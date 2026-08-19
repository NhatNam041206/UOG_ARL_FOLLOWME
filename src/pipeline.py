import os
import csv
import cv2
import math
import time
import yaml
import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import Optional, List, Dict, Any, Tuple

from src.types import AngleResult
from src.detector import YoloDetector
from src.verifier import OSNetVerifier
from src.view_estimator import ViewEstimator
from src import registry

logger = logging.getLogger(__name__)


class FollowPipeline:
    """
    Main interface class for CV Follow-Me system (Stage 2).
    Multi-threaded execution:
    - Parallel crop feature extraction & verification via ThreadPoolExecutor.
    - Temporal smoothing for identical track_ids across consecutive frames (EMA or Voting mode).
    - Cold-start instability fixes for Voting mode:
      * Mechanism 1: Per-track fallback to raw threshold when buffer < min_ready_frames.
      * Mechanism 2: Pipeline startup warm-up for the first N frames.
    - Dynamic ROI-constrained YOLO detection once a target is sticky-tracked, with automatic
      fallback to full-frame detection after repeated ROI failures.
    - Aspect ratio hard gate (independent of appearance similarity) using a reference aspect
      ratio captured at registration time.
    - Thread-safe CSV logging with raw_score, smoothed_val, voting_ready, aspect ratio gate
      fields, and ROI usage.
    - Sticky Target Locking & Exact Angle Calculation.
    """
    def __init__(self, config_path: str = "config/settings.yaml", reference_npz_path: str = None):
        # 1. Load configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: '{os.path.abspath(config_path)}'")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.threshold = float(self.config.get("similarity_threshold", 0.55))
        self.fov_horizontal = float(self.config.get("camera_fov_horizontal_deg", 60.0))
        self.input_resolution = self.config.get("input_resolution", [640, 480])

        # Dynamic ROI-constrained detection configuration
        self.roi_margin_percent = float(self.config.get("roi_margin_percent", 0.5))
        self.roi_failure_max_frames = int(self.config.get("roi_failure_max_frames", 5))
        self._roi_failure_count: int = 0

        # Aspect ratio hard gate configuration
        self.aspect_ratio_tolerance_percent = float(self.config.get("aspect_ratio_tolerance_percent", 0.30))

        # Temporal smoothing configuration parsing
        # Note: Temporal smoothing applies ONLY to identical track_ids across consecutive frames (does NOT handle ID switches).
        smoothing_cfg = self.config.get("verification_smoothing", {})
        self.smoothing_mode = str(smoothing_cfg.get("mode", "")).strip().lower()

        if self.smoothing_mode not in ("ema", "voting"):
            raise ValueError(
                f"Invalid verification_smoothing.mode '{self.smoothing_mode}' in configuration! "
                f"Must be explicitly set to either 'ema' or 'voting'."
            )

        self.ema_alpha = float(smoothing_cfg.get("ema_alpha", 0.3))
        self.voting_window_size = int(smoothing_cfg.get("voting_window_size", 5))
        self.voting_ratio = float(smoothing_cfg.get("voting_ratio", 0.6))
        self.voting_min_ready_percent = float(smoothing_cfg.get("voting_min_ready_percent", 0.6))

        # Temporal state tracking dicts keyed by track_id
        # Note: These states track history ONLY for identical track_ids. Track ID switches reset history.
        self.ema_scores: Dict[int, float] = {}
        self.voting_buffers: Dict[int, deque] = {}

        # Pipeline startup warm-up state (Mechanism 2, applies ONLY when mode == "voting")
        self._startup_frame_count: int = 0
        self._startup_warmup_done: bool = False
        self.startup_warmup_frames: int = math.ceil(self.voting_window_size * self.voting_min_ready_percent)

        # 2. Load the selected person's reference data (embedding + aspect ratio) from the
        # registry .npz file chosen by PersonRegistrySelector. Required, no default — a caller
        # forgetting to pass this should fail loudly here, not deep inside process_frame().
        if not reference_npz_path:
            raise ValueError(
                "reference_npz_path is required — pass the registry .npz path for the person "
                "selected via PersonRegistrySelector (or TargetRegistrar.run()'s return value)."
            )
        person_data = registry.load_person(reference_npz_path)
        self.reference_embedding = person_data["embedding"]
        self.reference_aspect_ratio = person_data["aspect_ratio"]
        if self.reference_aspect_ratio <= 1e-6:
            raise ValueError(
                f"Reference aspect ratio loaded from '{reference_npz_path}' is invalid "
                f"({self.reference_aspect_ratio}). Please re-register this person."
            )

        # Multi-view reference embeddings
        self.reference_embeddings_multiview = person_data.get("multi_views", {})
        self.has_multiview = any(
            self.reference_embeddings_multiview.get(v) is not None
            for v in ['front', 'right', 'back', 'left']
        ) if self.reference_embeddings_multiview else False

        # Reference pose proportions
        self.reference_pose_proportions = person_data.get("pose_proportions", {})

        logger.info(
            f"Loaded reference from '{reference_npz_path}': embedding shape {self.reference_embedding.shape}, "
            f"has_multiview={self.has_multiview}, pose_props={list(self.reference_pose_proportions.keys())}, "
            f"aspect_ratio {self.reference_aspect_ratio:.4f}, sample_count {person_data['sample_count']}, "
            f"created_at {person_data['created_at']}"
        )

        # 3. Load modules
        yolo_path = self.config.get("yolo_model_path", "yolo11n.onnx")
        osnet_variant = self.config.get("osnet_variant", "osnet_x1_0")
        pose_model_path = self.config.get("pose_model_path", "yolo11n-pose.pt")
        detection_imgsz = self.config.get("detection_imgsz")  # None -> ultralytics default (640)

        self.detector = YoloDetector(yolo_path, imgsz=detection_imgsz)
        self.verifier = OSNetVerifier(osnet_variant)
        self.view_estimator = ViewEstimator(pose_model_path)

        # Verification frame-skip: how often (in frames) an already-seen, non-active track gets
        # a FRESH OSNet+pose re-verification. 1 (default) = every frame, identical to the
        # original always-verify behavior. >1 skips re-extraction on the skipped frames and
        # reuses that track's last raw_score instead — ByteTrack already carries identity
        # between frames, so this trades a little verification staleness for a direct N-x cut
        # in the dominant per-frame cost (OSNet forward passes), for tracks whose identity
        # doesn't matter for this frame's decision. A brand-new track and the CURRENT sticky
        # active_target_id are always verified fresh every frame regardless of this setting —
        # see the "always verify" set built in process_frame().
        self.verify_every_n_frames = max(1, int(self.config.get("verify_every_n_frames", 1)))
        self._frame_counter: int = 0
        self._track_last_verified_frame: Dict[int, int] = {}
        self._track_cached_result: Dict[int, Dict[str, Any]] = {}

        # 4. Sticky Target state tracking & cached detections
        self.active_target_id: Optional[int] = None
        self.last_detections: List[Dict[str, Any]] = []
        # Exposed for external consumers (e.g. main.py's debug UI overlay) — reflects the most
        # recent process_frame() call's ROI decision, not just an internal implementation detail.
        self.last_used_roi: bool = False
        self.last_roi_bounds: Optional[Tuple[int, int, int, int]] = None
        # Per-stage wall-clock timing (ms) for the most recent process_frame() call — for
        # diagnosing where per-frame time actually goes (detect vs. verify vs. everything else)
        # instead of guessing. Exposed the same way as last_used_roi above. verified_count/
        # reused_count let a caller see the frame-skip mechanism actually working (reused_count
        # is always 0 when verify_every_n_frames == 1, the default).
        self.last_timing_ms: Dict[str, float] = {"detect_ms": 0.0, "verify_ms": 0.0, "total_ms": 0.0}
        self.last_verify_stats: Dict[str, int] = {"verified_count": 0, "reused_count": 0}

        # 5. CSV verification log setup & Thread Lock
        self.log_csv_path = "logs/verification_log.csv"
        os.makedirs(os.path.dirname(os.path.abspath(self.log_csv_path)), exist_ok=True)
        self.csv_lock = threading.Lock()
        self._init_csv_file()

        # 6. Multi-threaded ThreadPoolExecutor for parallel feature extraction
        self.num_workers = min(4, os.cpu_count() or 4)
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers, thread_name_prefix="FollowPipelineWorker")
        logger.info(
            f"FollowPipeline initialized (Mode: '{self.smoothing_mode}', Threshold: {self.threshold}, "
            f"MultiView: {self.has_multiview}, PoseGate: {bool(self.reference_pose_proportions)}, "
            f"AspectRatioTolerance: {self.aspect_ratio_tolerance_percent}, "
            f"ROI margin/failure_max: {self.roi_margin_percent}/{self.roi_failure_max_frames}, "
            f"Warmup Frames: {self.startup_warmup_frames if self.smoothing_mode == 'voting' else 0})"
        )

    def _init_csv_file(self) -> None:
        """Initialize verification log CSV file with headers if not present."""
        if not os.path.exists(self.log_csv_path):
            with self.csv_lock:
                with open(self.log_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "track_id", "raw_score", "smoothed_val", "voting_ready",
                        "candidate_aspect_ratio", "ar_diff_ratio", "aspect_ratio_pass",
                        "used_roi", "roi_bounds",
                        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                    ])
            logger.info(f"Created verification log CSV file at '{self.log_csv_path}'")

    def _compute_roi(self, bbox: Tuple[int, int, int, int], frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        """
        Expand a bbox by `roi_margin_percent` of its own width/height on each side, clamped to
        frame bounds. Used to build a search region around the target's last known position.
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        mx = w * self.roi_margin_percent
        my = h * self.roi_margin_percent
        rx1 = max(0, int(x1 - mx))
        ry1 = max(0, int(y1 - my))
        rx2 = min(frame_w, int(x2 + mx))
        ry2 = min(frame_h, int(y2 + my))
        return rx1, ry1, rx2, ry2

    @staticmethod
    def _get_crop(det: Dict[str, Any], frame, frame_w: int, frame_h: int):
        """
        Crop `det["bbox"]` out of `frame`, clamped to frame bounds. `det["bbox"]` is always
        expected in FULL-FRAME coordinates (already converted back from ROI-local coordinates
        by the caller if ROI-constrained detection was used) — verification always reads from
        the original full `frame`, never from a detection ROI crop.
        """
        x1, y1, x2, y2 = det["bbox"]
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(frame_w, x2), min(frame_h, y2)
        return frame[cy1:cy2, cx1:cx2]

    def _finish_verify_track(self, det: Dict[str, Any], crop, emb: np.ndarray) -> Dict[str, Any]:
        """
        Worker function executed in parallel thread pool. Takes an ALREADY-EXTRACTED OSNet
        embedding (`emb`, produced by a single batched verifier.extract_batch() call over every
        track needing fresh verification this frame — see process_frame()) and does the
        remaining per-track work: multi-view/pose similarity and score fusion. Kept separate
        from embedding extraction so extraction can be batched across all tracks in one model
        launch instead of N sequential ones (the dominant per-frame cost, see README perf
        section), while this remaining work still parallelizes across tracks via the executor.
        """
        track_id = det["track_id"]

        # Multi-view verification
        view_name = None
        if self.has_multiview and self.reference_embeddings_multiview:
            view_name, angle = self.view_estimator.estimate_view_from_crop(crop)
            ref_emb = self.reference_embeddings_multiview.get(view_name) if view_name else None
            if ref_emb is not None:
                similarity_osnet = self.verifier.compare(emb, ref_emb)
            else:
                # Fallback to median similarity across available view embeddings or composite
                available_embs = [e for e in self.reference_embeddings_multiview.values() if e is not None]
                if available_embs:
                    sims = [self.verifier.compare(emb, e) for e in available_embs]
                    similarity_osnet = float(np.median(sims))
                else:
                    similarity_osnet = self.verifier.compare(emb, self.reference_embedding)
        else:
            similarity_osnet = self.verifier.compare(emb, self.reference_embedding)

        # Pose proportions similarity
        if self.reference_pose_proportions:
            candidate_ratios = self.view_estimator.extract_pose_proportions_from_crop(crop)
            similarity_pose = self.view_estimator.compute_pose_similarity(candidate_ratios, self.reference_pose_proportions)
        else:
            similarity_pose = 0.5

        # Combined weighted score fusion (75% OSNet appearance + 25% Pose proportions)
        if self.reference_pose_proportions:
            similarity_combined = 0.75 * similarity_osnet + 0.25 * similarity_pose
        else:
            similarity_combined = similarity_osnet

        return {
            "track_id": track_id,
            "bbox": det["bbox"],
            "raw_score": similarity_combined,
            "similarity_osnet": similarity_osnet,
            "similarity_pose": similarity_pose,
            "view_name": view_name,
        }

    def process_frame(self, frame) -> AngleResult:
        """
        Process a single video frame with Multi-threaded parallel feature extraction:
        1. Decide ROI-constrained vs full-frame detection, run detector, convert ROI-local
           bboxes back to full-frame coordinates.
        2. Concurrently extract OSNet re-id embeddings & calculate raw similarity scores.
        3. Apply the aspect ratio hard gate (independent of similarity) per track.
        4. Apply Temporal Smoothing (EMA or Voting mode with Mechanisms 1 & 2) to similarity,
           then AND the result with the aspect ratio gate to get the final is_pass per track.
        5. Log raw_score, smoothed_val, voting_ready, aspect ratio gate fields, and ROI usage
           for ALL detected tracks to logs/verification_log.csv.
        6. Apply Sticky Target logic based on the combined is_pass evaluation.
        7. Calculate horizontal angle offset (deg) and size ratio for the target.
        """
        t_frame_start = time.time()

        if frame is None or frame.size == 0:
            logger.warning("Empty frame passed to FollowPipeline.process_frame()")
            self.last_detections = []
            return AngleResult(target_found=False)

        if self.input_resolution and len(self.input_resolution) == 2:
            rw, rh = int(self.input_resolution[0]), int(self.input_resolution[1])
            if frame.shape[1] != rw or frame.shape[0] != rh:
                frame = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)

        frame_h, frame_w = frame.shape[:2]
        timestamp = time.time()

        # Track pipeline startup frame counter for Mechanism 2 (only when mode == "voting")
        if self.smoothing_mode == "voting":
            self._startup_frame_count += 1

        self._frame_counter += 1

        # 1. Decide ROI-constrained vs full-frame detection.
        # ROI is only ever attempted once a target is already sticky-tracked (active_target_id
        # set) AND we have its bbox from the previous frame (last_detections) — there is no
        # position to build a ROI around otherwise. If too many consecutive ROI attempts have
        # failed, this frame is forced back to a full-frame scan and the failure streak is
        # cleared immediately (not conditional on this forced scan itself succeeding).
        t_detect_start = time.time()
        used_roi = False
        roi_bounds: Optional[Tuple[int, int, int, int]] = None
        force_full_frame = self._roi_failure_count >= self.roi_failure_max_frames

        if force_full_frame:
            self._roi_failure_count = 0
        elif self.active_target_id is not None:
            prev_target = next((t for t in self.last_detections if t["track_id"] == self.active_target_id), None)
            if prev_target is not None:
                roi_bounds = self._compute_roi(prev_target["bbox"], frame_w, frame_h)
                used_roi = True

        if used_roi:
            rx1, ry1, rx2, ry2 = roi_bounds
            roi_crop = frame[ry1:ry2, rx1:rx2]
            if roi_crop.size == 0:
                # Degenerate ROI (shouldn't normally happen given clamping) — don't count this
                # as a ROI attempt at all, just fall back to full frame for this single frame.
                used_roi = False
                roi_bounds = None
                detections = self.detector.track(frame)
            else:
                raw_dets = self.detector.track(roi_crop)
                # Convert ROI-local bbox coordinates back to full-frame coordinates. Forgetting
                # this step is the classic bug here: every downstream consumer (verification
                # crop, angle_offset_deg, size_ratio, CSV log) assumes full-frame coordinates.
                detections = [
                    {
                        "track_id": d["track_id"],
                        "bbox": (
                            d["bbox"][0] + rx1, d["bbox"][1] + ry1,
                            d["bbox"][2] + rx1, d["bbox"][3] + ry1,
                        ),
                        "confidence": d["confidence"],
                    }
                    for d in raw_dets
                ]
        else:
            detections = self.detector.track(frame)

        detect_ms = (time.time() - t_detect_start) * 1000.0

        # Expose this frame's final ROI decision for external consumers (e.g. main.py's debug
        # UI overlay) — reflects the decision AFTER the degenerate-ROI fallback above, so it's
        # always in sync no matter which branch below returns.
        self.last_used_roi = used_roi
        self.last_roi_bounds = roi_bounds

        if not detections:
            self.last_timing_ms = {
                "detect_ms": round(detect_ms, 1), "verify_ms": 0.0,
                "total_ms": round((time.time() - t_frame_start) * 1000.0, 1),
            }
            if used_roi:
                # A ROI-constrained scan finding nobody does NOT necessarily mean the target is
                # gone — it may simply have stepped outside the (deliberately tight) ROI margin
                # for one frame. Treat this as a ROI failure (feeds the fallback counter) but
                # deliberately do NOT clear active_target_id / last_detections here, so the next
                # frame can still build a ROI from the last known position and retry. Only a
                # genuine full-frame miss (branch below) is treated as real target loss.
                self._roi_failure_count += 1
                if self.smoothing_mode == "voting" and not self._startup_warmup_done:
                    if self._startup_frame_count > self.startup_warmup_frames:
                        self._startup_warmup_done = True
                return AngleResult(target_found=False)

            self.active_target_id = None
            self.last_detections = []
            self._roi_failure_count = 0
            # Clean up memory buffers when no tracks present
            self.ema_scores.clear()
            self.voting_buffers.clear()

            # Check startup warm-up state transition even when no tracks present
            if self.smoothing_mode == "voting" and not self._startup_warmup_done:
                if self._startup_frame_count > self.startup_warmup_frames:
                    self._startup_warmup_done = True

            return AngleResult(target_found=False)

        # 2. Crop feature extraction & raw verification (always reads from the original full
        # frame — ROI constraining only applies to detection, never to verification, per spec).
        #
        # Two-part cost reduction over the original per-track "extract+verify in one executor
        # task" design:
        #   (a) BATCHED extraction: every track needing a fresh OSNet embedding this frame is
        #       extracted in ONE verifier.extract_batch() call instead of N sequential
        #       verifier.extract() calls — collapses N model launches into 1.
        #   (b) FRAME-SKIP verification: a track that isn't brand-new and isn't the current
        #       sticky active_target_id only needs a *fresh* embedding every
        #       `verify_every_n_frames` frames; in between, its last raw_score is reused
        #       (ByteTrack already carries track_id identity across frames on its own). The
        #       active target is always verified fresh every frame since sticky-lock stability
        #       depends on it, and a brand-new track_id is always verified fresh on its first
        #       appearance since there's no prior score to reuse yet. With the default
        #       verify_every_n_frames=1 this set is just "every detection", so behavior and
        #       verifier.compare() call count/order are unchanged from before.
        t_verify_start = time.time()

        to_verify: List[Dict[str, Any]] = []
        to_reuse: List[Dict[str, Any]] = []
        for det in detections:
            tid = det["track_id"]
            is_new = tid not in self._track_last_verified_frame
            is_active = (tid == self.active_target_id)
            due = (self._frame_counter - self._track_last_verified_frame.get(tid, -1)) >= self.verify_every_n_frames
            if is_new or is_active or due:
                to_verify.append(det)
            else:
                to_reuse.append(det)

        crops_to_verify = [self._get_crop(det, frame, frame_w, frame_h) for det in to_verify]
        embeddings = self.verifier.extract_batch(crops_to_verify) if crops_to_verify else []

        futures = [
            self.executor.submit(self._finish_verify_track, det, crop, emb)
            for det, crop, emb in zip(to_verify, crops_to_verify, embeddings)
        ]
        fresh_tracks = [f.result() for f in futures]
        for t in fresh_tracks:
            tid = t["track_id"]
            self._track_last_verified_frame[tid] = self._frame_counter
            # Cache a SNAPSHOT, not a reference to `t` itself — the "3-4." block below mutates
            # `t` in place (adds candidate_aspect_ratio/smoothed_val/is_pass/...) as part of
            # every frame's processing, fresh or reused alike. Caching the live reference would
            # let next frame's reuse pick up this frame's mutated/stale extra keys instead of a
            # clean extraction snapshot (harmless today since those keys get overwritten fresh
            # every frame regardless, but aliasing mutable shared state here is a bug waiting to
            # happen the next time this block changes).
            self._track_cached_result[tid] = dict(t)

        reused_tracks = []
        for det in to_reuse:
            cached = dict(self._track_cached_result[det["track_id"]])
            cached["bbox"] = det["bbox"]  # position is always fresh from this frame's tracker
            reused_tracks.append(cached)

        # Restore original detection order (cosmetic — keeps CSV rows/consumers in a stable,
        # tracker-reported order regardless of which tracks were verified vs. reused).
        by_track_id = {t["track_id"]: t for t in fresh_tracks + reused_tracks}
        processed_tracks = [by_track_id[det["track_id"]] for det in detections]

        verify_ms = (time.time() - t_verify_start) * 1000.0
        self.last_verify_stats = {"verified_count": len(to_verify), "reused_count": len(to_reuse)}
        current_track_ids = {t["track_id"] for t in processed_tracks}

        # Clean up disappeared tracks from smoothing + frame-skip cache state to prevent memory leaks
        # Note: Temporal smoothing applies ONLY to identical track_ids across consecutive frames (does NOT handle ID switches).
        for tid in list(self.ema_scores.keys()):
            if tid not in current_track_ids:
                del self.ema_scores[tid]

        for tid in list(self.voting_buffers.keys()):
            if tid not in current_track_ids:
                del self.voting_buffers[tid]

        for tid in list(self._track_last_verified_frame.keys()):
            if tid not in current_track_ids:
                del self._track_last_verified_frame[tid]
                self._track_cached_result.pop(tid, None)

        # 3-4. Aspect ratio hard gate + Temporal Smoothing (EMA or Voting mode) & CSV Logging
        min_ready_frames = math.ceil(self.voting_window_size * self.voting_min_ready_percent)
        roi_bounds_str = f"{roi_bounds[0]},{roi_bounds[1]},{roi_bounds[2]},{roi_bounds[3]}" if roi_bounds else ""

        for t in processed_tracks:
            tid = t["track_id"]
            raw_score = t["raw_score"]

            # Aspect ratio gate: independent of appearance similarity, computed fresh every
            # frame from the CURRENT bbox (not smoothed/averaged over time) — a same-frame
            # instantaneous veto, deliberately kept separate from EMA/voting so smoothed_val
            # stays a clean similarity-only signal (calibratable independently via the CSV).
            tx1, ty1, tx2, ty2 = t["bbox"]
            bbox_w = float(tx2 - tx1)
            bbox_h = float(ty2 - ty1)
            if bbox_h > 1e-6:
                candidate_ar = bbox_w / bbox_h
                ar_diff_ratio = abs(candidate_ar - self.reference_aspect_ratio) / self.reference_aspect_ratio
                aspect_ratio_pass = ar_diff_ratio <= self.aspect_ratio_tolerance_percent
            else:
                candidate_ar = 0.0
                ar_diff_ratio = -1.0
                aspect_ratio_pass = False
            t["candidate_aspect_ratio"] = round(candidate_ar, 4)
            t["ar_diff_ratio"] = round(ar_diff_ratio, 4)
            t["aspect_ratio_pass"] = aspect_ratio_pass

            if self.smoothing_mode == "ema":
                # EMA: smoothed = alpha * raw + (1 - alpha) * prev_smoothed
                # Initialized with first raw_score if track_id is new
                if tid not in self.ema_scores:
                    smoothed_val = raw_score
                else:
                    smoothed_val = self.ema_alpha * raw_score + (1.0 - self.ema_alpha) * self.ema_scores[tid]

                self.ema_scores[tid] = smoothed_val
                t["smoothed_val"] = round(smoothed_val, 4)
                similarity_is_pass = (smoothed_val >= self.threshold)
                t["voting_ready"] = True  # EMA mode is always ready

            elif self.smoothing_mode == "voting":
                # Voting: Sliding window buffer of boolean pass/fail (raw_score >= threshold)
                if tid not in self.voting_buffers:
                    self.voting_buffers[tid] = deque(maxlen=self.voting_window_size)

                is_raw_pass = (raw_score >= self.threshold)
                self.voting_buffers[tid].append(is_raw_pass)

                # Compute vote ratio over actual elements present in deque
                buf = self.voting_buffers[tid]
                vote_ratio = sum(buf) / float(len(buf))
                t["smoothed_val"] = round(vote_ratio, 4)

                # Mechanism 1: Per-Track Fallback when buffer < min_ready_frames
                if len(buf) < min_ready_frames:
                    t["voting_ready"] = False
                    similarity_is_pass = is_raw_pass  # Fallback directly to raw_score >= self.threshold
                else:
                    t["voting_ready"] = True
                    similarity_is_pass = (vote_ratio >= self.voting_ratio)

            # Hard gate: candidate is only valid if BOTH similarity (post-smoothing) AND aspect
            # ratio agree. Combined AFTER smoothing so EMA/voting state itself stays untouched
            # by aspect ratio — only the final decision is vetoed.
            t["is_pass"] = similarity_is_pass and aspect_ratio_pass

            # Thread-safe CSV logging with raw_score, smoothed_val, voting_ready, aspect ratio
            # gate fields, and ROI usage for this frame
            x1, y1, x2, y2 = t["bbox"]
            with self.csv_lock:
                with open(self.log_csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp, tid, round(raw_score, 4), t["smoothed_val"], t["voting_ready"],
                        t["candidate_aspect_ratio"], t["ar_diff_ratio"], t["aspect_ratio_pass"],
                        used_roi, roi_bounds_str,
                        x1, y1, x2, y2,
                    ])

        self.last_detections = processed_tracks
        # Remaining work below (sticky selection + angle math) is pure in-memory arithmetic, no
        # I/O or model inference — negligible next to detect/verify, so total is finalized here.
        self.last_timing_ms = {
            "detect_ms": round(detect_ms, 1), "verify_ms": round(verify_ms, 1),
            "total_ms": round((time.time() - t_frame_start) * 1000.0, 1),
        }

        # Mechanism 2: Startup Warm-up check (runs ONCE at pipeline start when mode == "voting")
        # Uses <= (not <) so exactly `startup_warmup_frames` frames are forced target_found=False,
        # and normal processing resumes starting at frame (startup_warmup_frames + 1).
        if self.smoothing_mode == "voting" and not self._startup_warmup_done:
            if self._startup_frame_count <= self.startup_warmup_frames:
                # Log CSV was written above, but return target_found=False immediately during startup warm-up
                return AngleResult(target_found=False)
            else:
                self._startup_warmup_done = True

        # 6. Sticky Target Selection Logic using the combined (similarity AND aspect ratio) is_pass
        target_track = None

        # Check if active target is still present and valid
        if self.active_target_id is not None:
            active_candidate = next((t for t in processed_tracks if t["track_id"] == self.active_target_id), None)
            if active_candidate is not None and active_candidate["is_pass"]:
                target_track = active_candidate

        # If active target is lost or failed verification, select candidate with highest smoothed value
        if target_track is None:
            valid_candidates = [t for t in processed_tracks if t["is_pass"]]
            if valid_candidates:
                target_track = max(valid_candidates, key=lambda t: t["smoothed_val"])
                self.active_target_id = target_track["track_id"]
                logger.info(
                    f"Target locked onto track_id: {self.active_target_id} "
                    f"(Mode: '{self.smoothing_mode}', Ready: {target_track['voting_ready']}, "
                    f"Val: {target_track['smoothed_val']:.3f}, Raw: {target_track['raw_score']:.3f}, "
                    f"AR: {target_track['candidate_aspect_ratio']:.3f})"
                )

        # ROI failure-count bookkeeping + deciding whether this is a REAL loss of the target.
        # Mirrors the "zero detections during ROI" branch above: a ROI-constrained frame that
        # found tracks but had NONE pass the combined gate (similarity dip, motion blur, bad
        # angle for one frame — all routine) is a ROI failure to retry, NOT proof the target is
        # gone. Only a genuine full-frame miss clears active_target_id. Previously this branch
        # unconditionally cleared active_target_id on any failure, which defeated the ROI retry
        # mechanism the moment verification dipped for a single frame — confirmed by a real run
        # where the target was lost ~1s after being locked and then never got ROI-constrained
        # again (see Stage1_2_Implementation_Review_Log.md mục 15).
        if used_roi:
            if target_track is not None:
                self._roi_failure_count = 0
                # else: falls through — do NOT clear active_target_id, let it retry via ROI
            else:
                self._roi_failure_count += 1
                return AngleResult(target_found=False)
        elif target_track is None:
            self.active_target_id = None
            self._roi_failure_count = 0
            return AngleResult(target_found=False)

        # 7. Angle Offset & Size Ratio Calculation
        tx1, ty1, tx2, ty2 = target_track["bbox"]
        bbox_center_x = (tx1 + tx2) / 2.0

        # Normalized x offset from frame center in range [-1.0, 1.0]
        offset_x_norm = (bbox_center_x - (frame_w / 2.0)) / (frame_w / 2.0)

        # Exact trigonometric horizontal angle calculation:
        fov_half_rad = math.radians(self.fov_horizontal / 2.0)
        angle_offset_deg = math.degrees(math.atan(offset_x_norm * math.tan(fov_half_rad)))

        # Coarse distance proxy ratio: bbox_height / frame_height
        bbox_height = float(ty2 - ty1)
        size_ratio = bbox_height / float(frame_h)

        return AngleResult(
            target_found=True,
            angle_offset_deg=angle_offset_deg,
            size_ratio=size_ratio,
            track_id=target_track["track_id"],
            similarity_score=target_track["smoothed_val"]
        )

    def close(self):
        """Cleanly shutdown worker thread pool."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
