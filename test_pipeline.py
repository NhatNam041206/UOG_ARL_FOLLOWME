"""
Unit tests for src/pipeline.py: temporal smoothing (EMA + Voting), dynamic ROI-constrained
detection, and the aspect ratio hard gate.

YoloDetector and OSNetVerifier are mocked so these tests run without a real camera, YOLO
weights, or OSNet forward pass. Since every scenario below has at most one detection per
frame, verifier.compare() calls happen strictly in process_frame() call order, so a plain
ordered side_effect list is deterministic (no thread-race concerns from the pipeline's
internal ThreadPoolExecutor).
"""
import os
import csv
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from src.pipeline import FollowPipeline
from src import registry


def _det(track_id, bbox=(10, 10, 50, 50), confidence=0.9):
    return {"track_id": track_id, "bbox": bbox, "confidence": confidence}


class FollowPipelineTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="follow_pipeline_test_")
        self._pipelines = []

    def tearDown(self):
        for p in self._pipelines:
            p.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _build_pipeline(self, mode, threshold=0.5, ema_alpha=0.3,
                         voting_window_size=5, voting_ratio=0.6,
                         voting_min_ready_percent=0.6, reference_aspect_ratio=1.0,
                         verify_every_n_frames=1):
        config = {
            "camera_index": 0,
            "input_resolution": [640, 480],
            "yolo_model_path": "unused.onnx",
            "osnet_variant": "osnet_x1_0",
            "similarity_threshold": threshold,
            "camera_fov_horizontal_deg": 60.0,
            "verify_every_n_frames": verify_every_n_frames,
            "verification_smoothing": {
                "mode": mode,
                "ema_alpha": ema_alpha,
                "voting_window_size": voting_window_size,
                "voting_ratio": voting_ratio,
                "voting_min_ready_percent": voting_min_ready_percent,
            },
        }
        config_path = os.path.join(self.tmpdir, f"settings_{len(self._pipelines)}.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f)

        # Registry .npz file (embedding + aspect_ratio + metadata) — default reference_aspect_ratio=1.0
        # matches _det()'s default 40x40 bbox (AR=1.0) so tests that don't care about the aspect
        # ratio gate aren't accidentally vetoed by it.
        registry.REGISTRY_DIR = self.tmpdir
        npz_path = registry.save_person(
            name=f"test_person_{len(self._pipelines)}",
            embedding=np.zeros(4, dtype=np.float32),
            aspect_ratio=reference_aspect_ratio,
            sample_count=5,
        )

        with patch("src.pipeline.YoloDetector"), patch("src.pipeline.OSNetVerifier"):
            pipeline = FollowPipeline(config_path=config_path, reference_npz_path=npz_path)

        # process_frame() now extracts embeddings via ONE verifier.extract_batch(crops) call
        # instead of one verifier.extract(crop) call per track (see pipeline.py's "2. Crop
        # feature extraction" section) — give the mocked verifier a matching-shape default so
        # that call returns a real (mocked-object) list instead of an unconfigured MagicMock,
        # which isn't iterable and would break the zip() over it. Individual tests below only
        # ever assert on verifier.compare.side_effect, never on the embedding values themselves.
        pipeline.verifier.extract_batch.side_effect = lambda crops: [object() for _ in crops]

        # Redirect CSV logging to a temp file so tests never touch the real
        # logs/verification_log.csv, then rewrite the header there.
        pipeline.log_csv_path = os.path.join(self.tmpdir, f"verification_log_{len(self._pipelines)}.csv")
        pipeline._init_csv_file()

        self._pipelines.append(pipeline)
        return pipeline

    @staticmethod
    def _frame():
        return np.zeros((480, 640, 3), dtype=np.uint8)

    # ---- Config validation -------------------------------------------------

    def test_invalid_smoothing_mode_raises(self):
        with self.assertRaises(ValueError):
            self._build_pipeline(mode="bogus_mode")

    def test_csv_header_columns(self):
        pipeline = self._build_pipeline(mode="ema")
        with open(pipeline.log_csv_path, "r", newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        self.assertEqual(
            header,
            ["timestamp", "track_id", "raw_score", "smoothed_val", "voting_ready",
             "candidate_aspect_ratio", "ar_diff_ratio", "aspect_ratio_pass",
             "used_roi", "roi_bounds",
             "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"],
        )

    # ---- EMA mode (spec 5.4) ------------------------------------------------

    def test_ema_first_frame_uses_raw_score_not_zero(self):
        pipeline = self._build_pipeline(mode="ema", threshold=0.3, ema_alpha=0.3)
        pipeline.detector.track.side_effect = [[_det(1)]]
        pipeline.verifier.compare.side_effect = [0.42]

        result = pipeline.process_frame(self._frame())

        self.assertTrue(result.target_found)
        self.assertEqual(result.track_id, 1)
        self.assertAlmostEqual(result.similarity_score, 0.42, places=4)

    def test_ema_smooths_transient_dip_but_reacts_to_sustained_drop(self):
        # alpha=0.5, threshold=0.5:
        #   frame1 raw=0.9 -> smoothed=0.9 (pass)
        #   frame2 raw=0.3 -> smoothed=0.5*0.3+0.5*0.9=0.6 (still pass: smoothing absorbs 1 bad frame)
        #   frame3 raw=0.3 -> smoothed=0.5*0.3+0.5*0.6=0.45 (fail: sustained drop is not masked)
        pipeline = self._build_pipeline(mode="ema", threshold=0.5, ema_alpha=0.5)
        pipeline.detector.track.side_effect = [[_det(1)], [_det(1)], [_det(1)]]
        pipeline.verifier.compare.side_effect = [0.9, 0.3, 0.3]

        r1 = pipeline.process_frame(self._frame())
        r2 = pipeline.process_frame(self._frame())
        r3 = pipeline.process_frame(self._frame())

        self.assertTrue(r1.target_found)
        self.assertAlmostEqual(r1.similarity_score, 0.9, places=4)

        self.assertTrue(r2.target_found)
        self.assertAlmostEqual(r2.similarity_score, 0.6, places=4)

        self.assertFalse(r3.target_found)

    def test_ema_mode_has_no_startup_warmup_delay(self):
        pipeline = self._build_pipeline(mode="ema", threshold=0.5)
        pipeline.detector.track.side_effect = [[_det(1)]]
        pipeline.verifier.compare.side_effect = [0.9]

        result = pipeline.process_frame(self._frame())

        self.assertTrue(result.target_found)
        self.assertEqual(pipeline._startup_frame_count, 0)

    # ---- Voting mode: Mechanism 2 - startup warm-up (spec 6.3) --------------

    def test_voting_startup_warmup_forces_no_target(self):
        # voting_window_size=5, voting_min_ready_percent=0.6 -> warmup_frames=ceil(3.0)=3
        pipeline = self._build_pipeline(
            mode="voting", threshold=0.5, voting_window_size=5,
            voting_ratio=0.6, voting_min_ready_percent=0.6,
        )
        pipeline.detector.track.side_effect = [[_det(1)]] * 4
        pipeline.verifier.compare.side_effect = [0.99, 0.99, 0.99, 0.99]

        results = [pipeline.process_frame(self._frame()) for _ in range(4)]

        # First 3 frames: forced target_found=False regardless of raw_score.
        self.assertFalse(results[0].target_found)
        self.assertFalse(results[1].target_found)
        self.assertFalse(results[2].target_found)

        # Frame 4: warm-up done, buffer already has 4/4 True -> vote_ratio=1.0.
        self.assertTrue(results[3].target_found)
        self.assertEqual(results[3].track_id, 1)
        self.assertAlmostEqual(results[3].similarity_score, 1.0, places=4)

    # ---- Voting mode: Mechanism 1 - per-track fallback (spec 6.2) -----------

    def test_voting_per_track_fallback_before_buffer_ready(self):
        # voting_window_size=5, voting_min_ready_percent=0.6 -> min_ready_frames=ceil(3.0)=3
        pipeline = self._build_pipeline(
            mode="voting", threshold=0.5, voting_window_size=5,
            voting_ratio=0.6, voting_min_ready_percent=0.6,
        )
        # 4 empty frames first (warmup_frames=3, so exactly 3 frames are forced
        # False and the 4th is where warmup_done flips true) to clear pipeline
        # startup warm-up (Mechanism 2) without building any per-track history,
        # then a NEW track (id=2) appears mid-stream to exercise Mechanism 1
        # in isolation.
        pipeline.detector.track.side_effect = [
            [], [], [], [],
            [_det(2)], [_det(2)], [_det(2)],
        ]
        pipeline.verifier.compare.side_effect = [0.9, 0.3, 0.9]  # only the last 3 frames have a detection

        for _ in range(4):
            pipeline.process_frame(self._frame())
        self.assertTrue(pipeline._startup_warmup_done)

        # Frame 5: buffer=[True] (len=1 < 3) -> fallback to raw (0.9 >= 0.5 -> pass)
        r5 = pipeline.process_frame(self._frame())
        self.assertTrue(r5.target_found)
        self.assertEqual(r5.track_id, 2)
        self.assertFalse(pipeline.last_detections[0]["voting_ready"])

        # Frame 6: buffer=[True, False] (len=2 < 3) -> fallback to raw (0.3 >= 0.5 -> fail)
        r6 = pipeline.process_frame(self._frame())
        self.assertFalse(r6.target_found)

        # Frame 7: buffer=[True, False, True] (len=3 >= 3) -> ready, vote_ratio=2/3 >= 0.6 -> pass
        r7 = pipeline.process_frame(self._frame())
        self.assertTrue(r7.target_found)
        self.assertEqual(r7.track_id, 2)
        self.assertAlmostEqual(r7.similarity_score, 2.0 / 3.0, places=4)
        self.assertTrue(pipeline.last_detections[0]["voting_ready"])

    # ---- Dynamic ROI-constrained detection -----------------------------------

    def test_roi_constrained_detection_and_coordinate_conversion(self):
        pipeline = self._build_pipeline(mode="ema", threshold=0.5)

        # Square boxes (AR=1.0) throughout so this ROI-focused test isn't also incidentally
        # exercising the (separately-tested) aspect ratio gate.
        frame1_bbox = (300, 200, 350, 250)  # w=50, h=50
        pipeline.detector.track.side_effect = [
            [_det(1, bbox=frame1_bbox)],       # frame 1: no active target yet -> full frame
            [_det(1, bbox=(10, 10, 50, 50))],  # frame 2: ROI-local coords (crop-relative), w=40 h=40
        ]
        pipeline.verifier.compare.side_effect = [0.9, 0.9]

        r1 = pipeline.process_frame(self._frame())
        self.assertTrue(r1.target_found)
        self.assertEqual(pipeline.active_target_id, 1)

        # First call must have used the FULL frame (no ROI possible yet — no prior position).
        first_call_arg = pipeline.detector.track.call_args_list[0][0][0]
        self.assertEqual(first_call_arg.shape[:2], (480, 640))

        r2 = pipeline.process_frame(self._frame())

        # Second call must have received the ROI CROP (roi_margin_percent default 0.5):
        # bbox (300,200,350,250) expanded by 50% of its own w/h (25,25) -> (275,175,375,275).
        second_call_arg = pipeline.detector.track.call_args_list[1][0][0]
        expected_roi = (275, 175, 375, 275)
        expected_shape = (expected_roi[3] - expected_roi[1], expected_roi[2] - expected_roi[0])
        self.assertEqual(second_call_arg.shape[:2], expected_shape)

        # The ROI-local bbox (10,10,50,50) returned by the mocked detector must be converted
        # back to full-frame coordinates by adding the ROI offset (275,175).
        self.assertTrue(r2.target_found)
        self.assertEqual(pipeline.last_detections[0]["bbox"], (285, 185, 325, 225))

    def test_roi_fallback_to_full_frame_after_max_consecutive_failures(self):
        pipeline = self._build_pipeline(mode="ema", threshold=0.5)
        pipeline.roi_failure_max_frames = 2  # override default for a fast test

        pipeline.detector.track.side_effect = [
            [_det(1, bbox=(100, 100, 150, 150))],  # frame 1: full frame, target acquired (w=50 h=50)
            [],                                      # frame 2: ROI attempt #1 -> nobody found
            [],                                      # frame 3: ROI attempt #2 -> nobody found
            [_det(1, bbox=(50, 50, 100, 100))],     # frame 4: forced full-frame -> reacquired (w=50 h=50)
        ]
        pipeline.verifier.compare.side_effect = [0.9, 0.9]  # only frames 1 and 4 reach verification

        r1 = pipeline.process_frame(self._frame())
        self.assertTrue(r1.target_found)

        r2 = pipeline.process_frame(self._frame())
        self.assertFalse(r2.target_found)
        # A single ROI miss must NOT immediately discard the sticky target.
        self.assertEqual(pipeline.active_target_id, 1)
        self.assertEqual(pipeline._roi_failure_count, 1)
        second_call_arg = pipeline.detector.track.call_args_list[1][0][0]
        self.assertNotEqual(second_call_arg.shape[:2], (480, 640))

        r3 = pipeline.process_frame(self._frame())
        self.assertFalse(r3.target_found)
        self.assertEqual(pipeline._roi_failure_count, 2)
        third_call_arg = pipeline.detector.track.call_args_list[2][0][0]
        self.assertNotEqual(third_call_arg.shape[:2], (480, 640))

        r4 = pipeline.process_frame(self._frame())
        # 4th call must have used the FULL frame (roi_failure_count reached max -> forced fallback).
        fourth_call_arg = pipeline.detector.track.call_args_list[3][0][0]
        self.assertEqual(fourth_call_arg.shape[:2], (480, 640))
        self.assertTrue(r4.target_found)
        self.assertEqual(pipeline._roi_failure_count, 0)

    def test_roi_gate_rejection_does_not_immediately_discard_target(self):
        # Regression test for a real bug found via live webcam logs (2026-08-13): a
        # ROI-constrained frame where the SAME track is detected but its similarity/aspect
        # ratio momentarily dips below the gate (motion blur, bad angle for one frame — all
        # routine) must be treated as a ROI failure to retry, exactly like "found nobody in
        # ROI" is — NOT as immediate proof the target is gone. Previously active_target_id was
        # unconditionally cleared here, which defeated ROI retry the moment verification dipped
        # for a single frame.
        # ema_alpha=1.0 (no smoothing memory, smoothed_val == raw_score every frame) isolates
        # the gate-rejection path from EMA's OWN dip-absorbing behavior (alpha<1 would mask a
        # single bad frame by design -- that's a different, already-covered test).
        pipeline = self._build_pipeline(mode="ema", threshold=0.5, ema_alpha=1.0)

        pipeline.detector.track.side_effect = [
            [_det(1, bbox=(300, 200, 350, 250))],  # frame 1: full frame, target acquired (w=50 h=50)
            [_det(1, bbox=(10, 10, 50, 50))],      # frame 2: ROI, same track, but gate rejects it
            [_det(1, bbox=(10, 10, 50, 50))],      # frame 3: ROI retry, same track, now passes
        ]
        pipeline.verifier.compare.side_effect = [0.9, 0.1, 0.9]

        r1 = pipeline.process_frame(self._frame())
        self.assertTrue(r1.target_found)
        self.assertEqual(pipeline.active_target_id, 1)

        r2 = pipeline.process_frame(self._frame())
        self.assertFalse(r2.target_found)
        # The whole point of this test: identity must survive a single failed gate check.
        self.assertEqual(pipeline.active_target_id, 1)
        self.assertEqual(pipeline._roi_failure_count, 1)
        second_call_arg = pipeline.detector.track.call_args_list[1][0][0]
        self.assertNotEqual(second_call_arg.shape[:2], (480, 640))  # confirms ROI was used

        r3 = pipeline.process_frame(self._frame())
        self.assertTrue(r3.target_found)
        self.assertEqual(r3.track_id, 1)
        self.assertEqual(pipeline._roi_failure_count, 0)
        third_call_arg = pipeline.detector.track.call_args_list[2][0][0]
        self.assertNotEqual(third_call_arg.shape[:2], (480, 640))  # still ROI, not forced full-frame

    # ---- Aspect ratio hard gate -----------------------------------------------

    def test_aspect_ratio_gate_blocks_high_similarity_wrong_shape(self):
        # reference_aspect_ratio defaults to 1.0; tolerance 0.3 -> pass band [0.7, 1.3].
        pipeline = self._build_pipeline(mode="ema", threshold=0.5)
        pipeline.aspect_ratio_tolerance_percent = 0.3

        # bbox 40x200 -> AR=0.2, far outside the tolerance band.
        wrong_shape_bbox = (0, 0, 40, 200)
        pipeline.detector.track.side_effect = [[_det(1, bbox=wrong_shape_bbox)]]
        pipeline.verifier.compare.side_effect = [0.95]  # similarity alone would easily pass

        result = pipeline.process_frame(self._frame())

        self.assertFalse(result.target_found)
        self.assertFalse(pipeline.last_detections[0]["aspect_ratio_pass"])
        self.assertFalse(pipeline.last_detections[0]["is_pass"])

    # ---- Verification frame-skip (verify_every_n_frames) ---------------------

    def test_default_verify_every_frame_never_reuses(self):
        pipeline = self._build_pipeline(mode="ema", threshold=0.5)  # verify_every_n_frames defaults to 1
        pipeline.detector.track.side_effect = [[_det(1)], [_det(1)]]
        pipeline.verifier.compare.side_effect = [0.9, 0.9]

        pipeline.process_frame(self._frame())
        self.assertEqual(pipeline.last_verify_stats, {"verified_count": 1, "reused_count": 0})
        pipeline.process_frame(self._frame())
        self.assertEqual(pipeline.last_verify_stats, {"verified_count": 1, "reused_count": 0})
        self.assertEqual(pipeline.verifier.extract_batch.call_count, 2)

    def test_non_active_track_reuses_cached_score_between_verification_intervals(self):
        # verify_every_n_frames=3: a non-active, already-seen track should only get a fresh
        # OSNet call every 3rd frame; in between it must reuse its last raw_score untouched
        # while still picking up its CURRENT frame's bbox.
        pipeline = self._build_pipeline(mode="ema", threshold=0.5, verify_every_n_frames=3)

        # Frame 1: only the stranger (track 2) is present -> verified fresh (new track), fails
        # threshold so it never becomes the active target.
        # Frame 2: target (track 1, new) appears alongside the stranger (track 2, due=1 < 3 ->
        # reused). Frame 3: target (active -> always fresh) alongside stranger (due=2 < 3 ->
        # still reused). Exactly one fresh verification per frame throughout, so
        # verifier.compare's ordered side_effect list stays deterministic despite the executor.
        pipeline.detector.track.side_effect = [
            [_det(2, bbox=(0, 0, 40, 40))],
            [_det(1, bbox=(10, 10, 50, 50)), _det(2, bbox=(5, 5, 45, 45))],
            [_det(1, bbox=(10, 10, 50, 50)), _det(2, bbox=(20, 20, 60, 60))],
        ]
        pipeline.verifier.compare.side_effect = [0.2, 0.9, 0.9]

        pipeline.process_frame(self._frame())  # frame 1
        self.assertEqual(pipeline.last_verify_stats, {"verified_count": 1, "reused_count": 0})

        pipeline.process_frame(self._frame())  # frame 2
        self.assertEqual(pipeline.last_verify_stats, {"verified_count": 1, "reused_count": 1})
        self.assertEqual(pipeline.active_target_id, 1)
        track2 = next(t for t in pipeline.last_detections if t["track_id"] == 2)
        self.assertAlmostEqual(track2["raw_score"], 0.2, places=4)  # reused from frame 1
        self.assertEqual(track2["bbox"], (5, 5, 45, 45))  # bbox still refreshed this frame

        pipeline.process_frame(self._frame())  # frame 3
        self.assertEqual(pipeline.last_verify_stats, {"verified_count": 1, "reused_count": 1})
        track2 = next(t for t in pipeline.last_detections if t["track_id"] == 2)
        self.assertAlmostEqual(track2["raw_score"], 0.2, places=4)  # still reused
        self.assertEqual(track2["bbox"], (20, 20, 60, 60))  # but bbox is frame 3's

        # extract_batch should have been called once per frame with exactly the ONE crop that
        # actually needed a fresh embedding (never the reused stranger's crop).
        self.assertEqual(pipeline.verifier.extract_batch.call_count, 3)
        for call in pipeline.verifier.extract_batch.call_args_list:
            self.assertEqual(len(call[0][0]), 1)

    def test_active_target_always_verified_fresh_regardless_of_interval(self):
        # Even with a very long skip interval, the STICKY active target must never be skipped —
        # sticky-lock stability depends on catching a real drop immediately, not up to N frames late.
        pipeline = self._build_pipeline(mode="ema", threshold=0.5, verify_every_n_frames=10)
        pipeline.detector.track.side_effect = [[_det(1)], [_det(1)], [_det(1)]]
        pipeline.verifier.compare.side_effect = [0.9, 0.9, 0.9]

        pipeline.process_frame(self._frame())
        self.assertEqual(pipeline.active_target_id, 1)
        for _ in range(2):
            pipeline.process_frame(self._frame())
            self.assertEqual(pipeline.last_verify_stats, {"verified_count": 1, "reused_count": 0})

        self.assertEqual(pipeline.verifier.extract_batch.call_count, 3)

    def test_aspect_ratio_gate_allows_matching_shape(self):
        pipeline = self._build_pipeline(mode="ema", threshold=0.5)
        # Default _det() bbox is 40x40 -> AR=1.0, matches reference_aspect_ratio=1.0 exactly.
        pipeline.detector.track.side_effect = [[_det(1)]]
        pipeline.verifier.compare.side_effect = [0.9]

        result = pipeline.process_frame(self._frame())

        self.assertTrue(result.target_found)
        self.assertTrue(pipeline.last_detections[0]["aspect_ratio_pass"])


if __name__ == "__main__":
    unittest.main()
