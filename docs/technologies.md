# Technology Stack

## Runtime & core libraries

| Library | Version (`requirements.txt`) | Used for |
|---|---|---|
| Python | 3.x | Everything |
| `opencv-python` (cv2) | ≥4.8.0 | BGR frame I/O everywhere (the project's one universal image convention); also hosts two of the models below (YuNet, DNN blob preprocessing) and all debug drawing (`cv2.rectangle`, `cv2.putText`, `cv2.line`, `cv2.circle`) |
| `numpy` | ≥1.24.0 | Array/vector math throughout — embeddings, keypoints, trajectory vectors |
| `pyyaml` | ≥6.0 | Loads `config/thresholds.yaml`, the single source of truth for every tunable parameter |
| `ultralytics` | ≥8.3.0 | YOLO11 object detection + ByteTrack multi-object tracking |
| `torch` / `torchvision` | ≥2.0.0 / ≥0.15.0 | Backing runtime for `ultralytics` |
| `onnx` / `onnxruntime` / `onnxslim` | ≥1.14.0 / ≥1.15.0 / ≥0.1.0 | Runs the two ONNX-format models (YuNet, EdgeFace) via `onnxruntime.InferenceSession` |
| `lapx` | ≥0.9.0 | Linear-assignment solver ByteTrack depends on |
| `Pillow` | ≥10.0.0 | Transitive imaging dependency |
| `tensorflow` / `tensorflow-hub` | ≥2.13.0 / ≥0.14.0 | Loads and runs MoveNet Lightning (`hub.load(...)`, auto-downloads and caches on first use — no model file committed to the repo) |
| `mediapipe` | ≥1.0.0 | Hand landmark detection, via the modern Tasks API (`HandLandmarker`) — the legacy `mp.solutions.hands` API was dropped in this MediaPipe version |
| `torchreid` | ≥1.4.0 | OSNet model builder + the official `FeatureExtractor` preprocessing/inference utility (`appearance_verifier`) — KaiyangZhou/deep-person-reid, the reference implementation OSNet's own paper is published through |
| `gdown` | ≥4.7.0 | Downloads the Market1501-pretrained OSNet checkpoint from Google Drive on first use (see this doc's OSNet row — `torchreid`'s own `pretrained=True` shortcut does NOT fetch this checkpoint, only an ImageNet-classification backbone; `gdown` fetches the correct one directly by its published file id) |

## Models

Every model below is loaded as its **own independent instance** per module that uses it — see
[`architecture.md`](architecture.md)'s isolation rule. Two modules loading the "same" weights
file (e.g. `yolo11n.onnx`) still get two unrelated `YOLO(...)` objects with no shared state.

| Model | Format | Used by | Role |
|---|---|---|---|
| **YOLO11n** (`yolo11n.onnx`, COCO-pretrained, nano variant) | ONNX, via `ultralytics.YOLO` | `emergency_stop` (all 80 COCO classes, generic obstacle detection), `human_detection` (person class only, whole-frame + ByteTrack), `human_detection_roi` (person class only, single-frame, ROI-scoped) | Object/person bounding-box detection. Nano chosen for inference speed. |
| **ByteTrack** (`bytetrack.yaml`, bundled with `ultralytics`) | — | `emergency_stop`, `human_detection` (via `model.track(..., persist=True)`) | Multi-object tracking — assigns a stable `track_id` across frames per detected object/person. Deliberately **not** used by `human_detection_roi`, whose ROI crop shifts every frame and can't offer ByteTrack a stable coordinate frame. |
| **YuNet** (`face_detection_yunet_2023mar.onnx`) | ONNX, via `cv2.FaceDetectorYN` (bundled with OpenCV — no extra dependency) | `face_identity` | Face detection + 5-point landmark localization (eyes, nose, mouth corners) in one pass. |
| **EdgeFace-XS** (`edgeface_xs_gamma_06.onnx`, ~1.77M params, 99.73% LFW) | ONNX, via `onnxruntime` | `face_identity` | Face embedding (512-D, L2-normalized) for identity matching. Chosen over ArcFace/InsightFace specifically to avoid InsightFace's non-commercial pretrained-model licensing question. |
| **MoveNet Lightning** (singlepose, TF Hub `google/movenet/singlepose/lightning/4`) | TensorFlow SavedModel, via `tensorflow_hub` | `wave_facing_gate` (Method 1), `gesture_trajectory_verifier` (Method 3) | Single-person 17-keypoint pose estimation (COCO keypoint layout). Reused as a *model* by both gesture methods — never as shared code/instances (each loads its own `hub.load(...)`). ~15-20ms/frame on CPU per this project's own benchmarking. |
| **MediaPipe Hand Landmarker** (`hand_landmarker.task`) | MediaPipe Tasks bundle | `gesture_hand_keypoint` (Method 2) | 21-point-per-hand landmark detection (fixed layout: wrist + 4 joints × 5 fingers), up to 2 hands. Outputs are model-fixed — all 21 landmarks are always computed in one forward pass; there's no "detect fewer keypoints" option, though `num_hands` (currently 2) does trade off speed vs. simultaneous-hand coverage. |
| **OSNet** (`osnet_x1_0`, Market1501-pretrained, 94.2% rank-1 / 82.6% mAP) | PyTorch (`.pth`), via `torchreid` | `appearance_verifier` | Person re-identification embedding (512-D) — "does this crop look like the same person as this reference set." Chosen over a generic ResNet backbone because OSNet is purpose-trained for the same-person-across-views matching task, not repurposed generic classification features — at the accepted cost of two documented risks: similar-clothing confusion, and accuracy drop outside its Market1501-family training domain (see `docs/modules.md`'s `appearance_verifier` section). |

## Storage formats

| Format | Used for | Written by | Read by |
|---|---|---|---|
| `.npz` (one file per person) | Registered face identity: L2-normalized composite embedding (mean of samples) + every individual sample embedding + metadata | `modules/face_identity/registry.py` (`FaceRegistry.save_person`), invoked from `build_face_registry.py` | `FaceRegistry.load_all()` in `face_identity`'s matching stage |
| `.npz` (one file per reference gesture) | Method 3's shared, generic reference trajectory set (flattened, normalized, resampled wrist/elbow/shoulder path) | `modules/gesture_trajectory_verifier/reference_store.py`, invoked from `capture_reference_trajectory.py` | `ReferenceTrajectoryStore.load_all()` each `evaluate()` call |
| `config/thresholds.yaml` | Every tunable parameter, one YAML section per module, plus `camera.camera_index` | Hand-edited | Each module's own `load_config()` |

The `.npz`-per-person storage *shape* for face identity deliberately mirrors a sibling project's
(`UOG_ARL_FOLLOWME`, a separate git repository) OSNet-based registry convention — the format was
read for reference, then reimplemented from scratch here with no shared code or import path
between the two repositories.

## Why these choices (notable decisions)

- **Nano/lightweight models throughout** (YOLO11**n**, MoveNet **Lightning** not Thunder,
  EdgeFace-**XS**) — every model on the hot path was picked for CPU-friendly inference speed
  over maximum accuracy, consistent with running this live on a robot without a dedicated GPU.
- **Time-based trajectory resampling, not arc-length-based** (Method 3) — simpler, and a wave
  gesture is roughly periodic, so non-uniform speed along the path is a smaller risk than for
  arbitrary motion. Documented as a well-scoped future upgrade if shape fidelity ever proves
  insufficient.
- **Cosine similarity, not DTW** (Method 3) — an explicit non-goal per the module's spec unless
  empirically proven insufficient; fixed-length resampling + cosine similarity is far cheaper.
- **`cv2.estimateAffinePartial2D` instead of a third-party alignment package** (`face_identity`)
  — solves the same 5-point similarity-transform problem the reference implementation's own
  `uniface` dependency does, without adding a new dependency for a few dozen lines of geometry.
- **OSNet over a generic ResNet backbone** (`appearance_verifier`) — purpose-trained for the
  same-person-across-views re-identification task rather than repurposed image-classification
  features. Traded off against two accepted, explicitly documented risks: OSNet-based matching
  leans heavily on clothing as a feature (confuses similarly-dressed different people), and its
  Market1501-family training data is a different domain from this project's own campus footage
  (published benchmarks show accuracy can drop sharply outside that training distribution). Both
  risks apply to every caller of `appearance_verifier`, not just one — see `docs/modules.md`.
- **Hand-implemented PID over a library dependency** (`followme_orchestrator.SteeringController`)
  — no new package added for this. A standard PID loop is a few lines of arithmetic; pulling in
  a dependency for it would follow the opposite pattern of every other lightweight-over-heavier
  choice in this table. Same reasoning as `face_identity`'s `cv2.estimateAffinePartial2D` choice.
