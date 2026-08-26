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
| `lapx` | ≥0.9.0 | Linear-assignment solver this project's OWN ByteTrack (`emergency_stop`/`human_detection`) depends on — NOT used by the vendored `modules/autocar` tracker, which has its own scipy-based assignment (see below) |
| `Pillow` (PIL) | ≥10.0.0 | `register_person.py`'s Tkinter UI — converts each BGR camera frame to a `PIL.Image` → `ImageTk.PhotoImage` for display in a `tk.Label`. Was a transitive-only dependency before the registration UI. |
| `tkinter` | stdlib | `register_person.py`'s `RegistrationApp`/`CaptureWindow` — the CRUD UI and live capture preview. Not in `requirements.txt` (ships with the standard Python install). |
| `mediapipe` | ≥1.0.0 | Hand landmark detection, via the modern Tasks API (`HandLandmarker`) — the legacy `mp.solutions.hands` API was dropped in this MediaPipe version |
| `torchreid` | ≥1.4.0 | OSNet model builder + the official `FeatureExtractor` preprocessing/inference utility (`appearance_verifier`, SUPERSEDED module) — KaiyangZhou/deep-person-reid, the reference implementation OSNet's own paper is published through. Also used ONE TIME, offline, to export `modules/autocar/models/osnet_x1_0_msmt17.onnx` (see the `autocar` vendor row below) — despite the vendored repo's own code comment claiming this package "fails to import," it installs and works fine in this project's actual environment. |
| `gdown` | ≥4.7.0 | Downloads the Market1501-pretrained OSNet checkpoint from Google Drive on first use for `appearance_verifier` (`torchreid`'s own `pretrained=True` shortcut does NOT fetch this checkpoint, only an ImageNet-classification backbone). Also used once to fetch the MSMT17-pretrained checkpoint the `autocar` ONNX export below was built from, from the official `kaiyangzhou/deep-person-reid` Model Zoo. |
| `scipy` | ≥1.10.0 | The vendored `modules/autocar` tracker's own Hungarian assignment (`scipy.optimize.linear_sum_assignment`) and Kalman filter (`scipy.linalg.cho_factor`/`cho_solve`) — a from-scratch ByteTrack reimplementation, deliberately dependency-light (no `lap`/`filterpy`) per their own code comment, targeting a future Jetson Nano deploy. Not used by anything else in this project. |

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
| **MediaPipe Hand Landmarker** (`hand_landmarker.task`) | MediaPipe Tasks bundle | `gesture_hand_keypoint` (the TRIGGER gesture method) | 21-point-per-hand landmark detection (fixed layout: wrist + 4 joints × 5 fingers), up to 2 hands. Outputs are model-fixed — all 21 landmarks are always computed in one forward pass; there's no "detect fewer keypoints" option, though `num_hands` (currently 2) does trade off speed vs. simultaneous-hand coverage. |
| **OSNet** (`osnet_x1_0`, Market1501-pretrained, 94.2% rank-1 / 82.6% mAP) | PyTorch (`.pth`), via `torchreid` | `appearance_verifier` (SUPERSEDED, not in the live call path — see below) | Person re-identification embedding (512-D) — "does this crop look like the same person as this reference set." Chosen over a generic ResNet backbone because OSNet is purpose-trained for the same-person-across-views matching task, not repurposed generic classification features — at the accepted cost of two documented risks: similar-clothing confusion, and accuracy drop outside its Market1501-family training domain (see `docs/modules.md`'s `appearance_verifier` section). |
| **YOLOv8n-pose** (`yolov8n-pose.pt`, ultralytics) | PyTorch, via `ultralytics.YOLO` (their `detector/yolov8_pose_torch.py`) | `modules/autocar` (vendored, driven by `autocar_adapter.py`) | Person detection + full-body pose keypoints in one pass — a SEPARATE weights file/instance from this project's own `yolo11n.onnx` uses above; auto-downloads to the repo root on first use, same as `yolo11n.onnx`. |
| Their own ByteTrack (`modules/autocar/tracker/`) | Pure numpy+scipy, hand-written | `modules/autocar` | Multi-object tracking, functionally equivalent to `ultralytics`' bundled ByteTrack above but independently implemented (no `lap` dependency) — see the `scipy` row above. |
| **OSNet** (`osnet_x1_0`, **MSMT17**-pretrained — a DIFFERENT checkpoint from `appearance_verifier`'s Market1501 one above) | ONNX, via `onnxruntime` directly (their `identity/osnet_embedder.py` — no `torchreid` at inference time) | `modules/autocar` (`TargetLock`, driven by `autocar_adapter.py`) | Person re-identification embedding for the live tracking+recovery path — front-head, back-of-head, and lower-body regions each get their own embedding (see `docs/modules.md`'s `autocar`/`autocar_adapter` section). **This exact `.onnx` file isn't part of the vendored repo** (their own `.gitignore` excludes it, and no export script for it exists in their tree either) — obtained by downloading the MSMT17 checkpoint via `gdown` from the official Model Zoo and exporting it with `torchreid` + `torch.onnx.export` (opset 12, `dynamo=False` — torch 2.13's default dynamo-based exporter needs `onnxscript`, not installed), a one-time offline step, output verified end-to-end through their own unmodified `OSNetEmbedder` (self-similarity ≈ 1.0) before being placed at `modules/autocar/models/osnet_x1_0_msmt17.onnx`. |

## Storage formats

| Format | Used for | Written by | Read by |
|---|---|---|---|
| `.npz` (one file per person) | Registered face identity: L2-normalized composite embedding (mean of samples) + every individual sample embedding + metadata | `modules/face_identity/registry.py` (`FaceRegistry.save_person`), invoked from `build_face_registry.py` | `FaceRegistry.load_all()` in `face_identity`'s matching stage |
| `config/thresholds.yaml` | Every tunable parameter, one YAML section per module, plus `camera.camera_index` | Hand-edited | Each module's own `load_config()` |
| `.npz` (`modules/autocar/models/enrolled_<name>.npz`) | A person's re-id profile — front-head, back-of-head, and lower-body OSNet embeddings + bbox aspect ratio (their own format, `identity/target_profile.py`, unmodified) | `registration_data.build_target_profile()` | `TargetLock.__init__` (inside `autocar_adapter.py`'s `start()`) |
| `.jpg` (`registration_captures/<name>/{raw,cropped}/{front,back}/`) | Intermediate registration photos — RAW (exact camera frame) then CROPPED (ROI-cropped, a real inspectable file, not a value computed and discarded) | `registration_data.save_raw_capture()` / `build_cropped_roi()` | `build_face_registry()` / `build_target_profile()` (never anything live) |

The `.npz`-per-person storage *shape* for face identity deliberately mirrors a sibling project's
(`UOG_ARL_FOLLOWME`, a separate git repository) OSNet-based registry convention — the format was
read for reference, then reimplemented from scratch here with no shared code or import path
between the two repositories.

## Why these choices (notable decisions)

- **Nano/lightweight models throughout** (YOLO11**n**, EdgeFace-**XS**) — every model on the hot
  path was picked for CPU-friendly inference speed over maximum accuracy, consistent with running
  this live on a robot without a dedicated GPU.
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
- **Vendoring a teammate's already-built tracking+recovery engine over building a second one** —
  `modules/target_tracking`/`target_recovery` already worked, but `modules/autocar`
  (`vinhh9608-byte/Autocar`) already solved the same problem, independently, with a different
  design (their `TargetLock` folds tracking and recovery into one state machine, avoiding the
  handoff complexity a two-module split requires). Pulled in via `git clone` — not hand-copied
  through an API, and not a git submodule — kept byte-for-byte unmodified, with all
  project-specific glue (`autocar_adapter.py`) living outside that directory entirely. Eager
  warmup at `configure()` time (see `docs/modules.md`'s `followme_orchestrator` section) exists
  specifically because this backbone's own model-loading cost, left lazy, showed up as a
  multi-second stutter at the exact moment a gesture trigger fired.
