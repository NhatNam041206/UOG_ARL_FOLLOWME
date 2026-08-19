# Technology Stack

## Existing (unchanged by this work)

| Purpose | Technology | Where |
|---|---|---|
| Person detection + multi-object tracking | Ultralytics YOLO11 (`yolo11n.onnx`) + ByteTrack | `src/detector.py` |
| Person re-identification | OSNet (torchreid), `osnet_x1_0` variant, 512-d L2-normalized embeddings, cosine similarity | `src/verifier.py` |
| Body-orientation / pose-proportions gate | YOLO11-pose (`yolo11n-pose.pt`) | `src/view_estimator.py` |
| Inference runtimes | ONNX Runtime (YOLO detector), PyTorch (OSNet, YOLO-pose) | `.venv` |
| Registry storage | NumPy `.npz` (embedding, aspect ratio, multi-view embeddings, pose proportions) | `src/registry.py`, `logs/registry/*.npz` |
| Camera I/O | OpenCV `cv2.VideoCapture`, MJPG capture, threaded non-blocking reader | `src/camera_utils.py`, `main.py` |
| Registration UI | Tkinter | `src/person_selector.py`, `src/registration.py` |
| Config | YAML | `config/settings.yaml` |

## New for the Wave + Facing Trigger Gate demo

| Purpose | Technology | Why this choice | Where |
|---|---|---|---|
| Pose estimation (17 COCO keypoints) | **MoveNet Lightning (singlepose)**, loaded via **TF Hub** (`tensorflow-hub`) | Explicitly mandated by the spec (fixed COCO keypoint order, `[1,1,17,3]` output). No official PyTorch build exists, so the OSNet/ByteTrack precedent doesn't apply here — TF is the only supported runtime. TF Hub's `hub.load(url)` auto-downloads and caches the model on first use, mirroring the auto-download pattern Ultralytics already uses for YOLO weights in this project (`src/detector.py`), instead of a manually-managed `.tflite` file. | `src/pose_estimator.py` |
| Inference runtime for MoveNet | `tensorflow` (CPU) | Needed as the host runtime for the TF Hub SavedModel; also provides `tf.image.resize_with_pad`, which is used to reproduce MoveNet's official centered-letterbox preprocessing exactly (verified empirically — see Implementation Audit). | `requirements.txt` |
| Wave detection | Rule-based (no ML): wrist-above-shoulder posture + wrist-x direction-change/amplitude over a rolling buffer | Per spec §4 — deliberately simple/interpretable for a proof-of-concept, all thresholds calibratable. | `src/wave_detector.py` |
| Facing-camera proxy | Rule-based: 4-keypoint (both eyes, both shoulders) confidence check | Per spec §5 — crude visibility proxy, explicitly not a real yaw-angle estimate. | `src/wave_detector.py` |

### Dependency note: `setuptools<81` pin

`tensorflow-hub` still imports the deprecated `pkg_resources` API. `setuptools>=81` removed that
module outright, which breaks `import tensorflow_hub` with `ModuleNotFoundError: No module named
'pkg_resources'`. Confirmed directly on this environment (installed setuptools was 84.0.0) before
adding the pin — this isn't a hypothetical, it reproduced immediately. Pinned in
`requirements.txt` until upstream `tensorflow-hub` drops the dependency.

## Model files

| Model | Source | Auto-fetched? |
|---|---|---|
| `yolo11n.onnx` | Ultralytics | Already present in repo |
| `yolo11n-pose.pt` | Ultralytics | Already present in repo |
| `osnet_x1_0_reid.pth` | torchreid Model Zoo (Google Drive) | Downloaded on first `OSNetVerifier` init if missing (`src/verifier.py`) |
| MoveNet Lightning (singlepose) | `https://tfhub.dev/google/movenet/singlepose/lightning/4` | Downloaded + cached by `tensorflow_hub` on first `MoveNetPoseEstimator` init |
