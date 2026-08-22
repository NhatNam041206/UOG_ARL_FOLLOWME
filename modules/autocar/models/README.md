Place generated artifacts here:

- `yolov8n-pose_fp16.engine` - built by `scripts/build_trt_engine.sh` on the Nano (see main README).
- `action_gru.pt` - trained by `action/train_gru.py`.
- `osnet_x0_25_msmt17.pth` - downloaded by `scripts/download_osnet_weights.py` (only needed for
  `--reid-backend osnet`, the default CPU embedder).
- `osnet_x1_0_msmt17.pth` / `.onnx` / `.engine` - downloaded/exported/built via
  `scripts/download_osnet_weights.py --variant x1_0` -> `scripts/export_osnet_onnx.py` ->
  `scripts/build_osnet_trt_engine.sh` (only needed for `--reid-backend osnet_trt`, the
  recommended GPU embedder - see README "appearance re-identification").
- `enrolled_<name>.npz` - created by `scripts/enroll_person.py` (only needed for `--reid-enroll`).

None of these are checked into git by default (see `.gitignore`) - the
TensorRT engine is hardware+version locked, the GRU checkpoint depends on
your labeled training data, the OSNet weights are a large third-party
download, and enrollment files contain a real person's appearance embedding
(mild biometric data - `.gitignore` currently does NOT exclude
`enrolled_*.npz` since it's small and useful to keep, but reconsider before
pushing one to a public repo).
