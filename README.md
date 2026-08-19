# UOG AIS Follow-Me — CV Module (Stage 1 + Stage 2)

Module thị giác máy tính độc lập, chạy test trên laptop (không phải bản deploy Pi 5).
Nhiệm vụ: khóa mục tiêu là một người cụ thể qua webcam, rồi mỗi frame trả về góc lệch
ngang (so với tâm khung hình) và một tỉ lệ kích thước thô dùng làm proxy khoảng cách —
dữ liệu này để module điều khiển chuyển động (nằm ngoài repo này) dùng để bám theo người đó.

Xem [plan.md](plan.md) để biết spec gốc mà codebase này bám theo.

## Mục tiêu chính

1. **Stage 1 — Registration**: người dùng đứng vào vùng ROI trước webcam, đặt tên, hệ thống
   chụp nhiều mẫu ảnh, trích embedding đặc trưng ngoại hình bằng OSNet (person re-identification),
   và lưu vào **registry nhiều người** (`logs/registry/<tên>.npz`) — có thể đăng ký nhiều người,
   trước khi chạy Stage 2 người dùng chọn 1 người trong số đó làm target phiên đó.
2. **Stage 2 — Follow**: mỗi frame, hệ thống phát hiện + track tất cả người trong khung
   hình (YOLOv11 + ByteTrack tích hợp sẵn của Ultralytics), so khớp từng track với embedding
   tham chiếu bằng cosine similarity, chọn ra track khớp nhất là "target" (có cơ chế
   "sticky" để không nhảy target liên tục), rồi tính góc lệch ngang + size ratio của target đó.

Phạm vi **không bao gồm**: Stage 3 (re-assignment/extrapolation khi mất track hẳn), phép
chiếu IPM/world-frame đầy đủ, và tối ưu/quantize cho Raspberry Pi 5.

## Cấu trúc file

```
UOG_AIS_FOLLOWME/
├── main.py                    # Entry point CLI: mode "register" hoặc "run"
├── plan.md                    # Spec gốc (nguồn sự thật cho behavior mong đợi)
├── Stage1_2_Implementation_Review_Log.md  # Log review + quyết định thiết kế qua các vòng
├── test_pipeline.py           # Unit test cho temporal smoothing (EMA/Voting) trong pipeline.py
├── test_verifier.py           # Unit test cho OSNetVerifier (extract/compare), dùng ảnh giả
├── test_registry.py           # Unit test cho registry CRUD (sanitize/save/load/delete/rename)
├── test_person_selector.py    # Unit test cho logic điều phối PersonRegistrySelector.run()
├── requirements.txt           # Dependency Python
├── config/
│   └── settings.yaml          # Toàn bộ số cấu hình (camera, ROI, threshold, FOV...)
├── src/
│   ├── types.py                # Dataclass AngleResult — kiểu dữ liệu trả về của pipeline
│   ├── detector.py             # YoloDetector — detect + track người (YOLOv11 + ByteTrack)
│   ├── verifier.py             # OSNetVerifier — trích embedding re-id & so khớp cosine similarity
│   ├── registry.py             # Data layer: sanitize tên, save/load/list/delete/rename .npz
│   ├── registration.py         # TargetRegistrar(GUI) — công cụ Stage 1 lấy mẫu, nhập tên & lưu vào registry
│   ├── person_selector.py      # PersonRegistrySelector(GUI) — chọn/CRUD người trước khi chạy Stage 2
│   ├── pipeline.py             # FollowPipeline — ghép detector+verifier, interface chính Stage 2
│   ├── debug_overlay.py        # Vẽ overlay debug (bbox, ROI, telemetry, timing) — chỉ import khi có --ui
│   └── camera_utils.py         # configure_capture() — set MJPG/resolution/buffer cho cv2.VideoCapture
├── models/
│   └── osnet_x1_0_reid.pth     # Checkpoint OSNet fine-tune trên Market1501 (auto-tải lần đầu)
├── logs/
│   ├── registry/                   # Mỗi người 1 file — <tên_đã_sanitize>.npz (embedding+aspect_ratio+metadata)
│   └── verification_log.csv        # Sinh ra khi chạy mode run — log similarity mọi track mọi frame
├── yolo11n.pt / yolo11n.onnx   # Weight YOLOv11n (pretrained COCO, class 0 = person)
└── .venv/                      # Virtualenv cục bộ
```

Mỗi file trong `src/` là một module phẳng, không có package con. Quy tắc phụ thuộc giữa
các module: `detector.py` và `verifier.py` độc lập với nhau (không import chéo); `registry.py`
là data layer thuần (không Tkinter), được `registration.py`, `person_selector.py`, và
`pipeline.py` cùng dùng; `registration.py`/`person_selector.py`/`pipeline.py` là 3 điểm gộp
mọi thứ lại (theo đúng vai trò Stage 1 đăng ký / chọn người / Stage 2 tương ứng).

## Luồng chạy

```
python main.py --mode register   # Stage 1: mở GUI nhập tên + lấy mẫu, lưu logs/registry/<tên>.npz
python main.py --mode run        # Stage 2: mở GUI chọn người trong registry, rồi mở webcam bám theo người đã chọn
```

`main.py --mode run` mở [PersonRegistrySelector](#srcperson_selectorpy--personregistryselector-chọn-người-trước-khi-chạy-stage-2)
TRƯỚC KHI đụng tới webcam/pipeline — nếu registry rỗng, tự động chuyển sang GUI registration
(giữ đúng tinh thần auto-redirect đã thống nhất trước đây); nếu người dùng đóng GUI mà không
chọn ai, `main.py` thoát sạch sẽ với lỗi rõ ràng, không chạy tiếp với dữ liệu không hợp lệ.

---

## Chi tiết từng module

### `src/types.py`

`AngleResult` — dataclass đơn giản là kiểu trả về duy nhất của `FollowPipeline.process_frame()`:

| Field | Kiểu | Ý nghĩa |
|---|---|---|
| `target_found` | `bool` (bắt buộc) | Có tìm thấy target đủ tin cậy trong frame này không |
| `angle_offset_deg` | `float \| None` | Góc lệch ngang (độ) của target so với tâm khung hình; âm = lệch trái, dương = lệch phải |
| `size_ratio` | `float \| None` | `bbox_height / frame_height` — proxy khoảng cách thô, chưa qua IPM |
| `track_id` | `int \| None` | ID track (từ ByteTrack) của target đang khóa |
| `similarity_score` | `float \| None` | Cosine similarity giữa embedding track này và embedding tham chiếu |

Khi `target_found=False`, 4 field còn lại giữ nguyên `None` — không có giá trị rác.

### `src/detector.py` — `YoloDetector`

Bọc model YOLOv11 (`ultralytics.YOLO`) để detect + track người (COCO class 0).

- `__init__(model_path)`: load model; nếu path rõ ràng không tồn tại và không giống tên
  model chuẩn của Ultralytics → raise `FileNotFoundError` kèm đường dẫn tuyệt đối đã thử.
  Mọi lỗi khác khi load model (kể cả lỗi download) đều bị bắt và bọc lại thành
  `FileNotFoundError` có thông tin rõ ràng, không để traceback thư viện lộ ra.
- `track(frame) -> list[dict]`: gọi `model.track(frame, persist=True, classes=[0],
  tracker="bytetrack.yaml")` — dùng ByteTrack tích hợp sẵn trong Ultralytics, không tự viết
  tracker. **`tracker="bytetrack.yaml"` phải chỉ định rõ** (sửa 2026-08-13, xem log mục 16) —
  nếu bỏ trống, Ultralytics âm thầm dùng BoT-SORT thay vì ByteTrack, gây warning `GMC failed`
  liên tục khi input đổi kích thước giữa các lần gọi (đúng trường hợp ROI-constrained detection
  ở `pipeline.py`). Trả về **mọi** track hiện có trong frame, mỗi phần tử dạng
  `{"track_id": int, "bbox": (x1,y1,x2,y2), "confidence": float}`. Nếu track chưa có `id`
  (thường ở frame đầu của 1 track mới), gán ID tạm `idx + 1000` và log warning — **đúng theo
  spec**, không phải bug.

### `src/verifier.py` — `OSNetVerifier`

**Thay thế `MobileNetVerifier` (2026-08-12)** — xem [Stage1_2_Implementation_Review_Log.md](Stage1_2_Implementation_Review_Log.md)
mục 11 để biết lý do đầy đủ (số liệu Scenario A/B chứng minh embedding MobileNetV3-ImageNet
không đủ phân biệt danh tính người). Dùng OSNet — model được train sẵn cho đúng bài toán
person re-identification (qua thư viện `torchreid`), không phải model classification chung
như MobileNetV3.

- `__init__(variant)`: load OSNet qua `torchreid.utils.FeatureExtractor`, variant mặc định
  `osnet_x1_0`. **Quan trọng**: `FeatureExtractor(pretrained=True)` mặc định của torchreid
  chỉ tải weight pretrained **ImageNet-classification** (giống hệt vấn đề của MobileNetV3) —
  KHÔNG phải weight đã train cho re-id. Code tự tải đúng checkpoint đã fine-tune trên dataset
  re-id thật (Market1501 cho `osnet_x1_0`; MSMT17+DukeMTMC+CUHK03 cho `osnet_ain_x1_0`, chọn
  vì tính robust cross-domain — xem comment chi tiết trong `verifier.py`) từ Google Drive
  (Model Zoo chính thức của torchreid) về `models/{variant}_reid.pth`, cache lại cho lần sau.
- Input size: **256×128 (H×W)**, khác 224×224 vuông của MobileNetV3 — do `FeatureExtractor`
  tự resize + normalize (ImageNet mean/std) nội bộ, `extract()` không tự làm lại bước này.
- `extract(image_crop) -> np.ndarray`: BGR→RGB, forward qua `FeatureExtractor`, L2-normalize
  vector output 512-chiều (giống nhau cho cả `osnet_x1_0` và `osnet_ain_x1_0`).
- `compare(embedding_a, embedding_b) -> float`: giữ nguyên logic cũ — cosine similarity qua
  dot product của 2 vector đã L2-normalize.
- Giữ nguyên tối ưu `torch.set_num_threads(1)` khi chạy CPU (tránh oversubscription khi dùng
  chung `ThreadPoolExecutor` với `pipeline.py`).

**Cài `torchreid`** (không có sẵn trong `pip install -r requirements.txt` — xem comment đầu
`requirements.txt` để biết lý do). Quy trình đã test và xác nhận chạy được trên máy Windows:

```bash
git clone --depth 1 https://github.com/KaiyangZhou/deep-person-reid.git /tmp/deep-person-reid
.venv/Scripts/pip.exe install -r /tmp/deep-person-reid/requirements.txt
cp -r /tmp/deep-person-reid/torchreid .venv/Lib/site-packages/torchreid
```

Lý do dùng `cp` thay vì `pip install -e .`: cả PyPI package `torchreid` (fork bên thứ 3,
thiếu khai báo dependency, `import torchreid` fail vì thiếu `scipy`) lẫn `pip install -e .`
từ source chính chủ (setup.py cần `numpy`+`Cython` ở BUILD time để build 1 extension Cython
tùy chọn, nhưng build isolation của pip không có sẵn 2 package đó) đều fail. Copy thẳng
thư mục package (thuần Python) vào `site-packages` bỏ qua hẳn bước build extension đó —
`torchreid` tự fallback sang eval bằng Python thuần (chỉ warning, không lỗi) khi thiếu
extension này, và verifier của ta chỉ cần `FeatureExtractor` (không cần extension đó).

### `src/registry.py` — data layer cho Person Registry (không có Tkinter)

Toàn bộ logic đọc/ghi `.npz` cho registry nhiều người, dùng chung bởi `registration.py`,
`person_selector.py`, và `pipeline.py`:

- `sanitize_person_name(raw_name) -> str`: chỉ giữ lại chữ (kể cả có dấu/Unicode), số,
  `_`, `-`; khoảng trắng → `_`; các ký tự khác (bao gồm `.`, `/`, `\`) bị loại bỏ hoàn
  toàn — ngăn path traversal bằng cấu trúc (không phải chỉ pattern-match `../`). Raise
  `ValueError` nếu tên rỗng hoặc không còn ký tự hợp lệ nào sau khi sanitize.
- `save_person(name, embedding, aspect_ratio, sample_count) -> path`: ghi
  `logs/registry/<sanitized_name>.npz` (`np.savez`, `allow_pickle=False`) gồm `embedding`,
  `aspect_ratio`, `created_at` (ISO timestamp), `sample_count`. Cùng tên → ghi đè (registry
  dùng tên làm khóa duy nhất, không cho phép trùng).
- `load_person(path)`, `list_registry()` (bỏ qua file lỗi/không đọc được, chỉ log warning
  chứ không crash cả danh sách), `person_exists(name)`, `delete_person(name)`,
  `rename_person(old, new)` (raise `FileNotFoundError`/`FileExistsError` rõ ràng, không bao
  giờ xóa file gốc nếu rename thất bại).

### `src/registration.py` — `TargetRegistrar` / `TargetRegistrarGUI` (Stage 1)

Công cụ lấy mẫu ảnh người mục tiêu và sinh embedding tham chiếu. Triển khai dưới dạng
GUI Tkinter đầy đủ (mở rộng hơn so với luồng OpenCV thuần mô tả trong `plan.md`), gồm:
live camera feed với ROI overlay (làm mờ vùng ngoài ROI), **ô nhập tên người** (bắt buộc
trước khi bắt đầu), slider chỉnh ROI trực tiếp và lưu lại vào `settings.yaml`, progress bar
số mẫu đã thu, gallery thumbnail các crop đã lấy, và trạng thái máy hữu hạn
`WAITING_SPACE → COUNTDOWN → COLLECTING → COMPLETED/FAILED`.

Luồng lấy mẫu (`_video_loop`):
1. Người dùng nhập tên vào ô Entry, nhấn SPACE hoặc nút Start. `start_sampling()` validate
   tên qua `registry.sanitize_person_name()` trước (báo lỗi rõ nếu rỗng/không hợp lệ, không
   cho bắt đầu); nếu tên đã tồn tại trong registry → hỏi xác nhận GHI ĐÈ (không tự động ghi
   đè im lặng). **Lưu ý xử lý phím SPACE**: vì SPACE cũng được bind toàn cục để bấm Start
   nhanh, nếu không guard thì gõ tên có khoảng trắng (vd "Nguyen Van A") sẽ vô tình trigger
   Start giữa chừng — `start_sampling()` kiểm tra `root.focus_get()` và bỏ qua nếu đang gõ
   trong ô tên. Qua bước validate → đếm ngược `registration_countdown_sec` giây.
2. Trong `registration_duration_sec` giây, cứ mỗi `registration_sample_interval_frames`
   frame lại chạy `YoloDetector.track()` trên vùng ROI:
   - Đúng 1 người trong ROI → crop, lưu vào danh sách mẫu.
   - 0 hoặc >1 người → skip frame, tăng `skipped_frames`, log warning (không raise).
3. Kết thúc thu thập (`_finish_collection`): nếu số mẫu hợp lệ < `registration_min_samples`
   → chuyển state `FAILED` + set `self.success = False` (wrapper `TargetRegistrar.run()` sẽ
   raise `RuntimeError` dựa vào cờ này — xem mục Vấn đề đã biết bên dưới về lịch sử bug này).
   Nếu đủ mẫu: trích embedding từng crop (đã L2-norm) song song bằng `ThreadPoolExecutor`,
   lấy mean vector, normalize lại lần nữa; tính aspect ratio (`bbox_width / bbox_height`,
   tọa độ bbox gốc lúc detect, không phải sau resize cho model) của từng crop, lấy **median**
   (ổn định hơn mean với outlier); gọi `registry.save_person(name, embedding, aspect_ratio,
   sample_count)` — lưu TẤT CẢ vào 1 file `.npz` duy nhất trong `logs/registry/`.
- `TargetRegistrar.run() -> str`: chạy GUI, raise nếu fail, trả về path `.npz` đã lưu khi
  thành công.

### `src/person_selector.py` — `PersonRegistrySelector` (chọn người trước khi chạy Stage 2)

GUI Tkinter hiện TRƯỚC KHI `main.py --mode run` đụng tới webcam/pipeline, để chọn AI trong
số người đã đăng ký sẽ là target phiên chạy đó (pipeline vẫn chỉ theo 1 người mỗi lần chạy —
tính năng này chỉ đổi CÁCH CHỌN người đó, không phải multi-target).

- `run() -> Optional[str]`: đọc `registry.list_registry()`. Nếu **rỗng** → tự động gọi
  `_register_new_blocking()` (mở luồng đăng ký mới) rồi đọc lại danh sách; nếu vẫn rỗng
  (đăng ký thất bại/bị hủy) → trả `None`. Nếu có người → mở bảng danh sách
  (`_show_list_and_wait`, dùng `ttk.Treeview` hiển thị tên/`created_at`/`sample_count`) và
  chờ người dùng thao tác, trả về `self.selected_path` (vẫn `None` nếu đóng cửa sổ mà không
  chọn ai — `main.py` dựa vào giá trị `None` này để thoát sạch, không chạy tiếp với dữ liệu
  không hợp lệ).
- Action trên bảng danh sách: **Select & Run** (đặt `selected_path`, đóng cửa sổ), **Delete**
  (hỏi xác nhận trước khi xóa thật, không xóa im lặng), **Rename** (dialog nhập tên mới,
  validate qua `registry.rename_person`), **Register New** (mở lại luồng đăng ký, sau đó
  refresh danh sách tại chỗ, không đóng cửa sổ chọn).
- `run()` cố tình được tách thành các method nhỏ (`_register_new_blocking`,
  `_show_list_and_wait`) thay vì 1 khối dựng Tk liền — nhờ vậy logic điều phối (auto-redirect
  khi rỗng, trả `None` khi đóng không chọn) test được bằng cách mock 2 method đó, không cần
  mở Tk mainloop thật (xem `test_person_selector.py`).

### `src/detector.py` — dùng trong ROI-constrained detection (Stage 2)

`YoloDetector.track()` không đổi interface, nhưng từ khi có ROI-constrained detection,
`pipeline.py` có thể gọi nó với 1 **crop** của frame thay vì cả frame — xem mục
`src/pipeline.py` bên dưới về cách tọa độ bbox trả về được convert lại về full-frame.

### `src/pipeline.py` — `FollowPipeline` (interface chính Stage 2)

Class duy nhất mà `main.py` (và sau này module điều khiển chuyển động trong codebase
lớn hơn) cần import và gọi. Toàn bộ logic detect/verify/tính góc nằm sau lớp này.

- `__init__(config_path, reference_npz_path)`: load config YAML, load embedding + aspect
  ratio tham chiếu qua `registry.load_person(reference_npz_path)` (bắt buộc — raise
  `ValueError` nếu thiếu param, `FileNotFoundError` nếu file không tồn tại), khởi tạo
  `YoloDetector` + `OSNetVerifier`, mở file CSV log, khởi tạo `ThreadPoolExecutor` để trích
  embedding song song cho nhiều track cùng lúc. `reference_npz_path` là path do
  `PersonRegistrySelector` (hoặc `TargetRegistrar.run()`'s return value) chọn ra, KHÔNG còn
  là 1 file cố định như trước.
- `process_frame(frame) -> AngleResult`:
  1. **Chọn detect toàn frame hay ROI-constrained** (xem mục riêng bên dưới).
  2. Với mỗi track, chạy song song (thread pool): crop bbox (**luôn từ full frame gốc**,
     không phải ROI crop — verification không bị giới hạn ROI) → `verifier.extract()` →
     `verifier.compare()` với embedding tham chiếu → `raw_score`.
  3. Tính **aspect ratio gate** cho từng track (xem mục riêng bên dưới) — độc lập với
     similarity, tính lại mỗi frame từ bbox hiện tại (không smooth theo thời gian).
  4. Temporal smoothing (EMA/voting) áp dụng lên `raw_score` như cũ, ra `similarity_is_pass`.
     `is_pass` cuối cùng = `similarity_is_pass AND aspect_ratio_pass`.
  5. Ghi **mọi** track (không chỉ target) vào `logs/verification_log.csv` — dữ liệu bắt
     buộc để calibrate threshold sau này bằng dữ liệu thực tế.
  6. **Sticky target logic**: nếu track đang là target (`active_target_id`) vẫn còn tồn
     tại trong frame này và `is_pass` (đã combine) vẫn đúng → giữ nguyên track đó làm
     target. Chỉ khi track cũ biến mất hoặc `is_pass` sai mới tìm track có `smoothed_val`
     cao nhất (trong số `is_pass=True`) làm target mới.
  7. Nếu không có target hợp lệ → trả `AngleResult(target_found=False)`.
  8. Nếu có: tính `offset_x_norm = (bbox_center_x - frame_w/2) / (frame_w/2)`, rồi
     `angle_offset_deg = degrees(atan(offset_x_norm * tan(FOV_horizontal/2)))` (công thức
     lượng giác chính xác, không phải xấp xỉ tuyến tính). `size_ratio = bbox_height / frame_h`.
- `close()`: shutdown thread pool khi dừng.

#### Dynamic ROI-constrained detection

Tối ưu compute: khi đã có `active_target_id` sticky-tracked VÀ biết vị trí bbox của nó ở
frame trước (`last_detections`), thay vì chạy YOLO trên cả frame, `pipeline.py` crop 1 vùng
quanh vị trí đó (mở rộng thêm `roi_margin_percent` kích thước bbox mỗi chiều, clamp trong
biên frame) và chỉ chạy YOLO trên crop đó — bbox trả về (tọa độ cục bộ trong crop) được
**cộng lại offset ROI** để quy về tọa độ full-frame trước khi dùng ở mọi bước sau (đây là
lỗi hay gặp nếu quên convert, `angle_offset_deg` sẽ sai hoàn toàn).

- Chưa có target sticky-tracked (mới khởi động, hoặc active_target_id vừa None) → luôn detect
  toàn frame, không có ROI nào để dùng.
- ROI-constrained detect KHÔNG tìm thấy ai (`detections` rỗng): **không** coi là mất target
  ngay — chỉ tăng `_roi_failure_count`, giữ nguyên `active_target_id`/`last_detections` để
  frame sau vẫn thử lại ROI quanh vị trí cũ. Đây là điểm suy luận thêm ngoài spec chữ: nếu để
  hành vi cũ (mất track hoàn toàn ngay khi 0 detection) áp dụng luôn cho case này, cơ chế
  fallback `roi_failure_max_frames` sẽ không bao giờ có cơ hội chạy — 1 frame ROI hụt là mất
  target ngay, không tích lũy được failure count.
- ROI-constrained detect có track nhưng không track nào pass `is_pass` (similarity AND aspect
  ratio) → cũng tăng `_roi_failure_count`.
- `_roi_failure_count >= roi_failure_max_frames` → frame kế tiếp ép detect toàn frame, reset
  count về 0 ngay lúc quyết định (không đợi kết quả frame đó).
- Tìm được target hợp lệ (dù qua ROI hay full-frame) → reset `_roi_failure_count = 0`.
- **Verification KHÔNG bị giới hạn ROI** — chạy trên mọi track mà bước detect (dù ROI hay
  toàn frame) trả về.
- **Rủi ro kiến trúc đã XÁC NHẬN THẬT qua webcam + đã sửa** (2026-08-13, xem log mục 16): frame
  đổi kích thước liên tục giữa ROI/full-frame khiến bước GMC (global motion compensation) của
  tracker mặc định BoT-SORT liên tục fail (`GMC failed, falling back to identity`). Nguyên nhân
  sâu hơn: `detector.py` chưa từng chỉ định `tracker=` nên Ultralytics âm thầm dùng BoT-SORT
  thay vì **ByteTrack** như spec gốc/docstring vẫn luôn nói — lệch spec có từ đầu, chỉ lộ ra khi
  ROI làm frame size đổi. Đã sửa: chỉ định rõ `tracker="bytetrack.yaml"` — ByteTrack không có
  bước GMC nên loại bỏ hẳn lớp lỗi này (không chỉ ẩn warning), đã verify bằng test feed frame
  đổi kích thước liên tục, 0 warning.

#### Aspect ratio hard gate

Gate cứng độc lập với similarity ngoại hình, giảm nhầm người mặc đồ giống nhau nhưng vóc
dáng khác (`bbox_width / bbox_height` khác biệt rõ):

```
candidate_ar = bbox_width / bbox_height
ar_diff_ratio = abs(candidate_ar - reference_aspect_ratio) / reference_aspect_ratio
aspect_ratio_pass = ar_diff_ratio <= aspect_ratio_tolerance_percent
```

Tính **mỗi frame, từ bbox hiện tại** (không smooth theo EMA/voting — giữ `smoothed_val` là
tín hiệu similarity thuần, tách biệt để dễ calibrate qua CSV riêng), rồi `AND` với kết quả
similarity **sau khi** đã qua EMA/voting để ra `is_pass` cuối cùng. **Lưu ý diễn giải spec**:
yêu cầu gốc có 2 câu mô tả hơi khác nhau về việc gate nên áp dụng TRƯỚC hay SAU temporal
smoothing — đã chọn "SAU" (AND với `is_pass` đã smooth) vì giữ `smoothed_val`/`vote_ratio`
là tín hiệu similarity sạch, không lẫn 2 nguồn tín hiệu vào 1 số, khớp với yêu cầu logging
riêng cột `candidate_aspect_ratio`/`ar_diff_ratio`/`aspect_ratio_pass` để calibrate độc lập.
Nếu ý định ban đầu là gate TRƯỚC smoothing (ảnh hưởng đến chính giá trị EMA/vote_ratio), cần
sửa lại đoạn này trong `pipeline.py` (đã ghi rõ trong code comment tại vị trí liên quan).

### `main.py`

Entry point CLI (`argparse`), 2 mode. Không còn flag `--embedding` — registry thay hẳn vai
trò đó (xem `src/registry.py`/`src/person_selector.py` phía trên).

- `--mode register`: gọi `TargetRegistrar.run()` để mở GUI Stage 1 (nhập tên + lấy mẫu), log
  path `.npz` đã lưu.
- `--mode run`: mở `PersonRegistrySelector` TRƯỚC KHI đụng tới webcam/`FollowPipeline` — chọn
  người xong mới khởi tạo pipeline với đúng path `.npz` đã chọn. Nếu selector trả về
  `None` (đóng cửa sổ không chọn ai, hoặc registry rỗng và đăng ký tự động cũng thất bại) →
  `sys.exit(1)` với lỗi rõ ràng, KHÔNG chạy tiếp — đây chính là loại silent-failure (chạy với
  dữ liệu cũ/rỗng còn sót) mà spec yêu cầu tránh lặp lại. Sau khi có pipeline: mở webcam qua
  `WebcamStreamThread` (đọc frame trong thread riêng để không block vòng lặp xử lý), loop: đọc
  frame → `pipeline.process_frame()` → (nếu `--ui`) vẽ overlay debug → thoát khi nhấn `q` (chế
  độ `--ui`) hoặc Ctrl+C (headless). FPS đo thực tế theo thời gian xử lý mỗi giây, log qua
  `logging` module (không dùng `print()`).

**`--ui`** (mặc định TẮT — chạy headless/background): bật cửa sổ debug hiển thị bbox mỗi
track (xanh=target, đỏ=track khác, tag cam "AR-MISMATCH" nếu track bị aspect ratio gate chặn
dù similarity cao), **vùng ROI-constrained detection đang dùng** (hình chữ nhật cyan, di
chuyển theo target — đúng thứ người dùng muốn thấy), mũi tên hướng target, và panel telemetry
(FPS, mode, ROI đang dùng hay full-frame, số lần ROI fail liên tiếp, trạng thái warm-up).
Toàn bộ logic vẽ overlay nằm ở [src/debug_overlay.py](src/debug_overlay.py) — file riêng,
tách khỏi `main.py`, chỉ được `import` khi có `--ui` để nhánh headless (mặc định, dùng khi
triển khai thật/background) không phải cõng theo bất kỳ lệnh gọi `cv2` GUI nào. Ở chế độ
headless, dòng log FPS mỗi giây (`logger.info`) là kênh duy nhất để biết pipeline đang làm
gì (ROI hay full-frame, đã lock target chưa, ROI fail bao nhiêu lần) — không cần cửa sổ.

```bash
.venv/Scripts/python.exe main.py --mode run          # headless — dùng cho chạy nền/production
.venv/Scripts/python.exe main.py --mode run --ui      # mở cửa sổ debug (bbox, ROI, telemetry)
```

### `config/settings.yaml`

Toàn bộ số cấu hình dùng chung cho mọi module — không có giá trị nào bị hard-code đè
trong code khi đã có key tương ứng ở đây.

| Key | Ý nghĩa |
|---|---|
| `camera_index` | Chỉ số webcam OpenCV mở |
| `input_resolution` | `[width, height]` resize frame về trước khi xử lý |
| `yolo_model_path` | Đường dẫn weight YOLO (`.pt` hoặc `.onnx`) |
| `osnet_variant` | `osnet_x1_0` (Market1501, mặc định) hoặc `osnet_ain_x1_0` (cross-domain) |
| `similarity_threshold` | Ngưỡng cosine similarity để coi là "đúng người" — **placeholder**, giữ nguyên giá trị từ thời MobileNetV3, CHƯA calibrate lại cho OSNet — cần đo Scenario A/B mới |
| `roi_percent` | `[x1,y1,x2,y2]` theo % kích thước frame, vùng ROI dùng khi registration |
| `registration_countdown_sec` / `registration_duration_sec` / `registration_sample_interval_frames` / `registration_min_samples` | Tham số quy trình lấy mẫu Stage 1 |
| `camera_fov_horizontal_deg` | Góc nhìn ngang camera (độ) — **giá trị mẫu**, cần xác nhận lại với webcam thực tế đang dùng |
| `roi_margin_percent` | Mở rộng ROI-constrained detect thêm % kích thước bbox target frame trước, mỗi chiều — **placeholder** |
| `roi_failure_max_frames` | Số frame ROI-constrained thất bại liên tiếp trước khi ép quét lại toàn frame — **placeholder** |
| `aspect_ratio_tolerance_percent` | Sai số aspect ratio tối đa cho phép so với `reference_aspect_ratio` — **placeholder**, 30% là điểm khởi đầu rộng |

## Kiểm thử

```bash
.venv/Scripts/python.exe -m unittest test_pipeline -v         # temporal smoothing, ROI-constrained detect, aspect ratio gate — mock verifier
.venv/Scripts/python.exe -m unittest test_verifier -v         # OSNetVerifier thật, ảnh giả — cần model đã tải (hoặc mạng lần đầu)
.venv/Scripts/python.exe -m unittest test_registry -v         # sanitize + CRUD registry (.npz) — thuần logic, không Tkinter
.venv/Scripts/python.exe -m unittest test_person_selector -v  # điều phối PersonRegistrySelector.run() — mock 2 method dựng Tk
```

`test_pipeline.py` mock `YoloDetector`/`OSNetVerifier` nên chạy nhanh, không cần camera/model
thật — bao gồm test riêng cho ROI crop + coordinate conversion, ROI fallback sau
`roi_failure_max_frames` lần thất bại liên tiếp, và aspect ratio gate chặn đúng track có
similarity cao nhưng sai hình dáng. `test_verifier.py` load model OSNet thật (không mock) để
verify `extract()` trả đúng shape 512-dim đã L2-normalize và `compare()` trả giá trị hợp lệ
trong [-1, 1]. `test_registry.py` verify sanitize (khoảng trắng/ký tự đặc biệt/path traversal/
rỗng) và CRUD (nhiều người tồn tại độc lập, delete chỉ xóa đúng người, rename giữ nguyên data
và không đè lên tên đã có). `test_person_selector.py` verify logic auto-redirect khi registry
rỗng và trả về `None` khi đóng cửa sổ không chọn ai — bằng cách mock `_register_new_blocking`/
`_show_list_and_wait` (2 method duy nhất chạm Tkinter), không cần mở cửa sổ thật, theo đúng
cách `PersonRegistrySelector` được thiết kế để test được.

## Vấn đề đã biết

Xem [Stage1_2_Implementation_Review_Log.md](Stage1_2_Implementation_Review_Log.md) mục 7-9,
11-14 để biết lịch sử đầy đủ. Tình trạng tính tới lần cập nhật gần nhất (2026-08-13):

- **FPS thấp (3-5fps) khi chạy thật — đã đo được nguyên nhân gốc bằng số liệu thật trên máy
  dev (CPU, không GPU)**: YOLO detect ~64ms/frame, OSNet verify ~76ms/track — CHỈ RIÊNG 1
  người trong khung đã mất ~140ms/frame (~7fps trần lý thuyết), chưa tính overhead webcam
  I/O/vẽ overlay/CSV. OSNet nặng hơn MobileNetV3 cũ đáng kể (~76ms vs ~10-14ms/crop theo
  benchmark cũ ở log mục 4.2) — đây là **đánh đổi có thật** giữa độ chính xác re-id (lý do đổi
  sang OSNet ở mục 11) và tốc độ, không phải bug. Đã thêm timing per-stage
  (`pipeline.last_timing_ms`, hiện trong log FPS mỗi giây và panel debug `--ui`) để đo trực
  tiếp trên máy thật của người dùng thay vì đoán.
- **Đã thêm 3 đòn bẩy hiệu năng mới (2026-08-16), tất cả đều opt-in qua config, KHÔNG đổi
  hành vi/accuracy mặc định**, để nhắm mục tiêu >15fps trên máy dev / ~10fps trên Pi 5:
  - `OSNetVerifier.extract_batch()` ([src/verifier.py](src/verifier.py)): gộp N crop trong 1
    frame thành 1 lần forward pass OSNet thay vì N lần `extract()` tuần tự. `src/pipeline.py`
    dùng path này mặc định. Đo được trên máy dev: tăng tốc ~1.1-1.4x (khiêm tốn — batching
    KHÔNG giảm tổng FLOPs, chỉ giảm overhead Python/preprocessing lặp lại; xem
    `test_verifier.py::OSNetVerifierBatchingSpeedupTestCase` để tái lập).
  - `verify_every_n_frames` config ([config/settings.yaml](config/settings.yaml), mặc định 1 =
    hành vi cũ): track KHÔNG phải active target chỉ được OSNet+pose re-verify mỗi N frame,
    tái dùng `raw_score` gần nhất ở các frame giữa (ByteTrack tự giữ track_id). Đòn bẩy LỚN
    hơn batching nhiều — với trung bình ~9 người/frame đo được trên video ablation
    ([notebooks/test1_single_frame.ipynb](notebooks/test1_single_frame.ipynb) Section 4b),
    N=3 giảm ~2.4x số lần gọi OSNet/frame. Active target và track mới luôn verify mỗi frame
    bất kể N (xem `test_pipeline.py` phần "Verification frame-skip").
  - `detection_imgsz` config: truyền thẳng xuống `model.track(..., imgsz=...)`, None (mặc
    định) = giữ nguyên hành vi ultralytics (640).
  - `osnet_variant` giờ hỗ trợ thêm `osnet_x0_75`/`osnet_x0_5`/`osnet_x0_25` (Market1501
    same-domain checkpoints thật từ Model Zoo, không phải ImageNet backbone) — đánh đổi
    accuracy lấy tốc độ có kiểm soát: x0_5 mất ~3.4% mAP so với x1_0 baseline, x0_25 mất
    ~9.2% mAP (ngay trong ngân sách 5-10% đã thống nhất). CHƯA áp dụng cho video ablation
    hiện tại (notebook cố định `osnet_x1_0` để đo baseline accuracy).
  - **Chưa đo trên Raspberry Pi 5 thật** (không có hardware) — các con số trên chỉ đo trên máy
    dev CPU. NCNN export / INT8 quantization (khuyến nghị ~46% latency reduction theo benchmark
    công khai cho YOLO11n trên Pi 5) CHƯA implement — xem phần "Còn mở" bên dưới.
- **Đã sửa cấu hình capture webcam** ([src/camera_utils.py](src/camera_utils.py)): trước đây
  `cv2.VideoCapture()` không set FOURCC/resolution/buffer, có thể khiến camera fallback về
  định dạng raw/độ phân giải thấp hơn rồi bị code tự upscale mờ, và buffer mặc định làm frame
  đọc được bị trễ (stale). Giờ ép MJPG + resolution theo `input_resolution` + buffer size 1,
  log ra resolution/FPS/FOURCC THỰC TẾ camera trả về để biết camera có tuân theo yêu cầu hay
  không — CHƯA xác nhận bằng webcam thật, cần người dùng tự chạy lại và xem log.

- **Đã sửa**: thiếu `import cv2` trong `pipeline.py`; `registration.py` không raise khi số mẫu
  không đủ; `save_roi_config()` xóa comment trong `settings.yaml` (đổi sang patch text tại chỗ);
  thiếu `Pillow` trong `requirements.txt`; bug off-by-one ở Cơ chế 2 (startup warm-up) trong
  temporal smoothing khiến chỉ chặn N-1 frame thay vì N frame như spec; thay verifier từ
  MobileNetV3 sang OSNet (mục 11); thêm dynamic ROI-constrained detection + aspect ratio hard
  gate (mục 9); thêm debug UI tách file + headless mặc định (mục 12); chuyển từ single
  `reference_embedding.npy` sang Person Registry nhiều người (mục 13).
- **[QUAN TRỌNG] Phải chạy lại `--mode register`** trước khi dùng `--mode run`: toàn bộ model
  lưu trữ cũ (`logs/reference_embedding.npy` + companion `.npy`) đã bị loại bỏ, thay bằng
  `logs/registry/<tên>.npz` — không tự động migrate theo đúng yêu cầu (camera đã đổi góc từ
  lần đăng ký cũ, dữ liệu cũ không còn hợp lệ để dùng tiếp). Tất cả file `.npy` cũ đã đổi tên
  `_OLD_*.bak` để tham khảo, `logs/registry/` hiện đang rỗng — lần chạy `--mode run` kế tiếp
  sẽ tự động mở GUI đăng ký (đúng thiết kế). Đây là lần thứ 3 phải re-register trong phiên làm
  việc gần đây (MobileNetV3→OSNet, thêm aspect ratio gate, chuyển sang registry) — nếu tiếp
  tục thêm feature cần data mới ở registration, cân nhắc gộp các thay đổi lại để giảm số lần
  người dùng phải đứng trước camera đăng ký lại.
- **`similarity_threshold` (hiện `0.80`) chưa được calibrate lại cho OSNet** — giá trị này là
  mốc cũ từ MobileNetV3, phân phối similarity của OSNet gần như chắc chắn khác hẳn (thường có
  khoảng cách rõ hơn giữa target thật/người lạ, nhưng cần đo lại bằng đúng quy trình Scenario
  A/B đã dùng trước đó để xác nhận, KHÔNG giả định).
- **`roi_margin_percent`, `roi_failure_max_frames`, `aspect_ratio_tolerance_percent` đều là
  placeholder chưa calibrate** — cần dữ liệu thực tế (cột `used_roi`/`roi_bounds`/
  `candidate_aspect_ratio`/`ar_diff_ratio`/`aspect_ratio_pass` mới trong
  `verification_log.csv`) trước khi coi là giá trị cuối.
- **Bug thật đã tìm + sửa qua webcam thật (mục 15-16 log)**: (a) ROI thất bại (gate reject 1
  frame) từng xóa `active_target_id` ngay thay vì retry — sửa xong, có regression test; (b)
  `detector.py` chưa từng chỉ định `tracker=` nên chạy BoT-SORT thay vì ByteTrack như spec gốc
  — gây warning `GMC failed` liên tục khi ROI đổi kích thước frame — sửa xong bằng
  `tracker="bytetrack.yaml"`. Xem chi tiết ở mục `src/pipeline.py` phía trên.
- **Diễn giải spec cần xác nhận lại**: thứ tự aspect ratio gate AND với similarity — đã chọn
  áp dụng SAU khi similarity qua EMA/voting (không phải trước) — xem lý do và cách đổi lại
  nếu cần ở mục `src/pipeline.py` phía trên ("Aspect ratio hard gate").
- **Còn mở, có chủ đích chưa làm**: benchmark hiệu năng trên Raspberry Pi 5 thật + đo baseline
  "sequential có `torch.set_num_threads(1)`" — chưa có hardware. OSNet (2.2M param) nặng hơn
  MobileNetV3-small nhưng input nhỏ hơn (256×128 vs 224×224) — chưa đo FPS thực tế trên máy dev
  lẫn Pi 5. ROI-constrained detection kỳ vọng giảm compute khi target đã sticky nhưng cũng
  CHƯA đo FPS trước/sau bằng webcam thật.
- `main.py --mode run` vẫn tự động redirect sang registration khi registry rỗng thay vì raise
  lỗi cứng như `plan.md` yêu cầu — đây là thay đổi có chủ đích của người dùng (xem log mục 4.1),
  không phải bug.
- **`PersonRegistrySelector` (GUI chọn người) chưa được xác nhận trực quan bằng webcam/click
  thật** — chỉ unit test được phần logic điều phối (mock 2 method chạm Tkinter, xem mục Kiểm
  thử). Việc hiển thị bảng danh sách, nút Rename/Delete/Register New có hoạt động đúng khi
  người dùng thật sự click chưa được verify — cùng hạn chế như `TargetRegistrarGUI` trước đây
  (GUI Tkinter nặng, chỉ smoke-test được ở mức construction/logic, không phải pixel-level).
