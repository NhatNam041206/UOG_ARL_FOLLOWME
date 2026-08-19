> **Provenance:** copied verbatim from
> `D:\GW_UNIVERSITY\AIS\AUTOBOT_V2\document\implementation\followme\Project_Master_Doc.md`
> (outside this repo) so the tech report is self-contained. Treat that external path as the
> source of truth if the two ever diverge — this is a snapshot, not a synced copy.
>
> **Note:** this spec references `CV_Verification_Handoff_Doc.md` throughout (muc 7, 9.3, 4.2,
> 5.1, 8.1). That file does not exist anywhere in this repository or the broader
> `D:\GW_UNIVERSITY\AIS\AUTOBOT_V2` tree (confirmed via a full-tree search at implementation
> time). Implementation proceeded using `src/pipeline.py`'s own inline comments instead, which
> independently document the same ROI-coordinate-conversion and sticky-target concerns the spec
> points to that missing doc for. See `implementation_audit.md` §"Deviations" for detail.

---

# QUICK DEMO SPEC — Wave + Facing Trigger Gate
## Bàn giao cho Claude Agent để code (KHÔNG tự code trong phiên thiết kế này)

> Tài liệu này CHỈ đặc tả demo nhanh để chứng minh khả thi (proof-of-concept), KHÔNG phải bản
> triển khai cuối cùng. Mọi ngưỡng số trong tài liệu đều là PLACEHOLDER cần calibrate lại bằng
> dữ liệu thật sau demo — theo nguyên tắc "empirical discipline" của project (xem
> `CV_Verification_Handoff_Doc.md` mục 7).

---

## 1. MỤC TIÊU DEMO (định nghĩa "xong") — ĐÃ CẬP NHẬT

Webcam trực tiếp → **dùng lại pipeline Follow-Me hiện có** (`src/{pipeline,detector,verifier}.py`,
đã có sẵn theo `CV_Verification_Handoff_Doc.md`) để detect + track + verify identity, LẤY bbox
của người đã match với 1 registered person đã đăng ký thật (không giả lập bằng flag `True` nữa)
→ crop đúng vùng bbox đó mỗi frame → feed crop vào MoveNet Lightning → rule-based wave detector +
facing proxy → hiển thị boolean trigger lên màn hình.

**Thay đổi so với bản đầu tiên của tài liệu này:** `registered_person` KHÔNG còn là giá trị giả
định cố định `True` nữa — đây là kết quả THẬT lấy từ pipeline verification đã có sẵn của project
(OSNet + các gate hiện có, dù đang ở mức độ hoàn thiện nào tại thời điểm demo — xem mục 8.1 của
`CV_Verification_Handoff_Doc.md` để biết phần nào đã code xong). Điều này khiến demo GẦN với thực
tế triển khai cuối cùng hơn nhiều so với việc test wave-detector trên khung hình đầy đủ hoặc trên
"bất kỳ ai".

**KHÔNG bao gồm trong demo này:**
- Approach/steering logic sau khi trigger.
- Emergency-stop, occlusion handling, signal-loss recovery.
- Sửa đổi logic bên trong `pipeline.py`/`detector.py`/`verifier.py` hiện có — demo này CHỈ ĐỌC
  output của chúng (bbox của track đã verify là registered person), KHÔNG thay đổi hành vi các
  module đó.

**Điều kiện trigger hiển thị (cả 3 điều kiện đều dùng giá trị THẬT):**
```
trigger = registered_person(kết quả thật từ verifier.py hiện có) AND is_waving AND is_facing_camera
```

---

## 2. THÀNH PHẦN 0: LẤY CROP TỪ PIPELINE HIỆN CÓ (mới, thay cho việc giả lập)

**Bắt buộc dùng lại code hiện có, KHÔNG viết detector/verifier mới cho demo này.**

- Khởi tạo/gọi pipeline hiện có (`src/pipeline.py` và các module nó phụ thuộc:
  `detector.py`, `verifier.py`) đúng theo cách chúng đang được dùng trong `main.py --mode run`
  hiện tại — agent cần đọc code hiện có trước để biết chính xác cách khởi tạo (cần path `.npz`
  registry nào, tham số gì) thay vì đoán API.
- Mỗi frame, lấy kết quả track đã được xác nhận là registered person (`track_id`,
  `bbox`, và cờ xác nhận danh tính — tên field chính xác cần đọc từ `types.py`/`pipeline.py`
  hiện có, KHÔNG bịa tên field mới nếu đã có sẵn tên tương đương).
- Nếu KHÔNG có track nào được verify trong frame đó → `registered_person = False`, bỏ qua wave
  detection cho frame này (không có bbox để crop) → `trigger = False` luôn, không cần tính is_waving/
  is_facing_camera.
- Nếu CÓ track được verify → dùng `bbox` đó để crop vùng ảnh tương ứng từ frame gốc (convert tọa
  độ nếu pipeline hiện tại có dùng Dynamic ROI — theo đúng cảnh báo đã có sẵn trong
  `CV_Verification_Handoff_Doc.md` mục 9.3 về bug tọa độ crop-relative vs. frame-absolute, ÁP DỤNG
  TƯƠNG TỰ ở đây dù là demo, để tránh thói quen sai ngay từ đầu).
- Crop này (không phải full frame) là input cho MoveNet ở Thành phần 1 bên dưới.

**Lý do thiết kế lại phần này:** giúp demo test đúng luồng dữ liệu thật sẽ có khi tích hợp cuối
cùng — bbox đến từ verifier thật, không phải giả định — đồng thời tận dụng lại pipeline đã có
thay vì viết trùng lặp logic detect/verify chỉ cho demo.

---

## 3. THÀNH PHẦN 1: POSE ESTIMATION (MoveNet Lightning)

**Chưa có sẵn trong codebase — cần setup mới cho demo này** (đã xác nhận trong hội thoại).

- Model: MoveNet Lightning (singlepose, đủ dùng cho demo vì input đã là crop 1 người từ Thành
  phần 0 — không cần xử lý multi-person ở bước MoveNet).
- Input: crop từ Thành phần 0 ở trên (KHÔNG phải full frame), resize về 192×192×3 giữ tỷ lệ, RGB.
- Output: tensor `[1, 1, 17, 3]` — mỗi keypoint có `[y, x, confidence_score]`, tọa độ normalized
  [0.0, 1.0] theo khung ảnh input.
- **Thứ tự 17 keypoint (theo COCO convention, cố định, KHÔNG được đổi index khi code):**
  ```
  0: nose            6: right_shoulder   12: right_hip
  1: left_eye        7: left_elbow       13: left_knee
  2: right_eye       8: right_elbow      14: right_knee
  3: left_ear        9: left_wrist       15: left_ankle
  4: right_ear       10: right_wrist     16: right_ankle
  5: left_shoulder   11: left_hip
  ```
- Runtime nguồn: TensorFlow Lite (`.tflite`, khớp stack hiện có của project — MobileNetV3/OSNet
  đều chạy qua các runtime nhẹ tương tự) hoặc TF Hub — agent chọn theo dependency đã có sẵn trong
  `requirements`/môi trường hiện tại của project, tránh thêm framework mới không cần thiết.

---

## 4. THÀNH PHẦN 2: WAVE DETECTION (rule-based, theo đề xuất đã thống nhất)

### 4.1. Điều kiện tư thế (per-frame, cần nhưng chưa đủ)
```
wrist_y < shoulder_y   (tọa độ y nhỏ hơn = cao hơn trong ảnh)
```
Áp dụng cho ÍT NHẤT 1 bên tay (trái HOẶC phải) — không yêu cầu cả 2 tay cùng lúc, vì vẫy tay tự
nhiên thường chỉ dùng 1 tay. Dùng `confidence_score` của MoveNet cho từng keypoint (wrist,
shoulder) — chỉ tính điều kiện này là "thỏa mãn" nếu cả 2 keypoint liên quan có
`confidence_score > threshold_keypoint_conf` (placeholder: `0.3`, theo ngưỡng phổ biến trong
literature MoveNet, CẦN calibrate lại).

### 4.2. Điều kiện chuyển động lặp lại (temporal, qua cửa sổ N frame)
- Giữ buffer tọa độ `wrist_x` (của bên tay đang thỏa điều kiện 3.1) qua N frame gần nhất
  (placeholder: `N = 20` frame, tương đương ~0.6-1s ở 20-30fps — CẦN đo lại theo fps thực tế của
  pipeline khi chạy trên phần cứng thật).
- Tính số lần đổi hướng (sign change) của đạo hàm `wrist_x` theo thời gian trong buffer.
- `is_waving = True` nếu số lần đổi hướng ≥ `threshold_direction_changes` (placeholder: `3`) VÀ
  biên độ dao động (`max(wrist_x) - min(wrist_x)` trong buffer) ≥ `threshold_amplitude_norm`
  (placeholder: `0.05`, theo tọa độ normalized 0-1 của bbox crop — CẦN đo lại, không có cơ sở
  thực nghiệm cho con số này).

### 4.3. Chịu lỗi khung xương tạm thời (bắt buộc cho demo, không phải optional)
- Nếu 1 frame có `confidence_score` dưới ngưỡng (occlusion/motion blur tạm thời) → BỎ QUA frame
  đó khỏi buffer (không đẩy giá trị lỗi vào, không reset buffer) — tránh reset toàn bộ chuỗi vì 1
  frame lỗi ngắn.
- Nếu lỗi liên tục quá `max_consecutive_bad_frames` (placeholder: `5`) → reset buffer, coi như
  mất track tạm thời cho gesture detection (không phải mất track tổng thể của Re-ID, chỉ riêng
  module này).

---

## 5. THÀNH PHẦN 3: FACING-CAMERA PROXY (crude, theo quyết định đã chốt cho demo)

**Đã xác nhận: dùng visibility-based proxy, KHÔNG tính góc yaw thật cho demo này.**

```
is_facing_camera = (
    confidence_score[left_eye] > threshold_keypoint_conf AND
    confidence_score[right_eye] > threshold_keypoint_conf AND
    confidence_score[left_shoulder] > threshold_keypoint_conf AND
    confidence_score[right_shoulder] > threshold_keypoint_conf
)
```
Lý do chọn 4 điểm này: cả 2 mắt cùng nhìn thấy rõ + cả 2 vai cùng nhìn thấy rõ là proxy thô cho
"quay mặt/thân về phía camera" — khi người quay nghiêng/quay lưng, thường ít nhất 1 trong 4 điểm
này bị che hoặc confidence thấp. Đây KHÔNG phải phép đo góc chính xác, chỉ là bước trung gian để
hoàn thiện demo — thiết kế facing-gate chính thức (dùng vector hip-shoulder + arctan2, đã có tiền
lệ ở Multi-View Ensemble mục 4.2 của `CV_Verification_Handoff_Doc.md`) là việc CẦN LÀM RIÊNG sau
demo, không nằm trong phạm vi tài liệu này.

Dùng lại `threshold_keypoint_conf` từ mục 3.1 (placeholder `0.3`) cho nhất quán, trừ khi agent
thấy cần tách riêng để dễ tune độc lập — nếu tách, đặt tên rõ ràng
(`threshold_keypoint_conf_wave` vs. `threshold_keypoint_conf_facing`) để không nhầm lẫn khi
calibrate sau này.

---

## 6. OUTPUT / HIỂN THỊ DEMO

Mỗi frame, overlay lên khung hình webcam (hoặc console log nếu không cần overlay):
```
registered_person: True/False   (kết quả thật từ verifier.py hiện có — không có track nào verify
                                   được thì hiển thị False, cắt luôn is_waving/is_facing_camera)
is_waving: True/False
is_facing_camera: True/False
TRIGGER: True/False   (= registered_person AND is_waving AND is_facing_camera)
```
Khuyến khích thêm hiển thị phụ để debug bằng mắt khi demo cho team xem (không bắt buộc, agent tự
quyết định mức độ):
- Vẽ skeleton overlay (dùng lại nếu MoveNet library đã có sẵn hàm vẽ).
- Hiển thị số lần đổi hướng hiện tại trong buffer + biên độ hiện tại, giúp trực quan hóa tại sao
  trigger bật/tắt khi demo trực tiếp cho team.

---

## 7. DANH SÁCH PLACEHOLDER CẦN CALIBRATE SAU DEMO (tổng hợp, không lặp lại giải thích)

| Tên biến | Giá trị demo | Ghi chú |
|---|---|---|
| `threshold_keypoint_conf` | 0.3 | Dùng chung wave + facing trừ khi tách riêng |
| `N` (buffer window) | 20 frame | Phụ thuộc fps thực tế trên phần cứng đích |
| `threshold_direction_changes` | 3 | Chưa có cơ sở thực nghiệm |
| `threshold_amplitude_norm` | 0.05 | Chưa có cơ sở thực nghiệm |
| `max_consecutive_bad_frames` | 5 | Theo tiền lệ `roi_failure_max_frames=5` của Dynamic ROI (mục 6, `CV_Verification_Handoff_Doc.md`) — không phải trùng hợp, dùng lại số đã quen thuộc với project làm điểm khởi đầu |

**Yêu cầu bắt buộc với agent khi code:** đặt TẤT CẢ giá trị trên thành constant/config ở đầu
file hoặc file config riêng — KHÔNG hard-code rải rác trong logic — để việc calibrate sau demo
chỉ cần đổi 1 chỗ, không phải sửa nhiều nơi trong code.

---

## 8. RANH GIỚI TRÁCH NHIỆM (nhắc lại để agent không mở rộng phạm vi ngoài ý)

- Agent code phần trong tài liệu này: đọc output từ pipeline hiện có (KHÔNG sửa logic bên trong
  `pipeline.py`/`detector.py`/`verifier.py`) + crop theo bbox + pose estimation setup (MoveNet,
  mới hoàn toàn) + wave rule + facing proxy + demo display. KHÔNG code identity matching mới
  (đã có sẵn, chỉ dùng lại) — KHÔNG code approach/steering logic.
- Nếu pipeline hiện có trả về NHIỀU track được verify cùng lúc trong 1 frame (ví dụ registry có
  nhiều người và hơn 1 người trong khung hình cùng khớp) → dừng lại, hỏi lại thay vì tự quyết
  định cách chọn — tài liệu này giả định luồng dữ liệu chỉ có 1 track verified tại 1 thời điểm
  (đúng theo thiết kế hiện có của Follow-Me: "mỗi lần robot chạy vẫn chỉ follow 1 người", xem
  `CV_Verification_Handoff_Doc.md` mục 5.1), nhưng agent cần xác nhận thực tế API trả về đúng như
  vậy trước khi code, không giả định suông.
- Nếu agent gặp quyết định thiết kế không có trong tài liệu này (ví dụ: pipeline hiện có chưa
  code xong 1 phần nào đó theo mục 8.1 của `CV_Verification_Handoff_Doc.md`, khiến không có bbox
  verified nào để test) → dừng lại, báo cáo tình trạng, không tự ý viết logic thay thế cho phần
  đó (đó là phạm vi của teammate/phiên làm việc khác).
