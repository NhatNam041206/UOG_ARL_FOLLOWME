# LOG REVIEW & IMPLEMENTATION — Stage 1 + Stage 2 CV Follow-Me Module

> Tài liệu tổng hợp toàn bộ quyết định, instruction, và kết quả review qua quá trình implement
> module Stage 1 (Target Registration) + Stage 2 (Detection + Tracking + Verification + Angle
> Calculation) của robot Follow-Me. Dùng để đối chiếu với codebase thật — mọi mục có đánh dấu
> **[CẦN XÁC NHẬN]** nghĩa là chưa được verify lại trên code thật sau lần review gần nhất.

---

## 0. BỐI CẢNH & QUYẾT ĐỊNH THAY ĐỔI SO VỚI PLAN GỐC

- Plan gốc của project (`Ke_hoach_Software.md`) quy định: bỏ Re-ID model khỏi Sprint 1, chỉ
  dùng track-ID (ByteTrack) + histogram màu áo thô để verification, nhằm giảm rủi ro tiến độ.
- Bản implement này **THAY ĐỔI CÓ CHỦ ĐÍCH** quyết định đó: dùng MobileNetV3 (pretrained
  ImageNet, KHÔNG fine-tune) làm feature extractor cho verification, thay histogram màu áo.
- Đây là thử nghiệm có chủ đích, **CHƯA merge ngược vào planning docs** cho tới khi có kết quả
  test thực địa.
- Phạm vi: CHỈ Stage 1 + Stage 2. KHÔNG bao gồm Stage 3 (Re-assignment logic khi mất track hoàn
  toàn / track_id đổi do occlusion) — thuộc phạm vi khác, chưa implement.
- Môi trường: test local/laptop, KHÔNG deploy Pi 5 ở giai đoạn này.
- Tool build: Antigravity, model Gemini 3.6 / 3.1 High.

---

## 1. CẤU TRÚC PROJECT ĐÃ CHỐT

```
project/
├── main.py                    # entry point, CLI --mode (register|run), OpenCV overlay
├── config/
│   └── settings.yaml          # toàn bộ threshold, path, FOV, ROI, smoothing config
├── src/
│   ├── pipeline.py            # class FollowPipeline — INTERFACE CHÍNH, process_frame(frame) -> AngleResult
│   ├── detector.py            # class YoloDetector — YOLOv11 + ByteTrack (qua model.track())
│   ├── verifier.py            # class MobileNetVerifier — feature extraction + cosine similarity
│   ├── registration.py        # class TargetRegistrar — Stage 1
│   └── types.py                # dataclass AngleResult
└── logs/
    ├── reference_embedding.npy
    └── verification_log.csv
```

Nguyên tắc kiến trúc: KHÔNG dùng MVC (không phù hợp cho pipeline xử lý ảnh 1 chiều, không có
user interaction 2 chiều kiểu MVC nhắm tới). Dùng facade pattern: `FollowPipeline` là class duy
nhất mà `main.py` và module movement (trong codebase lớn hơn sau này) cần gọi — expose đúng 1
method `process_frame(frame) -> AngleResult`. KHÔNG tách thành nhiều package con lồng nhau — mỗi
file trong `src/` là module phẳng.

---

## 2. SPEC CHI TIẾT TỪNG FILE

### 2.1 `src/types.py`
```python
@dataclass
class AngleResult:
    target_found: bool
    angle_offset_deg: float | None = None
    size_ratio: float | None = None
    track_id: int | None = None
    similarity_score: float | None = None
```

### 2.2 `src/detector.py` — YoloDetector
- Dùng `model.track(frame, persist=True, classes=[0])` — KHÔNG tự viết lại tracker, tận dụng
  ByteTrack tích hợp sẵn trong Ultralytics.
- Model file không tồn tại → raise lỗi rõ ràng kèm path đã thử, KHÔNG để traceback thư viện lộ
  ra ngoài.
- `track_id` trả về `None` (thường ở frame đầu) → cần xử lý fallback + log warning, KHÔNG để
  lỗi ngầm lan xuống pipeline.
- Trả về list đầy đủ MỌI track hiện có trong frame (không chỉ 1), mỗi item dạng
  `{track_id, bbox, confidence}`.

### 2.3 `src/verifier.py` — MobileNetVerifier
- MobileNetV3 pretrained ImageNet, KHÔNG fine-tune, KHÔNG có training loop / custom weight
  loading nào.
- Bỏ classification head (`model.classifier = nn.Identity()` hoặc tương đương) — lấy đúng
  embedding layer, KHÔNG giữ lại logits 1000 class ImageNet làm "embedding" (lỗi hay gặp).
- Resize crop về 224×224 + normalize theo ImageNet mean/std trước khi extract — **bắt buộc**,
  thiếu bước này embedding vô nghĩa.
- `extract(crop) -> np.ndarray`: trả về vector đã L2-normalize.
- `compare(emb_a, emb_b) -> float`: cosine similarity qua dot product của 2 vector đã
  normalize — KHÔNG dùng Euclidean distance.

### 2.4 `src/registration.py` — TargetRegistrar (Stage 1)
- ROI hiển thị theo % config trong settings.yaml (KHÔNG hard-code pixel).
- Có state COUNTDOWN (config được, mặc định 3s) trước khi vào state COLLECTING (mặc định 8s).
- Trong lúc thu thập: mỗi N frame (config được, mặc định mỗi 5 frame) detect person trong ROI.
  - Đúng 1 person trong ROI → crop, lưu vào list.
  - 0 hoặc >1 person → skip frame, log warning, **KHÔNG raise exception giữa chừng**.
- Kết thúc thu thập: extract embedding từng crop (đã L2-norm), tính **mean vector**, normalize
  lại mean — đúng thứ tự: normalize từng cái trước → mean → normalize lại lần nữa.
- Số sample hợp lệ < `registration_min_samples` (config được, mặc định 5) →
  **BẮT BUỘC raise lỗi rõ ràng**, KHÔNG lưu file `.npy` không đủ tin cậy, KHÔNG chỉ hiện
  messagebox/log rồi return êm.
- Log rõ: số sample thu được, số frame skip, đường dẫn file đã lưu.

### 2.5 `src/pipeline.py` — FollowPipeline (spec quan trọng nhất)
- `verification_log.csv` ghi similarity_score cho **TẤT CẢ track mỗi frame**, không chỉ track
  được chọn làm target — dữ liệu này bắt buộc phải đầy đủ để calibrate threshold sau.
- **Sticky target logic**: nếu track_id đã là target ở frame trước và (a) track đó vẫn còn tồn
  tại, (b) similarity_score của nó vẫn ≥ threshold → giữ nguyên track đó làm target, KHÔNG tự
  chuyển sang track có similarity cao hơn, trừ khi track cũ biến mất khỏi frame.
- Công thức góc: `offset_x_norm = (bbox_center_x - frame_w/2) / (frame_w/2)` — chia cho **nửa**
  frame_width, tránh lỗi factor-2.
- `angle_offset_deg = degrees(atan(offset_x_norm * tan(FOV/2)))`.
- `size_ratio = bbox_height / frame_height` — proxy khoảng cách thô, comment rõ CHƯA qua IPM.
- Threshold đọc từ config, có comment rõ đây là placeholder cần calibrate bằng dữ liệu thực tế.
- Không track nào ≥ threshold → `AngleResult(target_found=False)`, các field khác `None`.

### 2.6 `main.py`
- Mode `register` và `run` tách biệt qua CLI argument `--mode`.
- Mode `run` nếu thiếu `reference_embedding.npy`: **[QUYẾT ĐỊNH ĐÃ THAY ĐỔI]** ban đầu spec yêu
  cầu raise lỗi rõ ràng yêu cầu chạy `register` trước; sau đó người dùng yêu cầu đổi thành
  auto-redirect sang GUI registration khi thiếu file — **điều kiện bắt buộc đi kèm**: lỗi từ
  registration (nếu fail) PHẢI được propagate lên, KHÔNG bị nuốt/bỏ qua ở tầng auto-redirect.
- Overlay OpenCV: bbox mỗi track (màu phân biệt target/không), similarity_score,
  angle_offset_deg, FPS — hiển thị real-time.
- FPS đo thực tế theo thời gian xử lý mỗi frame, KHÔNG hard-code/giả định.
- Dùng `logging` module, KHÔNG có `print()` sót lại.
- Webcam lỗi / frame decode lỗi → xử lý rõ ràng, KHÔNG crash toàn loop.

### 2.7 `config/settings.yaml`
- Không giá trị nào bị hard-code đè trong code khi đã có key config tương ứng.
- `similarity_threshold`, `camera_fov_horizontal_deg` và các giá trị chưa calibrate khác **phải
  có comment placeholder** ghi rõ "cần calibrate bằng dữ liệu thực tế" / "cần xác nhận FOV thực
  tế của webcam đang dùng".

---

## 3. KẾT QUẢ REVIEW LẦN 1 (trước khi thêm temporal smoothing)

### 3.1 Đạt yêu cầu (không cần sửa)
- `types.py`, `detector.py`, `verifier.py` — PASS toàn bộ theo spec ở mục 2.
- `registration.py` — ROI, countdown, single-person check, thứ tự tính mean embedding — PASS.
- `pipeline.py` — full-track CSV logging, sticky target logic, công thức góc (không lỗi
  factor-2), xử lý no-target-found — PASS.
- `main.py` — mode tách biệt, overlay đầy đủ, FPS đo thực tế, không print() sót, xử lý lỗi
  webcam/frame — PASS.
- Không có cross-import ngoài dự kiến (`detector.py` không import `verifier.py`).
- Không có Stage 3 hay IPM/world-frame nào bị âm thầm implement ngoài scope.

### 3.2 BLOCKER đã phát hiện — **[CẦN XÁC NHẬN ĐÃ SỬA TRÊN CODE THẬT]**

| # | File:dòng (tại thời điểm review) | Vấn đề | Mức độ |
|---|---|---|---|
| 1 | `pipeline.py:120-123` | Thiếu `import cv2` — code có gọi `cv2.resize(...)` khi `frame.shape` khác `input_resolution` trong config, nhưng không import cv2 ở đầu file → sẽ `NameError` nếu bất kỳ caller nào (kể cả module movement sau này) gọi `process_frame()` với frame có resolution khác config. Hiện tại "may" không lộ vì `main.py` đã resize sẵn frame trước khi gọi pipeline — nhưng đây là bug tiềm ẩn nghiêm trọng vì `FollowPipeline` được thiết kế là interface public. | **BLOCKER** |
| 2 | `registration.py:459-469` | Khi số sample hợp lệ < `min_samples`, code chỉ set state = "FAILED" + hiện messagebox, **KHÔNG raise exception**. Hậu quả: nếu user đóng cửa sổ GUI sau khi thấy lỗi (thay vì bấm RESET), `mainloop()` thoát bình thường → `registrar.run()` return bình thường → `main.py` in ra "Stage 1 Registration completed successfully!" dù file `.npy` chưa từng được lưu. Đây là silent failure — caller không biết đã fail. | **BLOCKER** |
| 3 | `config/settings.yaml` + `registration.py:242-260` | Hàm `save_roi_config()` dùng `yaml.safe_load()` rồi `yaml.safe_dump()` để ghi lại config khi user chỉnh ROI qua GUI slider — cơ chế này **xóa sạch mọi comment** trong file YAML gốc, bao gồm comment placeholder bắt buộc ("cần calibrate", "cần xác nhận FOV thực tế"). | Nên sửa |
| 4 | `main.py:102-114` | Auto-redirect sang GUI registration khi thiếu `reference_embedding.npy`, thay vì raise lỗi rõ ràng như spec gốc yêu cầu. **Đây là thay đổi có chủ đích của người dùng** (xem mục 4.3) — nhưng làm tăng mức nghiêm trọng của blocker #2, vì giờ `run` mode phụ thuộc trực tiếp vào registration không được silent-fail. | Cần xác nhận đã ràng buộc đúng propagate lỗi |
| 5 | `requirements.txt` | `registration.py` dùng `from PIL import Image, ImageTk` nhưng `requirements.txt` không có `Pillow`. Cài đặt sạch theo `requirements.txt` sẽ `ImportError` ngay khi chạy mode register. | Nên sửa |

### 3.3 Deviation từ spec — không phải bug, nhưng cần lưu ý
- `registration.py` dùng Tkinter GUI đầy đủ (sliders, gallery, ThreadPoolExecutor) thay vì luồng
  OpenCV đơn giản (`cv2.putText` countdown, vòng lặp cv2 thuần) như plan ban đầu mô tả. Không sai
  về hành vi nhưng vượt xa "simple structure" đã yêu cầu — là mở rộng có chủ đích, không phải lỗi.

---

## 4. CÁC YÊU CẦU BỔ SUNG SAU REVIEW LẦN 1

### 4.1 Auto-redirect khi thiếu file embedding (thay đổi spec `main.py`)
- Quyết định: khi `run` mode thiếu `reference_embedding.npy`, tự động điều hướng user sang GUI
  registration, KHÔNG raise lỗi cứng như spec gốc.
- **Điều kiện bắt buộc đi kèm**: sau khi bug #2 (registration silent-fail) được sửa để raise
  đúng khi fail, auto-redirect logic trong `main.py` PHẢI propagate lỗi đó lên, KHÔNG tự bắt và
  tiếp tục chạy `run` với file cũ/không tồn tại.

### 4.2 Multi-threading để optimize performance
Yêu cầu 2 điểm:
1. Webcam I/O (đọc frame) không bị block bởi xử lý YOLO/MobileNet chậm → giải pháp:
   `WebcamStreamThread` trong `main.py`, 1 thread nền liên tục capture, main loop lấy frame mới
   nhất có sẵn, độ trễ I/O ~0ms.
2. YOLO detect + MobileNet verify chạy song song cho nhiều track trong 1 frame → giải pháp:
   `ThreadPoolExecutor` trong `src/pipeline.py`.

**Vấn đề kỹ thuật đã phát hiện và XÁC NHẬN qua benchmark thật:**
- PyTorch's OpenMP/MKL backend tự dùng nhiều thread nội bộ (8-16) cho mỗi forward pass. Chạy
  `ThreadPoolExecutor(max_workers=4)` mà KHÔNG set `torch.set_num_threads(1)` gây oversubscription
  (4 workers × 8-16 OpenMP threads = 32-64 thread tranh chấp CPU) — làm CHẬM HƠN, không nhanh hơn.
- **Fix đã áp dụng**: `verifier.py` set `torch.set_num_threads(1)` và
  `torch.set_num_interop_threads(1)` khi `device.type == "cpu"`, giới hạn mỗi worker chỉ dùng 1
  thread PyTorch, cho phép song song hóa giữa các crop scale sạch qua các core.

**Benchmark thực đo (trên máy dev, KHÔNG phải Pi 5):**

| N crops | Sequential (ms) | Multi-thread KHÔNG set num_threads=1 (ms) | Multi-thread ĐÃ set num_threads=1 (ms) | Speedup |
|---|---|---|---|---|
| 1 | 14.54 (68 FPS) | 11.32 | **9.65** (103 FPS) | 1.51x |
| 3 | 44.17 (22 FPS) | 20.79 | **22.40** (44 FPS) | 1.97x |
| 5 | 73.92 (13 FPS) | 49.37 | **28.06** (35 FPS) | 2.63x |
| 10 | 152.53 (6.5 FPS) | 81.48 | **53.88** (18 FPS) | 2.83x |

**[CẦN XÁC NHẬN — CHƯA ĐƯỢC ĐO]:**
1. Chưa có baseline "Sequential nhưng CŨNG set `torch.set_num_threads(1)`" để tách bạch 2 nguồn
   lợi ích: (a) tránh oversubscription vs (b) song song hóa thật. Nếu baseline đó gần bằng cột
   "Optimized" hiện tại, phần lớn speedup đến từ việc set num_threads(1), không phải từ
   ThreadPoolExecutor — cần đo để biết chắc.
2. Benchmark trên chưa chạy trên Raspberry Pi 5 (chỉ 4 core, khác hẳn máy dev nhiều core hơn) —
   KHÔNG đại diện cho môi trường triển khai thật. Cần benchmark lại trên Pi 5 trước khi coi
   `max_workers` hiện tại là hợp lý cho production.
3. `max_workers` có set cố định = 4 hay dynamic theo `min(N, cpu_count)`? Chưa xác nhận —
   ảnh hưởng hành vi khi N > max_workers (ví dụ N=10 người cùng lúc).

### 4.3 Auto-redirect được XÁC NHẬN là quyết định có chủ đích của người dùng
Ghi chú tham chiếu tới mục 4.1 — không lặp lại.

---

## 5. TEMPORAL SMOOTHING CHO VERIFICATION (2 mode switch được)

### 5.1 Vấn đề gốc
Pipeline gốc so sánh similarity_score TỨC THỜI (mỗi frame độc lập) với threshold. 1 frame dao
động dưới threshold (motion blur, ánh sáng, góc nghiêng) khiến track bị coi "không phải target"
ngay lập tức dù frame trước/sau đều đúng — thiếu tính nhất quán theo thời gian (inconsistent).

### 5.2 Phạm vi giới hạn đã xác nhận
Giải pháp temporal smoothing CHỈ làm mượt tín hiệu cho CÙNG một track_id qua thời gian. KHÔNG
giải quyết trường hợp track_id bị đổi do occlusion/ID switch (ByteTrack gán ID mới) — track mới
vẫn phải tích lũy lịch sử từ đầu. Đây thuộc phạm vi Stage 3 (chưa làm, ngoài scope).

### 5.3 Spec — Config
```yaml
verification_smoothing:
  mode: "ema"              # "ema" hoặc "voting" — bắt buộc 1 trong 2, raise lỗi nếu sai
  ema_alpha: 0.3            # PLACEHOLDER — cần calibrate. Alpha nhỏ = mượt hơn nhưng trễ hơn.
  voting_window_size: 5     # PLACEHOLDER — cần calibrate. Số frame trong cửa sổ trượt.
  voting_ratio: 0.6         # PLACEHOLDER — cần calibrate. Tỷ lệ frame PASS tối thiểu để confirm.
```

### 5.4 Spec — Mode "ema"
- Mỗi track_id giữ 1 `smoothed_score` riêng:
  `smoothed_score[t] = alpha * raw_score + (1-alpha) * smoothed_score[t-1]`.
- Track_id mới (chưa có lịch sử) → khởi tạo `smoothed_score = raw_score` (giá trị đầu tiên),
  **KHÔNG khởi tạo bằng 0** (nếu khởi tạo 0 sẽ kéo lệch nặng giả tạo ở frame đầu).
- So threshold với `smoothed_score`, KHÔNG so với `raw_score` trực tiếp.
- Track biến mất khỏi frame → xóa entry khỏi dict, tránh memory leak.

### 5.5 Spec — Mode "voting"
- Mỗi track_id giữ `deque(maxlen=voting_window_size)` chứa bool (`raw_score >= threshold`) của
  N frame gần nhất — chỉ lưu pass/fail, không lưu score.
- `vote_ratio = (số True trong deque) / len(deque)` — dùng `len(deque)` THỰC TẾ (không chia cố
  định cho `voting_window_size`), để track mới không bị kẹt "chưa đủ evidence" quá lâu.
- Track được xác nhận target nếu `vote_ratio >= voting_ratio`.
- Track biến mất → xóa deque khỏi state.

### 5.6 Spec — Logging
- `verification_log.csv` phải có cả cột `raw_score` VÀ cột giá trị đã smooth: `smoothed_val`
  (chứa `smoothed_score` khi mode="ema", hoặc `vote_ratio` khi mode="voting").

### 5.7 KẾT QUẢ REVIEW LẦN 2 (sau khi implement temporal smoothing) — ĐÃ XÁC NHẬN QUA CODE

| Điểm kiểm tra | Kết quả | Vị trí code (tại thời điểm review) |
|---|---|---|
| `smoothed_val` chứa đúng `vote_ratio` khi mode="voting" | ✅ PASS | `pipeline.py:180-188` |
| EMA khởi tạo bằng `raw_score` đầu tiên, không phải 0 | ✅ PASS | `pipeline.py:170-173` |
| Voting dùng đúng `self.threshold` (= `similarity_threshold` config) để tính pass/fail mỗi frame | ✅ PASS | `pipeline.py:176-179` |
| Unit test cho 2 mode + invalid mode + CSV header | ✅ PASS (test chạy, nhưng xem 5.8 về giới hạn của test) | `test_pipeline.py` |

### 5.8 VẤN ĐỀ MỚI PHÁT HIỆN TỪ CHÍNH UNIT TEST — Cold-start instability của mode "voting"

Test case minh họa (`voting_window_size=3, voting_ratio=0.6`):
- Frame 1: raw_score=0.80 (Pass) → deque=[True], vote_ratio=1/1=1.0 ≥ 0.6 → Target Locked.
- Frame 2: raw_score=0.40 (Fail) → deque=[True, False], vote_ratio=1/2=0.50 < 0.6 →
  **Target Lost**.
- Frame 3: raw_score=0.80 (Pass) → deque=[True, False, True], vote_ratio=2/3=0.67 ≥ 0.6 →
  Target Regained.

**Vấn đề**: Ở Frame 2, buffer mới có 2/3 phần tử (chưa đầy), nên 1 frame fail đã đủ kéo
vote_ratio xuống dưới threshold — vì mẫu số nhỏ khiến 1 lần fail có tác động tỷ lệ lớn hơn bất
thường. Đây KHÔNG phải bug code (code đúng theo spec 5.5) — nhưng là giới hạn thiết kế: **trong
lúc buffer chưa đầy, mode "voting" nhạy với nhiễu tức thời HƠN mode "ema"**, ngược với mục tiêu
ban đầu (giảm nhiễu). Vote_ratio chỉ ổn định đúng như kỳ vọng khi buffer đã đầy đủ
`voting_window_size` frame.

**[CẦN XÁC NHẬN — CHƯA ĐƯỢC TEST]**: chưa có test case với buffer ĐÃ ĐẦY (đúng
`voting_window_size` thực tế dự kiến dùng, ví dụ 5) để xác nhận hành vi ổn định khi có đủ
evidence (ví dụ 1 frame fail giữa 5 frame pass → 4/5=0.8 ≥ 0.6 → không mất target).

---

## 6. FIX COLD-START INSTABILITY CHO MODE "VOTING" — SPEC ĐÃ CHỐT (CHƯA XÁC NHẬN ĐÃ IMPLEMENT)

> **[CẦN XÁC NHẬN TOÀN BỘ MỤC NÀY]** — đây là spec vừa được chốt trong phần cuối cuộc trò
> chuyện, CHƯA có báo cáo implement/test nào được đưa để review. Agent kiểm tra codebase cần xác
> nhận các điểm dưới đây đã được code đúng hay chưa tồn tại trong codebase.

### 6.1 Quyết định thiết kế đã chốt
- **Chỉ áp dụng cho `mode: "voting"`.** Mode "ema" không cần sửa (đã tự nhiên đúng nhờ khởi tạo
  bằng raw_score đầu tiên, xem mục 5.4).
- Ngưỡng "buffer đủ đầy để tin" tính theo **% của `voting_window_size`** (không dùng số tuyệt
  đối riêng biệt) — để chỉ cần tinh chỉnh 1 tham số (`voting_window_size`) khi calibrate, không
  phải đồng bộ 2 số độc lập.
- 2 cơ chế RIÊNG BIỆT, KHÔNG chồng lắp:

### 6.2 Cơ chế 1 — Per-track fallback (mọi lúc, cho track mới xuất hiện giữa lúc đang chạy)
- Config mới: `voting_min_ready_percent: 0.6` (PLACEHOLDER, cần calibrate) trong khối
  `verification_smoothing`. Chỉ có nghĩa khi `mode == "voting"`.
- `min_ready_frames = ceil(voting_window_size * voting_min_ready_percent)`.
- Nếu `len(voting_buffers[tid]) < min_ready_frames`:
  - KHÔNG dùng vote_ratio.
  - Fallback: dùng trực tiếp `raw_score >= self.threshold` để quyết định target_found cho
    track này ở frame hiện tại (giống hệt hành vi hệ thống KHÔNG có smoothing).
  - VẪN append kết quả pass/fail vào buffer bình thường (để buffer tiếp tục đầy dần).
- Nếu `len(voting_buffers[tid]) >= min_ready_frames`: dùng vote_ratio như cơ chế gốc (mục 5.5),
  không đổi.
- Log thêm cột `voting_ready: bool` trong `verification_log.csv` để phân biệt quyết định đến từ
  fallback hay từ vote_ratio thật.

### 6.3 Cơ chế 2 — Startup warm-up (chỉ 1 LẦN, lúc `FollowPipeline` khởi tạo, chỉ khi mode="voting")
- Biến trạng thái cấp pipeline (không phải cấp track): `_startup_frame_count = 0`,
  `_startup_warmup_done = False`, khởi tạo trong `__init__`.
- `startup_warmup_frames = ceil(voting_window_size * voting_min_ready_percent)` — dùng CHUNG
  công thức và tham số với Cơ chế 1 (không thêm config mới).
- Mỗi lần `process_frame()` được gọi trong giai đoạn warm-up:
  - Vẫn chạy đầy đủ detect + track + verify (để buffer build bình thường).
  - Tăng `_startup_frame_count`.
  - Nếu `_startup_warmup_done == False` và `_startup_frame_count < startup_warmup_frames`:
    - Ghi log CSV đầy đủ như bình thường.
    - Nhưng return `AngleResult(target_found=False)` ngay — KHÔNG tính angle_offset_deg/
      size_ratio, bất kể track nào có similarity cao thế nào.
  - Khi `_startup_frame_count >= startup_warmup_frames`: set `_startup_warmup_done = True`, từ
    đây xử lý bình thường qua Cơ chế 1.
- Nếu `mode == "ema"`: bỏ qua toàn bộ Cơ chế 2 — xử lý ngay từ frame đầu như hiện tại.
- Warm-up CHỈ chạy 1 lần trong đời `FollowPipeline` instance — KHÔNG reset nếu track bị mất/tìm
  lại giữa lúc chạy (đó là việc của Cơ chế 1).
- Output trong lúc warm-up: dùng chung `AngleResult(target_found=False)`, **KHÔNG thêm field
  mới** (đã xác nhận không cần field `warming_up: bool` riêng).

### 6.4 Phân biệt rõ 2 cơ chế (tránh nhầm khi review code)
- Cơ chế 2 (startup warm-up): chặn OUTPUT của TOÀN pipeline (mọi track) trong N frame đầu lúc
  khởi động chương trình — không quan tâm track nào.
- Cơ chế 1 (per-track fallback): chỉ chặn VOTE_RATIO cho TỪNG track riêng lẻ có buffer chưa đủ
  đầy, xảy ra ở BẤT KỲ thời điểm nào trong vòng đời chương trình (không chỉ lúc khởi động).
- Sau khi warm-up (Cơ chế 2) kết thúc, nó không còn tác dụng nữa — nhưng Cơ chế 1 vẫn luôn hoạt
  động cho track mới bất cứ lúc nào.

### 6.5 Test bắt buộc cần có (chưa xác nhận đã viết)
1. Test Cơ chế 1: track mới xuất hiện GIỮA lúc pipeline đã chạy lâu (không phải lúc khởi động)
   — với `voting_window_size=5, voting_min_ready_percent=0.6` (→ min_ready_frames=3) — verify 2
   frame đầu của track dùng raw threshold, frame thứ 3 trở đi mới dùng vote_ratio.
2. Test Cơ chế 2: verify trong N frame đầu của pipeline, dù raw_score rất cao (ví dụ 0.95) ở MỌI
   frame, `process_frame()` vẫn luôn trả `target_found=False`. Từ frame N+1, hành vi bình
   thường.
3. Test mode "ema" KHÔNG bị ảnh hưởng bởi thay đổi này — verify pipeline mode="ema" xử lý bình
   thường ngay từ frame đầu, không có warm-up.

---

## 7. TỔNG HỢP TÌNH TRẠNG — VIỆC CẦN AGENT KIỂM TRA NGAY

> **CẬP NHẬT 2026-08-12**: đã đối chiếu toàn bộ mục 7 với codebase thật (đọc lại từng file,
> không dùng kết quả review cũ) và sửa trực tiếp các mục còn mở, trừ mục 7.2 (benchmark Pi 5 —
> chưa có hardware). Chi tiết xem mục 8.3.

### 7.1 Blocker từ mục 3.2 — ĐÃ ĐỐI CHIẾU & SỬA XONG
- [x] #1 — `pipeline.py` thiếu `import cv2` → **đã có sẵn** `import cv2` ở dòng 3 khi kiểm tra
      lại (đã được sửa trước đó, không rõ thời điểm).
- [x] #2 — `registration.py` không raise khi sample < min_samples → **đã có sẵn** khi kiểm tra
      lại: wrapper `TargetRegistrar.run()` kiểm tra `gui.success` sau khi `gui.run()` return và
      `raise RuntimeError(...)` nếu fail, kể cả trường hợp user đóng cửa sổ giữa chừng (dùng
      fallback error message).
- [x] #3 — `save_roi_config()` xóa comment trong `settings.yaml` → **đã sửa trong phiên này**:
      đổi từ `yaml.safe_load`+`yaml.safe_dump` (round-trip qua dict, xóa mọi comment) sang patch
      trực tiếp ở mức text — chỉ tìm và thay đúng khối `roi_percent:` (hỗ trợ cả block-style và
      inline-style), giữ nguyên mọi dòng khác kể cả comment. Đã verify bằng script mô phỏng: sau
      patch, số lượng `#` trong file không đổi, các key khác (`similarity_threshold`,
      `verification_smoothing`...) không bị ảnh hưởng.
- [x] #4 — Auto-redirect trong `main.py` propagate lỗi đúng cách → **đã có sẵn** khi kiểm tra
      lại: `except Exception as e: logger.error(...); sys.exit(1)`, không nuốt lỗi. Kết hợp với
      fix #2 đã hoạt động đúng end-to-end.
- [ ] #5 — `requirements.txt` có `Pillow` chưa → **CHƯA ĐƯỢC XÁC NHẬN LÀ ĐÃ SỬA khi review**,
      nhưng **đã bổ sung `Pillow>=10.0.0`** vào `requirements.txt` trong phiên sửa này.

### 7.2 Cần đo/xác nhận thêm (từ mục 4.2) — VẪN CÒN MỞ, CHƯA CÓ HARDWARE
- [ ] Baseline "Sequential + `torch.set_num_threads(1)`" đã được đo để tách bạch nguồn lợi ích
      chưa. **[CẦN XÁC NHẬN — vẫn chưa đo, không có script/số liệu nào trong repo]**
- [ ] Benchmark trên Raspberry Pi 5 thật đã có chưa. **[CẦN XÁC NHẬN — vẫn chưa đo, chưa có
      hardware theo xác nhận của người dùng ngày 2026-08-12 — để ngỏ cho tới khi có Pi 5]**
- [x] `max_workers` trong `ThreadPoolExecutor` — đã xác nhận: [pipeline.py:98](src/pipeline.py:98)
      `min(4, os.cpu_count() or 4)` — **cố định trần 4, dynamic theo `cpu_count` KHÔNG dynamic
      theo N (số track/người trong frame)**. Nếu >4 người cùng lúc, crop dư sẽ xếp hàng chờ.

### 7.3 Cần xác nhận đã implement đúng (từ mục 6) — ĐÃ ĐỐI CHIẾU, XONG (khác dự đoán ban đầu)
- [x] Config `voting_min_ready_percent` đã được thêm vào `settings.yaml` — có, giá trị `0.6`.
- [x] Cơ chế 1 (per-track fallback) đã implement đúng theo mục 6.2 — verify qua unit test mới
      `test_voting_per_track_fallback_before_buffer_ready`.
- [x] Cơ chế 2 (startup warm-up) đã implement — **NHƯNG có bug off-by-one bị phát hiện khi viết
      test, đã sửa trong phiên này**, xem mục 8.1.
- [x] Cột `voting_ready` đã có trong `verification_log.csv` — verify cả ở code lẫn dữ liệu log
      thật đang có trong `logs/verification_log.csv`.
- [x] 3 test case ở mục 6.5 — **file `test_pipeline.py` mà mục 5.7 nhắc tới đã KHÔNG còn tồn tại
      trong codebase** (có dấu vết còn sót trong `verification_log.csv` từ lần chạy cũ, nhưng
      file test đã mất). Đã viết lại từ đầu trong phiên này, 7 test case, tất cả PASS.

---

## 8. PHIÊN SỬA 2026-08-12 — CHI TIẾT

### 8.1 Bug MỚI phát hiện khi viết lại test — off-by-one ở Cơ chế 2 (startup warm-up)

Khi viết test cho spec 6.5 mục 2 ("verify trong N frame đầu... `process_frame()` vẫn luôn trả
`target_found=False`. Từ frame N+1, hành vi bình thường"), test đầu tiên FAIL. Nguyên nhân: code
gốc dùng `if self._startup_frame_count < self.startup_warmup_frames: return False` — vì
`_startup_frame_count` tăng TRƯỚC (ở đầu `process_frame()`), điều kiện `<` khiến chỉ
`startup_warmup_frames - 1` frame bị chặn (ví dụ warmup=3 thì chỉ frame 1,2 bị chặn, frame 3 đã
xử lý bình thường), không đúng "N frame đầu" như spec 6.3/6.4 mô tả bằng lời.

**Đã sửa**: đổi `<` thành `<=` ở [pipeline.py:248](src/pipeline.py:248) (nhánh có detection), và
đổi `>=` thành `>` ở [pipeline.py:173](src/pipeline.py:173) (nhánh không có detection, để 2 nhánh
đồng nhất thời điểm `_startup_warmup_done` chuyển `True`). Sau fix: đúng `startup_warmup_frames`
frame đầu tiên bị chặn `target_found=False`, frame kế tiếp (N+1) mới xử lý bình thường — verify
bằng `test_voting_startup_warmup_forces_no_target`.

### 8.2 Danh sách file đã thay đổi
- [requirements.txt](requirements.txt): thêm `Pillow>=10.0.0`.
- [config/settings.yaml](config/settings.yaml): dọn nội dung bị lặp đôi (nguyên nhân chưa xác
  định rõ — không phải do `save_roi_config()` vì hàm đó parse qua dict Python nên không thể tạo
  key trùng; nhiều khả năng do sửa tay hoặc lỗi tool patch), khôi phục lại comment placeholder
  bắt buộc.
- [src/registration.py](src/registration.py): viết lại `save_roi_config()` theo cơ chế patch
  text tại chỗ thay vì `yaml.safe_load`/`yaml.safe_dump` — không còn xóa comment nữa; xóa import
  `yaml` không còn dùng.
- [src/pipeline.py](src/pipeline.py): sửa bug off-by-one ở mục 8.1.
- [test_pipeline.py](test_pipeline.py): file mới, 7 test case (invalid mode, CSV header, EMA init
  bằng raw_score, EMA làm mượt dao động tạm thời nhưng vẫn phản ứng khi drop kéo dài, EMA không
  có warm-up delay, Cơ chế 2 startup warm-up, Cơ chế 1 per-track fallback). Chạy bằng
  `.venv/Scripts/python.exe -m unittest test_pipeline -v` — tất cả PASS. Dùng `unittest.mock`
  để giả lập `YoloDetector`/`MobileNetVerifier`, không cần camera/model thật.

### 8.3 Còn treo (có chủ đích, chưa làm)
- Benchmark hiệu năng trên Raspberry Pi 5 thật + đo baseline "sequential có
  `set_num_threads(1)`" — người dùng xác nhận chưa có hardware, để ngỏ tới khi có Pi 5.

---

## 9. THÊM: Dynamic ROI-constrained detection + Aspect ratio hard gate (2026-08-12)

### 9.1 Mục tiêu
2 tính năng độc lập, implement song song:
1. **ROI-constrained detection**: khi target đã sticky-tracked, chỉ chạy YOLO trên vùng crop
   quanh vị trí bbox frame trước (mở rộng theo margin) thay vì cả frame — giảm compute.
2. **Aspect ratio hard gate**: chặn cứng track có similarity cao nhưng hình dáng
   (`bbox_width/height`) lệch quá xa so với `reference_aspect_ratio` đã đo lúc registration —
   giảm nhầm người mặc đồ giống nhau nhưng vóc dáng khác (tín hiệu độc lập với appearance
   embedding).

### 9.2 Quyết định thiết kế quan trọng — KHÔNG có trong spec chữ, tự suy luận thêm

**Vấn đề phát hiện khi thiết kế (trước khi code)**: code gốc, khi `detector.track()` trả về
rỗng (0 track), luôn reset `active_target_id = None` ngay lập tức — coi là mất target hoàn
toàn. Với ROI-constrained detection, một ROI hẹp (margin nhỏ) hoàn toàn có thể "hụt" mất
target ở 1 frame đơn lẻ (target bước ra rìa ROI dù vẫn còn trong full frame) — nếu giữ
nguyên hành vi reset ngay, cơ chế fallback `roi_failure_max_frames` sẽ **không bao giờ** có
cơ hội chạy, vì 1 frame ROI hụt là mất target ngay, không tích lũy được failure count.

**Quyết định**: tách 2 trường hợp "0 detection":
- ROI-constrained scan hụt (`used_roi=True`, detections rỗng) → tăng `_roi_failure_count`,
  **giữ nguyên** `active_target_id`/`last_detections` để frame sau vẫn thử ROI quanh vị trí
  cũ. Không coi là mất target.
- Full-frame scan hụt (`used_roi=False`) → mất target thật, reset như hành vi cũ.

Đây KHÔNG được ghi rõ trong spec gốc (spec chỉ nói "KHÔNG tìm thấy track nào có similarity
vượt threshold" cho việc đếm fail, chưa xét case 0 detection hoàn toàn) — cần xác nhận lại
đây có đúng ý định hay không.

**Vấn đề thứ 2 — thứ tự aspect ratio gate với temporal smoothing**: spec có 2 câu mô tả hơi
mâu thuẫn nhau:
- "kết hợp AND với logic threshold hiện có (EMA/voting)" → gợi ý AND ở SAU khi đã smooth.
- "chuỗi xử lý: tính raw similarity → tính aspect_ratio_pass → combine AND → mới đưa vào
  EMA/voting của is_pass đã combine đó" → gợi ý AND ở TRƯỚC khi đưa vào smoothing.

**Quyết định**: chọn AND ở SAU khi smooth (aspect ratio gate là veto tức thời mỗi frame,
không ảnh hưởng tới giá trị `smoothed_val`/`vote_ratio` — giữ tín hiệu similarity "sạch",
khớp với yêu cầu logging riêng 3 cột aspect ratio để calibrate độc lập với
`similarity_threshold`). Đã ghi rõ trong code comment ở `pipeline.py` và README — cần người
review xác nhận lại đúng ý định.

### 9.3 Implementation
- [src/pipeline.py](src/pipeline.py): thêm `_compute_roi()`, logic quyết định
  ROI-constrained/full-frame ở đầu `process_frame()`, convert tọa độ ROI-local → full-frame
  ngay sau khi detect, aspect ratio gate tính trong vòng lặp per-track (trước dòng
  `is_pass = similarity_is_pass and aspect_ratio_pass`), bookkeeping `_roi_failure_count` sau
  bước chọn target (1 chỗ duy nhất, xử lý cả 2 nhánh sticky/tìm-mới cùng lúc, tránh trùng lặp
  logic).
- [src/registration.py](src/registration.py): tính `collected_aspect_ratios` song song
  `collected_crops`, lưu median vào `logs/reference_aspect_ratio.npy` (companion file, cùng
  thư mục với `reference_embedding.npy`) trong `_finish_collection`.
- [config/settings.yaml](config/settings.yaml): thêm `roi_margin_percent: 0.5`,
  `roi_failure_max_frames: 5`, `aspect_ratio_tolerance_percent: 0.30` — tất cả PLACEHOLDER.
- CSV (`verification_log.csv`) thêm cột: `candidate_aspect_ratio`, `ar_diff_ratio`,
  `aspect_ratio_pass`, `used_roi`, `roi_bounds`.
- `logs/reference_embedding.npy` hiện tại (từ lần re-register cho OSNet ở mục 11) đã đổi tên
  thành `logs/reference_embedding_OLD_no_aspect_ratio.npy.bak` — thiếu file companion aspect
  ratio, **bắt buộc chạy lại `--mode register` lần nữa**.

### 9.4 Validation đã làm (theo đúng yêu cầu, KHÔNG tự chạy webcam thật)
Thêm 4 test case mới vào `test_pipeline.py` (mock `YoloDetector`/`OSNetVerifier`), tổng
11/11 test PASS:
1. `test_roi_constrained_detection_and_coordinate_conversion` — verify frame đầu (chưa có
   target) detect toàn frame; frame sau (đã có target) `detector.track()` nhận đúng CROP
   (kiểm tra qua `call_args_list[...].shape`, không phải full frame); bbox trả về (tọa độ
   cục bộ trong crop) được convert đúng về full-frame coords khớp phép tính tay.
2. `test_roi_fallback_to_full_frame_after_max_consecutive_failures` — 2 frame ROI hụt liên
   tiếp (`roi_failure_max_frames=2` để test nhanh) → xác nhận `active_target_id` KHÔNG bị xóa
   giữa chừng, `_roi_failure_count` tăng đúng; frame thứ 3 (đạt max) → xác nhận
   `detector.track()` nhận full frame (không phải crop), `_roi_failure_count` reset về 0.
3. `test_aspect_ratio_gate_blocks_high_similarity_wrong_shape` — track similarity=0.95 (dễ
   pass threshold) nhưng bbox aspect ratio lệch xa reference → `target_found=False`,
   `is_pass=False`, `aspect_ratio_pass=False` — xác nhận gate chặn đúng dù similarity cao.
4. `test_aspect_ratio_gate_allows_matching_shape` — bbox đúng aspect ratio tham chiếu →
   không bị chặn, `target_found=True`.

Smoke test tích hợp thật (không mock, YOLO + OSNet thật): dựng `FollowPipeline` với embedding
+ aspect_ratio giả (512-dim random + giá trị 0.5), gọi `process_frame()` 3 lần liên tiếp trên
frame nhiễu ngẫu nhiên — chạy hết không lỗi, log đúng config values đã đọc
(`roi_margin_percent=0.5`, `roi_failure_max_frames=5`, `aspect_ratio_tolerance_percent=0.3`).

### 9.5 Việc người dùng cần làm trước khi test lại bằng webcam thật
1. Chạy lại `python main.py --mode register` (embedding hiện tại thiếu file companion aspect
   ratio bắt buộc).
2. Test kịch bản gốc của task: người mặc đồ giống nhau, vóc dáng khác nhau — đo lại bằng cột
   mới trong `verification_log.csv` (`candidate_aspect_ratio`, `ar_diff_ratio`,
   `aspect_ratio_pass`) để xác nhận gate hoạt động đúng thực tế, và bằng cột `used_roi`/
   `roi_bounds` để xác nhận ROI-constrained detection hoạt động đúng, không có track_id bị
   "nhảy" bất thường do rủi ro kiến trúc đã ghi ở mục 9.2/README.
3. Xác nhận lại 2 quyết định diễn giải spec ở mục 9.2 (case 0-detection trong ROI, thứ tự
   aspect ratio gate với temporal smoothing) có đúng ý định ban đầu hay cần sửa lại.

---

## 10. NGOÀI PHẠM VI (KHÔNG implement, không nên xuất hiện trong codebase này)
- Stage 3 (re-assignment / world-frame extrapolation khi mất track hoàn toàn hoặc track_id đổi
  do occlusion).
- IPM / world-frame conversion đầy đủ (hiện tại chỉ có size_ratio thô qua bbox, chưa qua IPM).
- Quantize/optimize model cho Pi 5.
- Bất kỳ giá trị threshold/config nào được "chốt cứng" thành số cuối cùng mà không có comment
  placeholder — mọi threshold hiện tại đều là giá trị tạm, cần calibrate bằng dữ liệu thực địa.

---

## 11. THAY VERIFIER: MobileNetV3 → OSNet (2026-08-12)

### 11.1 Lý do

Đo thực nghiệm bằng đúng quy trình Scenario A (target thật trong khung) / Scenario B (chỉ
người lạ) đã dùng ở bước diagnose false-positive trước đó, xác nhận: mean similarity target
thật (0.808) và người lạ (0.787) chỉ cách nhau ~0.02, trong khi độ lệch chuẩn mỗi nhóm
(~0.04-0.05) lớn hơn khoảng cách đó — không tồn tại threshold nào tách 2 nhóm đáng tin cậy.
Đã xác nhận ở phần trace code (mục Bước 1 review trước) rằng đây KHÔNG phải bug logic chọn
target — nguyên nhân gốc là MobileNetV3 pretrained ImageNet-classification không sinh ra
embedding đủ phân biệt danh tính người (nó phân biệt "đây là loại vật thể gì", không phải
"đây là NGƯỜI NÀO"). Quyết định: thay bằng OSNet — model train sẵn cho đúng bài toán person
re-identification, qua thư viện `torchreid`.

### 11.2 Phát hiện quan trọng khi implement — `pretrained=True` mặc định KHÔNG phải Market1501

Giả định ban đầu (trong yêu cầu implement) là `torchreid.utils.FeatureExtractor(model_name=...,
pretrained=True)` mặc định sẽ tự cho weight đã train trên Market1501. **Sai** — đã verify bằng
code thật: `build_model(..., pretrained=True)` không kèm `model_path` chỉ tải weight
**ImageNet-classification** (file `{variant}_imagenet.pth`) — chính xác loại vấn đề mà việc
đổi sang OSNet vốn định tránh. Phải tự tải đúng checkpoint đã fine-tune re-id từ Model Zoo
chính thức của torchreid và truyền vào `model_path=`:
- `osnet_x1_0`: train+test trên Market1501 (Rank-1 94.2, mAP 82.6), Google Drive id
  `1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA`.
- `osnet_ain_x1_0`: không có checkpoint Market1501-only trong Model Zoo; dùng checkpoint train
  trên MSMT17+DukeMTMC+CUHK03, test zero-shot trên Market1501 (Rank-1 73.3, mAP 45.8), Google
  Drive id `1nIrszJVYSHf3Ej8-j6DTFdWz8EnO42PB` — chọn vì osnet_ain được thiết kế robust với
  domain gap, và webcam thực tế cũng là 1 domain chưa từng train qua.

Cả 2 checkpoint đã tải và verify bằng code thật (không giả định) — xem `test_verifier.py`.

### 11.3 Cài `torchreid` — 2 lần fail trước khi tìm ra cách chạy được

1. `pip install torchreid` (PyPI) — cài thành công nhưng `import torchreid` fail ngay:
   `ModuleNotFoundError: No module named 'scipy'`. Package PyPI này là fork bên thứ 3
   (`goksenin-uav/torchreid-pip`), `setup.py` không khai báo `install_requires` nào cả.
2. `pip install -e .` / `python setup.py develop` từ source chính chủ
   (`KaiyangZhou/deep-person-reid`) — fail ở bước "Getting requirements to build editable":
   `setup.py` cần `import numpy` (cho `cythonize()` build 1 Cython extension tùy chọn dùng để
   tăng tốc eval metrics) NGAY TẠI BUILD TIME, nhưng build isolation của pip tạo môi trường
   tạm không có numpy/Cython (không có `pyproject.toml` khai báo `[build-system] requires`).
   `--no-build-isolation` ở lệnh pip ngoài KHÔNG có tác dụng vì `setup.py develop` tự shell ra
   1 lệnh `pip install -e . --use-pep517` con không kế thừa cờ đó.
3. **Cách đã dùng thành công**: copy thẳng thư mục `torchreid/` (thuần Python) từ repo đã
   clone vào `.venv/Lib/site-packages/torchreid/`, bỏ qua hẳn bước build Cython extension.
   `torchreid/metrics/rank.py` tự fallback sang eval bằng Python thuần (chỉ warning, không
   lỗi) khi thiếu extension này — đã verify runtime thật, không phải giả định. Chi tiết đầy đủ
   ở [README.md](README.md) mục "Cài `torchreid`".

### 11.4 Thay đổi code
- [src/verifier.py](src/verifier.py): xóa hẳn `MobileNetVerifier`, thay bằng `OSNetVerifier`
  — giữ nguyên interface `extract()`/`compare()`, giữ `torch.set_num_threads(1)` khi CPU.
- [src/pipeline.py](src/pipeline.py), [src/registration.py](src/registration.py): đổi import +
  đọc config `osnet_variant` thay `mobilenet_variant`.
- [config/settings.yaml](config/settings.yaml): `mobilenet_variant: small` →
  `osnet_variant: osnet_x1_0`. `similarity_threshold` GIỮ NGUYÊN `0.80` theo đúng yêu cầu —
  chưa calibrate lại cho OSNet.
- [requirements.txt](requirements.txt): thêm runtime dependency của `torchreid` (scipy, h5py,
  six, matplotlib, tb-nightly, future, yacs, gdown, imageio, chardet) kèm comment giải thích
  không thể `pip install torchreid` trực tiếp.
- `logs/reference_embedding.npy` (cũ, 576-chiều từ MobileNetV3) đã đổi tên thành
  `logs/reference_embedding_OLD_mobilenetv3_INCOMPATIBLE.npy.bak` — **bắt buộc chạy lại
  `--mode register`** trước khi dùng `--mode run`, vì OSNet cho embedding 512-chiều, không
  tương thích kích thước với file cũ.

### 11.5 Validation đã làm (theo đúng yêu cầu, KHÔNG tự chạy webcam thật)
- `test_verifier.py` (mới): dùng ảnh ngẫu nhiên giả, verify `extract()` trả vector đúng shape
  `(512,)`, đã L2-normalize (norm ≈ 1.0), xử lý đúng crop rỗng (trả zero-vector), và
  `compare()` trả giá trị hợp lệ trong [-1, 1], tự-so-sánh ≈ 1.0. Chạy bằng model OSNet THẬT
  (không mock) — 5/5 test PASS.
- `test_pipeline.py` (cũ, mock verifier): cập nhật lại patch target/config key, chạy lại 7/7
  test PASS — xác nhận đổi verifier không ảnh hưởng logic pipeline (đúng như yêu cầu "không
  sửa gì ở pipeline.py logic chọn target").
- Smoke test tích hợp thật (không mock gì): dựng `FollowPipeline` với `YoloDetector` +
  `OSNetVerifier` thật (dùng đúng `yolo11n.onnx` + checkpoint OSNet đã tải), gọi
  `process_frame()` trên 1 frame giả — chạy xong không lỗi, trả `AngleResult` hợp lệ.
- Input size thực tế OSNet yêu cầu: **256×128 (H×W)** — khác 224×224 vuông của MobileNetV3,
  đã xác nhận qua `inspect.signature(FeatureExtractor.__init__)` trên code thật, không giả định.

### 11.6 Việc người dùng cần làm trước khi test lại bằng Scenario A/B
1. Chạy `python main.py --mode register` lại (embedding cũ không tương thích, xem 11.4).
2. Cài `torchreid` theo đúng 3 lệnh ở mục 11.3 nếu máy chưa có (`import torchreid` sẽ báo lỗi
   rõ nếu thiếu).
3. Chạy lại đúng quy trình Scenario A/B đã dùng lần trước (30s có target thật / 30s chỉ người
   lạ), so sánh phân phối similarity mới với số liệu MobileNetV3 cũ (target mean=0.808 / người
   lạ mean=0.787) để biết OSNet có thực sự tách 2 nhóm tốt hơn không — **chưa có số liệu thật
   nào ở bước này**, không được giả định OSNet chắc chắn tốt hơn tới khi có log thật.
4. `similarity_threshold` trong config vẫn là `0.80` (giá trị cũ) — cần tính lại từ log
   Scenario A/B mới, không dùng nguyên giá trị cũ cho model mới.

---

## 12. THÊM: Debug UI tách file riêng, mặc định chạy headless (2026-08-12)

### 12.1 Yêu cầu
Người dùng muốn thấy được vùng ROI-constrained detection đang di chuyển và các feature khác
của pipeline (aspect ratio gate, ROI failure count, trạng thái warm-up...) trực quan, nhưng
chỉ để debug — khi triển khai thật sau này, toàn bộ quá trình phải chạy background, không có
UI. Yêu cầu code vẽ overlay nằm ở 1 file riêng (tách khỏi `main.py`) để nhánh chạy nền không
phải mang theo code UI.

### 12.2 Implementation
- [src/debug_overlay.py](src/debug_overlay.py): file mới, hàm `render(frame, pipeline,
  angle_result, current_fps)` — toàn bộ logic vẽ overlay cũ (bbox target/non-target, mũi tên
  hướng target, status text, FPS) chuyển từ `main.py` sang đây, cộng thêm:
  - Vẽ hình chữ nhật cyan cho vùng ROI đang dùng (`pipeline.last_roi_bounds`, chỉ vẽ khi
    `pipeline.last_used_roi`).
  - Tag cam "AR-MISMATCH" cho track bị aspect ratio gate chặn (dù similarity có thể cao).
  - Panel telemetry: FPS, smoothing mode, ROI hay full-frame, `_roi_failure_count`/
    `roi_failure_max_frames`, banner cảnh báo khi đang trong giai đoạn voting warm-up.
- [src/pipeline.py](src/pipeline.py): expose 2 attribute mới `last_used_roi: bool` và
  `last_roi_bounds: Optional[Tuple[int,int,int,int]]`, cập nhật mỗi `process_frame()` — trước
  đây 2 giá trị này chỉ là biến local, không đọc được từ ngoài `pipeline.py`.
- [main.py](main.py): thêm flag `--ui` (`action="store_true"`, mặc định `False`). Mặc định
  (không có `--ui`) → **headless hoàn toàn**: không `cv2.namedWindow`/`imshow`/`waitKey`, không
  import `src/debug_overlay.py` (import lazy chỉ khi `args.ui`), thoát bằng Ctrl+C
  (`KeyboardInterrupt` bắt gọn, không in traceback xấu). Dòng log FPS mỗi giây (vẫn giữ, đã có
  từ trước) được bổ sung thêm `Detect=ROI/FULL-FRAME` và `ROIFail=x/y` để vẫn biết pipeline
  đang làm gì mà không cần cửa sổ. Có `--ui` → hành vi cũ (cửa sổ, thoát bằng phím `q`) cộng
  overlay mới từ `debug_overlay.render()`.

### 12.3 Validation đã làm
- Smoke test thật (YOLO + OSNet thật, không mock): dựng pipeline, gọi `process_frame()`, gọi
  `debug_overlay.render()` trên kết quả — chạy hết không lỗi, xác nhận `pipeline.last_used_roi`/
  `last_roi_bounds` tồn tại và đúng giá trị mặc định.
- Test riêng nhánh vẽ ROI: dựng `MagicMock` giả lập pipeline với `last_used_roi=True`,
  `last_roi_bounds` cụ thể, 1 track bị `aspect_ratio_pass=False`, mode voting đang warm-up —
  gọi `render()` trực tiếp, xác nhận cả 3 nhánh vẽ mới (ROI rectangle, tag AR-MISMATCH, banner
  warm-up) chạy không lỗi (nhánh này không được smoke test đầu tiên chạm tới vì frame nhiễu
  ngẫu nhiên không có detection thật).
- `test_pipeline.py` 11/11 PASS sau khi thêm 2 attribute mới vào `pipeline.py` (không phá logic
  cũ, chỉ thêm state đọc-ngoài).
- KHÔNG tự chạy webcam thật với `--ui` để xem cửa sổ hiển thị đúng hình — cần người dùng tự mở
  bằng `python main.py --mode run --ui` và xác nhận trực quan.

---

## 13. THAY ĐỔI LỚN: Single embedding → Person Registry nhiều người (2026-08-13)

### 13.1 Mục tiêu
Trước đây `logs/reference_embedding.npy` (+ `reference_aspect_ratio.npy`) là DUY NHẤT — mỗi
lần đăng ký GHI ĐÈ người trước, `--mode run` luôn theo đúng người đăng ký gần nhất. Chuyển
sang: nhiều người có thể đăng ký, lưu riêng biệt trong `logs/registry/`, và trước khi chạy
`--mode run` hiện GUI cho chọn ai trong số đã đăng ký làm target phiên đó. Vẫn KHÔNG
multi-target đồng thời — chỉ đổi cách CHỌN 1 người.

### 13.2 Kiến trúc — tách data layer khỏi 2 GUI dùng chung nó
[src/registry.py](src/registry.py) (mới) — thuần Python, không Tkinter — chứa toàn bộ CRUD:
`sanitize_person_name`, `save_person`/`load_person`, `list_registry`, `delete_person`,
`rename_person`. Cả `registration.py` (ghi) và `person_selector.py` (đọc/xóa/đổi tên) đều
import từ đây, tránh trùng lặp logic I/O ở 2 nơi và tránh import chéo lộn xộn giữa 2 module
GUI với nhau.

### 13.3 Quyết định thiết kế đáng chú ý
- **Sanitize tên bằng allowlist, không phải blocklist**: chỉ giữ `\w` (chữ kể cả Unicode/dấu,
  số, `_`) và `-`; MỌI ký tự khác (kể cả `.`, `/`, `\`) bị loại bỏ. Ngăn path traversal bằng
  cấu trúc (không có `.`/`/` nào sống sót) thay vì chỉ pattern-match chuỗi `"../"` — an toàn
  hơn trước các biến thể encode khác của path traversal.
- **`.npz` 1 file/người** thay vì 2 file `.npy` rời như trước — `np.savez(embedding=...,
  aspect_ratio=..., created_at=..., sample_count=...)`, đọc lại với `allow_pickle=False` (đã
  verify hoạt động — numpy lưu string/int scalar dạng dtype native, không cần pickle).
- **`PersonRegistrySelector.run()` tách thành method nhỏ có chủ đích** (`_register_new_blocking`,
  `_show_list_and_wait`) thay vì 1 khối dựng Tk liền mạch — để logic điều phối (auto-redirect
  khi rỗng, trả `None` khi đóng không chọn) unit-test được bằng cách mock 2 method đó, không
  cần mở Tk mainloop thật. Đây là pattern mới so với `TargetRegistrarGUI` cũ (chưa từng được
  unit test trực tiếp vì toàn bộ logic nằm trong 1 class Tk nặng) — nên áp dụng pattern tương
  tự nếu có thêm GUI mới sau này.
- **Bug tiềm ẩn tự phát hiện và sửa khi implement**: `start_sampling()` đã được bind SPACE để
  bấm nhanh (`root.bind("<space>", ...)`) TỪ TRƯỚC — thêm ô nhập tên có thể chứa khoảng trắng
  (vd "Nguyen Van A") mà không xử lý sẽ khiến gõ tên vô tình trigger Start giữa chừng mỗi lần
  gõ dấu cách. Đã thêm guard: `start_sampling()` kiểm tra `root.focus_get() is self.name_entry`
  và bỏ qua nếu đang gõ trong ô tên.
- **Không migrate dữ liệu cũ** (đúng yêu cầu) — 3 file `.npy` cũ (từ MobileNetV3, từ thời chưa
  có aspect ratio, và bản OSNet gần nhất) đều đã đổi tên `_OLD_*.bak`, không file nào còn được
  code đọc. `logs/registry/` hiện rỗng — lần `--mode run` kế tiếp sẽ tự động mở GUI đăng ký.

### 13.4 Thay đổi code
- [src/registry.py](src/registry.py): file mới, xem 13.2.
- [src/registration.py](src/registration.py): thêm `tk.Entry` cho tên (LabelFrame riêng phía
  trên status panel), `start_sampling()` validate tên (sanitize + hỏi xác nhận ghi đè nếu đã
  tồn tại) trước khi cho vào COUNTDOWN, `_finish_collection()` gọi `registry.save_person()`
  thay vì `np.save()` 2 file cố định, `TargetRegistrarGUI.__init__`/`TargetRegistrar.run()` bỏ
  param `output_path` (không còn ý nghĩa — đích lưu giờ suy ra từ tên nhập vào), `run()` trả
  về path `.npz` đã lưu.
- [src/person_selector.py](src/person_selector.py): file mới, `PersonRegistrySelector` — xem
  13.3 về cách tách method để test được.
- [src/pipeline.py](src/pipeline.py): `__init__` đổi tham số `reference_embedding_path` (có
  default trỏ file cố định) → `reference_npz_path` (bắt buộc, không default — thiếu thì raise
  `ValueError` ngay, không âm thầm dùng file cũ nào). Load qua `registry.load_person()` — gộp
  2 bước load embedding + aspect_ratio companion (cũ) thành 1 lệnh.
- [main.py](main.py): bỏ hẳn flag `--embedding`. Mode `register`: `registrar.run()` không
  tham số, log path trả về. Mode `run`: gọi `PersonRegistrySelector(config,
  config_path=args.config).run()` TRƯỚC KHI khởi tạo `FollowPipeline`/mở webcam — nếu trả về
  `None` (falsy) → `logger.error(...)` + `sys.exit(1)` ngay, không có đường nào rơi xuống chạy
  pipeline với dữ liệu không hợp lệ.

### 13.5 Validation đã làm (theo đúng yêu cầu 6 mục trong spec)
Tất cả chạy bằng code thật (không chỉ suy luận) — 40/40 unit test PASS tổng cộng across 4 file
test (`test_pipeline.py` 11, `test_verifier.py` 5, `test_registry.py` 19,
`test_person_selector.py` 5):
1. `test_save_multiple_people_stay_isolated` — 2 người, 2 file `.npz` riêng, không ghi đè,
   `list_registry()` trả đúng cả 2, data (embedding/aspect_ratio/sample_count) không lẫn nhau.
2. `test_delete_removes_only_target` — xóa 1 người, file mất, người còn lại + `list_registry()`
   không bị ảnh hưởng.
3. `test_rename_moves_file_and_preserves_data` — rename giữ nguyên toàn bộ data (embedding,
   aspect_ratio, sample_count), file cũ mất, file mới đúng tên;
   `test_rename_to_existing_name_raises_without_deleting_original` — rename sang tên đã tồn
   tại raise `FileExistsError`, KHÔNG đụng tới file gốc.
4. `test_empty_registry_auto_redirects_to_registration` — registry rỗng → tự gọi
   `_register_new_blocking()` rồi mới hiện danh sách (đã có data sau khi đăng ký).
5. `test_closing_without_selecting_returns_none` — đóng cửa sổ list không chọn ai →
   `run()` trả `None`; đã confirm `main.py` dùng giá trị này để `sys.exit(1)` thay vì chạy
   tiếp.
6. `test_spaces_become_underscore`, `test_special_characters_stripped`,
   `test_path_traversal_neutralized`, `test_empty_raises`, `test_whitespace_only_raises`,
   `test_only_special_characters_raises`, `test_diacritics_preserved` — toàn bộ case sanitize
   theo đúng spec.

Smoke test tích hợp thật (YOLO + OSNet thật, không mock): lưu 1 người thật vào
`logs/registry/` thật qua `registry.save_person()`, dựng `FollowPipeline` với path đó, gọi
`process_frame()` — chạy hết không lỗi, xóa dọn sau khi test xong.

### 13.6 Việc người dùng cần làm trước khi test lại bằng webcam thật
1. Chạy `python main.py --mode run` — vì `logs/registry/` đang rỗng, sẽ tự động mở GUI đăng ký
   (nhập tên trước khi bắt đầu).
2. Thử đăng ký 2+ người thật để xác nhận trực quan: GUI chọn người hiện đúng danh sách, nút
   Select/Rename/Delete/Register New hoạt động đúng khi click thật — phần này CHƯA được xác
   nhận bằng tương tác thật, chỉ unit test được phần logic điều phối (xem README mục "Vấn đề
   đã biết").
3. Xác nhận riêng: gõ tên có khoảng trắng vào ô Person Name có bị trigger Start giữa chừng
   không (guard đã thêm ở 13.3 nhưng chưa test bằng gõ phím thật).

---

## 14. CHẨN ĐOÁN HIỆU NĂNG: FPS thấp (3-5fps) + chất lượng ảnh xấu (2026-08-13)

### 14.1 Vấn đề người dùng báo cáo
Chạy `--mode run` thật bị lag nặng (~3-5fps) và hình ảnh xấu. Không có webcam thật để tự tái
hiện — đã chẩn đoán bằng 2 hướng: (a) đo timing thật trên máy dev cho phần compute, (b) sửa
1 nguyên nhân rất phổ biến cho đúng combo triệu chứng "lag + ảnh xấu" là cấu hình capture.

### 14.2 Đo được bằng số liệu thật (không đoán) — compute là nguyên nhân chính cho FPS thấp
Trên máy dev (CPU, không GPU — đã xác nhận `torch.cuda.is_available()==False` từ trước):
- `YoloDetector.track()` (YOLO11n ONNX, frame 640x480): **~64ms/frame**.
- `OSNetVerifier.extract()` (1 crop): **~76ms/track**.
- Tổng riêng detect + verify 1 người: **~140ms/frame → trần lý thuyết ~7fps**, CHƯA tính
  overhead webcam I/O, vẽ overlay (`--ui`), ghi CSV, dispatch thread.

So sánh với benchmark MobileNetV3 cũ ở mục 4.2 (~9.65-14.5ms/crop tùy N crop, đã set
`torch.set_num_threads(1)`): OSNet nặng hơn ~5-8 lần trên CPU cho 1 crop. Đây là hệ quả trực
tiếp, đã biết trước, của quyết định đổi MobileNetV3→OSNet ở mục 11 (đổi vì lý do độ chính xác
re-id, đã đánh đổi tốc độ — chưa có ai định lượng cụ thể mức đánh đổi này cho tới bây giờ).
**Kết luận**: 3-5fps thực tế của người dùng khớp hợp lý với ~7fps trần lý thuyết đo được cộng
thêm overhead thật (webcam, vẽ, I/O) — nhiều khả năng đây là giới hạn compute CPU thật, không
phải bug.

### 14.3 Sửa riêng cho "chất lượng ảnh xấu" — cấu hình capture webcam
`cv2.VideoCapture()` ở cả `main.py` (`WebcamStreamThread`) và `registration.py` trước đây
không set FOURCC/resolution/buffer — nguyên nhân rất phổ biến cho đúng combo "lag + ảnh xấu":
camera có thể fallback về format raw/độ phân giải thấp hơn config (`input_resolution`), rồi bị
`cv2.resize()` upscale mờ; buffer mặc định (thường nhiều frame) khiến `read()` trả frame cũ
(stale), cộng dồn cảm giác trễ khi xử lý không theo kịp tốc độ capture.

[src/camera_utils.py](src/camera_utils.py) (mới): `configure_capture(cap, width, height,
target_fps)` — set FOURCC=MJPG (nén trên camera, giải phóng băng thông USB cho độ phân giải/
FPS cao hơn), resolution theo `input_resolution`, `CAP_PROP_BUFFERSIZE=1`. Log lại resolution/
FPS/FOURCC THỰC TẾ camera trả về (`cap.get(...)` sau khi set) — vì không phải camera nào cũng
tuân theo yêu cầu, cần biết ngay có mismatch hay không thay vì đoán. Áp dụng ở cả
`WebcamStreamThread.start()` (main.py) và `TargetRegistrarGUI.run()` (registration.py).

### 14.4 Đo lường mới: `pipeline.last_timing_ms`
[src/pipeline.py](src/pipeline.py): thêm `t_frame_start`/`t_detect_start`/`t_verify_start`,
tính `detect_ms`/`verify_ms`/`total_ms` mỗi frame, lưu vào `self.last_timing_ms` (dict, cùng
pattern với `last_used_roi`). Hiện trong dòng log FPS mỗi giây ở `main.py` (cả headless lẫn
`--ui`) và panel telemetry trong [src/debug_overlay.py](src/debug_overlay.py). Mục đích: lần
người dùng chạy tiếp theo trên máy thật, có số liệu thật để biết chính xác detect hay verify
đang chiếm phần lớn thời gian, thay vì đoán dựa trên benchmark máy dev.

### 14.5 Validation đã làm
- Đo thật `detect_ms`/`verify_ms` trên máy dev bằng script riêng (không phải suy luận) — xem
  14.2.
- `test_pipeline.py` 11/11 PASS sau khi thêm timing instrumentation (chỉ thêm biến local + set
  1 instance attribute, không đổi control flow).
- Smoke test `debug_overlay.render()` với `pipeline.last_timing_ms` là dict thật (không phải
  `MagicMock` tự sinh, vốn không hỗ trợ subscript) — xác nhận panel timing vẽ không lỗi.
- Syntax-check toàn bộ file đã đổi + `main.py` import sạch.
- 40/40 test toàn bộ 4 file test vẫn PASS sau các thay đổi.

### 14.6 CHƯA làm / cần quyết định tiếp
- **CHƯA sửa `WebcamStreamThread`/`TargetRegistrarGUI` bằng webcam thật** — `configure_capture`
  chỉ mới verify bằng syntax-check + logic đọc, chưa có xác nhận camera thật của người dùng có
  tuân theo MJPG/resolution hay không (dòng log mới sẽ cho biết ngay khi chạy).
- **Không tự ý đổi `osnet_variant` sang bản nhẹ hơn** (`osnet_x0_75`/`x0_5`/`x0_25` — đều có
  checkpoint Market1501 riêng trong Model Zoo, nhẹ hơn tới ~12 lần GFLOPs so với `osnet_x1_0`
  nhưng Rank-1 thấp hơn: 93.7/92.5/91.2 so với 94.2). Đây là đánh đổi tốc độ/độ chính xác trực
  tiếp ảnh hưởng tới đúng vấn đề false-positive đã sửa ở phiên trước — để người dùng quyết định
  thay vì tự chọn.
- Chưa đo FPS thực tế bằng webcam thật sau khi áp dụng cả 2 fix (camera config + timing) — cần
  người dùng tự chạy `python main.py --mode run --ui` và đọc panel/log mới.

---

## 15. BUG THẬT PHÁT HIỆN QUA LOG WEBCAM THẬT: ROI bị xóa `active_target_id` quá sớm (2026-08-13)

### 15.1 Bằng chứng từ log thật (người dùng cung cấp)
Chạy `python main.py --mode run` thật (headless), log cho thấy:
- `15:41:09` — lock `track_id=3`.
- `15:41:10` — `Detect=ROI`, target vẫn found, `ROIFail=0/5` (đúng thiết kế).
- `15:41:11` — `Target Found=False`, `Detect=FULL-FRAME`, `ROIFail=1/5`.
- Từ đó tới `15:41:30` (~19 giây, ~17 dòng log) — kẹt liên tục ở `FULL-FRAME`, `ROIFail=1/5`
  ĐỨNG YÊN (không tăng thêm, không reset về 0), `Target Found=False` gần như toàn bộ.
- `15:41:34` — full-frame scan phát hiện **3 người cùng lúc trong khung hình** (3 dòng
  `Detection at bbox...`), giải thích `verify_ms` dao động mạnh 123-426ms (so với ~76-90ms khi
  chỉ 1 track qua ROI) — verify chạy cho CẢ 3 track mỗi frame thay vì 1.
- Cảnh báo `GMC failed, falling back to identity` lặp lại nhiều lần — Ultralytics ByteTrack
  không tính được optical flow giữa 2 lần gọi liên tiếp (kích thước frame khác nhau giữa ROI
  crop và full frame) — **xác nhận thực tế** rủi ro kiến trúc đã ghi ở mục 9.2/README
  ("ROI đổi kích thước crop mỗi frame có thể làm gián đoạn ByteTrack nội bộ"). Không fatal
  (tự fallback), nhưng là bằng chứng rủi ro đó có thật, không chỉ là lo ngại lý thuyết.

### 15.2 Nguyên nhân gốc — bug logic thật trong `pipeline.py`
Cơ chế "ROI thất bại không xóa `active_target_id` ngay" (mục 9.2) chỉ được implement cho ĐÚNG
1 trường hợp: ROI-constrained scan trả về **0 detection**. Trường hợp thứ 2 — ROI scan trả về
CÓ track (đúng track_id đang theo dõi) nhưng track đó KHÔNG pass gate (similarity dip do
motion blur/góc xấu/che khuất tạm thời — rất bình thường trong tracking thật) — rơi vào nhánh
chung cuối hàm:
```python
if target_track is None:
    self.active_target_id = None   # xóa NGAY, bất kể đang dùng ROI hay full-frame
    return AngleResult(target_found=False)
```
Nhánh này KHÔNG phân biệt `used_roi` — xóa `active_target_id` ngay cả khi đang dùng ROI, phá vỡ
đúng cơ chế retry mà mục 9.2 định xây. Log thật khớp chính xác kịch bản này: lock lúc `15:41:09`
→ ROI OK 1 lần → mất ngay ở lần verify tiếp theo → không bao giờ dùng lại ROI được nữa (vì
`active_target_id` đã `None`) → kẹt ở full-frame quét lại toàn bộ khung hình, verify nhiều
người hơn, chậm hơn, VÀ khó khóa lại (vì `similarity_threshold=0.80` khá chặt, chưa calibrate
cho OSNet — xem mục 11.6).

### 15.3 Fix
[src/pipeline.py](src/pipeline.py) — tách rõ 2 nhánh theo `used_roi`:
```python
if used_roi:
    if target_track is not None:
        self._roi_failure_count = 0
    else:
        self._roi_failure_count += 1
        return AngleResult(target_found=False)   # KHÔNG xóa active_target_id
elif target_track is None:
    self.active_target_id = None                  # chỉ xóa khi full-frame cũng miss thật
    self._roi_failure_count = 0
    return AngleResult(target_found=False)
```
Giờ hành vi nhất quán với nhánh "0 detection" đã có: ROI thất bại (dù do 0 detection hay do
gate reject) đều chỉ tăng `_roi_failure_count` và giữ `active_target_id`, cho retry tới khi đạt
`roi_failure_max_frames` mới ép full-frame; chỉ full-frame miss thật mới xóa identity.

### 15.4 Validation
Thêm `test_roi_gate_rejection_does_not_immediately_discard_target` vào `test_pipeline.py` —
dùng `ema_alpha=1.0` (tắt hẳn khả năng EMA tự hấp thụ dip, cô lập đúng path gate-rejection cần
test, tách biệt khỏi test đã có về khả năng EMA che 1 frame xấu). Xác nhận: frame lock →
frame gate-reject (raw thấp) → `active_target_id` KHÔNG bị xóa, `_roi_failure_count=1`, vẫn
dùng ROI → frame retry thành công → khóa lại đúng track, `_roi_failure_count` reset về 0.
41/41 test toàn bộ 4 file PASS (bao gồm test mới).

### 15.5 Còn treo — cần người dùng xác nhận lại bằng webcam thật
- Chạy lại `python main.py --mode run --ui`, kỳ vọng: sau khi lock target, các lần dip
  similarity/aspect-ratio thoáng qua không còn làm mất track ngay — cần verify FPS ổn định hơn
  và `ROIFail` không còn kẹt vĩnh viễn ở 1 mà dao động 0-1-0 bình thường khi target còn trong
  khung.
- Chưa xác nhận `similarity_threshold=0.80` có phải nguyên nhân phụ khiến khó khóa lại target
  hay không — cần log Scenario A/B mới để calibrate, như đã ghi ở mục 11.6.
- Chưa hỏi lại người dùng "chất lượng hình ảnh" có cải thiện sau fix `camera_utils.py` (mục 14)
  hay chưa — log mới chỉ xác nhận resolution/FPS request được camera chấp nhận, không xác nhận
  trực quan chất lượng ảnh.

**Cập nhật cùng ngày**: log webcam thật lần 2 (sau fix mục 15) xác nhận cải thiện — `ROIFail=0/5`
(không còn kẹt), `FPS=4.73`, `verify=112ms` (chỉ 1 track, so với 220-426ms lúc trước khi verify
nhiều track). Xem mục 16 về nguồn warning còn lại (GMC) và cách đã sửa.

---

## 16. BUG THẬT THỨ 2: sai tracker so với spec gốc — BoT-SORT thay vì ByteTrack (2026-08-13)

### 16.1 Phát hiện qua log thật
Log webcam thật (cả 2 lần chạy) đầy warning lặp lại:
```
WARNING GMC failed, falling back to identity: OpenCV(5.0.0) ...lkpyramid.cpp:1185:
error: (-215:Assertion failed) prevPyr[level*lvlStep1].size() == nextPyr[level*lvlStep2].size()
```
GMC (Global Motion Compensation) là bước trong **BoT-SORT** ước lượng chuyển động camera bằng
optical flow thưa (Lucas-Kanade pyramid) giữa 2 frame liên tiếp — cần 2 frame CÙNG kích thước.
Kiến trúc ROI-constrained detection (mục 9) cố ý feed frame khác kích thước liên tiếp (ROI crop
rồi full frame, xen kẽ) vào CÙNG 1 tracker instance persistent — đúng nguyên nhân assertion fail.

### 16.2 Nguyên nhân gốc — lệch spec đã có từ đầu, không phải do ROI mới thêm
`src/detector.py` — cả docstring class (`"Wrap YOLO model for person tracking using
Ultralytics ByteTrack"`) lẫn `plan.md` gốc đều nói rõ dùng **ByteTrack**. Nhưng dòng code thật:
```python
results = self.model.track(frame, persist=True, classes=[0], verbose=False)
```
**không hề chỉ định `tracker=`** — Ultralytics mặc định dùng **BoT-SORT** (`botsort.yaml`,
có `gmc_method: sparseOptFlow`) khi không chỉ định rõ, KHÔNG phải ByteTrack. Đây là lỗi lệch
spec đã tồn tại từ những review đầu tiên (mục 3, "ĐÃ ĐỐI CHIẾU & SỬA XONG" ở mục 7.1 không phát
hiện ra vì lúc đó tất cả input đều cùng kích thước, GMC chưa bao giờ fail) — chỉ lộ ra khi ROI
làm frame size thay đổi giữa các lần gọi.

### 16.3 Fix
[src/detector.py](src/detector.py): thêm `tracker="bytetrack.yaml"` vào lệnh gọi
`self.model.track(...)` — dùng đúng ByteTrack như spec gốc luôn yêu cầu. `bytetrack.yaml`
(xem `.venv/Lib/site-packages/ultralytics/cfg/trackers/bytetrack.yaml`) **không có bước GMC
nào cả** — không chỉ ẩn warning mà loại bỏ hẳn nguồn gây lỗi. Cũng không mất gì: BoT-SORT có hỗ
trợ appearance ReID tùy chọn (`with_reid`) nhưng vốn đã tắt mặc định (`False`) và ta tự làm
re-id bằng OSNet bên ngoài rồi.

### 16.4 Validation
Script test riêng: feed liên tiếp full-frame (480,640) → crop nhỏ (120,100) → full-frame lại →
crop khác (150,90) → full-frame vào CÙNG 1 `YoloDetector` instance (đúng kịch bản gây lỗi thật)
— xác nhận **0 warning GMC nào xuất hiện** (trước đây với BoT-SORT chắc chắn sẽ fail ở mọi lần
đổi kích thước). 41/41 test 4 file vẫn PASS sau khi đổi tracker.

### 16.5 Về warning còn lại trong log người dùng — không phải bug
`Detection at bbox (...) missing track_id. Assigned fallback temporary ID: 1000` — đây là hành
vi ĐÚNG THEO SPEC (plan.md yêu cầu rõ: track mới chưa có ID ổn định ở vài frame đầu → gán tạm +
log warning, xem [detector.py:64-71](src/detector.py:64)), không phải bug. Xảy ra khi ByteTrack
gán 1 track mới (người mới vào khung/track cũ bị mất rồi tái phát hiện) chưa qua đủ frame để
được xác nhận ID ổn định — bình thường, chỉ là log warning từng bị "chìm" giữa hàng loạt GMC
warning nên trông có vẻ nhiều lỗi hơn thực tế.
