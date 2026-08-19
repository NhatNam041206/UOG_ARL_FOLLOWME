import os
import glob
import time
import logging
import cv2
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from typing import Optional, List, Dict, Any

from src.detector import YoloDetector
from src.verifier import OSNetVerifier
from src.view_estimator import ViewEstimator
from src import registry
from src.camera_utils import configure_capture

logger = logging.getLogger(__name__)


class RawDataCapturerGUI:
    """
    Phase 1 GUI: Capture raw frames only into logs/raw_captures/<person_name>/frame_XXXX.jpg.
    Does NOT calculate embeddings, aspect ratios, or pose proportions during capture.
    """
    def __init__(self, config: dict, config_path: str = "config/settings.yaml"):
        self.config = config
        self.config_path = config_path
        self.output_dir: Optional[str] = None
        self.sanitized_name: Optional[str] = None
        self.success: bool = False
        self.error_message: Optional[str] = None

        self.camera_index = config.get("camera_index", 0)
        self.input_resolution = config.get("input_resolution", [640, 480])
        self.roi_percent = list(config.get("roi_percent", [0.3, 0.2, 0.7, 0.9]))
        self.min_capture_frames = config.get("min_capture_frames", config.get("registration_min_samples", 20))
        self.sample_interval = config.get("registration_sample_interval_frames", 3)

        yolo_path = config.get("yolo_model_path", "yolo11n.pt")
        logger.info(f"Initializing RawDataCapturerGUI (Camera index: {self.camera_index})...")
        self.detector = YoloDetector(yolo_path)

        self.cap = None
        self.state = "IDLE"  # IDLE, CAPTURING, COMPLETED, FAILED
        self.frame_counter = 0
        self.saved_counter = 0
        self.skipped_frames = 0
        self.thumbnail_images = []
        self.person_name_var = None

        # Build GUI Window
        self.root = tk.Tk()
        self.root.title("Phase 1: Raw Data Capture Tool")
        self.root.geometry("1280x820")
        self.root.minsize(1100, 750)
        self.root.configure(bg="#1e1e2e")

        self._style_gui()
        self._build_layout()
        self._bind_events()

    def _style_gui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background="#1e1e2e", foreground="#cdd6f4", font=("Segoe UI", 10))
        style.configure("TLabelFrame", background="#1e1e2e", foreground="#cba6f7", font=("Segoe UI", 10, "bold"))
        style.configure("TButton", font=("Segoe UI", 10, "bold"), background="#89b4fa", foreground="#11111b")
        style.map("TButton", background=[("active", "#b4befe")])
        style.configure("Start.TButton", font=("Segoe UI", 11, "bold"), background="#a6e3a1", foreground="#11111b")
        style.map("Start.TButton", background=[("active", "#94e2d5")])
        style.configure("Save.TButton", font=("Segoe UI", 10, "bold"), background="#fab387", foreground="#11111b")
        style.configure("TProgressbar", thickness=18, troughcolor="#313244", background="#a6e3a1")

    def _build_layout(self):
        header_frame = tk.Frame(self.root, bg="#181825", pady=10, padx=15)
        header_frame.pack(side=tk.TOP, fill=tk.X)

        title_lbl = tk.Label(
            header_frame,
            text="📷 PHASE 1: RAW DATA CAPTURER (360° Multi-Angle Data Collection)",
            font=("Segoe UI", 14, "bold"),
            fg="#cba6f7",
            bg="#181825"
        )
        title_lbl.pack(side=tk.LEFT)

        subtitle_lbl = tk.Label(
            header_frame,
            text=f"Camera Index: {self.camera_index}  |  Min Required: {self.min_capture_frames} frames",
            font=("Segoe UI", 10),
            fg="#a6adc8",
            bg="#181825"
        )
        subtitle_lbl.pack(side=tk.RIGHT)

        main_content = tk.Frame(self.root, bg="#1e1e2e", padx=10, pady=10)
        main_content.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        main_content.grid_columnconfigure(0, weight=1)
        main_content.grid_columnconfigure(1, weight=0, minsize=420)
        main_content.grid_rowconfigure(0, weight=1)

        # Left Column: Video Viewport
        self.video_panel = tk.LabelFrame(main_content, text=" Live Camera Feed & ROI Overlay ", bg="#1e1e2e", fg="#cba6f7", padx=5, pady=5)
        self.video_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.instruction_label = tk.Label(
            self.video_panel,
            text="💡 Nhập tên người, sau đó nhấn START CAPTURE. Xoay 360° để thu thập đủ các góc (trước/sau/trái/phải).",
            font=("Segoe UI", 10, "bold"),
            bg="#181825",
            fg="#89b4fa",
            pady=8,
            padx=10,
            wraplength=650,
            justify="left",
            anchor="w"
        )
        self.instruction_label.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        self.video_label = tk.Label(self.video_panel, bg="#11111b", text="Initializing Camera Feed...", fg="#6c7086", font=("Segoe UI", 12))
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Right Column: Controls & Metrics Sidebar
        sidebar = tk.Frame(main_content, bg="#1e1e2e", width=420)
        sidebar.grid(row=0, column=1, sticky="nsew")

        name_frame = tk.LabelFrame(sidebar, text=" Person Name (bắt buộc trước khi capture) ", bg="#1e1e2e", fg="#cba6f7", padx=10, pady=10)
        name_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        self.person_name_var = tk.StringVar()
        self.name_entry = tk.Entry(
            name_frame, textvariable=self.person_name_var,
            font=("Segoe UI", 11), bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
        )
        self.name_entry.pack(fill=tk.X)

        # Status & Control Frame
        status_frame = tk.LabelFrame(sidebar, text=" Capture Status & Control ", bg="#1e1e2e", fg="#cba6f7", padx=10, pady=10)
        status_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        self.badge_lbl = tk.Label(
            status_frame,
            text="READY - PRESS START CAPTURE",
            font=("Segoe UI", 11, "bold"),
            fg="#11111b",
            bg="#89b4fa",
            pady=6,
            relief=tk.RAISED
        )
        self.badge_lbl.pack(fill=tk.X, pady=(0, 10))

        prog_info_frame = tk.Frame(status_frame, bg="#1e1e2e")
        prog_info_frame.pack(fill=tk.X, pady=(0, 5))

        tk.Label(prog_info_frame, text="Raw Frames Saved:", bg="#1e1e2e", fg="#cdd6f4").pack(side=tk.LEFT)
        self.sample_count_lbl = tk.Label(prog_info_frame, text=f"0 / {self.min_capture_frames}", font=("Segoe UI", 10, "bold"), bg="#1e1e2e", fg="#a6e3a1")
        self.sample_count_lbl.pack(side=tk.RIGHT)

        self.progress_bar = ttk.Progressbar(status_frame, maximum=self.min_capture_frames, value=0)
        self.progress_bar.pack(fill=tk.X, pady=(0, 8))

        metrics_frame = tk.Frame(status_frame, bg="#1e1e2e")
        metrics_frame.pack(fill=tk.X, pady=(0, 10))

        self.skipped_lbl = tk.Label(metrics_frame, text="Skipped: 0", bg="#1e1e2e", fg="#f38ba8", font=("Segoe UI", 9))
        self.skipped_lbl.pack(side=tk.LEFT)

        btn_frame = tk.Frame(status_frame, bg="#1e1e2e")
        btn_frame.pack(fill=tk.X)

        self.start_btn = ttk.Button(btn_frame, text="▶ START CAPTURE", style="Start.TButton", command=self.toggle_capture)
        self.start_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.finish_btn = ttk.Button(btn_frame, text="💾 SAVE & FINISH", style="Save.TButton", command=self.save_and_finish)
        self.finish_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

        # ROI Sliders Frame
        roi_frame = tk.LabelFrame(sidebar, text=" Interactive ROI Region Config (%) ", bg="#1e1e2e", fg="#cba6f7", padx=10, pady=10)
        roi_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        self.roi_x1_var = tk.DoubleVar(value=round(self.roi_percent[0] * 100, 1))
        self.roi_y1_var = tk.DoubleVar(value=round(self.roi_percent[1] * 100, 1))
        self.roi_x2_var = tk.DoubleVar(value=round(self.roi_percent[2] * 100, 1))
        self.roi_y2_var = tk.DoubleVar(value=round(self.roi_percent[3] * 100, 1))

        self._create_roi_slider(roi_frame, "ROI X1 (Left %):", self.roi_x1_var, 0, 90)
        self._create_roi_slider(roi_frame, "ROI Y1 (Top %):", self.roi_y1_var, 0, 90)
        self._create_roi_slider(roi_frame, "ROI X2 (Right %):", self.roi_x2_var, 10, 100)
        self._create_roi_slider(roi_frame, "ROI Y2 (Bottom %):", self.roi_y2_var, 10, 100)

        # Thumbnails Gallery
        gallery_frame = tk.LabelFrame(sidebar, text=" Saved Raw Frame Crops ", bg="#1e1e2e", fg="#cba6f7", padx=5, pady=5)
        gallery_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.gallery_canvas = tk.Canvas(gallery_frame, bg="#11111b", highlightthickness=0)
        self.gallery_scrollbar = ttk.Scrollbar(gallery_frame, orient=tk.VERTICAL, command=self.gallery_canvas.yview)
        self.gallery_inner = tk.Frame(self.gallery_canvas, bg="#11111b")

        self.gallery_inner.bind("<Configure>", lambda e: self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all")))
        self.gallery_canvas.create_window((0, 0), window=self.gallery_inner, anchor="nw")
        self.gallery_canvas.configure(yscrollcommand=self.gallery_scrollbar.set)

        self.gallery_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.gallery_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _create_roi_slider(self, parent, label_text, var, from_val, to_val):
        frame = tk.Frame(parent, bg="#1e1e2e")
        frame.pack(fill=tk.X, pady=2)
        
        lbl_frame = tk.Frame(frame, bg="#1e1e2e")
        lbl_frame.pack(fill=tk.X)
        tk.Label(lbl_frame, text=label_text, bg="#1e1e2e", fg="#cdd6f4", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        val_lbl = tk.Label(lbl_frame, textvariable=var, bg="#1e1e2e", fg="#89b4fa", font=("Segoe UI", 9, "bold"))
        val_lbl.pack(side=tk.RIGHT)

        slider = ttk.Scale(frame, from_=from_val, to=to_val, variable=var, orient=tk.HORIZONTAL, command=lambda v: self._update_roi_from_sliders())
        slider.pack(fill=tk.X)

    def _update_roi_from_sliders(self):
        x1 = self.roi_x1_var.get() / 100.0
        y1 = self.roi_y1_var.get() / 100.0
        x2 = self.roi_x2_var.get() / 100.0
        y2 = self.roi_y2_var.get() / 100.0

        if x2 <= x1 + 0.05:
            x2 = min(1.0, x1 + 0.05)
            self.roi_x2_var.set(round(x2 * 100, 1))
        if y2 <= y1 + 0.05:
            y2 = min(1.0, y1 + 0.05)
            self.roi_y2_var.set(round(y2 * 100, 1))

        self.roi_percent = [x1, y1, x2, y2]

    def _bind_events(self):
        self.root.bind("<space>", lambda e: self.toggle_capture())
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def toggle_capture(self):
        if self.root.focus_get() is self.name_entry:
            return

        if self.state == "IDLE" or self.state == "FAILED":
            try:
                raw_name = self.person_name_var.get()
                self.sanitized_name = registry.sanitize_person_name(raw_name)
            except ValueError as e:
                messagebox.showerror("Invalid Name", str(e))
                return

            capture_dir = registry.raw_capture_dir(self.sanitized_name)
            os.makedirs(capture_dir, exist_ok=True)
            existing_files = sorted(glob.glob(os.path.join(capture_dir, "frame_*.jpg")))

            if existing_files:
                response = messagebox.askyesnocancel(
                    "Folder Exists",
                    f"Folder raw captures cho '{self.sanitized_name}' đã có {len(existing_files)} ảnh.\n\n"
                    f"• Bấm 'YES' để GHI ĐÈ (xóa ảnh cũ)\n"
                    f"• Bấm 'NO' để THÊM ẢNH (tiếp tục counter)\n"
                    f"• Bấm 'CANCEL' để HỦY"
                )
                if response is None:  # Cancel
                    return
                elif response is True:  # Overwrite
                    for f in existing_files:
                        try:
                            os.remove(f)
                        except Exception:
                            pass
                    self.saved_counter = 0
                    for widget in self.gallery_inner.winfo_children():
                        widget.destroy()
                    self.thumbnail_images.clear()
                else:  # Append
                    self.saved_counter = len(existing_files)

            self.output_dir = capture_dir
            self.state = "CAPTURING"
            self.start_btn.config(text="⏸ PAUSE CAPTURE")
            self.badge_lbl.config(text="CAPTURING IN PROGRESS", bg="#a6e3a1", fg="#11111b")
            logger.info(f"Raw data capture started for '{self.sanitized_name}' -> '{capture_dir}'")

        elif self.state == "CAPTURING":
            self.state = "IDLE"
            self.start_btn.config(text="▶ RESUME CAPTURE")
            self.badge_lbl.config(text="PAUSED", bg="#f9e2af", fg="#11111b")

    def save_and_finish(self):
        if self.saved_counter < self.min_capture_frames:
            messagebox.showerror(
                "Dữ liệu không đủ",
                f"Cần ít nhất {self.min_capture_frames} ảnh thô hợp lệ trước khi lưu!\n"
                f"Hiện tại mới thu thập được: {self.saved_counter} ảnh."
            )
            return

        self.state = "COMPLETED"
        self.success = True
        self.badge_lbl.config(text="RAW CAPTURE COMPLETED!", bg="#a6e3a1", fg="#11111b")
        messagebox.showinfo(
            "Capture Complete",
            f"Đã lưu thành công {self.saved_counter} ảnh thô cho '{self.sanitized_name}'!\n"
            f"Thư mục: '{os.path.abspath(self.output_dir)}'"
        )
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

    def _add_thumbnail(self, crop):
        try:
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_crop)
            pil_img.thumbnail((70, 90))
            tk_img = ImageTk.PhotoImage(pil_img)
            self.thumbnail_images.append(tk_img)

            idx = self.saved_counter
            thumb_frame = tk.Frame(self.gallery_inner, bg="#181825", bd=1, relief=tk.SOLID)
            thumb_frame.grid(row=(idx - 1) // 4, column=(idx - 1) % 4, padx=4, pady=4)

            lbl = tk.Label(thumb_frame, image=tk_img, bg="#181825")
            lbl.pack()
        except Exception as e:
            logger.warning(f"Failed to add thumbnail preview: {e}")

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            messagebox.showerror(
                "Camera Error",
                f"Unable to open Camera Index {self.camera_index}.\n"
                f"Please check your webcam connection or settings.yaml!"
            )
            self.root.destroy()
            raise RuntimeError(f"Camera index {self.camera_index} failed to open.")

        if self.input_resolution and len(self.input_resolution) == 2:
            configure_capture(self.cap, int(self.input_resolution[0]), int(self.input_resolution[1]))

        self.root.after(15, self._video_loop)
        self.root.mainloop()

    def _video_loop(self):
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.root.after(30, self._video_loop)
            return

        if self.config.get("flip_horizontal", False):
            frame = cv2.flip(frame, 1)

        if self.input_resolution and len(self.input_resolution) == 2:
            rw, rh = int(self.input_resolution[0]), int(self.input_resolution[1])
            frame = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)

        h, w = frame.shape[:2]
        rx1, ry1 = int(w * self.roi_percent[0]), int(h * self.roi_percent[1])
        rx2, ry2 = int(w * self.roi_percent[2]), int(h * self.roi_percent[3])

        display = cv2.GaussianBlur(frame, (51, 51), 0)
        if rx2 > rx1 and ry2 > ry1:
            display[ry1:ry2, rx1:rx2] = frame[ry1:ry2, rx1:rx2]

        if self.state == "IDLE":
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (255, 165, 0), 2)
            self.instruction_label.config(
                text="💡 Nhập tên người, sau đó nhấn START CAPTURE để bắt đầu thu thập ảnh thô.",
                fg="#89b4fa"
            )

        elif self.state == "CAPTURING":
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
            self.instruction_label.config(
                text=f"📸 Đang thu thập ảnh thô (Đã lưu: {self.saved_counter} ảnh)... Xoay 360° để thu thập đa góc.",
                fg="#a6e3a1"
            )

            self.frame_counter += 1
            if self.frame_counter % self.sample_interval == 0:
                roi_crop = frame[ry1:ry2, rx1:rx2]
                if roi_crop.size > 0:
                    detections = self.detector.track(roi_crop)
                    if len(detections) == 1:
                        bx1, by1, bx2, by2 = detections[0]["bbox"]
                        person_crop = roi_crop[max(0, by1):min(roi_crop.shape[0], by2),
                                               max(0, bx1):min(roi_crop.shape[1], bx2)]
                        if person_crop.size > 0 and person_crop.shape[0] > 15 and person_crop.shape[1] > 15:
                            self.saved_counter += 1
                            img_name = f"frame_{self.saved_counter:04d}.jpg"
                            img_path = os.path.join(self.output_dir, img_name)
                            cv2.imwrite(img_path, person_crop)

                            self._add_thumbnail(person_crop)
                            self.progress_bar["value"] = self.saved_counter
                            self.sample_count_lbl.config(text=f"{self.saved_counter} / {self.min_capture_frames}")
                    else:
                        self.skipped_frames += 1
                        self.skipped_lbl.config(text=f"Skipped: {self.skipped_frames} (Need 1 person in ROI)")

        rgb_frame = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        panel_w = max(320, self.video_panel.winfo_width() - 20)
        panel_h = max(240, self.video_panel.winfo_height() - 35)

        img_h, img_w = display.shape[:2]
        aspect = img_w / float(img_h)

        target_w = panel_w
        target_h = int(target_w / aspect)
        if target_h > panel_h:
            target_h = panel_h
            target_w = int(target_h * aspect)

        target_w = max(100, target_w)
        target_h = max(100, target_h)

        pil_img = Image.fromarray(rgb_frame)
        pil_img = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
        tk_img = ImageTk.PhotoImage(pil_img)

        self.video_label.config(image=tk_img)
        self.video_label.image = tk_img

        self.root.after(15, self._video_loop)

    def _on_close(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


class RawDataCapturer:
    """Standalone launcher for Phase 1 RawDataCapturerGUI."""
    def __init__(self, config: dict):
        self.config = config

    def run(self) -> str:
        gui = RawDataCapturerGUI(self.config)
        gui.run()
        if not gui.success:
            err_msg = gui.error_message or "Raw data capture was cancelled or incomplete."
            raise RuntimeError(err_msg)
        return gui.sanitized_name


class EmbeddingBuilder:
    """
    Phase 2: Builds multi-view OSNet embeddings and aspect ratio metadata from captured raw frames.
    Outputs logs/registry/<person_name>.npz.
    """
    def __init__(self, config: dict):
        self.config = config
        yolo_path = config.get("yolo_model_path", "yolo11n.pt")
        osnet_variant = config.get("osnet_variant", "osnet_x1_0")
        pose_model_path = config.get("pose_model_path", "yolo11n-pose.pt")

        self.detector = YoloDetector(yolo_path)
        self.verifier = OSNetVerifier(osnet_variant)
        self.view_estimator = ViewEstimator(pose_model_path)
        self.min_build_frames = config.get("min_build_frames", config.get("registration_min_samples", 20))

    def build_embedding(self, person_name: str) -> str:
        sanitized_name = registry.sanitize_person_name(person_name)
        capture_dir = registry.raw_capture_dir(sanitized_name)

        if not os.path.exists(capture_dir):
            raise FileNotFoundError(f"Raw capture directory does not exist: '{capture_dir}'")

        image_paths = sorted(glob.glob(os.path.join(capture_dir, "frame_*.jpg")))
        if not image_paths:
            image_paths = sorted(glob.glob(os.path.join(capture_dir, "*.jpg")) + glob.glob(os.path.join(capture_dir, "*.png")))

        if len(image_paths) < self.min_build_frames:
            raise ValueError(
                f"Không đủ ảnh để build embedding cho '{sanitized_name}': "
                f"{len(image_paths)} < min required ({self.min_build_frames})"
            )

        logger.info(f"Building multi-view embedding for '{sanitized_name}' from {len(image_paths)} raw frames...")

        view_groups: Dict[str, List[np.ndarray]] = {'front': [], 'right': [], 'back': [], 'left': []}
        aspect_ratios = []
        valid_embeddings = []
        valid_crops = []

        for img_path in image_paths:
            crop = cv2.imread(img_path)
            if crop is None or crop.size == 0:
                logger.warning(f"Skipping corrupt or unreadable frame: {img_path}")
                continue

            detections = self.detector.track(crop)
            if not detections:
                logger.warning(f"No person detected in raw frame, skipping: '{img_path}'")
                continue
            bx1, by1, bx2, by2 = detections[0]["bbox"]

            bw = max(1, bx2 - bx1)
            bh = max(1, by2 - by1)
            aspect_ratio = bw / float(bh)
            aspect_ratios.append(aspect_ratio)

            if bx2 > bx1 and by2 > by1:
                sub_crop = crop[by1:by2, bx1:bx2]
                person_crop = sub_crop if sub_crop.size > 0 else crop
            else:
                person_crop = crop

            try:
                emb = self.verifier.extract(person_crop)
                norm = np.linalg.norm(emb)
                if norm > 1e-6:
                    emb = emb / norm
                    valid_embeddings.append(emb)
                    valid_crops.append(person_crop)

                    # Estimate view
                    view_name, angle = self.view_estimator.estimate_view_from_crop(person_crop)
                    if view_name in view_groups:
                        view_groups[view_name].append(emb)
                    else:
                        logger.warning(
                            f"View estimation failed for '{img_path}' (no confident pose "
                            f"keypoints) — falling back to 'front' view group for this sample."
                        )
                        view_groups['front'].append(emb)
            except Exception as e:
                logger.warning(f"Failed to extract embedding from '{img_path}': {e}")
                continue

        if not valid_embeddings:
            raise ValueError(f"Không build được embedding cho bất kỳ view nào — dữ liệu capture không hợp lệ for '{sanitized_name}'")

        composite_mean = np.mean(valid_embeddings, axis=0)
        c_norm = np.linalg.norm(composite_mean)
        if c_norm > 1e-6:
            composite_mean = composite_mean / c_norm

        reference_embeddings = {}
        for view_name, emb_list in view_groups.items():
            if len(emb_list) > 0:
                mean_emb = np.mean(emb_list, axis=0)
                norm = np.linalg.norm(mean_emb)
                reference_embeddings[view_name] = (mean_emb / norm) if norm > 1e-6 else mean_emb
            else:
                # Per spec: a view with zero confident samples stays unavailable (None), it does
                # NOT get silently backfilled with the composite mean. pipeline.py's runtime
                # fallback (median similarity across whatever views ARE available) exists
                # precisely to handle a missing view — substituting composite_mean here would
                # make that fallback path unreachable and silently misrepresent this view as
                # having real reference data.
                reference_embeddings[view_name] = None

        # Extract pose proportions across all valid crops
        reference_pose_proportions = self.view_estimator.extract_reference_pose_proportions(valid_crops)

        reference_aspect_ratio = float(np.median(aspect_ratios))
        output_path = registry.registry_path(sanitized_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        save_dict = {
            "embedding": composite_mean,
            "aspect_ratio": np.array(reference_aspect_ratio, dtype=np.float64),
            "created_at": np.array(datetime.now().isoformat()),
            "sample_count": np.array(len(image_paths)),
            "valid_sample_count": np.array(len(valid_embeddings))
        }
        # Only persist a per-view key when that view actually has reference data — an absent key
        # (rather than a fabricated composite-mean stand-in) is how registry.load_person()/
        # pipeline.py tell "this view has no data, use the runtime median fallback" apart from
        # "this view was genuinely captured".
        for view_name in ['front', 'right', 'back', 'left']:
            if reference_embeddings[view_name] is not None:
                save_dict[f"embedding_{view_name}"] = reference_embeddings[view_name]

        if reference_pose_proportions:
            for k in ['shoulder_hip_ratio', 'leg_torso_ratio', 'shoulder_width']:
                val = reference_pose_proportions.get(k)
                if val is not None:
                    save_dict[f"pose_{k}"] = np.array(float(val), dtype=np.float64)

        np.savez(output_path, **save_dict)

        logger.info(
            f"Successfully built embedding for '{sanitized_name}' -> '{output_path}' "
            f"(samples={len(image_paths)}, valid={len(valid_embeddings)}, "
            f"views=[front:{len(view_groups['front'])}, right:{len(view_groups['right'])}, "
            f"back:{len(view_groups['back'])}, left:{len(view_groups['left'])}], "
            f"aspect_ratio={reference_aspect_ratio:.4f}, "
            f"pose={reference_pose_proportions})"
        )
        return output_path


# Backward compatibility wrappers
TargetRegistrarGUI = RawDataCapturerGUI


class TargetRegistrar:
    """Backward compatible TargetRegistrar runner executing Phase 1 + Phase 2 sequentially."""
    def __init__(self, config: dict):
        self.config = config

    def run(self) -> str:
        capturer = RawDataCapturer(self.config)
        sanitized_name = capturer.run()

        builder = EmbeddingBuilder(self.config)
        return builder.build_embedding(sanitized_name)
