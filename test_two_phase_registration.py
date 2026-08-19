import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import cv2

from src import registry
from src.registration import EmbeddingBuilder, RawDataCapturer


class TwoPhaseRegistrationTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_registry = tempfile.mkdtemp(prefix="registry_test_")
        self.tmp_raw = tempfile.mkdtemp(prefix="raw_captures_test_")
        
        self._orig_registry = registry.REGISTRY_DIR
        self._orig_raw = registry.RAW_CAPTURES_DIR
        
        registry.REGISTRY_DIR = self.tmp_registry
        registry.RAW_CAPTURES_DIR = self.tmp_raw

        self.config = {
            "yolo_model_path": "yolo11n.pt",
            "osnet_variant": "osnet_x1_0",
            "min_build_frames": 5,
            "min_capture_frames": 5,
            "registration_min_samples": 5
        }

    def tearDown(self):
        registry.REGISTRY_DIR = self._orig_registry
        registry.RAW_CAPTURES_DIR = self._orig_raw
        shutil.rmtree(self.tmp_registry, ignore_errors=True)
        shutil.rmtree(self.tmp_raw, ignore_errors=True)

    def test_raw_capture_dir_helpers(self):
        name = "Test Person 1"
        sanitized = registry.sanitize_person_name(name)
        dir_path = registry.raw_capture_dir(name)
        self.assertTrue(dir_path.endswith(sanitized))
        self.assertFalse(registry.raw_capture_exists(name))
        os.makedirs(dir_path, exist_ok=True)
        self.assertTrue(registry.raw_capture_exists(name))

    def test_embedding_builder_insufficient_frames_raises(self):
        person_name = "Alice"
        raw_dir = registry.raw_capture_dir(person_name)
        os.makedirs(raw_dir, exist_ok=True)

        # Create only 2 dummy frame files (< min_build_frames=5)
        for i in range(2):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(raw_dir, f"frame_{i+1:04d}.jpg"), img)

        builder = EmbeddingBuilder(self.config)
        with self.assertRaises(ValueError) as ctx:
            builder.build_embedding(person_name)
        self.assertIn("Không đủ ảnh", str(ctx.exception))

    def test_embedding_builder_successful_build(self):
        person_name = "Bob"
        raw_dir = registry.raw_capture_dir(person_name)
        os.makedirs(raw_dir, exist_ok=True)

        # Create 5 dummy (solid-color, not a real person) images. Real YOLO would detect
        # nobody in these, so detector.track() is mocked to return a fixed synthetic bbox —
        # this test is about EmbeddingBuilder's own logic (view/aspect-ratio/save), not YOLO's
        # detection accuracy, and must stay deterministic without a real photo of a person.
        for i in range(5):
            img = np.full((120, 80, 3), (100, 150, 200), dtype=np.uint8)
            cv2.imwrite(os.path.join(raw_dir, f"frame_{i+1:04d}.jpg"), img)

        builder = EmbeddingBuilder(self.config)
        fake_detection = [{"track_id": 1, "bbox": (5, 5, 75, 115), "confidence": 0.9}]
        with patch.object(builder.detector, "track", return_value=fake_detection):
            out_path = builder.build_embedding(person_name)

        self.assertTrue(os.path.exists(out_path))

        # Load person via registry
        data = registry.load_person(out_path)
        self.assertIn("embedding", data)
        self.assertIn("aspect_ratio", data)
        self.assertIn("multi_views", data)
        self.assertIn("front", data["multi_views"])
        self.assertGreater(data["sample_count"], 0)

    def test_embedding_builder_skips_frames_with_no_detected_person(self):
        # Spec requirement: a raw frame where person re-detection finds nobody must be SKIPPED
        # (logged, not fatal) and must NEVER fall back to treating the whole image as the
        # person's bbox — that would silently poison the aspect ratio / embedding crop with
        # background pixels. 2 of 5 frames simulate "nobody detected"; the build must still
        # succeed using only the 3 frames where a person actually was detected.
        person_name = "Carol"
        raw_dir = registry.raw_capture_dir(person_name)
        os.makedirs(raw_dir, exist_ok=True)

        for i in range(5):
            img = np.full((120, 80, 3), (100, 150, 200), dtype=np.uint8)
            cv2.imwrite(os.path.join(raw_dir, f"frame_{i+1:04d}.jpg"), img)

        builder = EmbeddingBuilder(self.config)
        fake_detection = [{"track_id": 1, "bbox": (5, 5, 75, 115), "confidence": 0.9}]
        side_effects = [[], [], fake_detection, fake_detection, fake_detection]
        with patch.object(builder.detector, "track", side_effect=side_effects):
            out_path = builder.build_embedding(person_name)

        raw_npz = np.load(out_path, allow_pickle=False)
        self.assertEqual(int(raw_npz["sample_count"]), 5)       # total raw frames on disk
        self.assertEqual(int(raw_npz["valid_sample_count"]), 3)  # only the 3 with a person detected

    def test_embedding_builder_all_frames_undetected_raises(self):
        # If every single raw frame fails person re-detection, the build must raise rather than
        # silently saving an empty/meaningless .npz (spec: "TẤT CẢ view rỗng sau khi build →
        # raise lỗi rõ ràng, không lưu file .npz rỗng vô nghĩa").
        person_name = "Dave"
        raw_dir = registry.raw_capture_dir(person_name)
        os.makedirs(raw_dir, exist_ok=True)

        for i in range(5):
            img = np.full((120, 80, 3), (100, 150, 200), dtype=np.uint8)
            cv2.imwrite(os.path.join(raw_dir, f"frame_{i+1:04d}.jpg"), img)

        builder = EmbeddingBuilder(self.config)
        with patch.object(builder.detector, "track", return_value=[]):
            with self.assertRaises(ValueError) as ctx:
                builder.build_embedding(person_name)
        self.assertIn("Không build được embedding", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
