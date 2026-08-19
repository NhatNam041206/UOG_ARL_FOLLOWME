import os
import shutil
import tempfile
import unittest
import numpy as np
import yaml

from src.view_estimator import ViewEstimator
from src.pipeline import FollowPipeline
from src import registry


class ViewEstimatorTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="view_estimator_test_")
        self.estimator = ViewEstimator(pose_model=None)  # unit test in offline mode

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_classify_view(self):
        # 0 deg -> front
        self.assertEqual(self.estimator.classify_view(0), 'front')
        self.assertEqual(self.estimator.classify_view(350), 'front')
        self.assertEqual(self.estimator.classify_view(10), 'front')

        # 90 deg -> right
        self.assertEqual(self.estimator.classify_view(90), 'right')
        self.assertEqual(self.estimator.classify_view(45), 'right')
        self.assertEqual(self.estimator.classify_view(130), 'right')

        # 180 deg -> back
        self.assertEqual(self.estimator.classify_view(180), 'back')
        self.assertEqual(self.estimator.classify_view(135), 'back')
        self.assertEqual(self.estimator.classify_view(220), 'back')

        # 270 deg -> left
        self.assertEqual(self.estimator.classify_view(270), 'left')
        self.assertEqual(self.estimator.classify_view(225), 'left')
        self.assertEqual(self.estimator.classify_view(310), 'left')

        # None angle
        self.assertIsNone(self.estimator.classify_view(None))

    def test_extract_pose_proportions_from_keypoints(self):
        # Create mock 17 COCO keypoints
        # 5: left_shoulder (x=60, y=30, conf=0.9)
        # 6: right_shoulder (x=40, y=30, conf=0.9) -> width = 20
        # 11: left_hip (x=55, y=60, conf=0.9)
        # 12: right_hip (x=45, y=60, conf=0.9) -> width = 10
        # 13: left_knee (x=55, y=80)
        # 15: left_ankle (x=55, y=100) -> leg = (100-80)*2 = 40
        # torso = mid shoulder (50, 30) to mid hip (50, 60) = 30
        kpts = np.zeros((17, 3), dtype=np.float32)
        kpts[5] = [60, 30, 0.9]
        kpts[6] = [40, 30, 0.9]
        kpts[11] = [55, 60, 0.9]
        kpts[12] = [45, 60, 0.9]
        kpts[13] = [55, 80, 0.9]
        kpts[15] = [55, 100, 0.9]

        props = ViewEstimator.extract_pose_proportions_from_keypoints(kpts)
        self.assertAlmostEqual(props['shoulder_width'], 20.0, places=2)
        self.assertAlmostEqual(props['shoulder_hip_ratio'], 20.0 / 10.0, places=2)
        self.assertAlmostEqual(props['leg_torso_ratio'], 40.0 / 30.0, places=2)

    def test_compute_pose_similarity(self):
        ref_props = {
            'shoulder_hip_ratio': 1.5,
            'leg_torso_ratio': 1.2,
            'shoulder_width': 25.0
        }

        # Identical candidate
        cand_same = {
            'shoulder_hip_ratio': 1.5,
            'leg_torso_ratio': 1.2,
            'shoulder_width': 25.0
        }
        sim_same = ViewEstimator.compute_pose_similarity(cand_same, ref_props)
        self.assertAlmostEqual(sim_same, 1.0, places=3)

        # Different body shape
        cand_diff = {
            'shoulder_hip_ratio': 2.5,
            'leg_torso_ratio': 0.6,
            'shoulder_width': 40.0
        }
        sim_diff = ViewEstimator.compute_pose_similarity(cand_diff, ref_props)
        self.assertLess(sim_diff, 0.7)
        self.assertGreater(sim_diff, 0.0)

        # Missing props returns 0.5 neutral
        self.assertEqual(ViewEstimator.compute_pose_similarity(None, ref_props), 0.5)
        self.assertEqual(ViewEstimator.compute_pose_similarity(cand_same, None), 0.5)

    def test_multi_view_pipeline_verification(self):
        # Create multi-view reference person .npz
        emb_front = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb_right = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        emb_back = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        emb_left = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        multi_views = {
            'front': emb_front,
            'right': emb_right,
            'back': emb_back,
            'left': emb_left,
        }
        pose_props = {
            'shoulder_hip_ratio': 1.5,
            'leg_torso_ratio': 1.2,
            'shoulder_width': 25.0
        }

        registry.REGISTRY_DIR = self.tmpdir
        npz_path = registry.save_person(
            name="multiview_person",
            embedding=emb_front,
            aspect_ratio=1.0,
            sample_count=20,
            multi_views=multi_views,
            pose_proportions=pose_props
        )

        loaded = registry.load_person(npz_path)
        self.assertIn("multi_views", loaded)
        self.assertIn("front", loaded["multi_views"])
        self.assertIn("right", loaded["multi_views"])
        self.assertIn("pose_proportions", loaded)
        self.assertAlmostEqual(loaded["pose_proportions"]["shoulder_hip_ratio"], 1.5, places=2)


if __name__ == "__main__":
    unittest.main()
