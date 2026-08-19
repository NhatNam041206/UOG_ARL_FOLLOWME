import os
import logging
from typing import Optional, Tuple, Dict, Any, List
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ViewEstimator:
    """
    Body rotation view estimator & pose proportions gate.
    - Estimates body rotation angle (0-360°) from 17 COCO pose keypoints.
    - Classifies body orientation into 4 view groups: 'front', 'right', 'back', 'left'.
    - Extracts body proportions (shoulder/hip ratio, leg/torso ratio, shoulder width).
    - Computes pose similarity between candidate detections and reference registration.
    """
    def __init__(self, pose_model: Any = "yolo11n-pose.pt"):
        self.model = None
        self.is_ready = False

        if pose_model is not None:
            self._init_model(pose_model)

    def _init_model(self, pose_model: Any) -> None:
        """Initialize pose model (e.g. YOLO pose model from ultralytics or custom callable)."""
        if callable(pose_model) and not isinstance(pose_model, str):
            self.model = pose_model
            self.is_ready = True
            return

        try:
            from ultralytics import YOLO
            if isinstance(pose_model, str):
                self.model = YOLO(pose_model)
            else:
                self.model = pose_model
            self.is_ready = True
            logger.info(f"ViewEstimator initialized successfully with pose model: {pose_model}")
        except Exception as e:
            logger.warning(f"Failed to load pose model '{pose_model}': {e}. ViewEstimator will use fallback mode.")
            self.model = None
            self.is_ready = False

    def extract_keypoints(self, image_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 17 COCO keypoints from person image crop.
        Returns:
            np.ndarray of shape (17, 3) where columns are [x, y, confidence], or None if failed.
        """
        if not self.is_ready or self.model is None or image_crop is None or image_crop.size == 0:
            return None

        if image_crop.shape[0] < 15 or image_crop.shape[1] < 15:
            return None

        try:
            # If model is a callable (like MoveNet function or lambda)
            if callable(self.model) and not hasattr(self.model, "predict"):
                res = self.model(image_crop)
                if isinstance(res, np.ndarray):
                    if res.ndim == 3:
                        res = res[0]
                    return res
                return np.asarray(res)

            # Ultralytics YOLO pose
            results = self.model(image_crop, verbose=False)
            if not results or len(results) == 0:
                return None

            res0 = results[0]
            if res0.keypoints is None or res0.keypoints.data is None:
                return None

            kpts_data = res0.keypoints.data
            if kpts_data.shape[0] == 0:
                return None

            # Pick highest confidence / largest detection keypoints
            kpts = kpts_data[0].cpu().numpy()  # shape (17, 3)
            return kpts

        except Exception as e:
            logger.debug(f"Keypoint extraction failed on crop: {e}")
            return None

    def estimate_angle_from_keypoints(self, keypoints: Optional[np.ndarray]) -> Optional[float]:
        """
        Estimate body rotation angle (0-360°) from 17 COCO keypoints.
        0°: Front, 90°: Right, 180°: Back, 270°: Left.
        """
        if keypoints is None or len(keypoints) < 17:
            return None

        # Extract joints
        # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
        # 5: left_shoulder, 6: right_shoulder, 11: left_hip, 12: right_hip
        s_left = keypoints[5]
        s_right = keypoints[6]
        h_left = keypoints[11]
        h_right = keypoints[12]

        conf_thresh = 0.2
        valid_shoulders = (s_left[2] >= conf_thresh and s_right[2] >= conf_thresh)
        valid_hips = (h_left[2] >= conf_thresh and h_right[2] >= conf_thresh)

        if not valid_shoulders and not valid_hips:
            return None

        # Face confidence to distinguish front vs back
        nose = keypoints[0]
        l_eye = keypoints[1]
        r_eye = keypoints[2]
        l_ear = keypoints[3]
        r_ear = keypoints[4]

        face_conf = np.mean([nose[2], l_eye[2], r_eye[2]])
        ear_conf = max(l_ear[2], r_ear[2])

        # Compute shoulder midpoint and hip midpoint if available
        if valid_shoulders and valid_hips:
            s_mid = (s_left[:2] + s_right[:2]) / 2.0
            h_mid = (h_left[:2] + h_right[:2]) / 2.0
            torso_vec = s_mid - h_mid
            shoulder_vec = s_left[:2] - s_right[:2]  # in image coords, left shoulder is typically to the right
        elif valid_shoulders:
            shoulder_vec = s_left[:2] - s_right[:2]
        else:
            shoulder_vec = h_left[:2] - h_right[:2]

        # In image coordinates, person facing FRONT:
        # s_left.x > s_right.x (person's left shoulder is on camera right)
        # face (nose/eyes) has high confidence (face_conf >= 0.3)
        shoulder_dx = shoulder_vec[0]
        shoulder_width = np.linalg.norm(shoulder_vec)

        # Determine side or front/back orientation
        if face_conf >= 0.3:
            # Facing frontwards (front / slight side)
            if shoulder_dx > 0.3 * shoulder_width:
                angle = 0.0  # Front
            elif l_ear[2] > r_ear[2] + 0.2:
                angle = 270.0  # Turning Left
            elif r_ear[2] > l_ear[2] + 0.2:
                angle = 90.0   # Turning Right
            else:
                angle = 0.0
        else:
            # Facing backwards or side with back visible
            if shoulder_dx < -0.3 * shoulder_width:
                angle = 180.0  # Back
            elif l_ear[2] > r_ear[2] + 0.2:
                angle = 270.0  # Left
            elif r_ear[2] > l_ear[2] + 0.2:
                angle = 90.0   # Right
            else:
                angle = 180.0  # Back

        return float(angle)

    def classify_view(self, angle: Optional[float]) -> Optional[str]:
        """
        Classify rotation angle into 4 view groups: 'front', 'right', 'back', 'left'.
        """
        if angle is None:
            return None

        angle = (float(angle) % 360.0 + 360.0) % 360.0

        if angle < 45.0 or angle >= 315.0:
            return 'front'
        elif 45.0 <= angle < 135.0:
            return 'right'
        elif 135.0 <= angle < 225.0:
            return 'back'
        else:  # 225.0 <= angle < 315.0
            return 'left'

    def estimate_view_from_crop(self, crop: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """
        Estimate body orientation view and angle directly from person crop.
        Returns:
            (view_name, angle) where view_name is in ['front', 'right', 'back', 'left'] or None.
        """
        keypoints = self.extract_keypoints(crop)
        if keypoints is None:
            return None, None

        angle = self.estimate_angle_from_keypoints(keypoints)
        view = self.classify_view(angle)
        return view, angle

    def estimate_view(self, frame: np.ndarray, crop_bbox: Tuple[int, int, int, int]) -> Tuple[Optional[str], Optional[float]]:
        """
        Extract pose from frame bbox and estimate view.
        """
        x1, y1, x2, y2 = crop_bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None, None
        crop = frame[y1:y2, x1:x2]
        return self.estimate_view_from_crop(crop)

    @staticmethod
    def extract_pose_proportions_from_keypoints(keypoints: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
        """
        Calculate body proportion ratios from 17 COCO keypoints:
        - shoulder_hip_ratio: shoulder_width / hip_width
        - leg_torso_ratio: leg_length / torso_length
        - shoulder_width: Euclidean distance between shoulders
        """
        default_result = {
            'shoulder_hip_ratio': None,
            'leg_torso_ratio': None,
            'shoulder_width': None,
        }

        if keypoints is None or len(keypoints) < 17:
            return default_result

        s_left = keypoints[5][:2]
        s_right = keypoints[6][:2]
        h_left = keypoints[11][:2]
        h_right = keypoints[12][:2]
        knee_left = keypoints[13][:2]
        knee_right = keypoints[14][:2]
        ankle_left = keypoints[15][:2]
        ankle_right = keypoints[16][:2]

        shoulder_width = float(np.linalg.norm(s_right - s_left))
        hip_width = float(np.linalg.norm(h_right - h_left))

        s_mid = (s_left + s_right) / 2.0
        h_mid = (h_left + h_right) / 2.0
        torso_length = float(np.linalg.norm(h_mid - s_mid))

        leg_l = np.linalg.norm(ankle_left - knee_left) * 2.0
        leg_r = np.linalg.norm(ankle_right - knee_right) * 2.0
        leg_length = float(max(leg_l, leg_r))

        sh_ratio = (shoulder_width / hip_width) if hip_width > 1e-3 else 1.0
        lt_ratio = (leg_length / torso_length) if torso_length > 1e-3 else 1.0

        return {
            'shoulder_hip_ratio': float(sh_ratio),
            'leg_torso_ratio': float(lt_ratio),
            'shoulder_width': float(shoulder_width),
        }

    def extract_pose_proportions_from_crop(self, crop: np.ndarray) -> Dict[str, Optional[float]]:
        """Extract pose proportions directly from a crop."""
        keypoints = self.extract_keypoints(crop)
        return self.extract_pose_proportions_from_keypoints(keypoints)

    def extract_reference_pose_proportions(self, crops: List[np.ndarray]) -> Dict[str, Optional[float]]:
        """
        Extract reference body proportions averaged (median) across all valid registration crops.
        """
        ratios: Dict[str, List[float]] = {
            'shoulder_hip_ratio': [],
            'leg_torso_ratio': [],
            'shoulder_width': [],
        }

        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            prop = self.extract_pose_proportions_from_crop(crop)
            for k in ratios:
                val = prop.get(k)
                if val is not None and not np.isnan(val) and val > 0:
                    ratios[k].append(val)

        return {
            'shoulder_hip_ratio': float(np.median(ratios['shoulder_hip_ratio'])) if ratios['shoulder_hip_ratio'] else None,
            'leg_torso_ratio': float(np.median(ratios['leg_torso_ratio'])) if ratios['leg_torso_ratio'] else None,
            'shoulder_width': float(np.median(ratios['shoulder_width'])) if ratios['shoulder_width'] else None,
        }

    @staticmethod
    def compute_pose_similarity(candidate_ratios: Optional[Dict[str, Any]], reference_ratios: Optional[Dict[str, Any]]) -> float:
        """
        Compare candidate pose proportions with reference proportions.
        Returns:
            Similarity score in range [0.0, 1.0]. Returns 0.5 (neutral) if either is missing/None.
        """
        if not reference_ratios or not candidate_ratios:
            return 0.5

        diffs = []
        for key in ['shoulder_hip_ratio', 'leg_torso_ratio']:
            ref_val = reference_ratios.get(key)
            cand_val = candidate_ratios.get(key)
            if ref_val is not None and cand_val is not None:
                try:
                    ref_f = float(ref_val)
                    cand_f = float(cand_val)
                    if ref_f > 1e-4 and cand_f > 1e-4:
                        ratio = cand_f / ref_f
                        ratio = max(ratio, 1.0 / ratio)
                        diff = 1.0 / (1.0 + abs(ratio - 1.0))
                        diffs.append(diff)
                except (ValueError, TypeError, ZeroDivisionError):
                    continue

        return float(np.mean(diffs)) if diffs else 0.5
