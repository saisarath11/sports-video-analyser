"""
Human body posture analysis and pose landmark estimation using YOLOv8 Pose (yolov8n-pose.pt).
Provides 17 COCO keypoints + joint angle computation + posture observations.
"""

import cv2
import numpy as np
import torch
from typing import List, Optional, Dict, Tuple
from core.models import Landmark, PoseLandmarks, BoundingBox
from utils.config import YOLO_POSE_MODEL_NAME, MODELS_DIR, FORCE_CPU
from utils.logging_utils import logger

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class PoseEstimator:
    """
    Extracts 17 keypoints for detected players using YOLOv8 Pose model.
    Computes athletic joint angles and posture characteristics.
    """
    COCO_KEYPOINT_INDICES = {
        "nose": 0,
        "left_eye": 1, "right_eye": 2,
        "left_ear": 3, "right_ear": 4,
        "left_shoulder": 5, "right_shoulder": 6,
        "left_elbow": 7, "right_elbow": 8,
        "left_wrist": 9, "right_wrist": 10,
        "left_hip": 11, "right_hip": 12,
        "left_knee": 13, "right_knee": 14,
        "left_ankle": 15, "right_ankle": 16
    }

    def __init__(self, model_name: str = YOLO_POSE_MODEL_NAME):
        self.device = "cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        if not ULTRALYTICS_AVAILABLE:
            logger.error("Ultralytics library unavailable for PoseEstimator.")
            return

        model_path = MODELS_DIR / self.model_name
        try:
            self.model = YOLO(str(model_path) if model_path.exists() else self.model_name)
            logger.info(f"Initialized YOLO Pose Estimator ({self.model_name}) on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO Pose model {self.model_name}: {e}")
            self.model = None

    def estimate(self, frame: np.ndarray, bbox: BoundingBox, player_id: int,
                 frame_idx: int, timestamp: float) -> Optional[PoseLandmarks]:
        """Estimate 17 pose landmarks on player crop using YOLO Pose."""
        if not self.model:
            return None

        # Crop player region with padding
        crop, cx, cy, cw, ch = self._crop_player(frame, bbox)
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return None

        results = self.model(crop, conf=0.25, device=self.device, verbose=False)

        if not results or len(results) == 0 or results[0].keypoints is None or len(results[0].keypoints) == 0:
            return PoseLandmarks(
                player_id=player_id, frame_idx=frame_idx, timestamp=timestamp,
                pose_detected=False, crop_x=cx, crop_y=cy, crop_w=cw, crop_h=ch
            )

        keypoints_data = results[0].keypoints.data[0].cpu().numpy()  # shape (17, 3): x, y, conf
        
        landmarks = []
        for i in range(len(keypoints_data)):
            kx, ky, kconf = keypoints_data[i]
            # Normalize relative to crop width and height
            norm_x = float(kx / max(1.0, cw))
            norm_y = float(ky / max(1.0, ch))
            landmarks.append(Landmark(x=round(norm_x, 4), y=round(norm_y, 4), z=0.0, visibility=round(float(kconf), 3)))

        joint_angles = self._compute_joint_angles(landmarks)
        posture_notes = self._generate_posture_notes(joint_angles)

        return PoseLandmarks(
            player_id=player_id,
            frame_idx=frame_idx,
            timestamp=timestamp,
            landmarks=landmarks,
            joint_angles=joint_angles,
            posture_notes=posture_notes,
            pose_detected=True,
            crop_x=cx,
            crop_y=cy,
            crop_w=cw,
            crop_h=ch
        )

    def _crop_player(self, frame: np.ndarray, bbox: BoundingBox, padding: float = 0.1) -> Tuple[np.ndarray, float, float, float, float]:
        """Crop player bounding box with percentage padding."""
        h, w = frame.shape[:2]
        pad_x = int(bbox.width * padding)
        pad_y = int(bbox.height * padding)
        x1 = max(0, int(bbox.x1) - pad_x)
        y1 = max(0, int(bbox.y1) - pad_y)
        x2 = min(w, int(bbox.x2) + pad_x)
        y2 = min(h, int(bbox.y2) + pad_y)
        crop_w = max(1, x2 - x1)
        crop_h = max(1, y2 - y1)
        return frame[y1:y2, x1:x2], float(x1), float(y1), float(crop_w), float(crop_h)

    def _compute_angle(self, p1: Landmark, p2: Landmark, p3: Landmark) -> float:
        """Compute interior angle at p2 formed by p1-p2-p3 in degrees."""
        if p1.visibility < 0.2 or p2.visibility < 0.2 or p3.visibility < 0.2:
            return 180.0

        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 180.0

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        return round(float(angle), 1)

    def _compute_joint_angles(self, landmarks: List[Landmark]) -> Dict[str, float]:
        """Compute key athletic joint angles from 17 COCO keypoints."""
        if len(landmarks) < 17:
            return {}

        angles = {}
        idx = self.COCO_KEYPOINT_INDICES

        # Left Knee angle (Hip - Knee - Ankle)
        angles["left_knee_angle"] = self._compute_angle(landmarks[idx["left_hip"]], landmarks[idx["left_knee"]], landmarks[idx["left_ankle"]])
        # Right Knee angle
        angles["right_knee_angle"] = self._compute_angle(landmarks[idx["right_hip"]], landmarks[idx["right_knee"]], landmarks[idx["right_ankle"]])
        # Left Elbow angle (Shoulder - Elbow - Wrist)
        angles["left_elbow_angle"] = self._compute_angle(landmarks[idx["left_shoulder"]], landmarks[idx["left_elbow"]], landmarks[idx["left_wrist"]])
        # Right Elbow angle
        angles["right_elbow_angle"] = self._compute_angle(landmarks[idx["right_shoulder"]], landmarks[idx["right_elbow"]], landmarks[idx["right_wrist"]])
        # Left Hip angle (Shoulder - Hip - Knee)
        angles["left_hip_angle"] = self._compute_angle(landmarks[idx["left_shoulder"]], landmarks[idx["left_hip"]], landmarks[idx["left_knee"]])
        # Right Hip angle
        angles["right_hip_angle"] = self._compute_angle(landmarks[idx["right_shoulder"]], landmarks[idx["right_hip"]], landmarks[idx["right_knee"]])

        # Trunk lean angle relative to vertical axis
        sh_mid = ((landmarks[idx["left_shoulder"]].x + landmarks[idx["right_shoulder"]].x) / 2,
                  (landmarks[idx["left_shoulder"]].y + landmarks[idx["right_shoulder"]].y) / 2)
        hip_mid = ((landmarks[idx["left_hip"]].x + landmarks[idx["right_hip"]].x) / 2,
                   (landmarks[idx["left_hip"]].y + landmarks[idx["right_hip"]].y) / 2)
        
        dx = sh_mid[0] - hip_mid[0]
        dy = sh_mid[1] - hip_mid[1]
        lean = np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6))
        angles["trunk_lean_angle"] = round(float(lean), 1)

        return angles

    def _generate_posture_notes(self, angles: Dict[str, float]) -> List[str]:
        """Generate human-readable athletic stance observations."""
        notes = []
        if not angles:
            return notes

        avg_knee = (angles.get("left_knee_angle", 180) + angles.get("right_knee_angle", 180)) / 2
        if 100 <= avg_knee <= 155:
            notes.append("Knees flexed in athletic ready stance.")
        elif avg_knee > 165:
            notes.append("Upright leg posture with high extension.")

        trunk_lean = angles.get("trunk_lean_angle", 0)
        if trunk_lean > 15:
            notes.append("Forward trunk lean detected (active drive posture).")

        return notes

    def close(self):
        """Release resources."""
        self.model = None
