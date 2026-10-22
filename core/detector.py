"""
YOLOv8-based person detector module with post-processing NMS deduplication.
Prevents double-detection of a single player by applying overlap suppression.
"""

import numpy as np
import torch
from typing import List
from pathlib import Path

from core.models import Detection, BoundingBox
from utils.config import YOLO_MODEL_NAME, DEFAULT_DETECTION_CONFIDENCE, FORCE_CPU, MODELS_DIR
from utils.logging_utils import logger

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("Ultralytics package not installed.")


def _compute_iou(b1: BoundingBox, b2: BoundingBox) -> float:
    """Compute IoU between two bounding boxes."""
    ix1 = max(b1.x1, b2.x1)
    iy1 = max(b1.y1, b2.y1)
    ix2 = min(b1.x2, b2.x2)
    iy2 = min(b1.y2, b2.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = b1.area + b2.area - inter
    return inter / union if union > 0 else 0.0


def _nms_suppress_duplicates(detections: List[Detection], iou_threshold: float = 0.45) -> List[Detection]:
    """
    Remove duplicate bounding boxes using Non-Maximum Suppression.
    Keeps the detection with the highest confidence when boxes significantly overlap.
    This is the key fix for a single player being detected twice.
    """
    if len(detections) <= 1:
        return detections

    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept = []
    suppressed = set()

    for i, det_i in enumerate(sorted_dets):
        if i in suppressed:
            continue
        kept.append(det_i)
        for j, det_j in enumerate(sorted_dets):
            if j <= i or j in suppressed:
                continue
            iou = _compute_iou(det_i.bbox, det_j.bbox)
            if iou > iou_threshold:
                suppressed.add(j)

    return kept


class PlayerDetector:
    """
    Detects players in frames using YOLOv8 models.
    Filters for person class (COCO class 0) and applies NMS deduplication.
    """
    def __init__(self, confidence: float = DEFAULT_DETECTION_CONFIDENCE, model_name: str = YOLO_MODEL_NAME,
                 nms_iou_threshold: float = 0.4):
        self.confidence = confidence
        self.nms_iou_threshold = nms_iou_threshold
        self.device = "cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load YOLO model weights (auto-downloads if missing)."""
        if not ULTRALYTICS_AVAILABLE:
            logger.error("Ultralytics library unavailable.")
            return

        model_path = MODELS_DIR / self.model_name
        try:
            self.model = YOLO(str(model_path) if model_path.exists() else self.model_name)
            logger.info(f"Loaded YOLO detector ({self.model_name}) on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model {self.model_name}: {e}")
            self.model = None

    def detect(self, frame: np.ndarray, frame_idx: int = 0, timestamp: float = 0.0) -> List[Detection]:
        """
        Run object detection on a single frame, filter for persons, and apply NMS deduplication.
        Returns clean list with no duplicate detections for the same physical player.
        """
        if self.model is None:
            return []

        # Use YOLO's built-in NMS + add our own post-processing on top
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.nms_iou_threshold,  # YOLO's NMS IoU threshold
            device=self.device,
            verbose=False,
            classes=[0]  # Only persons
        )
        detections = []

        if not results or len(results) == 0:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if cls_id != 0:
                continue

            conf = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].cpu().numpy()

            bbox = BoundingBox(
                x1=float(xyxy[0]),
                y1=float(xyxy[1]),
                x2=float(xyxy[2]),
                y2=float(xyxy[3])
            )

            det = Detection(
                frame_idx=frame_idx,
                timestamp=timestamp,
                bbox=bbox,
                confidence=round(conf, 3),
                class_id=cls_id,
                class_name="person",
                player_id=None
            )
            detections.append(det)

        # Apply secondary NMS to catch any remaining overlaps YOLO missed
        deduplicated = _nms_suppress_duplicates(detections, iou_threshold=self.nms_iou_threshold)

        # Filter out very small bounding boxes (partial/ghost detections)
        min_bbox_area = (frame.shape[0] * frame.shape[1]) * 0.002  # At least 0.2% of frame
        deduplicated = [d for d in deduplicated if d.bbox.area >= min_bbox_area]

        return deduplicated
