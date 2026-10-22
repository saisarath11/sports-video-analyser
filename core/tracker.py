"""
Multi-object player tracker maintaining persistent IDs across frames using IoU and spatial tracking.
"""

import numpy as np
from typing import List, Dict, Set, Optional
from core.models import Detection, PlayerTrack, TrackPoint, BoundingBox
from utils.config import MIN_TRACK_FRAMES
from utils.logging_utils import logger


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box1_area = box1.area
    box2_area = box2.area

    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def compute_center_distance(box1: BoundingBox, box2: BoundingBox) -> float:
    """Compute Euclidean distance between centers of two bounding boxes."""
    c1 = box1.center
    c2 = box2.center
    return float(np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2))


class PlayerTracker:
    """
    Tracks player detections across consecutive frames.
    Extracts persistent player IDs using IoU matching and tracker results.
    """
    def __init__(self, max_missing_frames: int = 30, iou_threshold: float = 0.2):
        self.tracks: Dict[int, PlayerTrack] = {}  # player_id -> PlayerTrack
        self._active_ids: Set[int] = set()
        self.next_id = 1
        self.max_missing_frames = max_missing_frames
        self.iou_threshold = iou_threshold

    def update_from_yolo_results(self, yolo_results, frame_idx: int, timestamp: float) -> List[Detection]:
        """
        Process Ultralytics tracking/detection results.
        Extract bounding boxes and update tracks using IoU matching and ID persistence.
        """
        detections = []
        if yolo_results is None or len(yolo_results) == 0:
            return detections

        result = yolo_results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes
        has_ids = boxes.id is not None

        # Extract current frame detections
        frame_dets: List[Detection] = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if cls_id != 0:  # Only person class
                continue

            conf = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].cpu().numpy()

            raw_id = int(boxes.id[i].item()) if has_ids else None

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
                player_id=raw_id
            )
            frame_dets.append(det)

        return self.update_from_detections(frame_dets, frame_idx, timestamp)

    def update_from_detections(self, detections: List[Detection], frame_idx: int, timestamp: float) -> List[Detection]:
        """
        Match current frame detections to active tracks using IoU and spatial proximity.
        """
        if not detections:
            return []

        # Find active tracks from recent frames
        recent_tracks = {
            tid: t for tid, t in self.tracks.items()
            if t.track_points and (frame_idx - t.track_points[-1].frame_idx) <= self.max_missing_frames
        }

        assigned_track_ids = set()
        assigned_detections = set()

        # If detections already have IDs from model.track(), use them to maintain consistency
        for i, det in enumerate(detections):
            if det.player_id is not None:
                track_id = det.player_id
                assigned_detections.add(i)
                assigned_track_ids.add(track_id)
                self._add_point_to_track(track_id, det, frame_idx, timestamp)

        # Match remaining unassigned detections using IoU to active tracks
        unassigned_det_indices = [i for i in range(len(detections)) if i not in assigned_detections]
        unassigned_track_ids = [tid for tid in recent_tracks.keys() if tid not in assigned_track_ids]

        if unassigned_det_indices and unassigned_track_ids:
            # Build IoU matrix
            iou_matrix = np.zeros((len(unassigned_det_indices), len(unassigned_track_ids)))
            for d_idx, det_i in enumerate(unassigned_det_indices):
                det_box = detections[det_i].bbox
                for t_idx, tid in enumerate(unassigned_track_ids):
                    last_box = recent_tracks[tid].track_points[-1].bbox
                    iou = compute_iou(det_box, last_box)
                    dist = compute_center_distance(det_box, last_box)
                    # Normalize score: IoU + proximity weight
                    score = iou + max(0.0, 1.0 - (dist / 200.0)) * 0.3
                    iou_matrix[d_idx, t_idx] = score

            # Greedy matching
            while True:
                if iou_matrix.size == 0 or np.max(iou_matrix) < self.iou_threshold:
                    break
                max_pos = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)
                d_idx, t_idx = max_pos[0], max_pos[1]
                
                det_i = unassigned_det_indices[d_idx]
                tid = unassigned_track_ids[t_idx]

                detections[det_i].player_id = tid
                self._add_point_to_track(tid, detections[det_i], frame_idx, timestamp)

                assigned_detections.add(det_i)
                assigned_track_ids.add(tid)

                iou_matrix[d_idx, :] = -1
                iou_matrix[:, t_idx] = -1

        # Create new track IDs for remaining unmatched detections
        for i, det in enumerate(detections):
            if det.player_id is None:
                new_id = self.next_id
                self.next_id += 1
                det.player_id = new_id
                self._add_point_to_track(new_id, det, frame_idx, timestamp)

        return detections

    def _add_point_to_track(self, track_id: int, det: Detection, frame_idx: int, timestamp: float):
        if track_id not in self.tracks:
            self.tracks[track_id] = PlayerTrack(track_id=track_id)

        tp = TrackPoint(
            frame_idx=frame_idx,
            timestamp=timestamp,
            bbox=det.bbox,
            center=det.bbox.center
        )
        self.tracks[track_id].track_points.append(tp)
        self._active_ids.add(track_id)

    def get_valid_tracks(self) -> Dict[int, PlayerTrack]:
        """Return tracks with at least MIN_TRACK_FRAMES recorded points, or all tracks if none meet threshold."""
        valid = {tid: t for tid, t in self.tracks.items() if t.frame_count >= MIN_TRACK_FRAMES}
        return valid if valid else self.tracks

    def get_active_player_ids(self) -> List[int]:
        """Get sorted list of all active tracked player IDs."""
        return sorted(self._active_ids)

    def reset(self):
        """Reset internal tracker state."""
        self.tracks.clear()
        self._active_ids.clear()
        self.next_id = 1
