"""
Annotated output video writer with player bounding boxes, IDs, pose skeletons, trails, and event labels.
Encodes web-compatible H.264 (libx264) MP4 video files for seamless Streamlit video playback.

FPS Fix: Each sampled/processed frame is written N times to maintain correct video duration.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from core.models import Detection, PlayerTrack, PoseLandmarks, SportEvent, VideoInfo
from utils.logging_utils import logger

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False


class AnnotatedVideoWriter:
    """
    Renders bounding boxes, IDs, pose skeletons, trails, and timestamps onto video.
    Outputs web-compatible MP4 (H.264 / yuv420p) at the ORIGINAL video FPS.
    Each sampled frame is repeated to maintain correct playback speed.
    """
    PLAYER_COLORS = [
        (0, 200, 255),   # Cyan
        (0, 255, 120),   # Green
        (255, 80, 80),   # Red
        (255, 200, 0),   # Yellow
        (200, 0, 255),   # Purple
        (255, 120, 0),   # Orange
        (0, 180, 180),   # Teal
        (180, 0, 180),   # Magenta
        (80, 255, 80),   # Lime
    ]

    # COCO 17-keypoint skeleton connections
    POSE_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),         # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Arms
        (5, 11), (6, 12), (11, 12),               # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)   # Legs
    ]

    def __init__(self, output_path: str, video_info: VideoInfo, frame_skip: int = 1):
        self.output_path = output_path
        self.video_info = video_info
        self.frame_skip = max(1, frame_skip)  # How many times to repeat each frame
        self.writer = None
        self.use_imageio = False
        self._init_writer()

    def _init_writer(self):
        fps = float(self.video_info.fps) if self.video_info.fps > 0 else 25.0

        if IMAGEIO_AVAILABLE:
            try:
                self.writer = imageio.get_writer(
                    self.output_path,
                    fps=fps,
                    codec="libx264",
                    format="FFMPEG",
                    ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "23"]
                )
                self.use_imageio = True
                logger.info(f"Initialized H.264 imageio VideoWriter @ {fps} FPS: {self.output_path}")
                return
            except Exception as e:
                logger.warning(f"imageio H.264 writer failed: {e}. Falling back to OpenCV.")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            fps,
            (self.video_info.width, self.video_info.height)
        )
        self.use_imageio = False
        logger.info(f"Initialized OpenCV VideoWriter @ {fps} FPS: {self.output_path}")

    def get_color(self, player_id: int) -> Tuple[int, int, int]:
        return self.PLAYER_COLORS[player_id % len(self.PLAYER_COLORS)]

    def annotate_frame(self, frame: np.ndarray, frame_idx: int, timestamp: float,
                       detections: List[Detection],
                       tracks: Dict[int, PlayerTrack],
                       pose_landmarks: Optional[Dict[int, PoseLandmarks]] = None,
                       events: Optional[List[SportEvent]] = None,
                       sport: str = "football") -> np.ndarray:
        """Draw annotations on video frame."""
        annotated = frame.copy()

        # 1. Semi-transparent overlay for better visibility
        overlay = annotated.copy()

        # 2. Draw motion trajectory trails
        self._draw_trajectories(overlay, tracks, frame_idx)

        # 3. Draw pose skeletons if present
        if pose_landmarks:
            for pid, pose in pose_landmarks.items():
                self._draw_pose(overlay, pose)

        # Blend overlay with transparency
        cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

        # 4. Draw detection boxes & Player ID labels (solid, after blend)
        for det in detections:
            if det.player_id is not None:
                self._draw_detection(annotated, det)

        # 5. Draw sport label + HUD bar
        self._draw_hud(annotated, timestamp, sport, len(detections))

        return annotated

    def _draw_detection(self, frame: np.ndarray, det: Detection):
        """Draw bounding box with rounded corners effect and ID badge."""
        color = self.get_color(det.player_id if det.player_id else 0)
        box = det.bbox
        x1, y1 = int(box.x1), int(box.y1)
        x2, y2 = int(box.x2), int(box.y2)

        # Thick colored bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Corner accents (thick L-shaped corners)
        clen = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        thick = 3
        for sx, sy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            cv2.line(frame, (sx, sy), (sx + dx * clen, sy), color, thick)
            cv2.line(frame, (sx, sy), (sx, sy + dy * clen), color, thick)

        # ID badge background
        label = f"Player {det.player_id}"
        conf_label = f"{int(det.confidence * 100)}%"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.45
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)

        badge_x1 = x1
        badge_y1 = max(0, y1 - th - 10)
        badge_x2 = x1 + tw + 10
        badge_y2 = y1

        # Filled badge with color
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(frame, label, (badge_x1 + 4, badge_y2 - 3), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

        # Confidence on the side
        cv2.putText(frame, conf_label, (x2 + 3, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    def _draw_trajectories(self, frame: np.ndarray, tracks: Dict[int, PlayerTrack], current_frame: int):
        """Draw fading motion trail for each tracked player."""
        for pid, track in tracks.items():
            color = self.get_color(pid)
            recent = [tp for tp in track.track_points if current_frame - 50 <= tp.frame_idx <= current_frame]
            if len(recent) < 2:
                continue
            pts = [tp.center for tp in recent]
            for i in range(1, len(pts)):
                # Fade older parts of trail
                alpha = i / len(pts)
                c = tuple(int(ch * alpha) for ch in color)
                p1 = (int(pts[i-1][0]), int(pts[i-1][1]))
                p2 = (int(pts[i][0]), int(pts[i][1]))
                cv2.line(frame, p1, p2, c, 2, cv2.LINE_AA)

    def _draw_pose(self, frame: np.ndarray, pose: PoseLandmarks):
        """Draw COCO 17-keypoint skeleton on frame."""
        if not pose.pose_detected or len(pose.landmarks) < 17:
            return

        cx, cy = pose.crop_x, pose.crop_y
        cw, ch = pose.crop_w, pose.crop_h
        if cw <= 0 or ch <= 0:
            return

        pts = {}
        for idx, lm in enumerate(pose.landmarks):
            if lm.visibility < 0.35:
                continue
            px = int(cx + lm.x * cw)
            py = int(cy + lm.y * ch)
            pts[idx] = (px, py)

        # Draw bones
        for p1_idx, p2_idx in self.POSE_CONNECTIONS:
            if p1_idx in pts and p2_idx in pts:
                cv2.line(frame, pts[p1_idx], pts[p2_idx], (50, 255, 50), 2, cv2.LINE_AA)

        # Draw joints
        for idx, pt in pts.items():
            cv2.circle(frame, pt, 4, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 4, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray, timestamp: float, sport: str, player_count: int):
        """Draw HUD overlay bar at the top of the frame."""
        h, w = frame.shape[:2]
        bar_h = 38

        # Dark top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (15, 15, 25), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Sport label left
        sport_label = f"AI Sports Analyser  |  {sport.upper()}"
        cv2.putText(frame, sport_label, (12, 24), cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 200, 255), 1, cv2.LINE_AA)

        # Timestamp center
        ts_str = f"{int(timestamp // 60):02d}:{timestamp % 60:05.2f}"
        (tw, _), _ = cv2.getTextSize(ts_str, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        cv2.putText(frame, ts_str, (w // 2 - tw // 2, 26), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Players detected right
        pd_label = f"Players: {player_count}"
        (pw, _), _ = cv2.getTextSize(pd_label, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
        cv2.putText(frame, pd_label, (w - pw - 12, 24), cv2.FONT_HERSHEY_DUPLEX, 0.45, (100, 255, 100), 1, cv2.LINE_AA)

        # Bottom thin colored bar
        cv2.line(frame, (0, bar_h), (w, bar_h), (0, 200, 255), 1)

    def write_frame(self, frame: np.ndarray, repeat: int = 1):
        """Write a frame N times to maintain correct playback speed when using frame skipping."""
        if not self.writer:
            return

        for _ in range(max(1, repeat)):
            if self.use_imageio:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.writer.append_data(rgb_frame)
            else:
                self.writer.write(frame)

    def release(self):
        if self.writer:
            try:
                if self.use_imageio:
                    self.writer.close()
                else:
                    self.writer.release()
            except Exception as e:
                logger.warning(f"Error releasing VideoWriter: {e}")
            self.writer = None
