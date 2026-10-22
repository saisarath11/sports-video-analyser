"""
Video loader, frame extraction, metadata detection, and intelligent frame sampling.
"""

import cv2
import numpy as np
from typing import Generator, Tuple, Optional
from core.models import VideoInfo
from utils.config import PROCESSING_MODES
from utils.file_utils import get_video_info
from utils.logging_utils import logger


class VideoProcessor:
    """
    Handles video opening, metadata extraction, frame resizing,
    and intelligent frame sampling based on selected processing mode.
    """
    def __init__(self, filepath: str, mode: str = "balanced"):
        self.filepath = filepath
        self.mode = mode.lower()
        self.config = PROCESSING_MODES.get(self.mode, PROCESSING_MODES["balanced"])
        self.cap = None
        self.video_info: Optional[VideoInfo] = None
        
    def load(self) -> VideoInfo:
        """Open video file and extract VideoInfo metadata."""
        self.video_info = get_video_info(self.filepath)
        if not self.video_info.is_valid:
            raise ValueError(f"Invalid video file: {self.video_info.error_message}")
            
        self.cap = cv2.VideoCapture(self.filepath)
        logger.info(f"Loaded video: {self.video_info.filename} ({self.video_info.width}x{self.video_info.height} @ {self.video_info.fps} FPS)")
        return self.video_info

    def get_frame_generator(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Yields sampled frames as (frame_idx, timestamp_seconds, frame_bgr).
        Applies frame skipping and resolution resizing specified by processing mode.
        """
        if self.cap is None or not self.cap.isOpened():
            self.load()
            
        frame_skip = self.config["frame_skip"]
        resize_width = self.config["resize_width"]
        max_frames = self.config.get("max_frames", 5000)
        
        frame_idx = 0
        processed_count = 0
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or processed_count >= max_frames:
                break
                
            if frame_idx % frame_skip == 0:
                timestamp = round(frame_idx / self.video_info.fps, 3) if self.video_info.fps > 0 else 0.0
                
                # Resize if necessary
                h, w = frame.shape[:2]
                if resize_width and w > resize_width:
                    aspect_ratio = h / w
                    new_w = resize_width
                    new_h = int(new_w * aspect_ratio)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                yield frame_idx, timestamp, frame
                processed_count += 1
                
            frame_idx += 1
            
        self.release()

    def extract_frame_at(self, timestamp: float) -> Optional[np.ndarray]:
        """Extract a single frame at given timestamp in seconds."""
        if not self.video_info:
            self.load()
            
        cap = cv2.VideoCapture(self.filepath)
        target_frame = int(timestamp * self.video_info.fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    def get_total_frames_to_process(self) -> int:
        """Calculate total number of frames that will be processed under current mode."""
        if not self.video_info:
            self.load()
        frame_skip = self.config["frame_skip"]
        total = self.video_info.frame_count // frame_skip
        max_frames = self.config.get("max_frames", 5000)
        return min(total, max_frames)

    def release(self):
        """Release VideoCapture resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
