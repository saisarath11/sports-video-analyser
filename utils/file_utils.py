"""
File management, video validation, metadata extraction, and session cleanup.
"""

import os
import uuid
import shutil
import cv2
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from utils.config import ALLOWED_VIDEO_FORMATS, MAX_UPLOAD_SIZE_MB
from core.models import VideoInfo
from utils.logging_utils import logger


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return str(uuid.uuid4())


def get_file_size_mb(filepath: str) -> float:
    """Return file size in Megabytes."""
    try:
        return round(os.path.getsize(filepath) / (1024 * 1024), 2)
    except Exception:
        return 0.0


def validate_video_file(filepath: str) -> Tuple[bool, str]:
    """
    Validate uploaded video format, file size, and playability.
    Returns (is_valid, error_message).
    """
    path = Path(filepath)
    if not path.exists():
        return False, "File does not exist."
    
    if path.suffix.lower() not in ALLOWED_VIDEO_FORMATS:
        return False, f"Unsupported format '{path.suffix}'. Supported: {', '.join(ALLOWED_VIDEO_FORMATS)}"
    
    size_mb = get_file_size_mb(filepath)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        return False, f"File size ({size_mb} MB) exceeds maximum allowed size of {MAX_UPLOAD_SIZE_MB} MB."
    
    # Check if OpenCV can open the file
    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        cap.release()
        return False, "Unable to open video stream. File may be corrupted or unreadable."
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if frame_count <= 0 or fps <= 0:
        return False, "Invalid video metadata (zero frames or FPS)."
    
    return True, ""


def get_video_info(filepath: str) -> VideoInfo:
    """Extract full VideoInfo metadata object using OpenCV."""
    path = Path(filepath)
    is_valid, err_msg = validate_video_file(filepath)
    
    if not is_valid:
        return VideoInfo(
            filename=path.name,
            filepath=str(filepath),
            duration=0.0,
            fps=0.0,
            width=0,
            height=0,
            frame_count=0,
            file_size_mb=get_file_size_mb(filepath),
            format=path.suffix.lower(),
            is_valid=False,
            error_message=err_msg
        )
    
    cap = cv2.VideoCapture(str(filepath))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    duration = round(frame_count / fps, 2) if fps > 0 else 0.0
    
    return VideoInfo(
        filename=path.name,
        filepath=str(filepath),
        duration=duration,
        fps=round(fps, 2),
        width=width,
        height=height,
        frame_count=frame_count,
        file_size_mb=get_file_size_mb(filepath),
        format=path.suffix.lower(),
        is_valid=True,
        error_message=""
    )


def save_uploaded_file(uploaded_file, dest_dir: str) -> str:
    """Save Streamlit UploadedFile to specified destination directory."""
    os.makedirs(dest_dir, exist_ok=True)
    file_path = os.path.join(dest_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    logger.info(f"Saved uploaded file to {file_path}")
    return file_path


def cleanup_temp_files(temp_dir: str, session_id: Optional[str] = None):
    """Clean up temporary files in directory or for specific session."""
    try:
        temp_path = Path(temp_dir)
        if not temp_path.exists():
            return
            
        for item in temp_path.iterdir():
            if session_id:
                if session_id in item.name:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
            else:
                if item.is_file() and not item.name.startswith("."):
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        logger.info(f"Cleaned up temporary files in {temp_dir}")
    except Exception as e:
        logger.warning(f"Error cleaning up temp files: {e}")
