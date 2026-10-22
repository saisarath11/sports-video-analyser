"""
Application configuration and settings for the AI Sports Video Analyser.
All tuneable parameters are centralised here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_INPUT_DIR = BASE_DIR / os.getenv("DATA_INPUT_DIR", "data/input")
DATA_OUTPUT_DIR = BASE_DIR / os.getenv("DATA_OUTPUT_DIR", "data/output")
DATA_TEMP_DIR = BASE_DIR / os.getenv("DATA_TEMP_DIR", "data/temp")
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
for _d in [DATA_INPUT_DIR, DATA_OUTPUT_DIR, DATA_TEMP_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ============================================================
# App Settings
# ============================================================
APP_TITLE = "AI Sports Video Analyser"
APP_ICON = "🏅"
APP_VERSION = "1.0.0"
APP_DEBUG = os.getenv("APP_DEBUG", "false").lower() == "true"

MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
ALLOWED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]
ALLOWED_MIME_TYPES = ["video/mp4", "video/x-msvideo", "video/quicktime", "video/x-matroska"]

# ============================================================
# AI / Model Settings
# ============================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# YOLO model — downloads automatically on first run
YOLO_MODEL_NAME = "yolov8n.pt"          # nano: fast, good accuracy
YOLO_POSE_MODEL_NAME = "yolov8n-pose.pt"  # pose variant

FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() == "true"

# ============================================================
# Processing Modes
# ============================================================
PROCESSING_MODES = {
    "fast": {
        "display": "⚡ Fast",
        "description": "Lower resolution, more frame skipping. Quick preview.",
        "frame_skip": 10,           # process every Nth frame
        "detection_every_n": 10,    # run YOLO every Nth processed frame
        "resize_width": 640,
        "pose_every_n": 10,
        "max_frames": 500,
    },
    "balanced": {
        "display": "⚖️ Balanced",
        "description": "Medium resolution, moderate sampling. Recommended.",
        "frame_skip": 5,
        "detection_every_n": 5,
        "resize_width": 854,
        "pose_every_n": 5,
        "max_frames": 1500,
    },
    "detailed": {
        "display": "🔬 Detailed",
        "description": "Higher resolution, more frequent analysis. Slower.",
        "frame_skip": 2,
        "detection_every_n": 2,
        "resize_width": 1280,
        "pose_every_n": 3,
        "max_frames": 5000,
    },
}

DEFAULT_PROCESSING_MODE = os.getenv("DEFAULT_PROCESSING_MODE", "balanced")
DEFAULT_DETECTION_CONFIDENCE = float(os.getenv("DEFAULT_DETECTION_CONFIDENCE", "0.5"))
DEFAULT_TRACKING_CONFIDENCE = 0.3

# ============================================================
# Motion Analysis Settings
# ============================================================
# Pixel speed thresholds (pixels per second) — tunable per sport via sport config
STATIONARY_THRESHOLD_PX = 5.0     # below this = stationary
WALKING_THRESHOLD_PX = 30.0
JOGGING_THRESHOLD_PX = 80.0
RUNNING_THRESHOLD_PX = 130.0
# above RUNNING = sprinting

# Estimated real-world conversion (rough heuristic — labelled as ESTIMATE)
# Assumes ~25px per metre at typical broadcast zoom — very rough
PX_TO_METRE_RATIO = 0.04           # 1 pixel ≈ 0.04 metres (ESTIMATE)

# Minimum track length to include in analysis
MIN_TRACK_FRAMES = 5

# ============================================================
# Pose Estimation Settings
# ============================================================
MEDIAPIPE_MODEL_COMPLEXITY = 1   # 0=lite, 1=full, 2=heavy
MEDIAPIPE_MIN_DETECTION_CONF = 0.5
MEDIAPIPE_MIN_TRACKING_CONF = 0.5

# MediaPipe landmark indices (33-point model)
LANDMARK_INDICES = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13,    "right_elbow": 14,
    "left_wrist": 15,    "right_wrist": 16,
    "left_hip": 23,      "right_hip": 24,
    "left_knee": 25,     "right_knee": 26,
    "left_ankle": 27,    "right_ankle": 28,
}

# ============================================================
# Cleanup Settings
# ============================================================
AUTO_CLEANUP_TEMP = os.getenv("AUTO_CLEANUP_TEMP", "true").lower() == "true"
KEEP_ANNOTATED_VIDEO = os.getenv("KEEP_ANNOTATED_VIDEO", "true").lower() == "true"

# ============================================================
# Report Settings
# ============================================================
PDF_REPORT_TITLE = "AI Sports Video Analysis Report"
PDF_AUTHOR = "AI Sports Video Analyser v1.0"
PDF_PRIMARY_COLOR = (33, 150, 243)    # Blue
PDF_SECONDARY_COLOR = (76, 175, 80)   # Green
PDF_WARNING_COLOR = (255, 152, 0)     # Orange
PDF_TEXT_COLOR = (33, 33, 33)

# ============================================================
# Streamlit UI Settings
# ============================================================
STREAMLIT_THEME = {
    "primaryColor": "#2196F3",
    "backgroundColor": "#0E1117",
    "secondaryBackgroundColor": "#1A1A2E",
    "textColor": "#FAFAFA",
}

# Performance grade thresholds
GRADE_THRESHOLDS = {
    "A+": 90,
    "A": 80,
    "B": 65,
    "C": 50,
    "D": 35,
    "E": 0,
}

def get_grade(score: float) -> str:
    """Convert numeric score (0-100) to letter grade."""
    for grade, threshold in GRADE_THRESHOLDS.items():
        if score >= threshold:
            return grade
    return "E"
