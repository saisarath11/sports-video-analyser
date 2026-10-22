"""
Shared data models for the AI Sports Video Analyser.
All pipeline modules import from here to ensure consistent data structures.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import time


# ============================================================
# Enumerations
# ============================================================

class ProcessingMode(str, Enum):
    FAST = "fast"          # Every 10th frame, lower res — quick preview
    BALANCED = "balanced"  # Every 5th frame, medium res — recommended
    DETAILED = "detailed"  # Every 2nd frame, full res — thorough analysis


class Sport(str, Enum):
    FOOTBALL = "football"
    CRICKET = "cricket"
    BASKETBALL = "basketball"
    BADMINTON = "badminton"
    TENNIS = "tennis"


class EventType(str, Enum):
    # Universal events (all sports)
    RUNNING = "running"
    SPRINTING = "sprinting"
    JOGGING = "jogging"
    WALKING = "walking"
    STATIONARY = "stationary"
    DIRECTION_CHANGE = "direction_change"
    # Sport specific added dynamically via config


# ============================================================
# Video & Detection Models
# ============================================================

@dataclass
class VideoInfo:
    """Metadata about the uploaded video."""
    filename: str
    filepath: str
    duration: float        # seconds
    fps: float
    width: int
    height: int
    frame_count: int
    file_size_mb: float
    format: str
    is_valid: bool = True
    error_message: str = ""


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return abs(self.x2 - self.x1)

    @property
    def height(self) -> float:
        return abs(self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_xywh(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)

    def contains_point(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass
class Detection:
    """A single player detection in one frame."""
    frame_idx: int
    timestamp: float        # seconds from start
    bbox: BoundingBox
    confidence: float       # 0.0 – 1.0
    class_id: int = 0
    class_name: str = "person"
    player_id: Optional[int] = None   # assigned by tracker


# ============================================================
# Pose Estimation Models
# ============================================================

@dataclass
class Landmark:
    """A single body keypoint from pose estimation."""
    x: float            # Normalized 0.0–1.0 relative to frame/crop
    y: float            # Normalized 0.0–1.0
    z: float = 0.0      # Depth (MediaPipe provides this)
    visibility: float = 1.0   # Confidence of landmark


@dataclass
class PoseLandmarks:
    """Full pose estimate for one player in one frame."""
    player_id: int
    frame_idx: int
    timestamp: float
    landmarks: List[Landmark] = field(default_factory=list)
    # Computed joint angles (degrees)
    joint_angles: Dict[str, float] = field(default_factory=dict)
    # Human-readable posture observations
    posture_notes: List[str] = field(default_factory=list)
    pose_detected: bool = False
    crop_x: float = 0.0
    crop_y: float = 0.0
    crop_w: float = 0.0
    crop_h: float = 0.0


@dataclass
class PostureObservation:
    """A noteworthy posture event for a player."""
    player_id: int
    timestamp: float
    observation: str
    confidence: float = 0.5
    joint_name: str = ""


# ============================================================
# Tracking Models
# ============================================================

@dataclass
class TrackPoint:
    """One position sample in a player's track."""
    frame_idx: int
    timestamp: float
    bbox: BoundingBox
    center: Tuple[float, float]


@dataclass
class PlayerTrack:
    """Complete trajectory of one tracked player."""
    track_id: int
    track_points: List[TrackPoint] = field(default_factory=list)

    @property
    def centers(self) -> List[Tuple[float, float]]:
        return [tp.center for tp in self.track_points]

    @property
    def timestamps(self) -> List[float]:
        return [tp.timestamp for tp in self.track_points]

    @property
    def frame_indices(self) -> List[int]:
        return [tp.frame_idx for tp in self.track_points]

    @property
    def first_seen(self) -> float:
        return self.track_points[0].timestamp if self.track_points else 0.0

    @property
    def last_seen(self) -> float:
        return self.track_points[-1].timestamp if self.track_points else 0.0

    @property
    def duration(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def frame_count(self) -> int:
        return len(self.track_points)


# ============================================================
# Motion Analysis Models
# ============================================================

@dataclass
class MotionMetrics:
    """Computed motion statistics for one player."""
    player_id: int
    total_distance_px: float = 0.0       # pixel distance
    avg_speed_px_per_s: float = 0.0      # pixels/second
    max_speed_px_per_s: float = 0.0
    active_duration_s: float = 0.0       # time in motion
    stationary_duration_s: float = 0.0  # time not moving
    total_duration_s: float = 0.0
    movement_count: int = 0              # number of movement bursts
    direction_changes: int = 0
    positions: List[Tuple[float, float]] = field(default_factory=list)
    speed_timeline: List[Tuple[float, float]] = field(default_factory=list)  # (ts, speed_px/s)
    coverage_area_px: float = 0.0        # convex hull area of positions
    # Estimated real-world values — CLEARLY LABELLED AS ESTIMATES
    est_distance_m: float = 0.0
    est_avg_speed_kmh: float = 0.0
    est_max_speed_kmh: float = 0.0


# ============================================================
# Event Detection Models
# ============================================================

@dataclass
class SportEvent:
    """A detected sports activity/event."""
    player_id: int
    event_type: str           # e.g. "running", "sprinting", "shot_attempt"
    timestamp: float          # start time (seconds)
    duration: float = 0.0    # duration (seconds)
    confidence: float = 1.0
    frame_idx: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Performance Evaluation Models
# ============================================================

@dataclass
class PerformanceIndicators:
    """
    AI-generated performance indicators for one player.
    NOTE: Scores are PROTOTYPE indicators based on video-extracted features.
    They are NOT scientifically validated metrics.
    """
    player_id: int
    overall_score: float = 0.0      # 0–100, prototype score
    movement_score: float = 0.0     # based on distance & speed
    activity_score: float = 0.0     # based on active time ratio
    consistency_score: float = 0.0  # based on movement regularity
    coverage_score: float = 0.0     # based on area covered
    posture_score: float = 0.0      # based on pose landmarks
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    raw_metrics: Dict[str, Any] = field(default_factory=dict)
    grade: str = "N/A"              # A+, A, B, C, D


# ============================================================
# Feedback Models
# ============================================================

@dataclass
class PlayerFeedback:
    """Personalized AI-generated feedback for one player."""
    player_id: int
    sport: str
    summary: str = ""
    strengths: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    training_suggestions: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    generated_by: str = "template"   # "template" or "gemini"
    timestamp: float = field(default_factory=time.time)


# ============================================================
# Processing Statistics
# ============================================================

@dataclass
class ProcessingStats:
    """Timing and throughput statistics for one analysis run."""
    total_frames: int = 0
    processed_frames: int = 0
    detection_frames: int = 0
    pose_frames: int = 0
    processing_time_s: float = 0.0
    avg_fps: float = 0.0
    model_load_time_s: float = 0.0
    detection_time_s: float = 0.0
    tracking_time_s: float = 0.0
    pose_time_s: float = 0.0
    motion_analysis_time_s: float = 0.0
    device: str = "cpu"


# ============================================================
# Master Analysis Result
# ============================================================

@dataclass
class AnalysisResult:
    """
    Complete analysis result for one video session.
    This is the single object passed between pipeline stages and stored to DB.
    """
    session_id: str
    video_info: Optional[VideoInfo] = None
    sport: str = "football"
    processing_mode: str = "balanced"
    detections: List[Detection] = field(default_factory=list)
    tracks: Dict[int, PlayerTrack] = field(default_factory=dict)
    pose_data: Dict[int, List[PoseLandmarks]] = field(default_factory=dict)
    posture_observations: List[PostureObservation] = field(default_factory=list)
    motion_metrics: Dict[int, MotionMetrics] = field(default_factory=dict)
    events: List[SportEvent] = field(default_factory=list)
    performance: Dict[int, PerformanceIndicators] = field(default_factory=dict)
    feedback: Dict[int, PlayerFeedback] = field(default_factory=dict)
    processing_stats: ProcessingStats = field(default_factory=ProcessingStats)
    annotated_video_path: Optional[str] = None
    report_path: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    is_complete: bool = False

    def get_player_ids(self) -> List[int]:
        return sorted(self.tracks.keys())

    def get_player_count(self) -> int:
        return len(self.tracks)

    def get_events_for_player(self, player_id: int) -> List[SportEvent]:
        return [e for e in self.events if e.player_id == player_id]

    def get_total_events(self) -> int:
        return len(self.events)
