"""
Unit tests for Motion Analyzer & Kinematics calculations.
"""

from core.models import PlayerTrack, TrackPoint, BoundingBox
from core.motion_analyzer import MotionAnalyzer


def test_motion_analyzer_distance():
    analyzer = MotionAnalyzer()
    track = PlayerTrack(track_id=1)
    
    # 2 points separated by 100px over 1 second
    tp1 = TrackPoint(frame_idx=0, timestamp=0.0, bbox=BoundingBox(0, 0, 10, 10), center=(0.0, 0.0))
    tp2 = TrackPoint(frame_idx=25, timestamp=1.0, bbox=BoundingBox(100, 0, 110, 10), center=(100.0, 0.0))
    
    track.track_points = [tp1, tp2]
    metrics = analyzer.analyze_track(track)
    
    assert metrics.total_distance_px == 100.0
    assert metrics.avg_speed_px_per_s == 100.0
    assert metrics.est_distance_m > 0
