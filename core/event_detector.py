"""
Modular rule-based sports event and activity detection engine.
"""

from typing import List, Dict
from core.models import PlayerTrack, MotionMetrics, PoseLandmarks, SportEvent
from sports.config import get_sport_config, get_speed_category


class EventDetector:
    """
    Detects events (sprints, walks, stance, direction changes) per sport config.
    """
    def __init__(self, sport: str = "football"):
        self.sport = sport
        self.config = get_sport_config(sport)
        self.thresholds = self.config["speed_thresholds"]

    def detect_events(self, tracks: Dict[int, PlayerTrack],
                      motion_metrics: Dict[int, MotionMetrics],
                      pose_data: Dict[int, List[PoseLandmarks]]) -> List[SportEvent]:
        """Detect sports activities and events across all player tracks."""
        events = []
        for player_id, track in tracks.items():
            metrics = motion_metrics.get(player_id)
            poses = pose_data.get(player_id, [])
            
            if metrics:
                events.extend(self._detect_motion_events(player_id, track, metrics))
            if poses:
                events.extend(self._detect_pose_events(player_id, poses))

        return sorted(events, key=lambda e: e.timestamp)

    def _detect_motion_events(self, player_id: int, track: PlayerTrack, metrics: MotionMetrics) -> List[SportEvent]:
        """Detect movement category events based on speed timeline."""
        events = []
        speed_timeline = metrics.speed_timeline
        if not speed_timeline:
            return events

        current_category = None
        event_start_ts = None
        event_start_frame = 0

        for ts, speed in speed_timeline:
            category = get_speed_category(speed, self.sport)
            if category != current_category:
                if current_category is not None and event_start_ts is not None:
                    duration = round(ts - event_start_ts, 2)
                    if duration >= 0.3:
                        events.append(SportEvent(
                            player_id=player_id,
                            event_type=current_category,
                            timestamp=event_start_ts,
                            duration=duration,
                            confidence=0.85,
                            frame_idx=event_start_frame
                        ))
                current_category = category
                event_start_ts = ts

        # Close open interval
        if current_category and event_start_ts and speed_timeline:
            last_ts = speed_timeline[-1][0]
            duration = round(last_ts - event_start_ts, 2)
            if duration >= 0.3:
                events.append(SportEvent(
                    player_id=player_id,
                    event_type=current_category,
                    timestamp=event_start_ts,
                    duration=duration,
                    confidence=0.85
                ))

        # Detect direction changes as distinct event
        if metrics.direction_changes > 0:
            events.append(SportEvent(
                player_id=player_id,
                event_type="direction_change",
                timestamp=track.first_seen,
                duration=0.5,
                confidence=0.8,
                metadata={"count": metrics.direction_changes}
            ))

        return events

    def _detect_pose_events(self, player_id: int, poses: List[PoseLandmarks]) -> List[SportEvent]:
        """Detect posture-based action events."""
        events = []
        for pose in poses:
            if not pose.pose_detected:
                continue

            angles = pose.joint_angles
            avg_knee = (angles.get("left_knee_angle", 180) + angles.get("right_knee_angle", 180)) / 2

            if 100 <= avg_knee <= 155:
                events.append(SportEvent(
                    player_id=player_id,
                    event_type="ready_position",
                    timestamp=pose.timestamp,
                    duration=0.2,
                    confidence=0.75,
                    frame_idx=pose.frame_idx
                ))

        return events
