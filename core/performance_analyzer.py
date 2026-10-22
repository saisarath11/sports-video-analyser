"""
Performance analysis engine computing prototype indicators and identifying strengths/weaknesses.
"""

from typing import Dict, List
from core.models import MotionMetrics, PlayerTrack, PoseLandmarks, SportEvent, PerformanceIndicators
from sports.config import get_sport_config
from utils.config import get_grade


class PerformanceAnalyzer:
    """
    Evaluates player performance metrics extracted from video.
    NOTE: Output scores are AI-generated prototype indicators, NOT scientifically validated metrics.
    """
    def __init__(self, sport: str = "football"):
        self.sport = sport
        self.config = get_sport_config(sport)
        self.weights = self.config["performance_weights"]

    def analyze_player(self, player_id: int, track: PlayerTrack,
                       metrics: MotionMetrics, events: List[SportEvent],
                       poses: List[PoseLandmarks],
                       all_metrics: Dict[int, MotionMetrics]) -> PerformanceIndicators:
        """Compute performance evaluation for a single player."""
        perf = PerformanceIndicators(player_id=player_id)

        # 1. Movement Score (distance relative to max player)
        all_dists = [m.total_distance_px for m in all_metrics.values() if m.total_distance_px > 0]
        max_dist = max(all_dists) if all_dists else 1.0
        perf.movement_score = round(min(100.0, (metrics.total_distance_px / max_dist) * 100), 1)

        # 2. Activity Score (active time ratio)
        if metrics.total_duration_s > 0:
            active_ratio = metrics.active_duration_s / metrics.total_duration_s
            perf.activity_score = round(min(100.0, active_ratio * 120), 1)

        # 3. Coverage Score (area covered relative to max player)
        all_areas = [m.coverage_area_px for m in all_metrics.values() if m.coverage_area_px > 0]
        max_area = max(all_areas) if all_areas else 1.0
        perf.coverage_score = round(min(100.0, (metrics.coverage_area_px / max_area) * 100), 1)

        # 4. Consistency Score (speed variance regularity)
        if metrics.speed_timeline and len(metrics.speed_timeline) > 1:
            speeds = [s for _, s in metrics.speed_timeline]
            mean_spd = np.mean(speeds)
            std_spd = np.std(speeds)
            cv = std_spd / (mean_spd + 1e-6)
            perf.consistency_score = round(max(0.0, min(100.0, 100.0 - (cv * 40))), 1)
        else:
            perf.consistency_score = 50.0

        # 5. Posture Score
        valid_poses = [p for p in poses if p.pose_detected]
        if valid_poses:
            p_scores = []
            for p in valid_poses:
                s = 60.0
                avg_k = (p.joint_angles.get("left_knee_angle", 180) + p.joint_angles.get("right_knee_angle", 180)) / 2
                if 110 <= avg_k <= 160:
                    s += 20.0
                if p.joint_angles.get("trunk_lean_angle", 0) > 10:
                    s += 20.0
                p_scores.append(min(100.0, s))
            perf.posture_score = round(float(np.mean(p_scores)), 1)
        else:
            perf.posture_score = 50.0

        # Overall weighted prototype score
        perf.overall_score = round(
            perf.movement_score * self.weights.get("movement_score", 0.3) +
            perf.activity_score * self.weights.get("activity_score", 0.25) +
            perf.coverage_score * self.weights.get("coverage_score", 0.20) +
            perf.consistency_score * self.weights.get("consistency_score", 0.15) +
            perf.posture_score * self.weights.get("posture_score", 0.10),
            1
        )
        perf.grade = get_grade(perf.overall_score)

        # Strengths & Weaknesses identification
        perf.strengths, perf.weaknesses = self._identify_strengths_weaknesses(perf, metrics)

        # Raw metrics summary
        perf.raw_metrics = {
            "est_distance_m": metrics.est_distance_m,
            "est_avg_speed_kmh": metrics.est_avg_speed_kmh,
            "est_max_speed_kmh": metrics.est_max_speed_kmh,
            "active_duration_s": metrics.active_duration_s,
            "stationary_duration_s": metrics.stationary_duration_s,
            "direction_changes": metrics.direction_changes
        }

        return perf

    def analyze_all_players(self, tracks: Dict[int, PlayerTrack],
                             motion_metrics: Dict[int, MotionMetrics],
                             events: List[SportEvent],
                             pose_data: Dict[int, List[PoseLandmarks]]) -> Dict[int, PerformanceIndicators]:
        """Compute performance evaluations for all tracked players."""
        results = {}
        for pid, track in tracks.items():
            metrics = motion_metrics.get(pid)
            if not metrics:
                continue
            p_events = [e for e in events if e.player_id == pid]
            p_poses = pose_data.get(pid, [])
            results[pid] = self.analyze_player(pid, track, metrics, p_events, p_poses, motion_metrics)
        return results

    def _identify_strengths_weaknesses(self, perf: PerformanceIndicators, metrics: MotionMetrics) -> tuple:
        strengths = []
        weaknesses = []

        if perf.movement_score >= 70:
            strengths.append("High movement activity and total distance covered.")
        elif perf.movement_score < 40:
            weaknesses.append("Low overall distance covered; consider increasing work rate.")

        if perf.activity_score >= 70:
            strengths.append("Excellent active work-rate to stationary ratio.")
        elif perf.activity_score < 40:
            weaknesses.append("High stationary time detected during active play phase.")

        if perf.coverage_score >= 70:
            strengths.append("Broad spatial field/court coverage.")
        elif perf.coverage_score < 40:
            weaknesses.append("Restricted spatial coverage; try expanding active positioning.")

        if metrics.direction_changes >= 4:
            strengths.append("Good agility and frequent directional adjustments.")

        if not strengths:
            strengths.append("Consistent participation maintained across session.")
        if not weaknesses:
            weaknesses.append("Maintain performance standards with continuous training.")

        return strengths[:4], weaknesses[:4]


import numpy as np
