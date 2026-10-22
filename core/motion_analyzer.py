"""
Motion analysis engine computing pixel distances, estimated speeds, active time, and trajectories.
Includes spatial noise filtering and position smoothing.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial import ConvexHull

from core.models import PlayerTrack, MotionMetrics
from utils.config import STATIONARY_THRESHOLD_PX, PX_TO_METRE_RATIO


class MotionAnalyzer:
    """
    Computes kinematics metrics (distance, speed, acceleration estimates, activity ratio)
    from multi-object player tracks with position smoothing.
    """
    def _smooth_positions(self, positions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Apply 3-point weighted moving average filter to reduce detection jitter noise."""
        if len(positions) < 3:
            return positions

        smoothed = [positions[0]]
        for i in range(1, len(positions) - 1):
            sx = (positions[i-1][0] + 2 * positions[i][0] + positions[i+1][0]) / 4.0
            sy = (positions[i-1][1] + 2 * positions[i][1] + positions[i+1][1]) / 4.0
            smoothed.append((round(sx, 2), round(sy, 2)))
        smoothed.append(positions[-1])
        return smoothed

    def analyze_track(self, track: PlayerTrack) -> MotionMetrics:
        """Compute full MotionMetrics for a single player trajectory."""
        metrics = MotionMetrics(player_id=track.track_id)
        
        if len(track.track_points) < 2:
            return metrics

        raw_positions = track.centers
        timestamps = track.timestamps
        positions = self._smooth_positions(raw_positions)

        distances = []
        speeds = []
        speed_timeline = []

        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            dist = np.sqrt(dx**2 + dy**2)
            dt = max(timestamps[i] - timestamps[i-1], 1e-6)
            speed = dist / dt

            # Filter unphysical speed spikes (> 500 px/s is usually detection teleporting)
            if speed > 500.0:
                speed = speeds[-1] if speeds else 0.0
                dist = speed * dt

            distances.append(dist)
            speeds.append(speed)
            speed_timeline.append((timestamps[i], round(speed, 2)))

        metrics.total_distance_px = round(float(np.sum(distances)), 1)
        metrics.avg_speed_px_per_s = round(float(np.mean(speeds)), 2) if speeds else 0.0
        metrics.max_speed_px_per_s = round(float(np.max(speeds)), 2) if speeds else 0.0
        metrics.speed_timeline = speed_timeline
        metrics.positions = list(positions)
        metrics.total_duration_s = round(timestamps[-1] - timestamps[0], 2)

        # Active vs Stationary duration
        active_time = sum(
            dt for spd, dt in zip(
                speeds, 
                [timestamps[i+1] - timestamps[i] for i in range(len(speeds))]
            )
            if spd > STATIONARY_THRESHOLD_PX
        )
        metrics.active_duration_s = round(float(active_time), 2)
        metrics.stationary_duration_s = round(float(max(0.0, metrics.total_duration_s - active_time)), 2)

        # Count movement bursts
        in_motion = False
        count = 0
        for spd in speeds:
            if spd > STATIONARY_THRESHOLD_PX and not in_motion:
                in_motion = True
                count += 1
            elif spd <= STATIONARY_THRESHOLD_PX:
                in_motion = False
        metrics.movement_count = count

        # Direction changes and area coverage
        metrics.direction_changes = self._count_direction_changes(positions)
        metrics.coverage_area_px = self._compute_coverage_area(positions)

        # Estimated real-world metrics (CLEARLY LABELLED AS ESTIMATES)
        metrics.est_distance_m = round(metrics.total_distance_px * PX_TO_METRE_RATIO, 2)
        if metrics.total_duration_s > 0:
            metrics.est_avg_speed_kmh = round((metrics.est_distance_m / metrics.total_duration_s) * 3.6, 2)
            metrics.est_max_speed_kmh = round(metrics.max_speed_px_per_s * PX_TO_METRE_RATIO * 3.6, 2)

        return metrics

    def analyze_all_tracks(self, tracks: Dict[int, PlayerTrack]) -> Dict[int, MotionMetrics]:
        """Compute MotionMetrics for all recorded player tracks."""
        return {tid: self.analyze_track(track) for tid, track in tracks.items()}

    def _count_direction_changes(self, positions: List[Tuple[float, float]], threshold_angle: float = 40.0) -> int:
        """Count significant direction changes in trajectory."""
        if len(positions) < 3:
            return 0
        changes = 0
        for i in range(1, len(positions) - 1):
            v1 = (positions[i][0] - positions[i-1][0], positions[i][1] - positions[i-1][1])
            v2 = (positions[i+1][0] - positions[i][0], positions[i+1][1] - positions[i][1])
            mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
            if mag1 < 3.0 or mag2 < 3.0:  # Ignore direction changes during near-stationary frames
                continue
            cos_a = (v1[0]*v2[0] + v1[1]*v2[1]) / (mag1 * mag2)
            angle = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
            if angle > threshold_angle:
                changes += 1
        return changes

    def _compute_coverage_area(self, positions: List[Tuple[float, float]]) -> float:
        """Compute convex hull area of player path positions."""
        if len(positions) < 3:
            return 0.0
        try:
            pts = np.array(positions)
            hull = ConvexHull(pts)
            return round(float(hull.volume), 1)
        except Exception:
            return 0.0

    def generate_heatmap_data(self, metrics_dict: Dict[int, MotionMetrics], frame_width: int, frame_height: int) -> np.ndarray:
        """Generate 2D heatmap matrix from positional points."""
        heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
        for metrics in metrics_dict.values():
            for x, y in metrics.positions:
                xi = int(np.clip(x, 0, frame_width - 1))
                yi = int(np.clip(y, 0, frame_height - 1))
                heatmap[yi, xi] += 1.0

        if np.max(heatmap) > 0:
            heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        return heatmap
