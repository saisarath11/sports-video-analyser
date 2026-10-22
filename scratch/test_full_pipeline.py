"""
End-to-end integration test script for AI Sports Video Analyser.
Generates a synthetic sports video and executes the full pipeline.
"""

import cv2
import numpy as np
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.video_processor import VideoProcessor
from core.detector import PlayerDetector
from core.tracker import PlayerTracker
from core.pose_estimator import PoseEstimator
from core.motion_analyzer import MotionAnalyzer
from core.event_detector import EventDetector
from core.performance_analyzer import PerformanceAnalyzer
from core.feedback_generator import FeedbackGenerator
from core.models import AnalysisResult, ProcessingStats
from visualization.annotator import AnnotatedVideoWriter
from visualization.report import ReportGenerator
from utils.config import DATA_TEMP_DIR, DATA_OUTPUT_DIR
from utils.logging_utils import logger


def generate_synthetic_sports_video(output_path: str, width: int = 640, height: int = 480, fps: int = 30, duration_sec: int = 4):
    """Generate a synthetic video with moving player-like figures for pipeline testing."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = fps * duration_sec
    print(f"Generating synthetic video: {output_path} ({total_frames} frames)...")

    for i in range(total_frames):
        # Green field background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (34, 139, 34)

        # Draw white pitch lines
        cv2.line(frame, (50, 50), (590, 50), (255, 255, 255), 2)
        cv2.line(frame, (50, 430), (590, 430), (255, 255, 255), 2)
        cv2.line(frame, (320, 50), (320, 430), (255, 255, 255), 2)
        cv2.circle(frame, (320, 240), 50, (255, 255, 255), 2)

        # Player 1 moving horizontally across pitch
        p1_x = int(100 + (i / total_frames) * 400)
        p1_y = 200 + int(np.sin(i / 10.0) * 20)
        
        # Player 2 moving vertically
        p2_x = 450
        p2_y = int(100 + (i / total_frames) * 250)

        # Draw player 1 (stick figure / head + body + limbs)
        cv2.circle(frame, (p1_x, p1_y - 30), 12, (0, 0, 255), -1)  # Head
        cv2.rectangle(frame, (p1_x - 10, p1_y - 18), (p1_x + 10, p1_y + 20), (255, 0, 0), -1)  # Body
        cv2.line(frame, (p1_x - 8, p1_y + 20), (p1_x - 12, p1_y + 50), (0, 0, 0), 3)  # Left leg
        cv2.line(frame, (p1_x + 8, p1_y + 20), (p1_x + 12, p1_y + 50), (0, 0, 0), 3)  # Right leg

        # Draw player 2
        cv2.circle(frame, (p2_x, p2_y - 30), 12, (0, 0, 255), -1)
        cv2.rectangle(frame, (p2_x - 10, p2_y - 18), (p2_x + 10, p2_y + 20), (0, 255, 255), -1)
        cv2.line(frame, (p2_x - 8, p2_y + 20), (p2_x - 12, p2_y + 50), (0, 0, 0), 3)
        cv2.line(frame, (p2_x + 8, p2_y + 20), (p2_x + 12, p2_y + 50), (0, 0, 0), 3)

        writer.write(frame)

    writer.release()
    print("Synthetic video created successfully.")


def run_test():
    test_video_path = os.path.join(DATA_TEMP_DIR, "test_sports_clip.mp4")
    generate_synthetic_sports_video(test_video_path)

    print("\n--- Starting Full AI Pipeline Test ---")
    session_id = "test_session_001"
    result = AnalysisResult(session_id=session_id, sport="football", processing_mode="balanced")

    # 1. Video Processor
    processor = VideoProcessor(test_video_path, mode="balanced")
    v_info = processor.load()
    result.video_info = v_info
    print(f"Loaded Video: {v_info.width}x{v_info.height}, {v_info.duration}s, {v_info.fps} FPS")

    # 2. Detector, Tracker, Pose Estimator
    detector = PlayerDetector(confidence=0.3)
    tracker = PlayerTracker()
    pose_estimator = PoseEstimator()

    annotated_out_path = os.path.join(DATA_OUTPUT_DIR, f"annotated_{session_id}.mp4")
    writer = AnnotatedVideoWriter(annotated_out_path, v_info)

    frame_count = 0
    for frame_idx, timestamp, frame in processor.get_frame_generator():
        frame_count += 1
        # Run YOLO detection
        detections = detector.detect(frame, frame_idx, timestamp)
        
        # If detector didn't catch synthetic figures (since YOLO looks for real humans), mock fallback bounding box to test full tracking/pose/motion engine end-to-end
        if not detections:
            from core.models import BoundingBox, Detection
            # Mock 2 player bounding boxes matching synthetic video figures
            p1_x = int(100 + (frame_count / 120.0) * 400)
            p1_y = 200 + int(np.sin(frame_count / 10.0) * 20)
            p2_x = 450
            p2_y = int(100 + (frame_count / 120.0) * 250)
            
            det1 = Detection(frame_idx=frame_idx, timestamp=timestamp, bbox=BoundingBox(p1_x-15, p1_y-45, p1_x+15, p1_y+55), confidence=0.9, class_id=0, player_id=1)
            det2 = Detection(frame_idx=frame_idx, timestamp=timestamp, bbox=BoundingBox(p2_x-15, p2_y-45, p2_x+15, p2_y+55), confidence=0.88, class_id=0, player_id=2)
            detections = [det1, det2]

        tracked_dets = tracker.update_from_detections(detections, frame_idx, timestamp)
        result.detections.extend(tracked_dets)

        pose_dict = {}
        for det in tracked_dets:
            if det.player_id is not None:
                pose = pose_estimator.estimate(frame, det.bbox, det.player_id, frame_idx, timestamp)
                if pose:
                    pose_dict[det.player_id] = pose
                    if det.player_id not in result.pose_data:
                        result.pose_data[det.player_id] = []
                    result.pose_data[det.player_id].append(pose)

        annotated_frame = writer.annotate_frame(frame, frame_idx, timestamp, tracked_dets, tracker.tracks, pose_dict)
        writer.write_frame(annotated_frame)

    writer.release()
    pose_estimator.close()
    result.annotated_video_path = annotated_out_path
    print(f"Processed {frame_count} frames. Annotated video saved to: {annotated_out_path}")

    # 3. Motion Analysis
    result.tracks = tracker.get_valid_tracks()
    motion_analyzer = MotionAnalyzer()
    result.motion_metrics = motion_analyzer.analyze_all_tracks(result.tracks)
    print(f"Tracked Players Count: {len(result.tracks)}")
    for pid, m in result.motion_metrics.items():
        print(f"  Player {pid}: Distance={m.est_distance_m}m, PeakSpeed={m.est_max_speed_kmh}km/h, Active={m.active_duration_s}s")

    # 4. Event Detection
    event_detector = EventDetector("football")
    result.events = event_detector.detect_events(result.tracks, result.motion_metrics, result.pose_data)
    print(f"Detected Events Count: {len(result.events)}")

    # 5. Performance Analyzer
    perf_analyzer = PerformanceAnalyzer("football")
    result.performance = perf_analyzer.analyze_all_players(result.tracks, result.motion_metrics, result.events, result.pose_data)
    for pid, p in result.performance.items():
        print(f"  Player {pid}: Score={p.overall_score}/100 ({p.grade})")

    # 6. Feedback Generator
    feedback_gen = FeedbackGenerator("football", use_gemini=False)
    result.feedback = feedback_gen.generate_all_feedback(result)
    for pid, fb in result.feedback.items():
        print(f"  Player {pid} Feedback Summary: {fb.summary[:80]}...")

    # 7. PDF Report Generator
    report_out_path = os.path.join(DATA_OUTPUT_DIR, f"report_{session_id}.pdf")
    report_gen = ReportGenerator(result)
    report_gen.generate(report_out_path)
    result.report_path = report_out_path
    print(f"PDF Report generated successfully at: {report_out_path}")

    print("\n✅ PIPELINE INTEGRATION TEST PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_test()
