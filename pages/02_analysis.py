"""
Streamlit Page 2 - AI Pipeline Analysis Execution with progress tracking.
"""

import streamlit as st
import time
import os
from pathlib import Path
from ultralytics import YOLO

from core.video_processor import VideoProcessor
from core.detector import PlayerDetector
from core.tracker import PlayerTracker
from core.pose_estimator import PoseEstimator
from core.motion_analyzer import MotionAnalyzer
from core.event_detector import EventDetector
from core.performance_analyzer import PerformanceAnalyzer
from core.feedback_generator import FeedbackGenerator
from core.models import AnalysisResult, ProcessingStats, Detection
from visualization.annotator import AnnotatedVideoWriter

from utils.config import YOLO_MODEL_NAME, MODELS_DIR, DATA_OUTPUT_DIR, PROCESSING_MODES
from utils.file_utils import generate_session_id
from utils.logging_utils import logger


# ─── Helper Functions (Must be defined at top before page execution) ──────────
def _dedup_by_iou(detections: list, iou_thresh: float = 0.4) -> list:
    """Remove duplicate detections that significantly overlap."""
    if len(detections) <= 1:
        return detections
    # Sort largest area first (keep the more complete bounding box)
    sorted_dets = sorted(detections, key=lambda d: d.bbox.area, reverse=True)
    kept = []
    suppressed = set()
    for i, di in enumerate(sorted_dets):
        if i in suppressed:
            continue
        kept.append(di)
        for j, dj in enumerate(sorted_dets):
            if j <= i or j in suppressed:
                continue
            ix1 = max(di.bbox.x1, dj.bbox.x1)
            iy1 = max(di.bbox.y1, dj.bbox.y1)
            ix2 = min(di.bbox.x2, dj.bbox.x2)
            iy2 = min(di.bbox.y2, dj.bbox.y2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = di.bbox.area + dj.bbox.area - inter
            if union > 0 and (inter / union) > iou_thresh:
                suppressed.add(j)
    return kept


st.set_page_config(page_title="Analysis | AI Sports Analyser", page_icon="🔬", layout="wide")

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #12121f; }
.stProgress > div > div > div { background: linear-gradient(90deg, #00c8ff, #7b2ff7); }
.step-card {
    background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 10px;
    padding: 14px 18px; margin: 6px 0;
}
.step-active  { border-color: #00c8ff; color: #00c8ff; }
.step-done    { border-color: #00e676; color: #00e676; }
</style>
""", unsafe_allow_html=True)

st.title("🔬 AI Pipeline Analysis")

if "video_path" not in st.session_state:
    st.warning("⚠️ No video loaded. Go to **Upload Video** page first.")
    st.stop()

video_path = st.session_state["video_path"]
sport      = st.session_state.get("selected_sport", "football")
mode       = st.session_state.get("selected_mode", "balanced")
confidence = st.session_state.get("confidence", 0.5)
use_gemini = st.session_state.get("use_gemini", True)

mode_cfg = PROCESSING_MODES.get(mode, PROCESSING_MODES["balanced"])
frame_skip = mode_cfg["frame_skip"]

col_info1, col_info2, col_info3 = st.columns(3)
col_info1.info(f"**Video:** {Path(video_path).name}")
col_info2.info(f"**Sport:** {sport.capitalize()}  |  **Mode:** {mode.capitalize()}")
col_info3.info(f"**Confidence:** {confidence:.0%}  |  **Frame Skip:** every {frame_skip}th frame")

st.divider()

if st.button("▶️  Start Full AI Analysis", type="primary", use_container_width=True):
    progress_bar = st.progress(0)
    status_box   = st.empty()
    log_box      = st.empty()

    session_id = generate_session_id()
    result     = AnalysisResult(session_id=session_id, sport=sport, processing_mode=mode)
    start_time = time.time()
    logs       = []

    def step(msg: str, pct: int):
        progress_bar.progress(pct)
        status_box.markdown(f"**⚙️ {msg}**")
        logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        log_box.code("\n".join(logs[-6:]))

    try:
        # ── Step 1: Video Load ──────────────────────────────────────────────
        step("Initialising video processor & loading metadata…", 5)
        processor   = VideoProcessor(video_path, mode)
        video_info  = processor.load()
        result.video_info = video_info
        total_proc_frames = processor.get_total_frames_to_process()

        # ── Step 2: Load Models ─────────────────────────────────────────────
        step("Loading YOLOv8 detector, tracker & pose models…", 15)
        detector       = PlayerDetector(confidence=confidence)
        tracker        = PlayerTracker()
        pose_estimator = PoseEstimator()

        # YOLO model for tracking
        yolo_path = MODELS_DIR / YOLO_MODEL_NAME
        model     = YOLO(str(yolo_path) if yolo_path.exists() else YOLO_MODEL_NAME)

        annotated_out = os.path.join(DATA_OUTPUT_DIR, f"annotated_{session_id}.mp4")
        # Pass frame_skip so each processed frame is written N times → correct speed
        writer = AnnotatedVideoWriter(annotated_out, video_info, frame_skip=frame_skip)

        # ── Step 3: Frame Loop ──────────────────────────────────────────────
        step("Running detection, tracking & pose estimation on video…", 20)
        frame_counter = 0

        for frame_idx, timestamp, frame in processor.get_frame_generator():
            frame_counter += 1

            # YOLO ByteTrack (person class only)
            yolo_results = model.track(
                frame, persist=True, tracker="bytetrack.yaml",
                classes=[0], conf=confidence, iou=0.4, verbose=False
            )

            detections = tracker.update_from_yolo_results(yolo_results, frame_idx, timestamp)

            # Secondary deduplication: if 2 boxes overlap > 40%, keep the bigger (more complete) one
            detections = _dedup_by_iou(detections, iou_thresh=0.4)

            result.detections.extend(detections)

            # Pose estimation for each tracked player
            pose_dict = {}
            for det in detections:
                if det.player_id is not None:
                    pose = pose_estimator.estimate(frame, det.bbox, det.player_id, frame_idx, timestamp)
                    if pose:
                        pose_dict[det.player_id] = pose
                        result.pose_data.setdefault(det.player_id, []).append(pose)

            # Annotate and write — repeat frame_skip times for correct playback speed
            annotated = writer.annotate_frame(
                frame, frame_idx, timestamp,
                detections, tracker.tracks, pose_dict, sport=sport
            )
            writer.write_frame(annotated, repeat=frame_skip)

            # Progress
            pct = 20 + int((frame_counter / max(1, total_proc_frames)) * 45)
            progress_bar.progress(min(65, pct))

        writer.release()
        pose_estimator.close()
        result.annotated_video_path = annotated_out

        # ── Step 4: Motion Analysis ─────────────────────────────────────────
        step("Computing motion metrics, trajectories & heatmaps…", 70)
        result.tracks = tracker.get_valid_tracks()
        motion_analyzer = MotionAnalyzer()
        result.motion_metrics = motion_analyzer.analyze_all_tracks(result.tracks)

        # ── Step 5: Event Detection ─────────────────────────────────────────
        step("Detecting sports activity events…", 78)
        event_detector = EventDetector(sport)
        result.events  = event_detector.detect_events(result.tracks, result.motion_metrics, result.pose_data)

        # ── Step 6: Performance Evaluation ─────────────────────────────────
        step("Evaluating prototype performance indicators…", 86)
        perf_analyzer   = PerformanceAnalyzer(sport)
        result.performance = perf_analyzer.analyze_all_players(
            result.tracks, result.motion_metrics, result.events, result.pose_data
        )

        # ── Step 7: AI Feedback ─────────────────────────────────────────────
        step("Generating personalised AI coaching feedback…", 93)
        feedback_gen   = FeedbackGenerator(sport, use_gemini=use_gemini)
        result.feedback = feedback_gen.generate_all_feedback(result)

        elapsed = round(time.time() - start_time, 2)
        result.processing_stats = ProcessingStats(
            total_frames=video_info.frame_count,
            processed_frames=frame_counter,
            processing_time_s=elapsed,
            avg_fps=round(frame_counter / max(0.1, elapsed), 2)
        )
        result.is_complete = True
        st.session_state["analysis_result"] = result

        progress_bar.progress(100)
        status_box.empty()
        st.success(f"✅ Analysis complete in **{elapsed}s**! Processed **{frame_counter}** frames. Players tracked: **{result.get_player_count()}**")

    except Exception as e:
        st.error(f"❌ Pipeline error: {e}")
        logger.exception("Error during pipeline execution")

# ─── Results Panel ────────────────────────────────────────────────────────────
if "analysis_result" in st.session_state:
    res = st.session_state["analysis_result"]
    st.divider()
    st.subheader("📊 Session Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players Tracked", res.get_player_count())
    c2.metric("Events Detected", res.get_total_events())
    c3.metric("Frames Processed", res.processing_stats.processed_frames)
    c4.metric("Processing Time", f"{res.processing_stats.processing_time_s}s")

    if res.annotated_video_path and os.path.exists(res.annotated_video_path):
        st.subheader("📹 Annotated Video")
        st.caption("Video plays at correct real-time speed. Each frame is repeated to match original FPS.")
        st.video(res.annotated_video_path)
