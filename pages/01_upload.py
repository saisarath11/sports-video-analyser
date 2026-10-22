"""
Streamlit Page 1 — Video Upload & Configuration
"""
import streamlit as st
import os
from pathlib import Path

from utils.file_utils import validate_video_file, save_uploaded_file, get_video_info, generate_session_id
from utils.config import MAX_UPLOAD_SIZE_MB, ALLOWED_VIDEO_FORMATS, DATA_INPUT_DIR, PROCESSING_MODES
from sports.config import list_sports

st.set_page_config(page_title="Upload | AI Sports Analyser", page_icon="📤", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0d0d1a; }
.upload-card {
    background: #12121f; border: 1px solid #2a2a4a; border-radius: 14px;
    padding: 24px; margin-bottom: 20px;
}
.info-chip {
    background: #1a1a2e; border: 1px solid #3a3a6a; border-radius: 8px;
    padding: 10px 14px; font-size: 0.85rem; color: #aac0d8; margin: 4px 0;
}
.option-header { color: #00c8ff; font-weight: 700; font-size: 1.05rem; margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

st.title("📤 Upload Video & Configure")
st.caption("Upload a recorded sports video to begin AI analysis. Supported: MP4, AVI, MOV, MKV | Max size: 500 MB")

# ─── Upload Zone ──────────────────────────────────────────────────────────────
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drop your sports video here or click Browse",
    type=[f.replace(".", "") for f in ALLOWED_VIDEO_FORMATS],
    help=f"Supported: {', '.join(ALLOWED_VIDEO_FORMATS)} · Max {MAX_UPLOAD_SIZE_MB} MB"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Save to disk
    temp_path = save_uploaded_file(uploaded_file, str(DATA_INPUT_DIR))
    is_valid, err_msg = validate_video_file(temp_path)

    if not is_valid:
        st.error(f"❌ **Validation Failed:** {err_msg}")
        st.stop()

    # Load metadata
    info = get_video_info(temp_path)
    st.session_state["video_info"]  = info
    st.session_state["video_path"]  = temp_path

    st.success(f"✅ **{info.filename}** uploaded and validated successfully!")

    # ─── Video Metadata Row ───────────────────────────────────────────────
    st.subheader("📋 Video Information")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Duration",    f"{info.duration:.1f} s")
    m2.metric("Resolution",  f"{info.width}×{info.height}")
    m3.metric("Frame Rate",  f"{info.fps:.1f} FPS")
    m4.metric("File Size",   f"{info.file_size_mb} MB")
    m5.metric("Format",      info.format.upper())

    # ─── Video Preview ────────────────────────────────────────────────────
    st.subheader("🎬 Preview")
    st.video(temp_path)

    st.divider()

    # ─── Analysis Options ─────────────────────────────────────────────────
    st.subheader("⚙️ Analysis Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="option-header">🏆 Select Sport</div>', unsafe_allow_html=True)
        sports = list_sports()
        sport_map = {k: f"{icon} {name}" for k, name, icon in sports}
        selected_sport = st.selectbox(
            "Sport", list(sport_map.keys()), format_func=lambda x: sport_map[x], label_visibility="collapsed"
        )

    with col2:
        st.markdown('<div class="option-header">⚡ Processing Mode</div>', unsafe_allow_html=True)
        mode_map = {k: v["display"] for k, v in PROCESSING_MODES.items()}
        selected_mode = st.selectbox(
            "Mode", list(mode_map.keys()), index=1, format_func=lambda x: mode_map[x], label_visibility="collapsed"
        )
        mode_info = PROCESSING_MODES[selected_mode]
        st.caption(f"Frame skip: every **{mode_info['frame_skip']}th** frame · Resize: **{mode_info['resize_width']}px**")

    with col3:
        st.markdown('<div class="option-header">🎯 Detection Threshold</div>', unsafe_allow_html=True)
        conf = st.slider("Confidence", 0.25, 0.90, 0.50, 0.05, label_visibility="collapsed")
        st.caption("Higher = fewer but more accurate detections. Recommended: **0.5**")

    # Gemini API toggle
    use_gemini = st.checkbox(
        "🤖 Enable Gemini AI Coaching Feedback",
        value=True,
        help="Requires GEMINI_API_KEY in your .env file. Falls back to template feedback if unavailable."
    )

    # Save to session
    st.session_state["selected_sport"] = selected_sport
    st.session_state["selected_mode"]  = selected_mode
    st.session_state["confidence"]     = conf
    st.session_state["use_gemini"]     = use_gemini

    st.divider()
    st.info("👉 Navigate to **🔬 Analysis** in the sidebar to run the AI pipeline.")

else:
    st.markdown("""
    <div style="background:#0f1a2a; border:1px dashed #2a4a6a; border-radius:14px; padding:40px; text-align:center; color:#6a8aaa;">
      <div style="font-size:3rem;">📁</div>
      <div style="font-size:1.1rem; margin-top:12px;">Upload a video above to get started</div>
      <div style="font-size:0.85rem; margin-top:8px;">Supports MP4, AVI, MOV, MKV · Up to 500 MB</div>
    </div>
    """, unsafe_allow_html=True)
