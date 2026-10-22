"""
AI Sports Video Analyser — Home Page
"""
import streamlit as st
import torch
from sports.config import list_sports

st.set_page_config(
    page_title="AI Sports Video Analyser",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
[data-testid="stSidebar"] {
    background: #0d0d1a;
    border-right: 1px solid #1e1e3a;
}
/* Hero gradient banner */
.hero-banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 50%, #0f3460 100%);
    border: 1px solid #1a3a5c;
    border-radius: 16px;
    padding: 40px 36px;
    margin-bottom: 28px;
    text-align: center;
}
.hero-banner h1 { font-size: 2.6rem; color: #00c8ff; margin: 0 0 10px 0; }
.hero-banner p  { font-size: 1.1rem; color: #aeccd8; margin: 0; }

/* Feature cards */
.feature-card {
    background: #12121f;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 20px;
    height: 100%;
    transition: border-color 0.2s;
}
.feature-card:hover { border-color: #00c8ff; }
.feature-card h4 { color: #00c8ff; margin: 0 0 8px 0; font-size: 1rem; }
.feature-card p  { color: #9999bb; font-size: 0.85rem; margin: 0; line-height: 1.5; }

/* Sport pill */
.sport-pill {
    display: inline-block;
    background: #1a1a2e;
    border: 1px solid #3a3a6a;
    border-radius: 40px;
    padding: 10px 20px;
    text-align: center;
    font-size: 1rem;
    color: #cce0ff;
}
.sport-pill span { display: block; font-size: 1.6rem; margin-bottom: 4px; }

/* Pipeline step */
.pipeline-step {
    background: #0f1a2a;
    border-left: 3px solid #00c8ff;
    border-radius: 0 8px 8px 0;
    padding: 8px 14px;
    margin: 4px 0;
    font-size: 0.85rem;
    color: #b0c8e0;
}

/* Sidebar nav hint */
.sidebar-hint {
    background: #1a2a1a;
    border: 1px solid #2a4a2a;
    border-radius: 8px;
    padding: 10px 14px;
    color: #90ee90;
    font-size: 0.82rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Hero Banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>🏅 AI Sports Video Analyser</h1>
  <p>Computer Vision · Deep Learning · Pose Estimation · Gemini AI Coaching</p>
</div>
""", unsafe_allow_html=True)

# ─── Quick Status ─────────────────────────────────────────────────────────────
device = "🚀 CUDA GPU" if torch.cuda.is_available() else "💻 CPU"
c0, c1, c2, c3, c4 = st.columns(5)
c0.metric("Inference Device", device)
c1.metric("Supported Sports", "5")
c2.metric("Pipeline Stages", "7")
c3.metric("Detection Model", "YOLOv8n")
c4.metric("Pose Model", "YOLOv8n-Pose")

st.divider()

# ─── Feature Cards ────────────────────────────────────────────────────────────
st.subheader("📦 Platform Features")

cols = st.columns(3)
features = [
    ("🎯 Player Detection", "YOLOv8 person detection with NMS deduplication. High-confidence bounding boxes, no duplicate player boxes."),
    ("🔗 Persistent Tracking", "ByteTrack multi-object tracking with IoU matching. Stable player IDs across the entire video."),
    ("🦴 Pose Estimation", "YOLOv8-Pose 17-keypoint COCO skeleton overlay. Joint angle computation and posture analysis."),
    ("⚡ Activity Events", "Rule-based sport-specific event detection: sprints, ready stances, direction changes, activity bursts."),
    ("📊 Performance Score", "Prototype 0–100 indicator with movement, activity, coverage, consistency and posture sub-scores."),
    ("🤖 AI Feedback", "Gemini API personalised coaching feedback with training suggestions. Template fallback when offline."),
    ("🔥 Heatmaps", "Gaussian-blurred position density heatmaps showing where players spent time on the field/court."),
    ("📈 Statistics Dashboard", "Interactive Plotly charts: speed timelines, trajectory maps, radar comparisons."),
    ("📄 PDF Report", "Downloadable FPDF2 PDF report with all metrics, grades, strengths/weaknesses, and AI advice."),
]

for i, (title, desc) in enumerate(features):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="feature-card">
          <h4>{title}</h4>
          <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")

st.divider()

# ─── Supported Sports ─────────────────────────────────────────────────────────
st.subheader("🏆 Supported Sports")
sports = list_sports()
sport_cols = st.columns(len(sports))
for idx, (key, name, icon) in enumerate(sports):
    with sport_cols[idx]:
        st.markdown(f"""
        <div class="sport-pill">
          <span>{icon}</span>
          <b>{name}</b>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ─── Analysis Pipeline ────────────────────────────────────────────────────────
st.subheader("🔄 Analysis Pipeline")
pipeline_steps = [
    "1️⃣  Upload & Validate Video (MP4 / AVI / MOV)",
    "2️⃣  Video Preprocessing — Frame extraction, FPS sampling, resolution resize",
    "3️⃣  Player Detection — YOLOv8 with NMS deduplication (no false doubles)",
    "4️⃣  Player Tracking — ByteTrack persistent ID assignment across frames",
    "5️⃣  Pose Estimation — YOLOv8-Pose 17 keypoints, joint angles, posture notes",
    "6️⃣  Motion Analysis — Distance, speed (estimated), heatmaps, trajectories",
    "7️⃣  Event Detection — Sport-specific rule-based activity classification",
    "8️⃣  Performance Evaluation — Prototype score (0–100), Grade (A+–E)",
    "9️⃣  AI Feedback — Gemini API personalised coaching (template fallback)",
    "🔟  Reports — PDF, CSV, annotated video download",
]
left_col, right_col = st.columns(2)
for i, step in enumerate(pipeline_steps):
    (left_col if i % 2 == 0 else right_col).markdown(f'<div class="pipeline-step">{step}</div>', unsafe_allow_html=True)

st.divider()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗺️ Navigation")
    st.markdown("""
    1. **📤 Upload Video** — Upload & configure
    2. **🔬 Analysis** — Run the pipeline
    3. **👥 Players** — Per-player stats
    4. **🏃 Movement** — Trajectories & heatmaps
    5. **⚡ Events** — Activity timeline
    6. **📊 Performance** — Scores & radar
    7. **🤖 Feedback** — AI coaching
    8. **📋 Report** — Download PDF/CSV
    """)
    st.divider()
    st.markdown('<div class="sidebar-hint">💡 Start at <b>Upload Video</b> to begin a new analysis session.</div>', unsafe_allow_html=True)
    st.divider()
    if st.button("🔄 Clear Session Data", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.caption("v1.0.0 · AI Sports Video Analyser")
