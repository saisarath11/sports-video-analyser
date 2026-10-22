"""
Streamlit Page 8 — Reports & Data Export
"""
import streamlit as st
import os
import json
import pandas as pd
from pathlib import Path

from visualization.report import ReportGenerator
from utils.config import DATA_OUTPUT_DIR

st.set_page_config(page_title="Reports & Export | AI Sports Analyser", page_icon="📋", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0d0d1a; }
.export-card {
    background: #12121f; border: 1px solid #2a2a4a; border-radius: 12px;
    padding: 24px; text-align: center; height: 100%;
}
.export-card h3 { color: #00c8ff; margin: 0 0 8px 0; font-size: 1.2rem; }
.export-card p { color: #8a9ab0; font-size: 0.85rem; margin-bottom: 16px; }
</style>
""", unsafe_allow_html=True)

st.title("📋 Performance Reports & Data Export")

if "analysis_result" not in st.session_state:
    st.warning("⚠️ No analysis data. Run analysis on the **🔬 Analysis** page first.")
    st.stop()

res = st.session_state["analysis_result"]

st.caption("Generate PDF performance reports, download annotated videos, or export raw CSV/JSON data.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="export-card">
      <h3>📄 PDF Performance Report</h3>
      <p>Download a comprehensive, multi-page PDF report containing video metadata, player statistics, performance scores, strengths/weaknesses, and AI coaching recommendations.</p>
    """, unsafe_allow_html=True)

    pdf_path = os.path.join(DATA_OUTPUT_DIR, f"report_{res.session_id}.pdf")

    if st.button("🔨 Generate PDF Report", type="primary", use_container_width=True):
        with st.spinner("Generating PDF report..."):
            gen = ReportGenerator(res)
            gen.generate(pdf_path)
            res.report_path = pdf_path
            st.success("PDF report generated!")

    if res.report_path and os.path.exists(res.report_path):
        with open(res.report_path, "rb") as f:
            st.download_button(
                label="📥 Download PDF Report",
                data=f,
                file_name=f"sports_report_{res.session_id[:8]}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="export-card">
      <h3>📊 Raw Data Export (CSV)</h3>
      <p>Export player kinematics metrics, prototype scores, grades, active durations, and direction changes into CSV format for further analysis in Excel or Python.</p>
    """, unsafe_allow_html=True)

    rows = []
    for pid in res.get_player_ids():
        p = res.performance.get(pid)
        m = res.motion_metrics.get(pid)
        if p and m:
            rows.append({
                "Player_ID": pid,
                "Prototype_Score": p.overall_score,
                "Grade": p.grade,
                "Total_Distance_px": m.total_distance_px,
                "Est_Distance_m": m.est_distance_m,
                "Est_Avg_Speed_kmh": m.est_avg_speed_kmh,
                "Est_Peak_Speed_kmh": m.est_max_speed_kmh,
                "Active_Duration_s": m.active_duration_s,
                "Stationary_Duration_s": m.stationary_duration_s,
                "Movement_Count": m.movement_count,
                "Direction_Changes": m.direction_changes
            })

    if rows:
        df = pd.DataFrame(rows)
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV Statistics",
            data=csv_data,
            file_name=f"player_stats_{res.session_id[:8]}.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

if res.annotated_video_path and os.path.exists(res.annotated_video_path):
    st.subheader("📹 Annotated Video Download")
    st.caption("Download the output video with bounding boxes, player IDs, fading motion trails, pose skeletons, and HUD overlays.")
    with open(res.annotated_video_path, "rb") as f:
        st.download_button(
            label="📥 Download Annotated MP4 Video",
            data=f,
            file_name=f"annotated_video_{res.session_id[:8]}.mp4",
            mime="video/mp4"
        )
