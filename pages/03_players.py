"""
Streamlit Page 3 — Player Statistics
"""
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Players | AI Sports Analyser", page_icon="👥", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0d0d1a; }
.player-card {
    background: #12121f; border: 1px solid #2a2a4a; border-radius: 12px; padding: 20px; margin-bottom: 12px;
}
.score-badge {
    display: inline-block; background: linear-gradient(135deg, #0f3460, #1a237e);
    border: 2px solid #00c8ff; border-radius: 50px; padding: 6px 18px;
    color: #00c8ff; font-size: 1.4rem; font-weight: 700;
}
.grade-badge {
    display: inline-block; background: #1a1a2e; border: 2px solid #7b2ff7;
    border-radius: 50px; padding: 6px 18px;
    color: #c080ff; font-size: 1.2rem; font-weight: 700; margin-left: 12px;
}
.strength-tag {
    display: inline-block; background: #0f2a1a; border: 1px solid #2a6a3a;
    border-radius: 20px; padding: 4px 12px; color: #80ff80; font-size: 0.8rem; margin: 3px;
}
.weakness-tag {
    display: inline-block; background: #2a1a0f; border: 1px solid #6a3a2a;
    border-radius: 20px; padding: 4px 12px; color: #ffb080; font-size: 0.8rem; margin: 3px;
}
</style>
""", unsafe_allow_html=True)

st.title("👥 Player Statistics")

if "analysis_result" not in st.session_state:
    st.warning("⚠️ No analysis data. Run analysis on the **🔬 Analysis** page first.")
    st.stop()

res = st.session_state["analysis_result"]
player_ids = res.get_player_ids()

if not player_ids:
    st.info("No players were tracked in this video.")
    st.stop()

# ─── Player Selector ──────────────────────────────────────────────────────────
selected_pid = st.selectbox(
    "Select Player", player_ids, format_func=lambda x: f"Player {x}"
)

perf   = res.performance.get(selected_pid)
motion = res.motion_metrics.get(selected_pid)
poses  = res.pose_data.get(selected_pid, [])

if perf and motion:
    # ─── Score Header ─────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="player-card">
      <div style="margin-bottom:14px;">
        <span class="score-badge">{perf.overall_score} / 100</span>
        <span class="grade-badge">Grade {perf.grade}</span>
        <span style="color:#6a8aaa; font-size:0.8rem; margin-left:14px;">
          AI-generated prototype indicator — not scientifically validated
        </span>
      </div>
    """, unsafe_allow_html=True)

    # Score dimensions
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.metric("Movement",    f"{perf.movement_score:.0f}")
    sc2.metric("Activity",    f"{perf.activity_score:.0f}")
    sc3.metric("Coverage",    f"{perf.coverage_score:.0f}")
    sc4.metric("Consistency", f"{perf.consistency_score:.0f}")
    sc5.metric("Posture",     f"{perf.posture_score:.0f}")
    st.markdown("</div>", unsafe_allow_html=True)

    # ─── Motion Metrics Grid ──────────────────────────────────────────────
    st.subheader("📐 Kinematic Metrics")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Est. Distance (Estimate)", f"{motion.est_distance_m} m")
        st.metric("Total Distance (px)", f"{motion.total_distance_px:.0f} px")
    with c2:
        st.metric("Est. Avg Speed (Estimate)", f"{motion.est_avg_speed_kmh} km/h")
        st.metric("Est. Peak Speed (Estimate)", f"{motion.est_max_speed_kmh} km/h")
    with c3:
        st.metric("Active Duration", f"{motion.active_duration_s:.1f} s")
        st.metric("Stationary Duration", f"{motion.stationary_duration_s:.1f} s")

    cm1, cm2, cm3 = st.columns(3)
    cm1.metric("Movement Bursts", motion.movement_count)
    cm2.metric("Direction Changes", motion.direction_changes)
    cm3.metric("Poses Analysed", len([p for p in poses if p.pose_detected]))

    # ─── Strengths & Weaknesses ───────────────────────────────────────────
    st.divider()
    col_s, col_w = st.columns(2)
    with col_s:
        st.markdown("**✅ Strengths**")
        if perf.strengths:
            tags = " ".join(f'<span class="strength-tag">✔ {s}</span>' for s in perf.strengths)
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.caption("No specific strengths identified.")

    with col_w:
        st.markdown("**⚠️ Areas for Improvement**")
        if perf.weaknesses:
            tags = " ".join(f'<span class="weakness-tag">⚡ {w}</span>' for w in perf.weaknesses)
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.caption("No specific weaknesses identified.")

# ─── Multi-Player Table ───────────────────────────────────────────────────────
st.divider()
st.subheader("📊 All Players Comparison")
rows = []
for pid in player_ids:
    p = res.performance.get(pid)
    m = res.motion_metrics.get(pid)
    if p and m:
        rows.append({
            "Player":              f"Player {pid}",
            "Score":               f"{p.overall_score:.0f}",
            "Grade":               p.grade,
            "Est. Distance (m)":   m.est_distance_m,
            "Peak Speed (km/h)":   m.est_max_speed_kmh,
            "Active (s)":          m.active_duration_s,
            "Stationary (s)":      m.stationary_duration_s,
            "Dir. Changes":        m.direction_changes,
        })

if rows:
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
