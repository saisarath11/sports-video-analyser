"""
Streamlit Page 6 — Performance Evaluation Engine
"""
import streamlit as st
from visualization.charts import plot_performance_bars, plot_performance_radar

st.set_page_config(page_title="Performance | AI Sports Analyser", page_icon="📊", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0d0d1a; }
.disclaimer-banner {
    background: #1a1a2e; border: 1px solid #3a3a6a; border-radius: 10px;
    padding: 12px 18px; margin-bottom: 20px; color: #a0b8d0; font-size: 0.85rem;
}
.score-pill {
    background: #12121f; border: 1px solid #00c8ff; border-radius: 12px;
    padding: 16px; text-align: center;
}
.score-num { font-size: 2.2rem; font-weight: 700; color: #00c8ff; }
.score-lbl { font-size: 0.82rem; color: #8a9ab0; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Performance Evaluation Engine")

if "analysis_result" not in st.session_state:
    st.warning("⚠️ No analysis data. Run analysis on the **🔬 Analysis** page first.")
    st.stop()

res = st.session_state["analysis_result"]
performance = res.performance

if not performance:
    st.info("No performance evaluation data generated.")
    st.stop()

st.markdown("""
<div class="disclaimer-banner">
  ⚠️ <b>Important Notice:</b> Performance scores (0–100) are <b>AI-generated prototype indicators</b> extracted from video observations. They are intended for demonstration and baseline tracking — they are not scientifically or medically validated athletic scores.
</div>
""", unsafe_allow_html=True)

# ─── Score Overview ───────────────────────────────────────────────────────────
st.subheader("🏆 Prototype Score Overview")
st.plotly_chart(plot_performance_bars(performance), use_container_width=True)

st.divider()

# ─── Radar Comparison ─────────────────────────────────────────────────────────
st.subheader("🕸️ Performance Dimensions Radar")
pids = res.get_player_ids()
selected_pids = st.multiselect("Select Players to Compare", pids, default=pids[:min(3, len(pids))])

if selected_pids:
    st.plotly_chart(plot_performance_radar(performance, selected_pids), use_container_width=True)

st.divider()

# ─── Individual Player Breakdown ──────────────────────────────────────────────
st.subheader("🔍 Individual Player Dimension Breakdown")
sel_player = st.selectbox("Select Player", pids, format_func=lambda x: f"Player {x}")
p_info = performance.get(sel_player)

if p_info:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="score-pill"><div class="score-num">{p_info.movement_score:.0f}</div><div class="score-lbl">Movement</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="score-pill"><div class="score-num">{p_info.activity_score:.0f}</div><div class="score-lbl">Activity</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="score-pill"><div class="score-num">{p_info.coverage_score:.0f}</div><div class="score-lbl">Coverage</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="score-pill"><div class="score-num">{p_info.consistency_score:.0f}</div><div class="score-lbl">Consistency</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="score-pill"><div class="score-num">{p_info.posture_score:.0f}</div><div class="score-lbl">Posture</div></div>', unsafe_allow_html=True)

    st.write("")
    col_str, col_wk = st.columns(2)
    with col_str:
        st.markdown("##### ✅ Strengths")
        for s in p_info.strengths:
            st.success(f"✔ {s}")
    with col_wk:
        st.markdown("##### ⚠️ Areas for Improvement")
        for w in p_info.weaknesses:
            st.warning(f"⚡ {w}")
