"""
Streamlit Page 4 — Movement, Trajectories & Heatmaps
"""
import streamlit as st
from visualization.charts import plot_player_speed_timeline, plot_activity_breakdown, plot_distance_comparison
from visualization.trajectory import plot_player_trajectories
from visualization.heatmap import generate_heatmap_figure

st.set_page_config(page_title="Movement | AI Sports Analyser", page_icon="🏃", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0d0d1a; }
.metric-disclaimer {
    background: #1a1a2e; border-left: 3px solid #00c8ff; border-radius: 0 8px 8px 0;
    padding: 8px 14px; font-size: 0.82rem; color: #8aa0c0; margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

st.title("🏃 Movement & Motion Analysis")

if "analysis_result" not in st.session_state:
    st.warning("⚠️ No analysis data. Run analysis on the **🔬 Analysis** page first.")
    st.stop()

res = st.session_state["analysis_result"]
player_ids = res.get_player_ids()

st.markdown("""
<div class="metric-disclaimer">
  ℹ️ <b>Kinematic Disclaimer:</b> Distance (metres) and Speed (km/h) are pixel-derived estimates based on field heuristic ratios. They are clearly labelled as estimates.
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Speed Timeline",
    "🗺️ Spatial Trajectories",
    "🔥 Position Heatmap",
    "📊 Distance & Work Rate"
])

with tab1:
    st.subheader("Player Speed Timeline")
    selected_p = st.selectbox(
        "Filter Player", [None] + player_ids,
        format_func=lambda x: "All Tracked Players" if x is None else f"Player {x}"
    )
    fig_speed = plot_player_speed_timeline(res.motion_metrics, selected_p)
    st.plotly_chart(fig_speed, use_container_width=True)

with tab2:
    st.subheader("2D Spatial Trajectory Path")
    st.caption("Traces player path over the video frame coordinates. Green dot = start, Red dot = end.")
    fig_traj = plot_player_trajectories(res.tracks, res.video_info)
    st.plotly_chart(fig_traj, use_container_width=True)

with tab3:
    st.subheader("Position Density Heatmap")
    h_player = st.selectbox(
        "Select Heatmap Target", [None] + player_ids,
        format_func=lambda x: "Combined All Players" if x is None else f"Player {x}"
    )
    if res.video_info:
        fig_heat = generate_heatmap_figure(res.motion_metrics, res.video_info, h_player)
        st.pyplot(fig_heat)

with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Estimated Distance Comparison")
        st.plotly_chart(plot_distance_comparison(res.motion_metrics), use_container_width=True)
    with col2:
        st.subheader("Active vs. Stationary Time")
        st.plotly_chart(plot_activity_breakdown(res.motion_metrics), use_container_width=True)
