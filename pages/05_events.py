"""
Streamlit Page 5 — Activity & Event Detection
"""
import streamlit as st
import pandas as pd
from visualization.charts import plot_event_timeline

st.set_page_config(page_title="Events | AI Sports Analyser", page_icon="⚡", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0d0d1a; }
.event-stat-card {
    background: #12121f; border: 1px solid #2a2a4a; border-radius: 10px;
    padding: 16px; text-align: center;
}
.event-stat-card h3 { color: #00c8ff; margin: 0; font-size: 1.8rem; }
.event-stat-card p { color: #8a9ab0; margin: 4px 0 0 0; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Sports Activity & Event Detection")

if "analysis_result" not in st.session_state:
    st.warning("⚠️ No analysis data. Run analysis on the **🔬 Analysis** page first.")
    st.stop()

res = st.session_state["analysis_result"]
events = res.events

if not events:
    st.info("No activity events recorded for this session.")
    st.stop()

# ─── Event Stats Bar ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
unique_types = len(set(e.event_type for e in events))
with c1:
    st.markdown(f'<div class="event-stat-card"><h3>{len(events)}</h3><p>Total Events</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="event-stat-card"><h3>{unique_types}</h3><p>Unique Activity Types</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="event-stat-card"><h3>{res.get_player_count()}</h3><p>Participating Players</p></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="event-stat-card"><h3>{res.sport.capitalize()}</h3><p>Sport Config</p></div>', unsafe_allow_html=True)

st.divider()

# ─── Event Gantt Timeline ─────────────────────────────────────────────────────
st.subheader("⏱️ Activity Timeline")
fig_tl = plot_event_timeline(events, res.get_player_ids())
st.plotly_chart(fig_tl, use_container_width=True)

st.divider()

# ─── Filterable Event Log Table ───────────────────────────────────────────────
st.subheader("📋 Detected Activity Log")

pids = res.get_player_ids()
selected_pid_filter = st.selectbox("Filter by Player", [None] + pids, format_func=lambda x: "All Players" if x is None else f"Player {x}")

filtered_events = events
if selected_pid_filter is not None:
    filtered_events = [e for e in events if e.player_id == selected_pid_filter]

df_events = pd.DataFrame([
    {
        "Player": f"Player {e.player_id}",
        "Event Type": e.event_type.replace("_", " ").title(),
        "Start (s)": round(e.timestamp, 2),
        "Duration (s)": round(e.duration, 2),
        "Confidence": f"{int(e.confidence * 100)}%"
    }
    for e in filtered_events
])

st.dataframe(df_events, use_container_width=True)
