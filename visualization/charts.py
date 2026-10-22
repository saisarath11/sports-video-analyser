"""
Plotly interactive chart visualizers for Streamlit dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List, Optional
from core.models import MotionMetrics, PerformanceIndicators, SportEvent


def plot_player_speed_timeline(motion_metrics: Dict[int, MotionMetrics], selected_player: Optional[int] = None) -> go.Figure:
    """Line chart showing player speed timeline over video timestamps."""
    fig = go.Figure()
    
    players_to_plot = [selected_player] if selected_player and selected_player in motion_metrics else list(motion_metrics.keys())
    
    for pid in players_to_plot:
        metrics = motion_metrics[pid]
        if not metrics.speed_timeline:
            continue
        
        ts = [t[0] for t in metrics.speed_timeline]
        spds = [t[1] for t in metrics.speed_timeline]
        
        fig.add_trace(go.Scatter(
            x=ts, y=spds,
            mode="lines",
            name=f"Player {pid}"
        ))

    fig.update_layout(
        title="Player Speed Timeline (pixels/sec)",
        xaxis_title="Time (seconds)",
        yaxis_title="Speed (px/s)",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_activity_breakdown(motion_metrics: Dict[int, MotionMetrics]) -> go.Figure:
    """Grouped bar chart showing active vs stationary time per player."""
    pids = list(motion_metrics.keys())
    active_times = [motion_metrics[p].active_duration_s for p in pids]
    stationary_times = [motion_metrics[p].stationary_duration_s for p in pids]

    fig = go.Figure(data=[
        go.Bar(name="Active Time (s)", x=[f"Player {p}" for p in pids], y=active_times, marker_color="#4CAF50"),
        go.Bar(name="Stationary Time (s)", x=[f"Player {p}" for p in pids], y=stationary_times, marker_color="#FF9800")
    ])

    fig.update_layout(
        barmode="group",
        title="Active vs Stationary Duration",
        yaxis_title="Time (seconds)",
        template="plotly_dark"
    )
    return fig


def plot_performance_radar(performance: Dict[int, PerformanceIndicators], player_ids: List[int]) -> go.Figure:
    """Radar chart comparing player performance dimensions."""
    categories = ["Movement", "Activity", "Coverage", "Consistency", "Posture"]
    fig = go.Figure()

    for pid in player_ids:
        if pid not in performance:
            continue
        p = performance[pid]
        values = [p.movement_score, p.activity_score, p.coverage_score, p.consistency_score, p.posture_score]
        values.append(values[0])  # Close loop

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            name=f"Player {pid}"
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Performance Indicators Radar Comparison",
        template="plotly_dark"
    )
    return fig


def plot_performance_bars(performance: Dict[int, PerformanceIndicators]) -> go.Figure:
    """Bar chart showing overall prototype scores across players."""
    pids = sorted(performance.keys())
    scores = [performance[p].overall_score for p in pids]
    grades = [performance[p].grade for p in pids]

    fig = go.Figure(data=[
        go.Bar(
            x=[f"Player {p} ({g})" for p, g in zip(pids, grades)],
            y=scores,
            marker_color="#2196F3",
            text=scores,
            textposition="auto"
        )
    ])

    fig.update_layout(
        title="Overall Prototype Performance Score (0-100)",
        yaxis_title="Score",
        yaxis=dict(range=[0, 100]),
        template="plotly_dark"
    )
    return fig


def plot_event_timeline(events: List[SportEvent], player_ids: List[int]) -> go.Figure:
    """Timeline chart showing detected activity events."""
    filtered = [e for e in events if e.player_id in player_ids] if player_ids else events
    if not filtered:
        fig = go.Figure()
        fig.update_layout(title="No Events Detected", template="plotly_dark")
        return fig

    df = pd.DataFrame([
        {
            "Player": f"Player {e.player_id}",
            "Start": e.timestamp,
            "Finish": e.timestamp + max(e.duration, 0.5),
            "Event": e.event_type
        }
        for e in filtered
    ])

    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Player", color="Event", title="Sports Activity Event Timeline")
    fig.update_layout(template="plotly_dark", xaxis_title="Timestamp (seconds)")
    return fig


def plot_distance_comparison(motion_metrics: Dict[int, MotionMetrics]) -> go.Figure:
    """Bar chart comparing estimated distances."""
    pids = sorted(motion_metrics.keys())
    dists = [motion_metrics[p].est_distance_m for p in pids]

    fig = go.Figure(data=[
        go.Bar(
            x=[f"Player {p}" for p in pids],
            y=dists,
            marker_color="#00BCD4",
            text=[f"{d} m" for d in dists],
            textposition="auto"
        )
    ])

    fig.update_layout(
        title="Estimated Distance Covered (Metres - Estimate)",
        yaxis_title="Distance (m)",
        template="plotly_dark"
    )
    return fig
