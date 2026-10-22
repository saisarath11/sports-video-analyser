"""
Player movement trajectory scatter & line plot visualizers.
"""

import plotly.graph_objects as go
from typing import Dict, List, Optional
from core.models import PlayerTrack, VideoInfo


def plot_player_trajectories(tracks: Dict[int, PlayerTrack], video_info: Optional[VideoInfo] = None, selected_players: Optional[List[int]] = None) -> go.Figure:
    """Plot 2D spatial trajectory paths for tracked players."""
    fig = go.Figure()

    targets = selected_players if selected_players else list(tracks.keys())

    for pid in targets:
        if pid not in tracks:
            continue
        track = tracks[pid]
        if not track.centers:
            continue

        xs = [pt[0] for pt in track.centers]
        ys = [pt[1] for pt in track.centers]

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            marker=dict(size=4),
            name=f"Player {pid}"
        ))

        # Start/End markers
        fig.add_trace(go.Scatter(
            x=[xs[0]], y=[ys[0]],
            mode="markers",
            marker=dict(symbol="circle", size=10, color="green"),
            name=f"Player {pid} Start",
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[xs[-1]], y=[ys[-1]],
            mode="markers",
            marker=dict(symbol="square", size=10, color="red"),
            name=f"Player {pid} End",
            showlegend=False
        ))

    w = video_info.width if video_info else 1280
    h = video_info.height if video_info else 720

    fig.update_layout(
        title="Player Trajectories (Spatial Path)",
        xaxis=dict(title="X (pixels)", range=[0, w]),
        yaxis=dict(title="Y (pixels)", range=[h, 0]),  # Invert Y to match computer vision coords
        template="plotly_dark",
        height=550
    )
    return fig
