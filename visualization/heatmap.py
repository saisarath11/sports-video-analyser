"""
Heatmap visualization generator using Matplotlib.
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import Dict, Optional
from matplotlib.figure import Figure
from core.models import MotionMetrics, VideoInfo


def generate_heatmap_figure(motion_metrics: Dict[int, MotionMetrics], video_info: VideoInfo, player_id: Optional[int] = None) -> Figure:
    """Generate Matplotlib figure heatmap for player positions."""
    w, h = video_info.width, video_info.height
    heatmap = np.zeros((h, w), dtype=np.float32)

    targets = [player_id] if player_id and player_id in motion_metrics else list(motion_metrics.keys())

    for pid in targets:
        metrics = motion_metrics[pid]
        for x, y in metrics.positions:
            xi = int(np.clip(x, 0, w - 1))
            yi = int(np.clip(y, 0, h - 1))
            heatmap[yi, xi] += 1.0

    if np.max(heatmap) > 0:
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")

    im = ax.imshow(heatmap, cmap="hot", interpolation="nearest")
    ax.set_title(f"Movement Position Heatmap {'(Player ' + str(player_id) + ')' if player_id else '(All Players)'}", color="white")
    ax.set_xlabel("Video X (pixels)", color="white")
    ax.set_ylabel("Video Y (pixels)", color="white")
    ax.tick_params(colors="white")

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax, "yticklabels"), color="white")

    plt.tight_layout()
    return fig


def overlay_heatmap_on_frame(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay colorized heatmap onto video frame."""
    norm_heat = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    color_heat = cv2.applyColorMap(norm_heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 1.0 - alpha, color_heat, alpha, 0)
    return overlay
