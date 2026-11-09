#!/usr/bin/env python3
"""Regenerate fig1_architecture (图 1) to match the current "1 method + 1 system"
framing: highlight 贡献1 = semantic-geometric loop verification, 贡献2 =
gradient-aware opacity decay. Run locally (matplotlib) or on AutoDL."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from sci_style import apply_publication_style, PALETTE, finalize_figure

apply_publication_style(font_size=8, axes_linewidth=1)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "figures")
os.makedirs(OUT, exist_ok=True)

def fig_architecture():
    fig, ax = plt.subplots(figsize=(11.0, 3.6))
    ax.set_xlim(0, 11.0); ax.set_ylim(0, 3.6)
    ax.axis("off")

    # palette: neutral (soft blue) stages, warm accent for the two contributions
    C_STAGE = "#dbe9f6"      # light blue
    C_C1    = "#fde0d9"      # light orange (贡献1)
    C_C2    = "#fde0d9"      # light orange (贡献2)
    EC_C    = PALETTE["red_strong"]  # red edge for contributions

    def box(x, y, w, h, title, sub, fc, ec="#444444", lw=1.1, tag=None):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                           linewidth=lw, edgecolor=ec, facecolor=fc)
        ax.add_patch(p)
        ty = y + h * 0.66
        if tag:
            ax.text(x + w/2, y + h + 0.06, tag, ha="center", va="bottom",
                    fontsize=7.5, color=ec, weight="bold")
        ax.text(x + w/2, ty, title, ha="center", va="center", fontsize=9.3, weight="bold")
        ax.text(x + w/2, y + h * 0.28, sub, ha="center", va="center", fontsize=7.0, color="#333333")

    def arrow(x1, y1, x2, y2, label=None):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                            linewidth=1.2, color="#444444")
        ax.add_patch(a)
        if label:
            ax.text((x1+x2)/2, y1 + 0.26, label, ha="center", va="bottom",
                    fontsize=6.6, color="#555555")

    # main flow (y baseline ~1.55)
    box(0.15, 1.55, 2.05, 1.30, "Stage 1\nVGGT-SLAM\nfeed-forward front-end",
        "poses · dense depth ·\nloop candidates", C_STAGE)
    box(2.78, 1.55, 1.95, 1.30, "Stage 2\nSAM3 semantic\ngrounding",
        "per-frame category\nmasks", C_STAGE)
    box(5.30, 1.55, 2.35, 1.30, "Semantic-Geometric\nLoop Verification",
        "3D consistency (thr 0.10)\nreject false loops", C_C1, ec=EC_C, lw=1.6, tag="Contribution 1")
    box(8.20, 1.55, 2.65, 1.30, "Stage 3\nSplat-SLAM 3DGS\nreconstruction",
        "gaussian map from verified\nposes & depth", C_STAGE)

    arrow(2.22, 2.20, 2.76, 2.20)
    arrow(4.75, 2.20, 5.28, 2.20)
    arrow(7.67, 2.20, 8.18, 2.20)

    # 贡献2 = regularization applied during Stage 3 training
    box(8.30, 0.55, 2.45, 0.70, "Gradient-aware Opacity Decay",
        "anti-floater regularization", C_C2, ec=EC_C, lw=1.6, tag="Contribution 2")
    arrow(9.30, 1.27, 9.30, 1.53)

    # input / output (below the pipeline, no overlap)
    ax.text(1.17, 0.22, "RGB sequence", ha="center", fontsize=8, color="#222222")
    arrow(1.17, 0.42, 1.17, 1.52)
    ax.text(9.52, 0.22, "metric-scale\nphotorealistic map", ha="center", fontsize=8, color="#222222")

    finalize_figure(fig, os.path.join(OUT, "fig1_architecture"))
    print("fig1_architecture regenerated ->", os.path.join(OUT, "fig1_architecture.png"))

if __name__ == "__main__":
    fig_architecture()
