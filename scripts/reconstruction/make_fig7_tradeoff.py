#!/usr/bin/env python3
"""Regenerate 图 8 (paper fig8_tradeoff) — the opacity-decay hard tradeoff (honest negative result).
x = training-view PSNR (surface preservation), y = held-out NVS PSNR.
Shows that no config reaches the ideal top-right (high train AND high NVS):
0.995 is the only config that lifts NVS but collapses train; semantic gating
preserves surface but does not lift NVS. Run locally (matplotlib)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sci_style import (apply_publication_style, PALETTE, add_panel_label,
                       finalize_figure)

apply_publication_style(font_size=8, axes_linewidth=1)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "figures")
os.makedirs(OUT, exist_ok=True)

# (label, train PSNR, NVS PSNR, color, marker, highlight)
# Color roles: baseline=neutral, 0.995=red (train collapse), decay sweep=blue
# family, refine=teal, semantic-gated=green (surface preserved).
CONFIGS = [
    ("baseline (no decay)", 22.81, 9.21, PALETTE["neutral_mid"], "o", False),
    ("decay 0.995",         14.52, 10.08, PALETTE["red_strong"], "s", True),   # only NVS lift, train collapses
    ("decay 0.999",         21.19, 9.30, PALETTE["blue_secondary"], "o", False),
    ("decay 0.9995",        22.43, 9.28, PALETTE["blue_secondary"], "o", False),
    ("decay 0.9998",        22.40, 9.26, PALETTE["blue_secondary"], "o", False),
    ("refine 5000 (no decay)", 19.64, 9.36, PALETTE["teal"], "^", False),
    ("semantic-gated 0.995", 22.97, 9.29, PALETTE["green_3"], "D", True),  # surface preserved, NVS no lift
]

def main():
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for label, tr, nvs, color, marker, hl in CONFIGS:
        ax.scatter(tr, nvs, c=color, marker=marker, s=90 if not hl else 150,
                   zorder=3, edgecolors="black" if hl else "none",
                   linewidths=0.9 if hl else 0, label=label)

    # ideal (top-right) region is empty
    ax.axvspan(22.0, 23.5, color=PALETTE["green_3"], alpha=0.08, zorder=0)
    ax.axhspan(9.9, 10.6, color=PALETTE["green_3"], alpha=0.08, zorder=0)
    ax.text(22.72, 10.18, "ideal: high train\n+ high NVS\n(empty)", ha="center",
            fontsize=7.0, color=PALETTE["neutral_dark"], style="italic")

    # annotate the two key regimes
    ax.annotate("0.995: only NVS lift (+0.87 dB)\nbut train collapses (−8.3 dB)",
                xy=(14.52, 10.08), xytext=(15.2, 10.35),
                fontsize=7.4, arrowprops=dict(arrowstyle="->", color=PALETTE["red_strong"], lw=1.0))
    ax.annotate("semantic gating: surface preserved\n(train 22.97) but NVS no lift (9.29)",
                xy=(22.97, 9.29), xytext=(20.4, 9.55),
                fontsize=7.4, arrowprops=dict(arrowstyle="->", color=PALETTE["green_3"], lw=1.0))

    ax.set_xlim(14.0, 23.5)
    ax.set_ylim(8.9, 10.6)
    ax.set_xlabel("training-view PSNR (dB)  →  surface preservation")
    ax.set_ylabel("held-out NVS PSNR (dB)  →  novel-view quality")
    ax.set_title("Opacity-decay regularization: a hard trade-off, no config reaches top-right",
                 fontsize=9.5)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=7.0)

    finalize_figure(fig, os.path.join(OUT, "fig8_tradeoff"))
    print("fig8_tradeoff regenerated ->", os.path.join(OUT, "fig8_tradeoff.png"))

if __name__ == "__main__":
    main()
