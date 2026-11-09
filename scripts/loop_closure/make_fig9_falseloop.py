#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone: regenerate 图9 (fig6_false_loop.png) with larger panels and labels.

Reads local loop data (paper/figures/loop_figures/semantic_out/room/loop_figures).
Does NOT touch v3dsam source. Same 2x2 layout, bigger canvas + bigger fonts.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
QD = os.path.join(HERE, "figures", "loop_figures", "semantic_out", "room", "loop_figures")
OUT = os.path.join(HERE, "figures", "fig6_false_loop.png")


def overlay_mask(rgb, cat_mask):
    H, W = cat_mask.shape[1], cat_mask.shape[2]
    cmap = plt.get_cmap("tab20")
    canvas = np.asarray(rgb, dtype=np.float32).copy()
    for c in range(cat_mask.shape[0]):
        m = cat_mask[c]
        if m.sum() < 500:
            continue
        col = np.array(cmap.colors[c % len(cmap.colors)])[:3] * 255
        canvas[m > 0] = 0.55 * canvas[m > 0] + 0.45 * col
    return np.clip(canvas, 0, 255).astype(np.uint8)


def main():
    q_img = Image.open(os.path.join(QD, "loop0_query.png")).convert("RGB")
    d_img = Image.open(os.path.join(QD, "loop0_detected.png")).convert("RGB")
    mz = np.load(os.path.join(QD, "loop0_masks.npz"))
    mq, md = mz["mask_q"], mz["mask_d"]

    # larger canvas + larger fonts (was 8.4 x 8.2, fs 9/10)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 11.0))
    panels = [
        (axes[0, 0], q_img, "Query frame (RGB)"),
        (axes[0, 1], d_img, "Detected frame (RGB)"),
        (axes[1, 0], Image.fromarray(overlay_mask(q_img, mq)), "Query + SAM3 categories"),
        (axes[1, 1], Image.fromarray(overlay_mask(d_img, md)), "Detected + SAM3 categories"),
    ]
    for ax, im, title in panels:
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(title, fontsize=18, fontweight="bold", pad=12)
    fig.suptitle("False loop closure (fr1/room #0): same categories, different viewpoint\n"
                 "category overlap (Jaccard) = 0.64   vs.   3D semantic consistency = 0.078",
                 fontsize=19, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("->", OUT, Image.open(OUT).size)


if __name__ == "__main__":
    main()
