#!/usr/bin/env python3
"""E4 figure: 3-way instance re-ID strategy comparison (geometry-first vs category-gate vs appearance)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/root/autodl-tmp"
seqs = ["room", "360"]
data = {}
for s in seqs:
    d = json.load(open(f"{BASE}/psnr_tum/{s}/semantic_instances/reid_appearance.json"))
    data[s] = d

strategies = [
    ("geometry-first\n(ours)", "recall_geometry_first", "#1f77b4"),
    ("category-gate", "recall_category_gate", "#ff7f0e"),
    ("appearance\n(color hist)", "recall_appearance", "#d62728"),
]

x = np.arange(len(seqs))
w = 0.26
fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=200)
for i, (name, key, c) in enumerate(strategies):
    vals = [data[s][key] for s in seqs]
    bars = ax.bar(x + (i - 1) * w, vals, w, color=c, label=name, edgecolor="k", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)

ax.set_xticks(x); ax.set_xticklabels(["room", "360"], fontsize=11)
ax.set_ylabel("Instance re-ID recall", fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_title("Instance association: geometry-first vs appearance", fontsize=11)
ax.grid(alpha=0.25, lw=0.5, axis="y")
ax.legend(frameon=False, fontsize=9, loc="upper right")
fig.tight_layout()
import os
os.makedirs(f"{BASE}/figures_sci", exist_ok=True)
fig.savefig(f"{BASE}/figures_sci/figS3_reid_appearance.png", bbox_inches="tight")
fig.savefig(f"{BASE}/figures_sci/figS3_reid_appearance.svg", bbox_inches="tight")
print("[saved] figures_sci/figS3_reid_appearance.png/.svg")

# markdown table
hdr = "| 序列 | 几何优先 (本文) | 类别硬门控 | 外观匹配 (颜色直方图) |"
sep = "|---|---|---|---|"
print(hdr); print(sep)
for s in seqs:
    d = data[s]
    n = d["n_tracked"]
    g = int(round(d["recall_geometry_first"] * n))
    c = int(round(d["recall_category_gate"] * n))
    a = int(round(d["recall_appearance"] * n))
    print(f"| {s} | {d['recall_geometry_first']:.3f} ({g}/{n}) "
          f"| {d['recall_category_gate']:.3f} ({c}/{n}) "
          f"| **{d['recall_appearance']:.3f}** ({a}/{n}) |")
print("E4 DONE")
