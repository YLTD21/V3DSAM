#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Redraw Fig2 (Jaccard motivation) and Fig6 (centroid displacement) honestly.

Fig2 uses the set-level category Jaccard values already committed in Table 4
(= gen_honest_tables.py: |classes_q & classes_d| / |classes_q | classes_d|).
Fig6 uses id_propagate.json dmin (fr1/room, 76 instances), median 6.9cm.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"E:\0722v3dsam\paper"

# ---------- Fig 2: category-level Jaccard scatter (set-level, Table 4) ----------
# (label, seq#, jaccard, is_false)
cand = [
    ("room",  "#0", 0.640, True),
    ("room",  "#1", 0.895, False),
    ("room",  "#2", 1.000, False),
    ("360",   "#0", 0.357, True),
    ("360",   "#1", 0.450, True),
    ("360",   "#2", 0.562, False),
    ("360",   "#3", 0.636, True),
    ("desk2", "#0", 0.950, False),
    ("desk2", "#1", 0.850, False),
    ("floor", "#0", 0.750, False),
    ("plant", "#0", 0.905, False),
    ("plant", "#1", 0.741, False),
    ("rpy",   "#0", 0.842, False),
    ("teddy", "#0", 0.500, False),
    ("teddy", "#1", 0.630, False),
]
labels = [f"{s}{n}" for s, n, _, _ in cand]
jac = np.array([c[2] for c in cand])
isf = np.array([c[3] for c in cand])
x = np.arange(len(cand))

fig, ax = plt.subplots(figsize=(9.2, 3.8), dpi=200)
# true (green circle), false (red cross)
for i in range(len(cand)):
    if isf[i]:
        ax.scatter(x[i], jac[i], marker="x", s=90, c="#d62728",
                   linewidths=2.0, zorder=3)
    else:
        ax.scatter(x[i], jac[i], marker="o", s=64, c="#2ca02c",
                   edgecolors="k", linewidths=0.6, zorder=3)

# annotate the key interleaving example
i3603 = labels.index("360#3")
itedd = labels.index("teddy#0")
ax.annotate("false 360#3\n0.636", xy=(i3603, 0.636), xytext=(i3603 - 1.6, 0.70),
            fontsize=8.5, color="#d62728", arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8))
ax.annotate("true teddy#0\n0.500", xy=(itedd, 0.500), xytext=(itedd - 1.2, 0.34),
            fontsize=8.5, color="#2ca02c", arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.8))

# shade the overlap band [min true, max false] to show no clean threshold
lo_overlap = min(jac[~isf]); hi_overlap = max(jac[isf])
ax.axhspan(lo_overlap, hi_overlap, color="grey", alpha=0.14, zorder=1)
ax.text(14.6, (lo_overlap + hi_overlap) / 2, " overlap:\nno threshold\n separates",
        fontsize=8, color="#555", ha="left", va="center")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, fontsize=8)
ax.set_ylabel("category-level Jaccard overlap")
ax.set_ylim(0.0, 1.12)
ax.set_xlim(-0.6, 15.6)
ax.grid(alpha=0.25, lw=0.4, axis="y")
ax.set_title("Category-level Jaccard cannot separate false/true loops (red=false, green=true)",
             fontsize=10.5)
fig.tight_layout()
fig.savefig(f"{BASE}\\figs\\fig2_motivation_honest.png", bbox_inches="tight")
plt.close(fig)
print("[saved] figs/fig2_motivation_honest.png")

# ---------- Fig 6: centroid displacement histogram (fr1/room) ----------
d = json.load(open(f"{BASE}\\figures\\id_propagate.json", encoding="utf-8"))
dmin = np.array([s["dmin"] for s in d["stats"]])          # metres, 76 values
thr = 0.30
rec = dmin < thr                                         # recovered within threshold
med = float(np.median(dmin[rec]))                        # median over recovered (6.9cm)
n_rec = int(rec.sum())

fig, ax = plt.subplots(figsize=(7.8, 3.9), dpi=200)
bins = np.arange(0, 0.86, 0.05)
ax.hist(dmin[rec], bins=bins, color="#1f77b4", edgecolor="white", alpha=0.9,
        label=f"recovered (n={n_rec})", zorder=3)
ax.hist(dmin[~rec], bins=bins, color="#bbbbbb", edgecolor="white", alpha=0.9,
        label=f"not recovered (n={len(dmin)-n_rec})", zorder=3)
ax.axvline(med, color="#d62728", ls="--", lw=1.4, zorder=4)
ax.axvline(thr, color="#555555", ls=":", lw=1.4, zorder=4)
ymax = ax.get_ylim()[1]
ax.text(med + 0.015, ymax * 0.90, f"median {med*100:.1f}cm", color="#d62728",
        fontsize=9, va="top")
ax.text(thr + 0.015, ymax * 0.66, "match threshold 30cm", color="#333333",
        fontsize=9, va="top")
ax.set_xlabel("centroid displacement between two half-view subsets (m)")
ax.set_ylabel("instances")
ax.set_title(f"fr1/room: geometric-anchored centroid displacement "
             f"(n=76, {n_rec}/76 recovered, median {med*100:.1f}cm)")
ax.set_xlim(0, 0.85)
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.25, lw=0.4, axis="y")
fig.tight_layout()
fig.savefig(f"{BASE}\\figs\\fig6_centroid_honest.png", bbox_inches="tight")
plt.close(fig)
print(f"[saved] figs/fig6_centroid_honest.png  (median recovered={med:.4f} m, {n_rec}/76)")

print("DONE")
