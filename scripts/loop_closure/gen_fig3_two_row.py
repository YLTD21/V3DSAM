#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig3 判别器对比 —— 两行：上=TUM-RGBD (fr1)，下=ScanNet。

每行 5 个判别器：(a) αmatch (b) 外观 (c) 类别重叠 (d) 几何 (e) 语义-几何三维一致性(本文)。
红=假回环，绿=真回环。αmatch = image_match_ratio（VGGT-SLAM 2.0 官方回环判据）。
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/root/autodl-tmp"
SEQ = ["room", "360", "desk2", "floor", "plant", "rpy", "teddy"]
SCENES = ["scene0000_00", "scene0059_00", "scene0106_00", "scene0207_00"]

RED, GREEN = "#d62728", "#2ca02c"

# ---------------- TUM (row 0) ----------------
gt = json.load(open(f"{BASE}/gt_loop_pose.json"))
label = {(e["seq"], e["loop"]): (e["paper"] == "假") for e in gt}
tum = []
for s in SEQ:
    alpha = json.load(open(f"{BASE}/semantic_out/{s}/loop_log_alpha.json"))["loops"]
    cons = json.load(open(f"{BASE}/semantic_out/{s}/loop_log_3d.json"))["loops"]
    geo = json.load(open(f"{BASE}/semantic_out/{s}/loop_log_geo.json"))["loops"]
    base = json.load(open(f"{BASE}/semantic_out/{s}/loop_log.json"))["loops"]
    for k in range(len(alpha)):
        cq = set(base[k]["classes_q"]); cd = set(base[k]["classes_d"])
        jac = len(cq & cd) / len(cq | cd) if (cq | cd) else 0.0
        tum.append({"is_false": label[(s, k)], "alpha": alpha[k]["alpha"],
                    "salad": alpha[k]["score"], "jaccard": jac,
                    "geo": geo[k]["geo"], "cons": cons[k]["cons"]})

# ---------------- ScanNet (row 1) ----------------
scannet = []
for sc in SCENES:
    d = json.load(open(f"{BASE}/ablation/scannet/{sc}/loop_log.json"))
    a = json.load(open(f"{BASE}/ablation/scannet/{sc}/loop_log_alpha.json"))["loops"]
    amap = {(x["query_sid"], x["query_fi"], x["detected_sid"], x["detected_fi"]): x["alpha"]
            for x in a}
    amap_lc = {x["lc_num"]: x["alpha"] for x in a if x.get("lc_num") is not None}
    for l in d["loops"]:
        key = (l["query_sid"], l["query_fi"], l["detected_sid"], l["detected_fi"])
        am = amap.get(key)
        if am is None and l.get("lc_num") in amap_lc:
            am = amap_lc[l["lc_num"]]
        scannet.append({"is_false": l["verdict"] == "reject",
                        "alpha": am,
                        "score": l["score"], "overlap": l["overlap"],
                        "resid": l["resid_med"], "cons": l["cons_avg"]})

print(f"TUM: {len(tum)} loops ({sum(r['is_false'] for r in tum)} false)")
print(f"ScanNet: {len(scannet)} loops ({sum(r['is_false'] for r in scannet)} false)")
miss = sum(1 for r in scannet if r["alpha"] is None)
print(f"ScanNet alpha missing: {miss}/{len(scannet)}")


def rng(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0, 0.0
    return float(arr.min()), float(arr.max())


def fmt_range(lo, hi, is_inf=False, is_empty=False):
    if is_empty:
        return "—"
    if is_inf:
        return "∞"
    return f"[{lo:.2f},{hi:.2f}]"


# 每个元素: (false_vals, true_vals, metric_subtitle, has_inf)
rows_data = [
    # TUM
    [(np.array([r["alpha"] for r in tum if r["is_false"]], dtype=float),
      np.array([r["alpha"] for r in tum if not r["is_false"]], dtype=float),
      "image match ratio  (αmatch)", False),
     (np.array([r["salad"] for r in tum if r["is_false"]]),
      np.array([r["salad"] for r in tum if not r["is_false"]]),
      "SALAD L2 distance", False),
     (np.array([r["jaccard"] for r in tum if r["is_false"]]),
      np.array([r["jaccard"] for r in tum if not r["is_false"]]),
      "class Jaccard", False),
     (np.array([r["geo"] for r in tum if r["is_false"]]),
      np.array([r["geo"] for r in tum if not r["is_false"]]),
      "depth residual", False),
     (np.array([r["cons"] for r in tum if r["is_false"]]),
      np.array([r["cons"] for r in tum if not r["is_false"]]),
      "3D consistency", False)],
    # ScanNet
    [(np.array([r["alpha"] for r in scannet if r["is_false"]], dtype=float),
      np.array([r["alpha"] for r in scannet if not r["is_false"]], dtype=float),
      "image match ratio  (αmatch)", False),
     (np.array([r["score"] for r in scannet if r["is_false"]]),
      np.array([r["score"] for r in scannet if not r["is_false"]]),
      "SALAD L2 distance", False),
     (np.array([r["overlap"] for r in scannet if r["is_false"]]),
      np.array([r["overlap"] for r in scannet if not r["is_false"]]),
      "class overlap", False),
     (np.array([r["resid"] for r in scannet if r["is_false"]]),
      np.array([r["resid"] for r in scannet if not r["is_false"]]),
      "3D residual (∞=no overlap)", True),
     (np.array([r["cons"] for r in scannet if r["is_false"]]),
      np.array([r["cons"] for r in scannet if not r["is_false"]]),
      "3D consistency", False)],
]

col_titles = ["(a) αmatch", "(b) Appearance", "(c) Category overlap",
              "(d) Geometry", "(e) Semantic-geom. 3D (ours)"]
row_titles = ["TUM-RGBD  (fr1)", "ScanNet"]

fig, axes = plt.subplots(2, 5, figsize=(18.5, 6.4), dpi=200)

for ri in range(2):
    for ci in range(5):
        ax = axes[ri, ci]
        f, t, metric, has_inf = rows_data[ri][ci]

        # cap inf (ScanNet geometry false loops) at a value above all finite
        f_plot = f.astype(np.float64).copy()
        t_plot = t.astype(np.float64).copy()
        cap = None
        if has_inf:
            finite = np.concatenate([f_plot[np.isfinite(f_plot)],
                                     t_plot[np.isfinite(t_plot)]])
            cap = finite.max() * 1.3 if len(finite) else 1.0
            f_plot = np.where(np.isfinite(f_plot), f_plot, cap)
            t_plot = np.where(np.isfinite(t_plot), t_plot, cap)

        for v in f_plot[np.isfinite(f_plot)]:
            ax.scatter(0, v, s=55, c=RED, edgecolors="k", linewidths=0.5, zorder=3)
        for v in t_plot[np.isfinite(t_plot)]:
            ax.scatter(1, v, s=55, c=GREEN, edgecolors="k", linewidths=0.5, zorder=3)

        lo_f, hi_f = rng(f)
        lo_t, hi_t = rng(t)
        f_has_inf = has_inf and bool(np.any(~np.isfinite(f)))
        f_empty = len(f[np.isfinite(f)]) == 0
        ax.annotate(f"false  {fmt_range(lo_f, hi_f, f_has_inf, f_empty)}",
                    xy=(0, 1.0), xytext=(0.04, 1.03),
                    textcoords="axes fraction", fontsize=8, color=RED, va="bottom")
        ax.annotate(f"true  {fmt_range(lo_t, hi_t, False, len(t) == 0)}",
                    xy=(1, 1.0), xytext=(0.55, 1.03),
                    textcoords="axes fraction", fontsize=8, color=GREEN, va="bottom")

        ax.set_xticks([0, 1]); ax.set_xticklabels(["false", "true"])

        # ylim: 0..1 metrics fixed; others dynamic from plotted (finite) data
        finite_all = np.concatenate([f_plot[np.isfinite(f_plot)],
                                     t_plot[np.isfinite(t_plot)]])
        if ci in (2, 4):  # category overlap, consistency -> 0..1
            ax.set_ylim(-0.04, 1.08)
        elif ci == 0:  # αmatch: dynamic, but keep a floor so -0.95 context is implied
            lo = finite_all.min() - 0.12 if len(finite_all) else -1.0
            hi = finite_all.max() + 0.08 if len(finite_all) else 1.0
            ax.set_ylim(lo, hi)
        else:  # appearance L2 / geometry residual -> dynamic
            lo = finite_all.min() - 0.08 if len(finite_all) else 0.0
            hi = finite_all.max() + 0.08 if len(finite_all) else 1.0
            if ci == 3 and ri == 1:  # ScanNet geometry: cap already applied
                hi = cap + 0.08 if cap else hi
            ax.set_ylim(lo, hi)

        ax.grid(alpha=0.25, lw=0.4, axis="y")
        ax.text(0.5, -0.22, metric, ha="center", va="top",
                transform=ax.transAxes, fontsize=8.5, color="#555555", style="italic")

        if ri == 0:
            ax.set_title(col_titles[ci], fontsize=10)
        if ci == 0:
            ax.set_ylabel(row_titles[ri], fontsize=10)

fig.suptitle("Loop-candidate discriminator scores  (red = false, green = true)  —  "
             "TUM-RGBD (top) vs ScanNet (bottom)",
             fontsize=12, y=1.03)
fig.tight_layout()
fig.savefig(f"{BASE}/figures_sci/fig3_discriminator_honest.png", bbox_inches="tight")
fig.savefig(f"{BASE}/figures_sci/fig3_discriminator_honest.svg", bbox_inches="tight")
plt.close(fig)
print("[saved] figures_sci/fig3_discriminator_honest.png / .svg")
print("DONE")
