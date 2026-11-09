#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""4 张纯示意图（无需远端数据，本地可跑）:
  1) fig_venn.png               — 图1 三圈 Venn，三圈交汇 = V3DSAM
  2) fig_framework_v2.png       — 更详细的方法框架图
  3) fig_falseloop_method.png   — 假回环检测方法示意图
  4) fig_semconsistency_method.png — 语义-几何一致性方法示意图
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# ---------- CJK 字体 ----------
def _pick_cjk():
    cands = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimSun"]
    avail = {f.name for f in fm.fontManager.ttflist}
    for c in cands:
        if c in avail:
            return c
    return "DejaVu Sans"

CJK = _pick_cjk()
plt.rcParams["font.sans-serif"] = [CJK, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "dejavusans"

# 统一配色
C_GEO   = "#1f77b4"   # VGGT 几何 蓝
C_SEM   = "#2ca02c"   # SAM3 语义 绿
C_GSS   = "#9467bd"   # 3DGS 重建 紫
C_NOVEL = "#d62728"   # 创新点 红
C_LOOP  = "#ff7f0e"   # 回环 橙
C_GRAY  = "#4d4d4d"

def box(ax, x, y, w, h, fc, ec, lw=1.4, text="", tc="black", fs=11, weight="normal",
        radius=0.02, ls="-", alpha=1.0):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                       linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls, alpha=alpha,
                       mutation_aspect=1.0, zorder=2)
    ax.add_patch(b)
    if text:
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
                color=tc, fontweight=weight, zorder=3)
    return b

def txt(ax, x, y, s, fs=10, color="black", weight="normal", ha="center", va="center",
        zorder=4, rotation=0):
    ax.text(x, y, s, fontsize=fs, color=color, fontweight=weight, ha=ha, va=va, zorder=zorder,
            rotation=rotation)

def arr(ax, p0, p1, color=C_GRAY, lw=1.8, style="-|>", ls="-", ms=16, zorder=1):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color,
                        linewidth=lw, linestyle=ls, zorder=zorder, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a

def new_ax(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax

def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    try:
        fig.savefig(f"figures/{name}.pdf", bbox_inches="tight", facecolor="white")
    except PermissionError:
        print(f"-> (skip pdf, locked) figures/{name}.pdf")
    plt.close(fig)
    print(f"-> figures/{name}.png")

# ======================================================================
# 1) 三圈 Venn —— 方法定位
# ======================================================================
def venn():
    fig, ax = new_ax(10, 7)
    r = 0.26
    cA = (0.30, 0.40)   # VGGT 前馈几何
    cB = (0.70, 0.40)   # 3DGS 稠密重建
    cC = (0.50, 0.72)   # SAM3 语义

    for c, fc, ec in [(cA, C_GEO, C_GEO), (cB, C_GSS, C_GSS), (cC, C_SEM, C_SEM)]:
        ax.add_patch(Circle(c, r, facecolor=fc, edgecolor=ec, alpha=0.18, lw=2.2, zorder=1))
        ax.add_patch(Circle(c, r, facecolor="none", edgecolor=ec, alpha=0.9, lw=2.4, zorder=2))

    txt(ax, cA[0], cA[1] - 0.55, "VGGT Feed-Forward Geometry\n(monocular depth · intrinsics · poses · point map)",
        fs=12.5, color=C_GEO, weight="bold")
    txt(ax, cB[0], cB[1] - 0.55, "3DGS Dense Reconstruction\n(3D Gaussian map · differentiable rendering)",
        fs=12.5, color=C_GSS, weight="bold")
    txt(ax, cC[0], cC[1] + 0.24, "SAM3 Open-Vocab Semantics\n(pixel masks · category/instance identity)",
        fs=12.5, color=C_SEM, weight="bold")

    txt(ax, 0.50, 0.565, "V3DSAM", fs=19, color=C_NOVEL, weight="bold")
    txt(ax, 0.50, 0.502, "VGGT + SAM3 + 3DGS", fs=9.5, color=C_NOVEL)

    # 单一能力区（代表性系统名）
    txt(ax, 0.20, 0.285, "VGGT-SLAM\nVGGT-SLAM 2.0", fs=8.5, color=C_GEO, weight="bold")
    txt(ax, 0.80, 0.285, "MonoGS · Splat-SLAM\nGS-SLAM · SplaTAM", fs=8.5, color=C_GSS, weight="bold")
    txt(ax, 0.50, 0.825, "SAM3\n(open-vocab segmentation)", fs=8.5, color=C_SEM, weight="bold")

    # 两两交汇区（代表性系统名）
    txt(ax, 0.50, 0.33, "PF3plat · Flash-Mono\n(feed-forward 3DGS)", fs=8.5, color=C_GRAY, weight="bold")
    txt(ax, 0.335, 0.585, "Dinya et al.\n(VGGT + semantic identity)", fs=8.5, color=C_GRAY, weight="bold")
    txt(ax, 0.665, 0.585, "SGS-SLAM\nSemGauss-SLAM", fs=8.5, color=C_GRAY, weight="bold")

    ax.set_title("Positioning: V3DSAM is the only system unifying feed-forward geometry + open-vocabulary semantics + dense 3DGS",
                 fontsize=13.5, color=C_GRAY, pad=14)
    save(fig, "fig_venn")

# ======================================================================
# 2) 更详细的方法框架图
# ======================================================================
def framework():
    fig, ax = new_ax(15, 8.5)

    box(ax, 0.05, 0.915, 0.90, 0.06, "#f2f2f2", C_GRAY,
        text="Input: monocular RGB video sequence  $\\{I_1, I_2, \\dots, I_T\\}$", fs=13, tc=C_GRAY, weight="bold")

    # ---- Stage 1 (VGGT feed-forward) ----
    x1, y1, w1 = 0.05, 0.44, 0.27
    box(ax, x1, y1, w1, 0.40, "#eaf1fb", C_GEO, lw=2.0, text="① VGGT Feed-Forward Front-End", fs=13, weight="bold", tc=C_GEO)
    items1 = [("per-frame depth  $\\hat{D}_t$", "scale-ambiguous"),
              ("per-frame intrinsics  $\\hat{K}_t$", ""),
              ("per-frame poses  $\\hat{P}_t$", "no BA / no per-frame opt."),
              ("global-aligned point map", "global scale alignment")]
    yy = y1 + 0.29
    for main, sub in items1:
        box(ax, x1+0.025, yy, w1-0.05, 0.052, "white", C_GEO, lw=1.1, text=main, fs=10.5, tc="#222")
        if sub:
            txt(ax, x1+w1/2, yy-0.022, sub, fs=8.5, color=C_GRAY)
        yy -= 0.078
    box(ax, x1+0.025, y1+0.012, w1-0.05, 0.045, "#eaf1fb", C_GEO, lw=0, text="output: consistent poses + 3D point map", fs=9, tc=C_GEO)

    # ---- Stage 2 (SAM3 semantics) ----
    x2, y2, w2 = 0.365, 0.44, 0.27
    box(ax, x2, y2, w2, 0.40, "#e9f6e9", C_SEM, lw=2.0, text="② SAM3 Open-Vocab Semantics", fs=13, weight="bold", tc=C_SEM)
    items2 = [("per-pixel semantic masks", "open-vocab / no closed set"),
              ("multi-view category voting", "category identity (semantic)"),
              ("geometric centroid anchoring", "instance identity (geometric)"),
              ("persistent instance ID", "consistent across loops")]
    yy = y2 + 0.29
    for main, sub in items2:
        box(ax, x2+0.025, yy, w2-0.05, 0.052, "white", C_SEM, lw=1.1, text=main, fs=10.5, tc="#222")
        if sub:
            txt(ax, x2+w2/2, yy-0.022, sub, fs=8.5, color=C_GRAY)
        yy -= 0.078
    box(ax, x2+0.025, y2+0.012, w2-0.05, 0.045, "#e9f6e9", C_SEM, lw=0, text="output: semantic labels + instance identity", fs=9, tc=C_SEM)

    # ---- Stage 3 (3DGS) ----
    x3, y3, w3 = 0.68, 0.44, 0.27
    box(ax, x3, y3, w3, 0.40, "#f4eefa", C_GSS, lw=2.0, text="③ Splat-SLAM 3DGS", fs=13, weight="bold", tc=C_GSS)
    items3 = [("3D Gaussian map", "dense explicit repr."),
              ("differentiable rasterization", "novel-view synthesis"),
              ("tracking + mapping", "Gaussian add / remove"),
              ("photometric / geometric loss", "RGB-D alignment")]
    yy = y3 + 0.29
    for main, sub in items3:
        box(ax, x3+0.025, yy, w3-0.05, 0.052, "white", C_GSS, lw=1.1, text=main, fs=10.5, tc="#222")
        if sub:
            txt(ax, x3+w3/2, yy-0.022, sub, fs=8.5, color=C_GRAY)
        yy -= 0.078
    box(ax, x3+0.025, y3+0.012, w3-0.05, 0.045, "#f4eefa", C_GSS, lw=0, text="output: dense 3DGS map + RGB rendering", fs=9, tc=C_GSS)

    # 阶段间箭头
    arr(ax, (x1+w1, 0.64), (x2-0.01, 0.64), color=C_GRAY, lw=2.0)
    arr(ax, (x2+w2, 0.64), (x3-0.01, 0.64), color=C_GRAY, lw=2.0)

    # ---- 底部：回环 + 语义-几何一致性验证（创新） ----
    xl, yl, wl = 0.05, 0.12, 0.90
    box(ax, xl, yl, wl, 0.20, "#fdf0e6", C_LOOP, lw=2.0, text="④ Pose-Graph Optimization + Semantic-Geometric Loop Verification (ours)", fs=13, weight="bold", tc=C_LOOP)
    box(ax, xl+0.02, yl+0.018, 0.36, 0.13, "white", C_LOOP, lw=1.1, text="loop candidate $(I_i, I_j)$\nhigh appearance similarity", fs=10, tc="#222")
    arr(ax, (xl+0.40, yl+0.083), (xl+0.45, yl+0.083), color=C_LOOP, lw=1.8)
    box(ax, xl+0.455, yl+0.018, 0.30, 0.13, "white", C_LOOP, lw=1.1, text="semantic-geometric 3D consistency\nper-pixel label consistency ≥ threshold", fs=10, tc="#222")
    arr(ax, (xl+0.77, yl+0.083), (xl+0.82, yl+0.083), color=C_LOOP, lw=1.8)
    box(ax, xl+0.83, yl+0.018, 0.14, 0.13, "#d62728", "white", lw=1.1, text="reject false loop\n(keep graph clean)", fs=9.5, tc="white")

    # 回环反馈箭头（从 ④ 回到 ③ / ①）
    arr(ax, (xl+0.90, yl+0.20), (x3+w3/2, y3+0.01), color=C_LOOP, lw=1.8, style="<|-|>", ls=(0, (4, 2)))

    # 语义标签从 ② 向下馈入回环验证
    arr(ax, (x2+w2/2, y2+0.01), (xl+0.60, yl+0.20), color=C_SEM, lw=1.8, style="-|>", ls=(0, (3, 2)))
    txt(ax, x2+w2/2+0.06, 0.34, "semantic labels", fs=9.5, color=C_SEM, rotation=38)

    # 位姿从 ① 向下馈入
    arr(ax, (x1+w1/2, y1+0.01), (xl+0.20, yl+0.20), color=C_GEO, lw=1.8, style="-|>", ls=(0, (3, 2)))
    txt(ax, x1+w1/2-0.10, 0.30, "poses / point map", fs=9.5, color=C_GEO, rotation=-30)

    save(fig, "fig_framework_v2")

# ======================================================================
# 3) 假回环检测方法示意图
# ======================================================================
def falseloop():
    fig, ax = new_ax(13, 6.5)

    box(ax, 0.03, 0.30, 0.16, 0.30, "#cfe2f3", C_GEO, lw=1.6, text="Frame $I_i$", fs=13, weight="bold", tc=C_GEO)
    box(ax, 0.03, 0.62, 0.16, 0.30, "#cfe2f3", C_GEO, lw=1.6, text="Frame $I_j$", fs=13, weight="bold", tc=C_GEO)
    txt(ax, 0.11, 0.50, "high appearance similarity\n(score > $\\tau_{app}$)", fs=10, color=C_GRAY, weight="bold")

    arr(ax, (0.19, 0.77), (0.34, 0.77), color=C_LOOP, lw=2.2)
    box(ax, 0.20, 0.70, 0.16, 0.14, "#fdf0e6", C_LOOP, lw=1.6, text="candidate relative pose  $T_{ij}$\n(from loop detection)", fs=10, tc=C_LOOP, weight="bold")

    arr(ax, (0.36, 0.77), (0.50, 0.77), color=C_NOVEL, lw=2.2)
    box(ax, 0.51, 0.36, 0.30, 0.52, "#fdecea", C_NOVEL, lw=2.2, text="Semantic-Geometric 3D Consistency\nVerification (ours, core)", fs=13, weight="bold", tc=C_NOVEL)
    items = [("① transform $I_i$ 3D points by $T_{ij}$ into $I_j$", ""),
             ("② reproject to pixels $q$ in $I_j$", ""),
             ("③ per-pixel compare SAM3 labels", "$L_i(p) =?\\; L_j(q)$"),
             ("④ consistency $c$ = matched-pixel fraction", "scale-invariant · robust to depth drift")]
    yy = 0.79
    for main, sub in items:
        txt(ax, 0.66, yy, main, fs=10.5, color="#222", ha="center", weight="bold")
        if sub:
            txt(ax, 0.66, yy-0.052, sub, fs=9, color=C_GRAY)
        yy -= 0.115

    arr(ax, (0.81, 0.62), (0.90, 0.62), color=C_GRAY, lw=2.0)
    box(ax, 0.90, 0.56, 0.08, 0.12, "#e9f6e9", C_SEM, lw=1.6, text="$c \\geq \\tau_c$\nACCEPT\n(true loop)", fs=9.5, tc="#1a7a1a", weight="bold")
    arr(ax, (0.81, 0.48), (0.90, 0.48), color=C_GRAY, lw=2.0)
    box(ax, 0.90, 0.36, 0.08, 0.12, "#fdecea", C_NOVEL, lw=1.6, text="$c < \\tau_c$\nREJECT\n(false loop)", fs=9.5, tc=C_NOVEL, weight="bold")

    txt(ax, 0.50, 0.16, "appearance-similar but 3D-semantic-inconsistent → false loop, removed from pose graph (prevents drift / misalignment)",
        fs=12, color=C_NOVEL, weight="bold")

    save(fig, "fig_falseloop_method")

# ======================================================================
# 4) 语义-几何一致性方法示意图
# ======================================================================
def semconsistency():
    fig, ax = new_ax(13, 6.5)

    box(ax, 0.03, 0.30, 0.22, 0.42, "#cfe2f3", C_GEO, lw=1.6, text="Frame $I_i$\n(depth $\\hat{D}_i$ + semantics $L_i$)", fs=12, weight="bold", tc=C_GEO)
    box(ax, 0.75, 0.30, 0.22, 0.42, "#cfe2f3", C_GEO, lw=1.6, text="Frame $I_j$\n(semantics $L_j$)", fs=12, weight="bold", tc=C_GEO)

    box(ax, 0.34, 0.30, 0.32, 0.42, "#f4f4f4", C_GRAY, lw=1.4, text="", fs=11)
    txt(ax, 0.50, 0.66, "① back-project  $p \\rightarrow$ 3D point  $X$", fs=11, weight="bold", color="#222")
    txt(ax, 0.50, 0.58, "$X = \\pi^{-1}(p, \\hat{D}_i)$", fs=12, color=C_GEO)
    txt(ax, 0.50, 0.50, "② transform  $X' = T_{ij} \\cdot X$", fs=11, weight="bold", color="#222")
    txt(ax, 0.50, 0.42, "③ project  $q = \\pi(X')$", fs=11, weight="bold", color="#222")
    txt(ax, 0.50, 0.34, "④ compare  $L_i(p)$ vs $L_j(q)$", fs=11, weight="bold", color=C_NOVEL)

    arr(ax, (0.25, 0.51), (0.34, 0.51), color=C_GRAY, lw=1.8)
    arr(ax, (0.66, 0.51), (0.75, 0.51), color=C_GRAY, lw=1.8)

    txt(ax, 0.13, 0.14, "pixel $p$: label = \"chair\"", fs=10, color=C_GEO)
    txt(ax, 0.87, 0.14, "pixel $q$: label = \"chair\" (consistent)", fs=10, color=C_SEM)
    txt(ax, 0.87, 0.06, "pixel $q'$: label = \"table\" (inconsistent)", fs=10, color=C_NOVEL)

    box(ax, 0.30, 0.04, 0.40, 0.16, "#fdecea", C_NOVEL, lw=1.8,
        text="$c = \\frac{|\\{p : L_i(p) = L_j(\\pi(T_{ij}\\,\\pi^{-1}(p)))\\}|}{|p|}$", fs=11, tc=C_NOVEL, weight="bold")

    txt(ax, 0.50, 0.88, "Key property: semantic labels are scale-invariant — monocular depth scale drift does not flip semantic categories",
        fs=11.5, color=C_NOVEL, weight="bold")

    save(fig, "fig_semconsistency_method")

if __name__ == "__main__":
    venn()
    framework()
    falseloop()
    semconsistency()
    print("done")
