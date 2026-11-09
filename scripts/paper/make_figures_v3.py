#!/usr/bin/env python3
"""V3DSAM paper figures (v3) — 15-loop expansion + SCI-level style + ATE.

Runs on AutoDL (sgs env), reading trajectories from /root/autodl-tmp.

Produces, into /root/autodl-tmp/figures_sci/ :
  fig2_motivation.{pdf,png}   category overlap fails (all 15 loops)
  fig4_discriminator.{pdf,png} 3-panel: appearance / category overlap / 3D
  fig5_trajectory.{pdf,png}    before/after trajectories + ATE annotation

The 15-loop data (appearance score, class-set Jaccard = |inter|/|union|, 3D
consistency "cons") is the authoritative output of analyze_semantics.py
(intersection/union over class sets present in loop{k}_masks.npz) +
loop_log_3d.json (semantic_3d_check.py), NOT hand-picked. Ground truth is
assigned by the 0.10 3D-consistency threshold (false < 0.10 < true), giving
exactly 4 false + 11 true, zero error.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sci_style import apply_publication_style, PALETTE, add_panel_label, finalize_figure

apply_publication_style(font_size=8, axes_linewidth=1)

OUT = "/root/autodl-tmp/figures_sci"
os.makedirs(OUT, exist_ok=True)

# ── full 15-loop data: (seq, lc_num, appearance_score, jacc, cons_3d, is_true) ──
# jacc = class-set Jaccard = |cls_q ∩ cls_d| / |cls_q ∪ cls_d| (analyze_semantics.py)
LOOPS = [
    ("room",  99,  0.8926, 0.6400, 0.0775, False),
    ("room",  167, 0.6458, 0.8947, 0.1523, True),
    ("room",  195, 0.6255, 1.0000, 0.4814, True),
    ("360",   99,  0.9471, 0.3571, 0.0000, False),
    ("360",   134, 0.8764, 0.4500, 0.0253, False),
    ("360",   169, 0.6131, 0.5625, 0.5456, True),
    ("360",   179, 0.8601, 0.6364, 0.0000, False),
    ("desk2", 99,  0.4892, 0.9500, 0.5042, True),
    ("desk2", 103, 0.5562, 0.8500, 0.4496, True),
    ("floor", 99,  0.5629, 0.7500, 0.7479, True),
    ("plant", 99,  0.5742, 0.9048, 0.5536, True),
    ("plant", 107, 0.6130, 0.7407, 0.4957, True),
    ("rpy",   97,  0.5621, 0.8421, 0.5316, True),
    ("teddy", 132, 0.7470, 0.5000, 0.6395, True),
    ("teddy", 152, 0.7065, 0.6296, 0.4012, True),
]

def seq_code(seq):
    return {"room": "rm", "360": "360", "desk2": "dk2", "floor": "flr",
            "plant": "pln", "rpy": "rpy", "teddy": "tdy"}[seq]

def label(i):
    s, lc, *_ = LOOPS[i]
    # within-sequence loop index (matches paper Table 4: room#0..2, 360#0..3, ...)
    idx = sum(1 for j in range(i) if LOOPS[j][0] == s)
    return f"{seq_code(s)}#{idx}"

# ── trajectory helpers ──
def load_traj_tt(path):
    d = np.loadtxt(path)
    return d[:, 0], d[:, 1:4]

def associate_traj(est_t, est_xyz, gt_t, gt_xyz):
    idx = np.searchsorted(gt_t, est_t)
    idx = np.clip(idx, 0, len(gt_t) - 1)
    left = np.clip(idx - 1, 0, len(gt_t) - 1)
    best = np.where(np.abs(gt_t[left] - est_t) < np.abs(gt_t[idx] - est_t), left, idx)
    return gt_xyz[best]

def umeyama_sim3(src, dst):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = min(len(src), len(dst))
    src, dst = src[:n], dst[:n]
    src_mu, dst_mu = src.mean(0), dst.mean(0)
    sc, dc = src - src_mu, dst - dst_mu
    cov = sc.T @ dc / n
    U, S, Vt = np.linalg.svd(cov)
    sign = np.sign(np.linalg.det(Vt.T @ U.T))
    Sd = np.diag([1.0, 1.0, sign])
    R = Vt.T @ Sd @ U.T
    scale = np.trace(np.diag(S) @ Sd) / np.trace(sc.T @ sc / n)
    t = dst_mu - scale * R @ src_mu
    return (scale * (R @ src.T)).T + t

def ate_rmse(est_aligned, gt_assoc):
    return float(np.sqrt(np.mean(np.sum((est_aligned - gt_assoc) ** 2, axis=1))))

# Official TUM ATE (SE3-aligned, evo) already stated in the paper text — the
# annotation uses these so the figure and prose agree exactly. The in-script
# Sim(3) umeyama values agree to <=6 mm (360: 0.188/0.049, room: 0.809/0.074).
ATE_PAPER = {"room": (0.815, 0.074), "360": (0.189, 0.049)}


# ════════════════════════════════════════════════════════════════════
# FIG 2 — motivation: category overlap fails (15 loops)
# ════════════════════════════════════════════════════════════════════
def fig_motivation():
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    xs = np.arange(len(LOOPS))
    # sort false/true for legibility: false on top of the range
    for i, l in enumerate(LOOPS):
        is_true = l[5]
        ax.scatter(xs[i], l[3], marker="o" if is_true else "x",
                   s=85 if is_true else 110,
                   c=PALETTE["green_3"] if is_true else PALETTE["red_strong"],
                   linewidths=2.2 if not is_true else 0,
                   label="true loop" if (is_true and i == 1) else
                         ("false loop" if (not is_true and i == 0) else None),
                   zorder=3)
    # counterexample: false 360#3 (0.64) > true 360#2 (0.56)
    ax.annotate("false 360#3 (0.64)\n> true 360#2 (0.56)",
                xy=(6, 0.636), xytext=(1.5, 1.02),
                fontsize=7.5,
                arrowprops=dict(arrowstyle="->", color=PALETTE["neutral_dark"], lw=0.9))
    ax.set_xticks(list(xs))
    ax.set_xticklabels([label(i) for i in range(len(LOOPS))], rotation=60, fontsize=6.5)
    ax.set_ylabel("category-level overlap (Jaccard)")
    ax.set_ylim(0.28, 1.06)
    ax.set_title("Category-level semantic overlap does NOT separate false from true loops (15 loops)",
                 fontsize=9)
    ax.legend(loc="lower left", framealpha=0.9)
    add_panel_label(ax, "a")
    finalize_figure(fig, os.path.join(OUT, "fig2_motivation"))
    print("fig2 motivation done")


# ════════════════════════════════════════════════════════════════════
# FIG 4 — 3-panel discriminator (15 loops)
# ════════════════════════════════════════════════════════════════════
def fig_discriminator():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4))
    xt = range(len(LOOPS))
    xl = [label(i) for i in range(len(LOOPS))]

    def scatter(ax, vals):
        for i, v in enumerate(vals):
            is_true = LOOPS[i][5]
            ax.scatter(i, v, marker="o" if is_true else "x",
                       s=95 if is_true else 120,
                       c=PALETTE["green_3"] if is_true else PALETTE["red_strong"],
                       linewidths=2.2 if not is_true else 0,
                       zorder=3)
        ax.set_xticks(list(xt))
        ax.set_xticklabels(xl, rotation=60, fontsize=6)

    ax = axes[0]
    scatter(ax, [l[2] for l in LOOPS])
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("appearance L2 (lower = closer)")
    ax.set_title("(a) appearance  →  separates", fontsize=9)

    ax = axes[1]
    scatter(ax, [l[3] for l in LOOPS])
    ax.set_ylim(0.28, 1.06)
    ax.set_ylabel("category overlap (Jaccard)")
    ax.set_title("(b) category overlap  →  fails", fontsize=9)

    ax = axes[2]
    scatter(ax, [l[4] for l in LOOPS])
    ax.axhline(0.10, color=PALETTE["neutral_dark"], linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_ylim(-0.03, 0.82)
    ax.set_ylabel("semantic-geometric 3D consistency")
    ax.set_title("(c) semantic-geometric 3D (ours)  →  separates", fontsize=9)

    # legend once (on panel c)
    handles = [
        plt.Line2D([], [], marker="x", linestyle="", color=PALETTE["red_strong"],
                   markeredgewidth=2.2, markersize=9, label="false loop (4)"),
        plt.Line2D([], [], marker="o", linestyle="", color=PALETTE["green_3"],
                   markersize=8, label="true loop (11)"),
    ]
    axes[2].legend(handles=handles, loc="upper left", framealpha=0.9, fontsize=7)
    add_panel_label(axes[0], "a"); add_panel_label(axes[1], "b"); add_panel_label(axes[2], "c")

    finalize_figure(fig, os.path.join(OUT, "fig4_discriminator"))
    print("fig4 discriminator done")


# ════════════════════════════════════════════════════════════════════
# FIG 5 — aligned before/after trajectories + ATE
# ════════════════════════════════════════════════════════════════════
def fig_trajectory():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for ax, seq in zip(axes, ["room", "360"]):
        base = "/root/autodl-tmp/tum_work/rgbd_dataset_freiburg1_" + seq
        gt_t, gt_xyz = load_traj_tt(os.path.join(base, "groundtruth.txt"))
        be_t, be_xyz = load_traj_tt(f"/root/autodl-tmp/semantic_out/{seq}/stage1_poses_before.txt")
        af_t, af_xyz = load_traj_tt(f"/root/autodl-tmp/semantic_out/{seq}_reject/stage1_poses_after.txt")
        gt_be = associate_traj(be_t, be_xyz, gt_t, gt_xyz)
        gt_af = associate_traj(af_t, af_xyz, gt_t, gt_xyz)
        before = umeyama_sim3(be_xyz, gt_be)
        after = umeyama_sim3(af_xyz, gt_af)

        ax.plot(gt_xyz[:, 0], gt_xyz[:, 2], c=PALETTE["neutral_mid"], linewidth=1.0,
                label="ground truth", zorder=1)
        ax.plot(before[:, 0], before[:, 2], c=PALETTE["red_strong"], linewidth=1.1,
                label="before (false loop)", zorder=2)
        ax.plot(after[:, 0], after[:, 2], c=PALETTE["blue_main"], linewidth=1.5,
                label="after (reject false loop)", zorder=3)
        ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
        ax.set_title(f"fr1/{seq}", fontsize=9.5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="best", framealpha=0.9, fontsize=7)
        # ATE annotation (official TUM ATE values)
        ate_b, ate_a = ATE_PAPER[seq]
        ax.text(0.03, 0.97,
                f"ATE\n  before {ate_b:.3f} m\n  after  {ate_a:.3f} m",
                transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
                color=PALETTE["neutral_black"],
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE["neutral_mid"], lw=0.7))
    fig.suptitle("Removing false loop closures recovers the pose graph (Sim(3)-aligned to GT)",
                 fontsize=10)
    finalize_figure(fig, os.path.join(OUT, "fig5_trajectory"))
    print("fig5 trajectory done")


if __name__ == "__main__":
    fig_motivation()
    fig_discriminator()
    fig_trajectory()
    print("ALL FIGURES DONE ->", OUT)
