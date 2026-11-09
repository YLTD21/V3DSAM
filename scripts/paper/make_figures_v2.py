#!/usr/bin/env python3
"""V3DSAM paper figures — SCI-level. Run on AutoDL (sgs env).

Produces, into /root/autodl-tmp/figures_sci/ :
  fig1_architecture.{pdf,png}   system architecture
  fig2_motivation.{pdf,png}     why category overlap fails
  fig3_discriminator.{pdf,png}  3-panel loop discriminator
  fig4_trajectory.{pdf,png}     aligned before/after trajectories
  fig5_false_loop.png           false-loop frames + SAM3 semantic masks
  fig6_rendering.png            novel-view rendering (GT vs ours)
"""
import os, glob, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

# ── SCI-level global style (nature-figure skill) ──
from sci_style import apply_publication_style, PALETTE, add_panel_label, finalize_figure

apply_publication_style(font_size=8, axes_linewidth=1)

OUT = os.environ.get("FIG_OUT", "/root/autodl-tmp/figures_sci")
os.makedirs(OUT, exist_ok=True)

# ── hardcoded loop-verification results (from semantic_3d_check.py) ──
# (seq, loop_id, appearance_L2, category_Jaccard, cons_3d, is_true)
LOOPS = [
    ("room", 0, 0.893, 0.64, 0.078, False),
    ("room", 1, 0.646, 0.89, 0.152, True),
    ("room", 2, 0.626, 1.00, 0.481, True),
    ("360",  0, 0.947, 0.36, 0.000, False),
    ("360",  1, 0.876, 0.45, 0.025, False),
    ("360",  2, 0.613, 0.56, 0.546, True),
    ("360",  3, 0.860, 0.64, 0.000, False),
]

ALL_PROMPTS = [
    "wall", "floor", "ceiling", "door", "window", "stairs", "railing",
    "bed", "cabinet", "bookshelf", "shelf", "counter", "desk", "table", "sofa", "couch",
    "sink", "toilet", "bathtub", "mirror", "lamp", "light",
    "television", "monitor", "laptop", "keyboard", "mouse", "phone", "refrigerator", "oven", "microwave",
    "chair", "plant", "vase", "picture", "painting", "curtain", "pillow", "clock",
    "book", "bottle", "cup", "bowl", "plate", "bag", "backpack", "box", "bin", "basket", "person",
]


def load_traj_tt(path):
    """TUM-format trajectory -> (timestamps, xyz)."""
    d = np.loadtxt(path)
    return d[:, 0], d[:, 1:4]


def associate_traj(est_t, est_xyz, gt_t, gt_xyz):
    """Match each estimated pose to the nearest-timestamp GT pose.
    The estimate is a sparse subset of keyframes; GT is dense (30 Hz),
    so index-truncation would pair unrelated points and corrupt Umeyama."""
    idx = np.searchsorted(gt_t, est_t)
    idx = np.clip(idx, 0, len(gt_t) - 1)
    left = np.clip(idx - 1, 0, len(gt_t) - 1)
    best = np.where(np.abs(gt_t[left] - est_t) < np.abs(gt_t[idx] - est_t), left, idx)
    return gt_xyz[best]


def umeyama_sim3(src, dst):
    """Correct Sim(3) (rotation+scale+translation) alignment, Umeyama 1991.
    Assumes src/dst are already in point correspondence."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = min(len(src), len(dst))
    src, dst = src[:n], dst[:n]
    src_mu, dst_mu = src.mean(0), dst.mean(0)
    sc, dc = src - src_mu, dst - dst_mu
    cov = sc.T @ dc / n                       # cross-covariance (3x3)
    U, S, Vt = np.linalg.svd(cov)
    sign = np.sign(np.linalg.det(Vt.T @ U.T))
    Sd = np.diag([1.0, 1.0, sign])            # reflection correction
    R = Vt.T @ Sd @ U.T
    scale = np.trace(np.diag(S) @ Sd) / np.trace(sc.T @ sc / n)
    t = dst_mu - scale * R @ src_mu
    return (scale * (R @ src.T)).T + t


# ════════════════════════════════════════════════════════════════════
# FIG 1 — system architecture
# ════════════════════════════════════════════════════════════════════
def fig_architecture():
    fig, ax = plt.subplots(figsize=(10.2, 3.4))
    ax.set_xlim(0, 10.2); ax.set_ylim(0, 3.4)
    ax.axis("off")

    def box(x, y, w, h, title, sub, fc, ec="#333333"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                           linewidth=1.1, edgecolor=ec, facecolor=fc)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=9.5, weight="bold")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=7.2, color="#333333")

    def arrow(x1, y1, x2, y2, label=None):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                            linewidth=1.2, color="#444444")
        ax.add_patch(a)
        if label:
            ax.text((x1 + x2) / 2, y1 + 0.24, label, ha="center", va="bottom", fontsize=6.6, color="#555555")

    box(0.15, 1.35, 2.05, 1.25, "Stage 1\nVGGT-SLAM\nfeed-forward front-end",
        "poses · dense depth ·\nloop candidates", "#dbe9f6")
    box(2.75, 1.35, 2.05, 1.25, "Stage 2\nSAM3 semantic\ngrounding",
        "per-frame category\nmasks", "#e8e3f7")
    box(5.35, 1.35, 2.30, 1.25, "Semantic-Geometric\nLoop Verification",
        "3D consistency (thr 0.10)\nreject false loops", "#fde9d9")
    box(8.20, 1.35, 1.85, 1.25, "Stage 3\nSplat-SLAM\nsemantic 3DGS",
        "photorealistic\nreconstruction", "#e2f0d9")

    arrow(2.20, 1.97, 2.73, 1.97)
    arrow(4.80, 1.97, 5.33, 1.97)
    arrow(7.65, 1.97, 8.18, 1.97)

    ax.text(1.17, 0.75, "RGB sequence", ha="center", fontsize=8, color="#222222")
    arrow(1.17, 0.95, 1.17, 1.32)
    ax.text(9.12, 0.75, "metric-scale\n3D semantic map", ha="center", fontsize=8, color="#222222")
    arrow(9.12, 1.32, 9.12, 0.98)

    ax.text(0.5, 0.18, "(a)  input  →  (b) front-end  →  (c) semantics  →  (d) loop verification  →  (e) reconstruction",
            fontsize=7.5, color="#555555")
    fig.savefig(os.path.join(OUT, "fig1_architecture.pdf"))
    fig.savefig(os.path.join(OUT, "fig1_architecture.png"))
    plt.close(fig)
    print("fig1 architecture done")


# ════════════════════════════════════════════════════════════════════
# FIG 2 — motivation: category overlap fails
# ════════════════════════════════════════════════════════════════════
def fig_motivation():
    # sort by Jaccard to show false loop 360#3 (0.64) > true loop 360#2 (0.56)
    pairs = [(l[3], l[5], l[0] + "#" + str(l[1]), l[5]) for l in LOOPS]
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    xs = np.arange(len(LOOPS))
    false_mask = [not l[5] for l in LOOPS]
    true_mask = [l[5] for l in LOOPS]
    xs_f = [i for i, m in enumerate(false_mask) if m]
    xs_t = [i for i, m in enumerate(true_mask) if m]
    ax.scatter(xs_f, [LOOPS[i][3] for i in xs_f], marker="x", s=110, c="#d62728", linewidths=2.2, label="false loop", zorder=3)
    ax.scatter(xs_t, [LOOPS[i][3] for i in xs_t], marker="o", s=90, c="#2ca02c", label="true loop", zorder=3)
    # annotate the counterexample
    ax.annotate("false loop (overlap 0.64)\n> true loop (0.56)", xy=(6, 0.64), xytext=(3.2, 0.90),
                fontsize=7.5, arrowprops=dict(arrowstyle="->", color="#333333", lw=0.9))
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{l[0]}#{l[1]}" for l in LOOPS], rotation=45, fontsize=7)
    ax.set_ylabel("category-level overlap (Jaccard)")
    ax.set_ylim(0.2, 1.1)
    ax.set_title("Category-level semantic overlap does NOT separate false from true loops", fontsize=9)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.savefig(os.path.join(OUT, "fig2_motivation.pdf"))
    fig.savefig(os.path.join(OUT, "fig2_motivation.png"))
    plt.close(fig)
    print("fig2 motivation done")


# ════════════════════════════════════════════════════════════════════
# FIG 3 — 3-panel loop discriminator
# ════════════════════════════════════════════════════════════════════
def fig_discriminator():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4))
    xt = range(len(LOOPS))
    xl = [f"{s}#{lid}" for s, lid, *_ in LOOPS]

    def scatter(ax, vals):
        fx, fv, tx, tv = [], [], [], []
        for i, v in enumerate(vals):
            (tx if LOOPS[i][-1] else fx).append(i)
            (tv if LOOPS[i][-1] else fv).append(v)
        ax.scatter(fx, fv, c="#d62728", marker="x", s=150, linewidths=2.4, label="false loop", zorder=3)
        ax.scatter(tx, tv, c="#2ca02c", marker="o", s=120, label="true loop", zorder=3)
        ax.set_xticks(list(xt)); ax.set_xticklabels(xl, rotation=45, fontsize=6.5)
        ax.legend(loc="best", framealpha=0.9)

    ax = axes[0]
    scatter(ax, [l[2] for l in LOOPS])
    ax.axhline(0.85, color="k", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_ylim(0.5, 1.05); ax.set_ylabel("appearance L2 (lower = closer)")
    ax.set_title("(a) appearance  →  ambiguous", fontsize=9.5)

    ax = axes[1]
    scatter(ax, [l[3] for l in LOOPS])
    ax.set_ylim(0.2, 1.1); ax.set_ylabel("category overlap (Jaccard)")
    ax.set_title("(b) category overlap  →  fails", fontsize=9.5)

    ax = axes[2]
    scatter(ax, [l[5] for l in LOOPS])
    ax.axhline(0.10, color="k", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_ylim(-0.03, 0.62); ax.set_ylabel("3D semantic-geometric consistency")
    ax.set_title("(c) semantic-geometric 3D (ours)  →  separates", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_discriminator.pdf"))
    fig.savefig(os.path.join(OUT, "fig3_discriminator.png"))
    plt.close(fig)
    print("fig3 discriminator done")


# ════════════════════════════════════════════════════════════════════
# FIG 4 — aligned before/after trajectories
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
        ax.plot(gt_xyz[:, 0], gt_xyz[:, 2], c="#888888", linewidth=1.0, label="ground truth", zorder=1)
        ax.plot(before[:, 0], before[:, 2], c="#d62728", linewidth=1.1, label="before (false loop)", zorder=2)
        ax.plot(after[:, 0], after[:, 2], c="#2ca02c", linewidth=1.5, label="after (reject false loop)", zorder=3)
        ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
        ax.set_title(f"fr1/{seq}", fontsize=9.5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="best", framealpha=0.9)
    fig.suptitle("Removing false loop closures recovers the pose graph (trajectories Sim(3)-aligned to GT)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "fig4_trajectory.pdf"))
    fig.savefig(os.path.join(OUT, "fig4_trajectory.png"))
    plt.close(fig)
    print("fig4 trajectory done")


# ════════════════════════════════════════════════════════════════════
# FIG 5 — false-loop frames + SAM3 semantic masks
# ════════════════════════════════════════════════════════════════════
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


def fig_false_loop():
    qd = "/root/autodl-tmp/semantic_out/room/loop_figures"
    q_img = Image.open(os.path.join(qd, "loop0_query.png")).convert("RGB")
    d_img = Image.open(os.path.join(qd, "loop0_detected.png")).convert("RGB")
    mz = np.load(os.path.join(qd, "loop0_masks.npz"))
    mq, md = mz["mask_q"], mz["mask_d"]

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.2))
    panels = [
        (axes[0, 0], q_img, "query frame (RGB)"),
        (axes[0, 1], d_img, "detected frame (RGB)"),
        (axes[1, 0], Image.fromarray(overlay_mask(q_img, mq)), "query + SAM3 categories"),
        (axes[1, 1], Image.fromarray(overlay_mask(d_img, md)), "detected + SAM3 categories"),
    ]
    for ax, im, title in panels:
        ax.imshow(im); ax.axis("off"); ax.set_title(title, fontsize=9)
    fig.suptitle("False loop closure (fr1/room #0): same categories, different viewpoint\n"
                 "category overlap (Jaccard) = 0.64  vs.  3D semantic consistency = 0.078",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "fig6_false_loop.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig6 false loop done")


# ════════════════════════════════════════════════════════════════════
# FIG 6 — novel-view rendering (GT vs ours)
# ════════════════════════════════════════════════════════════════════
def _psnr(gt, pred):
    mse = np.mean((gt.astype(np.float32) - pred.astype(np.float32)) ** 2)
    return 20 * math.log10(255.0 / math.sqrt(mse + 1e-10))


def fig_rendering():
    scenes = ["room", "floor", "360"]
    rows = []
    for seq in scenes:
        d = f"/root/autodl-tmp/psnr_tum/{seq}/splatslam/plots_after_refine"
        files = sorted(glob.glob(os.path.join(d, "*.png")))
        scored = []
        for f in files:
            im = np.asarray(Image.open(f).convert("RGB"))
            h, w, _ = im.shape
            gt = im[:, :w // 2]
            pred = im[:, w // 2:]
            if gt.shape != pred.shape:
                continue
            scored.append((_psnr(gt, pred), f, gt, pred))
        scored.sort(key=lambda x: -x[0])
        rows.append(scored[:3])

    fig, axes = plt.subplots(len(scenes), 6, figsize=(13.5, len(scenes) * 2.35))
    if len(scenes) == 1:
        axes = axes[None, :]
    for r, row in enumerate(rows):
        for c, (psnr, f, gt, pred) in enumerate(row):
            axes[r, 2 * c].imshow(gt); axes[r, 2 * c].axis("off")
            axes[r, 2 * c + 1].imshow(pred); axes[r, 2 * c + 1].axis("off")
            axes[r, 2 * c + 1].set_title(f"{psnr:.1f} dB", fontsize=8)
        axes[r, 0].set_ylabel(scenes[r], fontsize=9, rotation=0, labelpad=24)
    # column headers
    for c in range(3):
        axes[0, 2 * c].set_title("Ground truth", fontsize=9)
        axes[0, 2 * c + 1].set_title("Rendered (ours)", fontsize=9)
    fig.suptitle("Novel-view rendering: Splat-SLAM reconstruction from VGGT-SLAM poses (top-3 PSNR frames per scene)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "fig6_rendering.png"))
    plt.close(fig)
    print("fig6 rendering done")


if __name__ == "__main__":
    # Only fig6 (false loop) lives here now. fig1 moved to make_fig1_architecture.py,
    # fig2/fig4/fig5 to make_figures_v3.py, fig7 to make_fig7_nvs.py.
    fig_false_loop()
    print("FIG6 DONE ->", OUT)
