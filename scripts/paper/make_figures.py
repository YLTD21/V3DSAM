import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT = "/root/autodl-tmp/figures"
os.makedirs(OUT, exist_ok=True)

# ── hardcoded experiment results (from semantic_filter.py + analyze_semantics.py + semantic_3d_check.py) ──
# each: (seq, loop_id, score_l2, jaccard, layout_rms, cons_3d, is_true)
LOOPS = [
    ("room", 0, 0.893, 0.64, 0.290, 0.078, False),
    ("room", 1, 0.646, 0.89, 0.197, 0.152, True),
    ("room", 2, 0.626, 1.00, 0.234, 0.481, True),
    ("360",  0, 0.947, 0.36, 0.158, 0.000, False),
    ("360",  1, 0.876, 0.45, 0.223, 0.025, False),
    ("360",  2, 0.613, 0.56, 0.184, 0.546, True),
    ("360",  3, 0.860, 0.64, 0.258, 0.000, False),
]

def load_traj(path):
    d = np.loadtxt(path)
    return d[:, 1], d[:, 3]  # x, z (TUM: y is up)


def load_traj_xyz(path):
    d = np.loadtxt(path)
    return d[:, 1:4]  # x, y, z


def umeyama_sim3(src, dst):
    """Align src (estimated, arbitrary monocular frame) to dst (GT, metric)
    via a similarity transform (scale + rotation + translation).
    Handles the monocular gauge freedom that otherwise makes the two
    trajectories appear "far apart" when plotted raw."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = min(len(src), len(dst))
    src, dst = src[:n], dst[:n]
    src_mu, dst_mu = src.mean(0), dst.mean(0)
    H = (src - src_mu).T @ (dst - dst_mu)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    scale = np.trace((dst - dst_mu).T @ (src - src_mu) @ R) / \
            np.trace((src - src_mu).T @ (src - src_mu))
    t = dst_mu - scale * R @ src_mu
    return (scale * (R @ src.T)).T + t


# ═════════════ FIG 1: three candidate discriminators ═════════════
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), dpi=200)
xt = range(len(LOOPS))
xl = [f"{s}#{lid}" for s, lid, *_ in LOOPS]

def scatter_split(ax, vals):
    true_x = []; true_v = []
    false_x = []; false_v = []
    for i, v in enumerate(vals):
        (true_x if LOOPS[i][-1] else false_x).append(i)
        (true_v if LOOPS[i][-1] else false_v).append(v)
    ax.scatter(false_x, false_v, c="#d62728", marker="x", s=150, linewidths=2.5,
               label="false loop", zorder=3)
    ax.scatter(true_x, true_v, c="#2ca02c", marker="o", s=120, label="true loop", zorder=3)
    ax.set_xticks(list(xt)); ax.set_xticklabels(xl, rotation=45, fontsize=7)
    ax.legend(loc="best", framealpha=0.9, fontsize=8)

# (a) appearance L2
ax = axes[0]
scatter_split(ax, [l[2] for l in LOOPS])
ax.axhline(0.85, color="k", linestyle="--", linewidth=1.2, alpha=0.7)
ax.set_ylim(0.5, 1.05)
ax.set_ylabel("appearance distance (L2, lower = closer)")
ax.set_title("(a) Appearance L2  →  separates", fontsize=10)

# (b) semantic class overlap (Jaccard)
ax = axes[1]
scatter_split(ax, [l[3] for l in LOOPS])
ax.set_ylim(0.2, 1.1)
ax.set_ylabel("semantic class overlap (Jaccard)")
ax.set_title("(b) Category-level overlap  →  FAILS", fontsize=10)

# (c) 3D semantic-geometric consistency
ax = axes[2]
scatter_split(ax, [l[5] for l in LOOPS])
ax.axhline(0.10, color="k", linestyle="--", linewidth=1.2, alpha=0.7)
ax.set_ylim(-0.02, 0.62)
ax.set_ylabel("3D semantic consistency")
ax.set_title("(c) Semantic-geometric 3D  →  separates", fontsize=10)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_appearance_vs_semantic.png"), bbox_inches="tight")
plt.close(fig)
print("fig1 done")


# ═════════════ FIG 2: trajectory before / after ═════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=200)
for ax, seq in zip(axes, ["room", "360"]):
    base = "/root/autodl-tmp/tum_work/rgbd_dataset_freiburg1_" + seq
    gt_xyz = load_traj_xyz(os.path.join(base, "groundtruth.txt"))
    before_xyz = load_traj_xyz(f"/root/autodl-tmp/semantic_out/{seq}/stage1_poses_before.txt")
    after_xyz = load_traj_xyz(f"/root/autodl-tmp/semantic_out/{seq}_reject/stage1_poses_after.txt")
    before_ali = umeyama_sim3(before_xyz, gt_xyz)
    after_ali = umeyama_sim3(after_xyz, gt_xyz)
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 2], c="#888888", linewidth=1.0, label="ground truth")
    ax.plot(before_ali[:, 0], before_ali[:, 2], c="#d62728", linewidth=1.2, label="before (false loop)")
    ax.plot(after_ali[:, 0], after_ali[:, 2], c="#2ca02c", linewidth=1.4, label="after (reject false loop)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.set_title(f"fr1/{seq}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
fig.suptitle("Trajectory: removing false loop closures recovers the pose graph")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_trajectory.png"), bbox_inches="tight")
plt.close(fig)
print("fig2 done")


# ═════════════ FIG 3: false-loop frame + SAM3 masks (room #0) ═════════════
COLORS = plt.get_cmap("tab20").colors
def overlay_mask(rgb, cat_mask):
    H, W = cat_mask.shape[1], cat_mask.shape[2]
    canvas = np.array(rgb).astype(np.float32).copy()
    for c in range(cat_mask.shape[0]):
        m = cat_mask[c]
        if m.sum() < 500:
            continue
        col = np.array(COLORS[c % len(COLORS)])[:3] * 255
        canvas[m > 0] = 0.5 * canvas[m > 0] + 0.5 * col
    return np.clip(canvas, 0, 255).astype(np.uint8)

fig = plt.figure(figsize=(9, 6), dpi=200)
qd = "/root/autodl-tmp/semantic_out/room/loop_figures"
q_img = Image.open(os.path.join(qd, "loop0_query.png")).convert("RGB")
d_img = Image.open(os.path.join(qd, "loop0_detected.png")).convert("RGB")
mz = np.load(os.path.join(qd, "loop0_masks.npz"))
mq, md = mz["mask_q"], mz["mask_d"]

for r, (im, mk, title) in enumerate([
        (q_img, mq, "query frame"), (d_img, md, "detected frame")]):
    plt.subplot(2, 2, 1 + r)
    plt.imshow(im); plt.axis("off"); plt.title(f"{title} (RGB)", fontsize=9)
    plt.subplot(2, 2, 3 + r)
    plt.imshow(overlay_mask(im, mk)); plt.axis("off")
    plt.title(f"{title} (SAM3 categories)", fontsize=9)
fig.suptitle("False loop closure (fr1/room #0): same categories, different viewpoint — "
             "class overlap (Jaccard) = 0.64, 3D consistency = 0.078", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_false_loop_example.png"), bbox_inches="tight")
plt.close(fig)
print("fig3 done")


# ═════════════ TABLE 1: loop verification log ═════════════
rows = []
for seq, lid, score, jacc, rms, cons3d, is_true in LOOPS:
    rows.append({
        "sequence": seq, "loop": lid,
        "appearance_L2": f"{score:.3f}",
        "semantic_Jaccard": f"{jacc:.2f}",
        "layout_RMS": f"{rms:.3f}",
        "cons_3d": f"{cons3d:.3f}",
        "ground_truth": "true" if is_true else "false",
        "appearance(0.85)_rejects": "YES" if score > 0.85 else "no",
        "jaccard_rejects": "no",
        "cons3d(0.10)_rejects": "YES" if cons3d < 0.10 else "no",
    })
import csv
with open(os.path.join(OUT, "table1_loop_log.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("table1 done")
print("ALL FIGURES DONE ->", OUT)
