#!/usr/bin/env python3
"""Semantic 3D map = V3DSAM Stage-3 Gaussian-init point cloud (pts_gs, world frame)
colored by SAM3 category, using the SAME projection as v3dsam's own
`semantic_sam3/semantic_pointcloud.py::SemanticPointCloudEnhancer`:
    world point --w2c--> camera --K_full--> image pixel --> sample SAM3 label map
then majority-vote the label across all keyframes that see the point.

This is the honest "semantic point cloud result": the actual aggregated world-frame
point cloud that initializes the 3DGS map (pts_gs), NOT a re-unprojection of raw
single-view VGGT depth (which is noisy and mis-scaled).

CPU-only (numpy + matplotlib). No GPU, no v3dsam import.

Inputs:
  --stage3     _stage3_input.npz   (has pts_gs, rgb_gs, K_i, w2c_i, vggt_kf_indices)
  --semantic   kf_semantic.npz     (labels (K,480,640) int16, kf_indices (K,), prompts)
  --out_dir    where to write semantic_cloud.ply + semantic_cloud.png + rgb_cloud.ply
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sci_style import apply_publication_style

apply_publication_style(font_size=8, axes_linewidth=1)

# 50 distinct colors (tab20 + tab20b + tab20c), one per prompt category.
_tab = [np.array(plt.get_cmap(cm).colors) for cm in ("tab20", "tab20b", "tab20c")]
PALETTE = np.vstack(_tab)[:50]          # (50, 4) RGBA


def load_stage3(path):
    d = np.load(path, allow_pickle=True)
    pts = d["pts_gs"].astype(np.float64)   # (N,3) world frame
    rgb = d["rgb_gs"].astype(np.float64)   # (N,3) in [0,1]
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    return d, pts, rgb


def subsample(pts, rgb, n):
    if len(pts) <= n:
        return pts, rgb, np.arange(len(pts))
    idx = np.random.RandomState(0).choice(len(pts), n, replace=False)
    idx.sort()
    return pts[idx], rgb[idx], idx


def vote_labels(pts, stage3, semantic, min_votes=2):
    """Project world points into each keyframe and majority-vote the SAM3 label."""
    s = np.load(semantic, allow_pickle=True)
    labels = s["labels"]                     # (K, H, W) int16, -1 = unlabeled
    kf_indices = [int(x) for x in s["kf_indices"].tolist()]
    prompts = [str(p) for p in s["prompts"]]
    n_cats = len(prompts)
    H, W = labels.shape[1], labels.shape[2]
    N = len(pts)

    pts_h = np.hstack([pts, np.ones((N, 1), dtype=np.float64)])   # (N,4)
    all_pidx, all_lab = [], []
    n_seen = 0
    for k, fi in enumerate(kf_indices):
        if f"w2c_{fi}" not in stage3.files or f"K_{fi}" not in stage3.files:
            continue
        w2c = stage3[f"w2c_{fi}"].astype(np.float64)              # (4,4)
        K = stage3[f"K_{fi}"].astype(np.float64)                  # (3,3)
        cam = (w2c @ pts_h.T).T                                   # (N,4) camera frame
        z = cam[:, 2]
        front = z > 0.05
        if not front.any():
            continue
        Xc, Yc, Zc = cam[:, 0], cam[:, 1], cam[:, 2]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        u = fx * Xc / Zc + cx
        v = fy * Yc / Zc + cy
        ui = np.floor(u).astype(np.int64)
        vi = np.floor(v).astype(np.int64)
        inb = front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if not inb.any():
            continue
        lab = labels[k][vi[inb], ui[inb]]                         # sampled labels
        keep = lab >= 0
        if not keep.any():
            continue
        pidx = np.nonzero(inb)[0][keep]
        all_pidx.append(pidx)
        all_lab.append(lab[keep])
        n_seen += 1

    if not all_pidx:
        return np.full(N, -1, np.int32), prompts, 0

    flat = np.concatenate([p * n_cats + l for p, l in zip(all_pidx, all_lab)])
    votes = np.bincount(flat, minlength=N * n_cats).reshape(N, n_cats)
    total = votes.sum(axis=1)
    sid = votes.argmax(axis=1).astype(np.int32)
    sid[total < min_votes] = -1                                   # too few views -> unlabeled
    return sid, prompts, n_seen


def write_ply(path, pts, rgb, sid):
    """Write an ASCII PLY with per-point color (category color if labeled, else rgb)."""
    n = len(pts)
    cols = np.zeros((n, 3), np.uint8)
    labeled = sid >= 0
    cols[labeled] = (PALETTE[sid[labeled]][:, :3] * 255).astype(np.uint8)
    # unlabeled points keep their true RGB so the map still reads as a scene
    cols[~labeled] = np.clip(rgb[~labeled] * 255, 0, 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property int label\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b), lab in zip(pts, cols, sid):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)} {int(lab)}\n")
    print(f"[semantic cloud] wrote {n} points -> {path} ({len(pts)*3*8/1e6:.1f}MB ascii-ish)")


def render_fig(out_png, pts, rgb, sid, prompts, n_show=250000):
    if len(pts) > n_show:
        idx = np.random.RandomState(0).choice(len(pts), n_show, replace=False)
        pts, rgb, sid = pts[idx], rgb[idx], sid[idx]

    rgb255 = np.clip(rgb * 255, 0, 255).astype(np.uint8) / 255.0
    labeled = sid >= 0
    sem_cols = np.ones((len(pts), 3)) * 0.75                     # unlabeled = light gray
    sem_cols[labeled] = PALETTE[sid[labeled]][:, :3]

    fig = plt.figure(figsize=(13.5, 4.6))
    # (a) true-color perspective
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=rgb255, s=0.25, alpha=0.6,
                linewidths=0, rasterized=True)
    ax1.set_title("(a) RGB point cloud", fontsize=9)
    ax1.view_init(elev=28, azim=-62)
    set_axes_equal(ax1, pts)

    # (b) semantic perspective
    ax2 = fig.add_subplot(132, projection="3d")
    ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=sem_cols, s=0.25, alpha=0.7,
                linewidths=0, rasterized=True)
    ax2.set_title("(b) semantic perspective", fontsize=9)
    ax2.view_init(elev=28, azim=-62)
    set_axes_equal(ax2, pts)

    # (c) semantic top-down
    ax3 = fig.add_subplot(133, projection="3d")
    ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=sem_cols, s=0.25, alpha=0.7,
                linewidths=0, rasterized=True)
    ax3.set_title("(c) semantic top-down", fontsize=9)
    ax3.view_init(elev=88, azim=0)
    set_axes_equal(ax3, pts)

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")

    # legend: categories actually present
    present = sorted(int(c) for c in np.unique(sid) if c >= 0)
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=PALETTE[c][:3], markersize=6,
                          label=prompts[c]) for c in present]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=6,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("V3DSAM semantic point cloud (SAM3 category → world-frame 3DGS init points)", fontsize=10)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[semantic cloud] figure -> {out_png}")


def set_axes_equal(ax, P):
    mid = P.mean(0); rng = np.ptp(P, 0).max() / 2.0
    ax.set_xlim(mid[0]-rng, mid[0]+rng)
    ax.set_ylim(mid[1]-rng, mid[1]+rng)
    ax.set_zlim(mid[2]-rng, mid[2]+rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage3", required=True)
    ap.add_argument("--semantic", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_pts", type=int, default=1000000)
    ap.add_argument("--n_show", type=int, default=250000)
    ap.add_argument("--min_votes", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    stage3, pts, rgb = load_stage3(args.stage3)
    print(f"[semantic cloud] loaded {len(pts):,} world-frame points")
    pts, rgb, _ = subsample(pts, rgb, args.max_pts)
    print(f"[semantic cloud] subsampled to {len(pts):,} points")

    sid, prompts, n_seen = vote_labels(pts, stage3, args.semantic, args.min_votes)
    n_lab = (sid >= 0).sum()
    print(f"[semantic cloud] voted over {n_seen} keyframes; "
          f"{n_lab:,}/{len(pts):,} points labeled ({n_lab/len(pts)*100:.1f}%)")

    write_ply(os.path.join(args.out_dir, "semantic_cloud.ply"), pts, rgb, sid)
    render_fig(os.path.join(args.out_dir, "semantic_cloud.png"), pts, rgb, sid,
               prompts, args.n_show)
    print("DONE")


if __name__ == "__main__":
    main()
