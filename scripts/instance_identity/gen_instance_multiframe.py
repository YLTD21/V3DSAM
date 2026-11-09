#!/usr/bin/env python3
"""Persistent-instance multi-frame figure (Contribution 2).

Picks K keyframes of fr1/room from disjoint viewpoints; on each frame projects
every foreground object instance's 3D world centroid into 2D and marks it with
a colour that is *fixed per instance id* plus its all-view voted semantic name.
The same physical object (e.g. chair #31) then carries the same colour + label
across frames, so the reader sees identity anchored to a stable world position
(median cross-view centroid displacement ~7 cm) rather than to per-view pixels.

Reuses the exact projection + data paths of gen_instance_video4.py.  Static,
SCI-styled (sans-serif, no chart junk) version for the paper.

Run on AutoDL (sgs env):
  python gen_instance_multiframe.py --seq room
"""
import os, json, argparse, glob
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sci_style import apply_publication_style, finalize_figure

apply_publication_style(font_size=8, axes_linewidth=1)


# ───────────────────────────────────────────────────────────── helpers ──
def project_pixels(pts, w2c, K):
    ph = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
    cam = (w2c @ ph.T).T
    z = cam[:, 2]
    front = z > 0.05
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = np.where(front, fx * cam[:, 0] / np.maximum(z, 1e-6) + cx, -9999.0)
    v = np.where(front, fy * cam[:, 1] / np.maximum(z, 1e-6) + cy, -9999.0)
    return u, v, z, front


# 8 print-friendly, mutually distinct instance colours (dark enough for text)
INST_COLORS = [
    "#D62728",  # red
    "#1F77B4",  # blue
    "#2CA02C",  # green
    "#FF7F0E",  # orange
    "#9467BD",  # purple
    "#8C564B",  # brown
    "#17BECF",  # cyan
    "#E377C2",  # pink
]


def hex2bgr(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)

FOREGROUND = {"chair", "desk", "table", "television", "monitor", "plant",
              "picture", "cabinet", "lamp", "keyboard", "laptop", "sofa",
              "book", "bottle", "cup", "phone", "mouse", "clock", "shelf"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="room")
    ap.add_argument("--n_frames", type=int, default=6)
    ap.add_argument("--n_hero", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = f"/root/autodl-tmp/psnr_tum/{args.seq}"
    stage3 = np.load(f"{base}/stage1_scaled/_stage3_input.npz", allow_pickle=True)
    sem = np.load(f"{base}/stage1_scaled/kf_semantic.npz", allow_pickle=True)
    inst_meta = json.load(open(f"{base}/semantic_instances_postfilter/instances.json"))
    voted_meta = json.load(open(f"{base}/semantic_instances_postfilter/voted_instances.json"))

    prompts = [str(p) for p in sem["prompts"]]
    kf_indices = [int(x) for x in sem["kf_indices"].tolist()]
    img_W, img_H = int(stage3["img_W"]), int(stage3["img_H"])

    # instance id -> (centroid, voted class, n_vis, consistency)
    inst_list = inst_meta["instances"]
    cent = {}; n_vis = {}; cons = {}; cls = {}
    for o in inst_list:
        i = int(o["id"])
        cent[i] = np.array(o["centroid"], np.float64)
        n_vis[i] = int(o.get("n_vis", 0))
        cons[i] = float(o.get("consistency", 0))
        cls[i] = int(o.get("class", -1))
    voted_class = {}
    for o in voted_meta["instances"]:
        i = int(o["id"]); voted_class[i] = int(o["voted_class"])

    # hero instances: one per foreground class (diverse), most-visible first
    cand = [i for i in cent if i in voted_class
            and prompts[voted_class[i]] in FOREGROUND]
    best_per_class = {}
    for i in cand:
        nm = prompts[voted_class[i]]
        if nm not in best_per_class or n_vis[i] > n_vis[best_per_class[nm]]:
            best_per_class[nm] = i
    diverse = sorted(best_per_class.values(), key=lambda i: n_vis[i], reverse=True)
    hero = diverse[:args.n_hero]
    hero_name = {i: prompts[voted_class[i]] for i in hero}
    print(f"[multiframe] hero instances: {[(i, hero_name[i], n_vis[i]) for i in hero]}")

    # frames spread across the trajectory (disjoint viewpoints)
    step = max(1, len(kf_indices) // args.n_frames)
    pick = [kf_indices[min(k * step, len(kf_indices) - 1)]
            for k in range(args.n_frames)]
    # de-duplicate (small trajectories)
    pick = sorted(set(pick))
    print(f"[multiframe] frames: {pick}")

    fig, axes = plt.subplots(2, (len(pick) + 1) // 2,
                             figsize=(4.1 * ((len(pick) + 1) // 2), 2.9 * 2))
    axes = np.atleast_2d(axes)
    for r in range(axes.shape[0]):
        for c in range(axes.shape[1]):
            axes[r, c].axis("off")

    # project + overlay per picked frame
    for k, fi in enumerate(pick):
        r, c = divmod(k, axes.shape[1])
        ax = axes[r, c]
        img = cv2.imread(str(stage3[f"img_path_{fi}"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        w2c = stage3[f"w2c_{fi}"].astype(np.float64)
        K = stage3[f"K_{fi}"].astype(np.float64)

        # project hero centroids
        hids = list(hero)
        u, v, z, front = project_pixels(np.array([cent[i] for i in hids]), w2c, K)
        H, W = img.shape[:2]
        sx = W / img_W
        # K is defined on img_W x img_H; image may be stored at same size
        inb = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        for j, i in enumerate(hids):
            if not inb[j]:
                continue
            col = INST_COLORS[i % len(INST_COLORS)]
            colb = hex2bgr(col)
            cu, cv_ = int(round(u[j])), int(round(v[j]))
            cv2.circle(img, (cu, cv_), 11, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (cu, cv_), 9, colb, -1, cv2.LINE_AA)
            cv2.putText(img, f"#{i} {hero_name[i]}", (cu + 12, cv_ - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f"#{i} {hero_name[i]}", (cu + 12, cv_ - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        ax.imshow(img)
        ax.set_title(f"frame {fi}", fontsize=8, color="#272727")

    # legend: one entry per hero instance (id + semantic name + colour)
    handles = [Patch(facecolor=INST_COLORS[i % len(INST_COLORS)],
                     edgecolor="white", label=f"#{i} {hero_name[i]}")
               for i in hero]
    fig.legend(handles=handles, loc="upper center", ncol=len(hero),
               bbox_to_anchor=(0.5, 1.01), fontsize=7.5, frameon=False)
    fig.suptitle(
        "Persistent instance identity (fr1/room): same object keeps its colour + semantic "
        "label across disjoint viewpoints\n(identity anchored to world coordinate; median "
        "cross-view centroid displacement ~7 cm)",
        fontsize=10, y=1.08)

    out = args.out or f"{base}/semantic_instances_postfilter/fig12_instance_multiframe"
    finalize_figure(fig, out)
    print(f"[multiframe] -> {out}.png")


if __name__ == "__main__":
    main()
