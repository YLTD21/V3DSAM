#!/usr/bin/env python3
"""3x5 persistent-instance figure (innovation 3): 3 scenes x (3 overlapping keyframes
+ 2 loop-closure frames).

Each row = one scene (room / 360 / desk2). Columns 0-2 = 3 consecutive (overlapping)
keyframes; columns 3-4 = 2 temporally-distant loop-closure keyframes that revisit the
same objects. Every frame annotates all visible foreground instances with a
fixed-per-instance colour dot at its 3D world centroid, a soft semantic mask in the
instance colour (fig6 style: 55% image + 45% colour blend, no contour), and a
"#id <name>" label. The same physical object keeps the same colour + #id + category
across the loop — the geometric centroid (in the globally-consistent world frame) is
the identity anchor, not per-view pixels.

Reuses the projection + data paths of gen_instance_multiframe.py.

Run on AutoDL (sgs env):
  python gen_instance_grid_3x5.py
"""
import os, json, argparse
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sci_style import apply_publication_style, finalize_figure
    apply_publication_style(font_size=8, axes_linewidth=1)
except Exception:
    def finalize_figure(fig, out):
        fig.savefig(out + ".png", dpi=180, bbox_inches="tight")
        fig.savefig(out + ".pdf", bbox_inches="tight")
        print(f"-> {out}.png / .pdf")


FOREGROUND = {"chair","desk","table","television","monitor","plant","picture",
              "cabinet","lamp","keyboard","laptop","sofa","book","bottle","cup",
              "phone","mouse","clock","shelf","bin","bag","pillow","vase","box",
              "backpack","bowl","plate","mirror","sink","counter","stairs","door","window"}

# scene -> (base, stage3_rel, semantic_rel, instance_dir_rel)
SCENES = {
    "room":  ("/root/autodl-tmp/psnr_tum/room",  "stage1_scaled/_stage3_input.npz", "stage1_scaled/kf_semantic.npz", "semantic_instances_postfilter"),
    "360":   ("/root/autodl-tmp/psnr_tum/360",   "stage1/_stage3_input.npz",         "stage1/kf_semantic.npz",         "semantic_instances"),
    "desk2": ("/root/autodl-tmp/psnr_tum/desk2", "stage1/_stage3_input.npz",         "stage1/kf_semantic.npz",         "semantic_instances"),
}

# 20 print-friendly, mutually distinct colours (tab20)
TAB20 = ["#1f77b4","#aec7e8","#ff7f0e","#ffbb78","#2ca02c","#98df8a","#d62728",
         "#ff9896","#9467bd","#c5b0d5","#8c564b","#c49c94","#e377c2","#f7b6d2",
         "#7f7f7f","#c7c7c7","#bcbd22","#dbdb8d","#17becf","#9edae5"]


def hex2rgb(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def project_pixels(pts, w2c, K):
    ph = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
    cam = (w2c @ ph.T).T
    z = cam[:, 2]
    front = z > 0.05
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = np.where(front, fx * cam[:, 0] / np.maximum(z, 1e-6) + cx, -9999.0)
    v = np.where(front, fy * cam[:, 1] / np.maximum(z, 1e-6) + cy, -9999.0)
    return u, v, z, front


def load_objs(base, inst_dir):
    inst = json.load(open(f"{base}/{inst_dir}/instances.json"))
    voted = json.load(open(f"{base}/{inst_dir}/voted_instances.json"))
    prompts = [str(p) for p in inst["prompts"]]
    vc = {}
    for o in voted["instances"]:
        vc[int(o["id"])] = int(o["voted_class"])
    objs = []
    for o in inst["instances"]:
        i = int(o["id"])
        if i not in vc:
            continue
        c = int(vc[i])
        if prompts[c] not in FOREGROUND:
            continue
        if int(o.get("n_vis", 0)) < 3:
            continue
        objs.append({"id": i, "class": c, "name": prompts[c],
                     "centroid": np.array(o["centroid"], np.float64),
                     "n_vis": int(o.get("n_vis", 0))})
    objs.sort(key=lambda o: -o["n_vis"])
    return objs


def visible_mask(objs, w2c, K, W, H):
    if not objs:
        return np.zeros(0, dtype=bool)
    u, v, z, front = project_pixels(np.array([o["centroid"] for o in objs]), w2c, K)
    return front & (u >= 0) & (u < W) & (v >= 0) & (v < H), u, v


def instance_mask(lab, cls, u, v, W, H):
    """Tight per-instance mask: the connected component of the class mask that
    contains the projected centroid. This separates multiple instances of the
    same class (e.g. several desks) into distinct objects instead of tinting the
    whole class region."""
    cmask = (lab == cls)
    if not cmask.any():
        return None
    num, cc = cv2.connectedComponents(cmask.astype(np.uint8), connectivity=8)
    cu, cv_ = int(round(u)), int(round(v))
    if 0 <= cv_ < H and 0 <= cu < W:
        cid = cc[cv_, cu]
        if cid > 0:
            return cc == cid
    # centroid just off the mask: fall back to the largest component
    if num > 1:
        areas = np.bincount(cc.ravel())[1:]
        cid = int(np.argmax(areas)) + 1
        return cc == cid
    return cmask


def draw_label(img, text, cu, cv_, colrgb, W, H):
    """White chip + black text (readable on any background), with a thin border in
    the instance colour to tie the label to its dot + mask."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.9, 2
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    tx = cu + 15
    ty = cv_ - 12  # baseline
    x0, y0 = tx - 3, ty - th - 3
    x1, y1 = tx + tw + 3, ty + base + 3
    # clamp the chip into the frame
    if x1 > W - 2:
        x0 -= (x1 - (W - 2)); x1 = W - 2
    if x0 < 2:
        x0 = 2
    if y0 < 2:
        y0 = 2
    if y1 > H - 2:
        y1 = H - 2
    tx, ty = x0 + 3, y0 + th + 3
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0), (x1, y1), colrgb, 2, cv2.LINE_AA)
    cv2.putText(img, text, (tx, ty), font, scale, (0, 0, 0), thick, cv2.LINE_AA)


def pick_window(vis, n_local=3, n_loop=2, gap_min=None):
    """Pick a 5-frame window = n_local consecutive (overlapping) keyframes + n_loop
    loop-closure keyframes (temporally distant, revisiting the same objects). Returns
    (score, win_pos, vis_local, vis_loop) where win_pos are positions into kf_indices.

    The loop-closure frames are chosen to maximise the number of objects shared with
    the local window, so the SAME object (same #id) appears in both the local views and
    the loop-closure views — the core evidence for persistent instance identity."""
    K = vis.shape[0]  # number of keyframes
    if gap_min is None:
        gap_min = max(10, int(0.20 * K))
    best = None
    for s in range(K - n_local + 1):
        vis_local = vis[s:s + n_local].sum(axis=0)
        local_objs = vis_local >= 2
        if local_objs.sum() == 0:
            continue
        # score every temporally-distant keyframe by #local objects it also sees
        scores = np.zeros(K, dtype=int)
        for k in range(K):
            if s - gap_min < k < s + n_local - 1 + gap_min:
                continue
            scores[k] = int((vis[k] & local_objs).sum())
        order = np.argsort(-scores)
        loop = []
        for k in order:
            if scores[k] == 0:
                break
            loop.append(int(k))
            if len(loop) == n_loop:
                break
        if len(loop) < n_loop:
            continue
        vis_loop = vis[loop].sum(axis=0)
        n_persist = int((local_objs & (vis_loop >= 1)).sum())
        if best is None or n_persist > best[0]:
            win_pos = list(range(s, s + n_local)) + loop
            best = (n_persist, win_pos, vis_local, vis_loop)
    if best is None:
        # fallback (no distant loop found): first n_local+n_loop consecutive keyframes
        s = 0
        vis_local = vis[s:s + n_local].sum(axis=0)
        vis_loop = vis[s + n_local:s + n_local + n_loop].sum(axis=0)
        win_pos = list(range(s, s + n_local + n_loop))
        best = (int(((vis_local >= 2) & (vis_loop >= 1)).sum()), win_pos, vis_local, vis_loop)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_frames", type=int, default=5)
    ap.add_argument("--out", default="/root/autodl-tmp/psnr_tum/fig13_instance_grid_3x5")
    ap.add_argument("--max_objs", type=int, default=18)
    args = ap.parse_args()

    scenes = list(SCENES.keys())
    n = args.n_frames
    fig, axes = plt.subplots(len(scenes), n, figsize=(2.6 * n, 2.45 * len(scenes)))
    if len(scenes) == 1:
        axes = axes[None, :]
    for r in range(axes.shape[0]):
        for c in range(axes.shape[1]):
            axes[r, c].axis("off")

    for r, scene in enumerate(scenes):
        base, s3_rel, sem_rel, inst_rel = SCENES[scene]
        stage3 = np.load(f"{base}/{s3_rel}", allow_pickle=True)
        sem = np.load(f"{base}/{sem_rel}", allow_pickle=True)
        kf_indices = [int(x) for x in sem["kf_indices"].tolist()]
        kf_to_pos = {int(fi): k for k, fi in enumerate(kf_indices)}
        img_W, img_H = int(stage3["img_W"]), int(stage3["img_H"])
        objs = load_objs(base, inst_rel)
        # keep the most-visible foreground instances (cap for readability)
        objs = objs[:args.max_objs]
        print(f"[grid] {scene}: {len(objs)} foreground instances, {len(kf_indices)} keyframes")

        # precompute per-keyframe object visibility (centroid projection)
        K = len(kf_indices)
        vis = np.zeros((K, len(objs)), dtype=bool)
        for k, fi in enumerate(kf_indices):
            m, _, _ = visible_mask(objs, stage3[f"w2c_{fi}"].astype(np.float64),
                                   stage3[f"K_{fi}"].astype(np.float64), img_W, img_H)
            vis[k] = m

        score, win_pos, vis_local, vis_loop = pick_window(vis)
        win = [kf_indices[p] for p in win_pos]
        sel = [i for i in range(len(objs)) if vis_local[i] >= 2 and vis_loop[i] >= 1]
        print(f"[grid] {scene}: local kf={win[:3]}, loop kf={win[3:]}, "
              f"n_persist={score}, selected {len(sel)} objs "
              f"= {[(objs[i]['id'], objs[i]['name']) for i in sel]}")

        axes[r, 0].set_ylabel(scene, fontsize=10, rotation=0, labelpad=28,
                              va="center", color="#272727", fontweight="bold")

        for c, fi in enumerate(win):
            ax = axes[r, c]
            img = cv2.imread(str(stage3[f"img_path_{fi}"]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            w2c = stage3[f"w2c_{fi}"].astype(np.float64)
            K = stage3[f"K_{fi}"].astype(np.float64)
            H, W = img.shape[:2]

            # per-pixel semantic class map for this keyframe (mask source)
            lab = sem["labels"][kf_to_pos[fi]]
            if lab.shape[0] != H or lab.shape[1] != W:
                lab = cv2.resize(lab.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)

            m, u, v = visible_mask(objs, w2c, K, W, H)
            # pass 1: soft per-instance semantic masks (fig6 style: 55% image + 45% colour, no contour)
            canvas = img.astype(np.float32)
            for j in sel:
                if not m[j]:
                    continue
                colrgb = np.array(hex2rgb(TAB20[objs[j]["id"] % len(TAB20)]), np.float32)
                imask = instance_mask(lab, objs[j]["class"], u[j], v[j], W, H)
                if imask is None or not imask.any():
                    continue
                canvas[imask] = 0.55 * canvas[imask] + 0.45 * colrgb
            img = np.clip(canvas, 0, 255).astype(np.uint8)
            # pass 2: centroid dots + labels (on top of masks)
            for j in sel:
                if not m[j]:
                    continue
                colrgb = hex2rgb(TAB20[objs[j]["id"] % len(TAB20)])
                cu, cv_ = int(round(u[j])), int(round(v[j]))
                cv2.circle(img, (cu, cv_), 12, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (cu, cv_), 10, colrgb, -1, cv2.LINE_AA)
                draw_label(img, f"#{objs[j]['id']} {objs[j]['name']}", cu, cv_, colrgb, W, H)

            ax.imshow(img)
            if r == 0:
                ax.set_title(f"kf {fi}" if c < 3 else f"loop {fi}", fontsize=8, color="#272727")

    fig.suptitle(
        "Persistent instance identity: 3 scenes — 3 overlapping keyframes + 2 loop-closure frames; "
        "the same object keeps its #id + category across the loop\n(instance identity anchored to the "
        "globally-consistent world frame; category label from all-view voting)",
        fontsize=11, y=1.02)

    finalize_figure(fig, args.out)
    print(f"[grid] -> {args.out}.png")


if __name__ == "__main__":
    main()
