#!/usr/bin/env python3
"""Persistent semantic instance map + cross-view re-identification (超级创新点 pilot 1).

Goal: recognise the SAME physical object from any viewpoint / time, by combining
SAM3 category labels with SLAM world-frame coordinates.  "语义 + 坐标 = 实例身份"
-- no appearance matching needed, because the world frame (w2c / pts_gs) is already
globally consistent after the SL(4) pose-graph optimisation.

Method:
  1. Vote SAM3 labels onto the world-frame Gaussian-init cloud pts_gs (reuse
     gen_semantic_pointcloud.vote_labels) -> per-point category in a CONSISTENT world frame.
  2. Cluster each category's 3D points into instances via voxel grid + connected
     components (scipy.ndimage.label, 26-connectivity).  Fast and exact -- replaces
     the greedy KDTree BFS, which is too slow on the dense aggregated cloud.
  3. Re-identify each instance across all keyframes: project its points into every
     keyframe (w2c + K), do a cheap z-buffer (keep nearest point per pixel = the
     visible surface), then take the dominant SAM3 label at those pixels.
     "recognised" iff dominant label == instance class.
  4. Report per-instance observation count + recognition consistency; render an
     instance-coloured 3D cloud + a montage of the same instance overlaid on several frames.

CPU-only (numpy + scipy + matplotlib). Same inputs as gen_semantic_pointcloud.
"""
import os, sys, argparse
import numpy as np
from scipy import ndimage
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# reuse the voting + loading from gen_semantic_pointcloud (same dir on AutoDL)
from gen_semantic_pointcloud import load_stage3, subsample, vote_labels, PALETTE

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
                     "font.size": 8, "savefig.dpi": 150, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02})


def cluster_instances(pts, labels, voxel_size=0.04, min_voxels=15):
    """Cluster labeled world-frame points into instances (voxel + connected components).

    pts:   (N,3) world frame. labels: (N,) int, -1 = unlabeled.
    Returns (objects, inst_id): objects = list of {class, centroid, extent, count,
    point_indices}; inst_id = (N,) per-point instance id (-1 = none).
    """
    labeled = labels >= 0
    pidx = np.nonzero(labeled)[0]                    # full indices of labeled points
    P = pts[labeled].astype(np.float64)
    L = labels[labeled].astype(np.int64)

    v = np.floor(P / voxel_size).astype(np.int64)
    vmin = v.min(0)
    v = v - vmin
    shape = tuple((v.max(0) + 1).tolist())
    flat = np.ravel_multi_index((v[:, 0], v[:, 1], v[:, 2]), shape)
    uniq_flat, inverse = np.unique(flat, return_inverse=True)   # inverse: point -> voxel
    n_vox = len(uniq_flat)

    # majority label per voxel (vectorized scatter)
    n_lab = int(L.max()) + 1
    votes = np.zeros((n_vox, n_lab), dtype=np.int64)
    np.add.at(votes, (inverse, L), 1)
    vlab = votes.argmax(1).astype(np.int16)

    vcoords = np.column_stack(np.unravel_index(uniq_flat, shape))   # (n_vox, 3)
    vox_inst = np.full(n_vox, -1, dtype=np.int64)

    inst_count = 0
    inst_class = []
    struct = np.ones((3, 3, 3), dtype=bool)
    for c in np.unique(vlab):
        if c < 0:
            continue
        m = vlab == c
        grid = np.zeros(shape, dtype=bool)
        grid[vcoords[m, 0], vcoords[m, 1], vcoords[m, 2]] = True
        lab_cc, n_cc = ndimage.label(grid, structure=struct)
        for cc in range(1, n_cc + 1):
            cc_coords = np.column_stack(np.nonzero(lab_cc == cc))
            if len(cc_coords) < min_voxels:
                continue
            cc_flat = np.ravel_multi_index((cc_coords[:, 0], cc_coords[:, 1], cc_coords[:, 2]), shape)
            vox_ids = np.searchsorted(uniq_flat, cc_flat)
            vox_inst[vox_ids] = inst_count
            inst_class.append(int(c))
            inst_count += 1

    point_inst = vox_inst[inverse]                   # per labeled point
    valid = point_inst >= 0
    n_inst = inst_count
    centers = np.zeros((n_inst, 3))
    mins = np.full((n_inst, 3), np.inf); maxs = np.full((n_inst, 3), -np.inf)
    cnt = np.bincount(point_inst[valid], minlength=n_inst)
    np.add.at(centers, point_inst[valid], P[valid])
    np.minimum.at(mins, point_inst[valid], P[valid])
    np.maximum.at(maxs, point_inst[valid], P[valid])
    centers /= np.maximum(cnt, 1)[:, None]
    extent = maxs - mins

    full_inst = np.full(len(pts), -1, dtype=np.int64)
    full_inst[pidx] = point_inst

    objects = []
    for i in range(n_inst):
        objects.append({
            "class": inst_class[i], "centroid": centers[i], "extent": extent[i],
            "count": int(cnt[i]), "point_indices": pidx[point_inst == i], "id": i,
        })
    return objects, full_inst


def reidentify(instances, pts, stage3, semantic, max_proj=200):
    """For each instance, project its points into every keyframe and check whether the
    visible surface (z-buffer: nearest point per pixel) is dominated by its class."""
    s = np.load(semantic, allow_pickle=True)
    labels = s["labels"]                       # (K, H, W) int16, -1 = unlabeled
    kf_indices = [int(x) for x in s["kf_indices"].tolist()]
    H, W = labels.shape[1], labels.shape[2]

    for obj in instances:
        gidx = obj["point_indices"]
        if len(gidx) > max_proj:
            gidx = np.random.RandomState(0).choice(gidx, max_proj, replace=False)
        p = pts[gidx].astype(np.float64)                    # (m,3) world
        p_h = np.hstack([p, np.ones((len(p), 1))])
        n_vis = 0; n_match = 0; vis_fi = []; match_fi = set()
        for k, fi in enumerate(kf_indices):
            if f"w2c_{fi}" not in stage3.files or f"K_{fi}" not in stage3.files:
                continue
            w2c = stage3[f"w2c_{fi}"].astype(np.float64)
            K = stage3[f"K_{fi}"].astype(np.float64)
            cam = (w2c @ p_h.T).T
            z = cam[:, 2]
            front = z > 0.05
            if front.sum() < 5:
                continue
            Xc, Yc, Zc = cam[front, 0], cam[front, 1], cam[front, 2]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            u = fx * Xc / Zc + cx
            v = fy * Yc / Zc + cy
            ui = np.floor(u).astype(np.int64)
            vi = np.floor(v).astype(np.int64)
            inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            if inb.sum() < 5:
                continue
            zi, uii, vii = z[front][inb], ui[inb], vi[inb]
            lab = labels[k][vii, uii]
            # z-buffer: keep nearest point per pixel (visible surface)
            flat = uii.astype(np.int64) * H + vii
            order = np.lexsort((zi, flat))
            sorted_flat, sorted_lab = flat[order], lab[order]
            first = np.concatenate(([True], sorted_flat[1:] != sorted_flat[:-1]))
            vis_lab = sorted_lab[first]
            vis_lab = vis_lab[vis_lab >= 0]
            if len(vis_lab) == 0:
                continue
            n_vis += 1
            vis_fi.append(fi)
            dominant = np.bincount(vis_lab).argmax()
            if dominant == obj["class"]:
                n_match += 1
                match_fi.add(fi)
        obj["n_vis"] = n_vis
        obj["n_match"] = n_match
        obj["consistency"] = (n_match / n_vis) if n_vis > 0 else 0.0
        obj["vis_fi"] = vis_fi
        obj["match_fi"] = match_fi
    return instances


def write_instance_cloud(path, pts, rgb, inst_id, n_inst):
    n = len(pts)
    cols = np.zeros((n, 3), np.uint8)
    assigned = inst_id >= 0
    cmap = plt.get_cmap("hsv")
    for iid in range(n_inst):
        cols[inst_id == iid] = (np.array(cmap((iid % max(1, n_inst)) / max(1, n_inst))[:3]) * 255).astype(np.uint8)
    cols[~assigned] = np.clip(rgb[~assigned] * 255, 0, 255).astype(np.uint8)  # unassigned -> true rgb
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property int instance_id\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b), iid in zip(pts, cols, inst_id):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)} {int(iid)}\n")
    print(f"[instances] wrote {n} points -> {path}")


def render_cloud(out_png, pts, inst_id, n_inst, n_show=250000):
    if len(pts) > n_show:
        idx = np.random.RandomState(0).choice(len(pts), n_show, replace=False)
        pts, inst_id = pts[idx], inst_id[idx]
    cmap = plt.get_cmap("hsv")
    cols = np.ones((len(pts), 3)) * 0.75
    assigned = inst_id >= 0
    for iid in range(n_inst):
        cols[inst_id == iid] = cmap((iid % max(1, n_inst)) / max(1, n_inst))[:3]
    fig = plt.figure(figsize=(13, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=0.3, alpha=0.7,
               linewidths=0, rasterized=True)
    ax.set_title(f"Persistent semantic instances (N={n_inst}, colour = instance id)", fontsize=10)
    ax.view_init(elev=30, azim=-60)
    mid = pts.mean(0); rng = np.ptp(pts, 0).max() / 2.0
    ax.set_xlim(mid[0]-rng, mid[0]+rng); ax.set_ylim(mid[1]-rng, mid[1]+rng)
    ax.set_zlim(mid[2]-rng, mid[2]+rng)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[instances] cloud figure -> {out_png}")


def render_reid(out_png, chosen, pts, stage3, semantic, prompts):
    """Montage: same instance overlaid (z-buffer projected points) on several frames."""
    s = np.load(semantic, allow_pickle=True)
    labels = s["labels"]; kf_indices = [int(x) for x in s["kf_indices"].tolist()]
    H, W = labels.shape[1], labels.shape[2]
    n_inst = len(chosen)
    n_view = 4
    fig, axes = plt.subplots(n_inst, n_view, figsize=(3.2 * n_view, 3.0 * n_inst))
    if n_inst == 1:
        axes = axes[None, :]
    for r, obj in enumerate(chosen):
        vis = obj["vis_fi"]
        if len(vis) == 0:
            continue
        pick = np.linspace(0, len(vis) - 1, n_view).astype(int)
        gidx = obj["point_indices"]
        if len(gidx) > 200:
            gidx = np.random.RandomState(0).choice(gidx, 200, replace=False)
        p = pts[gidx].astype(np.float64)
        p_h = np.hstack([p, np.ones((len(p), 1))])
        cls_name = prompts[obj["class"]] if obj["class"] < len(prompts) else str(obj["class"])
        for c, vi in enumerate(pick):
            fi = vis[vi]
            ax = axes[r][c]
            img = cv2.imread(str(stage3[f"img_path_{fi}"]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            w2c = stage3[f"w2c_{fi}"].astype(np.float64)
            K = stage3[f"K_{fi}"].astype(np.float64)
            cam = (w2c @ p_h.T).T
            z = cam[:, 2]; front = z > 0.05
            Xc, Yc, Zc = cam[front, 0], cam[front, 1], cam[front, 2]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            u = fx * Xc / Zc + cx; v = fy * Yc / Zc + cy
            ui = np.floor(u).astype(int); vi = np.floor(v).astype(int)
            inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            ax.scatter(ui[inb], vi[inb], c="red", s=2, alpha=0.6, linewidths=0)
            ok = "recognised" if (fi in obj["match_fi"]) else "missed"
            ax.set_title(f"frame {fi}\n{cls_name} {ok}", fontsize=7)
            ax.axis("off")
    fig.suptitle("Cross-view re-identification: the SAME instance (red points) recognised across views",
                 fontsize=10)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[instances] reid figure -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage3", required=True)
    ap.add_argument("--semantic", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_pts", type=int, default=1000000)
    ap.add_argument("--voxel_size", type=float, default=0.04)
    ap.add_argument("--min_voxels", type=int, default=15)
    ap.add_argument("--n_show", type=int, default=250000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    stage3, pts, rgb = load_stage3(args.stage3)
    print(f"[instances] loaded {len(pts):,} world-frame points")
    pts, rgb, _ = subsample(pts, rgb, args.max_pts)
    print(f"[instances] subsampled to {len(pts):,} points")

    sid, prompts, n_seen = vote_labels(pts, stage3, args.semantic, min_votes=2)
    n_lab = (sid >= 0).sum()
    print(f"[instances] voted over {n_seen} keyframes; {n_lab:,}/{len(pts):,} labeled")

    objs, inst_id = cluster_instances(pts, sid, voxel_size=args.voxel_size, min_voxels=args.min_voxels)
    print(f"[instances] {len(objs)} instances clustered (voxel={args.voxel_size}, min_voxels={args.min_voxels})")

    # sort by size desc, remap ids so inst_id stays consistent
    order = np.argsort([-o["count"] for o in objs])
    remap = np.empty(len(objs), dtype=np.int64)
    for new_id, old_id in enumerate(order):
        remap[old_id] = new_id
    inst_id = np.where(inst_id >= 0, remap[inst_id], -1).astype(np.int32)
    objs = [objs[old_id] for old_id in order]
    for new_id, o in enumerate(objs):
        o["id"] = new_id

    objs = reidentify(objs, pts, stage3, args.semantic, max_proj=200)

    # ── report ──
    print(f"\n{'='*82}")
    print(f"  {'id':>3}  {'class':<16} {'n_pts':>7} {'centroid':>20} {'n_vis':>6} {'n_match':>7} {'cons':>6}")
    print(f"{'='*82}")
    for o in objs:
        cls_name = prompts[o["class"]] if o["class"] < len(prompts) else str(o["class"])
        print(f"  {o['id']:>3}  {cls_name:<16} {o['count']:>7} "
              f"({o['centroid'][0]:.2f},{o['centroid'][1]:.2f},{o['centroid'][2]:.2f}) "
              f"{o['n_vis']:>6} {o['n_match']:>7} {o['consistency']:.2f}")
    vis_objs = [o for o in objs if o["n_vis"] >= 3]
    avg_cons = float(np.mean([o["consistency"] for o in vis_objs])) if vis_objs else 0.0
    print(f"{'='*82}")
    print(f"  instances with >=3 visible views: {len(vis_objs)}/{len(objs)};  mean consistency = {avg_cons:.3f}")

    # ── outputs ──
    import json
    summary = [{"id": o["id"], "class": o["class"], "count": o["count"],
                "centroid": o["centroid"].tolist(), "extent": o["extent"].tolist(),
                "n_vis": o["n_vis"], "n_match": o["n_match"],
                "consistency": o["consistency"]} for o in objs]
    with open(os.path.join(args.out_dir, "instances.json"), "w") as f:
        json.dump({"n_instances": len(objs), "mean_consistency": avg_cons,
                   "prompts": prompts, "instances": summary}, f, indent=2)
    write_instance_cloud(os.path.join(args.out_dir, "instance_cloud.ply"), pts, rgb, inst_id, len(objs))
    render_cloud(os.path.join(args.out_dir, "instance_cloud.png"), pts, inst_id, len(objs), args.n_show)

    chosen = sorted(vis_objs, key=lambda o: -o["n_vis"])[:3]
    if chosen:
        render_reid(os.path.join(args.out_dir, "instance_reid.png"), chosen, pts, stage3, args.semantic, prompts)

    print("[instances] DONE")


if __name__ == "__main__":
    main()
