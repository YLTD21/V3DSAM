#!/usr/bin/env python3
"""Object-level semantic BA: persistent instance ID anchored to the SL(4) global frame.

What this adds over gen_semantic_instances.py (per-point voting + clustering):
  1. Occlusion-aware voting — when projecting world points into a keyframe, keep only
     the nearest point per pixel (z-buffer); occluded points do NOT vote.  (改进 C)
  2. Object-level voting — each object's category is the majority over ALL its member
     points' full vote histograms (not the per-point argmax), so a whole object votes
     as a unit.  This is the "reverse-calibration" the user asked about: the object's
     (category, position) is estimated from every view, in the globally-consistent frame.
  3. Object-level confidence — fraction of (point, view) observations agreeing with the
     object's consensus category.  Compare vs 51% single-view point-level reliability.
  4. Persistent ID map + cross-view re-ID (random-split two halves).

CPU-only. Reuses load_stage3/subsample (gen_semantic_pointcloud) and
cluster_instances (gen_semantic_instances).
"""
import os, argparse, json
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gen_semantic_pointcloud import load_stage3, subsample
from gen_semantic_instances import cluster_instances

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
                     "font.size": 8, "savefig.dpi": 150, "savefig.bbox": "tight"})


def accumulate_votes(pts, stage3, labels, kf_indices, positions=None, occlusion=True):
    """Per-point vote histogram over keyframes at `positions` (default: all).

    occlusion=True  -> per-pixel z-buffer, only the visible (nearest) point votes.
    Returns votes (N, n_cats), total (N,) = # views each point voted.
    """
    n_cats = int(labels.max()) + 1
    H, W = labels.shape[1], labels.shape[2]
    N = len(pts)
    pts_h = np.hstack([pts, np.ones((N, 1), dtype=np.float64)])
    votes = np.zeros((N, n_cats), dtype=np.int64)
    n_seen = 0
    if positions is None:
        positions = list(range(len(kf_indices)))
    for pos in positions:
        fi = int(kf_indices[pos])
        if f"w2c_{fi}" not in stage3.files or f"K_{fi}" not in stage3.files:
            continue
        w2c = stage3[f"w2c_{fi}"].astype(np.float64)
        K = stage3[f"K_{fi}"].astype(np.float64)
        cam = (w2c @ pts_h.T).T
        z = cam[:, 2]
        front = z > 0.05
        if front.sum() < 5:
            continue
        Xc, Yc, Zc = cam[:, 0], cam[:, 1], cam[:, 2]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        u = fx * Xc / Zc + cx
        v = fy * Yc / Zc + cy
        ui = np.floor(u).astype(np.int64)
        vi = np.floor(v).astype(np.int64)
        inb = front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if inb.sum() < 5:
            continue
        zi, uii, vii = z[inb], ui[inb], vi[inb]
        lab = labels[pos][vii, uii]
        keep = lab >= 0
        if not keep.any():
            continue
        zi, uii, vii, lab = zi[keep], uii[keep], vii[keep], lab[keep]
        pidx = np.nonzero(inb)[0][keep]
        if occlusion:
            # z-buffer: nearest point per pixel is the visible surface
            flat = uii.astype(np.int64) * H + vii
            order = np.lexsort((zi, flat))
            s_flat = flat[order]
            first = np.concatenate(([True], s_flat[1:] != s_flat[:-1]))
            pidx, lab = pidx[order][first], lab[order][first]
        np.add.at(votes, (pidx, lab), 1)
        n_seen += 1
    total = votes.sum(1)
    return votes, total, n_seen


def object_level_vote(objs, votes, total, prompts):
    """Aggregate each object's member-point vote histograms -> object category + confidence."""
    for o in objs:
        idx = o["point_indices"]
        V = votes[idx].sum(0)
        s = int(V.sum())
        if s > 0:
            o["category"] = int(V.argmax())
            o["confidence"] = float(V.max() / s)
            o["n_obs"] = s
            o["vote_hist"] = V.astype(np.int64)
        else:
            o["category"] = o["class"]
            o["confidence"] = 0.0
            o["n_obs"] = 0
    return objs


def same_class_match(objs_f, objs_b, dist_thr=0.30):
    """Nearest-centroid matching WITHIN each category (Hungarian, per category)."""
    cats = sorted(set([o["category"] for o in objs_f] + [o["category"] for o in objs_b]))
    pairs = []
    per_cat = {}
    for c in cats:
        A = [o for o in objs_f if o["category"] == c]
        B = [o for o in objs_b if o["category"] == c]
        if not A or not B:
            continue
        Ca = np.stack([o["centroid"] for o in A])
        Cb = np.stack([o["centroid"] for o in B])
        D = np.linalg.norm(Ca[:, None, :] - Cb[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(D)
        for r, cc in zip(rows, cols):
            if D[r, cc] <= dist_thr:
                pairs.append((A[r], B[cc], float(D[r, cc])))
        per_cat[c] = {"front_n": len(A), "back_n": len(B), "matched": len(pairs) if False else sum(
            1 for rr, cc2 in zip(rows, cols) if D[rr, cc2] <= dist_thr)}
    return pairs, per_cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage3", required=True)
    ap.add_argument("--semantic", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_pts", type=int, default=500000)
    ap.add_argument("--voxel_size", type=float, default=0.04)
    ap.add_argument("--min_voxels", type=int, default=15)
    ap.add_argument("--min_votes", type=int, default=2)
    ap.add_argument("--dist_thr", type=float, default=0.30)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    stage3, pts, rgb = load_stage3(args.stage3)
    pts, rgb, _ = subsample(pts, rgb, args.max_pts)
    s = np.load(args.semantic, allow_pickle=True)
    labels = s["labels"]
    kf_indices = [int(x) for x in s["kf_indices"].tolist()]
    prompts = [str(p) for p in s["prompts"]]
    K = len(kf_indices)

    # ── full-map object-level voting ──
    votes, total, n_seen = accumulate_votes(pts, stage3, labels, kf_indices)
    labeled = total >= args.min_votes
    per_point_label = np.where(labeled, votes.argmax(1), -1).astype(np.int16)
    print(f"[objba] {n_seen} keyframes voted; {labeled.sum():,}/{len(pts):,} labeled (>= {args.min_votes} views)")

    objs, inst_id = cluster_instances(pts, per_point_label, voxel_size=args.voxel_size,
                                      min_voxels=args.min_voxels)
    objs = object_level_vote(objs, votes, total, prompts)
    print(f"[objba] {len(objs)} objects after object-level voting")

    # object-level confidence distribution
    conf = np.array([o["confidence"] for o in objs])
    print(f"\n  object-level category confidence: mean={conf.mean():.3f} median={np.median(conf):.3f} "
          f"p10={np.percentile(conf,10):.3f} p90={np.percentile(conf,90):.3f}")
    print(f"  (contrast: single-view point-level reliability = 0.51 from denoise.json)")

    # ── two-halves re-ID (random split, object-level) ──
    rng = np.random.RandomState(0)
    perm = rng.permutation(K)
    front_pos = sorted(perm[:K // 2].tolist())
    back_pos = sorted(perm[K // 2:].tolist())
    vf, tf, _ = accumulate_votes(pts, stage3, labels, kf_indices, front_pos)
    vb, tb, _ = accumulate_votes(pts, stage3, labels, kf_indices, back_pos)
    lab_f = np.where(tf >= args.min_votes, vf.argmax(1), -1).astype(np.int16)
    lab_b = np.where(tb >= args.min_votes, vb.argmax(1), -1).astype(np.int16)
    objs_f, _ = cluster_instances(pts, lab_f, voxel_size=args.voxel_size, min_voxels=args.min_voxels)
    objs_b, _ = cluster_instances(pts, lab_b, voxel_size=args.voxel_size, min_voxels=args.min_voxels)
    objs_f = object_level_vote(objs_f, vf, tf, prompts)
    objs_b = object_level_vote(objs_b, vb, tb, prompts)

    pairs, per_cat = same_class_match(objs_f, objs_b, dist_thr=args.dist_thr)
    dists = [d for _, _, d in pairs]
    n_should = sum(min(pc["front_n"], pc["back_n"]) for pc in per_cat.values())
    acc = len(pairs) / n_should if n_should else 0.0
    print(f"\n  two-halves (random split, object-level): front {len(objs_f)} / back {len(objs_b)} objects")
    print(f"  same-category re-ID: {len(pairs)}/{n_should} = {acc:.3f}")
    if dists:
        print(f"  matched Δpos: mean={np.mean(dists):.4f}m median={np.median(dists):.4f}m "
              f"max={np.max(dists):.4f}m (n={len(dists)})")

    # ── outputs ──
    out = {
        "n_points": len(pts), "n_labeled": int(labeled.sum()), "n_objects": len(objs),
        "object_confidence_mean": float(conf.mean()),
        "object_confidence_median": float(np.median(conf)),
        "single_view_point_reliability_ref": 0.51,
        "reid": {"n_pairs": len(pairs), "n_should": n_should, "accuracy": float(acc),
                 "delta_median": float(np.median(dists)) if dists else None},
        "objects": [{"id": o["id"], "category": o["category"],
                     "category_name": prompts[o["category"]] if o["category"] < len(prompts) else str(o["category"]),
                     "centroid": o["centroid"].tolist(), "extent": o["extent"].tolist(),
                     "count": o["count"], "confidence": o["confidence"], "n_obs": o["n_obs"]}
                    for o in sorted(objs, key=lambda x: -x["count"])],
        "prompts": prompts,
    }
    with open(os.path.join(args.out_dir, "object_ba.json"), "w") as f:
        json.dump(out, f, indent=2)

    # per-category confidence table
    cat_conf = {}
    for o in objs:
        cat_conf.setdefault(o["category"], []).append(o["confidence"])
    print(f"\n  {'category':<16} {'#obj':>5} {'mean_conf':>10}")
    for c in sorted(cat_conf, key=lambda c: -len(cat_conf[c])):
        nm = prompts[c] if c < len(prompts) else str(c)
        print(f"  {nm:<16} {len(cat_conf[c]):>5} {np.mean(cat_conf[c]):>10.3f}")

    # colored object map (id = colour)
    cmap = plt.get_cmap("hsv")
    cols = np.zeros((len(pts), 3))
    assigned = inst_id >= 0
    for iid in range(len(objs)):
        cols[inst_id == iid] = cmap((iid % max(1, len(objs))) / max(1, len(objs)))[:3]
    cols[~assigned] = 0.8
    fig = plt.figure(figsize=(11, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    idx = np.random.RandomState(0).choice(len(pts), min(len(pts), 250000), replace=False)
    ax.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2], c=cols[idx], s=0.3, alpha=0.7,
               linewidths=0, rasterized=True)
    ax.set_title(f"Persistent object-ID map (object-level voting, N={len(objs)}, colour = ID)", fontsize=10)
    ax.view_init(elev=30, azim=-60)
    mid = pts.mean(0); rng = np.ptp(pts, 0).max() / 2.0
    ax.set_xlim(mid[0]-rng, mid[0]+rng); ax.set_ylim(mid[1]-rng, mid[1]+rng); ax.set_zlim(mid[2]-rng, mid[2]+rng)
    fig.savefig(os.path.join(args.out_dir, "object_map.png"))
    plt.close(fig)

    print(f"[objba] DONE -> {args.out_dir}/object_ba.json + object_map.png")


if __name__ == "__main__":
    main()
