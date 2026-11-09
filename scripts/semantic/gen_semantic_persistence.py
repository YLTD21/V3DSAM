#!/usr/bin/env python3
"""Persistence proof: same physical object -> same world position, independent of
which views you observe it from (超级创新点 pilot 2 / "语义定位做好").

Claim being tested: 语义 + 坐标 = 持久实例身份。"同一个椅子，无论从哪个视角/时间看，
都能认出是当时那个。"  Because the SLAM world frame is globally consistent, a physical
object's 3D position is invariant.  Here we PROVE it empirically:

  split the keyframes into two disjoint halves (front / back of the trajectory),
  vote SAM3 labels onto the SAME world-frame cloud pts_gs using ONLY that half,
  cluster into instances independently,
  then match instances across halves by nearest centroid (greedy).

Result to report: match rate + centroid displacement between the same object
observed from disjoint view sets.  ~cm-level agreement => "位置=身份" 成立.
CPU-only (numpy + scipy). Reuses cluster_instances from gen_semantic_instances.
"""
import os, sys, argparse
import numpy as np
from scipy.optimize import linear_sum_assignment
from gen_semantic_pointcloud import load_stage3, subsample
from gen_semantic_instances import cluster_instances


def vote_labels_subset(pts, stage3, labels, kf_indices, positions, prompts, min_votes=2):
    """Same as gen_semantic_pointcloud.vote_labels but over a SUBSET of keyframes
    given by their POSITIONS (indices into labels / kf_indices)."""
    n_cats = len(prompts); H, W = labels.shape[1], labels.shape[2]; N = len(pts)
    pts_h = np.hstack([pts, np.ones((N, 1), dtype=np.float64)])
    all_pidx, all_lab = [], []
    for pos in positions:
        fi = int(kf_indices[pos])
        if f"w2c_{fi}" not in stage3.files or f"K_{fi}" not in stage3.files:
            continue
        w2c = stage3[f"w2c_{fi}"].astype(np.float64); K = stage3[f"K_{fi}"].astype(np.float64)
        cam = (w2c @ pts_h.T).T; z = cam[:, 2]; front = z > 0.05
        if not front.any():
            continue
        Xc, Yc, Zc = cam[:, 0], cam[:, 1], cam[:, 2]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        u = fx * Xc / Zc + cx; v = fy * Yc / Zc + cy
        ui = np.floor(u).astype(np.int64); vi = np.floor(v).astype(np.int64)
        inb = front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if not inb.any():
            continue
        lab = labels[pos][vi[inb], ui[inb]]
        keep = lab >= 0
        if not keep.any():
            continue
        pidx = np.nonzero(inb)[0][keep]
        all_pidx.append(pidx); all_lab.append(lab[keep])
    if not all_pidx:
        return np.full(N, -1, np.int32)
    flat = np.concatenate([p * n_cats + l for p, l in zip(all_pidx, all_lab)])
    votes = np.bincount(flat, minlength=N * n_cats).reshape(N, n_cats)
    total = votes.sum(1)
    sid = votes.argmax(1).astype(np.int32)
    sid[total < min_votes] = -1
    return sid


def match_instances(objs_a, objs_b, dist_thr=0.25):
    """Greedy nearest-centroid matching. Returns list of (obj_a, obj_b, dist)."""
    if not objs_a or not objs_b:
        return []
    Ca = np.stack([o["centroid"] for o in objs_a])
    Cb = np.stack([o["centroid"] for o in objs_b])
    D = np.linalg.norm(Ca[:, None, :] - Cb[None, :, :], axis=2)   # (A,B)
    rows, cols = linear_sum_assignment(D)
    pairs = []
    for r, c in zip(rows, cols):
        if D[r, c] <= dist_thr:
            pairs.append((objs_a[r], objs_b[c], float(D[r, c])))
    pairs.sort(key=lambda x: -x[0]["count"])
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage3", required=True)
    ap.add_argument("--semantic", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_pts", type=int, default=1000000)
    ap.add_argument("--voxel_size", type=float, default=0.04)
    ap.add_argument("--min_voxels", type=int, default=15)
    ap.add_argument("--dist_thr", type=float, default=0.25)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    stage3, pts, rgb = load_stage3(args.stage3)
    pts, rgb, _ = subsample(pts, rgb, args.max_pts)
    s = np.load(args.semantic, allow_pickle=True)
    labels = s["labels"]; kf_indices = [int(x) for x in s["kf_indices"].tolist()]
    prompts = [str(p) for p in s["prompts"]]
    K = len(kf_indices)
    half = K // 2
    front_pos = list(range(0, half))
    back_pos = list(range(half, K))
    print(f"[persistence] {K} keyframes -> front {len(front_pos)} / back {len(back_pos)}")

    sid_f = vote_labels_subset(pts, stage3, labels, kf_indices, front_pos, prompts)
    sid_b = vote_labels_subset(pts, stage3, labels, kf_indices, back_pos, prompts)
    print(f"[persistence] labeled: front {(sid_f>=0).sum():,}  back {(sid_b>=0).sum():,}")

    objs_f, _ = cluster_instances(pts, sid_f, voxel_size=args.voxel_size, min_voxels=args.min_voxels)
    objs_b, _ = cluster_instances(pts, sid_b, voxel_size=args.voxel_size, min_voxels=args.min_voxels)
    print(f"[persistence] instances: front {len(objs_f)}  back {len(objs_b)}")

    pairs = match_instances(objs_f, objs_b, dist_thr=args.dist_thr)
    dists = [d for _, _, d in pairs]
    cls_mismatch = sum(1 for a, b, _ in pairs if a["class"] != b["class"])

    print(f"\n{'='*86}")
    print(f"  {'front pts':>9} {'class':<14} {'centroid (front)':>22} {'back pts':>9} "
          f"{'class':<14} {'Δpos(m)':>8}")
    print(f"{'='*86}")
    for a, b, d in pairs[:40]:
        na = prompts[a["class"]] if a["class"] < len(prompts) else str(a["class"])
        nb = prompts[b["class"]] if b["class"] < len(prompts) else str(b["class"])
        print(f"  {a['count']:>9} {na:<14} ({a['centroid'][0]:.2f},{a['centroid'][1]:.2f},"
              f"{a['centroid'][2]:.2f}) {b['count']:>9} {nb:<14} {d:>8.3f}")
    print(f"{'='*86}")
    print(f"  matched {len(pairs)} instances;  Δpos mean={np.mean(dists):.3f}m "
          f"median={np.median(dists):.3f}m max={np.max(dists):.3f}m")
    print(f"  class mismatches among matched: {cls_mismatch}/{len(pairs)}")

    import json
    out = {
        "n_keyframes": K, "front_n": len(front_pos), "back_n": len(back_pos),
        "front_instances": len(objs_f), "back_instances": len(objs_b),
        "n_matched": len(pairs), "class_mismatches": cls_mismatch,
        "delta_mean": float(np.mean(dists)) if dists else None,
        "delta_median": float(np.median(dists)) if dists else None,
        "delta_max": float(np.max(dists)) if dists else None,
        "pairs": [{"front_id": a["id"], "back_id": b["id"], "class": a["class"],
                   "delta": d, "front_count": a["count"], "back_count": b["count"]}
                  for a, b, d in pairs],
        "prompts": prompts,
    }
    with open(os.path.join(args.out_dir, "persistence.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[persistence] DONE")


if __name__ == "__main__":
    main()
