#!/usr/bin/env python3
"""几何优先 ID 传播：身份 = 世界坐标，重观测按最近 3D 质心继承 ID（不靠语义）。

对比两种跨视角重识别：
  A. 类别过滤匹配（语义 gate，baseline）——类别不一致的匹配被丢弃
  B. 几何优先匹配（无类别过滤，最近质心）——身份只靠几何

分解失败模式：
  recovered_same   几何找到正确对象，且类别一致
  recovered_flip   几何找到正确对象，但类别翻转（类别过滤会误丢 ← 语义的代价）
  fragmented       30cm 内无对象但 60cm 内有（聚类碎片化）
  unseen           back 半幅完全没重建出该区域

anchor = 全视角 instance_cloud.ply 的固定实例（voted class）。
back 半幅独立聚类（随机交错分割，保证共视）。
"""
import os, argparse, json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from gen_object_ba import accumulate_votes
from gen_semantic_instances import cluster_instances
from gen_semantic_pointcloud import load_stage3, subsample


def load_ply(path):
    df = pd.read_csv(path, skiprows=11, sep=r"\s+", header=None, usecols=[0, 1, 2, 6], engine="c")
    return df[[0, 1, 2]].to_numpy(np.float64), df[6].to_numpy(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage3", required=True)
    ap.add_argument("--semantic", required=True)
    ap.add_argument("--instance_cloud", required=True)
    ap.add_argument("--voted_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--max_pts", type=int, default=500000)
    ap.add_argument("--voxel_size", type=float, default=0.04)
    ap.add_argument("--min_voxels", type=int, default=15)
    ap.add_argument("--min_votes", type=int, default=2)
    ap.add_argument("--dist_thr", type=float, default=0.30)
    args = ap.parse_args()

    stage3 = np.load(args.stage3, allow_pickle=True)
    sem = np.load(args.semantic, allow_pickle=True)
    labels = sem["labels"]
    kf_indices = [int(x) for x in sem["kf_indices"].tolist()]
    prompts = [str(p) for p in sem["prompts"]]
    K = len(kf_indices)

    # ── anchor objects (all-view fixed instances) ──
    pts_anchor, iid_anchor = load_ply(args.instance_cloud)
    n_inst = int(iid_anchor.max()) + 1
    voted = json.load(open(args.voted_json))
    voted_class = np.full(n_inst, -1, np.int64)
    for o in voted["instances"]:
        if o["id"] < n_inst:
            voted_class[o["id"]] = int(o["voted_class"])
    anc_cent = np.zeros((n_inst, 3))
    anc_count = np.zeros(n_inst, np.int64)
    for h in range(n_inst):
        m = iid_anchor == h
        anc_cent[h] = pts_anchor[m].mean(0)
        anc_count[h] = m.sum()

    # ── back half (random interleaved split, co-visible) ──
    rng = np.random.RandomState(0)
    perm = rng.permutation(K)
    back_pos = sorted(perm[K // 2:].tolist())

    _, pts, rgb = load_stage3(args.stage3)
    pts, rgb, _ = subsample(pts, rgb, args.max_pts)
    vb, tb, _ = accumulate_votes(pts, stage3, labels, kf_indices, back_pos)
    lab_b = np.where(tb >= args.min_votes, vb.argmax(1), -1).astype(np.int16)
    objs_b, _ = cluster_instances(pts, lab_b, voxel_size=args.voxel_size, min_voxels=args.min_voxels)
    back_cent = np.stack([o["centroid"] for o in objs_b])
    back_cls = np.array([o["class"] for o in objs_b])
    print(f"[idprop] anchor {n_inst} instances; back half -> {len(objs_b)} objects")

    # ── geometry-only matching: each anchor -> nearest back object ──
    stats = []
    for h in range(n_inst):
        if anc_count[h] < 5 or voted_class[h] < 0:
            continue
        d = np.linalg.norm(back_cent - anc_cent[h], axis=1)
        j = int(d.argmin()); dmin = float(d[j])
        if dmin <= args.dist_thr:
            outcome = "recovered_same" if back_cls[j] == voted_class[h] else "recovered_flip"
        elif dmin <= 2 * args.dist_thr:
            outcome = "fragmented"
        else:
            outcome = "unseen"
        stats.append({"id": h, "voted_class": int(voted_class[h]),
                      "voted_name": prompts[voted_class[h]],
                      "back_class": int(back_cls[j]),
                      "back_name": prompts[back_cls[j]],
                      "dmin": dmin, "outcome": outcome, "count": int(anc_count[h])})

    # ── class-filtered matching (baseline, Hungarian within class) ──
    # anchor objects that are "recovered_same/flip" are the truth-able set; here we
    # compute the class-filtered re-ID the same way gen_reid_benchmark does.
    recovered = [s for s in stats if s["outcome"] in ("recovered_same", "recovered_flip")]
    flip = [s for s in stats if s["outcome"] == "recovered_flip"]
    frag = [s for s in stats if s["outcome"] == "fragmented"]
    unseen = [s for s in stats if s["outcome"] == "unseen"]
    n_tot = len(stats)

    recovered_same = [s for s in stats if s["outcome"] == "recovered_same"]
    print(f"\n  {'outcome':<16} {'n':>4} {'frac':>7}")
    for nm, lst in [("recovered_same", recovered_same), ("recovered_flip", flip),
                    ("fragmented", frag), ("unseen", unseen)]:
        print(f"  {nm:<16} {len(lst):>4} {len(lst)/n_tot:>7.3f}")

    dmin_rec = [s["dmin"] for s in recovered]
    print(f"\n  geometry-only re-ID recall = {(len(recovered))/n_tot:.3f} "
          f"({len(recovered)}/{n_tot})")
    print(f"  class-filtered re-ID (would drop flips) = {len(recovered)-len(flip)}/{n_tot} "
          f"= {(len(recovered)-len(flip))/n_tot:.3f}")
    if dmin_rec:
        print(f"  recovered Δpos: mean={np.mean(dmin_rec):.4f}m median={np.median(dmin_rec):.4f}m "
              f"max={np.max(dmin_rec):.4f}m")

    print(f"\n  class-flip cases (geometry right, semantic wrong) — {len(flip)} objects "
          f"would be LOST by class-filtered matching:")
    for s in sorted(flip, key=lambda x: -x["count"]):
        print(f"    #{s['id']:<3} anchor={s['voted_name']:<12} back={s['back_name']:<12} "
              f"Δpos={s['dmin']*100:.1f}cm  n={s['count']}")

    out = {
        "n_anchor": n_inst, "n_tracked": n_tot,
        "n_recovered": len(recovered), "n_flip": len(flip),
        "n_fragmented": len(frag), "n_unseen": len(unseen),
        "recall_geom": len(recovered) / n_tot if n_tot else 0,
        "recall_classfiltered": (len(recovered) - len(flip)) / n_tot if n_tot else 0,
        "delta_median": float(np.median(dmin_rec)) if dmin_rec else None,
        "stats": stats,
        "prompts": prompts,
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[idprop] wrote {args.out_json}")


if __name__ == "__main__":
    main()
