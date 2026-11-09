#!/usr/bin/env python3
"""E4: appearance-based instance re-ID baseline (reviewer 问题14 推荐实验).

在 gen_id_propagate 的同一 anchor/back 两半分割下，新增「外观匹配」策略：
每个实例取 RGB 颜色直方图(4^3=64维归一化)作为外观描述子，anchor 实例与
back 对象按外观 L2 最近匹配（不用质心、不用类别门控）。

三种策略同台对比（同一 anchor 集合、同一 back 半幅）：
  A. 几何优先 geometry-first : 最近质心 <= 30cm          （本文方法，已知 89.5%/86.6%）
  B. 类别硬门控 category-gate : 几何匹配但要求类别一致     （已知 59.2%/52.2%）
  C. 外观匹配 appearance     : 最近颜色直方图，且该对象 <= 30cm（新 baseline）

C 回答：外观能否找到"正确的那一个对象"（即最近颜色匹配的对象是否恰是几何正确匹配）。
"""
import os, argparse, json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from gen_object_ba import accumulate_votes
from gen_semantic_instances import cluster_instances
from gen_semantic_pointcloud import load_stage3, subsample


def load_ply_color(path):
    df = pd.read_csv(path, skiprows=11, sep=r"\s+", header=None,
                     usecols=[0, 1, 2, 3, 4, 5, 6], engine="c")
    xyz = df[[0, 1, 2]].to_numpy(np.float64)
    rgb = df[[3, 4, 5]].to_numpy(np.float64) / 255.0
    iid = df[6].to_numpy(np.int64)
    return xyz, rgb, iid


def rgb_hist(rgb, nbins=4):
    """归一化 RGB 联合直方图 (nbins^3 维)。rgb: (N,3) in [0,1]."""
    b = np.clip((rgb * nbins).astype(np.int64), 0, nbins - 1)
    idx = b[:, 0] * nbins * nbins + b[:, 1] * nbins + b[:, 2]
    h = np.bincount(idx, minlength=nbins ** 3).astype(np.float64)
    s = h.sum()
    return h / s if s > 0 else h


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
    ap.add_argument("--nbins", type=int, default=4)
    args = ap.parse_args()

    # ── anchor instances (with color) ──
    pts_anchor, rgb_anchor, iid_anchor = load_ply_color(args.instance_cloud)
    n_inst = int(iid_anchor.max()) + 1
    voted = json.load(open(args.voted_json))
    voted_class = np.full(n_inst, -1, np.int64)
    for o in voted["instances"]:
        if o["id"] < n_inst:
            voted_class[o["id"]] = int(o["voted_class"])
    anc_cent = np.zeros((n_inst, 3))
    anc_count = np.zeros(n_inst, np.int64)
    anc_app = np.zeros((n_inst, args.nbins ** 3))
    for h in range(n_inst):
        m = iid_anchor == h
        anc_cent[h] = pts_anchor[m].mean(0)
        anc_count[h] = m.sum()
        anc_app[h] = rgb_hist(rgb_anchor[m], args.nbins)

    # ── back half (random interleaved split, co-visible; 与 id_propagate 相同 seed) ──
    stage3 = np.load(args.stage3, allow_pickle=True)
    sem = np.load(args.semantic, allow_pickle=True)
    labels = sem["labels"]
    kf_indices = [int(x) for x in sem["kf_indices"].tolist()]
    prompts = [str(p) for p in sem["prompts"]]
    K = len(kf_indices)
    rng = np.random.RandomState(0)
    perm = rng.permutation(K)
    back_pos = sorted(perm[K // 2:].tolist())

    _, pts, rgb = load_stage3(args.stage3)
    pts, rgb, _ = subsample(pts, rgb, args.max_pts)
    vb, tb, _ = accumulate_votes(pts, stage3, labels, kf_indices, back_pos)
    lab_b = np.where(tb >= args.min_votes, vb.argmax(1), -1).astype(np.int16)
    objs_b, _ = cluster_instances(pts, lab_b, voxel_size=args.voxel_size,
                                  min_voxels=args.min_voxels)
    back_cent = np.stack([o["centroid"] for o in objs_b])
    back_cls = np.array([o["class"] for o in objs_b])
    back_app = np.stack([rgb_hist(rgb[o["point_indices"]], args.nbins) for o in objs_b])
    print(f"[appear] anchor {n_inst} instances; back half -> {len(objs_b)} objects")

    # ── tracked anchors (same filter as id_propagate) ──
    tracked = [h for h in range(n_inst) if anc_count[h] >= 5 and voted_class[h] >= 0]
    n_tot = len(tracked)

    geo_ok = 0          # geometry: nearest centroid within thr
    geo_flip = 0        # geometry right but class differs
    app_ok = 0          # appearance: nearest-color object also within thr (== same object)
    app_geom_match = 0  # appearance and geometry pick the SAME back object
    details = []
    for h in tracked:
        # geometry
        dg = np.linalg.norm(back_cent - anc_cent[h], axis=1)
        jg = int(dg.argmin()); dgmin = float(dg[jg])
        g_ok = dgmin <= args.dist_thr
        g_flip = g_ok and (back_cls[jg] != voted_class[h])
        if g_ok:
            geo_ok += 1
        if g_flip:
            geo_flip += 1
        # appearance
        da = np.linalg.norm(back_app - anc_app[h], axis=1)
        ja = int(da.argmin()); damin = float(da[ja])
        a_ok = np.linalg.norm(back_cent[ja] - anc_cent[h]) <= args.dist_thr
        if a_ok:
            app_ok += 1
        if ja == jg:
            app_geom_match += 1
        details.append({"id": h, "voted_name": prompts[voted_class[h]],
                        "geo_name": prompts[back_cls[jg]], "dgmin": dgmin,
                        "app_name": prompts[back_cls[ja]],
                        "app_geom_dist": float(np.linalg.norm(back_cent[ja] - anc_cent[h])),
                        "geo_ok": bool(g_ok), "geo_flip": bool(g_flip), "app_ok": bool(a_ok),
                        "same_obj": bool(ja == jg), "count": int(anc_count[h])})

    rec_geom = geo_ok / n_tot
    rec_class = (geo_ok - geo_flip) / n_tot
    rec_app = app_ok / n_tot
    same_rate = app_geom_match / n_tot

    print(f"\n  tracked anchors = {n_tot}")
    print(f"  A. geometry-first  recall = {rec_geom:.3f}  ({geo_ok}/{n_tot})")
    print(f"  B. category-gate   recall = {rec_class:.3f}  ({geo_ok-geo_flip}/{n_tot})  [drops {geo_flip} flips]")
    print(f"  C. appearance      recall = {rec_app:.3f}  ({app_ok}/{n_tot})")
    print(f"  appearance picks SAME object as geometry = {same_rate:.3f} ({app_geom_match}/{n_tot})")

    # appearance 找错对象时，到底错在哪（几何距离分布）
    wrong = [d for d in details if not d["app_ok"]]
    if wrong:
        gd = np.array([d["app_geom_dist"] for d in wrong])
        print(f"  appearance misses: n={len(wrong)}, 该对象的几何距离 mean={gd.mean():.2f}m "
              f"median={np.median(gd):.2f}m (>{args.dist_thr} => 找错对象)")

    out = {
        "n_anchor": n_inst, "n_tracked": n_tot,
        "recall_geometry_first": rec_geom,
        "recall_category_gate": rec_class,
        "recall_appearance": rec_app,
        "appearance_same_as_geometry_rate": same_rate,
        "n_appearance_miss": len(wrong),
        "prompts": prompts,
        "details": details,
    }
    json.dump(out, open(args.out_json, "w"), indent=2, ensure_ascii=False)
    print(f"[appear] wrote {args.out_json}")


if __name__ == "__main__":
    main()
