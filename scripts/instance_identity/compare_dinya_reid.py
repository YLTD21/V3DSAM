#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dinya 几何 re-ID head-to-head：复现「3D bbox IoU + median Chamfer, 阈值 30cm」，
对比本文 SL(4) 全局一致世界坐标锚定（~7cm）。

Dinya (arXiv 2511.16282) 的实例 re-ID = 3D bbox IoU + 中值点 Chamfer，阈值 30cm，
因为它用 rolling Sim3（无回环、无 SL(4)），对象位置会漂移 ~30cm。
本文 = SL(4)+回环全局一致，3D 聚类锚定，两半视图质心位移 ~7cm。

要证明的一层：同为几何锚定，本文 7cm 比竞品 30cm 紧 ~4×，
因此在更近的同类实例（7cm<Δ<30cm）之间，本文能分、竞品会混。

输出：
  1. 本文锚定精度 = 两半视图质心位移 median（~7cm）。
  2. Dinya re-ID（复现）= bbox IoU + median Chamfer，阈值 30cm。
  3. 同类实例对的距离分布：本文可分(Δ>7cm) vs 竞品可分(Δ>30cm)，
     统计落在 (7cm,30cm] 的「竞品混、本文分」的实例对数。

Run on AutoDL (sgs env):
  cd /root/autodl-tmp
  /root/miniconda3/envs/sgs/bin/python -u compare_dinya_reid.py \
    --stage3 /root/autodl-tmp/psnr_tum/room/stage1_scaled/_stage3_input.npz \
    --semantic /root/autodl-tmp/psnr_tum/room/stage1_scaled/kf_semantic.npz \
    --out /root/autodl-tmp/psnr_tum/room/dinya_reid
"""
import os, argparse, json
import numpy as np
from scipy.optimize import linear_sum_assignment
from gen_semantic_pointcloud import load_stage3, subsample
from gen_semantic_instances import cluster_instances
from gen_semantic_persistence import vote_labels_subset

# TUM 结构类（非"实例身份"意义上的前景对象），不计入同类实例区分统计
BACKGROUND_TUM = {"wall", "floor", "ceiling", "stairs", "railing"}


def chamfer_median(A, B, max_n=5000):
    """两片点云的中值 Chamfer 距离（Dinya 用的中值点 Chamfer）。"""
    if len(A) == 0 or len(B) == 0:
        return float("inf")
    if len(A) > max_n:
        A = A[np.random.RandomState(0).choice(len(A), max_n, replace=False)]
    if len(B) > max_n:
        B = B[np.random.RandomState(0).choice(len(B), max_n, replace=False)]
    d_ab = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2).min(1)
    d_ba = np.linalg.norm(B[:, None, :] - A[None, :, :], axis=2).min(1)
    return float(np.median(np.concatenate([d_ab, d_ba])))


def bbox_iou(o1, o2):
    lo1, hi1 = o1["lo"], o1["hi"]
    lo2, hi2 = o2["lo"], o2["hi"]
    inter = np.maximum(0.0, np.minimum(hi1, hi2) - np.maximum(lo1, lo2))
    v = float(inter[0] * inter[1] * inter[2])
    if v <= 0:
        return 0.0
    v1 = float(np.prod(hi1 - lo1)); v2 = float(np.prod(hi2 - lo2))
    return v / (v1 + v2 - v)


def add_bbox(objs, pts):
    for o in objs:
        p = pts[o["point_indices"]]
        o["lo"] = p.min(0); o["hi"] = p.max(0)
        o["points"] = p
    return objs


def two_half_anchoring(pts, stage3, labels, kf_indices, prompts, split="random"):
    """本文锚定精度：两半视图独立投票聚类，同类最近质心匹配的 Δpos。"""
    K = len(kf_indices)
    if split == "random":
        rng = np.random.RandomState(0)
        perm = rng.permutation(K)
        front = sorted(perm[:K // 2].tolist()); back = sorted(perm[K // 2:].tolist())
    else:
        front = list(range(0, K // 2)); back = list(range(K // 2, K))
    sid_f = vote_labels_subset(pts, stage3, labels, kf_indices, front, prompts)
    sid_b = vote_labels_subset(pts, stage3, labels, kf_indices, back, prompts)
    objs_f, _ = cluster_instances(pts, sid_f, voxel_size=0.04, min_voxels=15)
    objs_b, _ = cluster_instances(pts, sid_b, voxel_size=0.04, min_voxels=15)
    # 同类最近质心匹配（匈牙利，逐类别）
    classes = sorted(set([o["class"] for o in objs_f] + [o["class"] for o in objs_b]))
    deltas = []
    for c in classes:
        A = [o for o in objs_f if o["class"] == c]
        B = [o for o in objs_b if o["class"] == c]
        if not A or not B:
            continue
        Ca = np.stack([o["centroid"] for o in A]); Cb = np.stack([o["centroid"] for o in B])
        D = np.linalg.norm(Ca[:, None, :] - Cb[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(D)
        for r, cc in zip(rows, cols):
            if D[r, cc] <= 0.30:
                deltas.append(float(D[r, cc]))
    return np.array(deltas) if deltas else np.array([])


def same_class_pairs(objs, prompts, min_count=100):
    """同类前景实例两两配对的质心距离（考验区分能力的实例对）。

    过滤：结构类（wall/floor/ceiling/stairs/railing）不计；碎块（count<min_count）不计，
    避免过分割把同一物体劈成两半造成的伪"同类对"。"""
    by_cls = {}
    for o in objs:
        if prompts[o["class"]] in BACKGROUND_TUM:
            continue
        if o["count"] < min_count:
            continue
        by_cls.setdefault(o["class"], []).append(o)
    pairs = []
    for c, L in by_cls.items():
        if len(L) < 2:
            continue
        C = np.stack([o["centroid"] for o in L])
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                pairs.append((c, float(np.linalg.norm(C[i] - C[j]))))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage3", required=True)
    ap.add_argument("--semantic", required=True)
    ap.add_argument("--out", default="/root/autodl-tmp/psnr_tum/room/dinya_reid")
    ap.add_argument("--max_pts", type=int, default=1000000)
    ap.add_argument("--thr_dinya", type=float, default=0.30,
                    help="Dinya re-ID 的 Chamfer 阈值(m)")
    ap.add_argument("--thr_ours", type=float, default=0.07,
                    help="本文锚定精度(m)，用于对照")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    stage3, pts, rgb = load_stage3(args.stage3)
    pts, rgb, _ = subsample(pts, rgb, args.max_pts)
    s = np.load(args.semantic, allow_pickle=True)
    labels = s["labels"]; kf_indices = [int(x) for x in s["kf_indices"].tolist()]
    prompts = [str(p) for p in s["prompts"]]
    K = len(kf_indices)

    # 1) 本文锚定精度：两半视图质心位移
    deltas = two_half_anchoring(pts, stage3, labels, kf_indices, prompts)
    ours_median = float(np.median(deltas)) if len(deltas) else float("nan")
    print(f"[ours] 两半视图锚定 Δpos: n={len(deltas)} "
          f"mean={np.mean(deltas):.4f}m median={ours_median:.4f}m")

    # 2) 全量实例 + bbox（供 Dinya re-ID 复现 + 同类对距离）
    sid_full = vote_labels_subset(pts, stage3, labels, kf_indices,
                                  list(range(K)), prompts)
    objs_full, _ = cluster_instances(pts, sid_full, voxel_size=0.04, min_voxels=15)
    objs_full = add_bbox(objs_full, pts)
    print(f"[full] {len(objs_full)} instances")

    # 3) 同类实例对距离：本文可分(>7cm) vs 竞品可分(>30cm)
    pairs = same_class_pairs(objs_full, prompts)
    dists = np.array([d for _, d in pairs]) if pairs else np.array([])
    n_ours_only = int(np.sum((dists > args.thr_ours) & (dists <= args.thr_dinya)))
    n_both = int(np.sum(dists > args.thr_dinya))
    n_neither = int(np.sum(dists <= args.thr_ours))
    print(f"[discrimination] 同类实例对 n={len(pairs)}")
    print(f"  Δ<=7cm(都不分): {n_neither}")
    print(f"  7cm<Δ<=30cm(本文分,竞品混): {n_ours_only}")
    print(f"  Δ>30cm(都分): {n_both}")

    # 4) 复现 Dinya re-ID 的一个具体例子：最接近的一对同类前景实例
    #    （bbox IoU + Chamfer，阈值 30cm），看它落在哪个区间
    demo = None
    if pairs:
        c, dmin = sorted(pairs, key=lambda x: x[1])[0]
        L = [o for o in objs_full
             if o["class"] == c and prompts[o["class"]] not in BACKGROUND_TUM
             and o["count"] >= 100]
        L = sorted(L, key=lambda o: np.linalg.norm(o["centroid"]))
        best = None
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                dd = float(np.linalg.norm(L[i]["centroid"] - L[j]["centroid"]))
                if best is None or dd < best[0]:
                    best = (dd, L[i], L[j])
        if best is not None:
            dd, o1, o2 = best
            iou = bbox_iou(o1, o2)
            cham = chamfer_median(o1["points"], o2["points"])
            name = prompts[c] if c < len(prompts) else str(c)
            demo = {"class": name, "dist": dd, "bbox_iou": iou,
                    "chamfer_median": cham,
                    "dinya_same": bool(cham < args.thr_dinya),
                    "ours_same": bool(dd < args.thr_ours)}
            print(f"[demo] 最近同类前景对: {name} Δ={dd:.3f}m bboxIoU={iou:.3f} "
                  f"Chamfer={cham:.3f}m -> Dinya判定={'同' if demo['dinya_same'] else '异'}, "
                  f"本文判定={'同' if demo['ours_same'] else '异'}")

    out = {
        "ours_anchoring_median": ours_median,
        "dinya_threshold": args.thr_dinya,
        "precision_ratio": float(args.thr_dinya / ours_median) if ours_median > 0 else None,
        "n_instances": len(objs_full),
        "n_same_class_pairs": len(pairs),
        "n_neither_separable": n_neither,
        "n_ours_only_separable": n_ours_only,
        "n_both_separable": n_both,
        "same_class_pair_dists": [d for _, d in pairs],
        "demo": demo,
        "prompts": prompts,
    }
    with open(os.path.join(args.out, "dinya_reid.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] -> {os.path.join(args.out, 'dinya_reid.json')}")
    print("SUMMARY:", json.dumps({k: out[k] for k in [
        "ours_anchoring_median", "dinya_threshold", "precision_ratio",
        "n_same_class_pairs", "n_ours_only_separable"]}))


if __name__ == "__main__":
    main()
