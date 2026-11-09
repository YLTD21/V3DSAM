#!/usr/bin/env python3
"""Honest cross-view re-identification benchmark (路B 前置：修复版两半匹配).

gen_semantic_persistence.py 的两个缺陷在这里修复：
  (1) 匹配不按类别过滤 → wall 会被匹配到 window。现在**只在同类内做最近质心匹配**。
  (2) 结尾写 json 时 NameError(front/front_pos) 崩溃 → 这里用正确的变量名。

要回答的核心问题（贡献 2 的正确度量）：
  同一类别、不同实例（如两把椅子）能否被「世界坐标锚定」稳定区分？
  → 同类匹配的 Δpos（自一致性，应 ~cm 级）
  → 同类实例间的最近邻分离裕度（nearest vs second-nearest，越大越可分）
  → 同类别重识别准确率（正确匹配 / 应匹配总数）

CPU-only。复用 gen_semantic_pointcloud.load_stage3/subsample、
gen_semantic_instances.cluster_instances、gen_semantic_persistence.vote_labels_subset。
"""
import os, argparse, json
import numpy as np
from scipy.optimize import linear_sum_assignment
from gen_semantic_pointcloud import load_stage3, subsample
from gen_semantic_instances import cluster_instances
from gen_semantic_persistence import vote_labels_subset


def same_class_match(objs_f, objs_b, dist_thr=0.30):
    """同类内最近质心匹配（匈牙利，逐类别）。返回按类别聚合的结果。"""
    classes = sorted(set([o["class"] for o in objs_f] + [o["class"] for o in objs_b]))
    per_class = {}
    all_pairs = []
    for c in classes:
        A = [o for o in objs_f if o["class"] == c]
        B = [o for o in objs_b if o["class"] == c]
        if not A or not B:
            continue
        Ca = np.stack([o["centroid"] for o in A])
        Cb = np.stack([o["centroid"] for o in B])
        D = np.linalg.norm(Ca[:, None, :] - Cb[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(D)
        pairs = []
        for r, cc in zip(rows, cols):
            if D[r, cc] <= dist_thr:
                pairs.append((A[r], B[cc], float(D[r, cc])))
        pairs.sort(key=lambda p: -p[0]["count"])
        # 同类最近邻分离裕度：每个 front 实例到其最近 back 同类 vs 次近 back 同类的距离比
        margins = []
        for i in range(len(A)):
            d = np.sort(D[i])            # 升序：d[0]=最近，d[1]=次近
            if len(d) >= 2 and d[0] > 1e-9:
                margins.append(float(d[1] / d[0]))
        per_class[c] = {
            "front_n": len(A), "back_n": len(B), "matched": len(pairs),
            "missed": max(0, min(len(A), len(B)) - len(pairs)),
            "median_margin_ratio": float(np.median(margins)) if margins else None,
        }
        all_pairs.extend(pairs)
    return per_class, all_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage3", required=True)
    ap.add_argument("--semantic", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_pts", type=int, default=1000000)
    ap.add_argument("--voxel_size", type=float, default=0.04)
    ap.add_argument("--min_voxels", type=int, default=15)
    ap.add_argument("--dist_thr", type=float, default=0.30)
    ap.add_argument("--split", choices=["random", "frontback"], default="random",
                    help="random=随机交错两半(共视,正确设计); frontback=前/后两半(不重叠,旧设计)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    stage3, pts, rgb = load_stage3(args.stage3)
    pts, rgb, _ = subsample(pts, rgb, args.max_pts)
    s = np.load(args.semantic, allow_pickle=True)
    labels = s["labels"]; kf_indices = [int(x) for x in s["kf_indices"].tolist()]
    prompts = [str(p) for p in s["prompts"]]
    K = len(kf_indices)
    if args.split == "random":
        rng = np.random.RandomState(0)
        perm = rng.permutation(K)
        front_pos = sorted(perm[:K // 2].tolist())
        back_pos = sorted(perm[K // 2:].tolist())
    else:
        front_pos = list(range(0, K // 2))
        back_pos = list(range(K // 2, K))
    print(f"[reid] split={args.split} {K} keyframes -> front {len(front_pos)} / back {len(back_pos)}")

    sid_f = vote_labels_subset(pts, stage3, labels, kf_indices, front_pos, prompts)
    sid_b = vote_labels_subset(pts, stage3, labels, kf_indices, back_pos, prompts)
    objs_f, _ = cluster_instances(pts, sid_f, voxel_size=args.voxel_size, min_voxels=args.min_voxels)
    objs_b, _ = cluster_instances(pts, sid_b, voxel_size=args.voxel_size, min_voxels=args.min_voxels)
    print(f"[reid] instances: front {len(objs_f)} / back {len(objs_b)}")

    per_class, all_pairs = same_class_match(objs_f, objs_b, dist_thr=args.dist_thr)
    dists = [d for _, _, d in all_pairs]

    print(f"\n{'='*90}")
    print(f"  {'class':<12} {'front_n':>8} {'back_n':>8} {'matched':>8} {'missed':>7} {'median_margin(2nd/1st)':>24}")
    print(f"{'='*90}")
    for c in sorted(per_class):
        pc = per_class[c]
        nm = prompts[c] if c < len(prompts) else str(c)
        mm = f"{pc['median_margin_ratio']:.2f}" if pc['median_margin_ratio'] is not None else "n/a"
        print(f"  {nm:<12} {pc['front_n']:>8} {pc['back_n']:>8} {pc['matched']:>8} {pc['missed']:>7} {mm:>24}")
    print(f"{'='*90}")

    n_same_class = sum(1 for pc in per_class.values() if pc['front_n'] >= 2 or pc['back_n'] >= 2)
    n_matched = len(all_pairs)
    n_should = sum(min(pc['front_n'], pc['back_n']) for pc in per_class.values())
    acc = n_matched / n_should if n_should else 0.0
    print(f"  同类匹配: {n_matched}/{n_should} = re-ID 准确率 {acc:.3f}")
    print(f"  同类匹配 Δpos: mean={np.mean(dists):.4f}m median={np.median(dists):.4f}m "
          f"max={np.max(dists):.4f}m  (n={len(dists)})")
    print(f"  含 ≥2 实例的类别（真正考验区分能力的类别）: {n_same_class}")

    # 两把椅子的具体分离情况（若存在）
    chair_cls = None
    for i, p in enumerate(prompts):
        if p == "chair":
            chair_cls = i
    if chair_cls is not None:
        cf = [o for o in objs_f if o["class"] == chair_cls]
        cb = [o for o in objs_b if o["class"] == chair_cls]
        if len(cf) >= 2:
            print(f"\n  [chair 同类实例分离] front {len(cf)} 把椅子:")
            C = np.stack([o['centroid'] for o in cf])
            for i in range(len(cf)):
                d = np.linalg.norm(C - C[i], axis=1)
                nn = np.sort(d)[1] if len(d) > 1 else 0.0
                print(f"    chair#{i} centroid=({C[i][0]:.2f},{C[i][1]:.2f},{C[i][2]:.2f}) "
                      f"最近同类距离={nn:.3f}m")

    out = {
        "n_keyframes": K,
        "front_instances": len(objs_f), "back_instances": len(objs_b),
        "n_same_class_matched": n_matched, "n_should": n_should,
        "reid_accuracy": float(acc),
        "delta_mean": float(np.mean(dists)) if dists else None,
        "delta_median": float(np.median(dists)) if dists else None,
        "delta_max": float(np.max(dists)) if dists else None,
        "n_classes_with_ge2": n_same_class,
        "per_class": {str(k): v for k, v in per_class.items()},
        "prompts": prompts,
    }
    with open(os.path.join(args.out_dir, "reid_benchmark.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[reid] DONE -> {os.path.join(args.out_dir, 'reid_benchmark.json')}")


if __name__ == "__main__":
    main()
