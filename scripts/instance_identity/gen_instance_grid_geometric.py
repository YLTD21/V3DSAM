#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3×5 持久实例矩阵 —— 纯 3D 几何聚类（替代旧的「类别连通域」实例）。

旧实例（gen_semantic_instances.py 的 cluster_instances）用 4cm 体素 + 26 连通域，
把紧贴同类物体粘成一块、把稀疏/被遮挡单物体切成多块。本脚本从底层重做实例：

  1. 体素降采样（voxel=0.02m，均值坐标 + 多数类别）→ 规整密度。
  2. 按前景类别欧氏聚类（scipy cKDTree.query_pairs + 并查集）→ 每个连通分量 = 一个实例。
     语义（SAM3 类别）只切边界，几何（世界坐标连通性）定身份。
  3. 质心可见性 → pick_window 选 3 局部 + 2 回环帧。
  4. 持久实例（≥2 局部帧 + ≥1 回环帧可见）逐帧投影点集成掩膜，同 id 同色同类别渲染 3×5。

数据源：semantic_cloud_10m/semantic_cloud.npz（pts+label+prompts，全密度）
        + _stage3_input_after.npz（w2c/K/img_path，与 pts 同一 SLAM 世界帧）
        + kf_semantic.npz（kf_indices 选帧）

Run on AutoDL (sgs env):
  cd /root/autodl-tmp
  /root/miniconda3/envs/sgs/bin/python -u gen_instance_grid_geometric.py \
    --out /root/autodl-tmp/scannet_figs_all/fig_instance_grid_geometric
"""
import os, argparse
from collections import Counter
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

try:
    from sci_style import apply_publication_style, finalize_figure
    apply_publication_style(font_size=8, axes_linewidth=1)
except Exception:
    def finalize_figure(fig, out):
        fig.savefig(out + ".png", dpi=200, bbox_inches="tight", facecolor="white")
        print(f"-> {out}.png")

# 复用 gen_instance_grid_scannet.py 的渲染/选帧 helper（不改它）
from gen_instance_grid_scannet import (
    visible_mask, instance_mask_points, pick_window,
    draw_instance, draw_label, mask_label_pos, hex2rgb, TAB20, FOREGROUND,
    SCENES, BASE)

BACKGROUND = {"wall", "floor", "ceiling"}


def load_semantic(base):
    d = np.load(f"{base}/semantic_cloud_10m/semantic_cloud.npz", allow_pickle=True)
    pts = d["pts"].astype(np.float64)
    label = d["label"].astype(np.int32)
    prompts = [str(p) for p in d["prompts"]]
    return pts, label, prompts


def voxel_downsample(pts, label, voxel=0.02):
    """体素降采样：取体素内均值坐标 + 多数类别（-1 未标注在投票外）。"""
    origin = pts.min(axis=0)
    q = np.floor((pts - origin) / voxel).astype(np.int64)
    keys, inverse = np.unique(q, axis=0, return_inverse=True)
    n_vox = len(keys)
    cnt = np.bincount(inverse, minlength=n_vox).astype(np.float64)
    vpts = np.empty((n_vox, 3), np.float64)
    for d in range(3):
        vpts[:, d] = np.bincount(inverse, weights=pts[:, d], minlength=n_vox) / cnt

    lab_shift = label + 1                       # -1 -> 0；类别 0..49 -> 1..50
    n_lab = int(lab_shift.max()) + 1
    flat = inverse * n_lab + lab_shift
    cnt_lab = np.bincount(flat, minlength=n_vox * n_lab).reshape(n_vox, n_lab)
    non_unlab = cnt_lab[:, 1:]                  # 排除「未标注」桶
    vlab = non_unlab.argmax(1).astype(np.int32)
    vlab[non_unlab.max(1) == 0] = -1
    return vpts, vlab


def cluster_category(sub_pts, eps, min_voxels):
    """对单类别体素点做欧氏聚类（cKDTree.query_pairs + 并查集），返回连通分量列表。"""
    n = len(sub_pts)
    if n < min_voxels:
        return []
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree = cKDTree(sub_pts)
    pairs = tree.query_pairs(r=eps, output_type="ndarray")
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return [members for members in comps.values() if len(members) >= min_voxels]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--min_voxels", type=int, default=100)
    ap.add_argument("--max_inst", type=int, default=25)
    ap.add_argument("--out", default="/root/autodl-tmp/scannet_figs_all/fig_instance_grid_geometric")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    scenes = list(SCENES.keys())
    n_frames = 5
    fig, axes = plt.subplots(len(scenes), n_frames, figsize=(2.35 * n_frames, 3.1 * len(scenes)))
    if len(scenes) == 1:
        axes = axes[None, :]
    for r in range(axes.shape[0]):
        for c in range(axes.shape[1]):
            axes[r, c].axis("off")

    for r, scene in enumerate(scenes):
        name, s3_rel, sem_rel, _ = SCENES[scene]
        base = f"{BASE}/{name}"
        stage3 = np.load(f"{base}/{s3_rel}", allow_pickle=True)
        sem = np.load(f"{base}/{sem_rel}", allow_pickle=True)
        kf_indices = [int(x) for x in sem["kf_indices"].tolist()]
        img_W, img_H = int(stage3["img_W"]), int(stage3["img_H"])

        pts, label, prompts = load_semantic(base)
        print(f"\n[geo] {scene}: {len(pts):,} points, {len(prompts)} categories")

        vpts, vlab = voxel_downsample(pts, label, args.voxel)
        print(f"[geo] {scene}: voxel {args.voxel:.2f}m -> {len(vpts):,} voxel points")

        fg_cats = [c for c in range(len(prompts))
                   if prompts[c] in FOREGROUND and prompts[c] not in BACKGROUND]

        # 按类别欧氏聚类 → 实例
        instances = []
        for c in fg_cats:
            members = np.nonzero(vlab == c)[0]
            if members.size < args.min_voxels:
                continue
            clusters = cluster_category(vpts[members], args.eps, args.min_voxels)
            for cl in clusters:
                idx = members[cl]
                inst_pts = vpts[idx]
                instances.append({
                    "id": len(instances),
                    "class": c,
                    "name": prompts[c],
                    "centroid": inst_pts.mean(0),
                    "points": inst_pts,
                    "n_voxels": int(len(idx)),
                })
        instances.sort(key=lambda o: -o["n_voxels"])
        instances = instances[:args.max_inst]
        print(f"[geo] {scene}: {len(instances)} instances (cap {args.max_inst})")
        print("   by category:", dict(Counter(o["name"] for o in instances)))

        # 诊断：同类多实例的质心两两最小距离（<0.3m 大概率是同一物体被切碎）
        from itertools import combinations
        by_name = {}
        for o in instances:
            by_name.setdefault(o["name"], []).append(o)
        for name_, grp in by_name.items():
            if len(grp) < 2:
                continue
            cents = np.array([g["centroid"] for g in grp])
            ds = sorted(np.linalg.norm(a - b) for a, b in combinations(cents, 2))
            print(f"   [diag] {name_:<10} x{len(grp)} vox={[g['n_voxels'] for g in grp]} "
                  f"min_pair_dist={ds[0]:.2f}m ({', '.join(f'{d:.2f}' for d in ds[:5])})")

        # 质心可见性矩阵 + 清晰度
        K = len(kf_indices)
        vis = np.zeros((K, len(instances)), dtype=bool)
        blur = np.zeros(K, dtype=np.float64)
        for k, fi in enumerate(kf_indices):
            m, _, _ = visible_mask(instances, stage3[f"w2c_{fi}"].astype(np.float64),
                                   stage3[f"K_{fi}"].astype(np.float64), img_W, img_H)
            vis[k] = m
            g = cv2.imread(str(stage3[f"img_path_{fi}"]), cv2.IMREAD_GRAYSCALE)
            blur[k] = cv2.Laplacian(g, cv2.CV_64F).var()

        _, _, win_pos, _, _ = pick_window(vis, blur)
        win = [kf_indices[p] for p in win_pos]
        print(f"[geo] {scene}: local kf={win[:3]}, loop kf={win[3:]}")

        # 逐帧逐实例把 3D 点集投影成 2D 掩膜
        masks = {}
        for c, fi in enumerate(win):
            w2c = stage3[f"w2c_{fi}"].astype(np.float64)
            Kcam = stage3[f"K_{fi}"].astype(np.float64)
            for j, o in enumerate(instances):
                masks[(c, j)] = instance_mask_points(o["points"], w2c, Kcam, img_W, img_H)[0]

        # 持久实例：≥2 局部帧 + ≥1 回环帧可见
        sel = [j for j in range(len(instances))
               if sum(1 for c in range(3) if masks[(c, j)] is not None) >= 2
               and sum(1 for c in range(3, 5) if masks[(c, j)] is not None) >= 1]
        print(f"[geo] {scene}: {len(sel)} persistent = "
              f"{[(instances[j]['id'], instances[j]['name']) for j in sel]}")

        axes[r, 0].set_ylabel(f"{scene}\n{len(sel)} persistent instances",
                              fontsize=11, rotation=0, labelpad=24,
                              va="center", color="#272727", fontweight="bold")

        for c, fi in enumerate(win):
            ax = axes[r, c]
            img = cv2.imread(str(stage3[f"img_path_{fi}"]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]
            for j in sel:
                imask = masks[(c, j)]
                if imask is None:
                    continue
                colrgb = hex2rgb(TAB20[instances[j]["id"] % len(TAB20)])
                img = draw_instance(img, imask, colrgb)
            for j in sel:
                imask = masks[(c, j)]
                if imask is None:
                    continue
                colrgb = hex2rgb(TAB20[instances[j]["id"] % len(TAB20)])
                lu, lv = mask_label_pos(imask, W, H, 0, 0)
                draw_label(img, f"#{instances[j]['id']} {instances[j]['name']}",
                           lu, lv, colrgb, W, H)
            ax.imshow(img)
            if r == 0:
                ax.set_title(f"kf {fi}" if c < 3 else f"loop {fi}", fontsize=11, color="#272727")

    fig.suptitle(
        "Persistent instance identity on ScanNet — pure 3D geometric clustering (per-category Euclidean)\n"
        "(each instance = connected 3D component in the semantic point cloud; same object keeps "
        "the same colour + #id + category across the loop)",
        fontsize=14, y=1.02)
    finalize_figure(fig, args.out)
    print(f"[geo] -> {args.out}.png")


if __name__ == "__main__":
    main()
