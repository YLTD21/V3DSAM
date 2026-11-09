#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 oneshot 2D 实例 + 跨帧 3D 重叠关联，重做 3×5 实例矩阵（替代「类别连通域」）。

旧方法（gen_instance_grid_scannet.py + instance_cloud.ply）的实例来源是
「语义类别 + 3D 连通域」聚类：同一类别的相邻物被 26 连通粘成一个、单物体被
遮挡/稀疏点切成多个（bed×2、door×4、chair×5）。本脚本从底层重做实例：

  1. 对选中 5 帧（3 局部 + 2 回环）跑 SAM3 oneshot（一次 39 类 → argmax →
     类别并集连通域）→ 每帧干净 2D 实例 mask。
  2. 反投影：每帧每个 2D 实例 mask → pts_gs 里落在 mask 内的可见 3D 点。
  3. 跨帧重叠关联：两个 2D 实例（不同帧）共享足够多 3D 点 → 同一物理物体；
     union-find 连通分量 = 3D 实例（跨帧统一 id）。语义切边界，几何定身份。
  4. 画 3×5：同 id 同色同类别。

Run on AutoDL (sgs env):
  python gen_instance_grid_oneshot.py
"""
import os, sys, argparse, json
from collections import defaultdict, Counter
import numpy as np, torch, cv2
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/v3dsam")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/vggt")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/sam3")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/perception_models")
os.environ['XFORMERS_DISABLED'] = '1'

from sam3 import build_sam3_image_model

# 复用 oneshot 2D 实例（39 类合并版，min_score 由调用方传）
from dump_sam3_masks_oneshot import ALL_PROMPTS, N_cats, build_transform, oneshot_masks
# 复用选帧 + 画图（gen_instance_grid_scannet）
from gen_instance_grid_scannet import (
    TAB20, hex2rgb, draw_instance, draw_label, mask_label_pos,
    load_objs, load_instance_cloud, visible_mask, pick_window,
    SCENES, BASE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from sci_style import finalize_figure
except Exception:
    def finalize_figure(fig, out):
        fig.savefig(out + ".png", dpi=200, bbox_inches="tight", facecolor="white")
        print(f"-> {out}.png")

BACKGROUND = {"wall", "floor", "ceiling"}


def backproject(pts, stage3, frames, instances_per_frame, min_pts=10):
    """每个 2D 实例反投影得到 pts 里的 3D 点索引。

    frames: list of fi。instances_per_frame: fi -> [(cat, mask(bool HxW), score)]。
    返回 entries = [{fi, li, cat, score, pidx}]，pidx = 落在 mask 内的可见 3D 点索引。
    """
    pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
    entries = []
    for fi in frames:
        if fi not in instances_per_frame:
            continue
        w2c = stage3[f"w2c_{fi}"].astype(np.float64)
        K = stage3[f"K_{fi}"].astype(np.float64)
        cam = (w2c @ pts_h.T).T
        z = cam[:, 2]
        front = z > 0.05
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        u = fx * cam[:, 0] / np.maximum(z, 1e-6) + cx
        v = fy * cam[:, 1] / np.maximum(z, 1e-6) + cy
        insts = instances_per_frame[fi]
        H, W = insts[0][1].shape
        ui = np.floor(u).astype(np.int64)
        vi = np.floor(v).astype(np.int64)
        inb = front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        inb_idx = np.nonzero(inb)[0]
        ui_in = ui[inb_idx]; vi_in = vi[inb_idx]; zi_in = z[inb_idx]
        # z-buffer：每像素只保留最近点，避免前景 mask 反投影混入后方背景点
        flat = vi_in.astype(np.int64) * W + ui_in
        order = np.lexsort((zi_in, flat))
        sflat = flat[order]
        first = np.concatenate(([True], sflat[1:] != sflat[:-1]))
        surf_pidx = inb_idx[order][first]
        surf_ui = ui_in[order][first]; surf_vi = vi_in[order][first]
        for li, (cat, mask, score) in enumerate(insts):
            sel = mask[surf_vi, surf_ui]
            pidx = surf_pidx[sel]
            if len(pidx) < min_pts:
                continue
            entries.append({"fi": fi, "li": li, "cat": int(cat),
                            "score": float(score), "pidx": pidx})
    return entries


def associate(entries, pts, dist_thresh=0.3):
    """跨帧关联：同类别 + 反投影 3D 质心距离 < dist_thresh 的两个 2D 实例连边，
    union-find 连通分量 = 3D 实例。用质心而非可见点集 IoU——同一物体从正/反面
    看质心一致，可见面却可能不相交，故质心对视角（尤其回环反向视角）更鲁棒。
    返回 [{cat, members:[entry idx], per_frame:{fi:(li,cat,score,pidx)}}]。"""
    n = len(entries)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # 每个 2D 实例反投影点集的 3D 质心（世界帧，单位米）
    cents = np.empty((n, 3), dtype=np.float64)
    for i, e in enumerate(entries):
        cents[i] = pts[e["pidx"]].mean(axis=0)

    # 同类别 + 质心距离阈值 → union
    for a in range(n):
        for b in range(a + 1, n):
            if entries[a]["cat"] != entries[b]["cat"]:
                continue
            if np.linalg.norm(cents[a] - cents[b]) < dist_thresh:
                union(a, b)

    comp = {}
    for i in range(n):
        comp.setdefault(find(i), []).append(i)

    instances_3d = []
    for members in comp.values():
        cats = [entries[m]["cat"] for m in members]
        cat = Counter(cats).most_common(1)[0][0]
        per_frame = {}
        for m in members:
            e = entries[m]
            per_frame[e["fi"]] = (e["li"], e["cat"], e["score"], e["pidx"])
        instances_3d.append({"cat": cat, "per_frame": per_frame})
    return instances_3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_frames", type=int, default=5)
    ap.add_argument("--min_score", type=float, default=0.3)
    ap.add_argument("--dist_thresh", type=float, default=0.3)
    ap.add_argument("--max_pts", type=int, default=1000000)
    ap.add_argument("--out", default="/root/autodl-tmp/scannet_figs_all/fig_instance_grid_oneshot")
    args = ap.parse_args()

    print("[oneshot-grid] loading SAM3 ...")
    model = build_sam3_image_model(
        bpe_path="/root/autodl-tmp/v3dsam/third_party/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        checkpoint_path="/root/autodl-tmp/v3dsam/third_party/sam3/sam3.pt",
        device="cuda")
    transform = build_transform()
    device = torch.device("cuda")
    print("[oneshot-grid] SAM3 loaded")

    scenes = list(SCENES.keys())
    n = args.n_frames
    fig, axes = plt.subplots(len(scenes), n, figsize=(2.35 * n, 3.1 * len(scenes)))
    if len(scenes) == 1:
        axes = axes[None, :]
    for r in range(axes.shape[0]):
        for c in range(axes.shape[1]):
            axes[r, c].axis("off")

    for r, scene in enumerate(scenes):
        name, s3_rel, sem_rel, inst_rel = SCENES[scene]
        base = f"{BASE}/{name}"
        stage3 = np.load(f"{base}/{s3_rel}", allow_pickle=True)
        sem = np.load(f"{base}/{sem_rel}", allow_pickle=True)
        kf_indices = [int(x) for x in sem["kf_indices"].tolist()]
        img_W, img_H = int(stage3["img_W"]), int(stage3["img_H"])

        # 1) 选帧：复用旧 pick_window（只借它的「3 局部 + 2 回环」选择，实例来源换成 oneshot）
        objs = load_objs(base, inst_rel)
        pts_all, ids_all = load_instance_cloud(base)
        id2pts = {o["id"]: pts_all[ids_all == o["id"]] for o in objs}
        K = len(kf_indices)
        vis = np.zeros((K, len(objs)), dtype=bool)
        blur = np.zeros(K, dtype=np.float64)
        for k, fi in enumerate(kf_indices):
            m, _, _ = visible_mask(objs, stage3[f"w2c_{fi}"].astype(np.float64),
                                   stage3[f"K_{fi}"].astype(np.float64), img_W, img_H)
            vis[k] = m
            g = cv2.imread(str(stage3[f"img_path_{fi}"]), cv2.IMREAD_GRAYSCALE)
            blur[k] = cv2.Laplacian(g, cv2.CV_64F).var()
        _, _, win_pos, _, _ = pick_window(vis, blur)
        win = [kf_indices[p] for p in win_pos]
        print(f"[oneshot-grid] {scene}: selected frames local={win[:3]} loop={win[3:]}")

        # 2) 对 5 帧跑 oneshot 2D 实例
        instances_per_frame = {}
        for fi in win:
            p = str(stage3[f"img_path_{fi}"])
            insts, _ = oneshot_masks(p, model, transform, device, args.min_score)
            instances_per_frame[fi] = insts
            print(f"[oneshot-grid] {scene} kf{fi}: {len(insts)} 2D instances")

        # 3) 反投影 + 跨帧重叠关联
        pts = stage3["pts_gs"].astype(np.float64)
        if len(pts) > args.max_pts:
            idx = np.random.RandomState(0).choice(len(pts), args.max_pts, replace=False)
            idx.sort()
            pts = pts[idx]
        entries = backproject(pts, stage3, win, instances_per_frame)
        instances_3d = associate(entries, pts, dist_thresh=args.dist_thresh)
        print(f"[oneshot-grid] {scene}: {len(entries)} 2D entries -> {len(instances_3d)} 3D instances")

        # 4) 筛选持久实例（≥2 局部帧 + ≥1 回环帧都有 mask）
        local_set = set(win[:3]); loop_set = set(win[3:])
        keep3d = []
        for i, it in enumerate(instances_3d):
            if ALL_PROMPTS[it["cat"]] in BACKGROUND:
                continue
            pf = it["per_frame"]
            n_local = sum(1 for fi in pf if fi in local_set)
            n_loop = sum(1 for fi in pf if fi in loop_set)
            if n_local >= 2 and n_loop >= 1:
                it["id"] = len(keep3d)
                it["n_local"] = n_local; it["n_loop"] = n_loop
                keep3d.append(it)
        keep3d.sort(key=lambda x: -(x["n_local"] + x["n_loop"]))
        print(f"[oneshot-grid] {scene}: {len(keep3d)} persistent instances = "
              f"{[(it['id'], ALL_PROMPTS[it['cat']]) for it in keep3d]}")
        # 诊断：所有 3D 实例的帧覆盖（理解为何持久实例少）
        for it in instances_3d:
            frames = sorted(it["per_frame"].keys())
            nl = sum(1 for f in frames if f in local_set)
            nl_ = sum(1 for f in frames if f in loop_set)
            print(f"    [3D] {ALL_PROMPTS[it['cat']]:<12} frames={frames} local={nl} loop={nl_}")

        axes[r, 0].set_ylabel(f"{scene}\n{len(keep3d)} persistent",
                              fontsize=11, rotation=0, labelpad=24, va="center",
                              color="#272727", fontweight="bold")

        # 5) 画 3×5：每个 3D 实例用它在该帧的 oneshot 2D mask，同 id 同色
        for c, fi in enumerate(win):
            ax = axes[r, c]
            img = cv2.imread(str(stage3[f"img_path_{fi}"]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]
            for it in keep3d:
                if fi not in it["per_frame"]:
                    continue
                li, cat, score, pidx = it["per_frame"][fi]
                imask = instances_per_frame[fi][li][1]
                colrgb = hex2rgb(TAB20[it["id"] % len(TAB20)])
                img = draw_instance(img, imask, colrgb)
            for it in keep3d:
                if fi not in it["per_frame"]:
                    continue
                li, cat, score, pidx = it["per_frame"][fi]
                imask = instances_per_frame[fi][li][1]
                colrgb = hex2rgb(TAB20[it["id"] % len(TAB20)])
                lu, lv = mask_label_pos(imask, W, H, 0, 0)
                draw_label(img, f"#{it['id']} {ALL_PROMPTS[cat]}", lu, lv, colrgb, W, H)
            ax.imshow(img)
            if r == 0:
                ax.set_title(f"kf {fi}" if c < 3 else f"loop {fi}", fontsize=11, color="#272727")

    fig.suptitle(
        "Persistent instance identity on ScanNet — oneshot 2D instances + cross-frame 3D association\n"
        "(each instance = SAM3 oneshot 2D mask, linked across frames by 3D point overlap; "
        "same object keeps the same colour + #id + category across the loop)",
        fontsize=14, y=1.02)
    finalize_figure(fig, args.out)
    print(f"[oneshot-grid] -> {args.out}.png")


if __name__ == "__main__":
    main()
