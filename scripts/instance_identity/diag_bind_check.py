#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 scene0207 真实回环帧（近位置）能否绑定同一批实例。"""
import numpy as np, cv2
from gen_instance_grid_3d2d import (
    load_semantic, voxel_downsample, cluster_category,
    frame_2d_regions, bind_instance)
from gen_instance_grid_scannet import FOREGROUND, BASE

BACKGROUND = {"wall", "floor", "ceiling"}
name = "scene0207_00"
base = f"{BASE}/{name}"
stage3 = np.load(f"{base}/_stage3_input_after.npz", allow_pickle=True)
sem = np.load(f"{base}/kf_semantic.npz", allow_pickle=True)
kf_indices = [int(x) for x in sem["kf_indices"].tolist()]
img_W, img_H = int(stage3["img_W"]), int(stage3["img_H"])
labels_all = sem["labels"]
pts, label, prompts = load_semantic(base)
vpts, vlab = voxel_downsample(pts, label, 0.02)
fg = [c for c in range(len(prompts)) if prompts[c] in FOREGROUND and prompts[c] not in BACKGROUND]

inst = []
for c in fg:
    m = np.nonzero(vlab == c)[0]
    if m.size < 100:
        continue
    for cl in cluster_category(vpts[m], 0.05, 100):
        idx = m[cl]
        ip = vpts[idx]
        inst.append({"id": len(inst), "class": c, "name": prompts[c],
                     "centroid": ip.mean(0), "points": ip, "n_voxels": int(len(idx))})
inst.sort(key=lambda o: -o["n_voxels"])
inst = inst[:25]

def check(frames, tag):
    print(f"\n=== {tag}: frames={frames} ===")
    bound = {}
    for c, fi in enumerate(frames):
        p = kf_indices.index(fi)
        lf = labels_all[p].astype(np.int32)
        H, W = lf.shape
        regs = frame_2d_regions(lf, fg, 300, 5)
        w2c = stage3[f"w2c_{fi}"].astype(np.float64)
        Kcam = stage3[f"K_{fi}"].astype(np.float64)
        bound[c] = [j for j in range(len(inst))
                    if bind_instance(inst[j], regs, w2c, Kcam, W, H) is not None]
        print(f"  kf{fi}: {len(regs)}regions bound={[(j, inst[j]['name']) for j in bound[c]]}")
    # persistence: >=2 local + >=1 loop
    sel = [j for j in range(len(inst))
           if sum(1 for c in range(3) if j in bound[c]) >= 2
           and sum(1 for c in range(3, 5) if j in bound[c]) >= 1]
    print(f"  persistent={[(inst[j]['name']) for j in sel]}")


# 候选: 真 SE(3) 回环（位置+朝向都近）
check([146, 150, 154, 1714, 1718], "loopA (146/150/154 <-> 1714/1718) ang~3deg")
check([188, 191, 194, 1733, 1736], "loopB (188/191/194 <-> 1733/1736) ang~10deg")
check([119, 123, 128, 1919, 1924], "loopC (119/123/128 <-> 1919/1924) ang~20deg")
