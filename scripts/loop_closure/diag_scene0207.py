#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 scene0207：质心可见性 vs 2D 绑定，以及回环帧选择。"""
import numpy as np, cv2
from gen_instance_grid_3d2d import (
    load_semantic, voxel_downsample, cluster_category,
    frame_2d_regions, bind_instance)
from gen_instance_grid_scannet import FOREGROUND, BASE, visible_mask

BACKGROUND = {"wall", "floor", "ceiling"}


def build_instances(base, prompts, vpts, vlab, fg, voxel=0.02, eps=0.05, min_voxels=100, max_inst=25):
    inst = []
    for c in fg:
        m = np.nonzero(vlab == c)[0]
        if m.size < min_voxels:
            continue
        for cl in cluster_category(vpts[m], eps, min_voxels):
            idx = m[cl]
            ip = vpts[idx]
            inst.append({"id": len(inst), "class": c, "name": prompts[c],
                         "centroid": ip.mean(0), "points": ip, "n_voxels": int(len(idx))})
    inst.sort(key=lambda o: -o["n_voxels"])
    return inst[:max_inst]


def main():
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
    inst = build_instances(base, prompts, vpts, vlab, fg)
    print(f"n inst = {len(inst)}")
    for x in inst:
        print(f"  #{x['id']:2d} {x['name']:12s} n_vox={x['n_voxels']:6d}")

    K = len(kf_indices)
    # 全帧质心可见性
    vis = np.zeros((K, len(inst)), dtype=bool)
    for k, fi in enumerate(kf_indices):
        m, _, _ = visible_mask(inst, stage3[f"w2c_{fi}"].astype(np.float64),
                               stage3[f"K_{fi}"].astype(np.float64), img_W, img_H)
        vis[k] = m

    win = [1398, 1403, 1408, 615, 560]
    win_pos = [kf_indices.index(f) for f in win]
    print("\nper-frame: centroid-visible (V) vs bound (B)")
    for c, (p, fi) in enumerate(zip(win_pos, win)):
        lf = labels_all[p].astype(np.int32)
        H, W = lf.shape
        regs = frame_2d_regions(lf, fg, 300, 5)
        w2c = stage3[f"w2c_{fi}"].astype(np.float64)
        Kcam = stage3[f"K_{fi}"].astype(np.float64)
        m, _, _ = visible_mask(inst, w2c, Kcam, img_W, img_H)
        row = []
        for j in range(len(inst)):
            vis_j = bool(m[j])
            b_j = bind_instance(inst[j], regs, w2c, Kcam, W, H) is not None
            if vis_j or b_j:
                row.append(f"#{j}{inst[j]['name'][:4]}:{'V' if vis_j else '-'}{'B' if b_j else '-'}")
        print(f"  kf{fi}: {len(regs)}regions | {' '.join(row)}")

    # 全帧扫描：找能同时满足 ≥2 local 绑定 + ≥1 loop 绑定的 local window 起点
    print("\nscan all local windows for binding-based persistence ...")
    best = None
    for s in range(K - 2):
        local_pos = [s, s + 1, s + 2]
        # 快速：先算这 3 帧里每实例绑定情况
        bind3 = {}
        regs3 = {}
        for c, p in enumerate(local_pos):
            fi = kf_indices[p]
            lf = labels_all[p].astype(np.int32)
            H, W = lf.shape
            regs3[c] = frame_2d_regions(lf, fg, 300, 5)
            w2c = stage3[f"w2c_{fi}"].astype(np.float64)
            Kcam = stage3[f"K_{fi}"].astype(np.float64)
            for j in range(len(inst)):
                bind3[(c, j)] = bind_instance(inst[j], regs3[c], w2c, Kcam, W, H) is not None
        local_obj = [j for j in range(len(inst)) if sum(bind3[(c, j)] for c in range(3)) >= 2]
        if not local_obj:
            continue
        # loop 候选：离 local 至少 20% 帧
        gap = max(10, int(0.20 * K))
        cand = [k for k in range(K) if k < s - gap or k > s + 2 + gap]
        scored = []
        for k in cand:
            fi = kf_indices[k]
            lf = labels_all[k].astype(np.int32)
            H, W = lf.shape
            regs = frame_2d_regions(lf, fg, 300, 5)
            w2c = stage3[f"w2c_{fi}"].astype(np.float64)
            Kcam = stage3[f"K_{fi}"].astype(np.float64)
            nb = sum(1 for j in local_obj if bind_instance(inst[j], regs, w2c, Kcam, W, H) is not None)
            scored.append((nb, k, fi))
        scored.sort(key=lambda x: -x[0])
        if scored[0][0] + (scored[1][0] if len(scored) > 1 else 0) < 1:
            continue
        tot = scored[0][0] + (scored[1][0] if len(scored) > 1 else 0)
        n_persist = sum(1 for j in local_obj if any(
            bind_instance(inst[j], frame_2d_regions(labels_all[kf_indices.index(scored[t][2])].astype(np.int32), fg, 300, 5),
                          stage3[f"w2c_{scored[t][2]}"].astype(np.float64), stage3[f"K_{scored[t][2]}"].astype(np.float64),
                          labels_all[kf_indices.index(scored[t][2])].shape[1], labels_all[kf_indices.index(scored[t][2])].shape[0]) is not None
            for t in range(min(2, len(scored)))))
        if best is None or n_persist > best[0]:
            best = (n_persist, s, scored[:2], local_obj)
    if best:
        n_persist, s, loops, local_obj = best
        print(f"best local window s={s} (kf {[kf_indices[s+i] for i in range(3)]}), "
              f"loops={[(k, fi, nb) for nb, k, fi in loops]}, n_persist={n_persist}, "
              f"local_obj={[(j, inst[j]['name']) for j in local_obj]}")
    else:
        print("NO binding-based persistent window found")


if __name__ == "__main__":
    main()
