#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SE(3) 回环检测：位置近 + 朝向近 = 真回环。"""
import numpy as np

BASE = "/root/autodl-tmp/ablation/scannet/scene0207_00"
d = np.load(f"{BASE}/_stage3_input_after.npz", allow_pickle=True)
sem = np.load(f"{BASE}/kf_semantic.npz", allow_pickle=True)
kf_indices = [int(x) for x in sem["kf_indices"].tolist()]
K = len(kf_indices)

cents = np.zeros((K, 3), dtype=np.float64)
Rots = np.zeros((K, 3, 3), dtype=np.float64)
for k, fi in enumerate(kf_indices):
    w2c = d[f"w2c_{fi}"].astype(np.float64)
    R, t = w2c[:3, :3], w2c[:3, 3]
    cents[k] = -R.T @ t
    Rots[k] = R

def rot_angle(R):
    # 旋转角（弧度）
    c = (np.trace(R) - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    return np.arccos(c)

gap_k = 50
pos_thr = 0.5
ang_thr = np.deg2rad(30)  # 30 度内算同朝向
loops = []
for i in range(K):
    for j in range(i + gap_k, K):
        dp = np.linalg.norm(cents[i] - cents[j])
        if dp >= pos_thr:
            continue
        Rrel = Rots[j] @ Rots[i].T
        ang = rot_angle(Rrel)
        if ang < ang_thr:
            loops.append((dp, np.rad2deg(ang), i, j, kf_indices[i], kf_indices[j]))
loops.sort()
print(f"SE(3) genuine loops (pos<{pos_thr}m, ang<{np.rad2deg(ang_thr):.0f}deg, gap>={gap_k}): {len(loops)}")
for dp, ang, i, j, fi, fj in loops[:40]:
    print(f"  pos={dp:.3f}m ang={ang:.1f}deg  kf[{i}]={fi} <-> kf[{j}]={fj}  (gap {j-i})")

# 也打印 位置近但朝向差（假回环）的例子
print("\nfalse-loop examples (pos<0.5m but ang>90deg):")
cnt = 0
for i in range(K):
    for j in range(i + gap_k, K):
        dp = np.linalg.norm(cents[i] - cents[j])
        if dp >= 0.5:
            continue
        ang = np.rad2deg(rot_angle(Rots[j] @ Rots[i].T))
        if ang > 90:
            print(f"  pos={dp:.3f}m ang={ang:.1f}deg  kf[{i}]={kf_indices[i]} <-> kf[{j}]={kf_indices[j]}")
            cnt += 1
            if cnt >= 10:
                break
    if cnt >= 10:
        break
