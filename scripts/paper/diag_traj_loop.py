#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用相机轨迹找 scene0207 真实回环：远时间距 + 近空间距 = 回环。"""
import numpy as np

BASE = "/root/autodl-tmp/ablation/scannet/scene0207_00"
d = np.load(f"{BASE}/_stage3_input_after.npz", allow_pickle=True)
sem = np.load(f"{BASE}/kf_semantic.npz", allow_pickle=True)
kf_indices = [int(x) for x in sem["kf_indices"].tolist()]
K = len(kf_indices)

# 相机中心 = -R^T @ t  (w2c = [R|t])
cents = np.zeros((K, 3), dtype=np.float64)
for k, fi in enumerate(kf_indices):
    w2c = d[f"w2c_{fi}"].astype(np.float64)
    R, t = w2c[:3, :3], w2c[:3, 3]
    cents[k] = -R.T @ t

print(f"K={K} keyframes, frame range [{min(kf_indices)},{max(kf_indices)}]")

# 找回环：时间距 >= 50 keyframes，空间距 < 0.5m
gap_k = 50
thr = 0.5
loops = []
for i in range(K):
    for j in range(i + gap_k, K):
        dist = np.linalg.norm(cents[i] - cents[j])
        if dist < thr:
            loops.append((dist, i, j, kf_indices[i], kf_indices[j]))
loops.sort()
print(f"\ntop loop candidates (dist, i, j, kf_i, kf_j) dist<{thr}m, gap>={gap_k}:")
for dist, i, j, fi, fj in loops[:40]:
    print(f"  d={dist:.3f}m  kf[{i}]={fi} <-> kf[{j}]={fj}  (gap {j-i})")

# 也检查更宽的阈值
thr2 = 1.0
loops2 = [(np.linalg.norm(cents[i]-cents[j]), i, j, kf_indices[i], kf_indices[j])
          for i in range(K) for j in range(i+gap_k, K)
          if np.linalg.norm(cents[i]-cents[j]) < thr2]
loops2.sort()
print(f"\nwider thr<{thr2}m: {len(loops2)} candidates, top:")
for dist, i, j, fi, fj in loops2[:20]:
    print(f"  d={dist:.3f}m  kf[{i}]={fi} <-> kf[{j}]={fj}  (gap {j-i})")
