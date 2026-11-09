#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 scene0207 的 loop_log.json（submap sid+fi）映射到 keyframe 全局索引。"""
import numpy as np, json

BASE = "/root/autodl-tmp/ablation/scannet/scene0207_00"
d = np.load(f"{BASE}/_stage3_input_after.npz", allow_pickle=True)
sem = np.load(f"{BASE}/kf_semantic.npz", allow_pickle=True)
kf_indices = [int(x) for x in sem["kf_indices"].tolist()]

# 全局帧数 = 最大 img_path_{fi} 的 fi + 1
N = 0
while f"img_path_{N}" in d.files:
    N += 1
print(f"total frames (img_path_*) = {N}")
print(f"kf_indices len = {len(kf_indices)}, range [{min(kf_indices)},{max(kf_indices)}]")

# 检查 npz 里有没有 submap 相关的 key
submap_keys = [k for k in d.files if "submap" in k.lower() or "seg" in k.lower() or "sid" in k.lower()]
print(f"submap-like npz keys: {submap_keys[:20]}")

log = json.load(open(f"{BASE}/loop_log.json"))
print(f"\n{len(log['loops'])} loops:")
for i, lp in enumerate(log["loops"]):
    qs, qf = lp["query_sid"], lp["query_fi"]
    ds, df = lp["detected_sid"], lp["detected_fi"]
    # 假设 submap_size=32：全局帧 = sid*32 + fi
    for tag, (sid, fi) in (("Q", (qs, qf)), ("D", (ds, df))):
        g = sid * 32 + fi
        # 找到最近的 keyframe
        kf = min(kf_indices, key=lambda x: abs(x - g))
        print(f"  loop{i} {tag}: sid={sid} fi={fi} -> guess global={g} (in range: {0 <= g < N}), "
              f"nearest kf={kf}")
