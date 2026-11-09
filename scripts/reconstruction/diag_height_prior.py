#!/usr/bin/env python3
"""Verify the geometric (height-above-floor) prior, using signed distance to the
fitted floor PLANE (SLAM world frame is not gravity-aligned, so plain z fails)."""
import numpy as np

BASE = "/root/autodl-tmp/ablation/scannet/scene0000_00"
d = np.load(f"{BASE}/semantic_cloud_10m/semantic_cloud.npz", allow_pickle=True)
pts = d["pts"].astype(np.float64)
label = d["label"].astype(np.int32)
prompts = [str(p) for p in d["prompts"]]

floor = label == prompts.index("floor")
fp = pts[floor]
fp_mean = fp.mean(0)
_, _, Vt = np.linalg.svd(fp - fp_mean, full_matrices=False)
normal = Vt[-1]
normal /= np.linalg.norm(normal)
# orient "up" = away from floor toward scene interior (scene centroid positive)
scene_mean = pts.mean(0)
if (scene_mean - fp_mean) @ normal < 0:
    normal = -normal
print(f"floor normal = ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})  (unit)")

# signed height above floor plane
h = (pts - fp_mean) @ normal
print(f"floor pts height: median {np.median(h[floor]):.3f}, std {h[floor].std():.3f} (should be ~0)")

print("\ncategory height above floor (signed, meters):")
print(f"{'name':<12} {'n':>9} {'h_med':>7} {'p10':>6} {'p90':>6} {'hmax':>7}")
for name in ["bed", "table", "desk", "cabinet", "bookshelf", "shelf", "counter",
             "sofa", "chair", "dresser", "night stand"]:
    if name not in prompts:
        continue
    c = prompts.index(name)
    m = label == c
    if m.sum() < 500:
        continue
    hm = h[m]
    qs = np.percentile(hm, [10, 50, 90])
    print(f"{name:<12} {m.sum():>9,} {qs[1]:>7.2f} {qs[0]:>6.2f} {qs[2]:>6.2f} {hm.max():>7.2f}")
