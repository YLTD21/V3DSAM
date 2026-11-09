#!/usr/bin/env python3
"""Crop RGB frames showing the 'bed' blob (label==bed) so we can see what it actually is."""
import numpy as np, cv2, os

BASE = "/root/autodl-tmp/ablation/scannet/scene0000_00"
OUT = "/root/autodl-tmp/bed_crops"
os.makedirs(OUT, exist_ok=True)

d = np.load(f"{BASE}/semantic_cloud_10m/semantic_cloud.npz", allow_pickle=True)
pts = d["pts"].astype(np.float64)
label = d["label"].astype(np.int32)
prompts = [str(p) for p in d["prompts"]]
bed_pts = pts[label == prompts.index("bed")]
print(f"bed blob: {len(bed_pts):,} pts")

stage3 = np.load(f"{BASE}/_stage3_input_after.npz", allow_pickle=True)
sem = np.load(f"{BASE}/kf_semantic_v2.npz", allow_pickle=True)
kf_indices = [int(x) for x in sem["kf_indices"]]
img_W = int(stage3["img_W"]); img_H = int(stage3["img_H"])

ph = np.hstack([bed_pts, np.ones((len(bed_pts), 1), dtype=np.float64)])
rows = []
for fi in kf_indices:
    w2c = stage3[f"w2c_{fi}"].astype(np.float64)
    K = stage3[f"K_{fi}"].astype(np.float64)
    cam = (w2c @ ph.T).T
    z = cam[:, 2]
    front = z > 0.05
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * cam[:, 0] / np.maximum(z, 1e-6) + cx
    v = fy * cam[:, 1] / np.maximum(z, 1e-6) + cy
    inb = front & (u >= 0) & (u < img_W) & (v >= 0) & (v < img_H)
    n = int(inb.sum())
    if n < 300:
        continue
    umin, umax = u[inb].min(), u[inb].max()
    vmin, vmax = v[inb].min(), v[inb].max()
    rows.append((n, (umax - umin) * (vmax - vmin), fi, umin, umax, vmin, vmax))

rows.sort(key=lambda x: -x[0])
print("top frames by visible bed-pts:", [(r[2], r[0]) for r in rows[:12]])

# pick top 4, but spread across the sequence
top = rows[:12]
top.sort(key=lambda x: x[2])
sel = [top[i] for i in [0, len(top)//3, 2*len(top)//3, -1]]
# dedupe
seen = set(); sel = [r for r in sel if not (r[2] in seen or seen.add(r[2]))]

pad = 80
for r in sel:
    n, area, fi, umin, umax, vmin, vmax = r
    img = cv2.imread(str(stage3[f"img_path_{fi}"]))
    if img is None:
        print(f"  kf{fi}: img read fail"); continue
    x0 = max(0, int(umin) - pad); x1 = min(img_W, int(umax) + pad)
    y0 = max(0, int(vmin) - pad); y1 = min(img_H, int(vmax) + pad)
    crop = img[y0:y1, x0:x1]
    cv2.imwrite(f"{OUT}/crop_kf{fi}.jpg", crop)
    full = img.copy()
    cv2.rectangle(full, (int(umin), int(vmin)), (int(umax), int(vmax)), (0, 0, 255), 3)
    small = cv2.resize(full, (img_W // 4, img_H // 4))
    cv2.imwrite(f"{OUT}/full_kf{fi}.jpg", small)
    print(f"  kf{fi}: visible={n}, bbox=({umin:.0f},{vmin:.0f})-({umax:.0f},{vmax:.0f})")
print("done ->", OUT)
