#!/usr/bin/env python3
"""Re-vote the OLD 'bed' 3D points against NEW kf_semantic_v2 labels.
Answers: what does the blob that was labeled 'bed' become after the fix?"""
import numpy as np, time

BASE = "/root/autodl-tmp/ablation/scannet/scene0000_00"

old = np.load(f"{BASE}/semantic_cloud_10m/semantic_cloud.npz", allow_pickle=True)
pts = old["pts"].astype(np.float64)
old_label = old["label"].astype(np.int32)
old_prompts = [str(p) for p in old["prompts"]]

new = np.load(f"{BASE}/kf_semantic_v2.npz", allow_pickle=True)
new_labels = new["labels"]
new_kf = [int(x) for x in new["kf_indices"]]
new_prompts = [str(p) for p in new["prompts"]]

stage3 = np.load(f"{BASE}/_stage3_input_after.npz", allow_pickle=True)

bed_c = old_prompts.index("bed")
bed_mask = old_label == bed_c
bed_pts = pts[bed_mask]
print(f"old 'bed' points: {len(bed_pts):,} / {len(pts):,}")

pts_h = np.hstack([bed_pts, np.ones((len(bed_pts), 1), dtype=np.float64)])
n_cats = len(new_prompts)
H, W = new_labels.shape[1], new_labels.shape[2]
votes = np.zeros((n_cats, len(bed_pts)), dtype=np.uint16)

t0 = time.time()
for k, fi in enumerate(new_kf):
    w2c = stage3[f"w2c_{fi}"].astype(np.float64)
    K = stage3[f"K_{fi}"].astype(np.float64)
    cam = (w2c @ pts_h.T).T
    z = cam[:, 2]
    front = z > 0.05
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * cam[:, 0] / z + cx
    v = fy * cam[:, 1] / z + cy
    ui = np.floor(u).astype(np.int64)
    vi = np.floor(v).astype(np.int64)
    inb = front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    if not inb.any():
        continue
    lab = new_labels[k][vi[inb], ui[inb]]
    keep = lab >= 0
    if not keep.any():
        continue
    pidx = np.nonzero(inb)[0][keep]
    lab = lab[keep].astype(np.int64)
    for c in np.unique(lab):
        pc = pidx[lab == c]
        cnt = np.bincount(pc)
        idx = np.nonzero(cnt)[0]
        votes[c, idx] += cnt[idx].astype(np.uint16)

total = votes.sum(0)
sid = votes.argmax(0).astype(np.int32)
sid[total < 2] = -1
n_voted = int((total >= 2).sum())
print(f"voted over {len(new_kf)} kf in {time.time()-t0:.0f}s; {n_voted:,} bed-pts got >=2 votes")

cnt = np.bincount(sid[sid >= 0], minlength=n_cats)
order = np.argsort(-cnt)
print("new label of old-'bed' points:")
for c in order[:12]:
    if cnt[c] > 0:
        print(f"  {new_prompts[c]:<12} idx={c:<3} {cnt[c]:>10,}  ({cnt[c]/n_voted*100:.1f}%)")
