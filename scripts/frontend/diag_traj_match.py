#!/usr/bin/env python3
"""Diagnose the relationship between the video's raw (pre-filter) trajectory and
V3DSAM's corrected (post-semantic-filter) trajectory for fr1/360.

Loads:
  - corrected  : 360_reject/stage1_poses_after.txt  (camera centers, timestamps)
  - before     : stage3 _stage3_input.npz keyframe w2c -> camera centers
Matches keyframes by timestamp, reports alignment + per-frame drift.
"""
import os, sys, numpy as np

STAGE3 = "/root/autodl-tmp/psnr_tum/360/stage1/_stage3_input.npz"
AFTER  = "/root/autodl-tmp/semantic_out/360_reject/stage1_poses_after.txt"
BEFORE = "/root/autodl-tmp/semantic_out/360_reject/stage1_poses_before.txt"

def load_tum(path):
    d = np.loadtxt(path)
    ts = d[:, 0]
    xyz = d[:, 1:4]
    quat = d[:, 4:8]  # qx qy qz qw
    return ts, xyz, quat

def centers_from_stage3(npz):
    d = np.load(npz, allow_pickle=True)
    kf = [int(x) for x in d["vggt_kf_indices"].tolist()]
    ts = []
    xyz = []
    for fi in kf:
        p = str(d[f"img_path_{fi}"])
        ts.append(float(os.path.basename(p)[:-4]))
        w = d[f"w2c_{fi}"].astype(np.float64)
        R = w[:3, :3]; t = w[:3, 3]
        xyz.append(-R.T @ t)
    return np.array(ts), np.array(xyz)

def umeyama(src, dst, with_scale=True):
    s_ = src.mean(0); d_ = dst.mean(0)
    S = src - s_; D = dst - d_
    cov = S.T @ D
    U, sv, Vt = np.linalg.svd(cov)
    Sfix = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Sfix[-1, -1] = -1
    R = U @ Sfix @ Vt
    s = (np.trace(np.diag(sv) @ Sfix) / np.sum(S ** 2)) if with_scale else 1.0
    t = d_ - s * (R @ s_)
    aligned = (s * (R @ src.T)).T + t
    return aligned, R, t, s

after_ts, after_xyz, after_q = load_tum(AFTER)
before_ts, before_xyz, before_q = load_tum(BEFORE)
kf_ts, kf_xyz = centers_from_stage3(STAGE3)

print(f"after  : {len(after_ts)} poses, ts range [{after_ts[0]:.3f}, {after_ts[-1]:.3f}]")
print(f"before : {len(before_ts)} poses")
print(f"kf (stage3): {len(kf_ts)} keyframes, ts range [{kf_ts[0]:.3f}, {kf_ts[-1]:.3f}]")

# 1) before(file) vs after(file): same timestamps?
print(f"\nbefore==after timestamps identical: {np.allclose(before_ts, after_ts)}")

# 2) after vs kf(stage3): match by nearest timestamp
def match(ts_a, xyz_a, ts_b, xyz_b, tol=1e-3):
    pairs = []
    for i, t in enumerate(ts_a):
        j = np.argmin(np.abs(ts_b - t))
        if abs(ts_b[j] - t) < tol:
            pairs.append((i, j))
    return pairs

pairs_after_kf = match(after_ts, after_xyz, kf_ts, kf_xyz)
print(f"\nafter<->kf matched: {len(pairs_after_kf)} / {len(after_ts)}")

if pairs_after_kf:
    ia = [p[0] for p in pairs_after_kf]
    ik = [p[1] for p in pairs_after_kf]
    A = after_xyz[ia]; K = kf_xyz[ik]
    # align kf(before, stage3) -> after
    K_aligned, R, t, s = umeyama(K, A)
    resid = np.linalg.norm(K_aligned - A, axis=1)
    print(f"umeyama(kf_before -> after): scale={s:.4f}, rmse={np.sqrt(np.mean(resid**2)):.4f} m, "
          f"max={resid.max():.4f} m, median={np.median(resid):.4f} m")

    # raw before/after diff (same timestamps, no alignment)
    raw_diff = np.linalg.norm(A - K, axis=1)
    print(f"raw |after - kf_before| (matched): mean={raw_diff.mean():.4f} m, "
          f"median={np.median(raw_diff):.4f} m, max={raw_diff.max():.4f} m")

    # drift vs position along trajectory
    order = np.argsort(np.linalg.norm(A, axis=1))
    print(f"  drift at start (first 10%): {np.median(raw_diff[order[:len(order)//10]]):.4f} m")
    print(f"  drift at end   (last 10%):  {np.median(raw_diff[order[-len(order)//10:]]):.4f} m")
