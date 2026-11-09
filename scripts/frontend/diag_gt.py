#!/usr/bin/env python3
"""诊断 room GT 与估计轨迹的尺度/对齐。"""
import numpy as np


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
    return aligned, s


def main():
    gt = np.loadtxt("/root/autodl-tmp/tum_work/rgbd_dataset_freiburg1_room/groundtruth.txt",
                    comments="#")
    gt_ts = gt[:, 0]; gt_t = gt[:, 1:4]
    print(f"gt n={len(gt_ts)}  sorted={bool(np.all(np.diff(gt_ts) > 0))}  "
          f"ts=[{gt_ts.min():.4f}, {gt_ts.max():.4f}]")
    print(f"gt extent (m) xyz = {np.round(gt_t.max(0) - gt_t.min(0), 3)}")

    s = np.load("/root/autodl-tmp/psnr_tum/room/stage1_scaled/kf_semantic.npz", allow_pickle=True)
    paths = [str(p) for p in s["img_paths"]]
    kf_ts = np.array([float(p.rsplit("/", 1)[-1][:-4]) for p in paths])  # strip .png, keep fractional
    kf_indices = [int(x) for x in s["kf_indices"].tolist()]
    print(f"kf n={len(kf_ts)}  ts=[{kf_ts.min():.4f}, {kf_ts.max():.4f}]")

    d = np.load("/root/autodl-tmp/psnr_tum/room/stage1_scaled/_stage3_input.npz", allow_pickle=True)
    cam_w = []
    for fi in kf_indices:
        w2c = d[f"w2c_{fi}"].astype(np.float64)
        R, t = w2c[:3, :3], w2c[:3, 3]
        cam_w.append(-R.T @ t)
    cam_w = np.array(cam_w)
    print(f"cam_w extent = {np.round(cam_w.max(0) - cam_w.min(0), 3)}")

    # timestamp match
    gi = np.searchsorted(gt_ts, kf_ts).clip(1, len(gt_ts) - 1)
    gt_kf = np.stack([gt_t[j - 1] if (kf_ts[i] - gt_ts[j - 1] < gt_ts[j] - kf_ts[i]) else gt_t[j]
                      for i, j in enumerate(gi)])
    dt = np.abs(gt_ts[gi] - kf_ts)
    print(f"ts match max|dt| = {dt.max():.4f}s  median = {np.median(dt):.4f}s")

    # similarity align GT -> est
    aligned, scale = umeyama(gt_kf, cam_w, with_scale=True)
    rmse = np.sqrt(np.mean(np.sum((aligned - cam_w) ** 2, axis=1)))
    print(f"similarity align: scale={scale:.4f}  RMSE={rmse:.4f} m")
    # extent after align (should match cam_w extent)
    print(f"aligned gt extent = {np.round(aligned.max(0) - aligned.min(0), 3)}")


if __name__ == "__main__":
    main()
