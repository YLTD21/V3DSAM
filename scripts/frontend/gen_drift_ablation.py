#!/usr/bin/env python3
"""图一致性 → 实例坐标稳定性消融（路B 漂移消融的可行版）。

问题：贡献 2 声称「实例身份 7cm 几何自一致来自 SL(4) 全局一致性」。
要定量支持它，需要展示：当位姿图被一个坏约束（假回环）污染时，同一物体的
世界坐标会被拉开多远——即「没有正确的回环验证，身份坐标会漂移多远」。

这里用 room 序列的 lc_threshold 扫描（4 个阈值）：
  - 低阈值拒绝假回环 → 位姿图干净 → ATE ~0.07
  - 高阈值接受假回环 → 位姿图被污染 → ATE ~0.8
逐子图比较两者位姿位移 = 单个坏约束注入的坐标误差。

若该位移 >> 7cm（我们的几何自一致），则直接证明：
  没有回环验证（即没有全局一致性），身份坐标会偏移数十 cm，
  迫使采用宽松阈值（如竞品 Dinya 的 30cm Chamfer）。

CPU-only，仅 numpy，无需 evo（自己实现 SE3 Umeyama 对齐求 ATE）。
"""
import os, glob, argparse
import numpy as np


def load_tum(path):
    d = np.loadtxt(path)
    t = d[:, 1:4]          # (N,3) translation
    q = d[:, 4:8]          # (N,4) quaternion xyzw
    return t, q


def umeyama_align(src, dst):
    """SE(3) Umeyama alignment (scale=1). Returns aligned src and RMSE."""
    s = src.mean(0); d = dst.mean(0)
    S = src - s; D = dst - d
    H = S.T @ D
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = d - R @ s
    aligned = (R @ src.T).T + t
    rmse = np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1)))
    return aligned, rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lc_dir", required=True, help="tum_work/lc_sweep")
    ap.add_argument("--seq", default="room")
    ap.add_argument("--gt", required=True, help="groundtruth.txt")
    args = ap.parse_args()

    gt_t, _ = load_tum(args.gt)
    configs = {}
    for thr in ["0.95", "0.90", "0.85", "0.80"]:
        p = os.path.join(args.lc_dir, f"{args.seq}_lc{thr}", "stage1_poses.txt")
        if not os.path.exists(p):
            continue
        t, q = load_tum(p)
        # align each config to GT by timestamp order (assume same N, same order)
        n = min(len(t), len(gt_t))
        _, rmse = umeyama_align(t[:n], gt_t[:n])
        configs[thr] = {"t": t, "rmse": rmse, "path": p}
        print(f"  lc={thr}: ATE={rmse:.4f} m  (n={n})")

    # pick best (min rmse) and worst (max rmse)
    best = min(configs, key=lambda k: configs[k]["rmse"])
    worst = max(configs, key=lambda k: configs[k]["rmse"])
    tb = configs[best]["t"]; tw = configs[worst]["t"]
    n = min(len(tb), len(tw))
    disp = np.linalg.norm(tw[:n] - tb[:n], axis=1)
    print(f"\n  best lc={best} (ATE {configs[best]['rmse']:.4f}) vs worst lc={worst} (ATE {configs[worst]['rmse']:.4f})")
    print(f"  逐子图位姿位移: mean={disp.mean():.4f}m median={np.median(disp):.4f}m "
          f"p90={np.percentile(disp,90):.4f}m max={disp.max():.4f}m")
    print(f"\n  => 单个坏回环约束把世界坐标拉开 中位 {np.median(disp)*100:.1f} cm / 最大 {disp.max()*100:.1f} cm")
    print(f"     对比：干净图下实例几何自一致 = 7 cm（gen_reid_benchmark 随机分割）")


if __name__ == "__main__":
    main()
