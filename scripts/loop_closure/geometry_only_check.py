#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""纯几何(无语义)深度一致性回环判别 —— 消融对照 baseline。

与 semantic_3d_check.py 的唯一区别:
  semantic 版本:  agree = (label_Q[vv,uu] == labs[ok])   # 离散类别标签对齐
  geometry 版本:  agree = |z_proj - z_Q| / z_Q < thr        # 连续深度对齐

两者用**完全相同**的 Stage1 位姿 + VGGT 深度 + 相对变换 T_{D->Q},
唯一变量是"一致性判据"(标签 vs 深度),从而干净地回答:
  "语义标签相对纯几何深度,是否提供了额外的判别力?"

不需要 SAM3 mask,故 GPU 只跑 VGGT(bf16 ~2GB),可与其他任务并行。
"""
import os, sys, glob, time, argparse, json
import numpy as np, cv2, torch

sys.path.insert(0, "/root/autodl-tmp/v3dsam")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/vggt")

from vggt_slam_official.solver import Solver
from vggt.models.vggt import VGGT

device = "cuda"


def run_stage1(image_dir, submap_size, min_disparity, lc_thres, max_loops):
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(
        "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"))
    model.eval(); model = model.to(torch.bfloat16).to(device)

    solver = Solver(init_conf_threshold=25.0, lc_thres=lc_thres)
    names = sorted(glob.glob(os.path.join(image_dir, "*")))
    loops = []
    subset = []
    t0 = time.time()
    for name in names:
        img = cv2.imread(name)
        if solver.flow_tracker.compute_disparity(img, min_disparity, False):
            subset.append(name)
        if len(subset) == submap_size + 1 or name == names[-1]:
            pred = solver.run_predictions(subset, model, max_loops, None, None)
            for loop in pred.get("detected_loops", []):
                loops.append({
                    "q_sid": int(loop.query_submap_id),
                    "q_fi": int(loop.query_submap_frame),
                    "d_sid": int(loop.detected_submap_id),
                    "d_fi": int(loop.detected_submap_frame),
                    "score": float(loop.similarity_score),
                })
            solver.add_points(pred)
            solver.graph.optimize()
            subset = subset[-1:]
    print(f"[geo_check] stage1 done {time.time()-t0:.1f}s, {len(loops)} loops")
    return solver, loops


def frame_data(solver, sid, fi):
    sm = solver.map.get_submap(sid)
    pts = sm.get_frame_pointcloud(fi)          # (H, W, 3) camera frame
    K = sm.proj_mats[fi][:3, :3]               # (3,3) intrinsics
    T = solver.graph.get_homography(sid + fi)  # (4,4) cam2world
    return pts, K, T


def transform_and_project(P_cam_D, T_D, T_Q, K_Q):
    N = P_cam_D.shape[0]
    ones = np.ones((N, 1))
    X_w = (T_D @ np.hstack([P_cam_D, ones]).T).T
    X_w = X_w[:, :3] / X_w[:, 3:4]
    X_Q = (np.linalg.inv(T_Q) @ np.hstack([X_w, ones]).T).T
    X_Q = X_Q[:, :3] / X_Q[:, 3:4]
    uv = (K_Q @ X_Q.T).T
    u = uv[:, 0] / uv[:, 2]
    v = uv[:, 1] / uv[:, 2]
    return X_Q[:, 2], u, v


def geometry_consistency(pts_D, T_D, T_Q, K_Q, pts_Q, HQ, WQ,
                         max_pts=60000, thr=0.10):
    """Dense depth consistency (no semantics): project D's 3D points to Q,
    agree = |z_proj - z_Q| / z_Q < thr. Bidirectional handled by caller."""
    # sample ALL valid-depth points (not only labeled ones)
    valid = np.isfinite(pts_D[..., 2]) & (pts_D[..., 2] > 0.1)
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return 0.0, 0
    if len(xs) > max_pts:
        sel = np.random.default_rng(0).choice(len(xs), max_pts, replace=False)
        ys, xs = ys[sel], xs[sel]
    P = pts_D[ys, xs].reshape(-1, 3).astype(np.float64)
    z, u, v = transform_and_project(P, T_D, T_Q, K_Q)
    ok = (z > 0.1) & (u >= 0) & (u < WQ) & (v >= 0) & (v < HQ)
    if ok.sum() < 50:
        return 0.0, int(ok.sum())
    uu = np.clip(u[ok].astype(int), 0, WQ - 1)
    vv = np.clip(v[ok].astype(int), 0, HQ - 1)
    z_Q = pts_Q[vv, uu, 2]
    agree = (np.isfinite(z_Q) & (z_Q > 0.1)
             & (np.abs(z[ok] - z_Q) / np.maximum(z_Q, 1e-6) < thr))
    return float(agree.mean()), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--submap_size", type=int, default=32)
    ap.add_argument("--min_disparity", type=float, default=50)
    ap.add_argument("--lc_thres", type=float, default=0.95)
    ap.add_argument("--max_loops", type=int, default=1)
    ap.add_argument("--thr", type=float, default=0.10)
    args = ap.parse_args()

    solver, loops = run_stage1(args.image_dir, args.submap_size,
                               args.min_disparity, args.lc_thres, args.max_loops)
    print(f"\n{'='*72}")
    print(f"  {args.seq}: geometry-only (depth) consistency per loop  thr={args.thr}")
    print(f"{'='*72}")
    results = []
    for k, lp in enumerate(loops):
        pts_Q, K_Q, T_Q = frame_data(solver, lp["q_sid"], lp["q_fi"])
        pts_D, K_D, T_D = frame_data(solver, lp["d_sid"], lp["d_fi"])
        HQ, WQ = pts_Q.shape[:2]; HD, WD = pts_D.shape[:2]

        g_dq, n_dq = geometry_consistency(pts_D, T_D, T_Q, K_Q, pts_Q, HQ, WQ, thr=args.thr)
        g_qd, n_qd = geometry_consistency(pts_Q, T_Q, T_D, K_D, pts_D, HD, WD, thr=args.thr)
        g_both = 0.5 * (g_dq + g_qd)

        results.append({**lp, "geo_dq": g_dq, "geo_qd": g_qd, "geo": g_both,
                        "n_dq": n_dq, "n_qd": n_qd})
        print(f"loop{k}: score={lp['score']:.3f}  geo_dq={g_dq:.3f}  geo_qd={g_qd:.3f}  "
              f"geo_avg={g_both:.3f}  (n={n_dq}/{n_qd})")

    outdir = f"/root/autodl-tmp/semantic_out/{args.seq}"
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "loop_log_geo.json")
    with open(out, "w") as f:
        json.dump({"loops": results, "thr": args.thr}, f, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
