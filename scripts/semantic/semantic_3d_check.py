import os, sys, glob, time, argparse, json
import numpy as np, cv2, torch
import torch.nn.functional as F

sys.path.insert(0, "/root/autodl-tmp/v3dsam")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/vggt")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/sam3")

from vggt_slam_official.solver import Solver
from vggt.models.vggt import VGGT
from gtsam.symbol_shorthand import X

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
    lc_submaps = [sm for sm in solver.map.ordered_submaps_by_key() if sm.get_lc_status()]
    for k, lp in enumerate(loops):
        lp["lc_num"] = int(lc_submaps[k].get_id()) if k < len(lc_submaps) else None
    print(f"[3d_check] stage1 done {time.time()-t0:.1f}s, {len(loops)} loops")
    return solver, loops


def resize_mask(m, H, W):
    """m: (C, h, w) -> (C, H, W) via nearest."""
    if m.shape[1] == H and m.shape[2] == W:
        return m
    t = torch.from_numpy(m.astype(np.uint8))
    t = F.interpolate(t.unsqueeze(0).float(), (H, W), mode="nearest").squeeze(0)
    return t.numpy().astype(np.uint8)


def frame_data(solver, sid, fi):
    sm = solver.map.get_submap(sid)
    pts = sm.get_frame_pointcloud(fi)          # (H, W, 3) camera frame
    K = sm.proj_mats[fi][:3, :3]               # (3,3) intrinsics
    T = solver.graph.get_homography(sid + fi)  # (4,4) cam2world
    return pts, K, T


def transform_and_project(P_cam_D, T_D, T_Q, K_Q):
    """P_D (N,3) cam frame -> Q cam frame -> Q image pixels (N,2)."""
    N = P_cam_D.shape[0]
    ones = np.ones((N, 1))
    X_w = (T_D @ np.hstack([P_cam_D, ones]).T).T           # (N,4)
    X_w = X_w[:, :3] / X_w[:, 3:4]
    X_Q = (np.linalg.inv(T_Q) @ np.hstack([X_w, ones]).T).T  # (N,4)
    X_Q = X_Q[:, :3] / X_Q[:, 3:4]
    uv = (K_Q @ X_Q.T).T                                    # (N,3)
    u = uv[:, 0] / uv[:, 2]
    v = uv[:, 1] / uv[:, 2]
    return X_Q[:, 2], u, v                                  # depth, u, v


def consistency(label_D, label_Q, pts_D, T_D, T_Q, K_Q, HQ, WQ, max_pts=60000):
    """3D semantic consistency: D-labeled points -> Q frame -> label agreement."""
    ys, xs = np.nonzero(label_D >= 0)
    if len(xs) == 0:
        return 0.0, 0
    if len(xs) > max_pts:
        sel = np.random.default_rng(0).choice(len(xs), max_pts, replace=False)
        ys, xs = ys[sel], xs[sel]
    labs = label_D[ys, xs]
    P = pts_D[ys, xs].reshape(-1, 3).astype(np.float64)
    z, u, v = transform_and_project(P, T_D, T_Q, K_Q)
    ok = (z > 0.1) & (u >= 0) & (u < WQ) & (v >= 0) & (v < HQ)
    if ok.sum() < 50:
        return 0.0, int(ok.sum())
    uu = np.clip(u[ok].astype(int), 0, WQ - 1)
    vv = np.clip(v[ok].astype(int), 0, HQ - 1)
    agree = (label_Q[vv, uu] == labs[ok])
    return float(agree.mean()), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--submap_size", type=int, default=32)
    ap.add_argument("--min_disparity", type=float, default=50)
    ap.add_argument("--lc_thres", type=float, default=0.95)
    ap.add_argument("--max_loops", type=int, default=1)
    args = ap.parse_args()

    solver, loops = run_stage1(args.image_dir, args.submap_size,
                               args.min_disparity, args.lc_thres, args.max_loops)
    print(f"\n{'='*72}")
    print(f"  {args.seq}: 3D semantic consistency per loop")
    print(f"{'='*72}")
    results = []
    for k, lp in enumerate(loops):
        mz = os.path.join(args.mask_dir, f"loop{k}_masks.npz")
        if not os.path.exists(mz):
            print(f"loop{k}: no mask, skip")
            continue
        d = np.load(mz)
        mq, md = d["mask_q"], d["mask_d"]

        pts_Q, K_Q, T_Q = frame_data(solver, lp["q_sid"], lp["q_fi"])
        pts_D, K_D, T_D = frame_data(solver, lp["d_sid"], lp["d_fi"])
        HQ, WQ = pts_Q.shape[:2]; HD, WD = pts_D.shape[:2]

        mq_r = resize_mask(mq, HQ, WQ); md_r = resize_mask(md, HD, WD)
        label_Q = mq_r.argmax(0).astype(np.int16)
        label_Q[~mq_r.any(0)] = -1
        label_D = md_r.argmax(0).astype(np.int16)
        label_D[~md_r.any(0)] = -1

        # D -> Q direction
        c_dq, n_dq = consistency(label_D, label_Q, pts_D, T_D, T_Q, K_Q, HQ, WQ)
        # Q -> D direction (symmetric)
        c_qd, n_qd = consistency(label_Q, label_D, pts_Q, T_Q, T_D, K_D, HD, WD)
        c_both = 0.5 * (c_dq + c_qd)

        results.append({**lp, "cons_dq": c_dq, "cons_qd": c_qd, "cons": c_both,
                        "n_dq": n_dq, "n_qd": n_qd})
        print(f"loop{k}: score={lp['score']:.3f}  cons_dq={c_dq:.3f}  cons_qd={c_qd:.3f}  "
              f"cons_avg={c_both:.3f}  (n={n_dq}/{n_qd})")

    out = f"/root/autodl-tmp/semantic_out/{args.seq}/loop_log_3d.json"
    with open(out, "w") as f:
        json.dump({"loops": results}, f, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
