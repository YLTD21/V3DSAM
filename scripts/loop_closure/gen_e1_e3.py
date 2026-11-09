import os, sys, glob, time, json, argparse
import numpy as np, cv2, torch
import torch.nn.functional as F

sys.path.insert(0, "/root/autodl-tmp/v3dsam")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/vggt")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/sam3")
from vggt_slam_official.solver import Solver
from vggt.models.vggt import VGGT

device = "cuda"
BASE = "/root/autodl-tmp"


class AlphaCapture:
    """Wrap VGGT to record image_match_ratio every time compute_similarity=True."""
    def __init__(self, model):
        self._m = model
        self.alphas = []

    def __call__(self, *a, **k):
        out = self._m(*a, **k)
        if k.get("compute_similarity", False):
            self.alphas.append(float(out["image_match_ratio"]))
        return out

    def __getattr__(self, n):
        return getattr(self._m, n)


def load_model():
    model = VGGT()
    sd = torch.hub.load_state_dict_from_url(
        "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt")
    model.load_state_dict(sd)
    model.eval()
    return model.to(torch.bfloat16).to(device)


def run_stage1(image_dir, submap_size, min_disparity, lc_thres, max_loops):
    model = load_model()
    cap = AlphaCapture(model)
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
            pred = solver.run_predictions(subset, cap, max_loops, None, None)
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
    print(f"[stage1] {len(loops)} loops, {len(cap.alphas)} alpha-records, {time.time()-t0:.1f}s")
    return solver, loops, cap.alphas


def resize_mask(m, H, W):
    if m.shape[1] == H and m.shape[2] == W:
        return m
    t = torch.from_numpy(m.astype(np.uint8))
    t = F.interpolate(t.unsqueeze(0).float(), (H, W), mode="nearest").squeeze(0)
    return t.numpy().astype(np.uint8)


def frame_data(solver, sid, fi):
    sm = solver.map.get_submap(sid)
    pts = sm.get_frame_pointcloud(fi)
    K = sm.proj_mats[fi][:3, :3]
    T = solver.graph.get_homography(sid + fi)
    return pts, K, T


def transform_and_project(P_cam_D, T_D, T_Q, K_Q):
    N = P_cam_D.shape[0]
    ones = np.ones((N, 1))
    X_w = (T_D @ np.hstack([P_cam_D, ones]).T).T
    X_w = X_w[:, :3] / X_w[:, 3:4]
    X_Q = (np.linalg.inv(T_Q) @ np.hstack([X_w, ones]).T).T
    X_Q = X_Q[:, :3] / X_Q[:, 3:4]
    uv = (K_Q @ X_Q.T).T
    return X_Q[:, 2], uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]


def consistency(label_D, label_Q, pts_D, T_D, T_Q, K_Q, HQ, WQ, max_pts=60000):
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
    return float((label_Q[vv, uu] == labs[ok]).mean()), int(ok.sum())


def load_loop_data(solver, lp, mask_dir, k):
    mz = os.path.join(mask_dir, f"loop{k}_masks.npz")
    d = np.load(mz)
    mq, md = d["mask_q"], d["mask_d"]
    pts_Q, K_Q, T_Q = frame_data(solver, lp["q_sid"], lp["q_fi"])
    pts_D, K_D, T_D = frame_data(solver, lp["d_sid"], lp["d_fi"])
    HQ, WQ = pts_Q.shape[:2]
    HD, WD = pts_D.shape[:2]
    mq_r = resize_mask(mq, HQ, WQ)
    md_r = resize_mask(md, HD, WD)
    label_Q = mq_r.argmax(0).astype(np.int16)
    label_Q[~mq_r.any(0)] = -1
    label_D = md_r.argmax(0).astype(np.int16)
    label_D[~md_r.any(0)] = -1
    return (label_D, label_Q, pts_D, pts_Q, T_D, T_Q, K_D, K_Q,
            HD, WD, HQ, WQ, int(mq.shape[0]))


def cons_both(label_D, label_Q, pts_D, pts_Q, T_D, T_Q, K_D, K_Q, HD, WD, HQ, WQ):
    c_dq, n_dq = consistency(label_D, label_Q, pts_D, T_D, T_Q, K_Q, HQ, WQ)
    c_qd, n_qd = consistency(label_Q, label_D, pts_Q, T_Q, T_D, K_D, HD, WD)
    return 0.5 * (c_dq + c_qd), n_dq, n_qd


def inject_noise(label, p, rng):
    out = label.copy()
    ys, xs = np.nonzero(label >= 0)
    if len(xs) == 0 or p <= 0:
        return out
    n = int(round(p * len(xs)))
    if n == 0:
        return out
    sel = rng.choice(len(xs), n, replace=False)
    C = int(label.max()) + 1
    out[ys[sel], xs[sel]] = rng.integers(0, C, size=n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--submap_size", type=int, default=32)
    ap.add_argument("--min_disparity", type=float, default=50)
    ap.add_argument("--lc_thres", type=float, default=0.95)
    ap.add_argument("--max_loops", type=int, default=1)
    ap.add_argument("--noise_levels", default="0,0.05,0.10,0.15,0.20,0.30,0.40,0.50")
    ap.add_argument("--n_seeds", type=int, default=3)
    args = ap.parse_args()

    seq = args.seq
    img_dir = f"{BASE}/tum_work/rgbd_dataset_freiburg1_{seq}/rgb"
    mask_dir = f"{BASE}/semantic_out/{seq}/loop_figures"
    noise_levels = [float(x) for x in args.noise_levels.split(",")]

    solver, loops, alphas = run_stage1(img_dir, args.submap_size, args.min_disparity,
                                       args.lc_thres, args.max_loops)

    # ---- E1: attach alpha ----
    if len(alphas) == len(loops):
        for k, lp in enumerate(loops):
            lp["alpha"] = alphas[k]
    else:
        print(f"[warn] alpha count {len(alphas)} != loops {len(loops)}")
        for k in range(len(loops)):
            lp["alpha"] = alphas[-len(loops) + k] if len(alphas) >= len(loops) else None

    # ---- E3: noise robustness ----
    noise_result = []
    for k, lp in enumerate(loops):
        try:
            data = load_loop_data(solver, lp, mask_dir, k)
        except Exception as e:
            print(f"[skip] loop{k}: {e}")
            continue
        label_D, label_Q, pts_D, pts_Q, T_D, T_Q, K_D, K_Q, HD, WD, HQ, WQ, C = data
        row = {"loop": k, "q_sid": lp["q_sid"], "d_sid": lp["d_sid"],
               "score": lp["score"], "alpha": lp.get("alpha"), "C": C, "noise": {}}
        for p in noise_levels:
            cs = []
            for seed in range(args.n_seeds):
                rng = np.random.default_rng(seed)
                nD = inject_noise(label_D, p, rng)
                nQ = inject_noise(label_Q, p, rng)
                c, _, _ = cons_both(nD, nQ, pts_D, pts_Q, T_D, T_Q, K_D, K_Q,
                                    HD, WD, HQ, WQ)
                cs.append(float(c))
            row["noise"][str(p)] = cs
        noise_result.append(row)
        clean = float(np.mean(row["noise"]["0.0"]))
        print(f"[{seq}] loop{k}: alpha={lp.get('alpha')} score={lp['score']:.3f} clean={clean:.3f}")

    os.makedirs(f"{BASE}/semantic_out/{seq}", exist_ok=True)
    with open(f"{BASE}/semantic_out/{seq}/loop_log_alpha.json", "w") as f:
        json.dump({"loops": loops, "n_alpha": len(alphas)}, f, indent=2, ensure_ascii=False)
    with open(f"{BASE}/semantic_out/{seq}/loop_log_noise.json", "w") as f:
        json.dump({"loops": noise_result, "noise_levels": noise_levels,
                   "n_seeds": args.n_seeds}, f, indent=2, ensure_ascii=False)
    print(f"[saved] {seq}/loop_log_alpha.json + loop_log_noise.json")


if __name__ == "__main__":
    main()
