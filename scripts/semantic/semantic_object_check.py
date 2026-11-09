#!/usr/bin/env python3
"""
Direction 1 (object-level): semantic-geometric 3D object consistency for false-loop rejection.

Upgrades the per-pixel consistency in semantic_3d_check.py to OBJECT-level:
  1. Per frame: lift the SAM3 category mask onto the VGGT camera-frame pointcloud,
     cluster each class into objects via the user's own `cluster_objects_by_class`
     (vggt_slam.semantic_loop_verify) -> object centroids in CAMERA frame (pose-free,
     immune to graph corruption).
  2. Project D-frame object centroids into the Q-frame camera via the loop's estimated
     relative pose T_{D->Q} = inv(T_Q) @ T_D, then greedily match same-class centroids
     within a 3D distance threshold.
  3. Consistency = symmetric match ratio (matched objects / total objects). A TRUE loop
     has the same objects at the same 3D locations -> high ratio + low residual; a FALSE
     loop (same room, different viewpoint) has objects that do NOT line up -> low ratio.

Does NOT modify the user's v3dsam source. Imports cluster_objects_by_class verbatim.
"""

import os, sys, glob, time, argparse, json
import numpy as np, cv2, torch
import torch.nn.functional as F

sys.path.insert(0, "/root/autodl-tmp/v3dsam")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/vggt")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/sam3")

from vggt_slam_official.solver import Solver
from vggt_slam.semantic_loop_verify import cluster_objects_by_class
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
    lc_submaps = [sm for sm in solver.map.ordered_submaps_by_key() if sm.get_lc_status()]
    for k, lp in enumerate(loops):
        lp["lc_num"] = int(lc_submaps[k].get_id()) if k < len(lc_submaps) else None
    print(f"[obj_check] stage1 done {time.time()-t0:.1f}s, {len(loops)} loops")
    return solver, loops


def resize_mask(m, H, W):
    if m.shape[1] == H and m.shape[2] == W:
        return m
    t = torch.from_numpy(m.astype(np.uint8))
    t = F.interpolate(t.unsqueeze(0).float(), (H, W), mode="nearest").squeeze(0)
    return t.numpy().astype(np.uint8)


def frame_data(solver, sid, fi):
    sm = solver.map.get_submap(sid)
    pts = sm.get_frame_pointcloud(fi)          # (H, W, 3) camera frame
    K = sm.proj_mats[fi][:3, :3]
    T = solver.graph.get_homography(sid + fi)  # (4,4) cam2world
    return pts, K, T


def frame_objects(pts, label, T, min_points=80, cluster_eps=0.3, max_pts=120000):
    """Cluster per-class 3D points into objects, return centroids in WORLD frame.

    label: (H, W) int16, -1 = unlabeled. pts: (H, W, 3) camera frame. T: cam2world.
    Returns list of {class, centroid_world, extent, count}.
    """
    H, W = label.shape
    ys, xs = np.nonzero(label >= 0)
    if len(xs) == 0:
        return []
    if len(xs) > max_pts:
        sel = np.random.default_rng(0).choice(len(xs), max_pts, replace=False)
        ys, xs = ys[sel], xs[sel]
    labs = label[ys, xs]
    P = pts[ys, xs].reshape(-1, 3).astype(np.float64)
    ok = np.isfinite(P).all(1) & (P[:, 2] > 0.1)
    P, labs = P[ok], labs[ok]
    if len(P) == 0:
        return []
    # camera -> world
    ones = np.ones((len(P), 1))
    W = (T @ np.hstack([P, ones]).T).T
    W = W[:, :3] / W[:, 3:4]
    objs = cluster_objects_by_class(W, labs, min_points=min_points, cluster_eps=cluster_eps)
    return objs


def project_world_to_cam(c_world, T_Q):
    """world centroid -> Q camera frame (x,y,z)."""
    ones = np.ones((len(c_world), 1))
    X = (np.linalg.inv(T_Q) @ np.hstack([c_world, ones]).T).T
    return X[:, :3] / X[:, 3:4]


def object_consistency(objs_D, objs_Q, T_D, T_Q, match_thr=0.5):
    """Symmetric object match ratio + median residual.

    objs_D/objs_Q are in WORLD frame. We compare them via the loop's relative pose:
    D centroids -> Q camera frame via inv(T_Q)@T_D, then match to Q centroids (also in
    Q camera frame). This uses only the loop's own estimate, immune to graph corruption
    of the absolute world frame (which a false loop has already distorted).
    """
    nD, nQ = len(objs_D), len(objs_Q)
    if nD == 0 or nQ == 0:
        return 0.0, 0.0, 0, 0, 0.0

    T_dq = np.linalg.inv(T_Q) @ T_D           # D world -> Q world -> Q cam
    cD_world = np.array([o['centroid'] for o in objs_D])
    cQ_world = np.array([o['centroid'] for o in objs_Q])
    cD_qcam = project_world_to_cam(cD_world, T_Q)   # D centroids in Q camera frame
    cQ_qcam = project_world_to_cam(cQ_world, T_Q)   # Q centroids in Q camera frame
    clsD = np.array([o['class'] for o in objs_D])
    clsQ = np.array([o['class'] for o in objs_Q])

    # D -> Q greedy match
    matched_dq = 0
    resid_dq = []
    for i in range(nD):
        if cD_qcam[i, 2] < 0.1:              # D centroid behind Q camera -> invisible
            continue
        cand = np.where(clsQ == clsD[i])[0]
        if len(cand) == 0:
            continue
        d = np.linalg.norm(cQ_qcam[cand] - cD_qcam[i], axis=1)
        j = cand[int(np.argmin(d))]
        if d.min() < match_thr:
            matched_dq += 1
            resid_dq.append(float(d.min()))

    # Q -> D greedy match (symmetric)
    matched_qd = 0
    resid_qd = []
    T_qd = np.linalg.inv(T_D) @ T_Q
    cQ_dcam = project_world_to_cam(cQ_world, T_D)
    cD_dcam = project_world_to_cam(cD_world, T_D)
    for j in range(nQ):
        if cQ_dcam[j, 2] < 0.1:
            continue
        cand = np.where(clsD == clsQ[j])[0]
        if len(cand) == 0:
            continue
        d = np.linalg.norm(cD_dcam[cand] - cQ_dcam[j], axis=1)
        if d.min() < match_thr:
            matched_qd += 1
            resid_qd.append(float(d.min()))

    ratio = (matched_dq + matched_qd) / (nD + nQ)
    resid = float(np.median(resid_dq + resid_qd)) if (resid_dq or resid_qd) else float('inf')
    return ratio, resid, matched_dq, matched_qd, 0.5 * (nD + nQ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--submap_size", type=int, default=32)
    ap.add_argument("--min_disparity", type=float, default=50)
    ap.add_argument("--lc_thres", type=float, default=0.95)
    ap.add_argument("--max_loops", type=int, default=1)
    ap.add_argument("--match_thr", type=float, default=0.5)
    args = ap.parse_args()

    solver, loops = run_stage1(args.image_dir, args.submap_size,
                               args.min_disparity, args.lc_thres, args.max_loops)
    print(f"\n{'='*78}")
    print(f"  {args.seq}: OBJECT-level 3D semantic consistency (match_thr={args.match_thr}m)")
    print(f"{'='*78}")
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
        label_Q = mq_r.argmax(0).astype(np.int16); label_Q[~mq_r.any(0)] = -1
        label_D = md_r.argmax(0).astype(np.int16); label_D[~md_r.any(0)] = -1

        objs_D = frame_objects(pts_D, label_D, T_D)
        objs_Q = frame_objects(pts_Q, label_Q, T_Q)

        ratio, resid, mdq, mqd, tot = object_consistency(objs_D, objs_Q, T_D, T_Q,
                                                         match_thr=args.match_thr)
        # class overlap (reference, expected to fail)
        classes_D = set(label_D[label_D >= 0].tolist())
        classes_Q = set(label_Q[label_Q >= 0].tolist())
        jacc = len(classes_D & classes_Q) / max(1, len(classes_D | classes_Q))

        results.append({**lp, "n_obj_D": len(objs_D), "n_obj_Q": len(objs_Q),
                        "match_ratio": ratio, "resid_med": resid,
                        "matched_dq": mdq, "matched_qd": mqd,
                        "class_jaccard": jacc})
        print(f"loop{k}: score={lp['score']:.3f}  objs_D={len(objs_D)} objs_Q={len(objs_Q)}  "
              f"match_ratio={ratio:.3f}  resid_med={resid:.3f}  (matched {mdq}/{mqd}, jacc={jacc:.2f})")

    out = f"/root/autodl-tmp/semantic_out/{args.seq}/loop_log_obj.json"
    with open(out, "w") as f:
        json.dump({"loops": results, "match_thr": args.match_thr}, f, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
