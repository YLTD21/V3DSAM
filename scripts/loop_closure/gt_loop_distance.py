#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Independent ground-truth labeling of loop candidates.

For each of the 15 loop candidates (room/360/desk2/floor/plant/rpy/teddy),
re-run the VGGT-SLAM stage-1 front-end EXACTLY as in semantic_3d_check.py,
recover each loop's two frame timestamps (from submap.frame_ids), then read
the TUM-RGBD ground-truth camera trajectory and measure the Euclidean distance
between the two frames' GT world positions.

  dist small  -> genuine revisit  -> TRUE  loop
  dist large  -> visually similar, spatially different -> FALSE loop

This criterion is independent of the reported ATE and of any of our scores.
"""
import os, sys, glob, time, json
import numpy as np, cv2, torch

sys.path.insert(0, "/root/autodl-tmp/v3dsam")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/vggt")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/sam3")

from vggt_slam_official.solver import Solver
from vggt.models.vggt import VGGT

device = "cuda"


def run_stage1(image_dir, submap_size=32, min_disparity=50, lc_thres=0.95, max_loops=1):
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(
        "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"))
    model.eval(); model = model.to(torch.bfloat16).to(device)

    solver = Solver(init_conf_threshold=25.0, lc_thres=lc_thres)
    names = sorted(glob.glob(os.path.join(image_dir, "*")))
    loops = []
    subset = []
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
    return solver, loops


def load_gt(gt_path):
    ts, pos = [], []
    with open(gt_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            v = line.split()
            ts.append(float(v[0]))
            pos.append(np.array([float(v[1]), float(v[2]), float(v[3])]))
    return np.array(ts), np.array(pos)


def gt_pos_at(ts, pos, t):
    """Linear interpolation of GT position at timestamp t."""
    i = np.searchsorted(ts, t)
    if i == 0:
        return pos[0]
    if i >= len(ts):
        return pos[-1]
    t0, t1 = ts[i - 1], ts[i]
    p0, p1 = pos[i - 1], pos[i]
    w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    return (1 - w) * p0 + w * p1


def main():
    seqs = ["room", "360", "desk2", "floor", "plant", "rpy", "teddy"]
    base = "/root/autodl-tmp/tum_work"
    all_rows = []
    for seq in seqs:
        image_dir = os.path.join(base, f"rgbd_dataset_freiburg1_{seq}", "rgb")
        gt_path = os.path.join(base, f"rgbd_dataset_freiburg1_{seq}", "groundtruth.txt")
        solver, loops = run_stage1(image_dir)
        ts, pos = load_gt(gt_path)
        rows = []
        for k, lp in enumerate(loops):
            q_sm = solver.map.get_submap(lp["q_sid"])
            d_sm = solver.map.get_submap(lp["d_sid"])
            q_ts = float(q_sm.frame_ids[lp["q_fi"]])
            d_ts = float(d_sm.frame_ids[lp["d_fi"]])
            q_p = gt_pos_at(ts, pos, q_ts)
            d_p = gt_pos_at(ts, pos, d_ts)
            dist = float(np.linalg.norm(q_p - d_p))
            row = {**lp, "q_ts": q_ts, "d_ts": d_ts,
                   "q_xyz": q_p.tolist(), "d_xyz": d_p.tolist(), "gt_dist": dist}
            rows.append(row)
            print(f"[{seq} loop{k}] q=({lp['q_sid']},{lp['q_fi']}) d=({lp['d_sid']},{lp['d_fi']}) "
                  f"score={lp['score']:.4f}  GT_dist={dist:.4f} m")
        all_rows.append({"seq": seq, "loops": rows})

    out = "/root/autodl-tmp/gt_loop_distance.json"
    with open(out, "w") as f:
        json.dump(all_rows, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY: ground-truth distance (m) per loop candidate")
    print("=" * 80)
    for seq_rec in all_rows:
        for k, r in enumerate(seq_rec["loops"]):
            label = "TRUE " if r["gt_dist"] < 0.5 else "FALSE"
            print(f"  {seq_rec['seq']:6s} loop{k}  GT_dist={r['gt_dist']:7.4f} m  ->  {label}")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
