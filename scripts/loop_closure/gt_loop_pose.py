#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full SE(3) ground-truth difference for each loop candidate.

Reads the timestamps already recovered by gt_loop_distance.py (gt_loop_distance.json),
loads the full TUM ground-truth poses (tx ty tz qx qy qz qw), and for each loop
computes:

  pos_dist   = || t_Q - t_D ||                          (m)
  rot_deg    = angle of R_Q^T R_D, i.e. orientation gap  (deg)

A loop-closure constraint assumes T_Q ~= T_D (same position AND orientation).
A true loop has BOTH small; a false loop (same room, different viewpoint)
can have small pos_dist but LARGE rot_deg (e.g. camera rotated 90-180 deg).
"""
import os, json
import numpy as np


def load_gt(gt_path):
    ts, pos, quat = [], [], []
    with open(gt_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            v = line.split()
            ts.append(float(v[0]))
            pos.append(np.array([float(v[1]), float(v[2]), float(v[3])]))
            quat.append(np.array([float(v[4]), float(v[5]), float(v[6]), float(v[7])]))
    return np.array(ts), np.array(pos), np.array(quat)


def quat_to_rot(q):
    """qx qy qz qw -> 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def interpolate(ts, arr, t):
    i = np.searchsorted(ts, t)
    if i == 0:
        return arr[0]
    if i >= len(ts):
        return arr[-1]
    t0, t1 = ts[i - 1], ts[i]
    w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    return (1 - w) * arr[i - 1] + w * arr[i]


def rot_angle_deg(R_rel):
    """Rotation angle (deg) of a rotation matrix."""
    c = (np.trace(R_rel) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def main():
    base = "/root/autodl-tmp/tum_work"
    data = json.load(open("/root/autodl-tmp/gt_loop_distance.json"))

    # paper's labels (from Table 5) for reference
    paper_label = {
        ("room", 0): "假", ("room", 1): "真", ("room", 2): "真",
        ("360", 0): "假", ("360", 1): "假", ("360", 2): "真", ("360", 3): "假",
        ("desk2", 0): "真", ("desk2", 1): "真",
        ("floor", 0): "真",
        ("plant", 0): "真", ("plant", 1): "真",
        ("rpy", 0): "真",
        ("teddy", 0): "真", ("teddy", 1): "真",
    }

    print(f"{'seq':6s} {'loop':5s} {'paper':5s} {'pos_dist':>9s} {'rot_deg':>8s}   verdict")
    print("-" * 60)
    rows = []
    for rec in data:
        seq = rec["seq"]
        gt_path = os.path.join(base, f"rgbd_dataset_freiburg1_{seq}", "groundtruth.txt")
        ts, pos, quat = load_gt(gt_path)
        for k, r in enumerate(rec["loops"]):
            q_ts, d_ts = r["q_ts"], r["d_ts"]
            t_Q = interpolate(ts, pos, q_ts)
            t_D = interpolate(ts, pos, d_ts)
            pos_dist = float(np.linalg.norm(t_Q - t_D))

            # interpolate quaternions then normalize (slerp approx via lerp+norm)
            q_Q = interpolate(ts, quat, q_ts)
            q_D = interpolate(ts, quat, d_ts)
            q_Q = q_Q / np.linalg.norm(q_Q)
            q_D = q_D / np.linalg.norm(q_D)
            R_Q = quat_to_rot(q_Q)
            R_D = quat_to_rot(q_D)
            rot = rot_angle_deg(R_Q.T @ R_D)

            lab = paper_label.get((seq, k), "?")
            rows.append({"seq": seq, "loop": k, "paper": lab,
                         "pos_dist": pos_dist, "rot_deg": rot})
            print(f"{seq:6s} {k:<5d} {lab:5s} {pos_dist:9.4f} {rot:8.2f}")

    out = "/root/autodl-tmp/gt_loop_pose.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\n[saved] {out}")

    # separation stats
    print("\n" + "=" * 60)
    print("position-only separation:      false vs true")
    fp = [r for r in rows if r["paper"] == "假"]
    tp = [r for r in rows if r["paper"] == "真"]
    print(f"  false: {[round(r['pos_dist'],3) for r in fp]}")
    print(f"  true : {[round(r['pos_dist'],3) for r in tp]}")
    print(f"  false rot_deg: {[round(r['rot_deg'],1) for r in fp]}")
    print(f"  true  rot_deg: {[round(r['rot_deg'],1) for r in tp]}")


if __name__ == "__main__":
    main()
