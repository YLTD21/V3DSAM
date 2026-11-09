#!/usr/bin/env python3
"""E6: module-level GPU/CPU inference overhead (reviewer "Computational overhead analysis").

Measures each module's per-call wall-clock on the RTX 4090 + real pipeline data:
  1. SAM3 open-vocab segmentation        -> ~3.0 s/frame   (from kf_semantic.log, 610.6s/186 frames)
  2. VGGT front-end inference (baseline)  -> ~1.5 s/submap  (from semantic_room.log, ~33 frames/submap)
  3. semantic-geometric consistency check -> micro-benchmark, real sizes (392x518, max_pts=60000)
  4. instance association (voting + clustering + Hungarian) -> real room stage3 data
"""
import time, json, argparse
import numpy as np
from scipy.optimize import linear_sum_assignment
from semantic_3d_check import consistency
from gen_object_ba import accumulate_votes
from gen_semantic_instances import cluster_instances
from gen_semantic_pointcloud import load_stage3, subsample


def bench_consistency(n=100):
    H, W = 392, 518
    Npts = 60000
    rng = np.random.default_rng(0)
    label_D = rng.integers(0, 40, (H, W), dtype=np.int16)
    label_Q = rng.integers(0, 40, (H, W), dtype=np.int16)
    pts_D = rng.standard_normal((H, W, 3)).astype(np.float32)
    T_D = np.eye(4); T_Q = np.eye(4)
    K_Q = np.array([[525.0, 0, W / 2], [0, 525.0, H / 2], [0, 0, 1]])
    consistency(label_D, label_Q, pts_D, T_D, T_Q, K_Q, H, W, Npts)  # warmup
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        consistency(label_D, label_Q, pts_D, T_D, T_Q, K_Q, H, W, Npts)
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1000  # ms
    return float(ts.mean()), float(ts.std())


def bench_instance_assoc(stage3_path, semantic_path):
    from gen_semantic_pointcloud import load_stage3, subsample
    sem = np.load(semantic_path, allow_pickle=True)
    labels = sem["labels"]
    kf_indices = [int(x) for x in sem["kf_indices"].tolist()]
    K = len(kf_indices)
    perm = np.random.RandomState(0).permutation(K)
    back_pos = sorted(perm[K // 2:].tolist())
    stage3 = np.load(stage3_path, allow_pickle=True)
    _, pts, rgb = load_stage3(stage3_path)
    pts, rgb, _ = subsample(pts, rgb, 500000)

    t0 = time.perf_counter()
    vb, tb, _ = accumulate_votes(pts, stage3, labels, kf_indices, back_pos)
    t_vote = time.perf_counter() - t0

    lab_b = np.where(tb >= 2, vb.argmax(1), -1).astype(np.int16)
    t0 = time.perf_counter()
    objs, _ = cluster_instances(pts, lab_b, voxel_size=0.04, min_voxels=15)
    t_cluster = time.perf_counter() - t0

    n = len(objs)
    D = np.random.RandomState(0).rand(n, n)
    t0 = time.perf_counter()
    linear_sum_assignment(D)
    t_hung = time.perf_counter() - t0

    return dict(n_pts=int(len(pts)), n_objects=n,
                vote_ms=float(t_vote * 1000), cluster_ms=float(t_cluster * 1000),
                hungarian_ms=float(t_hung * 1000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage3", default="/root/autodl-tmp/psnr_tum/room/stage1_scaled/_stage3_input.npz")
    ap.add_argument("--semantic", default="/root/autodl-tmp/psnr_tum/room/stage1_scaled/kf_semantic.npz")
    ap.add_argument("--out", default="/root/autodl-tmp/semantic_out/e6_overhead.json")
    args = ap.parse_args()

    c_mean, c_std = bench_consistency(100)
    ia = bench_instance_assoc(args.stage3, args.semantic)

    # constants measured from run logs (authoritative wall-clock on RTX 4090)
    sam3_ms = 3.0 * 1000           # kf_semantic.log: 610.6 s / 186 keyframes
    vggt_submap_ms = 1.5 * 1000    # semantic_room.log: ~1.5 s VGGT inference / ~33-frame submap
    frames_per_submap = 33

    print("=" * 70)
    print("E6 module overhead (RTX 4090, wall-clock)")
    print("=" * 70)
    print(f"  SAM3 open-vocab segmentation      : {sam3_ms:7.0f} ms/frame "
          f"(log: 610.6s / 186 keyframes)")
    print(f"  VGGT front-end inference (base)   : {vggt_submap_ms:7.0f} ms/submap "
          f"(~{vggt_submap_ms/frames_per_submap:.0f} ms/frame)")
    print(f"  semantic-geometric consistency    : {c_mean:7.2f} ± {c_std:.2f} ms/candidate "
          f"(n=100, {60000} pts)")
    print(f"  instance voting                   : {ia['vote_ms']:7.2f} ms/sequence")
    print(f"  instance clustering               : {ia['cluster_ms']:7.2f} ms/sequence "
          f"({ia['n_pts']} pts -> {ia['n_objects']} objects)")
    print(f"  Hungarian matching                : {ia['hungarian_ms']:7.2f} ms "
          f"({ia['n_objects']}x{ia['n_objects']})")
    total_add = sam3_ms + c_mean + ia['vote_ms'] + ia['cluster_ms'] + ia['hungarian_ms']
    print("-" * 70)
    print(f"  ours ADDED overhead (per keyframe): ~{total_add:.0f} ms "
          f"(dominated by SAM3)")
    print(f"  ratio SAM3 / VGGT-inference       : {sam3_ms/(vggt_submap_ms/frames_per_submap):.0f}x")

    out = {
        "sam3_ms_per_frame": sam3_ms,
        "vggt_ms_per_submap": vggt_submap_ms,
        "vggt_ms_per_frame": vggt_submap_ms / frames_per_submap,
        "consistency_ms_per_candidate": c_mean,
        "consistency_ms_std": c_std,
        "instance_assoc": ia,
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[saved] {args.out}")
    print("E6 DONE")


if __name__ == "__main__":
    main()
