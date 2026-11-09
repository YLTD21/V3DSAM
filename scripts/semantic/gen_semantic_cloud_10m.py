#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全密度（1000 万点）语义点云 + RGB 点云生成。

复用 gen_semantic_pointcloud 的投影思路：把 SAM3 逐帧类别标签（kf_semantic.npz）
正向投影到全局世界帧点云 pts_gs 上，跨关键帧多数投票得每个 3D 点的类别。

关键点：
  - pts_gs / rgb_gs 已是 1000 万点（_stage3_input_after.npz），这里**不做降采样**。
  - w2c_{fi} 与 est_after_tum.txt 逐元素一致（已验证），故直接用 npz 里的 w2c 投影，
    无需额外对齐。
  - 投票改成增量 bincount（uint32 累加），避免 10M×695 帧把 (pidx,lab) 全堆进内存。

输出（每场景 out_dir）：
  rgb_cloud.ply       二进制 PLY：xyz + 真实 RGB
  semantic_cloud.ply  二进制 PLY：xyz + 类别色 + label(int)
  semantic_cloud.npz  pts(float32) + label(int16) + prompts(object) —— 方便 Jupyter 读

Run on AutoDL (sgs env):
  /root/miniconda3/envs/sgs/bin/python gen_semantic_cloud_10m.py
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_semantic_pointcloud import load_stage3, PALETTE

BASE = "/root/autodl-tmp/ablation/scannet"
SCENES = ["scene0000_00", "scene0059_00", "scene0106_00", "scene0207_00"]


def vote_labels(pts, stage3, semantic, min_votes=2):
    """把逐帧 SAM3 标签投票到 3D 点上。内存友好：增量 bincount。"""
    s = np.load(semantic, allow_pickle=True)
    labels = s["labels"]                     # (K, H, W) int16, -1 = unlabeled
    kf_indices = [int(x) for x in s["kf_indices"].tolist()]
    prompts = [str(p) for p in s["prompts"]]
    n_cats = len(prompts)
    H, W = labels.shape[1], labels.shape[2]
    N = len(pts)

    import time
    pts_h = np.hstack([pts, np.ones((N, 1), dtype=np.float64)])   # (N,4)
    votes = np.zeros((n_cats, N), dtype=np.uint16)                # (n_cats,N) 每点每类得票
    n_seen = 0
    t0 = time.time()
    for k, fi in enumerate(kf_indices):
        if f"w2c_{fi}" not in stage3.files or f"K_{fi}" not in stage3.files:
            continue
        w2c = stage3[f"w2c_{fi}"].astype(np.float64)              # (4,4)
        K = stage3[f"K_{fi}"].astype(np.float64)                  # (3,3)
        cam = (w2c @ pts_h.T).T                                   # (N,4)
        z = cam[:, 2]
        front = z > 0.05
        if not front.any():
            continue
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        u = fx * cam[:, 0] / z + cx
        v = fy * cam[:, 1] / z + cy
        ui = np.floor(u).astype(np.int64)
        vi = np.floor(v).astype(np.int64)
        inb = front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if not inb.any():
            continue
        lab = labels[k][vi[inb], ui[inb]]
        keep = lab >= 0
        if not keep.any():
            continue
        pidx = np.nonzero(inb)[0][keep]
        lab = lab[keep].astype(np.int64)
        # 按类别分桶 bincount + 仅非零散射累加（避免 np.add.at 慢 & 400M 密集分配）
        for c in np.unique(lab):
            pc = pidx[lab == c]
            cnt = np.bincount(pc)                       # 稀疏长度 = pc.max()+1
            idx = np.nonzero(cnt)[0]                    # 本帧该类别命中的点索引
            votes[c, idx] += cnt[idx].astype(np.uint16)
        n_seen += 1
        if n_seen % 25 == 0:
            print(f"    [{n_seen}/{len(kf_indices)}] frames, "
                  f"{time.time() - t0:.0f}s elapsed", flush=True)

    total = votes.sum(axis=0)
    sid = votes.argmax(axis=0).astype(np.int32)
    sid[total < min_votes] = -1
    return sid, prompts, n_seen


def write_ply_binary(path, pts, rgb, label=None):
    n = len(pts)
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if label is not None:
        dtype.append(("label", "<i4"))
    arr = np.empty(n, dtype=dtype)
    arr["x"] = pts[:, 0].astype(np.float32)
    arr["y"] = pts[:, 1].astype(np.float32)
    arr["z"] = pts[:, 2].astype(np.float32)
    arr["red"] = rgb[:, 0].astype(np.uint8)
    arr["green"] = rgb[:, 1].astype(np.uint8)
    arr["blue"] = rgb[:, 2].astype(np.uint8)
    if label is not None:
        arr["label"] = label.astype(np.int32)
    with open(path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {n}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if label is not None:
            f.write(b"property int label\n")
        f.write(b"end_header\n")
        arr.tofile(f)
    print(f"  -> {path} ({os.path.getsize(path)/1e6:.0f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--min_votes", type=int, default=2)
    args = ap.parse_args()

    for scene in args.scenes:
        base = f"{BASE}/{scene}"
        stage3_path = f"{base}/_stage3_input_after.npz"
        semantic_path = f"{base}/kf_semantic.npz"
        out_dir = f"{base}/semantic_cloud_10m"
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n[{scene}] loading pts_gs ...")
        stage3, pts, rgb = load_stage3(stage3_path)
        print(f"[{scene}] {len(pts):,} points (full density)")

        sid, prompts, n_seen = vote_labels(pts, stage3, semantic_path, args.min_votes)
        n_lab = int((sid >= 0).sum())
        print(f"[{scene}] voted over {n_seen} keyframes; "
              f"{n_lab:,}/{len(pts):,} labeled ({n_lab/len(pts)*100:.1f}%)")

        # RGB 点云
        rgb255 = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        write_ply_binary(f"{out_dir}/rgb_cloud.ply", pts, rgb255)

        # 语义点云：类别色 + label 字段
        cols = np.zeros((len(pts), 3), np.uint8)
        labeled = sid >= 0
        cols[labeled] = (PALETTE[sid[labeled]][:, :3] * 255).astype(np.uint8)
        cols[~labeled] = np.clip(rgb255[~labeled], 0, 255)
        write_ply_binary(f"{out_dir}/semantic_cloud.ply", pts, cols, sid)

        # 紧凑 npz 供 Jupyter 读
        np.savez_compressed(f"{out_dir}/semantic_cloud.npz",
                            pts=pts.astype(np.float32),
                            label=sid.astype(np.int16),
                            prompts=np.array(prompts, dtype=object))
        print(f"  -> {out_dir}/semantic_cloud.npz "
              f"({os.path.getsize(out_dir + '/semantic_cloud.npz')/1e6:.0f} MB)")
        print(f"[{scene}] DONE")

    print("\n[all] DONE")


if __name__ == "__main__":
    main()
