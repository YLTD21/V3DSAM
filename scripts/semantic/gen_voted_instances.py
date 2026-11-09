#!/usr/bin/env python3
"""全视角投票 → 固定实例的语义类别（用户想法：语义赋值到固定物体聚类上）。

对 instance_cloud.ply 里的每一个固定实例（世界帧聚类，ID 不变），把其成员点
在所有关键帧里的遮挡感知投票聚合 → voted_class + 置信度。这样每个物体只有
一个稳定类别，颜色(=类别)在所有视角下彻底一致。

复用 gen_object_ba.accumulate_votes（z-buffer 遮挡感知）。
输出 voted_instances.json: {id: {voted_class, confidence, n_obs, count}}。
"""
import os, argparse, json
import numpy as np
import pandas as pd
from gen_object_ba import accumulate_votes


def load_ply(path):
    df = pd.read_csv(path, skiprows=11, sep=r"\s+", header=None, usecols=[0, 1, 2, 6], engine="c")
    xyz = df[[0, 1, 2]].to_numpy(np.float64)
    iid = df[6].to_numpy(np.int64)
    return xyz, iid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage3", required=True)
    ap.add_argument("--semantic", required=True)
    ap.add_argument("--instance_cloud", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--cap_per_instance", type=int, default=5000)
    args = ap.parse_args()

    stage3 = np.load(args.stage3, allow_pickle=True)
    sem = np.load(args.semantic, allow_pickle=True)
    labels = sem["labels"]
    kf_indices = [int(x) for x in sem["kf_indices"].tolist()]
    prompts = [str(p) for p in sem["prompts"]]

    pts, iid = load_ply(args.instance_cloud)
    n_inst = int(iid.max()) + 1
    rng = np.random.RandomState(0)
    parts = []
    for h in range(n_inst):
        m = np.nonzero(iid == h)[0]
        if len(m) > args.cap_per_instance:
            m = m[rng.choice(len(m), args.cap_per_instance, replace=False)]
        parts.append(m)
    idx = np.concatenate(parts)
    pts_s, iid_s = pts[idx], iid[idx]
    print(f"[voted] {len(pts_s):,} points over {n_inst} instances")

    votes, total, n_seen = accumulate_votes(pts_s, stage3, labels, kf_indices)
    out = {"n_instances": n_inst, "prompts": prompts, "instances": []}
    for h in range(n_inst):
        m = iid_s == h
        V = votes[m].sum(0)
        s = int(V.sum())
        if s > 0:
            vc = int(V.argmax()); conf = float(V.max() / s); n_obs = s
        else:
            vc = -1; conf = 0.0; n_obs = 0
        out["instances"].append({"id": h, "voted_class": vc, "confidence": conf,
                                 "n_obs": n_obs, "count": int(m.sum())})

    confs = [o["confidence"] for o in out["instances"] if o["voted_class"] >= 0]
    print(f"[voted] instance-level voted confidence: mean={np.mean(confs):.3f} "
          f"median={np.median(confs):.3f}  n_valid={len(confs)}/{n_inst}")

    # per-category voted distribution (for debugging fragmentation)
    from collections import Counter
    c = Counter(o["voted_class"] for o in out["instances"] if o["voted_class"] >= 0)
    print("[voted] per-category instance count:")
    for vc, cnt in sorted(c.items(), key=lambda x: -x[1]):
        print(f"   {prompts[vc]:<14} {cnt}")

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[voted] wrote {args.out_json}")


if __name__ == "__main__":
    main()
