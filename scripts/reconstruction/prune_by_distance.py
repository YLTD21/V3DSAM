#!/usr/bin/env python3
"""Stage B: prune Gaussians whose center is farther than T from the fused
world-frame cloud (the geometric-anchor floater filter). Writes pruned PLYs
(binary little-endian, preserving the exact 3DGS property list) for NVS eval.

CPU-only. No v3dsam import.
"""
import os, argparse
import numpy as np
from scipy.spatial import cKDTree


def read_gaussian_ply(path):
    raw = open(path, "rb").read()
    off = raw.find(b"end_header\n") + len(b"end_header\n")
    hdr = raw[:off].decode("ascii", "ignore")
    lines = hdr.strip().split("\n")
    count = int([l for l in lines if l.startswith("element vertex")][0].split()[-1])
    prop_lines = [l for l in lines if l.startswith("property")]
    props = [l.split()[-1] for l in prop_lines]
    data = np.frombuffer(raw, dtype=np.float32, count=count * len(props), offset=off)
    data = data.reshape(count, len(props))
    return lines, prop_lines, props, data


def read_cloud(path):
    a = np.loadtxt(path, skiprows=11, usecols=(0, 1, 2), dtype=np.float64)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaussians", required=True)
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--thresholds", default="0.40,0.20,0.10,0.05")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    thresholds = [float(t) for t in args.thresholds.split(",")]

    lines, prop_lines, props, data = read_gaussian_ply(args.gaussians)
    xyz = np.stack([data[:, props.index("x")], data[:, props.index("y")],
                    data[:, props.index("z")]], axis=1).astype(np.float64)
    n = len(xyz)
    print(f"[prune] {n:,} Gaussians loaded, {len(props)} props")

    cloud = read_cloud(args.cloud)
    print(f"[prune] {len(cloud):,} cloud points loaded")
    tree = cKDTree(cloud)
    dist, _ = tree.query(xyz, k=1, workers=-1)

    for T in thresholds:
        keep = dist <= T
        tag = f"{T:.2f}".replace(".", "_")          # 0.40 -> "0_40"
        out = os.path.join(args.out_dir, f"pruned_{tag}.ply")
        # rebuild header with new count
        new_lines = []
        for l in lines:
            if l.startswith("element vertex"):
                new_lines.append(f"element vertex {int(keep.sum())}")
            else:
                new_lines.append(l)
        header = ("\n".join(new_lines) + "\n").encode("ascii")   # keep trailing newline
        body = data[keep].astype(np.float32).tobytes()
        with open(out, "wb") as f:
            f.write(header)
            f.write(body)
        print(f"[prune] T={T:.2f}m -> keep {int(keep.sum()):,}/{n:,} "
              f"({keep.mean()*100:.1f}%) -> {out}")

    print("[prune] DONE")


if __name__ == "__main__":
    main()
