#!/usr/bin/env python3
"""Stage A probe: does "off-surface distance to the fused cloud" identify floaters?

Hypothesis (超级创新点 -> reconstruction): the fused world-frame cloud (pts_gs,
globally consistent, Delta~8cm) IS the true surface.  Floaters that overfit the
train views during unregularized final_refine drift OFF this surface, while real
surface Gaussians stay ON it.  So a floater = HIGH opacity + LARGE distance to the
nearest fused-cloud point (and likely needle-like anisotropy).

This probes that signature on the BASELINE collapsed model (NVS 9.21 dB) and prints
the distance distribution + how opacity/anisotropy vary across distance bins.

CPU-only (numpy + scipy + matplotlib). No GPU, no v3dsam import.
"""
import os, argparse
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
                     "font.size": 9, "savefig.dpi": 160, "savefig.bbox": "tight"})


def read_gaussian_ply(path):
    """Parse a 3DGS-style binary_little_endian PLY -> dict of arrays."""
    raw = open(path, "rb").read()
    off = raw.find(b"end_header\n") + len(b"end_header\n")
    hdr = raw[:off].decode("ascii", "ignore")
    lines = hdr.strip().split("\n")
    count = int([l for l in lines if l.startswith("element vertex")][0].split()[-1])
    props = [l.split()[-1] for l in lines if l.startswith("property")]
    data = np.frombuffer(raw, dtype=np.float32, count=count * len(props), offset=off)
    data = data.reshape(count, len(props))
    return {p: data[:, i] for i, p in enumerate(props)}, count


def read_cloud_ascii(path, skiprows=11):
    """Read ascii PLY -> xyz (N,3), iid (N,) using only needed columns."""
    a = np.loadtxt(path, skiprows=skiprows, usecols=(0, 1, 2, 6), dtype=np.float64)
    return a[:, :3], a[:, 3].astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaussians", required=True, help="gaussians_final.ply (baseline)")
    ap.add_argument("--cloud", required=True, help="instance_cloud.ply (fused world-frame cloud)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    g, n_g = read_gaussian_ply(args.gaussians)
    print(f"[probe] {n_g:,} Gaussians loaded")
    xyz = np.stack([g["x"], g["y"], g["z"]], axis=1).astype(np.float64)
    op_logit = g["opacity"].astype(np.float64)
    opacity = 1.0 / (1.0 + np.exp(-op_logit))                 # sigmoid(logit) -> (0,1)
    scale = np.exp(np.stack([g["scale_0"], g["scale_1"], g["scale_2"]], axis=1))
    aniso = scale.max(1) / (scale.min(1) + 1e-9)              # anisotropy (needle-ness)

    cloud_xyz, cloud_iid = read_cloud_ascii(args.cloud)
    print(f"[probe] {len(cloud_xyz):,} fused-cloud points loaded")

    tree = cKDTree(cloud_xyz)
    dist, nn = tree.query(xyz, k=1, workers=-1)
    nn_iid = cloud_iid[nn]
    print(f"[probe] nearest-neighbour distance computed")

    bins = [(0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, np.inf)]
    labels = ["<2cm", "2-5cm", "5-10cm", "10-20cm", "20-40cm", ">40cm"]
    print(f"\n{'='*78}\n  {'bin':<10} {'n':>8} {'%':>6} {'mean_opac':>10} {'mean_aniso':>11} {'med_aniso':>10}\n{'='*78}")
    rows = []
    for (lo, hi), lab in zip(bins, labels):
        m = (dist >= lo) & (dist < hi)
        if m.sum() == 0:
            rows.append(dict(bin=lab, n=0)); continue
        rows.append(dict(bin=lab, n=int(m.sum()), pct=float(m.mean() * 100),
                         mean_opacity=float(opacity[m].mean()),
                         mean_aniso=float(aniso[m].mean()),
                         median_aniso=float(np.median(aniso[m]))))
        print(f"  {lab:<10} {m.sum():>8} {m.mean()*100:>5.1f}% {opacity[m].mean():>10.3f} "
              f"{aniso[m].mean():>11.2f} {np.median(aniso[m]):>10.2f}")

    # floaters = far + high opacity + high anisotropy
    far = dist > 0.10
    print(f"\n  floaters (dist>10cm): {far.sum():,} = {far.mean()*100:.1f}% of Gaussians")
    print(f"    of those, mean opacity={opacity[far].mean():.3f}, mean aniso={aniso[far].mean():.2f}")
    near = dist < 0.02
    print(f"  surfaces (dist<2cm): {near.sum():,} = {near.mean()*100:.1f}%")

    # ── figure ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.hist(np.clip(dist, 0, 0.5), bins=100, color="steelblue", alpha=0.85)
    ax.set_xlabel("distance to fused cloud (m)")
    ax.set_ylabel("# Gaussians")
    ax.set_title("(a) off-surface distance (baseline, NVS 9.2dB)")
    ax.axvline(0.02, color="g", ls="--", lw=1, label="2cm")
    ax.axvline(0.10, color="r", ls="--", lw=1, label="10cm")
    ax.legend()

    ax = axes[1]
    cnt = [r.get("n", 0) for r in rows]
    op = [r.get("mean_opacity", 0) for r in rows]
    an = [r.get("mean_aniso", 0) for r in rows]
    x = np.arange(len(labels))
    ax2 = ax.twinx()
    ax.bar(x, op, width=0.6, color="steelblue", alpha=0.7, label="mean opacity")
    ax2.plot(x, an, "o-", color="darkorange", lw=1.5, label="mean anisotropy")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("mean opacity (sigmoid)"); ax2.set_ylabel("mean anisotropy (needle-ness)")
    ax.set_title("(b) floater signature: far Gaussians = high opacity + needle-like")
    ax.set_ylim(0, 1)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)
    fig.suptitle("Floater probe: geometric-anchored off-surface distance vs opacity/anisotropy", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "floater_probe.png"))

    import json
    out = {"n_gaussians": n_g, "n_cloud": int(len(cloud_xyz)),
           "dist_p50": float(np.percentile(dist, 50)), "dist_p90": float(np.percentile(dist, 90)),
           "dist_p99": float(np.percentile(dist, 99)), "dist_max": float(dist.max()),
           "n_far_10cm": int(far.sum()), "pct_far_10cm": float(far.mean() * 100),
           "bins": rows}
    with open(os.path.join(args.out_dir, "floater_probe.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[probe] DONE -> {args.out_dir}/floater_probe.png + floater_probe.json")


if __name__ == "__main__":
    main()
