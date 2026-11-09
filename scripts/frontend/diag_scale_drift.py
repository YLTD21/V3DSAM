#!/usr/bin/env python3
"""Diagnose per-submap depth scale drift in _stage3_input.npz.

Key idea: VGGT-SLAM processes submaps with O overlap frames.  The SAME physical
frame appears once in submap k (as its last O frames) and again in submap k+1
(as its first O frames), each predicted by a DIFFERENT VGGT batch -> different
scale.  The depth ratio between those two copies is the inter-submap scale
factor.  We also report per-frame median depth to see global drift, and check
whether w2c poses are smooth (globally consistent) or jumpy (raw per-submap).
"""
import os, sys, argparse, collections
import numpy as np

def med(d):
    d = d.astype(np.float32).reshape(-1)
    d = d[np.isfinite(d) & (d > 0.1)]
    return float(np.median(d)) if d.size else float('nan')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    args = ap.parse_args()
    d = np.load(args.npz, allow_pickle=True)

    M = sum(1 for k in d.keys() if k.startswith("K_"))
    print(f"M={M} frames")

    # per-frame median depth
    meds = np.zeros(M)
    names = []
    for i in range(M):
        meds[i] = med(d[f"depth_{i}"])
        names.append(os.path.basename(str(d[f"img_path_{i}"])))

    # group by basename -> find overlapping copies
    idx_by_name = collections.defaultdict(list)
    for i, n in enumerate(names):
        idx_by_name[n].append(i)

    dup = {n: v for n, v in idx_by_name.items() if len(v) > 1}
    print(f"duplicate (overlap) frames: {len(dup)} names")

    ratios = []
    print("\n=== overlap depth ratios (adjacent-submap scale) ===")
    print(f"{'name':<22} {'idx_a':>5} {'idx_b':>5} {'med_a':>8} {'med_b':>8} {'ratio b/a':>9}")
    for n, idxs in sorted(dup.items()):
        for a, b in zip(idxs[:-1], idxs[1:]):
            r = meds[b] / meds[a] if meds[a] > 0 and np.isfinite(meds[a]) else float('nan')
            ratios.append(r)
            print(f"{n:<22} {a:>5} {b:>5} {meds[a]:>8.3f} {meds[b]:>8.3f} {r:>9.3f}")

    ratios = np.array([r for r in ratios if np.isfinite(r)])
    if ratios.size:
        print(f"\noverlap depth ratio: n={ratios.size} mean={ratios.mean():.3f} "
              f"std={ratios.std():.3f} min={ratios.min():.3f} max={ratios.max():.3f}")
        # how many are far from 1.0?
        print(f"  |ratio-1|>0.05 : {(np.abs(ratios-1)>0.05).sum()}/{ratios.size}")
        print(f"  |ratio-1|>0.10 : {(np.abs(ratios-1)>0.10).sum()}/{ratios.size}")

    # per-frame median depth sequence (to eyeball global drift)
    print("\n=== median depth sequence (every 40th frame) ===")
    for i in range(0, M, 40):
        seg = meds[i:i+40]
        print(f"  [{i:4d}-{i+39:4d}] median={np.nanmedian(seg):7.3f}  "
              f"min={np.nanmin(seg):7.3f} max={np.nanmax(seg):7.3f}")

    # pose smoothness: translation magnitude between consecutive frames
    print("\n=== w2c pose translation (consecutive frames) ===")
    trans = []
    for i in range(M-1):
        t_a = d[f"w2c_{i}"][:3, 3]
        t_b = d[f"w2c_{i+1}"][:3, 3]
        trans.append(float(np.linalg.norm(t_b - t_a)))
    trans = np.array(trans)
    print(f"  consecutive translation: mean={trans.mean():.4f} median={np.median(trans):.4f} "
          f"max={trans.max():.4f}  (jumps at submap boundaries would show as spikes)")

if __name__ == "__main__":
    main()
