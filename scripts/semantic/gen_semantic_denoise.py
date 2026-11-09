#!/usr/bin/env python3
"""Quantify multi-view semantic denoising (超级创新点 pilot 3).

Question: how UNRELIABLE is a single-view open-vocabulary (SAM3) label, and how
much does geometric-anchored multi-view majority voting stabilize it?

We already showed (pilot 2) that two disjoint 93-view consensus maps disagree on
73% of instances.  Here we measure the POINT level:

  for each world-frame point, we have its vote histogram over all views that see
  it (reuse the exact projection used by vote_labels).  The mode = consensus label.

  1. single-view reliability = mode_count / total_views  (if you look from ONE
     random view, how often does SAM3 agree with the point's consensus label?)
  2. voting curve: agreement of a majority vote over n views with the full
     consensus, for n = 1,3,5,10,20,50,100  (how few views suffice to denoise).

CPU-only (numpy + matplotlib). Outputs denoise.png + denoise.json.
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gen_semantic_pointcloud import load_stage3, subsample

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
                     "font.size": 9, "savefig.dpi": 160, "savefig.bbox": "tight"})


def accumulate_votes(pts, stage3, labels, kf_indices, prompts):
    """Per-point vote histogram over all keyframes. Returns (votes, total, n_seen)."""
    n_cats = len(prompts); H, W = labels.shape[1], labels.shape[2]; N = len(pts)
    pts_h = np.hstack([pts, np.ones((N, 1), dtype=np.float64)])
    votes = np.zeros((N, n_cats), dtype=np.int64)
    n_seen = 0
    for pos, fi in enumerate(kf_indices):
        fi = int(fi)
        if f"w2c_{fi}" not in stage3.files or f"K_{fi}" not in stage3.files:
            continue
        w2c = stage3[f"w2c_{fi}"].astype(np.float64); K = stage3[f"K_{fi}"].astype(np.float64)
        cam = (w2c @ pts_h.T).T; z = cam[:, 2]; front = z > 0.05
        if not front.any():
            continue
        Xc, Yc, Zc = cam[:, 0], cam[:, 1], cam[:, 2]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        u = fx * Xc / Zc + cx; v = fy * Yc / Zc + cy
        ui = np.floor(u).astype(np.int64); vi = np.floor(v).astype(np.int64)
        inb = front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if not inb.any():
            continue
        lab = labels[pos][vi[inb], ui[inb]]
        keep = lab >= 0
        if not keep.any():
            continue
        pidx = np.nonzero(inb)[0][keep]
        np.add.at(votes, (pidx, lab[keep]), 1)
        n_seen += 1
    total = votes.sum(1)
    return votes, total, n_seen


def voting_curve(votes, total, ns, rng_seed=0, n_sub=40000):
    """Agreement of an n-view majority vote with the full consensus (mode)."""
    labeled = total >= 2
    idx = np.nonzero(labeled)[0]
    if len(idx) > n_sub:
        idx = np.random.RandomState(rng_seed).choice(idx, n_sub, replace=False)
    V = votes[idx]; T = total[idx]
    mode = V.argmax(1)
    npts = len(idx)
    # precompute per-point categorical weights
    cum = np.cumsum(V, axis=1) / T[:, None]      # (npts, n_cats)
    rng = np.random.RandomState(rng_seed)
    curve = {}
    for n in ns:
        if n == 1:
            agree = (V.max(1) / T)               # single-view agreement = mode fraction
            curve[n] = float(np.mean(agree))
            continue
        n_trial = 3
        hits = np.zeros(npts)
        for _ in range(n_trial):
            r = rng.rand(npts, n)
            draw = (r[:, :, None] < cum[:, None, :]).argmax(2)   # (npts, n) labels
            mj = np.apply_along_axis(lambda row: np.bincount(row, minlength=V.shape[1]).argmax(), 1, draw)
            hits += (mj == mode)
        curve[n] = float(hits.mean() / n_trial)
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage3", required=True)
    ap.add_argument("--semantic", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_pts", type=int, default=500000)
    ap.add_argument("--min_votes", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    stage3, pts, rgb = load_stage3(args.stage3)
    pts, rgb, _ = subsample(pts, rgb, args.max_pts)
    s = np.load(args.semantic, allow_pickle=True)
    labels = s["labels"]; kf_indices = [int(x) for x in s["kf_indices"].tolist()]
    prompts = [str(p) for p in s["prompts"]]

    print(f"[denoise] accumulating votes over {len(kf_indices)} keyframes, {len(pts):,} points ...")
    votes, total, n_seen = accumulate_votes(pts, stage3, labels, kf_indices, prompts)
    labeled = total >= args.min_votes
    print(f"[denoise] {labeled.sum():,}/{len(pts):,} labeled (>= {args.min_votes} views)")

    mode_frac = votes.max(1) / np.maximum(total, 1)          # single-view reliability
    mf = mode_frac[labeled]
    print(f"\n  single-view reliability (fraction of views agreeing with consensus):")
    print(f"    mean={mf.mean():.3f}  median={np.median(mf):.3f}  "
          f"p10={np.percentile(mf,10):.3f}  p90={np.percentile(mf,90):.3f}")
    print(f"    single-view DISAGREEMENT (point-level, analog of the 73% flip) = {1-mf.mean():.3f}")

    # per-class stability
    sid = votes.argmax(1)
    cls_rows = []
    for c in np.unique(sid[labeled]):
        m = labeled & (sid == c)
        if m.sum() < 100:
            continue
        name = prompts[c] if c < len(prompts) else str(c)
        cls_rows.append((name, int(m.sum()), float(mode_frac[m].mean())))
    cls_rows.sort(key=lambda x: -x[1])

    ns = [1, 3, 5, 10, 20, 50, 100]
    curve = voting_curve(votes, total, ns)
    print("\n  n-view majority-vote agreement with full consensus:")
    for n in ns:
        print(f"    n={n:>3}: {curve[n]:.3f}")

    # ── figure ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.hist(mf, bins=40, range=(0.3, 1.0), color="steelblue", alpha=0.85)
    ax.axvline(mf.mean(), color="red", ls="--", lw=1.5, label=f"mean={mf.mean():.2f}")
    ax.set_xlabel("single-view agreement w/ consensus (mode fraction)")
    ax.set_ylabel("# points")
    ax.set_title("(a) single-view SAM3 reliability\n(geometric-anchored, per point)")
    ax.legend()

    ax = axes[1]
    ax.plot(ns, [curve[n] for n in ns], "o-", color="darkorange", lw=1.5)
    ax.set_xscale("log"); ax.set_xlabel("# views in majority vote")
    ax.set_ylabel("agreement with full consensus")
    ax.set_ylim(0.3, 1.02); ax.set_title("(b) multi-view voting denoises semantics")
    ax.grid(alpha=0.3)
    fig.suptitle("V3DSAM semantic denoising: open-vocabulary labels are view-dependent, voting stabilizes them",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "denoise.png"))

    import json
    out = {
        "n_points": len(pts), "n_labeled": int(labeled.sum()),
        "single_view_reliability_mean": float(mf.mean()),
        "single_view_reliability_median": float(np.median(mf)),
        "single_view_disagreement": float(1 - mf.mean()),
        "per_class": [{"class": n, "count": c, "reliability": r} for n, c, r in cls_rows],
        "voting_curve": curve,
    }
    with open(os.path.join(args.out_dir, "denoise.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[denoise] DONE -> {args.out_dir}/denoise.png + denoise.json")


if __name__ == "__main__":
    main()
