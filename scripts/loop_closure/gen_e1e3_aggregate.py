import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

BASE = "/root/autodl-tmp"
SEQ = ["room", "360", "desk2", "floor", "plant", "rpy", "teddy"]
TAU = 0.10          # paper's c-score threshold
ALPHA_THR = -0.95   # VGGT-SLAM native alpha-match rejection threshold

# ---- GT labels ----
gt = json.load(open(f"{BASE}/gt_loop_pose.json"))
label = {}
for e in gt:
    label[(e["seq"], e["loop"])] = (e["paper"] == "假")

# ---- gather per-candidate scores ----
rows = []
for s in SEQ:
    alpha = json.load(open(f"{BASE}/semantic_out/{s}/loop_log_alpha.json"))["loops"]
    cons = json.load(open(f"{BASE}/semantic_out/{s}/loop_log_3d.json"))["loops"]
    geo = json.load(open(f"{BASE}/semantic_out/{s}/loop_log_geo.json"))["loops"]
    obj = json.load(open(f"{BASE}/semantic_out/{s}/loop_log_obj.json"))["loops"]
    base = json.load(open(f"{BASE}/semantic_out/{s}/loop_log.json"))["loops"]
    for k in range(len(alpha)):
        rows.append({
            "seq": s, "loop": k,
            "is_false": label[(s, k)],
            "salad": alpha[k]["score"],                    # appearance L2 (SALAD distance)
            "alpha": alpha[k]["alpha"],                    # image_match_ratio (attention)
            "cons": cons[k]["cons"],                       # ours: semantic-geometric
            "geo": geo[k]["geo"],                          # pure geometry depth
            "jaccard": base[k]["overlap"],                 # category Jaccard
            "obj": obj[k]["match_ratio"] if k < len(obj) else None,  # object-level
        })

assert len(rows) == 15, f"rows={len(rows)}"
y = np.array([not r["is_false"] for r in rows])   # True = true loop

def metrics(pred_accept):
    pred_accept = np.asarray(pred_accept, dtype=bool)
    TP = int(np.sum(pred_accept & y));  FP = int(np.sum(pred_accept & ~y))
    TN = int(np.sum(~pred_accept & ~y)); FN = int(np.sum(~pred_accept & y))
    P = TP / (TP + FP) if (TP + FP) else 0.0
    R = TP / (TP + FN) if (TP + FN) else 0.0
    return dict(TP=TP, FP=FP, TN=TN, FN=FN, P=P, R=R)

def best_by_threshold(scores, higher_accept):
    best = None
    for t in np.unique(scores):
        acc = (scores >= t) if higher_accept else (scores <= t)
        m = metrics(acc)
        f1 = 2 * m["P"] * m["R"] / (m["P"] + m["R"]) if (m["P"] + m["R"]) else 0
        if best is None or f1 > best["F1"]:
            best = {**m, "thr": float(t), "F1": f1}
    return best

print("=" * 90)
print("Per-candidate scores (15 total: 4 false / 11 true)")
print("=" * 90)
print(f"{'seq':7}{'k':3} {'false':5} {'salad':7} {'alpha':7} {'cons':7} {'geo':7} {'jacc':7} {'obj':7}")
for r in rows:
    o = "-" if r["obj"] is None else f"{r['obj']:.3f}"
    print(f"{r['seq']:7}{r['loop']:3} {str(r['is_false']):5} {r['salad']:7.3f} {r['alpha']:7.3f} "
          f"{r['cons']:7.3f} {r['geo']:7.3f} {r['jaccard']:7.3f} {o:7}")

print()
print("=" * 90)
print(f"E1 HEAD-TO-HEAD: alpha-match (>= {ALPHA_THR}) vs ours (cons >= {TAU})")
print("=" * 90)
am = metrics(np.array([r["alpha"] for r in rows]) >= ALPHA_THR)
ours = metrics(np.array([r["cons"] for r in rows]) >= TAU)
hdr = f"{'method':26s}{'TP':>4}{'TN':>4}{'FP':>4}{'FN':>4}{'P':>8}{'R':>8}"
print(hdr)
print(f"{'alpha-match (native)':26s}{am['TP']:4d}{am['TN']:4d}{am['FP']:4d}{am['FN']:4d}{am['P']:8.3f}{am['R']:8.3f}")
print(f"{'ours (semantic-geometric)':26s}{ours['TP']:4d}{ours['TN']:4d}{ours['FP']:4d}{ours['FN']:4d}{ours['P']:8.3f}{ours['R']:8.3f}")

print()
print("=" * 90)
print("Ablation: best-threshold per discriminator (max F1 on the 15 candidates)")
print("=" * 90)
ablation = {}
for name, key, higher in [
    ("salad L2 (appearance)", "salad", False),
    ("alpha-match", "alpha", True),
    ("category Jaccard", "jaccard", True),
    ("pure geometry", "geo", True),
    ("object-level", "obj", True),
    ("ours (semantic-geom)", "cons", True),
]:
    sc = np.array([r[key] for r in rows])
    b = best_by_threshold(sc, higher)
    ablation[name] = b
    print(f"{name:26s} thr={b['thr']:7.3f}  TP={b['TP']:2d} TN={b['TN']:2d} FP={b['FP']:2d} "
          f"FN={b['FN']:2d}  P={b['P']:.3f} R={b['R']:.3f}  F1={b['F1']:.3f}")

summary = {
    "n_total": 15, "n_false": 4, "n_true": 11,
    "head_to_head": {"alpha_match_thr": ALPHA_THR, "alpha_match": am,
                     "ours_thr": TAU, "ours": ours},
    "ablation": ablation,
    "per_candidate": rows,
}
json.dump(summary, open(f"{BASE}/semantic_out/e1_head2head.json", "w"), indent=2, ensure_ascii=False)
print("\n[saved] semantic_out/e1_head2head.json")

# ---- E3: noise robustness (P/R at tau=0.10 vs SAM label-noise rate) ----
print()
print("=" * 90)
print(f"E3: SAM mask label-noise robustness (P/R at tau={TAU} vs noise rate, mean over seeds)")
print("=" * 90)
noise_rows = []
for s in SEQ:
    d = json.load(open(f"{BASE}/semantic_out/{s}/loop_log_noise.json"))["loops"]
    noise_rows.extend(d)
assert len(noise_rows) == 15, f"noise rows={len(noise_rows)}"

# use the exact keys stored in the noise dict (e.g. "0.0", "0.05", "0.1", ...)
levels = sorted(set().union(*[set(r["noise"].keys()) for r in noise_rows]),
                key=lambda k: float(k))

per_level = {}
for p in levels:
    acc = [float(np.mean(r["noise"][p])) >= TAU for r in noise_rows]
    m = metrics(np.array(acc))
    per_level[p] = m
    print(f"noise={float(p):4.2f}  TP={m['TP']:2d} TN={m['TN']:2d} FP={m['FP']:2d} FN={m['FN']:2d}  "
          f"P={m['P']:.3f} R={m['R']:.3f}")

json.dump({"tau": TAU, "per_level": {str(k): v for k, v in per_level.items()}},
          open(f"{BASE}/semantic_out/e3_noise.json", "w"), indent=2, ensure_ascii=False)
print("[saved] semantic_out/e3_noise.json")

# ---- E3 figure ----
fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
xs = [float(k) for k in levels]
P = [per_level[k]["P"] for k in levels]
R = [per_level[k]["R"] for k in levels]
ax.plot(xs, P, "-o", color="#1f77b4", lw=2, label="Precision")
ax.plot(xs, R, "-s", color="#d62728", lw=2, label="Recall")
ax.axvline(0.20, color="gray", ls="--", lw=1, alpha=0.6)
ax.set_xlabel("SAM mask label-noise rate")
ax.set_ylabel("Score")
ax.set_ylim(-0.02, 1.02)
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.grid(alpha=0.3, lw=0.5)
ax.legend(frameon=False, loc="center right")
fig.tight_layout()
import os
os.makedirs(f"{BASE}/figures_sci", exist_ok=True)
fig.savefig(f"{BASE}/figures_sci/figS2_noise_robustness.png")
fig.savefig(f"{BASE}/figures_sci/figS2_noise_robustness.svg")
plt.close(fig)
print("[saved] figures_sci/figS2_noise_robustness.png")
print("AGGREGATE DONE")
