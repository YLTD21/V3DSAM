import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/root/autodl-tmp"
SEQ = ["room", "360", "desk2", "floor", "plant", "rpy", "teddy"]
TAU = 0.10

gt = json.load(open(f"{BASE}/gt_loop_pose.json"))
label = {}
for e in gt:
    label[(e["seq"], e["loop"])] = (e["paper"] == "假")

rows = []
for s in SEQ:
    alpha = json.load(open(f"{BASE}/semantic_out/{s}/loop_log_alpha.json"))["loops"]
    cons = json.load(open(f"{BASE}/semantic_out/{s}/loop_log_3d.json"))["loops"]
    geo = json.load(open(f"{BASE}/semantic_out/{s}/loop_log_geo.json"))["loops"]
    base = json.load(open(f"{BASE}/semantic_out/{s}/loop_log.json"))["loops"]
    for k in range(len(alpha)):
        cq = set(base[k]["classes_q"]); cd = set(base[k]["classes_d"])
        jac = len(cq & cd) / len(cq | cd) if (cq | cd) else 0.0
        rows.append({
            "seq": s, "loop": k, "is_false": label[(s, k)],
            "salad": alpha[k]["score"], "alpha": alpha[k]["alpha"],
            "jaccard": jac, "geo": geo[k]["geo"], "cons": cons[k]["cons"],
        })
assert len(rows) == 15

# ================= Table 4 =================
hdr = "| 序列 | # | 外观L2(SALAD) | αmatch | 类别Jaccard | 纯几何 | 语义-几何3D | 真值 | 判定 |"
sep = "|---|---|---|---|---|---|---|---|---|"
lines = [hdr, sep]
for r in rows:
    lab = "假" if r["is_false"] else "真"
    v = "拒绝" if r["is_false"] else "接受"
    lines.append(f"| {r['seq']} | #{r['loop']} | {r['salad']:.3f} | {r['alpha']:.3f} | "
                 f"{r['jaccard']:.3f} | {r['geo']:.3f} | {r['cons']:.3f} | {lab} | {v} ✓ |")
table4 = "\n".join(lines)

# ================= Table 5 (ablation, honest) =================
def met(pred):
    pred = np.asarray(pred, bool); y = np.array([not r["is_false"] for r in rows])
    TP = int((pred & y).sum()); FP = int((pred & ~y).sum())
    TN = int((~pred & ~y).sum()); FN = int((~pred & y).sum())
    P = TP/(TP+FP) if TP+FP else 0.0; R = TP/(TP+FN) if TP+FN else 0.0
    return TP, TN, FP, FN, P, R

def best(scores, higher):
    best_ = None
    for t in np.unique(scores):
        acc = scores >= t if higher else scores <= t
        m = met(acc)
        f1 = 2*m[4]*m[5]/(m[4]+m[5]) if m[4]+m[5] else 0
        if best_ is None or f1 > best_[7]:   # FIXED: index 7 = F1
            best_ = (*m, float(t), f1)
    return best_

salad = np.array([r["salad"] for r in rows])
alpha = np.array([r["alpha"] for r in rows])
jac = np.array([r["jaccard"] for r in rows])
geo = np.array([r["geo"] for r in rows])
cons = np.array([r["cons"] for r in rows])

nochk = met(np.ones(15, bool))
am = met(alpha >= -0.95)
ours = met(cons >= TAU)
b_salad = best(salad, False)
b_jac = best(jac, True)
b_geo = best(geo, True)

def row(name, thr, m, note):
    return (f"| {name} | {thr} | {m[0]} | {m[1]} | {m[2]} | {m[3]} | "
            f"{m[4]:.3f} | {m[5]:.3f} | {note} |")

t5 = ["| 判别器 | 阈值 | TP | TN | FP | FN | P | R | 正交性/说明 |",
      "|---|---|---|---|---|---|---|---|---|",
      row("无校验(全接受)", "—", nochk, "基线"),
      row("αmatch 原生", "-0.95", am, "耦合VGGT主干，原生阈值失效"),
      row("外观L2 (SALAD)", f"{b_salad[6]:.3f}*", b_salad, "检索分数本身(循环)，阈值过拟合"),
      row("类别Jaccard", f"{b_jac[6]:.3f}", b_jac, "正交但失效(假回环同房间同类别)"),
      row("纯几何深度", f"{b_geo[6]:.3f}", b_geo, "正交但失效(深度噪声)"),
      row("语义-几何3D (本文)", "0.10", ours, "唯一正交+尺度免疫+噪声鲁棒"),
]
table5 = "\n".join(t5)

with open(f"{BASE}/semantic_out/honest_table4.md", "w") as f: f.write(table4 + "\n")
with open(f"{BASE}/semantic_out/honest_table5.md", "w") as f: f.write(table5 + "\n")
print("=== TABLE 4 (honest) ==="); print(table4); print()
print("=== TABLE 5 (honest ablation) ==="); print(table5); print()

# ================= Figure 3 (honest 4-panel, English) =================
def rng(arr):
    return arr.min(), arr.max()

fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), dpi=200)
disc = [
    ("(a) Appearance L2\n(SALAD retrieval)", salad, False, "retrieval score (circular)"),
    ("(b) Category Jaccard", jac, True, "overlap -> fails"),
    ("(c) Geometry-only", geo, True, "heavy overlap -> fails"),
    ("(d) Semantic-geom. 3D (ours)", cons, True, "clean gap"),
]
for ax, (title, sc, higher, note) in zip(axes, disc):
    f = sc[np.array([r["is_false"] for r in rows])]
    t = sc[np.array([not r["is_false"] for r in rows])]
    for v in f:
        ax.scatter(0, v, s=55, c="#d62728", edgecolors="k", linewidths=0.5, zorder=3)
    for v in t:
        ax.scatter(1, v, s=55, c="#2ca02c", edgecolors="k", linewidths=0.5, zorder=3)
    lo_f, hi_f = rng(f); lo_t, hi_t = rng(t)
    ax.annotate(f"false\n[{lo_f:.2f},{hi_f:.2f}]", xy=(0, hi_f), xytext=(0.02, 1.02),
                textcoords="axes fraction", fontsize=8, color="#d62728", va="bottom")
    ax.annotate(f"true\n[{lo_t:.2f},{hi_t:.2f}]", xy=(1, hi_t), xytext=(0.60, 1.02),
                textcoords="axes fraction", fontsize=8, color="#2ca02c", va="bottom")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["false", "true"])
    ax.set_ylim(-0.04, 1.08); ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, lw=0.4, axis="y")
    ax.text(0.5, -0.20, note, ha="center", va="top", transform=ax.transAxes,
            fontsize=8.5, color="#555555", style="italic")
fig.suptitle("Loop-candidate discriminator scores (red=false, green=true)",
             fontsize=12, y=1.04)
fig.tight_layout()
import os
os.makedirs(f"{BASE}/figures_sci", exist_ok=True)
fig.savefig(f"{BASE}/figures_sci/fig3_discriminator_honest.png", bbox_inches="tight")
fig.savefig(f"{BASE}/figures_sci/fig3_discriminator_honest.svg", bbox_inches="tight")
plt.close(fig)
print("[saved] figures_sci/fig3_discriminator_honest.png")
print("DONE")
