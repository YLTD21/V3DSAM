#!/usr/bin/env python3
"""Generate paper fig10 (geom-anchor schematic) + fig11 (re-ID recall bar chart).

fig11 uses the exact numbers from id_propagate.json, BOTH scenes:
  fr1/room: geometry-first 89.5% (68/76) vs class-filtered 59.2% (45/76), 23 flips (30.3%).
  fr1/360:  geometry-first 86.6% (58/67) vs class-filtered 52.2% (35/67), 23 flips (34.3%).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sci_style import apply_publication_style, PALETTE, add_panel_label, finalize_figure

apply_publication_style(font_size=8, axes_linewidth=1)

FIG = "E:/0722v3dsam/paper/figures"

# ─────────────────────────────────────────────────────────────── fig10 ──
def fig10():
    import os, json
    # real matched-pair centroid displacement (dmin, cm) from id_propagate.json
    dmins = {}
    for s in ["room", "360"]:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fig10_data", f"id_prop_{s}.json")
        try:
            st = json.load(open(p))["stats"]
            dmins[s] = np.array([x["dmin"] * 100.0
                                 for x in st if x["outcome"].startswith("recovered")])
        except Exception:
            dmins[s] = None

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    # Left: real matched-pair centroid displacement histogram (ours, SL(4) global)
    ax = axes[0]
    bins = np.arange(0, 31, 3)
    for s, c, lab in [("room", PALETTE["blue_main"], "fr1/room"),
                      ("360", PALETTE["teal"], "fr1/360")]:
        if dmins[s] is None:
            continue
        ax.hist(dmins[s], bins=bins, color=c, alpha=0.62, label=lab, edgecolor="white", lw=0.6)
    # competitor threshold (30 cm) — at/just beyond our distribution tail
    ax.axvline(30, color=PALETTE["red_strong"], ls="--", lw=1.6)
    ax.text(30.6, ax.get_ylim()[1] * 0.0, "", fontsize=1)  # noop to keep axis
    ax.annotate("competitor\n30 cm threshold", xy=(30, 0.0), xytext=(22, 12),
                fontsize=7.5, color=PALETTE["red_strong"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["red_strong"], lw=1.0))
    # median markers
    ax.axvline(6.9, color=PALETTE["blue_main"], lw=1.0, alpha=0.7)
    ax.axvline(6.0, color=PALETTE["teal"], lw=1.0, alpha=0.7)
    ax.set_xlabel("matched-pair centroid displacement (cm)")
    ax.set_ylabel("count")
    ax.set_title("Ours: SL(4) globally-consistent\n(real displacement, median 6.9 / 6.0 cm)",
                 fontsize=9)
    ax.set_xlim(0, 33)
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    ax.annotate("median\n6.9 cm", xy=(6.9, 0), xytext=(2, 16), fontsize=7,
                color=PALETTE["blue_main"], va="top")
    add_panel_label(ax, "a")

    # Right: rolling Sim3 (competitor) — identity drifts as blocks accumulate
    ax = axes[1]
    rng = np.random.RandomState(1)
    ax.set_title("Competitor (Dinya): rolling Sim3\n(identity drifts, 30 cm matching threshold)",
                 fontsize=9)
    chairs = [(0.0, 0.0), (0.44, 0.0)]  # two chairs 0.44 m apart
    drift = np.array([0.0, 0.05, 0.12, 0.20, 0.30])
    for (cx, cy) in chairs:
        xs = cx + drift
        ys = cy + np.linspace(0, 0.05, 5)
        ax.plot(xs, ys, "-", color=PALETTE["neutral_mid"], lw=1.5, alpha=0.8)
        for i in range(len(xs)):
            ax.scatter([xs[i]], [ys[i]], s=30, c=PALETTE["teal"], alpha=0.25 + 0.18 * i, zorder=3)
    ax.add_patch(plt.Circle((drift[-1], 0.05), 0.30, fill=False, ec=PALETTE["red_strong"], lw=2, ls="--"))
    ax.text(drift[-1], 0.05 + 0.34, "30 cm", ha="center", fontsize=9, color=PALETTE["red_strong"])
    ax.set_xlim(-0.3, 1.0); ax.set_ylim(-0.35, 0.6)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(0.5, -0.28, "centroid accumulates drift -> 30 cm threshold needed to re-match",
            ha="center", fontsize=8, color=PALETTE["neutral_dark"])
    add_panel_label(ax, "b")

    finalize_figure(fig, f"{FIG}/fig10_geom_anchor", dpi=200)
    print("fig10 done")


# ─────────────────────────────────────────────────────────────── fig11 ──
def fig11():
    # exact numbers from id_propagate.json, BOTH scenes
    scenes = ["fr1/room", "fr1/360"]
    geom = [89.5, 86.6]
    classf = [59.2, 52.2]
    n_flip = [23, 23]
    n_anchor = [76, 67]
    # 360 flip source-class distribution (voted/anchor class of recovered_flip)
    flip_src_360 = {"floor": 7, "desk": 3, "window": 2, "table": 2, "picture": 2,
                    "television": 2, "wall": 1, "lamp": 1, "bookshelf": 1,
                    "box": 1, "ceiling": 1}

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw={"width_ratios": [1.0, 1.1]})

    # Left: grouped bar — geometry-first vs class-filtered, per scene
    ax = axes[0]
    x = np.arange(len(scenes))
    w = 0.36
    b1 = ax.bar(x - w / 2, geom, w, color=PALETTE["blue_main"], alpha=0.92, label="geometry-first")
    b2 = ax.bar(x + w / 2, classf, w, color=PALETTE["neutral_dark"], alpha=0.92, label="class-filtered")
    for bars, vals in [(b1, geom), (b2, classf)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%",
                    ha="center", fontsize=12, fontweight="bold")
    for i in range(len(scenes)):
        ax.annotate("", xy=(x[i] + w / 2, classf[i]), xytext=(x[i] + w / 2, geom[i]),
                    arrowprops=dict(arrowstyle="<->", color=PALETTE["neutral_dark"], lw=1.2))
        ax.text(x[i] + w / 2 + 0.06, (geom[i] + classf[i]) / 2,
                f"+{geom[i] - classf[i]:.1f}\n({n_flip[i]}/{n_anchor[i]} flip)",
                fontsize=8, color=PALETTE["neutral_dark"], va="center")
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, fontsize=10)
    ax.set_ylabel("cross-view re-ID recall (%)", fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_title("Re-ID recall: geometry-first vs class-filtered", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    add_panel_label(ax, "a")

    # Right: flip source-class distribution (fr1/360, the demo scene)
    ax = axes[1]
    names = list(flip_src_360.keys())
    counts = [flip_src_360[n] for n in names]
    order = np.argsort(counts)
    names = [names[i] for i in order]
    counts = [counts[i] for i in order]
    ax.barh(names, counts, color=PALETTE["violet"], alpha=0.85)
    for i, c in enumerate(counts):
        ax.text(c + 0.12, i, str(c), va="center", fontsize=9)
    ax.set_xlabel("flipped objects", fontsize=10)
    ax.set_title("Semantic-label flips (fr1/360, 23 objects)\nby voted (anchor) class", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(counts) + 1.2)
    add_panel_label(ax, "b")

    fig.suptitle("Identity must be anchored to geometry, not semantic voting", fontsize=12,
                 fontweight="bold", y=1.02)
    finalize_figure(fig, f"{FIG}/fig11_reid_recall", dpi=200)
    print("fig11 done")


if __name__ == "__main__":
    fig10()
    fig11()
    print("ALL DONE")
