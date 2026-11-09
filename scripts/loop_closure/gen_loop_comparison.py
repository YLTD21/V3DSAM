#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the loop-discriminator comparison figure for fr1/room.

4 rows (discriminators) x 3 columns (room SALAD loop candidates).
Each cell = [query frame] --colored bar-- [detected frame].
Green = judged TRUE loop, Red = judged FALSE loop.
Also packages images into 回环对比图片/ with 5 sub-folders.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

FONT = None
for _fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc",
            r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc"]:
    if os.path.exists(_fp):
        try:
            FONT = ImageFont.truetype(_fp, 22)
            break
        except Exception:
            pass

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = r"E:\0722v3dsam\paper"
QD = os.path.join(HERE, "figures", "loop_figures", "semantic_out", "room", "loop_figures")
OUTDIR = os.path.join(HERE, "回环对比图片")

GREEN = (44, 160, 44)    # #2ca02c  true
RED = (214, 39, 40)      # #d62728  false

# ---- honest data: (id, is_false, geo, jaccard, alpha, ours) ----
CANDS = [
    ("room#0", True,  "loop0_query.png", "loop0_detected.png", 0.205, 0.640, 0.552, 0.078),
    ("room#1", False, "loop1_query.png", "loop1_detected.png", 0.660, 0.895, 1.007, 0.152),
    ("room#2", False, "loop2_query.png", "loop2_detected.png", 0.003, 1.000, 1.041, 0.481),
]
DISCS = [
    ("纯几何深度匹配", "geo",    0.003, 4),
    ("类别Jaccard重叠", "jac",   0.500, 5),
    ("VGGT αmatch 官方", "alpha", -0.95, 6),
    ("我们方法 语义-几何3D", "ours", 0.10, 7),
]
KEYIDX = {"geo": 4, "jac": 5, "alpha": 6, "ours": 7}


def resize_h(im, h):
    w = int(im.width * h / im.height)
    return im.resize((w, h), Image.LANCZOS)


def add_border(im, color, b):
    W, H = im.width + 2 * b, im.height + 2 * b
    out = Image.new("RGB", (W, H), color)
    out.paste(im, (b, b))
    return out


def compose_pair(q_path, d_path, color, bar_w=42, border=13, img_h=190):
    """query --bar-- detected, both frames bordered with `color`."""
    q = resize_h(Image.open(q_path).convert("RGB"), img_h)
    d = resize_h(Image.open(d_path).convert("RGB"), img_h)
    q = add_border(q, color, border)
    d = add_border(d, color, border)
    bar = Image.new("RGB", (bar_w, q.height), color)
    strip = Image.new("RGB", (q.width + bar.width + d.width, q.height), (255, 255, 255))
    strip.paste(q, (0, 0))
    strip.paste(bar, (q.width, 0))
    strip.paste(d, (q.width + bar.width, 0))
    return strip


def verdict(score, thr):
    return score >= thr


# ================= main 4x3 figure =================
fig, axes = plt.subplots(4, 3, figsize=(16.5, 11.5))
fig.subplots_adjust(left=0.13, top=0.86, bottom=0.04, right=0.98,
                    hspace=0.35, wspace=0.06)

for r, (dname, key, thr, _) in enumerate(DISCS):
    for c, (cid, isf, qp, dp, geo, jac, alpha, ours) in enumerate(CANDS):
        score = [None, None, None, None, geo, jac, alpha, ours][KEYIDX[key]]
        acc = verdict(score, thr)
        col = GREEN if acc else RED
        strip = compose_pair(os.path.join(QD, qp), os.path.join(QD, dp), col)
        ax = axes[r][c]
        ax.imshow(strip)
        ax.axis("off")
        label = f"{score:.3f}  {'接受 · 真回环' if acc else '拒绝 · 假回环'}"
        ax.set_title(label, fontsize=13, fontweight="bold",
                     color="#%02x%02x%02x" % col, pad=6)

# column headers (top)
for c, (cid, isf, *_rest) in enumerate(CANDS):
    gt = "假回环" if isf else "真回环"
    axes[0][c].text(0.5, 1.35, f"{cid}\n（真值：{gt}）", transform=axes[0][c].transAxes,
                    ha="center", va="bottom", fontsize=14, fontweight="bold")

# row labels (left)
for r, (dname, key, thr, _) in enumerate(DISCS):
    axes[r][0].text(-0.045, 0.5, f"{dname}\n(阈值 {thr:g})",
                    transform=axes[r][0].transAxes, ha="right", va="center",
                    fontsize=13.5, fontweight="bold")

fig.suptitle("fr1/room：四种判别器在 3 个 SALAD 回环候选上的判定\n"
             "（绿 = 判定为真回环并接受；红 = 判定为假回环并拒绝；横杠连接回环对应帧）",
             fontsize=16, fontweight="bold")
main_png = os.path.join(OUTDIR, "回环对比总图.png")
os.makedirs(OUTDIR, exist_ok=True)
fig.savefig(main_png, dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("[saved]", main_png)

# ================= package into sub-folders =================
# 1) SALAD candidates (raw 6 frames)
sub1 = os.path.join(OUTDIR, "1_SALAD候选所有图")
os.makedirs(sub1, exist_ok=True)
for cid, isf, qp, dp, *_ in CANDS:
    Image.open(os.path.join(QD, qp)).convert("RGB").save(os.path.join(sub1, f"{cid}_query.png"))
    Image.open(os.path.join(QD, dp)).convert("RGB").save(os.path.join(sub1, f"{cid}_detected.png"))
print("[saved] 6 raw frames ->", sub1)

# 2-5) one folder per method, pair strips colored by that method's verdict
for r, (dname, key, thr, _) in enumerate(DISCS):
    folder = os.path.join(OUTDIR, f"{r+2}_{dname}")
    os.makedirs(folder, exist_ok=True)
    for cid, isf, qp, dp, geo, jac, alpha, ours in CANDS:
        score = [None, None, None, None, geo, jac, alpha, ours][KEYIDX[key]]
        acc = verdict(score, thr)
        col = GREEN if acc else RED
        strip = compose_pair(os.path.join(QD, qp), os.path.join(QD, dp), col)

        # render with a title bar (method + score + verdict)
        tw = strip.width
        th = 44
        title_img = Image.new("RGB", (tw, strip.height + th), (255, 255, 255))
        title_img.paste(strip, (0, th))
        dr = ImageDraw.Draw(title_img)
        txt = f"{dname}  {cid}  分数 {score:.3f} (阈值 {thr:g})  →  {'接受·真回环' if acc else '拒绝·假回环'}"
        if FONT is not None:
            approx_w = int(FONT.getlength(txt))
        else:
            approx_w = int(len(txt) * 13)
        x = max(8, (tw - approx_w) // 2)
        dr.text((x, 12), txt, fill=col, font=FONT)
        title_img.save(os.path.join(folder, f"{cid}.png"))
    print(f"[saved] 3 strips -> {folder}")

print("DONE")
