#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出"最原始"的组件图（不做拼接），供用户自行组合：
  fig16 3×5 持久实例网格 -> 15 张全分辨率标注帧（3 场景 × 3 重叠关键帧 + 2 回环帧）
  fig11 轨迹 2×2        -> 4 张单场景轨迹图（GT vs 回环前 vs 回环后 + 放大细节小图）

复用远程已有模块 make_scannet_figures / gen_instance_grid_scannet 的算法与数据路径。
"""
import os, sys, json
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/root/autodl-tmp/psnr_tum")
import make_scannet_figures as m
import gen_instance_grid_scannet as g

OUT = "/root/autodl-tmp/psnr_tum/raw_export"
os.makedirs(f"{OUT}/fig16_instance_frames", exist_ok=True)
os.makedirs(f"{OUT}/fig11_traj", exist_ok=True)


def draw_label_big(img, text, cu, cv_, colrgb, W, H):
    """比 gen_instance_grid 里的 draw_label 更大的标签（全分辨率单帧用）。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 1.8, 4
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    tx, ty = cu + 20, cv_ - 16
    x0, y0 = tx - 4, ty - th - 4
    x1, y1 = tx + tw + 4, ty + base + 4
    x0 = max(2, min(x0, W - 2 - (x1 - x0)))
    x1 = x0 + tw + 8
    y0 = max(2, min(y0, H - 2 - (y1 - y0)))
    y1 = y0 + th + base + 8
    tx, ty = x0 + 4, y0 + th + 4
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0), (x1, y1), colrgb, 3, cv2.LINE_AA)
    cv2.putText(img, text, (tx, ty), font, scale, (0, 0, 0), thick, cv2.LINE_AA)


# ---------------------------------------------------------------- fig11 单场景轨迹
for scene in m.SCENES:
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    gt_path = f"{m.DAT}/{scene}/gt_tum.txt"
    be_path = f"{m.ABL}/{scene}/est_before_tum.txt"
    af_path = f"{m.ABL}/{scene}/est_after_tum.txt"
    if not all(os.path.exists(p) for p in (gt_path, be_path, af_path)):
        plt.close(fig); print(f"[fig11 raw] SKIP {scene} (missing)"); continue
    gt, be_a = m.align_to_gt_evo(be_path, gt_path)
    _, af_a = m.align_to_gt_evo(af_path, gt_path)
    ax.plot(gt[:, 0], gt[:, 1], color="#222222", lw=1.0, ls="--", label="GT", zorder=3)
    ax.plot(be_a[:, 0], be_a[:, 1], color="#d62728", lw=1.0, alpha=0.85, label="before LC", zorder=2)
    ax.plot(af_a[:, 0], af_a[:, 1], color="#1f77b4", lw=1.3, label="after LC (ours)", zorder=1)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(m.SCENE_TITLE[scene], fontsize=14, color="#272727")
    log = f"{m.ABL}/{scene}/loop_log.json"
    if os.path.exists(log):
        j = json.load(open(log))
        ax.text(0.03, 0.03, f"{len(j.get('loops', []))} loops, {len(j.get('reject_lc_ids', []))} false rejected",
                transform=ax.transAxes, fontsize=11, color="#d62728", va="bottom")
    m._add_zoom_inset(ax, gt, be_a, af_a)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    fig.savefig(f"{OUT}/fig11_traj/{scene}_traj.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig11 raw] -> {scene}_traj.png")

# ---------------------------------------------------------------- fig16 全分辨率标注帧
for scene in g.SCENES:
    name, s3_rel, sem_rel, inst_rel = g.SCENES[scene]
    base = f"{g.BASE}/{name}"
    stage3 = np.load(f"{base}/{s3_rel}", allow_pickle=True)
    sem = np.load(f"{base}/{sem_rel}", allow_pickle=True)
    kf_indices = [int(x) for x in sem["kf_indices"].tolist()]
    img_W, img_H = int(stage3["img_W"]), int(stage3["img_H"])
    objs = g.load_objs(base, inst_rel)[:18]
    K = len(kf_indices)
    vis = np.zeros((K, len(objs)), dtype=bool)
    for k, fi in enumerate(kf_indices):
        mm, _, _ = g.visible_mask(objs, stage3[f"w2c_{fi}"].astype(np.float64),
                                  stage3[f"K_{fi}"].astype(np.float64), img_W, img_H)
        vis[k] = mm
    score, win_pos, vis_local, vis_loop = g.pick_window(vis)
    win = [kf_indices[p] for p in win_pos]
    sel = [i for i in range(len(objs)) if vis_local[i] >= 2 and vis_loop[i] >= 1]
    safe = scene.replace(" ", "_").replace("(", "").replace(")", "")
    print(f"[fig16 raw] {scene}: win={win}, n_persist={score}, sel={len(sel)}")
    for c, fi in enumerate(win):
        img = cv2.imread(str(stage3[f"img_path_{fi}"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        w2c = stage3[f"w2c_{fi}"].astype(np.float64)
        Kk = stage3[f"K_{fi}"].astype(np.float64)
        H, W = img.shape[:2]
        mm, u, v = g.visible_mask(objs, w2c, Kk, W, H)
        for j in sel:
            if not mm[j]:
                continue
            colrgb = g.hex2rgb(g.TAB20[objs[j]["id"] % len(g.TAB20)])
            cu, cv_ = int(round(u[j])), int(round(v[j]))
            cv2.circle(img, (cu, cv_), 20, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (cu, cv_), 18, colrgb, -1, cv2.LINE_AA)
            draw_label_big(img, f"#{objs[j]['id']} {objs[j]['name']}", cu, cv_, colrgb, W, H)
        tag = "kf" if c < 3 else "loop"
        outp = f"{OUT}/fig16_instance_frames/{safe}__{tag}{fi}.png"
        cv2.imwrite(outp, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"  [fig16 raw] -> {safe}__{tag}{fi}.png")

print("ALL RAW EXPORT DONE")
