#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ScanNet A 类数据图（不需要 SAM3）：
  1) fig_scannet_traj.png      — 4 场景轨迹：GT(虚线) vs 回环前(红) vs 回环后(蓝)，标注假回环
  2) fig_scannet_pcd.png       — RGB 点云 + GT 语义点云（逐帧深度反投影，GT semantic_id 着色）
  3) fig_scannet_sem_traj.png  — GT 语义点云 + 预测轨迹 + 假回环标记

数据依赖（全部已存在或已上传）：
  - /root/autodl-tmp/ablation/scannet/<scene>/_stage3_input_after.npz  (w2c/K/depth/img_path/pts_gs/rgb_gs)
  - /root/autodl-tmp/ablation/scannet/<scene>/est_before_tum.txt / est_after_tum.txt
  - /root/autodl-tmp/ablation/scannet/<scene>/loop_log.json
  - /root/autodl-tmp/v3dsam/datasets/scannet/<scene>/gt_tum.txt
  - /root/autodl-tmp/v3dsam/datasets/scannet/<scene>/semantic_id/*.png   (GT, 已上传)

运行（AutoDL sgs 环境）：python make_scannet_figures.py
"""
import os, sys, json, argparse, glob
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

try:
    from sci_style import apply_publication_style, finalize_figure
    apply_publication_style(font_size=8, axes_linewidth=1)
except Exception:
    def finalize_figure(fig, out):
        fig.savefig(out + ".png", dpi=160, bbox_inches="tight")
        fig.savefig(out + ".pdf", bbox_inches="tight")
        print(f"-> {out}.png / .pdf")

ABL = "/root/autodl-tmp/ablation/scannet"
DAT = "/root/autodl-tmp/v3dsam/datasets/scannet"
OUT = "/root/autodl-tmp/psnr_tum/scannet_figs"
os.makedirs(OUT, exist_ok=True)

# ScanNet 语义 PNG 存 raw id，经 scannet_labels 映射到 nyu40id(0-based)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scannet_labels import NYU40, map_gt

SCENES = ["scene0000_00", "scene0059_00", "scene0106_00", "scene0207_00"]
SCENE_TITLE = {"scene0000_00": "scene0000 (bedroom)", "scene0059_00": "scene0059 (office)",
               "scene0106_00": "scene0106 (living)", "scene0207_00": "scene0207 (office)"}

# 40 个区分色（tab20 + tab20b）
_cmap = np.concatenate([plt.cm.tab20(np.arange(20)), plt.cm.tab20b(np.arange(20))])
NYU40_COLORS = (_cmap[:, :3] * 255).astype(np.uint8)  # (40,3) rgb 0-255
# 把背景/结构类调成灰调更直观（wall/floor/ceiling 不抢眼）
for i in [0, 1, 21]:
    NYU40_COLORS[i] = np.array([200, 200, 200])


# ---------------------------------------------------------------- 轨迹
def read_tum(path):
    if not os.path.exists(path):
        return None, None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 8:
                continue
            rows.append([float(x) for x in p[:8]])
    a = np.array(rows)
    t = a[:, 0]
    pos = a[:, 1:4]
    quat = a[:, 4:8]
    return pos, quat


def align_to_gt_evo(est_path, gt_path):
    """用 evo 的 Sim(3) Umeyama 对齐 est -> gt（等价 stage1 代码的 `evo_traj -as`）。
    返回 (gt_positions, est_aligned_positions)。

    注：此前自定义的 umeyama()（scale = S.sum()/var_s）缺少 det 符号修正与近线性回退，
    与 evo_ape -as 的对齐不一致；改用 evo 官方对齐与 stage1 的轨迹生成保持一致。"""
    from evo.tools import file_interface
    from evo.core import sync
    gt = file_interface.read_tum_trajectory_file(gt_path)
    est = file_interface.read_tum_trajectory_file(est_path)
    # 时间戳对齐（scene0059 的 GT 缺 20 帧，须先按时间戳关联再对齐）
    est, gt = sync.associate_trajectories(est, gt)
    est.align(gt, correct_scale=True)  # -as = Sim(3) Umeyama + scale correction
    return gt.positions_xyz, est.positions_xyz


def _add_zoom_inset(ax, gt, be, af):
    """在轨迹图上叠加放大细节小图：聚焦回环修正最大处（before vs after 偏差最大）。"""
    try:
        from matplotlib.patches import Rectangle
        pts = np.concatenate([gt, be, af], axis=0)[:, :2]
        dev = np.linalg.norm(be[:, :2] - af[:, :2], axis=1)
        i = int(np.argmax(dev))
        cx, cy = af[i, 0], af[i, 1]
        ext = np.ptp(pts, axis=0)
        hs = max(0.25 * float(max(ext)), 0.15)   # 放大窗口半宽（米）
        ax.add_patch(Rectangle((cx - hs, cy - hs), 2 * hs, 2 * hs,
                               fill=False, ec="#ff7f0e", lw=1.1, ls=":", zorder=5))
        inset = ax.inset_axes([0.52, 0.03, 0.45, 0.45])
        inset.set_facecolor("white")
        inset.plot(gt[:, 0], gt[:, 1], color="#222222", lw=1.3, ls="--", zorder=3)
        inset.plot(be[:, 0], be[:, 1], color="#d62728", lw=1.3, alpha=0.85, zorder=2)
        inset.plot(af[:, 0], af[:, 1], color="#1f77b4", lw=1.8, zorder=1)
        inset.set_xlim(cx - hs, cx + hs)
        inset.set_ylim(cy - hs, cy + hs)
        inset.set_xticks([]); inset.set_yticks([])
        inset.set_aspect("equal")
        for s in inset.spines.values():
            s.set_edgecolor("#ff7f0e"); s.set_linewidth(0.9)
    except Exception as e:
        print(f"  [zoom inset skip] {e}")


def traj_figure():
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.0))
    axes = axes.ravel()
    for ax, scene in zip(axes, SCENES):
        gt_path = f"{DAT}/{scene}/gt_tum.txt"
        be_path = f"{ABL}/{scene}/est_before_tum.txt"
        af_path = f"{ABL}/{scene}/est_after_tum.txt"
        if not all(os.path.exists(p) for p in (gt_path, be_path, af_path)):
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(SCENE_TITLE[scene], fontsize=12)
            ax.axis("off"); continue
        # 以 GT 为参照，用 evo 的 Sim(3) Umeyama 对齐（before/after 各自对齐到 GT）
        gt, be_a = align_to_gt_evo(be_path, gt_path)
        _, af_a = align_to_gt_evo(af_path, gt_path)
        ax.plot(gt[:, 0], gt[:, 1], color="#222222", lw=1.0, ls="--", label="GT", zorder=3)
        ax.plot(be_a[:, 0], be_a[:, 1], color="#d62728", lw=1.0, alpha=0.85, label="before LC", zorder=2)
        ax.plot(af_a[:, 0], af_a[:, 1], color="#1f77b4", lw=1.2, label="after LC (ours)", zorder=1)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(SCENE_TITLE[scene], fontsize=12, color="#272727")
        # 假回环标注（文本）
        log = f"{ABL}/{scene}/loop_log.json"
        if os.path.exists(log):
            j = json.load(open(log))
            nrej = len(j.get("reject_lc_ids", []))
            nloop = len(j.get("loops", []))
            ax.text(0.03, 0.03, f"{nloop} loops, {nrej} false rejected",
                    transform=ax.transAxes, fontsize=10, color="#d62728", va="bottom")
        # 放大细节小图
        _add_zoom_inset(ax, gt, be_a, af_a)
    axes[0].legend(fontsize=9, loc="upper left", framealpha=0.9, ncol=1)
    fig.suptitle("ScanNet trajectory: VGGT-SLAM front-end before / after semantic-geometric loop verification\n"
                 "(insets zoom into the region of largest loop correction)",
                 fontsize=13, y=1.02)
    finalize_figure(fig, f"{OUT}/fig_scannet_traj")
    print(f"[traj] -> {OUT}/fig_scannet_traj.png")


# ---------------------------------------------------------------- 点云
def backproject_frame(depth, K, w2c, step=8):
    """把深度图反投影到世界坐标（每 step 像素取 1 个，稀疏化）。返回 (N,3) 世界点 + (ys,xs) 像素坐标。"""
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    z = depth[ys, xs].astype(np.float64)
    valid = np.isfinite(z) & (z > 0.05) & (z < 50.0)
    ys, xs, z = ys[valid], xs[valid], z[valid]
    if len(z) == 0:
        return None, None, None
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    pc = np.stack([x, y, z], axis=1)  # (N,3) camera
    ones = np.ones((len(pc), 1))
    pw = (np.linalg.inv(w2c) @ np.hstack([pc, ones]).T).T[:, :3]
    return pw, ys, xs


def build_pointcloud(scene, frame_step=40, pix_step=8, max_pts=60000):
    npz = np.load(f"{ABL}/{scene}/_stage3_input_after.npz", allow_pickle=True)
    img_W, img_H = int(npz["img_W"]), int(npz["img_H"])
    # 收集所有帧号（K_0, K_1, ...）
    idxs = sorted(int(k.split("_")[1]) for k in npz.keys() if k.startswith("K_"))
    pts_all, rgb_all, sem_all = [], [], []
    sem_dir = f"{DAT}/{scene}/semantic_id"
    for i in idxs[::frame_step]:
        depth = npz[f"depth_{i}"]
        K = npz[f"K_{i}"].astype(np.float64)
        w2c = npz[f"w2c_{i}"].astype(np.float64)
        H, W = depth.shape
        if H != img_H or W != img_W:
            depth = cv2.resize(depth.astype(np.float32), (img_W, img_H),
                               interpolation=cv2.INTER_NEAREST)
        pw, ys, xs = backproject_frame(depth, K, w2c, step=pix_step)
        if pw is None:
            continue
        # RGB
        ip = str(npz[f"img_path_{i}"])
        im = np.array(Image.open(ip).convert("RGB"))  # (H,W,3)
        rgb = im[ys, xs].reshape(-1, 3)
        # GT semantic
        sem_path = f"{sem_dir}/{i}.png"
        sem = None
        if os.path.exists(sem_path):
            sm = np.array(Image.open(sem_path))  # raw id (H,W)
            sm = map_gt(sm)                       # -> nyu40id 0-based
            sem = sm[ys, xs].reshape(-1)
        pts_all.append(pw); rgb_all.append(rgb)
        if sem is not None:
            sem_all.append(sem)
    if not pts_all:
        return None, None, None
    pts = np.concatenate(pts_all).astype(np.float32)
    rgb = np.concatenate(rgb_all).astype(np.float32)
    sem = np.concatenate(sem_all).astype(np.int32) if sem_all else None
    # 去重/裁剪：裁掉离群点
    med = np.median(pts, axis=0)
    d = np.linalg.norm(pts - med, axis=1)
    keep = d < (np.percentile(d, 98) * 3)
    pts, rgb = pts[keep], rgb[keep]
    if sem is not None:
        sem = sem[keep]
    # 均匀降采样到 max_pts
    if len(pts) > max_pts:
        sel = np.random.RandomState(0).choice(len(pts), max_pts, replace=False)
        pts, rgb = pts[sel], rgb[sel]
        sem = sem[sel] if sem is not None else None
    return pts, rgb, sem


def _plot_pcd(ax, pts, colors, title, elev=25, azim=-60):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=0.6, linewidths=0,
               rasterized=True)
    ax.set_title(title, fontsize=10, color="#272727")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_box_aspect((1, 1, 0.7))
    ax.view_init(elev=elev, azim=azim)


def pcd_figure(scene="scene0000_00"):
    pts, rgb, sem = build_pointcloud(scene)
    if pts is None:
        print("[pcd] no points for", scene); return
    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    _plot_pcd(ax1, pts, rgb / 255.0, "RGB point cloud (reconstruction)")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    # GT 语义着色
    sem_col = np.zeros((len(pts), 3), np.float32)
    valid = (sem >= 0) & (sem < 40)
    if valid.any():
        sem_col[valid] = NYU40_COLORS[sem[valid]] / 255.0
    else:
        sem_col[:] = 0.6
    _plot_pcd(ax2, pts, sem_col, "GT semantic point cloud")
    fig.suptitle(f"ScanNet {SCENE_TITLE[scene]}: dense reconstruction vs ground-truth semantics",
                 fontsize=12, y=1.0)
    finalize_figure(fig, f"{OUT}/fig_scannet_pcd")
    print(f"[pcd] -> {OUT}/fig_scannet_pcd.png  ({len(pts)} pts)")


def sem_traj_figure(scene="scene0000_00"):
    """GT 语义点云 + 预测轨迹 + 假回环标记。"""
    pts, rgb, sem = build_pointcloud(scene, frame_step=50, max_pts=80000)
    if pts is None:
        print("[sem_traj] no points"); return
    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    sem_col = np.zeros((len(pts), 3), np.float32)
    valid = (sem >= 0) & (sem < 40)
    sem_col[valid] = NYU40_COLORS[sem[valid]] / 255.0
    sem_col[~valid] = 0.6
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=sem_col, s=0.5, linewidths=0, rasterized=True)
    # 预测轨迹（after）
    af_p, _ = read_tum(f"{ABL}/{scene}/est_after_tum.txt")
    if af_p is not None:
        # 轨迹与点云同处 est 世界系，直接叠加
        ax.plot(af_p[:, 0], af_p[:, 1], af_p[:, 2], color="#d62728", lw=1.6, label="predicted trajectory")
        ax.scatter(af_p[0:1, 0], af_p[0:1, 1], af_p[0:1, 2], color="green", s=40, marker="o", label="start")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=30, azim=-55)
    ax.set_title(f"{SCENE_TITLE[scene]}: GT semantic map + predicted trajectory", fontsize=10, color="#272727")
    ax.legend(fontsize=8)
    finalize_figure(fig, f"{OUT}/fig_scannet_sem_traj")
    print(f"[sem_traj] -> {OUT}/fig_scannet_sem_traj.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0000_00")
    ap.add_argument("--only", default="", help="comma: traj,pcd,semtraj")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    if only is None or "traj" in only:
        traj_figure()
    if only is None or "pcd" in only:
        pcd_figure(args.scene)
    if only is None or "semtraj" in only:
        sem_traj_figure(args.scene)
    print("done")


if __name__ == "__main__":
    main()
