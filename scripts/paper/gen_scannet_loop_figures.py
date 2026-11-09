#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ScanNet 假回环论文图生成器（v6：统一三图视角 + 遍历全部场景）。

每个场景产出（写入 /root/autodl-tmp/scannet_figs_all/<scene>/）：
  1) fig_scene_pcd_traj.png/.pdf — 1000万 RGB 点云 + 三轨迹(before红/after绿/GT深灰) + 回环放大 inset + XYZ 三轴
  2) fig_scene_sem.png/.pdf      — 1000万 GT 语义点云(官方 NYU40 配色) + 底部图例 + XYZ 三轴
  3) gif_scene_build.gif         — 轨迹生长动画(与上图同一视角/缩放)
  raw/README.txt                 — 数据来源清单(硬链接小文件)

三图视角/缩放严格一致：同一 figsize + 同一 full subplot + 同一 set_view(limits=pts_gt)。
语义图例用 fig.legend 放在图外下方，不再挤压 3D 轴（这是上一版视角不统一的根因）。

运行（AutoDL sgs 环境，nohup 自运行）：
  /root/miniconda3/envs/sgs/bin/python -u gen_scannet_loop_figures.py
"""
import os, sys, shutil, traceback
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

ABL = "/root/autodl-tmp/ablation/scannet"
DAT = "/root/autodl-tmp/v3dsam/datasets/scannet"
TOP = "/root/autodl-tmp/scannet_figs_all"
SCENES = ["scene0000_00", "scene0059_00", "scene0106_00", "scene0207_00"]
os.makedirs(TOP, exist_ok=True)

sys.path.insert(0, "/root/autodl-tmp/psnr_tum")
from scannet_labels import map_gt, NYU40

# ---------------------------------------------------------------- 官方 ScanNet NYU-40 配色（与 ScanNet benchmark util.py 一致）
NYU40_COLORS = np.array([
    (174, 199, 232), (152, 223, 138), (31, 119, 180), (255, 187, 120),
    (188, 189, 34),  (140, 86, 75),   (255, 152, 150), (214, 39, 40),
    (197, 176, 213), (148, 103, 189), (196, 156, 148), (23, 190, 207),
    (178, 34, 34),   (247, 182, 210), (66, 188, 102),  (219, 219, 141),
    (140, 150, 198), (136, 19, 19),   (119, 11, 32),   (143, 157, 178),
    (255, 0, 0),     (106, 137, 204), (90, 40, 50),    (85, 168, 233),
    (110, 109, 124), (197, 193, 170), (126, 116, 106), (201, 164, 205),
    (54, 72, 33),    (151, 151, 151), (211, 130, 130), (80, 180, 50),
    (80, 180, 50),   (80, 180, 50),   (150, 180, 200), (255, 255, 255),
    (140, 180, 140), (80, 90, 100),   (120, 120, 120), (100, 100, 100),
], dtype=np.float32) / 255.0

RED = "#d62728"; GREEN = "#2ca02c"; GTCOL = "#1a1a1a"; XYZ = ["#d62728", "#2ca02c", "#1f77b4"]

# ---------------------------------------------------------------- 两套视角参数（45°俯视 + 90°垂直俯视）+ 统一图尺寸
VIEW45 = dict(elev=45, azim=-68, box=(1, 1, 0.62), pad=0.02)
VIEW90 = dict(elev=90, azim=-90, box=(1, 1, 0.62), pad=0.02)
VIEWS = [("45", VIEW45), ("90", VIEW90)]
FIG_SIZE = (11.5, 9.5)


# ---------------------------------------------------------------- 轨迹对齐
def align_to_gt(est_path, gt_path):
    from evo.tools import file_interface
    from evo.core import sync, geometry
    gt = file_interface.read_tum_trajectory_file(gt_path)
    est = file_interface.read_tum_trajectory_file(est_path)
    est, gt = sync.associate_trajectories(est, gt)
    r_a, t_a, s_a = geometry.umeyama_alignment(est.positions_xyz.T, gt.positions_xyz.T,
                                               with_scale=True)
    est_aligned = (s_a * (r_a @ est.positions_xyz.T)).T + t_a
    return gt.positions_xyz, est_aligned, (r_a, t_a, s_a)


# ---------------------------------------------------------------- 点云
def load_gs_points(after_npz, n=10000000, seed=42):
    d = np.load(after_npz, allow_pickle=True)
    pts = d["pts_gs"].astype(np.float64)
    rgb = d["rgb_gs"]
    med = np.median(pts, axis=0)
    dist = np.linalg.norm(pts - med, axis=1)
    keep = dist < np.percentile(dist, 99.8)
    pts, rgb = pts[keep], rgb[keep]
    idx = np.random.RandomState(seed).choice(len(pts), min(n, len(pts)), replace=False)
    return pts[idx], rgb[idx]


def build_sem_on_ptsgs(scene, pts, frame_step=5):
    """把 GT 语义（NYU-40）正向投影赋到 pts_gs 的每个点上（与 v27 的 all_sem 机制一致）。"""
    from scipy.spatial.transform import Rotation as Rot
    npz = np.load(f"{ABL}/{scene}/_stage3_input_after.npz", allow_pickle=True)
    img_W, img_H = int(npz["img_W"]), int(npz["img_H"])
    idxs = sorted(int(k.split("_")[1]) for k in npz.keys() if k.startswith("K_"))
    est = np.loadtxt(f"{ABL}/{scene}/est_after_tum.txt")
    sem_dir = f"{DAT}/{scene}/semantic_id"
    N = len(pts)
    sem = np.full(N, -1, dtype=np.int32)
    pts_h = np.concatenate([pts, np.ones((N, 1))], axis=1)
    for i in idxs[::frame_step]:
        K = npz[f"K_{i}"].astype(np.float64)
        x, y, z, qx, qy, qz, qw = est[i][1:8]
        T = np.eye(4)
        T[:3, :3] = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
        T[:3, 3] = [x, y, z]
        w2c = np.linalg.inv(T)
        cam = pts_h @ w2c.T
        zz = cam[:, 2]
        im = cam[:, :3] @ K.T
        u = im[:, 0] / zz
        v = im[:, 1] / zz
        vis = (zz > 0.1) & (zz < 50) & (u >= 0) & (u < img_W) & (v >= 0) & (v < img_H)
        vis &= (sem == -1)
        if not vis.any():
            continue
        sm = np.array(Image.open(f"{sem_dir}/{i}.png"))
        sm = map_gt(sm)
        if sm.shape[0] != img_H or sm.shape[1] != img_W:
            import cv2
            sm = cv2.resize(sm.astype(np.float32), (img_W, img_H),
                            interpolation=cv2.INTER_NEAREST).astype(np.int32)
        ui = u[vis].astype(np.int32)
        vi = v[vis].astype(np.int32)
        labs = sm[vi, ui]
        ok = labs >= 0
        pidx = np.nonzero(vis)[0][ok]
        sem[pidx] = labs[ok]
    return sem


# ---------------------------------------------------------------- 视角 / 三轴
def set_view(ax, pts, view):
    lo, hi = pts.min(0), pts.max(0)
    ext = hi - lo
    pad = view["pad"]
    ax.set_xlim(lo[0] - pad * ext[0], hi[0] + pad * ext[0])
    ax.set_ylim(lo[1] - pad * ext[1], hi[1] + pad * ext[1])
    ax.set_zlim(lo[2] - pad * ext[2], hi[2] + pad * ext[2])
    ax.set_box_aspect(view["box"])
    ax.view_init(elev=view["elev"], azim=view["azim"])


def style_axes(ax):
    """带数轴刻度的坐标轴（替换原来的指向性 triad + set_axis_off）。"""
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel("X (m)", fontsize=12, labelpad=8)
    ax.set_ylabel("Y (m)", fontsize=12, labelpad=8)
    ax.set_zlabel("Z (m)", fontsize=12, labelpad=8)
    ax.tick_params(labelsize=10)


def add_zoom_inset(fig, gt, be, af):
    dev = np.linalg.norm(be[:, :2] - af[:, :2], axis=1)
    i = int(np.argmax(dev))
    cx, cy = af[i, 0], af[i, 1]
    hs = 0.45
    inset = fig.add_axes([0.66, 0.56, 0.27, 0.27])
    inset.set_facecolor("white")
    inset.plot(gt[:, 0], gt[:, 1], color=GTCOL, lw=1.7, zorder=3, label="GT")
    inset.plot(be[:, 0], be[:, 1], color=RED, lw=1.2, alpha=0.9, zorder=2, label="before")
    inset.plot(af[:, 0], af[:, 1], color=GREEN, lw=1.8, zorder=1, label="after")
    inset.set_xlim(cx - hs, cx + hs); inset.set_ylim(cy - hs, cy + hs)
    inset.set_xticks([]); inset.set_yticks([])
    inset.set_aspect("equal")
    inset.set_title("top view @ max deviation", fontsize=7.5, color="#ff7f0e", pad=3)
    for s in inset.spines.values():
        s.set_edgecolor("#ff7f0e"); s.set_linewidth(1.1)


# ---------------------------------------------------------------- 图1：RGB 点云 + 三轨迹
def fig_pcd_traj(scene, out, pts_gt, rgb, gt, be_a, af_a, view):
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts_gt[:, 0], pts_gt[:, 1], pts_gt[:, 2], c=rgb, s=0.28,
               linewidths=0, alpha=0.6, rasterized=True, zorder=1)
    ax.plot(be_a[:, 0], be_a[:, 1], be_a[:, 2], color=RED, lw=1.4,
            label="before loop closure (w/ false loop)", zorder=6)
    ax.plot(af_a[:, 0], af_a[:, 1], af_a[:, 2], color=GREEN, lw=2.0,
            label="after loop closure (ours)", zorder=7)
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color=GTCOL, lw=1.6, ls="-",
            label="GT", zorder=8)
    ax.scatter(gt[0:1, 0], gt[0:1, 1], gt[0:1, 2], color=GTCOL, s=45, marker="o",
               zorder=9)
    set_view(ax, pts_gt, view)
    style_axes(ax)
    ax.legend(fontsize=11, loc="upper right", framealpha=0.95, edgecolor="0.8")
    ax.set_title(f"{scene}: dense RGB reconstruction + trajectory "
                 "(before/after loop verification)", fontsize=13, color="#1a1a1a", pad=12)
    add_zoom_inset(fig, gt, be_a, af_a)
    fig.savefig(out + ".png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"    [fig1] {out}.png / .pdf  ({len(pts_gt):,} pts)")


# ---------------------------------------------------------------- 图2：语义点云 + 图例（图例在图外，不挤压3D轴）
def fig_sem(scene, out, pts_gt, sem, view):
    sem_col = np.zeros((len(pts_gt), 3), np.float32)
    valid = (sem >= 0) & (sem < 40)
    sem_col[valid] = NYU40_COLORS[sem[valid]]
    sem_col[~valid] = 0.72
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts_gt[:, 0], pts_gt[:, 1], pts_gt[:, 2], c=sem_col, s=0.5,
               linewidths=0, rasterized=True, zorder=1)
    set_view(ax, pts_gt, view)
    style_axes(ax)
    ax.set_title(f"{scene}: semantic point cloud (ScanNet NYU-40, official colors)",
                 fontsize=13, color="#1a1a1a", pad=12)
    # 图例：场景中出现的类别，按点数降序
    present = sorted([c for c in range(40) if (sem == c).any()],
                     key=lambda c: -int((sem == c).sum()))
    handles = [mpatches.Patch(facecolor=NYU40_COLORS[c], edgecolor="none",
                              label=f"{NYU40[c]} ({int((sem == c).sum()):,})")
               for c in present]
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.03),
                     ncol=7, fontsize=8.5, frameon=False, handlelength=1.0,
                     handleheight=1.0, columnspacing=0.8, labelspacing=0.45)
    fig.savefig(out + ".png", dpi=220, bbox_inches="tight", facecolor="white",
                bbox_extra_artists=(leg,))
    fig.savefig(out + ".pdf", bbox_inches="tight", bbox_extra_artists=(leg,))
    plt.close(fig)
    print(f"    [fig2] {out}.png / .pdf  ({len(pts_gt):,} pts, {int(valid.sum()):,} labeled)")


# ---------------------------------------------------------------- 图3：轨迹生长 GIF（同视角）
def gif_build(scene, out, pts_gt, rgb, gt, be_a, af_a, view, n_frames=60):
    from matplotlib.animation import FuncAnimation
    sub = np.random.RandomState(0).choice(len(pts_gt), min(400000, len(pts_gt)), replace=False)
    psub = pts_gt[sub]; csub = rgb[sub]
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(psub[:, 0], psub[:, 1], psub[:, 2], c=csub, s=0.3,
               linewidths=0, alpha=0.55, rasterized=True, zorder=1)
    ax.plot(be_a[:, 0], be_a[:, 1], be_a[:, 2], color=RED, lw=1.3, alpha=0.45,
            label="before loop closure", zorder=5)
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color=GTCOL, lw=1.6, ls="-", label="GT", zorder=6)
    line_af, = ax.plot([], [], [], color=GREEN, lw=2.2, label="after loop closure (ours)", zorder=7)
    set_view(ax, pts_gt, view)
    style_axes(ax)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95, edgecolor="0.8")
    ax.set_title(f"{scene}: trajectory build-up (green = after loop verification)",
                 fontsize=12, color="#1a1a1a", pad=12)

    N = len(af_a)
    def update(k):
        m = int((k + 1) / n_frames * N)
        line_af.set_data(af_a[:m, 0], af_a[:m, 1])
        line_af.set_3d_properties(af_a[:m, 2])
        return [line_af]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=90, blit=False)
    anim.save(out, writer="pillow", fps=10, dpi=130)
    plt.close(fig)
    print(f"    [gif] {out}")


# ---------------------------------------------------------------- 归档（轻量：清单 + 硬链接小文件）
def hardlink(src, dst):
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def archive(scene, out):
    raw = os.path.join(out, "raw")
    os.makedirs(raw, exist_ok=True)
    for f in ("_stage3_input_after.npz", "est_before_tum.txt", "est_after_tum.txt",
              "loop_log.json"):
        src = f"{ABL}/{scene}/{f}"
        if os.path.exists(src):
            hardlink(src, f"{raw}/{f}")
    for f in ("gt_tum.txt",):
        src = f"{DAT}/{scene}/{f}"
        if os.path.exists(src):
            hardlink(src, f"{raw}/{f}")
    with open(f"{out}/raw/README.txt", "w") as f:
        f.write(f"{scene} 假回环验证原始数据\n\n")
        f.write("raw/_stage3_input_after.npz  原始融合点云 pts_gs(1000万)/rgb_gs + 逐帧 depth/K/w2c/img_path\n")
        f.write("raw/est_before_tum.txt        消假回环前轨迹\n")
        f.write("raw/est_after_tum.txt         消假回环后轨迹\n")
        f.write("raw/gt_tum.txt                ScanNet 动捕真值\n")
        f.write("raw/loop_log.json             回环候选 + 假回环判定\n")
        f.write("\n大文件原地保留(未硬链接):\n")
        f.write(f"  {DAT}/{scene}/color/         原始 RGB 帧\n")
        f.write(f"  {DAT}/{scene}/semantic_id/   GT 语义 raw-id PNG\n")
    print(f"    [archive] {out}/raw/README.txt")


def run_scene(scene):
    print(f"== {scene} ==")
    out = f"{TOP}/{scene}"
    os.makedirs(out, exist_ok=True)
    gt_path = f"{DAT}/{scene}/gt_tum.txt"
    be_path = f"{ABL}/{scene}/est_before_tum.txt"
    af_path = f"{ABL}/{scene}/est_after_tum.txt"
    npz = f"{ABL}/{scene}/_stage3_input_after.npz"

    print("  [align] ...")
    gt, af_a, (R, t, s) = align_to_gt(af_path, gt_path)
    _, be_a, _ = align_to_gt(be_path, gt_path)

    print("  [load] 10M points ...")
    pts, rgb = load_gs_points(npz, n=10000000)

    print("  [sem] forward-project onto pts_gs ...")
    sem = build_sem_on_ptsgs(scene, pts)

    pts_gt = (s * (R @ pts.T)).T + t
    med = np.median(pts_gt, axis=0)
    dd = np.linalg.norm(pts_gt - med, axis=1)
    keep = dd < np.percentile(dd, 99.9)
    pts_gt, rgb, sem = pts_gt[keep], rgb[keep], sem[keep]

    for tag, v in VIEWS:
        fig_pcd_traj(scene, f"{out}/fig_{scene}_pcd_traj_{tag}", pts_gt, rgb, gt, be_a, af_a, v)
        fig_sem(scene, f"{out}/fig_{scene}_sem_{tag}", pts_gt, sem, v)
        gif_build(scene, f"{out}/gif_{scene}_build_{tag}.gif", pts_gt, rgb, gt, be_a, af_a, v)
    archive(scene, out)
    print(f"== {scene} DONE -> {out}\n")


if __name__ == "__main__":
    for sc in SCENES:
        try:
            run_scene(sc)
        except Exception as e:
            print(f"!! {sc} FAILED: {e}")
            traceback.print_exc()
    print("ALL SCENES DONE ->", TOP)
