#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""查看 10M 语义点云 / RGB 点云（Jupyter 或本地脚本）。

数据来源（gen_semantic_cloud_10m.py 产出，每场景 semantic_cloud_10m/ 下）：
  semantic_cloud.npz  pts(float32 Nx3) + label(int16 N, -1=未标注) + prompts(object)
  rgb_cloud.ply       xyz + 真实 RGB
  semantic_cloud.ply  xyz + 类别色 RGB + label(int)

Jupyter 用法（任选其一）：
  %run view_semantic_cloud.py                 # 直接跑，用下面默认参数
  # 或把整段复制进一个 cell 运行

本地脚本用法：
  python view_semantic_cloud.py

改下面 3 个参数即可切换场景 / 密度：
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============ 参数 ============
SCENE = "scene0000_00"                         # scene0000_00 / scene0059_00 / scene0106_00
BASE = r"E:\0722v3dsam\semantic_cloud_10m"     # 点云所在根目录
N_SHOW = 300_000                               # 交互/静态图下采样点数（10M 全量对浏览器太重）
# =============================

# 50 类调色板（与生成脚本 gen_semantic_pointcloud.py 完全一致）
_tab = [np.array(plt.get_cmap(cm).colors) for cm in ("tab20", "tab20b", "tab20c")]
PALETTE = (np.vstack(_tab)[:50][:, :3] * 255).astype(np.uint8)


def load_semantic(scene, base=BASE):
    d = np.load(os.path.join(base, scene, "semantic_cloud.npz"), allow_pickle=True)
    pts = d["pts"].astype(np.float32)
    label = d["label"].astype(np.int32)
    prompts = [str(p) for p in d["prompts"]]
    # 着色：标注点用类别色，未标注点浅灰
    cols = np.full((len(pts), 3), 200, np.uint8)
    ok = label >= 0
    cols[ok] = PALETTE[label[ok]]
    return pts, cols, label, prompts


def load_rgb(scene, base=BASE):
    """从 rgb_cloud.ply 读真实 RGB 点云（仅 xyz+rgb，无 label）。"""
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(os.path.join(base, scene, "rgb_cloud.ply"))
    pts = np.asarray(pcd.points, np.float32)
    cols = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    return pts, cols


def subsample(*arrays, n=N_SHOW):
    N = len(arrays[0])
    if N <= n:
        return arrays
    idx = np.random.RandomState(0).choice(N, n, replace=False)
    idx.sort()
    return [a[idx] for a in arrays]


# ================= 1) 语义点云（类别色） =================
pts, cols, label, prompts = load_semantic(SCENE)
print(f"\n[{SCENE}] semantic cloud: {len(pts):,} points, {len(prompts)} categories")
present = sorted(set(label.tolist()) - {-1})
print("  出现类别:", [(c, prompts[c]) for c in present])

# ================= 2) Open3D 交互视图（Jupyter 内可旋转/缩放） =================
try:
    import open3d as o3d
    pts_s, cols_s = subsample(pts, cols, n=N_SHOW)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_s)
    pcd.colors = o3d.utility.Vector3dVector(cols_s.astype(np.float64) / 255.0)
    print(f"[Open3D] 交互显示 {len(pts_s):,} 点（语义色）...")
    o3d.visualization.draw_plotly([pcd], width=1000, height=700)
except ImportError:
    print("[Open3D] 未安装，跳过交互视图（可 pip install open3d）")

# ================= 3) matplotlib 静态三视图（兜底，无需 Open3D） =================
pts_s, cols_s = subsample(pts, cols, n=N_SHOW)
cols_f = cols_s.astype(np.float32) / 255.0


def set_equal(ax, P):
    mid = P.mean(0)
    rng = np.ptp(P, 0).max() / 2.0
    ax.set_xlim(mid[0] - rng, mid[0] + rng)
    ax.set_ylim(mid[1] - rng, mid[1] + rng)
    ax.set_zlim(mid[2] - rng, mid[2] + rng)


fig = plt.figure(figsize=(15, 5))
for i, (elev, azim, title) in enumerate([(28, -62, "(a) perspective"),
                                         (28, 152, "(b) opposite"),
                                         (88, 0, "(c) top-down")]):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    ax.scatter(pts_s[:, 0], pts_s[:, 1], pts_s[:, 2], c=cols_f, s=0.25,
               alpha=0.7, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=elev, azim=azim)
    set_equal(ax, pts_s)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")

handles = [plt.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=PALETTE[c] / 255.0, markersize=6,
                      label=f"{prompts[c]}" if False else f"{c}:{prompts[c]}")
           for c in present]
fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=6,
           frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle(f"{SCENE} semantic point cloud ({len(pts_s):,} shown)", fontsize=11)
fig.tight_layout()
out_png = os.path.join(BASE, SCENE, "semantic_cloud_3view.png")
fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
print(f"[matplotlib] 三视图已保存 -> {out_png}")

# ================= 4) RGB 点云（真实颜色，可选） =================
try:
    import open3d as o3d
    pts_r, cols_r = load_rgb(SCENE)
    pts_r, cols_r = subsample(pts_r, cols_r, n=N_SHOW)
    pcd_r = o3d.geometry.PointCloud()
    pcd_r.points = o3d.utility.Vector3dVector(pts_r)
    pcd_r.colors = o3d.utility.Vector3dVector(cols_r.astype(np.float64) / 255.0)
    print(f"[Open3D] RGB 交互显示 {len(pts_r):,} 点...")
    o3d.visualization.draw_plotly([pcd_r], width=1000, height=700)
except ImportError:
    print("[Open3D] 未安装，跳过 RGB 交互视图")
