#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""渲染语义点云 + RGB 点云的静态对比图（open3d offscreen，降采样 ~500K 点）。

输出 6 张 PNG（每场景语义 2 视角 + RGB 2 视角），供本地查看点云效果。

Run on AutoDL (sgs env):
  /root/miniconda3/envs/sgs/bin/python render_pcd_compare.py \
    --scene scene0000_00 --out /root/autodl-tmp/pcd_preview
"""
import os, argparse
import numpy as np
import open3d as o3d


def render_pcd(points, colors, out_png, views, n_show=500_000):
    N = len(points)
    if N > n_show:
        idx = np.random.RandomState(0).choice(N, n_show, replace=False)
        idx.sort()
        points = points[idx]
        colors = colors[idx]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1200, height=900, visible=False)
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    for tag, (front, lookat, up, zoom) in views:
        ctr.set_front(front)
        ctr.set_lookat(lookat)
        ctr.set_up(up)
        ctr.set_zoom(zoom)
        vis.poll_events()
        vis.update_renderer()
        fn = out_png.replace(".png", f"_{tag}.png")
        vis.capture_screen_image(fn, do_render=True)
        print(f"  -> {fn}")
    vis.destroy_window()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0000_00")
    ap.add_argument("--out", default="/root/autodl-tmp/pcd_preview")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    base = f"/root/autodl-tmp/ablation/scannet/{args.scene}"
    sem_dir = f"{base}/semantic_cloud_10m"

    # 语义点云：xyz + 类别色（semantic_cloud.ply 里 red/green/blue 就是类别色）
    print(f"[sem] loading semantic_cloud.ply ...")
    sem_pcd = o3d.io.read_point_cloud(f"{sem_dir}/semantic_cloud.ply")
    sem_pts = np.asarray(sem_pcd.points, np.float32)
    sem_col = (np.asarray(sem_pcd.colors) * 255).astype(np.uint8)
    print(f"[sem] {len(sem_pts):,} points")

    # RGB 点云：xyz + 真实色
    print(f"[rgb] loading rgb_cloud.ply ...")
    rgb_pcd = o3d.io.read_point_cloud(f"{sem_dir}/rgb_cloud.ply")
    rgb_pts = np.asarray(rgb_pcd.points, np.float32)
    rgb_col = (np.asarray(rgb_pcd.colors) * 255).astype(np.uint8)
    print(f"[rgb] {len(rgb_pts):,} points")

    # 用质心 + 包围盒定 lookat 和 zoom
    c = sem_pts.mean(axis=0)
    rng = np.ptp(sem_pts, axis=0).max() / 2.0
    views = [
        ("persp", (0.5, -0.5, 0.7), c, (0, 0, 1), 0.55),
        ("top",   (0.0, 0.0, 1.0), c, (0, 1, 0), 0.55),
        ("side",  (1.0, 0.0, 0.1), c, (0, 0, 1), 0.55),
    ]

    render_pcd(sem_pts, sem_col, f"{args.out}/{args.scene}_semantic.png", views)
    render_pcd(rgb_pts, rgb_col, f"{args.out}/{args.scene}_rgb.png", views)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
