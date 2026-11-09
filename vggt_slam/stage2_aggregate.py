#!/usr/bin/env python3
"""
Stage 2: Aggregate world-frame points and camera data from the optimized
VGGT-SLAM pose graph.  Reads from the Solver's graph and map; outputs
a single point cloud and per-frame camera data ready for 3DGS.
"""

import os
import re
import numpy as np
from PIL import Image


def aggregate_world_points(solver, target_points=120000, voxel_size=0.02):
    """
    Collect world-frame points from all non-loop-closure submaps.
    Returns (N,3) points, (N,3) colors in [0,1].
    """
    all_pts, all_col = [], []
    for submap in solver.map.ordered_submaps_by_key():
        if submap.get_lc_status():
            continue
        pts = submap.get_points_in_world_frame(solver.graph)
        col = submap.get_points_colors()
        if col.max() > 1.0:
            col = col / 255.0
        all_pts.append(pts)
        all_col.append(col)
    all_pts = np.concatenate(all_pts, axis=0)
    all_col = np.concatenate(all_col, axis=0)
    print(f"  Aggregated {len(all_pts):,} world-frame points")

    if len(all_pts) > target_points:
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(all_pts)
            pcd.colors = o3d.utility.Vector3dVector(all_col)
            pcd = pcd.voxel_down_sample(voxel_size)
            all_pts = np.asarray(pcd.points)
            all_col = np.asarray(pcd.colors)
            if len(all_pts) > target_points:
                idx = np.random.choice(len(all_pts), target_points, replace=False)
                all_pts, all_col = all_pts[idx], all_col[idx]
            print(f"  Downsampled to {len(all_pts):,} points (voxel={voxel_size}m)")
        except ImportError:
            idx = np.random.choice(len(all_pts),
                                   min(target_points, len(all_pts)), replace=False)
            all_pts, all_col = all_pts[idx], all_col[idx]
            print(f"  Random subsample to {len(all_pts):,} points")
    return all_pts, all_col


def extract_frame_number(name):
    m = re.search(r'\d+', os.path.basename(name))
    return int(m.group()) if m else 0


def aggregate_camera_data(solver, image_dir, max_dim=640):
    """
    Collect per-frame camera data: w2c poses, intrinsics K, and images.
    Returns dict with keys: images (list of np), w2c (M,4,4), K (M,3,3),
                             paths, width, height.
    """
    from vggt_slam.slam_utils import decompose_camera

    # Get all camera-to-world poses from the optimized graph
    all_c2w = solver.map.get_all_cam_matricies(solver.graph, give_camera_mat=False)
    all_proj = solver.map.get_all_cam_matricies(solver.graph, give_camera_mat=True)

    # Gather image paths sorted by frame number
    img_paths = sorted(
        [os.path.join(image_dir, f) for f in os.listdir(image_dir)
         if os.path.isfile(os.path.join(image_dir, f))
         and f.lower().endswith(('.jpg', '.jpeg', '.png'))
         and 'depth' not in f.lower()],
        key=extract_frame_number)

    # Detect VGGT internal resolution from first submap's frames
    vggt_H, vggt_W = 392, 518  # default
    for submap in solver.map.ordered_submaps_by_key():
        if not submap.get_lc_status():
            frames = submap.get_all_frames()
            if frames is not None and hasattr(frames, 'shape') and len(frames.shape) >= 4:
                _, _, vggt_H, vggt_W = frames.shape
            break

    images, w2c_list, K_list, paths, indices = [], [], [], [], []
    count = 0
    K_scale_W, K_scale_H = None, None
    img_W, img_H = None, None

    for submap in solver.map.ordered_submaps_by_key():
        if submap.get_lc_status():
            continue
        frame_ids = submap.get_frame_ids()
        for fi, fid in enumerate(frame_ids):
            if count >= len(all_c2w):
                break
            c2w = all_c2w[count]
            proj = all_proj[count]

            K_vggt, _, _, _ = decompose_camera(proj)

            fid_int = int(fid)
            if fid_int < len(img_paths):
                path = img_paths[fid_int]
            else:
                match = [p for p in img_paths if extract_frame_number(p) == fid_int]
                path = match[0] if match else None
            if path is None:
                count += 1
                continue

            try:
                img = Image.open(path).convert("RGB")
                img_np = np.array(img)
                h, w = img_np.shape[:2]
                if max(h, w) > max_dim:
                    scl = max_dim / max(h, w)
                    img_np = np.array(Image.fromarray(img_np).resize(
                        (int(w * scl), int(h * scl)), Image.BILINEAR))
            except Exception:
                count += 1
                continue

            if K_scale_W is None:
                img_H, img_W = img_np.shape[:2]
                K_scale_W = img_W / vggt_W
                K_scale_H = img_H / vggt_H
                print(f"  Target resolution: {img_W}x{img_H}, "
                      f"K scale: W={K_scale_W:.3f} H={K_scale_H:.3f}")

            K = K_vggt.copy()
            K[0, 0] *= K_scale_W
            K[0, 2] *= K_scale_W
            K[1, 1] *= K_scale_H
            K[1, 2] *= K_scale_H

            w2c = np.linalg.inv(c2w)
            images.append(img_np)
            w2c_list.append(w2c)
            K_list.append(K)
            paths.append(path)
            indices.append(fid_int)
            count += 1

    print(f"  Collected {len(images)} frames ({img_W}x{img_H})")
    return {
        "images": images,
        "w2c": np.array(w2c_list),
        "K": np.array(K_list),
        "paths": paths,
        "indices": indices,
        "width": img_W,
        "height": img_H,
    }
