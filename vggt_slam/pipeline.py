#!/usr/bin/env python3
"""
VGGT-SLAM → 3DGS Clean Pipeline
================================
Four-stage pipeline with visualisation at every step:

  Stage 1 — VGGT-SLAM:   submaps → pose graph → optimised homographies
  Stage 2 — Aggregate:   world-frame points + camera data
  Stage 3 — 3DGS Init:   Gaussian parameters from point cloud
  Stage 4 — 3DGS Train:  L1+SSIM, densify, prune, joint BA

Usage:
  python vggt_slam/pipeline.py --image_dir datasets/416 --stage 1       # SLAM only
  python vggt_slam/pipeline.py --image_dir ... --stage all --num_iters 3000
"""

import os, sys, glob, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "third_party", "perception_models"))
sys.path.insert(0, "/media/yml/share/v3dsam/diff-gaussian-rasterization-modified")

from vggt_slam.slam_utils import sort_images_by_number
from vggt_slam.solver import Solver
from vggt.models.vggt import VGGT


def _load_vggt_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGT()
    url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(url))
    model.eval()
    model = model.to(torch.bfloat16).to(device)
    print("VGGT model loaded.")
    return model


# =========================================================================
# Stage 1: VGGT-SLAM
# =========================================================================

def run_stage1(image_names, model, solver, submap_size=16, overlap=1,
               max_loops=1, lc_thres=0.95):
    """
    Run VGGT-SLAM on all images in overlapping submap batches.
    After each submap the pose graph is optimised and viser updated.
    """
    ordered = sort_images_by_number(image_names)
    print(f"  Stage 1: {len(ordered)} images, submap_size={submap_size}, overlap={overlap}")

    buf = []
    t0 = time.time()
    for name in ordered:
        buf.append(name)
        ready = len(buf) == submap_size + overlap
        last = name == ordered[-1]
        if ready or last:
            print(f"\n  Submap {len(solver.map.get_submaps())+1}: {len(buf)} frames")
            preds = solver.run_predictions(buf, model, max_loops, None, None)
            solver.add_points(preds)
            solver.graph.optimize()
            solver.update_all_submap_vis()

            # ── Overlap check: compare world-frame points from the
            # overlapping FRAME only (same image, different batches) ──
            non_lc = [s for s in solver.map.ordered_submaps_by_key() if not s.get_lc_status()]
            if len(non_lc) >= 2:
                prev = non_lc[-2]
                curr = non_lc[-1]
                # Get world-frame points for the specific overlapping frames
                prev_overlap_idx = prev.get_last_non_loop_frame_index()
                curr_overlap_idx = 0
                prev_homo = solver.graph.get_homography(int(prev.get_id() + prev_overlap_idx))
                curr_homo = solver.graph.get_homography(int(curr.get_id() + curr_overlap_idx))
                prev_pts_cam = prev.get_frame_pointcloud(prev_overlap_idx).reshape(-1, 3)
                curr_pts_cam = curr.get_frame_pointcloud(curr_overlap_idx).reshape(-1, 3)
                # Transform to world frame
                pw_prev = (prev_homo[:3,:3] @ prev_pts_cam.T).T + prev_homo[:3,3]
                pw_curr = (curr_homo[:3,:3] @ curr_pts_cam.T).T + curr_homo[:3,3]
                # Compare
                n = min(2000, len(pw_prev), len(pw_curr))
                c_prev = pw_prev[:n].mean(axis=0)
                c_curr = pw_curr[:n].mean(axis=0)
                delta = c_curr - c_prev
                dist = np.linalg.norm(delta)
                # Also check camera pose at overlap
                t_prev = prev_homo[:3, 3]
                t_curr = curr_homo[:3, 3]
                cam_delta = np.linalg.norm(t_curr - t_prev)
                print(f"  [OVERLAP] node {int(prev.get_id()+prev_overlap_idx)}<->"
                      f"{int(curr.get_id()+curr_overlap_idx)}  "
                      f"point_Δ={dist:.3f}m  cam_Δ={cam_delta:.3f}m")

            buf = buf[-overlap:] if overlap > 0 else []

    print(f"  Stage 1 done in {time.time()-t0:.1f}s  "
          f"({solver.map.get_num_submaps()} submaps, "
          f"{solver.graph.get_num_loops()} loops)")
    return solver


# =========================================================================
# Stage 2: Aggregate
# =========================================================================

def run_stage2(solver, image_dir, solver_viewer, target_points=120000):
    """Aggregate world-frame points and camera data; show in viser."""
    print("\n" + "=" * 60)
    print("  Stage 2: Aggregation")
    print("=" * 60)
    from vggt_slam.stage2_aggregate import (
        aggregate_world_points, aggregate_camera_data)

    points, colors = aggregate_world_points(solver, target_points=target_points)
    cam_data = aggregate_camera_data(solver, image_dir)

    # Viser: unified world-frame point cloud
    if solver_viewer is not None:
        solver_viewer.server.scene.add_point_cloud(
            name="pcd_unified",
            points=points,
            colors=(colors * 255).astype(np.uint8),
            point_size=0.002,
            point_shape="circle",
        )
        # Trajectory line
        c2w = np.linalg.inv(cam_data["w2c"])
        traj = c2w[:, :3, 3]
        lines = np.stack([traj[:-1], traj[1:]], axis=1)
        solver_viewer.server.scene.add_line_segments(
            name="trajectory", points=lines,
            colors=(0, 255, 0), line_width=2.0,
        )
        print("  [viser] Unified point cloud + trajectory added")

    print(f"  Points: {points.shape[0]:,},  Frames: {len(cam_data['images'])}")
    return points, colors, cam_data


# =========================================================================
# Stage 3: 3DGS Init
# =========================================================================

def run_stage3(points, colors, cam_data, solver_viewer):
    """Initialise 3DGS from aggregated data; show Gaussian centers in viser."""
    print("\n" + "=" * 60)
    print("  Stage 3: 3DGS Initialization")
    print("=" * 60)
    from vggt_slam.stage3_gs_init import (
        initialize_params, init_camera_poses_from_slam, setup_optimizer)

    M = len(cam_data["images"])
    params, variables = initialize_params(points, colors, M)
    params = init_camera_poses_from_slam(params, cam_data["w2c"])
    optimizer = setup_optimizer(params)

    # Viser: Gaussian centers
    if solver_viewer is not None:
        pts = params["means3D"].detach().cpu().numpy()
        cols = params["rgb_colors"].detach().cpu().numpy()
        solver_viewer.server.scene.add_point_cloud(
            name="gs_init",
            points=pts,
            colors=(np.clip(cols, 0, 1) * 255).astype(np.uint8),
            point_size=0.003,
            point_shape="circle",
        )
        print(f"  [viser] {pts.shape[0]:,} initial Gaussians displayed")

    print(f"  {points.shape[0]:,} Gaussians, {M} frames")
    return params, variables, optimizer


# =========================================================================
# Stage 4: 3DGS Train
# =========================================================================

def run_stage4(params, variables, optimizer, cam_data,
               num_iters=3000, solver_viewer=None):
    """Train 3DGS; save renders; optionally show final Open3D view."""
    print("\n" + "=" * 60)
    print(f"  Stage 4: 3DGS Training ({num_iters} iters)")
    print("=" * 60)
    from vggt_slam.stage4_gs_train import train

    params, variables = train(params, variables, optimizer, cam_data,
                              num_iters=num_iters)

    # Viser: update with final Gaussians
    if solver_viewer is not None:
        pts = params["means3D"].detach().cpu().numpy()
        cols = params["rgb_colors"].detach().cpu().numpy()
        solver_viewer.server.scene.add_point_cloud(
            name="gs_final",
            points=pts,
            colors=(np.clip(cols, 0, 1) * 255).astype(np.uint8),
            point_size=0.002,
            point_shape="circle",
        )
        print("  [viser] Final Gaussians displayed")

    return params, variables


# =========================================================================
# Full pipeline driver
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="VGGT-SLAM → 3DGS Clean Pipeline")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--stage", type=str, default="all",
                        choices=["1", "2", "3", "4", "all"])
    parser.add_argument("--submap_size", type=int, default=16)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument("--max_loops", type=int, default=1)
    parser.add_argument("--num_iters", type=int, default=3000)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--target_points", type=int, default=120000)
    parser.add_argument("--output_dir", type=str, default="output_pipeline")
    parser.add_argument("--no_vis", action="store_true",
                        help="Disable Open3D viewer")
    parser.add_argument("--viser_port", type=int, default=8080)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- gather images ----
    image_names = sorted(glob.glob(os.path.join(args.image_dir, "*")))
    image_names = [f for f in image_names
                   if os.path.isfile(f) and f.lower().endswith(('.jpg','.jpeg','.png'))
                   and "depth" not in os.path.basename(f).lower()]
    if not image_names:
        raise FileNotFoundError(f"No images found in {args.image_dir}")
    if args.max_images > 0:
        image_names = image_names[:args.max_images]
        print(f"[FAST MODE] Using first {len(image_names)} images")
    print(f"Found {len(image_names)} images")

    # ---- load VGGT ----
    model = _load_vggt_model()

    # ---- Stage 1 ----
    solver = Solver(init_conf_threshold=25.0, lc_thres=0.95)
    solver = run_stage1(image_names, model, solver,
                        submap_size=args.submap_size,
                        overlap=args.overlap,
                        max_loops=args.max_loops)

    if args.stage == "1":
        print("\nStage 1 complete. Viser running at http://localhost:8080")
        print("Press Ctrl+C to stop.")
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            return

    # ---- Stage 2 ----
    points, colors, cam_data = run_stage2(solver, args.image_dir,
                                          solver.viewer,
                                          target_points=args.target_points)

    if args.stage == "2":
        print("\nStage 2 complete.")
        return

    # ---- Stage 3 ----
    params, variables, optimizer = run_stage3(points, colors, cam_data,
                                              solver.viewer)

    if args.stage == "3":
        print("\nStage 3 complete.")
        return

    # ---- Stage 4 ----
    params, variables = run_stage4(params, variables, optimizer, cam_data,
                                   num_iters=args.num_iters,
                                   solver_viewer=solver.viewer)

    # ---- Save ----
    print(f"\nSaving to {args.output_dir}/ ...")
    np.savez(os.path.join(args.output_dir, "gaussians.npz"),
             means3D=params["means3D"].detach().cpu().numpy(),
             rgb_colors=params["rgb_colors"].detach().cpu().numpy(),
             unnorm_rotations=params["unnorm_rotations"].detach().cpu().numpy(),
             logit_opacities=params["logit_opacities"].detach().cpu().numpy(),
             log_scales=params["log_scales"].detach().cpu().numpy())
    print(f"Done. Results saved to {args.output_dir}/")
    print("Viser running at http://localhost:8080 — press Ctrl+C to stop.")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
