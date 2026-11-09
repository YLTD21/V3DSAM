#!/usr/bin/env python3
"""
3DGS Interactive Viewer — mouse orbit/zoom, real-time 3DGS rendering.
Same approach as SGS-SLAM tk_recon: render 3DGS → RGBD → point cloud → Open3D window.

Usage:
  conda activate sgs
  python tools/view_3dgs.py --gaussians out_v9_final/gaussians.npz
"""
import os, sys, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/media/yml/share/v3dsam/diff-gaussian-rasterization-modified")
os.environ['XFORMERS_DISABLED'] = '1'

import torch, torch.nn.functional as F
from diff_gaussian_rasterization import GaussianRasterizer as Renderer, GaussianRasterizationSettings as Camera
import open3d as o3d

device = "cuda"

def load_gaussians(path):
    d = np.load(path, allow_pickle=True)
    params = {}
    for k in ['means3D', 'rgb_colors', 'log_scales', 'unnorm_rotations', 'logit_opacities']:
        params[k] = torch.tensor(d[k], dtype=torch.float32, device=device)
    if 'semantic_ids' in d:
        params['semantic_ids'] = d['semantic_ids']
    print(f"Loaded {params['means3D'].shape[0]:,} Gaussians")
    return params

def setup_camera(w, h, k, w2c):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c_t = torch.tensor(w2c).to(device).float()
    cam_center = torch.inverse(w2c_t)[:3, 3]
    w2c_t = w2c_t.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([
        [2*fx/w, 0.0, -(w-2*cx)/w, 0.0],
        [0.0, 2*fy/h, -(h-2*cy)/h, 0.0],
        [0.0, 0.0, 100/(100-0.01), -(100*0.01)/(100-0.01)],
        [0.0, 0.0, 1.0, 0.0]
    ]).to(device).float().unsqueeze(0).transpose(1, 2)
    return Camera(image_height=h, image_width=w, tanfovx=w/(2*fx), tanfovy=h/(2*fy),
                  bg=torch.tensor([0,0,0], dtype=torch.float32, device=device),
                  scale_modifier=1.0, viewmatrix=w2c_t, projmatrix=w2c_t.bmm(opengl_proj),
                  sh_degree=0, campos=cam_center, prefiltered=False, debug=False)

def render_to_rgbd(params, w2c, K, W, H):
    """Render 3DGS → RGB image + depth map."""
    with torch.no_grad():
        cam = setup_camera(W, H, K, w2c)
        # RGB
        rv_rgb = {
            'means3D': params['means3D'], 'colors_precomp': params['rgb_colors'],
            'rotations': F.normalize(params['unnorm_rotations']),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
            'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device=device) + 0,
        }
        im, radius = Renderer(raster_settings=cam)(**rv_rgb)

        # Depth (render each Gaussian's z-depth)
        pts4 = torch.cat((params['means3D'], torch.ones(params['means3D'].shape[0], 1, device=device)), dim=-1)
        w2c_t = torch.tensor(w2c, device=device).float()
        pts_cam = (w2c_t @ pts4.T).T
        depth_z = pts_cam[:, 2].unsqueeze(-1)
        depth_sil = torch.zeros((params['means3D'].shape[0], 3), device=device).float()
        depth_sil[:, 0] = depth_z.squeeze(-1)
        depth_sil[:, 1] = 1.0
        depth_sil[:, 2] = torch.square(depth_z).squeeze(-1)
        rv_depth = {
            'means3D': params['means3D'], 'colors_precomp': depth_sil,
            'rotations': F.normalize(params['unnorm_rotations']),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
            'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device=device) + 0,
        }
        depth_im, _ = Renderer(raster_settings=cam)(**rv_depth)
        depth = depth_im[0, :, :]  # (H, W)

    return im.permute(1,2,0).cpu().numpy().clip(0,1), depth.cpu().numpy()

def rgbd_to_pointcloud(color_np, depth_np, w2c, K):
    """Back-project RGBD → world-frame point cloud."""
    H, W = depth_np.shape
    fx, fy = K[0,0], K[1,1]; cx, cy = K[0,2], K[1,2]
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    xx = (xx - cx) / fx; yy = (yy - cy) / fy
    z = depth_np.reshape(-1)
    pts_cam = np.stack((xx.reshape(-1)*z, yy.reshape(-1)*z, z), axis=-1)
    c2w = np.linalg.inv(w2c)
    pts4 = np.concatenate((pts_cam, np.ones((pts_cam.shape[0], 1))), axis=1)
    pts_world = (c2w @ pts4.T).T[:, :3]
    colors = color_np.reshape(-1, 3)
    valid = (z > 0.01) & (z < 50) & np.all(np.isfinite(pts_world), axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world[valid].astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors[valid].astype(np.float64))
    return pcd

# ════════════════════ Main ════════════════════
parser = argparse.ArgumentParser()
parser.add_argument("--gaussians", required=True)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=720)
args = parser.parse_args()

params = load_gaussians(args.gaussians)

# Estimate intrinsics from scene bounds
means_np = params['means3D'].cpu().numpy()
center = means_np.mean(0)
extent = np.linalg.norm(means_np.max(0) - means_np.min(0))
focal = args.width / 2.0
K = np.array([[focal, 0, args.width/2], [0, focal, args.height/2], [0, 0, 1]], dtype=np.float64)

# Start camera looking at scene center from the front
cam_pos = center + np.array([0, 0, extent * 1.2])
forward = center - cam_pos; forward /= np.linalg.norm(forward)
up = np.array([0, -1, 0])
right = np.cross(forward, up); right /= np.linalg.norm(right)
up = np.cross(right, forward)
c2w = np.eye(4)
c2w[:3,0]=right; c2w[:3,1]=up; c2w[:3,2]=forward; c2w[:3,3]=cam_pos
init_w2c = np.linalg.inv(c2w)

# Open3D window
vis = o3d.visualization.Visualizer()
vis.create_window(width=args.width, height=args.height, visible=True, window_name="3DGS Viewer — Mouse to orbit/zoom  |  Q=quit")

im_np, depth_np = render_to_rgbd(params, init_w2c, K, args.width, args.height)
pcd = rgbd_to_pointcloud(im_np, depth_np, init_w2c, K)
vis.add_geometry(pcd)

vc = vis.get_view_control()
cparams = o3d.camera.PinholeCameraParameters()
cparams.extrinsic = init_w2c
cparams.intrinsic.intrinsic_matrix = K
cparams.intrinsic.height = args.height
cparams.intrinsic.width = args.width
vc.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)
vis.get_render_option().point_size = 1.0
vis.get_render_option().background_color = np.array([0.05, 0.05, 0.05])

last_w2c = init_w2c.copy()
print(f"Viewer ready. Mouse: orbit/zoom. Press Q in window to quit.")

# Main loop
while True:
    cam_p = vc.convert_to_pinhole_camera_parameters()
    curr_w2c = np.array(cam_p.extrinsic)
    if not np.allclose(curr_w2c, last_w2c, atol=1e-3):
        im_np, depth_np = render_to_rgbd(params, curr_w2c, K, args.width, args.height)
        new_pcd = rgbd_to_pointcloud(im_np, depth_np, curr_w2c, K)
        pcd.points = new_pcd.points
        pcd.colors = new_pcd.colors
        vis.update_geometry(pcd)
        last_w2c = curr_w2c.copy()
    if not vis.poll_events():
        break
    vis.update_renderer()

vis.destroy_window()
print("Done.")
