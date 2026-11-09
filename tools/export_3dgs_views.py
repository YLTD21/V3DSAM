#!/usr/bin/env python3
"""
Export trained 3DGS to multiple visualization formats + interactive viser viewer.

Three methods:
  1. PLY export → SIBR viewer, MeshLab, CloudCompare, etc.
  2. .splat export → antimatter15 web viewer (https://antimatter15.com/splat/)
  3. .ply_compressed → Supersplat web viewer (https://playcanvas.com/supersplat/editor)
  4. Interactive viser viewer (renders 3DGS in real-time via web browser)

Usage:
  python tools/export_3dgs_views.py --gaussians out_v9_final/gaussians.npz
  python tools/export_3dgs_views.py --gaussians out_v9_final/gaussians.npz --no-viewer
"""
import os, sys, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/media/yml/share/v3dsam/diff-gaussian-rasterization-modified")
os.environ['XFORMERS_DISABLED'] = '1'

import torch, torch.nn.functional as F
from diff_gaussian_rasterization import GaussianRasterizer as Renderer, GaussianRasterizationSettings as Camera
from PIL import Image
import time

def load_gaussians(path, device="cuda"):
    d = np.load(path)
    params = {}
    for k in ['means3D', 'rgb_colors', 'log_scales', 'unnorm_rotations', 'logit_opacities']:
        if k in d:
            params[k] = torch.tensor(d[k], dtype=torch.float32, device=device)
    if 'semantic_ids' in d:
        params['semantic_ids'] = d['semantic_ids']  # keep as numpy
    print(f"Loaded {params['means3D'].shape[0]:,} Gaussians")
    return params

# ═══════════════════════════════════════════════════════════════════════
# Method 1-3: Export to gsplat formats
# ═══════════════════════════════════════════════════════════════════════
def export_all_formats(params, output_dir):
    """导出 3 种格式. gsplat 期望 LOG 尺度 + LOGIT 不透明度 + SH 系数."""
    from gsplat import export_splats
    os.makedirs(output_dir, exist_ok=True)

    SH_C0 = 0.28209479177387814

    with torch.no_grad():
        means = params['means3D']                              # (N, 3)
        scales = params['log_scales']                          # (N,3) LOG 空间, 已是逐轴
        quats = F.normalize(params['unnorm_rotations'])        # (N, 4) 单位四元数
        opacities = params['logit_opacities'].squeeze(-1)      # (N,)  LOGIT 空间
        # NPZ 存的是 RGB colors (0-1), 需转回 SH DC 系数
        rgb = params['rgb_colors']                             # (N, 3)
        sh0 = ((rgb - 0.5) / SH_C0).unsqueeze(1)               # (N, 1, 3) SH DC
        shN = torch.zeros(means.shape[0], 0, 3, device=means.device)

        # 1. PLY (INRIA 标准, SIBR / MeshLab / CloudCompare / SuperSplat)
        path_ply = os.path.join(output_dir, "gaussians.ply")
        export_splats(means, scales, quats, opacities, sh0, shN, format='ply', save_to=path_ply)
        size_mb = os.path.getsize(path_ply) / 1e6
        print(f"  ✅ PLY:  {path_ply} ({size_mb:.1f} MB)")
        print(f"     → Use with SIBR viewer, MeshLab, CloudCompare, SuperSplat")

        # 2. .splat (antimatter15 web viewer)
        path_splat = os.path.join(output_dir, "gaussians.splat")
        export_splats(means, scales, quats, opacities, sh0, shN, format='splat', save_to=path_splat)
        size_mb = os.path.getsize(path_splat) / 1e6
        print(f"  ✅ .splat: {path_splat} ({size_mb:.1f} MB)")
        print(f"     → Drag into https://antimatter15.com/splat/")

        # 3. .ply_compressed (SuperSplat 原生压缩格式, 推荐)
        path_compressed = os.path.join(output_dir, "gaussians_compressed.ply")
        export_splats(means, scales, quats, opacities, sh0, shN, format='ply_compressed', save_to=path_compressed)
        size_mb = os.path.getsize(path_compressed) / 1e6
        print(f"  ✅ compressed PLY: {path_compressed} ({size_mb:.1f} MB)")
        print(f"     → Drag into https://playcanvas.com/supersplat/editor (推荐)")

# ═══════════════════════════════════════════════════════════════════════
# Method 4: Interactive viser viewer (renders 3DGS to viser image panel)
# ═══════════════════════════════════════════════════════════════════════
def launch_viser_viewer(params, img_W=640, img_H=480):
    """Launch interactive viser-based 3DGS viewer.
    Uses viser for camera control → renders 3DGS on GPU → displays in viser panel.
    """
    try:
        import viser
    except ImportError:
        print("  ⚠️  viser not installed. Install with: pip install viser")
        print("  → Use the exported PLY/.splat files with web viewers instead.")
        return

    print(f"\n  Launching viser interactive viewer...")
    server = viser.ViserServer(port=8082)

    # Add a GUI panel with controls
    with server.gui.add_folder("3DGS Viewer Controls"):
        mode_btn = server.gui.add_button("Toggle RGB/Semantic")
        screenshot_btn = server.gui.add_button("Screenshot")

    # Create a mesh/image panel to display rendered result
    render_handle = server.scene.add_image(
        name="rendered_view",
        image=np.zeros((img_H, img_W, 3), dtype=np.uint8),
        render_width=img_W,
        render_height=img_H,
        position=(0, 0, 0),
        w=1.0, h=img_H/max(img_W, img_H),
    )

    # Compute scene bounds for camera setup
    means_np = params['means3D'].cpu().numpy()
    center = means_np.mean(0)
    extent = np.linalg.norm(means_np.max(0) - means_np.min(0))

    # Setup camera
    server.scene.set_up_direction((0, -1, 0))
    server.scene.add_frame("origin", w=extent*0.1, position=center.tolist())

    # Camera state
    cam_fov = 60.0
    cam_radius = extent * 0.8
    cam_theta, cam_phi = 0.0, np.pi/4
    view_mode = [0]  # 0=RGB, 1=Semantic
    frame_count = [0]

    # Semantic colors
    sem_colors_gpu = None
    cat_names = []
    if 'semantic_ids' in params:
        # Try to load category info
        cat_names = params.get('_sem_cat_names', [])
        if len(cat_names) > 0:
            PAL = np.zeros((len(cat_names), 3), dtype=np.float32)
            from vggt_slam.v9_full_pipeline import CAT_COLORS, DEFAULT_COLOR
            for ci, cn in enumerate(cat_names):
                PAL[ci] = np.array(CAT_COLORS.get(cn, DEFAULT_COLOR), dtype=np.float32) / 255.0
            sem_colors_gpu = torch.tensor(PAL, device=params['means3D'].device)

    def get_camera_w2c():
        """Build w2c from current orbital parameters."""
        cx = center[0] + cam_radius * np.cos(cam_phi) * np.sin(cam_theta)
        cy = center[1] + cam_radius * np.sin(cam_phi)
        cz = center[2] + cam_radius * np.cos(cam_phi) * np.cos(cam_theta)
        cam_pos = np.array([cx, cy, cz])
        forward = center - cam_pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        up = np.array([0, -1, 0])
        right = np.cross(forward, up); right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward); up = up / (np.linalg.norm(up) + 1e-8)
        c2w = np.eye(4)
        c2w[:3, 0] = right; c2w[:3, 1] = up; c2w[:3, 2] = forward; c2w[:3, 3] = cam_pos
        return np.linalg.inv(c2w)

    def setup_camera(w, h, k, w2c, near=0.01, far=100, device="cuda"):
        fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
        w2c_t = torch.tensor(w2c).to(device).float()
        cam_center_pt = torch.inverse(w2c_t)[:3, 3]
        w2c_t = w2c_t.unsqueeze(0).transpose(1, 2)
        opengl_proj = torch.tensor([
            [2*fx/w, 0.0, -(w-2*cx)/w, 0.0], [0.0, 2*fy/h, -(h-2*cy)/h, 0.0],
            [0.0, 0.0, far/(far-near), -(far*near)/(far-near)], [0.0, 0.0, 1.0, 0.0]
        ]).to(device).float().unsqueeze(0).transpose(1, 2)
        return Camera(image_height=h, image_width=w, tanfovx=w/(2*fx), tanfovy=h/(2*fy),
                      bg=torch.tensor([0,0,0], dtype=torch.float32, device=device),
                      scale_modifier=1.0, viewmatrix=w2c_t, projmatrix=w2c_t.bmm(opengl_proj),
                      sh_degree=0, campos=cam_center_pt, prefiltered=False, debug=False)

    def render_frame(w2c, mode="rgb"):
        fx = img_W / (2 * np.tan(np.radians(cam_fov) / 2))
        fy = img_H / (2 * np.tan(np.radians(cam_fov) / 2))
        K = np.array([[fx, 0, img_W/2], [0, fy, img_H/2], [0, 0, 1]], dtype=np.float64)

        device = params['means3D'].device
        if mode == "semantic" and sem_colors_gpu is not None:
            sids = torch.tensor(params['semantic_ids'], device=device).long()
            valid = (sids >= 0) & (sids < len(cat_names))
            colors = torch.zeros((params['means3D'].shape[0], 3), dtype=torch.float32, device=device)
            if valid.any():
                colors[valid] = sem_colors_gpu[sids[valid]]
            rv = {
                'means3D': params['means3D'], 'colors_precomp': colors,
                'rotations': F.normalize(params['unnorm_rotations']),
                'opacities': torch.sigmoid(params['logit_opacities']),
                'scales': torch.exp(params['log_scales']),  # log_scales 已是 (N,3), exp→物理尺度
                'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device=device) + 0,
            }
        else:
            rv = {
                'means3D': params['means3D'], 'colors_precomp': params['rgb_colors'],
                'rotations': F.normalize(params['unnorm_rotations']),
                'opacities': torch.sigmoid(params['logit_opacities']),
                'scales': torch.exp(params['log_scales']),  # log_scales 已是 (N,3), exp→物理尺度
                'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device=device) + 0,
            }
        cam = setup_camera(img_W, img_H, K, w2c)
        with torch.no_grad():
            im, _ = Renderer(raster_settings=cam)(**rv)
        return (im.permute(1,2,0).cpu().numpy().clip(0,1) * 255).astype(np.uint8)

    # Add point cloud visualization in viser for context
    pts_sample = params['means3D'][::20].cpu().numpy()  # subsample
    colors_sample = np.clip(params['rgb_colors'][::20].cpu().numpy(), 0, 1)  # RGB
    server.scene.add_point_cloud(
        name="gaussian_centers",
        points=pts_sample,
        colors=colors_sample,
        point_size=0.005,
    )

    @mode_btn.on_click
    def _(event):
        view_mode[0] = 1 - view_mode[0]
        mode_name = "Semantic" if view_mode[0] == 1 else "RGB"
        print(f"  Mode: {mode_name}")

    @screenshot_btn.on_click
    def _(event):
        w2c = get_camera_w2c()
        img = render_frame(w2c, "semantic" if view_mode[0] == 1 else "rgb")
        path = f"vis_screenshot_{frame_count[0]:04d}.png"
        Image.fromarray(img).save(path)
        print(f"  Screenshot: {path}")

    print(f"  ✅ Viser viewer at http://localhost:8082")
    print(f"     Controls: GUI panel on the left")
    print(f"     The 3D view shows Gaussian centers (subsampled)")
    print(f"     Press Ctrl+C to stop")

    # Main loop: periodically update the rendered image based on viser camera
    def update_loop():
        while True:
            try:
                # Get camera from viser
                viser_cam = server.camera
                if hasattr(viser_cam, 'wxyz') and hasattr(viser_cam, 'position'):
                    # Convert viser quaternion (wxyz) + position to w2c
                    q = np.array(viser_cam.wxyz)  # (w, x, y, z)
                    pos = np.array(viser_cam.position)
                    # Build rotation from quaternion
                    w, x, y, z = q
                    R = np.array([
                        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
                    ])
                    c2w = np.eye(4)
                    c2w[:3, :3] = R
                    c2w[:3, 3] = pos
                    w2c = np.linalg.inv(c2w)

                    img_arr = render_frame(w2c, "semantic" if view_mode[0] == 1 else "rgb")
                    render_handle.image = img_arr
                    frame_count[0] += 1
            except Exception:
                pass
            time.sleep(0.05)  # 20 FPS

    try:
        update_loop()
    except KeyboardInterrupt:
        print("  Viewer stopped.")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export 3DGS to multiple visualization formats")
    parser.add_argument("--gaussians", required=True, help="Path to gaussians.npz")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: same dir as gaussians)")
    parser.add_argument("--no_viewer", action="store_true", help="Skip interactive viewer")
    args = parser.parse_args()

    gs_dir = args.output_dir or os.path.dirname(os.path.abspath(args.gaussians))
    params = load_gaussians(args.gaussians)

    print(f"\n{'='*60}")
    print(f"  Exporting 3DGS to multiple formats...")
    print(f"{'='*60}\n")
    export_all_formats(params, gs_dir)

    if not args.no_viewer:
        launch_viser_viewer(params)
    else:
        print(f"\n  Skipping viewer. To launch interactively:")
        print(f"    python tools/export_3dgs_views.py --gaussians {args.gaussians}")
