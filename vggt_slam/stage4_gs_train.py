#!/usr/bin/env python3
"""
Stage 4: Minimal 3DGS training loop — L1+SSIM loss, densify, prune,
joint camera optimization.  No semantics, no depth supervision, no
cosine LR, no memory streaming.
"""

import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys, os
sys.path.insert(0, "/media/yml/share/v3dsam/diff-gaussian-rasterization-modified")

from diff_gaussian_rasterization import (
    GaussianRasterizer as Renderer,
    GaussianRasterizationSettings as Camera,
)
from vggt_slam.stage3_gs_init import params2rendervar


# ---------------------------------------------------------------------------
# Camera builder (same as original, uses TRAINABLE camera params)
# ---------------------------------------------------------------------------

def build_camera(params, frame_idx, w, h, K_np, near=0.01, far=100.0):
    device = "cuda"
    r = params["cam_unnorm_rots"].squeeze(0)  # (4, M)
    t = params["cam_trans"].squeeze(0)         # (3, M)
    rot_q = F.normalize(r[:, frame_idx], dim=0)
    trans = t[:, frame_idx]
    wv, x, y, z = rot_q[0], rot_q[1], rot_q[2], rot_q[3]

    R = torch.zeros((3, 3), device=device, dtype=torch.float32)
    R[0,0]=1-2*(y*y+z*z); R[0,1]=2*(x*y-wv*z); R[0,2]=2*(x*z+wv*y)
    R[1,0]=2*(x*y+wv*z); R[1,1]=1-2*(x*x+z*z); R[1,2]=2*(y*z-wv*x)
    R[2,0]=2*(x*z-wv*y); R[2,1]=2*(y*z+wv*x); R[2,2]=1-2*(x*x+y*y)

    w2c = torch.eye(4, device=device, dtype=torch.float32)
    w2c[:3, :3] = R
    w2c[:3, 3] = trans
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c_t = w2c.unsqueeze(0).transpose(1, 2)

    fx, fy, cx, cy = K_np[0,0], K_np[1,1], K_np[0,2], K_np[1,2]
    proj = torch.tensor([
        [2*fx/w, 0.0, -(w-2*cx)/w, 0.0],
        [0.0, 2*fy/h, -(h-2*cy)/h, 0.0],
        [0.0, 0.0, far/(far-near), -(far*near)/(far-near)],
        [0.0, 0.0, 1.0, 0.0],
    ], device=device, dtype=torch.float32).unsqueeze(0).transpose(1, 2)

    return Camera(
        image_height=h, image_width=w,
        tanfovx=w/(2*fx), tanfovy=h/(2*fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device=device),
        scale_modifier=1.0,
        viewmatrix=w2c_t, projmatrix=w2c_t.bmm(proj),
        sh_degree=0, campos=cam_center,
        prefiltered=False, debug=False,
    )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def ssim(img1, img2, window_size=11):
    C1, C2 = 0.01**2, 0.03**2
    mu1 = F.avg_pool2d(img1, window_size, 1, 0)
    mu2 = F.avg_pool2d(img2, window_size, 1, 0)
    s1 = F.avg_pool2d(img1**2, window_size, 1, 0) - mu1**2
    s2 = F.avg_pool2d(img2**2, window_size, 1, 0) - mu2**2
    s12 = F.avg_pool2d(img1*img2, window_size, 1, 0) - mu1*mu2
    return ((2*mu1*mu2 + C1)*(2*s12 + C2) /
            ((mu1**2 + mu2**2 + C1)*(s1 + s2 + C2))).mean()


def compute_loss(params, frame_data, variables):
    """Render one frame and return L1+SSIM loss."""
    rendervar = params2rendervar(params)
    cam = build_camera(params, frame_data["idx"], frame_data["w"],
                       frame_data["h"], frame_data["K"])
    rendervar["means2D"].retain_grad()
    im, radius = Renderer(raster_settings=cam)(**rendervar)
    variables["means2D"] = rendervar["means2D"]
    loss = 0.8 * F.l1_loss(im, frame_data["im"]) + \
           0.2 * (1.0 - ssim(im, frame_data["im"]))
    seen = radius > 0
    variables["max_2D_radius"][seen] = torch.max(
        radius[seen], variables["max_2D_radius"][seen])
    variables["seen"] = seen
    return loss, variables


# ---------------------------------------------------------------------------
# Densify & Prune
# ---------------------------------------------------------------------------

def densify(params, variables, optimizer, iteration, cfg):
    if iteration < cfg.get("from_iter", 500): return params, variables
    if iteration > cfg.get("until_iter", 3000): return params, variables
    if iteration % 100 != 0: return params, variables

    threshold = cfg.get("grad_thresh", 0.0002)
    grads = variables["means2D_gradient_accum"] / variables["denom"]
    grads[grads.isnan()] = 0.0
    selected = grads > threshold
    if selected.sum() == 0:
        return params, variables

    for key in ["means3D", "rgb_colors", "unnorm_rotations", "logit_opacities", "log_scales"]:
        if key in params:
            add = params[key][selected].detach()
            params[key] = torch.nn.Parameter(
                torch.cat([params[key], add], dim=0).requires_grad_(True))

    N = params["means3D"].shape[0]
    for vk in ["means2D_gradient_accum", "denom", "max_2D_radius"]:
        variables[vk] = torch.zeros(N, device="cuda").float()

    for grp in optimizer.param_groups:
        nm = grp["name"]
        if nm in params:
            old = grp["params"][0]
            if old in optimizer.state: del optimizer.state[old]
            grp["params"] = [params[nm]]
    return params, variables


def prune(params, variables, optimizer, iteration, cfg):
    if iteration < cfg.get("from_iter", 500): return params, variables
    if iteration % cfg.get("interval", 300) != 0: return params, variables

    with torch.no_grad():
        opa = torch.sigmoid(params["logit_opacities"]).squeeze(-1)
        scl = torch.exp(params["log_scales"]).max(dim=-1).values
        keep = (opa > cfg.get("opacity_thresh", 0.005)) & \
               (scl < cfg.get("max_scale_thresh", 0.1)) & \
               (variables["max_2D_radius"] < cfg.get("max_world_radius", 1e6))
        if keep.all():
            return params, variables

        for key in ["means3D", "rgb_colors", "unnorm_rotations", "logit_opacities", "log_scales"]:
            if key in params:
                params[key] = torch.nn.Parameter(
                    params[key][keep].detach().clone().requires_grad_(True))

        N = params["means3D"].shape[0]
        for vk in ["means2D_gradient_accum", "denom", "max_2D_radius"]:
            variables[vk] = torch.zeros(N, device="cuda").float()

        for grp in optimizer.param_groups:
            nm = grp["name"]
            if nm in params and grp["params"][0] is not params[nm]:
                old = grp["params"][0]
                if old in optimizer.state: del optimizer.state[old]
                grp["params"] = [params[nm]]
    return params, variables


def reset_opacity(params, optimizer, iteration, cfg):
    if iteration < cfg.get("from_iter", 3000): return params
    if iteration % cfg.get("interval", 3000) != 0: return params
    target = cfg.get("target_value", 0.01)
    with torch.no_grad():
        inv_sig = np.log(target / (1 - target))
        params["logit_opacities"] = torch.nn.Parameter(
            torch.full_like(params["logit_opacities"], inv_sig).requires_grad_(True))
        for grp in optimizer.param_groups:
            if grp["name"] == "logit_opacities":
                grp["params"] = [params["logit_opacities"]]
    return params


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(params, variables, optimizer, camera_data, num_iters=3000,
          densify_cfg=None, prune_cfg=None, opacity_cfg=None):
    """
    Run the minimal 3DGS training loop.
    camera_data: dict with keys "images" (list of np), "w2c" (M,4,4),
                 "K" (M,3,3), "width", "height"
    """
    if densify_cfg is None:
        densify_cfg = {"from_iter": 500, "until_iter": 3000, "grad_thresh": 0.0002}
    if prune_cfg is None:
        prune_cfg = {"from_iter": 500, "interval": 300,
                     "opacity_thresh": 0.005, "max_scale_thresh": 0.1,
                     "max_world_radius": 1e6}
    if opacity_cfg is None:
        opacity_cfg = {"from_iter": 3000, "interval": 3000, "target_value": 0.01}

    images = camera_data["images"]
    K_list = camera_data["K"]
    w = camera_data["width"]
    h = camera_data["height"]
    M = len(images)

    # Pre-load images to GPU as float32 tensors
    imgs_gpu = []
    for img_np in images:
        t = torch.tensor(img_np, dtype=torch.float32, device="cuda") / 255.0
        imgs_gpu.append(t.permute(2, 0, 1))

    pbar = tqdm(range(num_iters), desc="3DGS train")
    t0 = time.time()
    for it in range(num_iters):
        fi = random.randint(0, M - 1)
        fdata = {
            "idx": fi, "w": w, "h": h, "K": K_list[fi],
            "im": imgs_gpu[fi],
        }
        loss, variables = compute_loss(params, fdata, variables)
        loss.backward()

        with torch.no_grad():
            if variables["means2D"].grad is not None:
                gn = torch.norm(variables["means2D"].grad, dim=-1)
                variables["means2D_gradient_accum"] += gn
                variables["denom"] += 1

        with torch.no_grad():
            params, variables = densify(params, variables, optimizer, it, densify_cfg)
            params, variables = prune(params, variables, optimizer, it, prune_cfg)
            params = reset_opacity(params, optimizer, it, opacity_cfg)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "N": params["means3D"].shape[0]})
        pbar.update(1)
    pbar.close()

    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.1f}s ({elapsed/num_iters:.3f}s/iter)")
    print(f"  Final Gaussians: {params['means3D'].shape[0]:,}")
    return params, variables
