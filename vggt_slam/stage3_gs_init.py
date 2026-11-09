#!/usr/bin/env python3
"""
Stage 3: Initialize 3D Gaussian Splatting from the aggregated point cloud
and camera data.  Produces trainable Gaussian parameters and an optimizer.
"""

import numpy as np
import torch
import torch.nn.functional as F


def matrix_to_quaternion(matrix):
    """Convert (..., 3, 3) rotation matrices to quaternions [w,x,y,z]."""
    batch_dim = matrix.shape[:-2]
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]

    q_abs = torch.sqrt(torch.clamp(torch.stack([
        1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22,
    ], dim=-1), min=0.0))

    quat_by_rijk = torch.stack([
        torch.stack([q_abs[..., 0]**2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1]**2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2]**2, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3]**2], dim=-1),
    ], dim=-2)
    flr = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def initialize_params(points3D, colors, num_frames):
    """
    Initialize 3DGS parameters from a point cloud.
    Args:
        points3D: (N,3) numpy array, world-frame positions
        colors:   (N,3) numpy array in [0,1]
        num_frames: int
    Returns: params dict, variables dict
    """
    N = points3D.shape[0]
    device = "cuda"

    mean_dist = float(np.mean(
        np.linalg.norm(points3D - points3D.mean(axis=0), axis=1)))
    init_scale = max(mean_dist / 10, 0.001)

    params = {
        "means3D": torch.nn.Parameter(
            torch.tensor(points3D, dtype=torch.float32, device=device).requires_grad_(True)),
        "rgb_colors": torch.nn.Parameter(
            torch.tensor(colors, dtype=torch.float32, device=device).requires_grad_(True)),
        "unnorm_rotations": torch.nn.Parameter(
            torch.tensor(np.tile([1, 0, 0, 0], (N, 1)), dtype=torch.float32, device=device).requires_grad_(True)),
        "logit_opacities": torch.nn.Parameter(
            torch.zeros((N, 1), dtype=torch.float32, device=device).requires_grad_(True)),
        "log_scales": torch.nn.Parameter(
            torch.full((N, 1), np.log(init_scale), dtype=torch.float32, device=device).requires_grad_(True)),
    }

    # Camera poses: identity init, filled from SLAM data below
    params["cam_unnorm_rots"] = torch.nn.Parameter(
        torch.tensor(np.tile([1, 0, 0, 0], (1, 1)), dtype=torch.float32, device=device)
        .unsqueeze(-1).expand(-1, -1, num_frames).clone().requires_grad_(True))
    params["cam_trans"] = torch.nn.Parameter(
        torch.zeros((1, 3, num_frames), dtype=torch.float32, device=device).requires_grad_(True))

    variables = {
        "max_2D_radius": torch.zeros(N, dtype=torch.float32, device=device),
        "means2D_gradient_accum": torch.zeros(N, dtype=torch.float32, device=device),
        "denom": torch.zeros(N, dtype=torch.float32, device=device),
    }
    return params, variables


def init_camera_poses_from_slam(params, w2c_poses):
    """
    Fill trainable camera parameters from SLAM w2c poses.
    Args:
        params: dict returned by initialize_params()
        w2c_poses: (M, 4, 4) numpy array, world-to-camera
    """
    device = "cuda"
    M = w2c_poses.shape[0]
    cam_q, cam_t = [], []
    for i in range(M):
        w2c = torch.tensor(w2c_poses[i], dtype=torch.float32, device=device)
        q = matrix_to_quaternion(w2c[:3, :3].unsqueeze(0))  # (1,4)
        cam_q.append(q)
        cam_t.append(w2c[:3, 3])
    params["cam_unnorm_rots"] = torch.nn.Parameter(
        torch.stack(cam_q, dim=-1).requires_grad_(True))  # (1, 4, M)
    params["cam_trans"] = torch.nn.Parameter(
        torch.stack(cam_t, dim=-1).unsqueeze(0).requires_grad_(True))  # (1, 3, M)
    return params


def setup_optimizer(params):
    """Create Adam optimizer with recommended learning rates."""
    lrs = {
        "means3D": 1.6e-5, "rgb_colors": 5e-4,
        "unnorm_rotations": 5e-4, "logit_opacities": 0.2,
        "log_scales": 0.01, "cam_unnorm_rots": 5e-4,
        "cam_trans": 5e-4,
    }
    groups = [{"params": [v], "name": k, "lr": lrs[k]}
              for k, v in params.items() if isinstance(v, torch.nn.Parameter)]
    return torch.optim.Adam(groups, lr=0.0, eps=1e-15)


def params2rendervar(params):
    """Convert training params to renderer variables."""
    return {
        "means3D": params["means3D"],
        "colors_precomp": params["rgb_colors"],
        "rotations": F.normalize(params["unnorm_rotations"]),
        "opacities": torch.sigmoid(params["logit_opacities"]),
        "scales": torch.exp(torch.tile(params["log_scales"], (1, 3))),
        "means2D": torch.zeros_like(params["means3D"], requires_grad=True, device="cuda") + 0,
    }
