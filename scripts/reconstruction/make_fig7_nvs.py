#!/usr/bin/env python3
"""Honest fig7 — novel-view rendering (held-out frames), GT vs ours.

The previous fig_rendering() in make_figures_v2.py read plots_after_refine/*.png
which are TRAINING-keyframe renders (~22 dB), yet the caption called them
"新视角渲染".  This script renders the actual HELD-OUT (novel-view) frames and
shows the honest "训练高、新视角糊" collapse (~9 dB).

For each scene it takes the top-3 held-out frames BY PSNR (from nvs_psnr.npy,
which nvs_eval.py wrote over all held-out frames) and renders them from the
scene's gaussians_final.ply, reusing the exact build_camera + render path of
nvs_eval.py (identity exposure).  Output is 3 scenes x (GT + 3 renders) with
the true per-frame PSNR.

Run on AutoDL (sgs env):
  /root/miniconda3/envs/sgs/bin/python make_fig7_nvs.py
"""
import os, sys, math, glob
import numpy as np, cv2, torch

_SPLATSLAM = "/root/autodl-tmp/Splat-SLAM-main"
os.chdir(_SPLATSLAM)
sys.path.insert(0, _SPLATSLAM)
sys.path.insert(0, os.path.join(_SPLATSLAM, "thirdparty", "simple-knn"))
_torch_lib = "/root/miniconda3/envs/sgs/lib/python3.12/site-packages/torch/lib"
os.environ["LD_LIBRARY_PATH"] = f"{_torch_lib}:{os.environ.get('LD_LIBRARY_PATH','')}".strip(":")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["MPLBACKEND"] = "Agg"

from munch import munchify
from thirdparty.gaussian_splatting.gaussian_renderer import render
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
from thirdparty.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, focal2fov
from thirdparty.monogs.utils.camera_utils import Camera
from thirdparty.glorie_slam.config import load_config

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/root/autodl-tmp")   # sci_style.py lives beside the scripts
from sci_style import apply_publication_style
apply_publication_style(font_size=8, axes_linewidth=1)


def build_camera(uid, img_path, w2c, K, img_W, img_H, W_out, H_out, W_edge, H_edge):
    W_out_edge = W_out + 2 * W_edge
    H_out_edge = H_out + 2 * H_edge
    fx = float(K[0, 0]) * W_out_edge / img_W
    fy = float(K[1, 1]) * H_out_edge / img_H
    cx = float(K[0, 2]) * W_out_edge / img_W - W_edge
    cy = float(K[1, 2]) * H_out_edge / img_H - H_edge
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img = cv2.resize(img, (W_out_edge, H_out_edge))
    if W_edge > 0 or H_edge > 0:
        img = img[H_edge:H_edge + H_out, W_edge:W_edge + W_out]
    color = torch.from_numpy(img).float().permute(2, 0, 1).cuda()
    proj = getProjectionMatrix2(znear=0.01, zfar=100.0, fx=fx, fy=fy,
                                cx=cx, cy=cy, W=W_out, H=H_out).transpose(0, 1).cuda()
    w2c_t = torch.from_numpy(w2c).float().cuda()
    cam = Camera(uid, color, None, torch.eye(4).cuda(), proj, fx, fy, cx, cy,
                 focal2fov(fx, W_out), focal2fov(fy, H_out), H_out, W_out)
    cam.update_RT(w2c_t[:3, :3], w2c_t[:3, 3])
    return cam


# scenes -> (npz, ply, nvs_psnr, config)
SCENES = [
    ("room",  "/root/autodl-tmp/psnr_tum/room/stage1_scaled/_stage3_input.npz",
              "/root/autodl-tmp/psnr_tum/room/splatslam/gaussians_final.ply"),
    ("floor", "/root/autodl-tmp/psnr_tum/floor/stage1/_stage3_input.npz",
              "/root/autodl-tmp/psnr_tum/floor/splatslam/gaussians_final.ply"),
    ("360",   "/root/autodl-tmp/psnr_tum/360/stage1/_stage3_input.npz",
              "/root/autodl-tmp/psnr_tum/360/splatslam/gaussians_final.ply"),
]
CONFIG = os.path.join(_SPLATSLAM, "configs/TUM_RGBD/freiburg1_room.yaml")
OUT = "/root/autodl-tmp/figures_sci/fig7_rendering"


def main():
    cfg = load_config(CONFIG, './configs/splat_slam.yaml')
    cam_cfg = cfg["cam"]
    W_out = int(cam_cfg["W_out"]); H_out = int(cam_cfg["H_out"])
    W_edge = int(cam_cfg.get("W_edge", 0)); H_edge = int(cam_cfg.get("H_edge", 0))
    sh_degree = 3 if cfg["mapping"]["Training"]["spherical_harmonics"] else 0
    pipe = munchify(cfg["mapping"]["pipeline_params"])
    bg = torch.tensor([0.0, 0.0, 0.0], device="cuda")

    rows = []  # per scene: list of (gt_np, ren_np, psnr)
    for seq, npz_path, ply_path in SCENES:
        data = np.load(npz_path, allow_pickle=True)
        M = sum(1 for k in data.keys() if k.startswith("K_"))
        kf = set(int(x) for x in data["vggt_kf_indices"].tolist())
        held_out = [i for i in range(M) if i not in kf]
        psnr_all = np.load(os.path.join(os.path.dirname(npz_path), "..", "nvs_eval", "nvs_psnr.npy"))
        assert len(psnr_all) == len(held_out), \
            f"{seq}: nvs_psnr {len(psnr_all)} != held_out {len(held_out)} (npz mismatch?)"

        gaussians = GaussianModel(sh_degree, config=cfg)
        gaussians.load_ply(ply_path)
        print(f"[fig7] {seq}: {M} frames, {len(kf)} kf, {len(held_out)} held-out, "
              f"{gaussians._xyz.shape[0]} gaussians")

        img_W = int(data["img_W"]); img_H = int(data["img_H"]); K0 = data["K_0"]
        mean_psnr = float(psnr_all.mean())
        # pick 3 held-out frames closest to the MEAN PSNR (representative "糊" frames,
        # consistent with the paper's quoted mean ~9 dB, not the misleading top-3 ~18 dB)
        rep_pos = np.argsort(np.abs(psnr_all - mean_psnr))[:3]
        rep_frames = [held_out[p] for p in rep_pos]
        rep_psnr = [float(psnr_all[p]) for p in rep_pos]
        order = np.argsort(rep_frames)
        rep_frames = [rep_frames[o] for o in order]
        rep_psnr = [rep_psnr[o] for o in order]
        print(f"[fig7]   {seq} mean={mean_psnr:.2f} rep held-out frames {rep_frames} "
              f"PSNR {[round(p,2) for p in rep_psnr]}")

        scene_rows = []
        for fi in rep_frames:
            cam = build_camera(fi, str(data[f"img_path_{fi}"]), data[f"w2c_{fi}"],
                               K0, img_W, img_H, W_out, H_out, W_edge, H_edge)
            with torch.no_grad():
                r = render(cam, gaussians, pipe, bg)
            img = torch.clamp(r["render"], 0.0, 1.0)
            gt = cam.original_image
            mse = torch.nn.functional.mse_loss(img, gt).item()
            psnr = 20 * math.log10(1.0 / math.sqrt(mse + 1e-10))
            gt_np = gt.permute(1, 2, 0).cpu().numpy()
            ren_np = img.permute(1, 2, 0).cpu().numpy()
            scene_rows.append((gt_np, ren_np, psnr))
        rows.append(scene_rows)
        del gaussians
        torch.cuda.empty_cache()

    # compose: 3 rows (scenes) x 6 cols (GT + 3 renders)
    fig, axes = plt.subplots(len(SCENES), 6, figsize=(13.5, len(SCENES) * 2.5))
    if len(SCENES) == 1:
        axes = axes[None, :]
    for r, (seq, _, _) in enumerate(SCENES):
        for c, (gt_np, ren_np, psnr) in enumerate(rows[r]):
            axes[r, 2 * c].imshow(gt_np); axes[r, 2 * c].axis("off")
            axes[r, 2 * c + 1].imshow(ren_np); axes[r, 2 * c + 1].axis("off")
            axes[r, 2 * c + 1].set_title(f"{psnr:.1f} dB", fontsize=8)
        axes[r, 0].set_ylabel(f"fr1/{seq}", fontsize=9, rotation=0, labelpad=28)
    for c in range(3):
        axes[0, 2 * c].set_title("Ground truth", fontsize=9)
        axes[0, 2 * c + 1].set_title("Rendered (ours)", fontsize=9)
    fig.suptitle("Novel-view rendering: representative held-out frames (PSNR ≈ per-scene mean) "
                 "from Splat-SLAM reconstruction — the train/novel-view collapse (~9 dB)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    print(f"[fig7] -> {OUT}.png / .pdf")


if __name__ == "__main__":
    main()
