#!/usr/bin/env python3
"""NVS (novel-view synthesis) evaluation for V3DSAM Stage3.

Quantifies "blob collapse": a model with high train-keyframe PSNR can collapse
when rendered from a held-out (novel) viewpoint.  We render frames that were
exported in _stage3_input.npz but NOT used as mapper keyframes, and compare
against their ground-truth images (PSNR / SSIM).

Matches v38_stage3_splatslam.py exactly: same config loading, same intrinsic
scaling (resize-to-edge then crop), same Splat-SLAM Camera + render.  Identity
exposure is used for held-out cameras (they have no learned exposure) — this
understates absolute PSNR but preserves the geometric-collapse signal.

Usage (AutoDL, sgs env):
  python nvs_eval.py --data_npz /root/autodl-tmp/psnr_tum/room/stage1/_stage3_input.npz \
                     --ply /root/autodl-tmp/psnr_tum/room/splatslam/gaussians_final.ply \
                     --output_dir /root/autodl-tmp/psnr_tum/room/nvs_eval \
                     --config /root/autodl-tmp/Splat-SLAM-main/configs/TUM_RGBD/freiburg1_room.yaml
"""
import os, sys, argparse, math
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
from thirdparty.gaussian_splatting.utils.loss_utils import ssim
from thirdparty.monogs.utils.camera_utils import Camera
from thirdparty.glorie_slam.config import load_config


def build_camera(uid, img_path, w2c, K, img_W, img_H, W_out, H_out, W_edge, H_edge):
    """Replicate v38 NPZDataset camera: resize->edge->crop, scale intrinsics."""
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
    gt_T = torch.eye(4).cuda()
    cam = Camera(uid, color, None, gt_T, proj, fx, fy, cx, cy,
                 focal2fov(fx, W_out), focal2fov(fy, H_out), H_out, W_out)
    cam.update_RT(w2c_t[:3, :3], w2c_t[:3, 3])
    return cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_npz", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--config", default=os.path.join(_SPLATSLAM, "configs/TUM_RGBD/freiburg1_desk2.yaml"))
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config, './configs/splat_slam.yaml')
    cam_cfg = cfg["cam"]
    W_out = int(cam_cfg["W_out"]); H_out = int(cam_cfg["H_out"])
    W_edge = int(cam_cfg.get("W_edge", 0)); H_edge = int(cam_cfg.get("H_edge", 0))
    sh_degree = 3 if cfg["mapping"]["Training"]["spherical_harmonics"] else 0

    data = np.load(args.data_npz, allow_pickle=True)
    M = sum(1 for k in data.keys() if k.startswith("K_"))
    kf = set(int(x) for x in data["vggt_kf_indices"].tolist()) if "vggt_kf_indices" in data else set()
    held_out = [i for i in range(M) if i not in kf]
    if args.max_frames > 0:
        held_out = held_out[:args.max_frames]
    print(f"[nvs_eval] {M} frames, {len(kf)} keyframes, {len(held_out)} held-out; "
          f"res {W_out}x{H_out} edge {W_edge}/{H_edge}")

    gaussians = GaussianModel(sh_degree, config=cfg)
    gaussians.load_ply(args.ply)
    pipe = munchify(cfg["mapping"]["pipeline_params"])
    bg = torch.tensor([0.0, 0.0, 0.0], device="cuda")
    print(f"[nvs_eval] gaussians: {gaussians._xyz.shape[0]}")

    os.makedirs(args.output_dir, exist_ok=True)
    K0 = data["K_0"]
    img_W = int(data["img_W"]); img_H = int(data["img_H"])
    psnrs, ssims, rows = [], [], []
    for k, i in enumerate(held_out):
        try:
            cam = build_camera(i, str(data[f"img_path_{i}"]), data[f"w2c_{i}"],
                               K0, img_W, img_H, W_out, H_out, W_edge, H_edge)
            with torch.no_grad():
                r = render(cam, gaussians, pipe, bg)
            img = torch.clamp(r["render"], 0.0, 1.0)
            gt = cam.original_image
            mse = torch.nn.functional.mse_loss(img, gt).item()
            psnr = 20 * math.log10(1.0 / math.sqrt(mse + 1e-10))
            s = ssim(img.unsqueeze(0), gt.unsqueeze(0)).item()
            psnrs.append(psnr); ssims.append(s); rows.append((i, psnr, s))
            if k < 8:
                out = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(args.output_dir, f"render_{i:05d}.png"),
                            cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"  [skip {i}] {e}")

    psnrs = np.array(psnrs); ssims = np.array(ssims)
    print(f"\n{'='*60}")
    if len(psnrs):
        print(f"  NVS held-out ({len(psnrs)}): PSNR mean={psnrs.mean():.2f} "
              f"min={psnrs.min():.2f} max={psnrs.max():.2f}  SSIM mean={ssims.mean():.4f}")
        order = np.argsort(psnrs)
        print(f"  worst5: {[(rows[j][0], round(psnrs[j],2)) for j in order[:5]]}")
        print(f"  best5 : {[(rows[j][0], round(psnrs[j],2)) for j in order[-5:]]}")
    print(f"{'='*60}")
    with open(os.path.join(args.output_dir, "nvs.txt"), "w") as f:
        f.write(f"N={len(psnrs)} PSNR_mean={psnrs.mean():.2f} PSNR_min={psnrs.min():.2f} "
                f"SSIM_mean={ssims.mean():.4f}\n")
    np.save(os.path.join(args.output_dir, "nvs_psnr.npy"), psnrs)
    print(f"\nDone -> {args.output_dir}")


if __name__ == "__main__":
    main()
