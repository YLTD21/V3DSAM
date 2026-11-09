#!/usr/bin/env python3
"""Qualitative rendering comparison for the anti-blob ablations (Contribution 2).

Renders the SAME set of frames from several Splat-SLAM gaussians_final.ply files
(baseline / opacity-decay 0.995 / semantic-gated) and lays them out against the
ground-truth image.  Shows, in one figure, the two halves of the decay tradeoff:
  * train keyframe view  -> decay 0.995 destroys surfaces (train PSNR 14.52)
  * held-out novel view  -> all variants stay blobby; decay only ~+0.9 dB

Reuses the exact camera build + render path of nvs_eval.py (identity exposure).

Usage (AutoDL sgs env):
  python render_compare.py --data_npz .../stage1_scaled/_stage3_input.npz \
      --config .../configs/TUM_RGBD/freiburg1_desk2.yaml \
      --ply baseline=.../splatslam/gaussians_final.ply \
      --ply decay995=.../opacity_decay/od_decay_a1/gaussians_final.ply \
      --ply semgate=.../semantic_gate/gaussians_final.ply \
      --output_dir .../room/render_compare
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
    gt_T = torch.eye(4).cuda()
    cam = Camera(uid, color, None, gt_T, proj, fx, fy, cx, cy,
                 focal2fov(fx, W_out), focal2fov(fy, H_out), H_out, W_out)
    cam.update_RT(w2c_t[:3, :3], w2c_t[:3, 3])
    return cam


def pick_textured_keyframe(data, kf):
    """在关键帧里选纹理最丰富的一帧(拉普拉斯方差最大),让 0.995 破坏表面的对比最明显。"""
    best_idx, best_var = None, -1.0
    for fi in kf:
        p = str(data[f"img_path_{fi}"])
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        var = cv2.Laplacian(img, cv2.CV_64F).var()
        if var > best_var:
            best_var, best_idx = var, fi
    return best_idx if best_idx is not None else kf[len(kf) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_npz", required=True)
    ap.add_argument("--config", default=os.path.join(_SPLATSLAM, "configs/TUM_RGBD/freiburg1_desk2.yaml"))
    ap.add_argument("--ply", action="append", required=True, help="name=path (repeatable, ordered)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--frames", type=str, default="",
                    help="comma-sep frame indices; if empty, auto-pick mid keyframe + 2 held-out")
    args = ap.parse_args()

    cfg = load_config(args.config, './configs/splat_slam.yaml')
    cam_cfg = cfg["cam"]
    W_out = int(cam_cfg["W_out"]); H_out = int(cam_cfg["H_out"])
    W_edge = int(cam_cfg.get("W_edge", 0)); H_edge = int(cam_cfg.get("H_edge", 0))
    sh_degree = 3 if cfg["mapping"]["Training"]["spherical_harmonics"] else 0

    data = np.load(args.data_npz, allow_pickle=True)
    M = sum(1 for k in data.keys() if k.startswith("K_"))
    kf = sorted(int(x) for x in data["vggt_kf_indices"].tolist()) if "vggt_kf_indices" in data else []
    held = [i for i in range(M) if i not in set(kf)]
    if args.frames:
        frames = [int(x) for x in args.frames.split(",") if x.strip() != ""]
    else:
        kf_pick = pick_textured_keyframe(data, kf)
        # 2 held-out frames at distinct viewpoints (matches caption "下两行为 held-out 新视角")
        held_sel = [held[len(held) // 3], held[2 * len(held) // 3]]
        frames = [kf_pick] + held_sel
    frames = [f for f in frames if f < M]
    print(f"[render_compare] M={M} keyframes={len(kf)} held={len(held)}; frames={frames}")

    img_W = int(data["img_W"]); img_H = int(data["img_H"])
    K0 = data["K_0"]

    # Parse ply list
    plies = []
    for spec in args.ply:
        name, path = spec.split("=", 1)
        plies.append((name, path))

    # Load all Gaussian models (keep on GPU; room ~ a few hundred K gaussians)
    models = []
    for name, path in plies:
        g = GaussianModel(sh_degree, config=cfg)
        g.load_ply(path)
        models.append((name, g))
        print(f"[render_compare] {name}: {g._xyz.shape[0]} gaussians <- {path}")

    pipe = munchify(cfg["mapping"]["pipeline_params"])
    bg = torch.tensor([0.0, 0.0, 0.0], device="cuda")

    os.makedirs(args.output_dir, exist_ok=True)

    # GT + render for each frame, each model
    # cache GT per frame
    def render_frame(fi, model):
        cam = build_camera(fi, str(data[f"img_path_{fi}"]), data[f"w2c_{fi}"],
                           K0, img_W, img_H, W_out, H_out, W_edge, H_edge)
        with torch.no_grad():
            r = render(cam, model, pipe, bg)
        img = torch.clamp(r["render"], 0.0, 1.0)
        gt = cam.original_image
        return img, gt

    gt_cache = {}
    rows = []   # each row: (label, list of (name, np_img, psnr))
    for fi in frames:
        gt_img = None
        for name, g in models:
            img, gt = render_frame(fi, g)
            mse = torch.nn.functional.mse_loss(img, gt).item()
            psnr = 20 * math.log10(1.0 / math.sqrt(mse + 1e-10))
            if gt_img is None:
                gt_img = gt
            rows.append((fi, name, img, psnr))

    # compose figure: rows = frames, cols = [GT] + [models]
    # 训练关键帧行(第 0 行)放大,held-out 行缩小
    nrows = len(frames)
    ncols = 1 + len(models)
    from matplotlib.gridspec import GridSpec
    heights = [2.0] + [1.0] * (nrows - 1)
    fig = plt.figure(figsize=(3.1 * ncols, sum(heights) * 2.0))
    gs = GridSpec(nrows, ncols, figure=fig, height_ratios=heights, hspace=0.12, wspace=0.04)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(ncols)] for r in range(nrows)])

    for r, fi in enumerate(frames):
        # GT column
        ax = axes[r][0]
        gt_np = None
        for (fi2, name, img, psnr) in rows:
            if fi2 == fi and name == models[0][0]:
                gt_np = img
                break
        # find gt from any row's gt (identity): recompute gt image directly
        cam = build_camera(fi, str(data[f"img_path_{fi}"]), data[f"w2c_{fi}"],
                           K0, img_W, img_H, W_out, H_out, W_edge, H_edge)
        gt_np = cam.original_image.permute(1, 2, 0).cpu().numpy()
        ax.imshow(gt_np)
        ax.set_title("GT" if r == 0 else "")
        kind = "keyframe" if fi in set(kf) else "held-out"
        ax.set_ylabel(f"frame {fi}\n({kind})", fontsize=8)
        ax.axis("off")

        for c, (name, _g) in enumerate(models):
            ax = axes[r][c + 1]
            entry = [e for e in rows if e[0] == fi and e[1] == name][0]
            img_np = entry[2].permute(1, 2, 0).cpu().numpy()
            psnr = entry[3]
            ax.imshow(img_np)
            ax.set_title(f"{name}\n{psnr:.1f} dB" if r == 0 else f"{psnr:.1f} dB",
                         fontsize=8)
            ax.axis("off")

    fig.suptitle("Anti-blob ablation (fr1/room): GT vs baseline vs opacity-decay 0.995 vs semantic-gated\n"
                 "(identity exposure; top = train keyframe, bottom = held-out novel view)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = os.path.join(args.output_dir, "render_compare.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[render_compare] figure -> {out_png}")

    # also dump individual renders for reuse
    for fi in frames:
        for name, _g in models:
            entry = [e for e in rows if e[0] == fi and e[1] == name][0]
            img_np = entry[2].permute(1, 2, 0).cpu().numpy()
            out = (img_np * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(args.output_dir, f"f{fi:05d}_{name}.png"),
                        cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print("DONE")


if __name__ == "__main__":
    main()
