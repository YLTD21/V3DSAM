#!/usr/bin/env python3
"""Standalone Stage1 -> _stage3_input.npz WITH per-submap SL(4) scale alignment.

Same as gen_stage3_npz.py (official VGGT-SLAM stage1 + export), EXCEPT:
  1. For each frame we keep the SL(4) scale factor `decompose_camera(proj)[3]`
     (the per-submap scale that the SL(4) pose-graph optimization already
     estimated to align submaps globally).
  2. We export BOTH:
       depth_{i}         = raw VGGT depth (submap-local scale)   [baseline]
       depth_scaled_{i}  = depth_{i} * scale  (globally aligned) [contribution-2]
  3. We print per-submap scale statistics so the drift is visible.

The raw depth is left unchanged so the existing v38 baseline (median-norm)
still reproduces; v40 consumes depth_scaled with no per-frame median norm.

NOTE: this is a standalone script. It does NOT modify v3dsam source.
"""
import os, sys, re, glob, time, argparse, collections
import numpy as np, cv2, torch
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/v3dsam")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/vggt")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/sam3")
os.environ['XFORMERS_DISABLED'] = '1'

import torch.nn.functional as F
from vggt_slam_official.solver import Solver
import vggt_slam_official.slam_utils as slam_utils
from vggt.models.vggt import VGGT
from vggt_slam.slam_utils import decompose_camera

parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--submap_size", type=int, default=16)
parser.add_argument("--min_disparity", type=float, default=50)
parser.add_argument("--lc_thres", type=float, default=0.95)
parser.add_argument("--max_loops", type=int, default=1)
parser.add_argument("--max_points", type=int, default=10000000)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = "cuda"

names = sorted(glob.glob(os.path.join(args.image_dir, "*")))
names = [f for f in names if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
names = slam_utils.sort_images_by_number(names)
print(f"[gen_npz_scaled] Images: {len(names)}")

print(f"[gen_npz_scaled] Stage1 VGGT-SLAM S={args.submap_size} min_disp={args.min_disparity} lc={args.lc_thres}")
model = VGGT()
model.load_state_dict(torch.hub.load_state_dict_from_url(
    "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"))
model.eval()
model = model.to(torch.bfloat16).to(device)

solver = Solver(init_conf_threshold=25.0, lc_thres=args.lc_thres)
image_names_subset = []
vggt_kf_names = set()
n_loops = 0
t0 = time.time()
for name in tqdm(names, desc="VGGT-SLAM"):
    img = cv2.imread(name)
    if solver.flow_tracker.compute_disparity(img, args.min_disparity, False):
        image_names_subset.append(name)
        vggt_kf_names.add(name)
    if len(image_names_subset) == args.submap_size + 1 or name == names[-1]:
        predictions = solver.run_predictions(image_names_subset, model, args.max_loops, None, None)
        n_loops += len(predictions.get("detected_loops", []))
        solver.add_points(predictions)
        solver.graph.optimize()
        image_names_subset = image_names_subset[-1:]
print(f"[gen_npz_scaled] Stage1 done {time.time()-t0:.1f}s, {n_loops} loops")
del model; torch.cuda.empty_cache()

pts_list, rgb_list = [], []
for sm in solver.map.ordered_submaps_by_key():
    if sm.get_lc_status(): continue
    pts_list.append(sm.get_points_in_world_frame(solver.graph))
    rgb_list.append(sm.get_points_colors())
_pts_all = np.concatenate(pts_list).astype(np.float32)
_rgb_all = np.concatenate(rgb_list).astype(np.float32)
if _rgb_all.max() > 1.0:
    _rgb_all = _rgb_all / 255.0
_min_pts = min(len(_pts_all), len(_rgb_all))
_target = min(_min_pts, args.max_points)
_idx = np.random.RandomState(42).choice(_min_pts, _target, replace=False)
pts_gs = _pts_all[_idx].astype(np.float32)
rgb_gs = _rgb_all[_idx].astype(np.float32)

_first_sm = [sm for sm in solver.map.ordered_submaps_by_key() if not sm.get_lc_status()][0]
_first_depth = _first_sm.get_frame_pointcloud(0)
vggt_H, vggt_W = _first_depth.shape[:2] if _first_depth is not None else (392, 518)
_first_img = Image.open(names[0])
img_H_full, img_W_full = _first_img.size[1], _first_img.size[0]
scale_x = img_W_full / vggt_W; scale_y = img_H_full / vggt_H
print(f"[gen_npz_scaled] VGGT res {vggt_W}x{vggt_H}, full res {img_W_full}x{img_H_full}")

_all_c2w = solver.map.get_all_cam_matricies(solver.graph, False)
_all_proj = solver.map.get_all_cam_matricies(solver.graph, True)
_submap_lookup = {}
_counter = 0
for sm in solver.map.ordered_submaps_by_key():
    if sm.get_lc_status(): continue
    for _fi in range(len(sm.poses)):
        if _counter >= len(_all_c2w): break
        if sm.img_names and _fi < len(sm.img_names):
            _bn = os.path.basename(sm.img_names[_fi])
            _depth = sm.get_frame_pointcloud(_fi)
            _d = _depth[..., 2].astype(np.float32) if (_depth is not None and _depth.ndim == 3) else None
            _submap_lookup[_bn] = (_all_c2w[_counter], _all_proj[_counter], _d)
        _counter += 1

save = {"img_W": img_W_full, "img_H": img_H_full, "pts_gs": pts_gs, "rgb_gs": rgb_gs}
_n_exported = 0
vggt_kf_export_indices = []
_scale_list = []          # per-frame SL(4) scale
_scale_by_submap = []     # (submap_id, first_scale)
_cur_submap = None
for _orig_path in names:
    _bn = os.path.basename(_orig_path)
    if _bn in _submap_lookup:
        _c2w, _proj, _d = _submap_lookup[_bn]
    else:
        _orig_num = int(re.search(r'\d+', _bn).group())
        _best_dist = float('inf'); _best = None
        for _sbn, _sv in _submap_lookup.items():
            _snum = int(re.search(r'\d+', _sbn).group())
            _dnum = abs(_orig_num - _snum)
            if _dnum < _best_dist:
                _best_dist = _dnum; _best = _sv
        if _best is None: continue
        _c2w, _proj, _d = _best
    _K_vggt, _, _, _scale = decompose_camera(_proj)
    _scale = float(_scale)
    _scale_list.append(_scale)
    _K_full = _K_vggt.copy()
    _K_full[0,0] *= scale_x; _K_full[0,2] *= scale_x
    _K_full[1,1] *= scale_y; _K_full[1,2] *= scale_y
    _w2c = np.linalg.inv(_c2w)
    save[f"K_{_n_exported}"] = _K_full.astype(np.float32)
    save[f"w2c_{_n_exported}"] = _w2c.astype(np.float32)
    save[f"img_path_{_n_exported}"] = _orig_path
    if _d is not None:
        save[f"depth_{_n_exported}"] = _d.astype(np.float32)
        save[f"depth_scaled_{_n_exported}"] = (_d * _scale).astype(np.float32)
    if _orig_path in vggt_kf_names:
        vggt_kf_export_indices.append(_n_exported)
    _n_exported += 1

save["vggt_kf_indices"] = np.array(vggt_kf_export_indices, dtype=np.int32)
npz_path = os.path.join(args.output_dir, "_stage3_input.npz")
np.savez_compressed(npz_path, **save)
print(f"[gen_npz_scaled] Saved {npz_path} ({os.path.getsize(npz_path)/1e6:.1f} MB, "
      f"{_target/1e6:.2f}M pts, {_n_exported} frames, {len(vggt_kf_export_indices)} keyframes)")

# ═══ Scale diagnostics ═══
_sc = np.array(_scale_list)
print(f"\n[gen_npz_scaled] === SL(4) per-frame scale diagnostics === ")
print(f"  n={_sc.size}  mean={_sc.mean():.4f}  std={_sc.std():.4f}  "
      f"min={_sc.min():.4f}  max={_sc.max():.4f}")
print(f"  |scale-1|>0.05 : {(np.abs(_sc-1)>0.05).sum()}/{_sc.size}")
print(f"  |scale-1|>0.10 : {(np.abs(_sc-1)>0.10).sum()}/{_sc.size}")
# show scale sequence (every ~1 submap, i.e. every 33rd exported frame)
print(f"  scale sequence (every 33rd frame):")
print(f"    " + " ".join(f"{_sc[i]:.3f}" for i in range(0, _sc.size, 33)))
with open(os.path.join(args.output_dir, "scale_stats.txt"), "w") as f:
    f.write(f"n={_sc.size} mean={_sc.mean():.4f} std={_sc.std():.4f} "
            f"min={_sc.min():.4f} max={_sc.max():.4f}\n")
    f.write("scale_per_frame:\n" + " ".join(f"{x:.4f}" for x in _sc) + "\n")
print(f"[gen_npz_scaled] done")
