#!/usr/bin/env python3
"""Create v40_stage3_scaled.py from v38_stage3_splatslam.py via surgical patch.

v40 adds a --depth_mode flag:
  raw_norm   (default) = v38 baseline: raw depth + per-frame median norm
  raw_none   = raw depth, no norm (isolates the median-norm effect)
  scaled_none= SL(4)-scaled depth (depth_scaled_*), no norm  [contribution-2]

Only the depth-field selection and the median-norm guard change; everything
else (mapper, keyframe pre-filter, stereo init, eval) is identical so the
comparison is matched.
"""
import sys

SRC = "/root/autodl-tmp/v3dsam/vggt_slam/v38_stage3_splatslam.py"
DST = "/root/autodl-tmp/v40_stage3_scaled.py"

src = open(SRC, encoding="utf-8").read()

# ── Patch 1: add --depth_mode argument ──
old1 = '''parser.add_argument("--metric_depth", action="store_true", help="Depth is metric (DROID), skip scale normalization and stereo init")
args = parser.parse_args()'''
new1 = '''parser.add_argument("--metric_depth", action="store_true", help="Depth is metric (DROID), skip scale normalization and stereo init")
parser.add_argument("--depth_mode", type=str, default="raw_norm", help="raw_norm (v38) | raw_none | scaled_none")
args = parser.parse_args()'''
assert src.count(old1) == 1, f"patch1 anchor count = {src.count(old1)}"
src = src.replace(old1, new1)

# ── Patch 2: depth field selection + median-norm guard ──
old2 = '''    elif f"depth_{frame_idx}" in data and data[f"depth_{frame_idx}"] is not None:
        d_np = data[f"depth_{frame_idx}"].astype(np.float32)
        # Match image pipeline: resize → crop (same as NPZDataset)
        d_np = cv2.resize(d_np, (W_out + 2*W_edge, H_out + 2*H_edge))
        if W_edge > 0 or H_edge > 0:
            d_np = d_np[H_edge:H_edge + H_out, W_edge:W_edge + W_out]
        if not args.metric_depth:
            # Global scale normalization: only for scale-ambiguous monocular depth (VGGT)
            d_med = np.median(d_np[d_np > 0.1]) if (d_np > 0.1).any() else 1.0
            if d_med > 0:
                d_np = d_np * (_VGGT_GLOBAL_MEDIAN / d_med)'''
new2 = '''    elif f"depth_{frame_idx}" in data and data[f"depth_{frame_idx}"] is not None:
        _dkey = f"depth_scaled_{frame_idx}" if (args.depth_mode.startswith("scaled") and f"depth_scaled_{frame_idx}" in data) else f"depth_{frame_idx}"
        d_np = data[_dkey].astype(np.float32)
        # Match image pipeline: resize → crop (same as NPZDataset)
        d_np = cv2.resize(d_np, (W_out + 2*W_edge, H_out + 2*H_edge))
        if W_edge > 0 or H_edge > 0:
            d_np = d_np[H_edge:H_edge + H_out, W_edge:W_edge + W_out]
        if args.depth_mode == "raw_norm" and not args.metric_depth:
            # Global scale normalization: only for scale-ambiguous monocular depth (VGGT)
            d_med = np.median(d_np[d_np > 0.1]) if (d_np > 0.1).any() else 1.0
            if d_med > 0:
                d_np = d_np * (_VGGT_GLOBAL_MEDIAN / d_med)'''
assert src.count(old2) == 1, f"patch2 anchor count = {src.count(old2)}"
src = src.replace(old2, new2)

open(DST, "w", encoding="utf-8").write(src)
print(f"[make_v40] wrote {DST} ({len(src)} bytes)")
print(f"[make_v40] patch1 applied, patch2 applied")
