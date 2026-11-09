#!/bin/bash
# Regenerate all remote figures with the official-aligned sci_style.
# Run on AutoDL (sgs env). Logs to /root/autodl-tmp/figs_official.log
set -u
PY=/root/miniconda3/envs/sgs/bin/python
CD=/root/autodl-tmp
FIG=$CD/figures_sci
LOG=$CD/figs_official.log
exec > >(tee -a "$LOG") 2>&1
echo "===== START $(date +%H:%M:%S) ====="
mkdir -p "$FIG"

# --- CPU figures ---
echo "[1/6] fig2/4/5 (make_figures_v3.py)"
cd "$CD" && $PY make_figures_v3.py || echo "  FAILED fig2/4/5"

echo "[2/6] fig6 false loop (make_figures_v2.py)"
cd "$CD" && $PY make_figures_v2.py || echo "  FAILED fig6"

echo "[3/6] fig3 semantic cloud (gen_semantic_pointcloud.py)"
cd "$CD" && $PY gen_semantic_pointcloud.py \
  --stage3 "$CD/psnr_tum/room/stage1_scaled/_stage3_input.npz" \
  --semantic "$CD/psnr_tum/room/stage1_scaled/kf_semantic.npz" \
  --out_dir "$CD/psnr_tum/room/semantic_cloud" || echo "  FAILED fig3"
cp -f "$CD/psnr_tum/room/semantic_cloud/semantic_cloud.png" "$FIG/fig3_semantic_cloud.png" 2>/dev/null && echo "  fig3 -> figures_sci/fig3_semantic_cloud.png"

# --- GPU figures ---
echo "[4/6] fig7 novel-view rendering (make_fig7_nvs.py)"
cd "$CD" && $PY make_fig7_nvs.py || echo "  FAILED fig7"

echo "[5/6] fig9 render compare (render_compare.py)"
cd "$CD" && $PY render_compare.py \
  --data_npz "$CD/psnr_tum/room/stage1_scaled/_stage3_input.npz" \
  --config "$CD/Splat-SLAM-main/configs/TUM_RGBD/freiburg1_room.yaml" \
  --ply baseline="$CD/psnr_tum/room/splatslam/gaussians_final.ply" \
  --ply decay995="$CD/psnr_tum/room/opacity_decay/od_decay_a1/gaussians_final.ply" \
  --ply semgate="$CD/psnr_tum/room/semantic_gate/gaussians_final.ply" \
  --output_dir "$CD/psnr_tum/room/render_compare" || echo "  FAILED fig9"
cp -f "$CD/psnr_tum/room/render_compare/render_compare.png" "$FIG/fig9_render_compare.png" 2>/dev/null && echo "  fig9 -> figures_sci/fig9_render_compare.png"

echo "[6/6] fig12 instance multiframe (gen_instance_multiframe.py)"
cd "$CD" && $PY gen_instance_multiframe.py --seq room || echo "  FAILED fig12"
cp -f "$CD/psnr_tum/room/semantic_instances_postfilter/fig12_instance_multiframe.png" "$FIG/fig12_instance_multiframe.png" 2>/dev/null && echo "  fig12 -> figures_sci/fig12_instance_multiframe.png"

echo "===== DONE $(date +%H:%M:%S) ====="
echo "--- figures_sci listing ---"
ls -la --time-style=+%H:%M "$FIG" | grep -E "fig(2|3|4|5|6|7|9|12)_"
