#!/bin/bash
# Scale-alignment ablation for V3DSAM contribution-2 (NVS / blob-collapse fix).
# Three matched conditions on the SAME npz (raw + depth_scaled_* from one VGGT run):
#   scaled_none : SL(4)-scaled depth, no median norm  [treatment, contribution-2]
#   raw_norm    : raw depth + per-frame median norm    [v38 baseline reproduction]
#   raw_none    : raw depth, no median norm            [isolate median-norm effect]
# Each: Stage3 (Splat-SLAM, pure RGB alpha=1.0) then held-out NVS eval.
# Self-running; writes a summary to $BASE/summary.txt.
set -u
PY=/root/miniconda3/envs/sgs/bin/python
export LD_LIBRARY_PATH=/root/miniconda3/envs/sgs/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export XFORMERS_DISABLED=1

NPZ=/root/autodl-tmp/psnr_tum/room/stage1_scaled/_stage3_input.npz
BASE=/root/autodl-tmp/psnr_tum/room/scale_ablation
CFG=/root/autodl-tmp/Splat-SLAM-main/configs/TUM_RGBD/freiburg1_desk2.yaml
V38=/root/autodl-tmp/v3dsam/vggt_slam/v38_stage3_splatslam.py
V40=/root/autodl-tmp/v40_stage3_scaled.py
mkdir -p "$BASE"

echo "===== scale ablation START $(date +%H:%M:%S) =====" | tee "$BASE/summary.txt"

run_one () {
  local MODE="$1"
  local OUT="$BASE/$MODE"
  local SCRIPT="$V38"
  local EXTRA=""
  case "$MODE" in
    scaled_none) SCRIPT="$V40"; EXTRA="--depth_mode scaled_none" ;;
    raw_none)    SCRIPT="$V40"; EXTRA="--depth_mode raw_none" ;;
    raw_norm)    SCRIPT="$V38"; EXTRA="" ;;  # default raw_norm
  esac
  echo "----- [$MODE] Stage3 $(date +%H:%M:%S) -----" | tee -a "$BASE/summary.txt"
  cd /root/autodl-tmp/Splat-SLAM-main
  $PY -u "$SCRIPT" --data_npz "$NPZ" --output_dir "$OUT" --alpha 1.0 $EXTRA > "$OUT.stage3.log" 2>&1
  echo -n "[$MODE] train PSNR: " | tee -a "$BASE/summary.txt"
  cat "$OUT/psnr.txt" 2>/dev/null | tee -a "$BASE/summary.txt" || echo "NO psnr.txt" | tee -a "$BASE/summary.txt"

  echo "----- [$MODE] NVS eval $(date +%H:%M:%S) -----" | tee -a "$BASE/summary.txt"
  cd /root/autodl-tmp
  $PY -u nvs_eval.py --data_npz "$NPZ" --ply "$OUT/gaussians_final.ply" \
    --output_dir "$OUT/nvs_eval" --config "$CFG" > "$OUT.nvs.log" 2>&1
  echo -n "[$MODE] NVS: " | tee -a "$BASE/summary.txt"
  cat "$OUT/nvs_eval/nvs.txt" 2>/dev/null | tee -a "$BASE/summary.txt" || echo "NO nvs.txt" | tee -a "$BASE/summary.txt"
  echo "" | tee -a "$BASE/summary.txt"
}

run_one scaled_none
run_one raw_norm
run_one raw_none

echo "===== scale ablation DONE $(date +%H:%M:%S) =====" | tee -a "$BASE/summary.txt"
echo "=== FINAL SUMMARY ===" | tee -a "$BASE/summary.txt"
cat "$BASE/summary.txt"
