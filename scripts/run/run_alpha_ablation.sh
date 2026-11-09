#!/bin/bash
# Decisive experiment for contribution-2:
#   Does SCALE-ALIGNED depth + depth supervision (alpha<1.0) fix NVS?
# Prior ablation showed scale alignment ALONE (alpha=1.0, pure RGB) has no effect:
#   scaled@1.0 NVS=9.21, raw@1.0 NVS=9.25  -> depth init is overridden by RGB overfit.
# Hypothesis: scale-consistent depth ENABLES depth supervision (alpha<1.0) to
#   regularize against sparse-view RGB overfitting -> fixes NVS.
# 2x2: {scaled, raw} x {alpha=0.8 (default), alpha=0.5 (stronger)}.
set -u
PY=/root/miniconda3/envs/sgs/bin/python
export LD_LIBRARY_PATH=/root/miniconda3/envs/sgs/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export XFORMERS_DISABLED=1

NPZ=/root/autodl-tmp/psnr_tum/room/stage1_scaled/_stage3_input.npz
BASE=/root/autodl-tmp/psnr_tum/room/alpha_ablation
CFG=/root/autodl-tmp/Splat-SLAM-main/configs/TUM_RGBD/freiburg1_desk2.yaml
V40=/root/autodl-tmp/v40_stage3_scaled.py
mkdir -p "$BASE"

echo "===== alpha/depth-supervision ablation START $(date +%H:%M:%S) =====" | tee "$BASE/summary.txt"

run_one () {
  local MODE="$1"   # scaled_none | raw_norm
  local ALPHA="$2"
  local TAG="${MODE}_a${ALPHA}"
  local OUT="$BASE/$TAG"
  echo "----- [$TAG] Stage3 (alpha=$ALPHA) $(date +%H:%M:%S) -----" | tee -a "$BASE/summary.txt"
  cd /root/autodl-tmp/Splat-SLAM-main
  $PY -u "$V40" --data_npz "$NPZ" --output_dir "$OUT" --alpha "$ALPHA" --depth_mode "$MODE" > "$OUT.stage3.log" 2>&1
  echo -n "[$TAG] train PSNR: " | tee -a "$BASE/summary.txt"
  cat "$OUT/psnr.txt" 2>/dev/null | tee -a "$BASE/summary.txt" || echo "NO psnr.txt" | tee -a "$BASE/summary.txt"

  echo "----- [$TAG] NVS eval $(date +%H:%M:%S) -----" | tee -a "$BASE/summary.txt"
  cd /root/autodl-tmp
  $PY -u nvs_eval.py --data_npz "$NPZ" --ply "$OUT/gaussians_final.ply" \
    --output_dir "$OUT/nvs_eval" --config "$CFG" > "$OUT.nvs.log" 2>&1
  echo -n "[$TAG] NVS: " | tee -a "$BASE/summary.txt"
  cat "$OUT/nvs_eval/nvs.txt" 2>/dev/null | tee -a "$BASE/summary.txt" || echo "NO nvs.txt" | tee -a "$BASE/summary.txt"
  echo "" | tee -a "$BASE/summary.txt"
}

run_one scaled_none 0.8
run_one raw_norm    0.8
run_one scaled_none 0.5
run_one raw_norm    0.5

echo "===== alpha ablation DONE $(date +%H:%M:%S) =====" | tee -a "$BASE/summary.txt"
