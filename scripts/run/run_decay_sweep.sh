#!/bin/bash
# Decay-rate sweep + refine-iters lever (contribution-2 fix).
#   no-decay baseline:  train=22.81  NVS=9.21  SSIM=0.3405
#   decay 0.995:        train=14.52  NVS=10.08 SSIM=0.4275  (over-decays surfaces)
# Goal: find rate that keeps train high AND lifts NVS. Candidate sweet spot ~0.9995-0.9998.
# Also test the orthogonal lever: fewer final_refine iters (less overfitting).
set -u
PY=/root/miniconda3/envs/sgs/bin/python
export LD_LIBRARY_PATH=/root/miniconda3/envs/sgs/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export XFORMERS_DISABLED=1

NPZ=/root/autodl-tmp/psnr_tum/room/stage1_scaled/_stage3_input.npz
BASE=/root/autodl-tmp/psnr_tum/room/decay_sweep
CFG=/root/autodl-tmp/Splat-SLAM-main/configs/TUM_RGBD/freiburg1_desk2.yaml
V40=/root/autodl-tmp/v40_stage3_scaled.py
mkdir -p "$BASE"

echo "===== decay sweep START $(date +%H:%M:%S) =====" | tee "$BASE/summary.txt"

run_one () {
  local TAG="$1"; shift
  local OUT="$BASE/$TAG"
  echo "----- [$TAG] Stage3 $(date +%H:%M:%S) -----" | tee -a "$BASE/summary.txt"
  cd /root/autodl-tmp/Splat-SLAM-main
  $PY -u "$V40" --data_npz "$NPZ" --output_dir "$OUT" --alpha 1.0 --depth_mode scaled_none "$@" > "$OUT.stage3.log" 2>&1
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

run_one r999   --opacity_decay --opacity_decay_rate 0.999
run_one r9995  --opacity_decay --opacity_decay_rate 0.9995
run_one r9998  --opacity_decay --opacity_decay_rate 0.9998
run_one refine5000 --refine_iters 5000

echo "===== decay sweep DONE $(date +%H:%M:%S) =====" | tee -a "$BASE/summary.txt"
