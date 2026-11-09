#!/bin/bash
# Opacity-decay experiment (contribution-2 real fix candidate).
# Wait for the alpha/depth-supervision ablation to finish (avoid GPU contention),
# then run: pure-RGB (alpha=1.0) + StereoGS gradient-aware opacity decay.
# Baseline (no decay): scaled@1.0 NVS=9.21, train=22.81. If decay lifts NVS,
# floaters were the collapse cause (not depth scale / not depth supervision).
set -u
PY=/root/miniconda3/envs/sgs/bin/python
export LD_LIBRARY_PATH=/root/miniconda3/envs/sgs/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export XFORMERS_DISABLED=1

NPZ=/root/autodl-tmp/psnr_tum/room/stage1_scaled/_stage3_input.npz
BASE=/root/autodl-tmp/psnr_tum/room/opacity_decay
CFG=/root/autodl-tmp/Splat-SLAM-main/configs/TUM_RGBD/freiburg1_desk2.yaml
V40=/root/autodl-tmp/v40_stage3_scaled.py
ALPHA_SUM=/root/autodl-tmp/psnr_tum/room/alpha_ablation/summary.txt
mkdir -p "$BASE"

echo "===== opacity decay START $(date +%H:%M:%S) =====" | tee "$BASE/summary.txt"
echo "waiting for alpha ablation to finish..." | tee -a "$BASE/summary.txt"

# Wait up to 3h for the alpha ablation to complete.
for i in $(seq 1 180); do
  if grep -q "alpha ablation DONE" "$ALPHA_SUM" 2>/dev/null; then break; fi
  sleep 60
done
echo "alpha ablation finished, starting opacity-decay run $(date +%H:%M:%S)" | tee -a "$BASE/summary.txt"

run_one () {
  local TAG="$1"; shift
  local OUT="$BASE/$TAG"
  echo "----- [$TAG] Stage3 $(date +%H:%M:%S) -----" | tee -a "$BASE/summary.txt"
  cd /root/autodl-tmp/Splat-SLAM-main
  $PY -u "$V40" --data_npz "$NPZ" --output_dir "$OUT" --alpha 1.0 --depth_mode scaled_none --opacity_decay > "$OUT.stage3.log" 2>&1
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

run_one od_decay_a1

echo "===== opacity decay DONE $(date +%H:%M:%S) =====" | tee -a "$BASE/summary.txt"
