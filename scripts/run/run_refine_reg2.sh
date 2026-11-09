#!/bin/bash
# 2区 attempt — sweep 2: isotropic scale regularizer (the one map() uses that
# final_refine silently dropped). Runs AFTER sweep 1 (run_refine_reg.sh) finishes
# to avoid GPU contention; chains on the same room scene.
set -euo pipefail
PY=/root/miniconda3/envs/sgs/bin/python
export LD_LIBRARY_PATH=/root/miniconda3/envs/sgs/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export XFORMERS_DISABLED=1

NPZ=/root/autodl-tmp/psnr_tum/room/stage1_scaled/_stage3_input.npz
CFG=/root/autodl-tmp/Splat-SLAM-main/configs/TUM_RGBD/freiburg1_desk2.yaml
V40=/root/autodl-tmp/v40_stage3_scaled.py
BASE=/root/autodl-tmp/psnr_tum/room/refine_reg2
SUM=$BASE/summary.txt
SW1=/root/autodl-tmp/psnr_tum/room/refine_reg/summary.txt
mkdir -p "$BASE"

echo "===== refine-reg sweep2 (isotropic) START $(date +%H:%M:%S) =====" | tee "$SUM"
echo "waiting for sweep1 to finish..." | tee -a "$SUM"

# Wait up to 3h for sweep1.
for i in $(seq 1 180); do
  if grep -q "refine-reg sweep DONE" "$SW1" 2>/dev/null; then break; fi
  sleep 60
done
echo "sweep1 done (or timeout), starting sweep2 $(date +%H:%M:%S)" | tee -a "$SUM"

# 2) patch (idempotent)
echo "----- patch isotropic $(date +%H:%M:%S) -----" | tee -a "$SUM"
$PY -u patch_v42_isotropic.py | tee -a "$SUM"

run_one () {
  local TAG="$1"; shift
  local OUT="$BASE/$TAG"
  echo "----- [$TAG] Stage3 $(date +%H:%M:%S) -----" | tee -a "$SUM"
  cd /root/autodl-tmp/Splat-SLAM-main
  $PY -u "$V40" --data_npz "$NPZ" --output_dir "$OUT" --alpha 1.0 --depth_mode scaled_none \
    "$@" > "$OUT.stage3.log" 2>&1
  echo -n "[$TAG] train PSNR: " | tee -a "$SUM"
  cat "$OUT/psnr.txt" 2>/dev/null | tee -a "$SUM" || echo "NO psnr.txt" | tee -a "$SUM"

  echo "----- [$TAG] NVS eval $(date +%H:%M:%S) -----" | tee -a "$SUM"
  cd /root/autodl-tmp
  $PY -u nvs_eval.py --data_npz "$NPZ" --ply "$OUT/gaussians_final.ply" \
    --output_dir "$OUT/nvs_eval" --config "$CFG" > "$OUT.nvs.log" 2>&1
  echo -n "[$TAG] NVS: " | tee -a "$SUM"
  cat "$OUT/nvs_eval/nvs.txt" 2>/dev/null | tee -a "$SUM" || echo "NO nvs.txt" | tee -a "$SUM"
  echo "" | tee -a "$SUM"
}

# isotropic only (weight 10, matches map() exactly)
run_one iso10 \
  --refine_isotropic_weight 10.0

# full canonical restore: isotropic 10 + opacity reset 3000 + prune 1000/0.005
run_one iso10_reset3k_p1k \
  --refine_isotropic_weight 10.0 --refine_opacity_reset_interval 3000 \
  --refine_prune_interval 1000 --refine_prune_opacity 0.005

# gentler isotropic (weight 1) + reset, in case 10 is too strong over 26000 iters
run_one iso1_reset3k_p1k \
  --refine_isotropic_weight 1.0 --refine_opacity_reset_interval 3000 \
  --refine_prune_interval 1000 --refine_prune_opacity 0.005

echo "===== refine-reg sweep2 DONE $(date +%H:%M:%S) =====" | tee -a "$SUM"
