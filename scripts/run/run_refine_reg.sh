#!/bin/bash
# 2区 attempt: restore 3DGS regularization to final_refine (opacity reset + prune).
#   baseline (scaled_none, no reg):  train=22.81  NVS=9.21  SSIM=0.3405
#   plain decay 0.995 (dead end):    train=14.52  NVS=10.08 (over-kills surfaces)
# Hypothesis: periodic opacity reset (->0.01) + low-opacity prune removes the
#   floating blobs that overfit train views, lifting held-out NVS without
#   destroying train surfaces.
# Go/no-go: NVS > 12 dB AND train stays >= ~20 dB -> positive contribution (2区).
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
BASE=/root/autodl-tmp/psnr_tum/room/refine_reg
SUM=$BASE/summary.txt
mkdir -p "$BASE"

echo "===== refine-reg sweep START $(date +%H:%M:%S) =====" | tee "$SUM"

# 1) patch (idempotent)
echo "----- patch splat-slam $(date +%H:%M:%S) -----" | tee -a "$SUM"
$PY -u patch_v41_refine_reg.py | tee -a "$SUM"

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

# canonical: reset every 3000 (opacity->0.01), no prune
run_one reset3k \
  --refine_opacity_reset_interval 3000

# reset + low-opacity prune (0.005) every 1000
run_one reset3k_p1k \
  --refine_opacity_reset_interval 3000 --refine_prune_interval 1000 --refine_prune_opacity 0.005

# more aggressive: reset 1000 + prune 500
run_one reset1k_p500 \
  --refine_opacity_reset_interval 1000 --refine_prune_interval 500 --refine_prune_opacity 0.005

echo "===== refine-reg sweep DONE $(date +%H:%M:%S) =====" | tee -a "$SUM"
