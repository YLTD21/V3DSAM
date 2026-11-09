#!/bin/bash
# Stage B: geometric-anchor floater pruning -> NVS eval sweep.
# Prunes the BASELINE collapsed model (scaled_none) by distance-to-fused-cloud and
# measures held-out NVS at several thresholds.  Waits for sweep2 (isotropic) to
# finish so the GPU is free, then runs (prune is CPU, nvs_eval is GPU).
set -euo pipefail
PY=/root/miniconda3/envs/sgs/bin/python
export LD_LIBRARY_PATH=/root/miniconda3/envs/sgs/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export XFORMERS_DISABLED=1

NPZ=/root/autodl-tmp/psnr_tum/room/stage1_scaled/_stage3_input.npz
CFG=/root/autodl-tmp/Splat-SLAM-main/configs/TUM_RGBD/freiburg1_desk2.yaml
BASE=/root/autodl-tmp/psnr_tum/room/floater_prune
SUM=$BASE/summary.txt
SW2=/root/autodl-tmp/psnr_tum/room/refine_reg2/summary.txt
mkdir -p "$BASE"

echo "===== floater-prune sweep START $(date +%H:%M:%S) =====" | tee "$SUM"
echo "waiting for sweep2 (isotropic) to finish..." | tee -a "$SUM"
for i in $(seq 1 180); do
  if grep -q "refine-reg sweep2 DONE" "$SW2" 2>/dev/null; then break; fi
  sleep 60
done
echo "sweep2 done (or timeout), pruning + NVS $(date +%H:%M:%S)" | tee -a "$SUM"

echo "----- prune $(date +%H:%M:%S) -----" | tee -a "$SUM"
$PY -u prune_by_distance.py \
  --gaussians /root/autodl-tmp/psnr_tum/room/scale_ablation/scaled_none/gaussians_final.ply \
  --cloud /root/autodl-tmp/psnr_tum/room/semantic_instances/instance_cloud.ply \
  --out_dir "$BASE" --thresholds "0.40,0.20,0.10,0.05" | tee -a "$SUM"

# baseline reference
echo -n "[baseline] NVS: " | tee -a "$SUM"
cat /root/autodl-tmp/psnr_tum/room/scale_ablation/scaled_none/nvs_eval/nvs.txt 2>/dev/null | tee -a "$SUM" || echo "no baseline nvs.txt" | tee -a "$SUM"
echo "" | tee -a "$SUM"

for T in 0_40 0_20 0_10 0_05; do
  PLY="$BASE/pruned_${T}.ply"
  [ -f "$PLY" ] || continue
  echo "----- [pruned_$T] NVS $(date +%H:%M:%S) -----" | tee -a "$SUM"
  cd /root/autodl-tmp
  $PY -u nvs_eval.py --data_npz "$NPZ" --ply "$PLY" --output_dir "$BASE/nvs_$T" --config "$CFG" \
    > "$BASE/nvs_$T.log" 2>&1
  echo -n "[pruned_$T] NVS: " | tee -a "$SUM"
  cat "$BASE/nvs_$T/nvs.txt" 2>/dev/null | tee -a "$SUM" || echo "NO nvs.txt" | tee -a "$SUM"
  echo "" | tee -a "$SUM"
done

echo "===== floater-prune sweep DONE $(date +%H:%M:%S) =====" | tee -a "$SUM"
