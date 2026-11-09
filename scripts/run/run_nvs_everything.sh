#!/bin/bash
# 全量 NVS 大扫：把所有还没测过 held-out NVS 的模型一次性测完，给"谁(若有)能救
# 塌缩"一个完整答案。包含：
#   (A) 13 个已有模型（scale/alpha 消融、opacity decay 各档、refine5000 少迭代、semantic_gate）
#   (B) Stage B 离面剪枝的 4 个阈值模型
# 等 sweep2（isotropic）收尾后串行跑，避免 GPU 争抢。
set -u
PY=/root/miniconda3/envs/sgs/bin/python
export LD_LIBRARY_PATH=/root/miniconda3/envs/sgs/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export XFORMERS_DISABLED=1

NPZ=/root/autodl-tmp/psnr_tum/room/stage1_scaled/_stage3_input.npz
CFG=/root/autodl-tmp/Splat-SLAM-main/configs/TUM_RGBD/freiburg1_desk2.yaml
ROOM=/root/autodl-tmp/psnr_tum/room
BASE=$ROOM/nvs_everything
SUM=$BASE/summary.txt
SW2=$ROOM/refine_reg2/summary.txt
mkdir -p "$BASE"

echo "===== nvs-everything sweep START $(date +%H:%M:%S) =====" | tee "$SUM"
echo "waiting for sweep2 (isotropic)..." | tee -a "$SUM"
for i in $(seq 1 180); do
  grep -q "refine-reg sweep2 DONE" "$SW2" 2>/dev/null && break
  sleep 60
done
echo "sweep2 done (or timeout). starting prune + NVS sweep $(date +%H:%M:%S)" | tee -a "$SUM"

# ---- Stage B: prune baseline by distance ----
echo "----- prune $(date +%H:%M:%S) -----" | tee -a "$SUM"
$PY -u prune_by_distance.py \
  --gaussians "$ROOM/scale_ablation/scaled_none/gaussians_final.ply" \
  --cloud "$ROOM/semantic_instances/instance_cloud.ply" \
  --out_dir "$BASE" --thresholds "0.40,0.20,0.10,0.05" | tee -a "$SUM"

eval_model () {
  local TAG="$1"; local PLY="$2"
  if [ ! -f "$PLY" ]; then echo "[$TAG] MISSING $PLY" | tee -a "$SUM"; return; fi
  echo "----- [$TAG] NVS $(date +%H:%M:%S) -----" | tee -a "$SUM"
  cd /root/autodl-tmp
  $PY -u nvs_eval.py --data_npz "$NPZ" --ply "$PLY" --output_dir "$BASE/nvs_$TAG" --config "$CFG" \
    > "$BASE/nvs_$TAG.log" 2>&1
  echo -n "[$TAG] NVS: " | tee -a "$SUM"
  cat "$BASE/nvs_$TAG/nvs.txt" 2>/dev/null | tee -a "$SUM" || echo "NO nvs.txt" | tee -a "$SUM"
  echo "" | tee -a "$SUM"
}

# reference (known): baseline scaled_none NVS=9.21
echo "[baseline scaled_none] NVS (reference): N=1176 PSNR_mean=9.21" | tee -a "$SUM"
echo "" | tee -a "$SUM"

# ---- (A) existing models ----
eval_model splatslam          "$ROOM/splatslam/gaussians_final.ply"
eval_model raw_norm           "$ROOM/scale_ablation/raw_norm/gaussians_final.ply"
eval_model raw_none           "$ROOM/scale_ablation/raw_none/gaussians_final.ply"
eval_model scaled_none_a0_8   "$ROOM/alpha_ablation/scaled_none_a0.8/gaussians_final.ply"
eval_model raw_norm_a0_8      "$ROOM/alpha_ablation/raw_norm_a0.8/gaussians_final.ply"
eval_model scaled_none_a0_5   "$ROOM/alpha_ablation/scaled_none_a0.5/gaussians_final.ply"
eval_model raw_norm_a0_5      "$ROOM/alpha_ablation/raw_norm_a0.5/gaussians_final.ply"
eval_model od_decay_a1        "$ROOM/opacity_decay/od_decay_a1/gaussians_final.ply"
eval_model r999               "$ROOM/decay_sweep/r999/gaussians_final.ply"
eval_model r9995              "$ROOM/decay_sweep/r9995/gaussians_final.ply"
eval_model r9998              "$ROOM/decay_sweep/r9998/gaussians_final.ply"
eval_model refine5000         "$ROOM/decay_sweep/refine5000/gaussians_final.ply"
eval_model semantic_gate      "$ROOM/semantic_gate/gaussians_final.ply"

# ---- (B) pruned by distance ----
eval_model pruned_0_40        "$BASE/pruned_0_40.ply"
eval_model pruned_0_20        "$BASE/pruned_0_20.ply"
eval_model pruned_0_10        "$BASE/pruned_0_10.ply"
eval_model pruned_0_05        "$BASE/pruned_0_05.ply"

echo "===== nvs-everything sweep DONE $(date +%H:%M:%S) =====" | tee -a "$SUM"
