#!/bin/bash
# Lever 1: semantic-edge-gated opacity decay (semantic -> PSNR).
#   baseline (no decay):        train=22.81  NVS=9.21   SSIM=0.3405
#   plain decay 0.995:          train=14.52  NVS=10.08  SSIM=0.4275  (over-kills)
#   plain decay 0.999/0.9995:   train=21.19/22.43  NVS=9.30/9.28  (no NVS gain)
# Hypothesis: gate the decay to SEMANTIC BOUNDARIES so we kill boundary
# floaters (NVS up) while sparing interior textureless surfaces (train stays).
# Go/no-go: NVS > 10.5 dB AND train not dropping below ~20 dB.
set -u
PY=/root/miniconda3/envs/sgs/bin/python
export LD_LIBRARY_PATH=/root/miniconda3/envs/sgs/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export XFORMERS_DISABLED=1

NPZ=/root/autodl-tmp/psnr_tum/room/stage1_scaled/_stage3_input.npz
KSEM=/root/autodl-tmp/psnr_tum/room/stage1_scaled/kf_semantic.npz
ESEM=/root/autodl-tmp/psnr_tum/room/stage1_scaled/semantic_edges.npz
CFG=/root/autodl-tmp/Splat-SLAM-main/configs/TUM_RGBD/freiburg1_desk2.yaml
V40=/root/autodl-tmp/v40_stage3_scaled.py
OUT=/root/autodl-tmp/psnr_tum/room/semantic_gate
SUM=$OUT/summary.txt
mkdir -p "$OUT"

echo "===== semantic-gate run START $(date +%H:%M:%S) =====" | tee "$SUM"

# 1) edges (skip if exists)
if [ ! -f "$ESEM" ]; then
  echo "----- gen semantic edges $(date +%H:%M:%S) -----" | tee -a "$SUM"
  $PY -u gen_semantic_edges.py --kf_semantic "$KSEM" --output "$ESEM" | tee -a "$SUM"
fi

# 2) patch (idempotent)
echo "----- patch splat-slam $(date +%H:%M:%S) -----" | tee -a "$SUM"
$PY -u patch_v40_semantic_gate.py | tee -a "$SUM"

# 3) Stage3 with edge-gated decay
echo "----- Stage3 (edge-gated 0.995) $(date +%H:%M:%S) -----" | tee -a "$SUM"
cd /root/autodl-tmp/Splat-SLAM-main
$PY -u "$V40" --data_npz "$NPZ" --output_dir "$OUT" --alpha 1.0 --depth_mode scaled_none \
  --opacity_decay --opacity_decay_rate 0.995 --semantic_edges "$ESEM" > "$OUT.stage3.log" 2>&1
echo -n "[edge-gate] train PSNR: " | tee -a "$SUM"
cat "$OUT/psnr.txt" 2>/dev/null | tee -a "$SUM" || echo "NO psnr.txt" | tee -a "$SUM"

# 4) NVS eval
echo "----- NVS eval $(date +%H:%M:%S) -----" | tee -a "$SUM"
cd /root/autodl-tmp
$PY -u nvs_eval.py --data_npz "$NPZ" --ply "$OUT/gaussians_final.ply" \
  --output_dir "$OUT/nvs_eval" --config "$CFG" > "$OUT.nvs.log" 2>&1
echo -n "[edge-gate] NVS: " | tee -a "$SUM"
cat "$OUT/nvs_eval/nvs.txt" 2>/dev/null | tee -a "$SUM" || echo "NO nvs.txt" | tee -a "$SUM"

echo "===== semantic-gate run DONE $(date +%H:%M:%S) =====" | tee -a "$SUM"
