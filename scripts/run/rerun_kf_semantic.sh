#!/bin/bash
# Regenerate kf_semantic for all 3 ScanNet scenes with fixed prompts + score-weighted argmax.
# Writes kf_semantic_v2.npz (old kf_semantic.npz preserved for comparison).
set -u
cd /root/autodl-tmp || exit 1
PY=/root/miniconda3/envs/sgs/bin/python
LOG=/root/autodl-tmp/kf_semantic_rerun.log
{
  echo "=== rerun start $(date) ==="
  for s in scene0000_00 scene0059_00 scene0106_00; do
    echo "=== $s start $(date) ==="
    $PY -u gen_kf_semantic.py \
      --data_npz /root/autodl-tmp/ablation/scannet/$s/_stage3_input_after.npz \
      --output   /root/autodl-tmp/ablation/scannet/$s/kf_semantic_v2.npz
    echo "=== $s done $(date) exit=$? ==="
  done
  echo "=== ALL DONE $(date) ==="
} >> "$LOG" 2>&1
