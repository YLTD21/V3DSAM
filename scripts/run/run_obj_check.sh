#!/bin/bash
PY=/root/miniconda3/envs/sgs/bin/python
for s in room 360; do
  echo "=== $s start $(date) ==="
  $PY /root/autodl-tmp/semantic_object_check.py \
    --image_dir /root/autodl-tmp/tum_work/rgbd_dataset_freiburg1_$s/rgb \
    --mask_dir /root/autodl-tmp/semantic_out/$s/loop_figures \
    --seq $s
  echo "=== $s done $(date) ==="
done
echo "=== ALL DONE $(date) ==="
