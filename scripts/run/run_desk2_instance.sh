#!/bin/bash
# desk2 实例身份流水线：SAM3 语义 -> 聚类 -> 投票 (standalone, 不改 v3dsam 源码)
set -e
cd /root/autodl-tmp
PY=/root/miniconda3/envs/sgs/bin/python
BASE=/root/autodl-tmp/psnr_tum/desk2
STAGE3=$BASE/stage1/_stage3_input.npz
SEM=$BASE/stage1/kf_semantic.npz
OUT=$BASE/semantic_instances
mkdir -p $OUT

echo "[pipeline] 1/3 gen_kf_semantic (SAM3, ~98 kf) ..."
$PY gen_kf_semantic.py --data_npz $STAGE3 --output $SEM 2>&1 | tail -4

echo "[pipeline] 2/3 gen_semantic_instances (cluster) ..."
$PY gen_semantic_instances.py --stage3 $STAGE3 --semantic $SEM --out_dir $OUT 2>&1 | tail -8

echo "[pipeline] 3/3 gen_voted_instances (vote) ..."
$PY gen_voted_instances.py --stage3 $STAGE3 --semantic $SEM \
  --instance_cloud $OUT/instance_cloud.ply --out_json $OUT/voted_instances.json 2>&1 | tail -8

echo "[pipeline] DONE -> $OUT"
