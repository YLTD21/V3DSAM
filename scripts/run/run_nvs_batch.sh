#!/bin/bash
# 表 8 补测：为 8 个 TUM fr1 序列跑 held-out NVS eval（room 已有 9.21，跳过）。
# 目的：量化"重建几何塌缩"是否在全部序列上普遍成立（训练关键帧 PSNR 高、held-out NVS 低）。
#
# 用法（AutoDL RTX 4090，sgs 环境）：
#   scp run_nvs_batch.sh root@autodl:/root/autodl-tmp/
#   ssh root@autodl
#   cd /root/autodl-tmp && nohup bash run_nvs_batch.sh > nvs_batch.log 2>&1 &
# 结果写入 /root/autodl-tmp/psnr_tum/nvs_batch_summary.txt，跑完再查。
#
# 关键：config 固定用 freiburg1_desk2.yaml —— 这是 v38_stage3_splatslam.py 的
# 默认 config（batch_psnr.sh 建 8 个序列重建时没传 --config，用的就是这个默认），
# 它 inherit_from tum.yaml（W_out=512 H_out=384 W_edge=8 H_edge=8），fr1 各序列
# 内参相同。nvs_eval 必须用同一 config 才能与重建口径一致。
set -u
PY=/root/miniconda3/envs/sgs/bin/python
export LD_LIBRARY_PATH=/root/miniconda3/envs/sgs/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export XFORMERS_DISABLED=1

CFG=/root/autodl-tmp/Splat-SLAM-main/configs/TUM_RGBD/freiburg1_desk2.yaml
NAMES="360 desk desk2 floor plant rpy teddy xyz"
SUMMARY=/root/autodl-tmp/psnr_tum/nvs_batch_summary.txt

echo "===== NVS batch START $(date +%F_%H:%M:%S) =====" > "$SUMMARY"
echo "seq  train_PSNR(from psnr.txt)  NVS_PSNR  NVS_SSIM" >> "$SUMMARY"

for SEQ in $NAMES; do
  NPZ=/root/autodl-tmp/psnr_tum/$SEQ/stage1/_stage3_input.npz
  PLY=/root/autodl-tmp/psnr_tum/$SEQ/splatslam/gaussians_final.ply
  OUT=/root/autodl-tmp/psnr_tum/$SEQ/nvs_eval

  if [ ! -f "$NPZ" ] || [ ! -f "$PLY" ]; then
    echo "$SEQ  MISSING npz/ply (skip): npz=$NPZ ply=$PLY" | tee -a "$SUMMARY"
    continue
  fi

  echo "----- [$SEQ] NVS eval $(date +%H:%M:%S) -----" | tee -a "$SUMMARY"
  cd /root/autodl-tmp
  $PY -u nvs_eval.py --data_npz "$NPZ" --ply "$PLY" --output_dir "$OUT" --config "$CFG" \
    > "/root/autodl-tmp/psnr_tum/$SEQ/nvs_eval.log" 2>&1

  TRAIN=$(cat /root/autodl-tmp/psnr_tum/$SEQ/splatslam/psnr.txt 2>/dev/null | tr -d '\n')
  NVS=$(cat "$OUT/nvs.txt" 2>/dev/null | tr -d '\n')
  echo "$SEQ  $TRAIN  $NVS" | tee -a "$SUMMARY"
  echo "" >> "$SUMMARY"
done

echo "===== NVS batch DONE $(date +%F_%H:%M:%S) =====" | tee -a "$SUMMARY"
echo "结果：$SUMMARY"
