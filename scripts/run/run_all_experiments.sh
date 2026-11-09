#!/bin/bash
# ============================================================================
# V3DSAM 3x5 持久实例图 —— 全自动幂等流水线（自运行，无需监控）
#
# 做三件事：
#   1) desk2 实例身份数据补齐（幂等：已存在则跳过）
#        gen_kf_semantic      -> stage1/kf_semantic.npz     (SAM3 语义, ~5min GPU)
#        gen_semantic_instances -> semantic_instances/      (聚类 -> instances.json)
#        gen_voted_instances  -> voted_instances.json       (全视角投票)
#   2) 生成 3x5 图（room / 360 / desk2 各 5 帧重叠关键帧，全前景实例 + 质心 + ID + 类别）
#   3) 自检：核对三场景实例数与产物文件，写完成标记
#
# 用法（远程后台，完全脱离 ssh 会话，立即返回）：
#   setsid bash run_all_experiments.sh </dev/null >run_all.log 2>&1 &
# 完成后看 /root/autodl-tmp/run_all.log 末尾的 "[ALL DONE]"。
# ============================================================================
set -u
cd /root/autodl-tmp
PY=/root/miniconda3/envs/sgs/bin/python
LOG=/root/autodl-tmp/run_all.log
DONE_MARK=/root/autodl-tmp/run_all.done
rm -f "$DONE_MARK"

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*"; }

exec > >(tee -a "$LOG") 2>&1

log "===== V3DSAM 3x5 pipeline start ====="

# ---------------------------------------------------------------- desk2 数据
BASE=/root/autodl-tmp/psnr_tum/desk2
STAGE3=$BASE/stage1/_stage3_input.npz
SEM=$BASE/stage1/kf_semantic.npz
OUT=$BASE/semantic_instances
mkdir -p "$OUT"

if [ -s "$SEM" ]; then
  log "desk2 kf_semantic.npz 已存在($(du -h "$SEM" | cut -f1))，跳过 SAM3"
else
  log "1/5 desk2 SAM3 语义 (gen_kf_semantic) ..."
  $PY gen_kf_semantic.py --data_npz "$STAGE3" --output "$SEM" 2>&1 | tail -6
fi

if [ -s "$OUT/instances.json" ] && [ -s "$OUT/instance_cloud.ply" ]; then
  log "desk2 instances.json 已存在，跳过聚类"
else
  log "2/5 desk2 实例聚类 (gen_semantic_instances) ..."
  $PY gen_semantic_instances.py --stage3 "$STAGE3" --semantic "$SEM" --out_dir "$OUT" 2>&1 | tail -8
fi

if [ -s "$OUT/voted_instances.json" ]; then
  log "desk2 voted_instances.json 已存在，跳过投票"
else
  log "3/5 desk2 全视角投票 (gen_voted_instances) ..."
  $PY gen_voted_instances.py --stage3 "$STAGE3" --semantic "$SEM" \
    --instance_cloud "$OUT/instance_cloud.ply" --out_json "$OUT/voted_instances.json" 2>&1 | tail -8
fi

# ---------------------------------------------------------------- 3x5 渲染
FIG=/root/autodl-tmp/psnr_tum/fig13_instance_grid_3x5.png
log "4/5 生成 3x5 持久实例图 (gen_instance_grid_3x5) ..."
$PY gen_instance_grid_3x5.py 2>&1 | tail -20

# ---------------------------------------------------------------- 自检
log "5/5 自检 ..."
if [ -s "$FIG" ]; then
  log "OK 图已生成: $FIG ($(du -h "$FIG" | cut -f1))"
else
  log "WARN 图未生成，请查上文报错"
fi
for s in room 360 desk2; do
  if [ -f "/root/autodl-tmp/psnr_tum/$s/semantic_instances_postfilter/instances.json" ]; then
    n=$($PY -c "import json;print(len(json.load(open('/root/autodl-tmp/psnr_tum/$s/semantic_instances_postfilter/instances.json'))['instances']))")
  elif [ -f "/root/autodl-tmp/psnr_tum/$s/semantic_instances/instances.json" ]; then
    n=$($PY -c "import json;print(len(json.load(open('/root/autodl-tmp/psnr_tum/$s/semantic_instances/instances.json'))['instances']))")
  else
    n="MISSING"
  fi
  log "  $s instances=$n"
done

touch "$DONE_MARK"
log "===== ALL DONE ====="
