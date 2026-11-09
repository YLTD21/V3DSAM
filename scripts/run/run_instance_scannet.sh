#!/bin/bash
# 自运行：ScanNet 持久实例身份流水线（图10 改 ScanNet，去掩膜，语义+ID 用我们流水线）
#   scene0000/0059/0106 三场景 -> kf_semantic(SAM3) -> instances(聚类) -> voted(投票) -> 3x5 grid
# 结果：/root/autodl-tmp/psnr_tum/scannet_figs/fig_instance_grid_scannet.png/.pdf
# 日志：/root/autodl-tmp/psnr_tum/run_instance_scannet.log
# 预计：~3s/kf x 1256 kf ~ 60min（SAM3 为主）
set -x
cd /root/autodl-tmp
PY=/root/miniconda3/envs/sgs/bin/python
BASE=/root/autodl-tmp/ablation/scannet

SCENES=(scene0000_00 scene0059_00 scene0106_00)

for s in "${SCENES[@]}"; do
  echo "===== [$s] 1/3 gen_kf_semantic (SAM3, all keyframes) ====="
  $PY /root/autodl-tmp/gen_kf_semantic.py \
      --data_npz "$BASE/$s/_stage3_input_after.npz" \
      --output   "$BASE/$s/kf_semantic.npz" \
      2>&1 | grep -vE "FutureWarning|warnings.warn|timm"

  echo "===== [$s] 2/3 gen_semantic_instances (vote + voxel-CC cluster + reid) ====="
  $PY /root/autodl-tmp/gen_semantic_instances.py \
      --stage3   "$BASE/$s/_stage3_input_after.npz" \
      --semantic "$BASE/$s/kf_semantic.npz" \
      --out_dir  "$BASE/$s/semantic_instances" \
      2>&1 | grep -vE "FutureWarning|warnings.warn|timm"

  echo "===== [$s] 3/3 gen_voted_instances (occlusion-aware all-view voting) ====="
  $PY /root/autodl-tmp/gen_voted_instances.py \
      --stage3         "$BASE/$s/_stage3_input_after.npz" \
      --semantic       "$BASE/$s/kf_semantic.npz" \
      --instance_cloud "$BASE/$s/semantic_instances/instance_cloud.ply" \
      --out_json       "$BASE/$s/semantic_instances/voted_instances.json" \
      2>&1 | grep -vE "FutureWarning|warnings.warn|timm"
done

echo "===== render 3x5 persistent-instance grid (no mask) ====="
$PY /root/autodl-tmp/psnr_tum/gen_instance_grid_scannet.py \
    2>&1 | grep -vE "FutureWarning|warnings.warn|timm"

echo "===== DONE ====="
ls -la /root/autodl-tmp/psnr_tum/scannet_figs/fig_instance_grid_scannet.* 2>/dev/null
echo "RUN_INSTANCE_SCANNET_DONE $(date +%s)"
