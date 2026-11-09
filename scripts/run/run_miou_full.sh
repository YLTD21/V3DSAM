#!/bin/bash
# 自运行：全量 mIoU（4 场景，stride 20 / 每场景 50 帧）+ 重生成语义点云图（用修正后的 raw->nyu40 映射）
# 结果：/root/autodl-tmp/psnr_tum/scannet_figs/{miou.json, fig_scannet_pcd.png, fig_scannet_sem_traj.png}
set -x
cd /root/autodl-tmp/psnr_tum
PY=/root/miniconda3/envs/sgs/bin/python

echo "==== [1/2] full mIoU (4 scenes) ===="
$PY measure_miou.py --stride 20 --max_frames 50 2>&1 | grep -vE "FutureWarning|warnings.warn|timm"

echo "==== [2/2] regenerate semantic point-cloud figures (fixed mapping) ===="
$PY make_scannet_figures.py --scene scene0000_00 --only pcd,semtraj 2>&1 | grep -vE "FutureWarning|warnings.warn|timm"

echo "==== DONE ===="
ls -la /root/autodl-tmp/psnr_tum/scannet_figs/*.png /root/autodl-tmp/psnr_tum/scannet_figs/miou.json
echo "RUN_MIOU_FULL_DONE $(date +%s)"
