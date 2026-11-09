#!/bin/bash
# Promote kf_semantic_v2 -> canonical, regenerate semantic cloud + instance grid figure.
set -u
cd /root/autodl-tmp || exit 1
PY=/root/miniconda3/envs/sgs/bin/python
LOG=/root/autodl-tmp/semantic_rerun.log
SCENES="scene0000_00 scene0059_00 scene0106_00"
{
  echo "=== pipeline start $(date) ==="

  for s in $SCENES; do
    b=/root/autodl-tmp/ablation/scannet/$s
    [ -f "$b/kf_semantic.npz" ] && [ ! -f "$b/kf_semantic_old.npz" ] && mv "$b/kf_semantic.npz" "$b/kf_semantic_old.npz"
    mv "$b/kf_semantic_v2.npz" "$b/kf_semantic.npz"
    [ -d "$b/semantic_cloud_10m" ] && [ ! -d "$b/semantic_cloud_10m_old" ] && mv "$b/semantic_cloud_10m" "$b/semantic_cloud_10m_old"
  done
  echo "=== promoted kf_semantic_v2 + backed up old clouds ==="

  $PY -u gen_semantic_cloud_10m.py --scenes $SCENES
  echo "=== semantic cloud done exit=$? ==="

  $PY -u gen_instance_grid_geometric.py --out /root/autodl-tmp/scannet_figs_all/fig_instance_grid_geometric_v2
  echo "=== figure done exit=$? ==="

  echo "=== ALL DONE $(date) ==="
} >> "$LOG" 2>&1
