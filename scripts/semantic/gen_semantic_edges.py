#!/usr/bin/env python3
"""Standalone: build semantic-BOUNDARY edge maps at TRAINING resolution from
kf_semantic.npz label maps (Lever 1 of "semantic -> PSNR").

A Gaussian is "on a semantic boundary" iff its projected centre lands on a
label-change pixel. The idea: floater Gaussians concentrate at semantic
boundaries (colour discontinuities the model can't explain with a real
surface), whereas the textureless *interior* of walls/floors is low-gradient
but semantically coherent and must be SPARED by opacity decay.

Input  kf_semantic.npz:
  labels       : (K, H_full, W_full) int16 argmax label map (-1 = no class)
  kf_indices   : (K,) export frame indices (== vggt_kf_indices)
  img_H, img_W : full-res mask H/W

Resize pipeline EXACTLY mirrors NPZDataset in v40 (resize -> crop):
  (H_full, W_full)  --NEAREST-->  (H_out+2*H_edge, W_out+2*W_edge)
  --crop--> (H_out, W_out)

Boundary = pixel label != any 4-neighbour label (pad 'edge' to avoid wrap
artefacts), then dilated by `dilate` px so the decay band catches Gaussian
centres that sit just off the boundary.

Output semantic_edges.npz:
  kf_indices : (K,)
  edges      : (K, H_out, W_out) uint8 (1 = semantic boundary)
  H_out, W_out, H_edge, W_edge, dilate

Does NOT modify v3dsam source. Run on AutoDL (any env with numpy + cv2):

  /root/miniconda3/envs/sgs/bin/python gen_semantic_edges.py \
    --kf_semantic /root/autodl-tmp/psnr_tum/room/stage1_scaled/kf_semantic.npz \
    --output      /root/autodl-tmp/psnr_tum/room/stage1_scaled/semantic_edges.npz
"""
import os, sys, argparse
import numpy as np
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kf_semantic", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--W_out", type=int, default=512)
    ap.add_argument("--H_out", type=int, default=384)
    ap.add_argument("--W_edge", type=int, default=8)
    ap.add_argument("--H_edge", type=int, default=8)
    ap.add_argument("--dilate", type=int, default=2, help="boundary dilation radius (px)")
    args = ap.parse_args()

    d = np.load(args.kf_semantic, allow_pickle=True)
    labels = d["labels"]                # (K, H_full, W_full) int16
    kf_indices = d["kf_indices"].astype(np.int32)
    K = labels.shape[0]
    W_out_edge = args.W_out + 2 * args.W_edge
    H_out_edge = args.H_out + 2 * args.H_edge

    edges = np.zeros((K, args.H_out, args.W_out), dtype=np.uint8)
    coverage = []
    for k in range(K):
        lab = labels[k]  # (H_full, W_full) int16
        # resize NEAREST (preserve labels) -> crop (mirror NPZDataset)
        lab_r = cv2.resize(lab, (W_out_edge, H_out_edge), interpolation=cv2.INTER_NEAREST)
        lab_r = lab_r[args.H_edge:args.H_edge + args.H_out,
                      args.W_edge:args.W_edge + args.W_out]
        # boundary = self != any 4-neighbour; pad 'edge' to avoid wrap artefact
        p = np.pad(lab_r, 1, mode="edge")
        b = (
            (lab_r != p[0:-2, 1:-1]) |   # up
            (lab_r != p[2:,   1:-1]) |   # down
            (lab_r != p[1:-1, 0:-2]) |   # left
            (lab_r != p[1:-1, 2:  ])     # right
        )
        if args.dilate > 0:
            kernel = np.ones((2 * args.dilate + 1, 2 * args.dilate + 1), np.uint8)
            b = cv2.dilate(b.astype(np.uint8), kernel).astype(bool)
        edges[k] = b.astype(np.uint8)
        coverage.append(b.mean())

    np.savez_compressed(
        args.output,
        kf_indices=kf_indices,
        edges=edges,
        H_out=args.H_out, W_out=args.W_out,
        H_edge=args.H_edge, W_edge=args.W_edge,
        dilate=args.dilate,
    )
    cov = np.asarray(coverage)
    print(f"[gen_semantic_edges] {K} keyframes -> {args.output}")
    print(f"  resolution {args.W_out}x{args.H_out} (edge crop {args.W_edge}/{args.H_edge}, dilate {args.dilate})")
    print(f"  edge coverage: mean {cov.mean()*100:.1f}%  min {cov.min()*100:.1f}%  max {cov.max()*100:.1f}%")
    print(f"  size: {os.path.getsize(args.output)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
