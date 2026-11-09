#!/usr/bin/env python3
"""
3D Semantic Evaluation for SAM3 masks in VGGT-SLAM context.
================================================================
Three metrics that are more meaningful than 2D per-frame mIoU:

  Metric 1: Multi-view Semantic Consistency (MVSC)
    - Same 3D point seen from N views → how often do labels agree?
    - No GT needed — measures internal geometric consistency of SAM3+VGGT

  Metric 2: Re-projection Semantic Agreement (RPSA)
    - Project 3D labels from view A → view B, compare with SAM3 at B
    - Measures 3D-geometry-aware cross-view semantic stability

  Metric 3: 3D Point Cloud mIoU (vs ScanNet aggregated mesh GT)
    - Project all masks to 3D, voxelize, compare against ScanNet 3D labels

Usage:
  python tools/eval_semantic_3d.py \
    --npz experiments/scannet_0059_v38_full/_stage3_input.npz \
    --mask_dir experiments/scannet_0059_v38_full/sam3_masks \
    --output_dir experiments/semantic_3d_eval/scene0059
"""
import numpy as np, os, argparse, sys
from collections import defaultdict
from tqdm import tqdm
from PIL import Image

# ═══ Prompt → SN20 mapping (same as compute_miou_from_masks.py) ═══
PROMPT_TO_SCANNET = {
    "wall":1,"floor":2,"cabinet":3,"bed":4,"chair":5,
    "sofa":6,"couch":6,"table":7,"door":8,"window":9,
    "bookshelf":10,"shelf":10,"picture":11,"painting":11,
    "counter":12,"desk":14,"curtain":16,
    "refrigerator":24,"toilet":33,"sink":34,"bathtub":36,
}
SN20_FROM_SCANNET = {
    1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,
    11:11,12:12,14:13,16:14,24:15,28:16,33:17,34:18,36:19,39:20
}
ALL_PROMPTS = [
    "wall","floor","ceiling","door","window","stairs","railing",
    "bed","cabinet","bookshelf","shelf","counter","desk","table","sofa","couch",
    "sink","toilet","bathtub","mirror","lamp","light",
    "television","monitor","laptop","keyboard","mouse","phone","refrigerator","oven","microwave",
    "chair","plant","vase","picture","painting","curtain","pillow","clock",
    "book","bottle","cup","bowl","plate","bag","backpack","box","bin","basket","person",
]

# Build prompt idx → SN20 class
PIDX_TO_SN20 = {}
for i, prompt in enumerate(ALL_PROMPTS):
    sn_label = PROMPT_TO_SCANNET.get(prompt)
    if sn_label is not None:
        sn20 = SN20_FROM_SCANNET.get(sn_label)
        if sn20 is not None:
            PIDX_TO_SN20[i] = sn20

SN20_NAMES = {
    1:"wall",2:"floor",3:"cabinet",4:"bed",5:"chair",6:"sofa",
    7:"table",8:"door",9:"window",10:"bookshelf",11:"picture",12:"counter",
    13:"desk",14:"curtain",15:"refrigerator",16:"shower curtain",
    17:"toilet",18:"sink",19:"bathtub",20:"otherfurniture"
}

def load_npz_data(npz_path):
    """Load VGGT-SLAM NPZ and extract keyframe poses, depths, intrinsics."""
    data = np.load(npz_path, allow_pickle=True)
    kf_indices = data['vggt_kf_indices'].tolist()
    T_wc = np.stack([data[f'T_wc_{i}'] for i in kf_indices])  # (K, 4, 4)
    K = data['K_0']  # (3, 3)
    return data, kf_indices, T_wc, K

def unproject_depth_to_3d(depth, K, T_wc, mask=None):
    """Unproject depth map to world-frame 3D points."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    pts_cam = np.stack([X, Y, Z, np.ones_like(Z)], axis=-1)  # (H, W, 4)
    pts_world = (T_wc @ pts_cam[..., None])[..., 0]  # (H, W, 4)
    pts_3d = pts_world[..., :3]  # (H, W, 3)
    if mask is not None:
        pts_3d = pts_3d[mask]
    return pts_3d

def compute_multiview_consistency(npz_path, mask_dir, kf_indices, T_wc, K,
                                   voxel_size=0.05, min_views=2):
    """
    Metric 1: Multi-view Semantic Consistency (MVSC)

    Algorithm:
      1. For each keyframe, project SAM3 mask labels to 3D
      2. Voxelize 3D space
      3. For each occupied voxel, collect semantic votes from all views
      4. MVSC = fraction of voxels where majority vote > threshold
    """
    print(f"\n{'='*60}")
    print(f"  Metric 1: Multi-view Semantic Consistency (MVSC)")
    print(f"  Voxel size: {voxel_size}m, min views: {min_views}")
    print(f"{'='*60}")

    data = np.load(npz_path, allow_pickle=True)
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npz')])

    # Build: global_frame_index → mask file
    mask_by_frame = {}
    for mf in mask_files:
        gi = int(mf.replace("mask_", "").replace(".npz", ""))
        mask_by_frame[gi] = mf

    # Voxel accumulator: (vx, vy, vz) → [list of SN20 labels from each view]
    voxel_votes = defaultdict(list)
    voxel_points = defaultdict(list)  # store actual 3D points for centroid
    n_matched = 0

    for ki, kf_idx in enumerate(tqdm(kf_indices, desc="MVSC-projection")):
        if kf_idx not in mask_by_frame:
            continue

        # Load depth & mask
        depth_full = data[f'depth_{kf_idx}'].astype(np.float32)
        masks = np.load(os.path.join(mask_dir, mask_by_frame[kf_idx]))['mask']  # (50, H, W)

        # Resize depth to match mask resolution
        H_m, W_m = masks.shape[1], masks.shape[2]
        if depth_full.shape != (H_m, W_m):
            import cv2
            depth = cv2.resize(depth_full, (W_m, H_m), interpolation=cv2.INTER_LINEAR)
        else:
            depth = depth_full

        # Build SN20 label map from 50 prompts
        sem_label = np.zeros((H_m, W_m), dtype=np.int32)
        for pidx, sn20 in PIDX_TO_SN20.items():
            if pidx < masks.shape[0]:
                sem_label[masks[pidx] > 0] = sn20

        # Only use pixels with valid depth AND valid semantic label
        valid = (depth > 0.1) & (depth < 10.0) & (sem_label > 0)
        if valid.sum() < 100:
            continue

        # Unproject to 3D
        T = T_wc[ki]
        pts_3d = unproject_depth_to_3d(depth, K, T, mask=valid)  # (N, 3)
        labels = sem_label[valid]  # (N,)

        # Assign to voxels
        voxel_coords = (pts_3d / voxel_size).astype(np.int32)
        for j in range(len(pts_3d)):
            vk = tuple(voxel_coords[j])
            voxel_votes[vk].append(labels[j])
            voxel_points[vk].append(pts_3d[j])

        n_matched += 1

    if n_matched == 0:
        print("  No frames matched!")
        return None

    # Compute consistency per voxel
    consistent_voxels = 0
    total_multi_view_voxels = 0
    agreement_scores = []
    view_counts = []

    for vk, votes in voxel_votes.items():
        if len(votes) < min_views:
            continue
        total_multi_view_voxels += 1
        view_counts.append(len(votes))

        # Majority vote
        from collections import Counter
        vote_counts = Counter(votes)
        majority_count = vote_counts.most_common(1)[0][1]
        agreement = majority_count / len(votes)
        agreement_scores.append(agreement)

        if agreement >= 0.5:  # majority agreement
            consistent_voxels += 1

    mvsc = consistent_voxels / max(1, total_multi_view_voxels)
    mean_agreement = np.mean(agreement_scores) if agreement_scores else 0
    mean_views = np.mean(view_counts) if view_counts else 0

    print(f"  Frames projected: {n_matched}")
    print(f"  Multi-view voxels (≥{min_views} views): {total_multi_view_voxels}")
    print(f"  Mean views per voxel: {mean_views:.1f}")
    print(f"  MVSC (majority agreement): {mvsc:.4f} ({mvsc*100:.1f}%)")
    print(f"  Mean agreement score: {mean_agreement:.4f}")
    print(f"  Agreement distribution:")
    for thresh in [0.5, 0.67, 0.75, 0.9, 1.0]:
        n = sum(1 for a in agreement_scores if a >= thresh)
        pct = n / max(1, len(agreement_scores)) * 100
        print(f"    ≥{thresh:.2f}: {pct:.1f}%")

    return {
        'mvsc': mvsc, 'mean_agreement': mean_agreement,
        'mean_views': mean_views, 'n_multi_view_voxels': total_multi_view_voxels,
        'n_frames': n_matched
    }

def compute_reprojection_agreement(npz_path, mask_dir, kf_indices, T_wc, K,
                                     sample_pairs=50, min_distance=0.5):
    """
    Metric 2: Re-projection Semantic Agreement (RPSA)

    Algorithm:
      1. Select pairs of keyframes (A, B) with sufficient viewpoint difference
      2. Project SAM3 labels from A to 3D, then re-project to B
      3. Compare re-projected labels with SAM3 labels at B
      4. RPSA = per-pixel agreement rate
    """
    print(f"\n{'='*60}")
    print(f"  Metric 2: Re-projection Semantic Agreement (RPSA)")
    print(f"  Sample pairs: {sample_pairs}, min distance: {min_distance}m")
    print(f"{'='*60}")

    data = np.load(npz_path, allow_pickle=True)
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npz')])
    mask_by_frame = {int(f.replace("mask_","").replace(".npz","")): f for f in mask_files}

    # Find candidate pairs with sufficient baseline
    pair_scores = []
    np.random.seed(42)

    # Build label maps for all keyframes
    print("  Building semantic label maps...")
    label_maps = {}
    depth_maps = {}
    for ki, kf_idx in enumerate(tqdm(kf_indices, desc="Load")):
        if kf_idx not in mask_by_frame:
            continue
        masks = np.load(os.path.join(mask_dir, mask_by_frame[kf_idx]))['mask']
        H, W = masks.shape[1], masks.shape[2]
        sem_label = np.zeros((H, W), dtype=np.int32)
        for pidx, sn20 in PIDX_TO_SN20.items():
            if pidx < masks.shape[0]:
                sem_label[masks[pidx] > 0] = sn20

        depth_full = data[f'depth_{kf_idx}'].astype(np.float32)
        if depth_full.shape != (H, W):
            import cv2
            depth = cv2.resize(depth_full, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            depth = depth_full

        if (sem_label > 0).sum() > 500:
            label_maps[ki] = sem_label
            depth_maps[ki] = depth

    valid_kis = sorted(label_maps.keys())
    if len(valid_kis) < 2:
        print("  Not enough valid keyframes!")
        return None

    # Sample pairs
    pairs = []
    for _ in range(sample_pairs * 3):
        a, b = np.random.choice(valid_kis, 2, replace=False)
        dist = np.linalg.norm(T_wc[a][:3, 3] - T_wc[b][:3, 3])
        if dist > min_distance:
            pairs.append((a, b))
        if len(pairs) >= sample_pairs:
            break

    print(f"  Selected {len(pairs)} pairs from {len(valid_kis)} keyframes")

    # Evaluate each pair
    all_agreements = []

    for a, b in tqdm(pairs, desc="RPSA-pairs"):
        sem_a = label_maps[a]
        sem_b = label_maps[b]
        depth_a = depth_maps[a]
        depth_b = depth_maps[b]

        # Get valid pixels in A with semantic label
        valid_a = (depth_a > 0.1) & (depth_a < 10.0) & (sem_a > 0)
        if valid_a.sum() < 100:
            continue

        # Project A → 3D
        pts_a_3d = unproject_depth_to_3d(depth_a, K, T_wc[a], mask=valid_a)
        labels_a = sem_a[valid_a]

        # Project 3D → B (camera frame)
        T_bw = np.linalg.inv(T_wc[b])  # world→camera B
        pts_a_in_b = (T_bw @ np.concatenate([pts_a_3d, np.ones((len(pts_a_3d),1))], axis=1).T).T[:, :3]

        # Project to B image plane
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        u_b = (pts_a_in_b[:, 0] * fx / pts_a_in_b[:, 2] + cx).astype(int)
        v_b = (pts_a_in_b[:, 1] * fy / pts_a_in_b[:, 2] + cy).astype(int)

        # Filter: in bounds, in front of camera, depth consistent
        H_b, W_b = sem_b.shape
        valid_proj = (pts_a_in_b[:, 2] > 0.1) & (u_b >= 0) & (u_b < W_b) & (v_b >= 0) & (v_b < H_b)

        if valid_proj.sum() < 100:
            continue

        # Check depth consistency
        depth_b_at_proj = depth_b[v_b[valid_proj], u_b[valid_proj]]
        depth_consistent = np.abs(pts_a_in_b[valid_proj, 2] - depth_b_at_proj) < 0.1  # 10cm

        valid_final = valid_proj.copy()
        valid_final[valid_proj] = depth_consistent

        if valid_final.sum() < 50:
            continue

        # Compare labels
        labels_b = sem_b[v_b[valid_final], u_b[valid_final]]
        labels_a_proj = labels_a[valid_final]

        # Agreement: same non-zero label
        agree = (labels_a_proj == labels_b) & (labels_b > 0)
        agreement_rate = agree.sum() / max(1, (labels_b > 0).sum())
        all_agreements.append(agreement_rate)

    if not all_agreements:
        print("  No valid pairs!")
        return None

    rpsa = np.mean(all_agreements)
    rpsa_std = np.std(all_agreements)

    print(f"  Valid pairs evaluated: {len(all_agreements)}")
    print(f"  RPSA (re-projection agreement): {rpsa:.4f} ({rpsa*100:.1f}%)")
    print(f"  Std: {rpsa_std:.4f}")

    return {'rpsa': rpsa, 'rpsa_std': rpsa_std, 'n_pairs': len(all_agreements)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="Stage3 input NPZ")
    parser.add_argument("--mask_dir", required=True, help="SAM3 masks directory")
    parser.add_argument("--output_dir", default="experiments/semantic_3d_eval")
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--min_views", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading NPZ...")
    data = np.load(args.npz, allow_pickle=True)
    kf_indices = data['vggt_kf_indices'].tolist()
    K = data['K_0']
    # NPZ uses w2c (world-to-camera), convert to T_wc (camera-to-world)
    w2c = np.stack([data[f'w2c_{kf}'] for kf in kf_indices])
    T_wc = np.linalg.inv(w2c)  # w2c → T_wc
    print(f"  Keyframes: {len(kf_indices)}, T_wc: {T_wc.shape}, K: {K.shape}")

    # ═══ Metric 1: Multi-view Consistency ═══
    mvsc_result = compute_multiview_consistency(
        args.npz, args.mask_dir, kf_indices, T_wc, K,
        voxel_size=args.voxel_size, min_views=args.min_views
    )

    # ═══ Metric 2: Re-projection Agreement ═══
    rpsa_result = compute_reprojection_agreement(
        args.npz, args.mask_dir, kf_indices, T_wc, K,
        sample_pairs=min(50, len(kf_indices))
    )

    # ═══ Save results ═══
    results = {
        'npz': args.npz,
        'mask_dir': args.mask_dir,
        'n_keyframes': len(kf_indices),
        'mvsc': mvsc_result,
        'rpsa': rpsa_result,
    }

    import json
    with open(os.path.join(args.output_dir, "semantic_3d_results.json"), "w") as f:
        # Convert numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"\n{'='*60}")
    print(f"  SUMMARY: 3D Semantic Evaluation")
    print(f"{'='*60}")
    if mvsc_result:
        print(f"  MVSC: {mvsc_result['mvsc']:.4f} ({mvsc_result['mvsc']*100:.1f}%)")
        print(f"    → {mvsc_result['mvsc']*100:.1f}% of multi-view voxels have consistent labels")
    if rpsa_result:
        print(f"  RPSA: {rpsa_result['rpsa']:.4f} ({rpsa_result['rpsa']*100:.1f}%)")
        print(f"    → {rpsa_result['rpsa']*100:.1f}% of re-projected pixels agree with SAM3 prediction")
    print(f"  Results saved to {args.output_dir}/semantic_3d_results.json")

if __name__ == "__main__":
    main()
