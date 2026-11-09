#!/usr/bin/env python3
"""Compute SAM3 mIoU by remapping pipeline 50-prompts → ScanNet 20 classes."""
import numpy as np, os, sys
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

# Pipeline 50 prompts → ScanNet label → SN20 mapping
PROMPT_TO_SCANNET = {
    "wall": 1, "floor": 2, "cabinet": 3, "bed": 4, "chair": 5,
    "sofa": 6, "couch": 6, "table": 7, "door": 8, "window": 9,
    "bookshelf": 10, "shelf": 10, "picture": 11, "painting": 11,
    "counter": 12, "desk": 14, "curtain": 16,
    "refrigerator": 24, "toilet": 33, "sink": 34, "bathtub": 36,
}

ALL_PROMPTS = [
    "wall","floor","ceiling","door","window","stairs","railing",
    "bed","cabinet","bookshelf","shelf","counter","desk","table","sofa","couch",
    "sink","toilet","bathtub","mirror","lamp","light",
    "television","monitor","laptop","keyboard","mouse","phone","refrigerator","oven","microwave",
    "chair","plant","vase","picture","painting","curtain","pillow","clock",
    "book","bottle","cup","bowl","plate","bag","backpack","box","bin","basket","person",
]

# ScanNet label → SN20 index (1-based)
SN20_FROM_SCANNET = {
    1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10,
    11:11, 12:12, 14:13, 16:14, 24:15, 28:16, 33:17, 34:18, 36:19, 39:20
}

# Build: channel_idx → sn20 class
ch_to_sn20 = {}
for i, prompt in enumerate(ALL_PROMPTS):
    sn_label = PROMPT_TO_SCANNET.get(prompt)
    if sn_label is not None:
        sn20 = SN20_FROM_SCANNET.get(sn_label)
        if sn20 is not None:
            ch_to_sn20.setdefault(sn20, []).append(i)

print(f"Channel mapping: {len(ch_to_sn20)} SN20 classes mapped")
for sn20, channels in sorted(ch_to_sn20.items()):
    names = [ALL_PROMPTS[c] for c in channels]
    print(f"  SN20-{sn20}: {names}")

# Raw ScanNet ID → SN20 (via NYU40)
RAW_TO_NYU40 = {
    1:1, 2:5, 3:2, 4:7, 5:8, 6:19, 7:3, 8:15, 9:14, 10:5,
    11:4, 13:18, 14:34, 15:11, 16:9, 18:10, 19:35, 21:16, 22:23,
    23:23, 26:29, 28:35, 31:27, 40:35, 41:22, 47:39, 48:40,
    56:39, 59:40, 71:19, 161:8, 165:40, 177:40,
}
NYU40_TO_20 = {
    1:1, 2:2, 3:3, 4:4, 5:5, 6:10, 7:6, 8:7, 9:8, 10:9,
    11:10, 12:12, 14:12, 15:9, 16:13, 18:19, 19:19, 22:2,
    23:19, 27:19, 29:19, 34:17, 35:19, 39:19, 40:19,
}
raw_to_sn20 = {}
for raw_id, nyu40 in RAW_TO_NYU40.items():
    sn20 = NYU40_TO_20.get(nyu40, 19)
    raw_to_sn20[raw_id] = sn20

SN20_NAMES = {
    1:"wall",2:"floor",3:"cabinet",4:"bed",5:"chair",6:"sofa",
    7:"table",8:"door",9:"window",10:"bookshelf",11:"picture",12:"counter",
    13:"desk",14:"curtain",15:"refrigerator",16:"shower curtain",
    17:"toilet",18:"sink",19:"bathtub",20:"otherfurniture"
}

def run_scene(mask_dir, gt_dir, num_frames, scene_name):
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npz')])[:num_frames]
    gt_files = set(os.listdir(gt_dir))

    per_class_intersection = defaultdict(int)
    per_class_union = defaultdict(int)
    matched = 0

    for mf in tqdm(mask_files, desc=f"mIoU-{scene_name}"):
        frame_num = mf.replace("mask_", "").replace(".npz", "").lstrip("0") or "0"
        gt_name = f"{int(frame_num)}.png"
        if gt_name not in gt_files:
            continue

        sam3 = np.load(os.path.join(mask_dir, mf))["mask"]  # (50, H, W)

        # Build SN20 label map
        sam3_label = np.zeros((sam3.shape[1], sam3.shape[2]), dtype=np.int32)
        for sn20, channels in ch_to_sn20.items():
            mask_union = np.zeros_like(sam3_label, dtype=bool)
            for ch in channels:
                if ch < sam3.shape[0]:
                    mask_union |= (sam3[ch] > 0)
            sam3_label[mask_union] = sn20

        # Load GT
        gt_raw = np.array(Image.open(os.path.join(gt_dir, gt_name)))
        gt_label = np.zeros_like(gt_raw, dtype=np.int32)
        for raw_id, sn20 in raw_to_sn20.items():
            gt_label[gt_raw == raw_id] = sn20

        # Resize SAM3 to match GT
        if sam3_label.shape != gt_label.shape:
            sam3_label = np.array(Image.fromarray(sam3_label.astype(np.uint8)).resize(
                (gt_label.shape[1], gt_label.shape[0]), Image.NEAREST))

        gt_present = set(np.unique(gt_label)) & set(range(1, 21))
        for c in gt_present:
            pred_c = (sam3_label == c)
            gt_c = (gt_label == c)
            per_class_intersection[c] += (pred_c & gt_c).sum()
            per_class_union[c] += (pred_c | gt_c).sum()
        matched += 1

    if matched == 0:
        print(f"\n{scene_name}: No frames matched!")
        return None

    print(f"\n{'='*60}")
    print(f"  SAM3 mIoU: {scene_name} ({matched} frames)")
    print(f"  {'Class':<22s} {'IoU':>6s}  {'Intersect':>10s} {'Union':>10s}")
    print(f"  {'-'*52}")
    ious = []
    all_classes = sorted(set(list(per_class_intersection.keys()) + list(per_class_union.keys())))
    for c in all_classes:
        inter = per_class_intersection[c]
        union = per_class_union[c]
        if union > 0:
            iou = inter / union
            ious.append(iou)
            print(f"  {c:2d} {SN20_NAMES.get(c,'?'):20s}: {iou:6.4f}  {inter/1e3:8.1f}k {union/1e3:8.1f}k")
        else:
            print(f"  {c:2d} {SN20_NAMES.get(c,'?'):20s}: {'N/A':>6s}  (no union)")

    miou = np.mean(ious) if ious else 0
    print(f"  {'─'*52}")
    print(f"  mIoU: {miou:.4f}  ({len(ious)} classes)")
    print(f"{'='*60}")
    return miou

if __name__ == "__main__":
    scenes = [
        ("scannet_0059", "/media/yml/share/v3dsam/experiments/scannet_0059_v38_full/sam3_masks",
         "/media/yml/share/SGS-SLAM-main2/datasets/scannet/scene0059_00/semantic_id", 200),
        ("scannet_0000", "/media/yml/share/v3dsam/experiments/scannet_0000_v38/sam3_masks",
         "/media/yml/share/SGS-SLAM-main2/datasets/scannet/scene0000_00/semantic_id", 45),
        ("scannet_0106", "/media/yml/share/v3dsam/experiments/scannet_0106_v38/sam3_masks",
         "/media/yml/share/SGS-SLAM-main2/datasets/scannet/scene0106_00/semantic_id", 40),
        ("scannet_0207", "/media/yml/share/v3dsam/experiments/scannet_0207_v38/sam3_masks",
         "/media/yml/share/SGS-SLAM-main2/datasets/scannet/scene0207_00/semantic_id", 85),
    ]

    results = {}
    for name, mask_dir, gt_dir, nf in scenes:
        if not os.path.exists(mask_dir):
            print(f"{name}: mask_dir missing: {mask_dir}")
            continue
        if not os.path.exists(gt_dir):
            print(f"{name}: gt_dir missing: {gt_dir}")
            continue
        miou = run_scene(mask_dir, gt_dir, nf, name)
        if miou is not None:
            results[name] = miou

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"  {'Scene':<25s} {'mIoU':>8s}")
    for name, miou in results.items():
        print(f"  {name:<25s} {miou:8.4f}")
    if results:
        print(f"  {'Average':<25s} {np.mean(list(results.values())):8.4f}")
    print(f"{'='*60}")
