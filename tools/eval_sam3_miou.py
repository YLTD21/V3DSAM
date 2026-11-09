#!/usr/bin/env python3
"""
Standalone SAM3 mIoU Evaluation on ScanNet GT
==============================================
Uses SAM3 with ScanNet's exact 20 class names as prompts.
Does NOT modify the main pipeline. Output: per-class IoU + mIoU.

Usage:
  python tools/eval_sam3_miou.py \
    --image_dir /path/to/scannet/color \
    --gt_dir /path/to/scannet/semantic_id \
    --output_dir experiments/sam3_miou_eval \
    --num_frames 50
"""
import numpy as np, cv2, os, sys, argparse, torch
from tqdm import tqdm
from collections import defaultdict

# ═══ ScanNet 20 classes (exact names as text prompts) ═══
SCANNET_CLASSES = [
    "wall",              # 0
    "floor",             # 1
    "chair",             # 2
    "table",             # 3
    "desk",              # 4
    "bed",               # 5
    "bookshelf",         # 6
    "sofa",              # 7
    "sink",              # 8
    "bathtub",           # 9
    "toilet",            # 10
    "curtain",           # 11
    "counter",           # 12
    "door",              # 13
    "window",            # 14
    "shower curtain",    # 15
    "refrigerator",      # 16
    "picture",           # 17
    "cabinet",           # 18
    "other furniture",   # 19
]

SCANNET_NAMES = {
    0:"wall",1:"floor",2:"chair",3:"table",4:"desk",5:"bed",
    6:"bookshelf",7:"sofa",8:"sink",9:"bathtub",10:"toilet",
    11:"curtain",12:"counter",13:"door",14:"window",15:"shower curtain",
    16:"refrigerator",17:"picture",18:"cabinet",19:"other furniture"
}


def run_sam3_on_frames(image_paths, prompts, output_dir):
    """Run SAM3 on frames — exact same API as v38_full_pipeline.py."""
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_processor import Sam3Processor
    from sam3.model.data_misc import FindStage
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)

    print("Loading SAM3 model...")
    sam3_model = build_sam3_image_model(
        bpe_path="/media/yml/share/sam3-main/assets/bpe_simple_vocab_16e6.txt.gz",
        checkpoint_path="/media/yml/share/sam3-main/sam3.pt", device="cuda")
    processor = Sam3Processor(sam3_model, confidence_threshold=0.1)
    dummy_geo = sam3_model._get_dummy_prompt(num_prompts=1)
    N_cats = len(prompts)

    print(f"SAM3 loaded. {N_cats} categories, {len(image_paths)} frames")

    for gi, img_path in enumerate(tqdm(image_paths, desc="SAM3->disk")):
        img = Image.open(img_path).convert("RGB")
        state = processor.set_image(img)
        backbone_out = state["backbone_out"]
        h0, w0 = state["original_height"], state["original_width"]

        text_outputs = sam3_model.backbone.forward_text(prompts, device="cuda")
        all_lang_features = text_outputs["language_features"]
        all_lang_mask = text_outputs["language_mask"]

        find_input = FindStage(
            img_ids=torch.tensor([0], device="cuda", dtype=torch.long),
            text_ids=torch.tensor([0], device="cuda", dtype=torch.long),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None)

        cat_mask = np.zeros((N_cats, h0, w0), dtype=np.uint8)

        for cat_idx in range(N_cats):
            try:
                backbone_out["language_features"] = all_lang_features[:, cat_idx:cat_idx+1, :]
                backbone_out["language_mask"] = all_lang_mask[cat_idx:cat_idx+1, :]
                outputs = sam3_model.forward_grounding(
                    backbone_out=backbone_out, find_input=find_input,
                    geometric_prompt=dummy_geo, find_target=None)
                out_masks = outputs.get("pred_masks")
                out_logits = outputs.get("pred_logits")
                if out_masks is None or out_logits is None:
                    continue
                out_probs = out_logits.sigmoid()
                if out_probs.max() > 0.3:
                    best_idx = out_probs.argmax().item()
                    mask = out_masks[best_idx, 0] > 0.5
                    mask_np = mask.cpu().numpy().astype(np.uint8)
                    # Resize to image size
                    if mask_np.shape != (h0, w0):
                        mask_np = cv2.resize(mask_np.astype(np.uint8), (w0, h0),
                                            interpolation=cv2.INTER_NEAREST)
                    cat_mask[cat_idx] = mask_np
            except Exception:
                continue

        np.savez_compressed(os.path.join(output_dir, f'mask_{gi:06d}.npz'), mask=cat_mask)

    return output_dir


def compute_miou(mask_dir, gt_dir, frame_indices, n_classes=20):
    """
    Compute per-class IoU and mIoU.

    SAM3 mask i → ScanNet class i (perfect 1:1 mapping)
    """
    intersection = defaultdict(int)
    union = defaultdict(int)
    pixel_acc = []

    # First: find which GT classes exist
    gt_present = set()
    for fi in frame_indices[:10]:
        gt = cv2.imread(os.path.join(gt_dir, f'{fi}.png'), cv2.IMREAD_UNCHANGED)
        if gt is not None:
            for c in range(n_classes):
                if (gt == c).sum() > 500:
                    gt_present.add(c)

    print(f"GT classes present in scene: {sorted(gt_present)}")
    print(f"  {[SCANNET_NAMES[c] for c in sorted(gt_present)]}")

    for fi in tqdm(frame_indices, desc="Computing mIoU"):
        # Load SAM3 masks
        mask_file = os.path.join(mask_dir, f'mask_{fi:06d}.npz')
        if not os.path.exists(mask_file):
            continue

        masks = np.load(mask_file)['mask']  # (20, H, W)
        if masks.ndim != 3 or masks.shape[0] < n_classes:
            continue

        # Argmax to get per-pixel label
        sam3_label = np.argmax(masks[:n_classes], axis=0)
        sam3_conf = masks[:n_classes].max(axis=0)

        # Load GT
        gt = cv2.imread(os.path.join(gt_dir, f'{fi}.png'), cv2.IMREAD_UNCHANGED)
        if gt is None:
            continue

        if gt.shape != sam3_label.shape:
            gt = cv2.resize(gt, (sam3_label.shape[1], sam3_label.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

        # Valid region: confident SAM3 prediction + valid GT
        valid = (sam3_conf > 0.3) & (gt < n_classes)

        # Per-class IoU
        for c in gt_present:
            pred_c = (sam3_label == c) & valid
            gt_c = (gt == c)
            intersection[c] += (pred_c & gt_c).sum()
            union[c] += (pred_c | gt_c).sum()

        # Pixel accuracy
        if valid.sum() > 0:
            correct = (sam3_label[valid] == gt[valid]).sum()
            pixel_acc.append(correct / valid.sum())

    return intersection, union, pixel_acc, gt_present


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--output_dir", default="experiments/sam3_miou_eval")
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--skip_inference", action="store_true",
                       help="Skip SAM3 inference (use existing masks)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Get frame list
    img_files = sorted(os.listdir(args.image_dir))
    img_paths = [os.path.join(args.image_dir, f) for f in img_files[:args.num_frames]]
    frame_indices = list(range(args.num_frames))

    print(f"Frames: {len(img_paths)}")
    print(f"Classes: {len(SCANNET_CLASSES)} (exact ScanNet 20-class mapping)")

    mask_dir = os.path.join(args.output_dir, "sam3_masks")

    if not args.skip_inference:
        run_sam3_on_frames(img_paths, SCANNET_CLASSES, mask_dir)
    else:
        print(f"Using existing masks from {mask_dir}")
        if not os.path.exists(mask_dir):
            print("ERROR: No existing masks found. Remove --skip_inference.")
            sys.exit(1)

    # Compute mIoU
    print("\n" + "="*60)
    print("Computing mIoU (SAM3 20-class ↔ ScanNet 20-class, perfect mapping)")
    print("="*60)

    intersection, union, pixel_acc, gt_present = compute_miou(
        mask_dir, args.gt_dir, frame_indices, n_classes=len(SCANNET_CLASSES)
    )

    print("\n=== Per-Class IoU ===")
    ious = []
    for c in sorted(gt_present):
        iou = intersection[c] / (union[c] + 1e-8)
        ious.append(iou)
        name = SCANNET_NAMES[c]
        n_inter = intersection[c]
        n_union = union[c]
        print(f"  {name:18s} (class {c:2d}): IoU={iou:.4f}  "
              f"(intersection={n_inter/1e3:.0f}k, union={n_union/1e3:.0f}k)")

    miou = np.mean(ious) if ious else 0
    mpa = np.mean(pixel_acc) * 100 if pixel_acc else 0

    print(f"\n{'='*60}")
    print(f"  mIoU: {miou:.4f} ({miou*100:.1f}%)")
    print(f"  mPA:  {mpa:.1f}%")
    print(f"  Evaluated on {len(pixel_acc)} frames, {len(gt_present)} active classes")
    print(f"{'='*60}")

    # Save results
    with open(os.path.join(args.output_dir, "miou_results.txt"), "w") as f:
        f.write(f"mIoU: {miou:.4f}\nmPA: {mpa:.1f}%\n")
        for c in sorted(gt_present):
            iou = intersection[c] / (union[c] + 1e-8)
            f.write(f"{SCANNET_NAMES[c]:18s}: IoU={iou:.4f}\n")

    print(f"\nResults saved to {args.output_dir}/")
