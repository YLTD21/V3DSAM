#!/usr/bin/env python3
"""Standalone: generate SAM3 semantic label maps for ALL VGGT keyframes of a
Stage3 input npz, to enable semantic-boundary-aware reconstruction
(Lever 1 of the "semantic -> PSNR" direction).

Reads _stage3_input.npz (which carries vggt_kf_indices + img_path_{i}), runs
SAM3 grounding per keyframe (same routine as gen_npz_semantic.py), and saves a
COMPACT per-keyframe argmax label map.

Output npz (<out>):
  prompts      : (N_cats,)   object array, class names in label-id order
  kf_indices   : (K,)        export indices == vggt_kf_indices
  labels       : (K, H, W)   int16 label map (score-weighted argmax over categories; -1 = none)
  img_paths    : (K,)        object array, keyframe image paths
  img_H, img_W : int         full-res mask H/W
  N_cats       : int

Does NOT modify v3dsam source. Needs the sgs env + the sys.path trick (vggt /
sam3 / perception_models). Run on AutoDL:

  /root/miniconda3/envs/sgs/bin/python gen_kf_semantic.py \
    --data_npz /root/autodl-tmp/psnr_tum/room/stage1_scaled/_stage3_input.npz \
    --output   /root/autodl-tmp/psnr_tum/room/stage1_scaled/kf_semantic.npz \
    [--max_kf 3]   # smoke test
"""
import os, sys, argparse, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/v3dsam")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/vggt")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/sam3")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/perception_models")
os.environ['XFORMERS_DISABLED'] = '1'

# 家具类对齐 ScanNet 官方 NYU40：补 dresser/night stand（官方独立类），
# 去 couch/light/painting 冗余同义词，减少逐类 query 互相抢同一物体。
ALL_PROMPTS = [
    "wall","floor","ceiling","door","window","stairs","railing",
    "bed","cabinet","bookshelf","shelf","counter","desk","table","sofa",
    "dresser","night stand",
    "sink","toilet","bathtub","mirror","lamp",
    "television","monitor","laptop","keyboard","mouse","phone","refrigerator","oven","microwave",
    "chair","plant","vase","picture","curtain","pillow","clock",
    "book","bottle","cup","bowl","plate","bag","backpack","box","bin","basket","person",
]
N_cats = len(ALL_PROMPTS)


def sam3_frame_mask(img_path, sam3_model, processor, dummy_geo):
    """Per-frame (N_cats, H, W) float32 category score map (mask_prob * detection_score)."""
    from sam3.model.data_misc import FindStage
    img = Image.open(img_path).convert("RGB")
    state = processor.set_image(img)
    backbone_out = state["backbone_out"]
    h0, w0 = state["original_height"], state["original_width"]
    text_outputs = sam3_model.backbone.forward_text(ALL_PROMPTS, device="cuda")
    all_lang_features = text_outputs["language_features"]
    all_lang_mask = text_outputs["language_mask"]
    find_input = FindStage(
        img_ids=torch.tensor([0], device="cuda", dtype=torch.long),
        text_ids=torch.tensor([0], device="cuda", dtype=torch.long),
        input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
        input_points=None, input_points_mask=None)
    cat_score = np.zeros((N_cats, h0, w0), dtype=np.float32)
    for cat_idx in range(N_cats):
        try:
            backbone_out["language_features"] = all_lang_features[:, cat_idx:cat_idx + 1, :]
            backbone_out["language_mask"] = all_lang_mask[cat_idx:cat_idx + 1, :]
            outputs = sam3_model.forward_grounding(
                backbone_out=backbone_out, find_input=find_input,
                geometric_prompt=dummy_geo, find_target=None)
            out_masks = outputs.get("pred_masks")
            out_logits = outputs.get("pred_logits")
            if out_masks is None or out_logits is None:
                continue
            out_probs = out_logits.sigmoid()
            presence = outputs.get("presence_logit_dec")
            if presence is not None:
                out_probs = (out_probs * presence.sigmoid().unsqueeze(1)).squeeze(-1)
            keep = out_probs > 0.1
            if keep.any():
                masks_kept = out_masks[keep].sigmoid()
                scores_kept = out_probs[keep]
                masks_kept = F.interpolate(masks_kept.unsqueeze(1), (h0, w0),
                    mode="bilinear", align_corners=False).squeeze(1)
                for j in range(masks_kept.shape[0]):
                    contrib = (masks_kept[j] * scores_kept[j]).detach().cpu().numpy().astype(np.float32)
                    np.maximum(cat_score[cat_idx], contrib, out=cat_score[cat_idx])
        except Exception:
            continue
    return cat_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_npz", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_kf", type=int, default=0, help="limit keyframes (0 = all), for smoke test")
    ap.add_argument("--score_thr", type=float, default=0.05,
                    help="min category score for a pixel to keep a label (soft argmax floor)")
    args = ap.parse_args()

    data = np.load(args.data_npz, allow_pickle=True)
    kf_indices = [int(x) for x in data["vggt_kf_indices"].tolist()]
    if args.max_kf > 0:
        kf_indices = kf_indices[:args.max_kf]
    img_paths = [str(data[f"img_path_{i}"]) for i in kf_indices]
    img_H = int(data["img_H"]); img_W = int(data["img_W"])
    print(f"[gen_kf_semantic] {len(kf_indices)} keyframes to process, res {img_W}x{img_H}")

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    print("[gen_kf_semantic] loading SAM3 ...")
    sam3_model = build_sam3_image_model(
        bpe_path="/root/autodl-tmp/v3dsam/third_party/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        checkpoint_path="/root/autodl-tmp/v3dsam/third_party/sam3/sam3.pt", device="cuda")
    processor = Sam3Processor(sam3_model, confidence_threshold=0.1)
    dummy_geo = sam3_model._get_dummy_prompt(num_prompts=1)
    print("[gen_kf_semantic] SAM3 loaded")

    labels = np.full((len(kf_indices), img_H, img_W), -1, dtype=np.int16)
    t0 = time.time()
    for k, (idx, p) in enumerate(zip(kf_indices, img_paths)):
        t1 = time.time()
        try:
            cat_score = sam3_frame_mask(p, sam3_model, processor, dummy_geo)
            lab = cat_score.argmax(0).astype(np.int16)
            lab[cat_score.max(0) < args.score_thr] = -1
            labels[k] = lab
            uniq = np.unique(lab)
            present = uniq[uniq >= 0].tolist()
            print(f"  [{k+1}/{len(kf_indices)}] kf_idx={idx} {os.path.basename(p)} "
                  f"present={len(present)} classes {present[:12]} ({time.time()-t1:.1f}s)")
        except Exception as e:
            print(f"  [{k+1}/{len(kf_indices)}] kf_idx={idx} FAILED: {e} (labels stay -1)")

    out = {
        "prompts": np.array(ALL_PROMPTS, dtype=object),
        "kf_indices": np.array(kf_indices, dtype=np.int32),
        "labels": labels,
        "img_paths": np.array(img_paths, dtype=object),
        "img_H": img_H, "img_W": img_W, "N_cats": N_cats,
    }
    np.savez_compressed(args.output, **out)
    n_ok = int((labels.max(axis=(1, 2)) >= 0).sum())
    print(f"[gen_kf_semantic] DONE {time.time()-t0:.1f}s: {n_ok}/{len(kf_indices)} frames with "
          f"semantics -> {args.output} ({os.path.getsize(args.output)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
