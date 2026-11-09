#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAM3(开放词汇) vs ScanNet GT(semantic_id, NYU-40) 的 mIoU 精度测量。

复用 gen_npz_scan.py 的 SAM3 调用方式（逐类 grounding）。对关键帧子集跑 SAM3，
把 50 个开放词汇 prompt 映射到 NYU-40 ID，与 GT semantic_id 逐像素对齐，输出
mIoU / 逐类 IoU / 像素精度。

运行（AutoDL sgs 环境，需先上传 semantic_id）：
  python measure_miou.py --scene scene0000_00 --stride 20 --max_frames 30   # smoke
  python measure_miou.py --stride 20 --max_frames 50                        # 全量
"""
import os, sys, argparse, json, time
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/v3dsam")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/sam3")
os.environ['XFORMERS_DISABLED'] = '1'

ABL = "/root/autodl-tmp/ablation/scannet"
DAT = "/root/autodl-tmp/v3dsam/datasets/scannet"
OUT = "/root/autodl-tmp/psnr_tum/scannet_figs"
os.makedirs(OUT, exist_ok=True)

# ScanNet 语义 PNG 存的是 raw id，需经 scannet_labels 映射到 nyu40id(0-based)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scannet_labels import RAW2NYU, NYU40, map_gt

ALL_PROMPTS = [
    "wall","floor","ceiling","door","window","stairs","railing",
    "bed","cabinet","bookshelf","shelf","counter","desk","table","sofa","couch",
    "sink","toilet","bathtub","mirror","lamp","light",
    "television","monitor","laptop","keyboard","mouse","phone","refrigerator","oven","microwave",
    "chair","plant","vase","picture","painting","curtain","pillow","clock",
    "book","bottle","cup","bowl","plate","bag","backpack","box","bin","basket","person",
]
N_cats = len(ALL_PROMPTS)

# 开放词汇 prompt -> NYU-40 id
PROMPT2NYU = {
    "wall":0,"floor":1,"ceiling":21,"door":7,"window":8,"stairs":37,"railing":37,
    "bed":3,"cabinet":2,"bookshelf":9,"shelf":14,"counter":11,"desk":13,"table":6,
    "sofa":5,"couch":5,"sink":33,"toilet":32,"bathtub":35,"mirror":18,"lamp":34,"light":34,
    "television":24,"monitor":24,"laptop":39,"keyboard":39,"mouse":39,"phone":39,
    "refrigerator":23,"oven":38,"microwave":38,"chair":4,"plant":39,"vase":39,
    "picture":10,"painting":10,"curtain":15,"pillow":17,"clock":39,"book":22,
    "bottle":39,"cup":39,"bowl":39,"plate":39,"bag":36,"backpack":36,"box":28,
    "bin":39,"basket":39,"person":30,
}
PROMPT2NYU_IDX = [PROMPT2NYU[p] for p in ALL_PROMPTS]  # (N_cats,) 每个 prompt 的 NYU id

def sam3_frame_mask(img_path, sam3_model, processor, dummy_geo):
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
    cat_mask = np.zeros((N_cats, h0, w0), dtype=np.uint8)
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
                masks_kept = F.interpolate(masks_kept.unsqueeze(1), (h0, w0),
                    mode="bilinear", align_corners=False).squeeze(1) > 0.5
                for m in masks_kept:
                    cat_mask[cat_idx] |= m.cpu().numpy().astype(np.uint8)
        except Exception:
            continue
    return cat_mask


def label_from_cat_mask(cat_mask):
    """cat_mask (N_cats,H,W) -> 每个像素的 NYU-40 id（无标签 -1）。"""
    any_mask = cat_mask.any(0)
    pred_cat = cat_mask.argmax(0)  # (H,W) 开放词汇类别索引
    pred_nyu = np.full(pred_cat.shape, -1, dtype=np.int16)
    pred_nyu[any_mask] = np.array(PROMPT2NYU_IDX, dtype=np.int16)[pred_cat[any_mask]]
    return pred_nyu


def load_gt(scene, fi, H, W):
    p = f"{DAT}/{scene}/semantic_id/{fi}.png"
    if not os.path.exists(p):
        return None
    a = np.array(Image.open(p))  # raw id (H0,W0)
    if a.shape[0] != H or a.shape[1] != W:
        a = F.interpolate(torch.from_numpy(a.astype(np.float32))[None, None],
                          (H, W), mode="nearest").squeeze().numpy().astype(np.int32)
    return map_gt(a)  # raw id -> nyu40id(0-based), 未知记 -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="", help="单场景 smoke test（空=全部 4 场景）")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--max_frames", type=int, default=50, help="每场景最多抽帧数")
    ap.add_argument("--out", default=f"{OUT}/miou.json")
    args = ap.parse_args()

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    print("[miou] loading SAM3 ...")
    sam3_model = build_sam3_image_model(
        bpe_path="/root/autodl-tmp/v3dsam/third_party/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        checkpoint_path="/root/autodl-tmp/v3dsam/third_party/sam3/sam3.pt", device="cuda")
    processor = Sam3Processor(sam3_model, confidence_threshold=0.1)
    dummy_geo = sam3_model._get_dummy_prompt(num_prompts=1)

    scenes = [args.scene] if args.scene else ["scene0000_00","scene0059_00","scene0106_00","scene0207_00"]
    global_inter = np.zeros(40, np.float64)
    global_union = np.zeros(40, np.float64)
    total_correct = 0
    total_valid = 0
    results = {}
    t_all = time.time()

    for scene in scenes:
        npz = np.load(f"{ABL}/{scene}/_stage3_input_after.npz", allow_pickle=True)
        idxs = sorted(int(k.split("_")[1]) for k in npz.keys() if k.startswith("K_"))
        sample = idxs[::args.stride][:args.max_frames]
        inter = np.zeros(40, np.float64)
        union = np.zeros(40, np.float64)
        corr = 0; valid = 0
        t0 = time.time()
        for n, fi in enumerate(sample):
            ip = str(npz[f"img_path_{fi}"])
            cat_mask = sam3_frame_mask(ip, sam3_model, processor, dummy_geo)
            H, W = cat_mask.shape[1], cat_mask.shape[2]
            pred = label_from_cat_mask(cat_mask)
            gt = load_gt(scene, fi, H, W)
            if gt is None:
                print(f"  [{scene}] frame {fi}: GT missing, skip")
                continue
            m = (pred >= 0) & (gt >= 0) & (gt < 40)
            if m.sum() < 1000:
                continue
            for c in range(40):
                pc = (pred == c) & m
                gc = (gt == c) & m
                inter[c] += int((pc & gc).sum())
                union[c] += int((pc | gc).sum())
            corr += int((pred[m] == gt[m]).sum())
            valid += int(m.sum())
            if (n + 1) % 10 == 0:
                print(f"  [{scene}] {n+1}/{len(sample)} frames, {time.time()-t0:.0f}s elapsed")
        dt = time.time() - t0
        iou = np.zeros(40)
        np.seterr(divide="ignore", invalid="ignore")
        iou[union > 0] = inter[union > 0] / union[union > 0]
        miou = float(iou[union > 0].mean())
        pacc = corr / valid if valid else 0.0
        results[scene] = {
            "n_frames": len(sample), "mIoU": miou, "pixel_acc": pacc,
            "per_class_iou": {NYU40[c]: round(float(iou[c]), 4) for c in range(40) if union[c] > 0},
        }
        global_inter += inter; global_union += union
        total_correct += corr; total_valid += valid
        print(f"[miou] {scene}: {len(sample)} frames, mIoU={miou:.4f}, pixel_acc={pacc:.4f} ({dt:.0f}s, {dt/max(len(sample),1):.1f}s/frame)")

    giou = np.zeros(40)
    np.seterr(divide="ignore", invalid="ignore")
    giou[global_union > 0] = global_inter[global_union > 0] / global_union[global_union > 0]
    gmiou = float(giou[global_union > 0].mean())
    gpacc = total_correct / total_valid if total_valid else 0.0
    summary = {"global": {"mIoU": gmiou, "pixel_acc": gpacc, "n_frames_total": sum(v["n_frames"] for v in results.values())},
               "per_scene": results,
               "elapsed_s": time.time() - t_all}
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\n[miou] GLOBAL mIoU={gmiou:.4f}, pixel_acc={gpacc:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
