#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAM3 一次多类别 grounding（替代循环 50 次单类别），验证重复 mask 是否下降。

关键改动：不再逐类别循环，而是把 50 个类别的 language_features 一次性喂给
forward_grounding（text_ids = [0..49]）。SAM3 decoder 的 200 个 object query 被
50 类分摊，同一物体被几十个 query 命中的重复会大幅减少（book 应从 50~110 降到 ~3-5）。

对比：dump_sam3_masks.py（循环+堆叠）、dump_sam3_masks_dedup.py（循环+IoU去重）。

Run on AutoDL (sgs env):
  /root/miniconda3/envs/sgs/bin/python dump_sam3_masks_multiclass.py \
    --semantic /root/autodl-tmp/ablation/scannet/scene0000_00/kf_semantic.npz \
    --n_frames 3 --out_dir /root/autodl-tmp/sam3_mask_mc
"""
import os, sys, argparse
import numpy as np, torch, torch.nn.functional as F
import cv2
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/v3dsam")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/vggt")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/sam3")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/perception_models")
os.environ['XFORMERS_DISABLED'] = '1'

ALL_PROMPTS = [
    "wall","floor","ceiling","door","window","stairs","railing",
    "bed","cabinet","bookshelf","shelf","counter","desk","table","sofa","couch",
    "sink","toilet","bathtub","mirror","lamp","light",
    "television","monitor","laptop","keyboard","mouse","phone","refrigerator","oven","microwave",
    "chair","plant","vase","picture","painting","curtain","pillow","clock",
    "book","bottle","cup","bowl","plate","bag","backpack","box","bin","basket","person",
]
N_cats = len(ALL_PROMPTS)

TAB40 = ["#1f77b4","#aec7e8","#ff7f0e","#ffbb78","#2ca02c","#98df8a","#d62728",
         "#ff9896","#9467bd","#c5b0d5","#8c564b","#c49c94","#e377c2","#f7b6d2",
         "#7f7f7f","#c7c7c7","#bcbd22","#dbdb8d","#17becf","#9edae5",
         "#393b79","#5254a3","#6b6ecf","#9c9ede","#637939","#8ca252","#b5cf6b",
         "#cedb9c","#8c6d31","#bd9e39","#e7ba52","#e7cb94","#843c39","#ad494a",
         "#d6616b","#e7969c","#7b4173","#a55194","#ce6dbd","#de9ed6"]


def hex2bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def sam3_multiclass_masks(img_path, sam3_model, processor, dummy_geo, min_score=0.1):
    """一次 forward_grounding 预测全部 50 类，返回 [(cat_idx, mask, score), ...]。"""
    from sam3.model.data_misc import FindStage
    img = Image.open(img_path).convert("RGB")
    state = processor.set_image(img)
    backbone_out = state["backbone_out"]
    h0, w0 = state["original_height"], state["original_width"]
    text_outputs = sam3_model.backbone.forward_text(ALL_PROMPTS, device="cuda")
    all_lang_features = text_outputs["language_features"]   # (1, N_cats, C)
    all_lang_mask = text_outputs["language_mask"]           # (N_cats, ...)

    # 一次传全部类别：text_ids = [0..N_cats-1]，language_features 全放
    text_ids = torch.arange(N_cats, device="cuda", dtype=torch.long)
    find_input = FindStage(
        img_ids=torch.tensor([0], device="cuda", dtype=torch.long),
        text_ids=text_ids,
        input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
        input_points=None, input_points_mask=None)
    backbone_out["language_features"] = all_lang_features
    backbone_out["language_mask"] = all_lang_mask

    outputs = sam3_model.forward_grounding(
        backbone_out=backbone_out, find_input=find_input,
        geometric_prompt=dummy_geo, find_target=None)
    out_masks = outputs.get("pred_masks")      # (1, 200, H, W)
    out_logits = outputs.get("pred_logits")    # (1, 200, N_cats)
    if out_masks is None or out_logits is None:
        return [], img

    out_probs = out_logits.sigmoid()           # (1, 200, N_cats)
    presence = outputs.get("presence_logit_dec")
    if presence is not None:
        p = presence.sigmoid()
        if p.dim() == 3:
            out_probs = out_probs * p
        elif p.dim() == 2:
            out_probs = out_probs * p.unsqueeze(-1)
        elif p.dim() == 1:
            out_probs = out_probs * p.unsqueeze(0).unsqueeze(-1)
    probs = out_probs[0]                        # (200, N_cats)
    scores, cat_ids = probs.max(1)              # (200,), (200,)
    keep = scores > min_score
    if not keep.any():
        return [], img

    masks_kept = out_masks[0][keep].sigmoid()   # (n_keep, H, W) 低分辨率
    masks_kept = F.interpolate(masks_kept.unsqueeze(1), (h0, w0),
        mode="bilinear", align_corners=False).squeeze(1) > 0.5
    instances = []
    for i in range(len(masks_kept)):
        m = masks_kept[i].cpu().numpy().astype(bool)
        if int(m.sum()) < 50:
            continue
        instances.append((int(cat_ids[keep][i]), m, float(scores[keep][i])))
    instances.sort(key=lambda x: -x[2])
    return instances, img


def draw_regions(img_bgr, instances, out_png):
    H, W = img_bgr.shape[:2]
    overlay = img_bgr.astype(np.float32)
    for rank, (cat_idx, mask, score) in enumerate(instances):
        col = hex2bgr(TAB40[rank % len(TAB40)])
        overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.array(col, np.float32)
    img_bgr = np.clip(overlay, 0, 255).astype(np.uint8)
    for rank, (cat_idx, mask, score) in enumerate(instances):
        col = hex2bgr(TAB40[rank % len(TAB40)])
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_bgr, cnts, -1, col, 2, cv2.LINE_AA)
        ys, xs = np.nonzero(mask)
        cy, cx = int(ys.mean()), int(xs.mean())
        txt = f"#{rank} {ALL_PROMPTS[cat_idx]}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(img_bgr, txt, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img_bgr, txt, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, col, 1, cv2.LINE_AA)
    cv2.imwrite(out_png, img_bgr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semantic", required=True)
    ap.add_argument("--n_frames", type=int, default=3)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_score", type=float, default=0.1)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sem = np.load(args.semantic, allow_pickle=True)
    img_paths = [str(p) for p in sem["img_paths"]]
    n = len(img_paths)
    idxs = np.linspace(0, n - 1, args.n_frames).astype(int)

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    print("[mc] loading SAM3 ...")
    sam3_model = build_sam3_image_model(
        bpe_path="/root/autodl-tmp/v3dsam/third_party/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        checkpoint_path="/root/autodl-tmp/v3dsam/third_party/sam3/sam3.pt", device="cuda")
    processor = Sam3Processor(sam3_model, confidence_threshold=0.1)
    dummy_geo = sam3_model._get_dummy_prompt(num_prompts=N_cats)
    print("[mc] SAM3 loaded")

    for k, ii in enumerate(idxs):
        p = img_paths[ii]
        inst, img = sam3_multiclass_masks(p, sam3_model, processor, dummy_geo, args.min_score)
        print(f"\n[mc] frame {k} (kf_idx {ii}) {os.path.basename(p)}: {len(inst)} instance masks")
        from collections import Counter
        cc = Counter(ALL_PROMPTS[c] for c, _, _ in inst)
        for name, cnt in sorted(cc.items(), key=lambda x: -x[1]):
            print(f"   {name:<14} {cnt}")
        img_bgr = cv2.imread(p)
        out_png = os.path.join(args.out_dir, f"mc_frame{k}_kf{ii}.png")
        draw_regions(img_bgr, inst, out_png)
        print(f"   -> {out_png}")

    print("[mc] DONE")


if __name__ == "__main__":
    main()
