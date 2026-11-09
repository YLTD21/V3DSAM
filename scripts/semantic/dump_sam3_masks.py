#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dump SAM3 per-instance masks (NO category-OR) for visual quality check.

目的：判断 SAM3 的 forward_grounding 对单个类别 prompt 是否真的能输出
「同一类别的多个实例 mask」，以及这些 mask 的边界质量。这正是「语义切边界、
几何定身份」方案的第一层，也是当前 kf_semantic.npz 在 OR 拍平前丢掉的信息。

对每帧：逐类别跑 grounding，保留 out_probs>0.1 的多个实例 mask（每个实例一个
mask，不 OR），叠加到原图保存，并打印实例清单（类别/序号/score/面积）。

Run on AutoDL (sgs env):
  /root/miniconda3/envs/sgs/bin/python dump_sam3_masks.py \
    --semantic /root/autodl-tmp/ablation/scannet/scene0000_00/kf_semantic.npz \
    --n_frames 3 --out_dir /root/autodl-tmp/sam3_mask_dump
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

# 40 色（tab20 + tab20b），按实例序号取模
TAB40 = ["#1f77b4","#aec7e8","#ff7f0e","#ffbb78","#2ca02c","#98df8a","#d62728",
         "#ff9896","#9467bd","#c5b0d5","#8c564b","#c49c94","#e377c2","#f7b6d2",
         "#7f7f7f","#c7c7c7","#bcbd22","#dbdb8d","#17becf","#9edae5",
         "#393b79","#5254a3","#6b6ecf","#9c9ede","#637939","#8ca252","#b5cf6b",
         "#cedb9c","#8c6d31","#bd9e39","#e7ba52","#e7cb94","#843c39","#ad494a",
         "#d6616b","#e7969c","#7b4173","#a55194","#ce6dbd","#de9ed6"]


def hex2bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))  # BGR


def sam3_instance_masks(img_path, sam3_model, processor, dummy_geo):
    """逐类别跑 grounding，保留每个类别的多个实例 mask（不 OR）。
    返回 instances = [(cat_idx, mask(H,W)bool, score), ...]，按 score 降序。"""
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

    instances = []
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
                probs = out_probs[keep]
                for i in range(len(masks_kept)):
                    m = masks_kept[i].cpu().numpy().astype(bool)
                    if int(m.sum()) < 50:
                        continue
                    instances.append((cat_idx, m, float(probs[i])))
        except Exception:
            continue
    instances.sort(key=lambda x: -x[2])
    return instances, img


def draw_instances(img_bgr, instances, out_png):
    """把实例 mask 叠加到原图：每个实例一个颜色（轮廓+轻填充），标签 = 类别#k。"""
    H, W = img_bgr.shape[:2]
    overlay = img_bgr.astype(np.float32)
    label_objs = [x for x in instances if x[2] >= 0.15][:30]   # 只画 top 30 的标签
    for rank, (cat_idx, mask, score) in enumerate(instances):
        col = hex2bgr(TAB40[rank % len(TAB40)])
        overlay[mask] = 0.82 * overlay[mask] + 0.18 * np.array(col, np.float32)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_bgr, cnts, -1, col, 2, cv2.LINE_AA)
    img_bgr = np.clip(overlay, 0, 255).astype(np.uint8)
    for rank, (cat_idx, mask, score) in enumerate(instances):
        if (cat_idx, score) not in [(x[0], x[2]) for x in label_objs]:
            continue
        ys, xs = np.nonzero(mask)
        cy, cx = int(ys.mean()), int(xs.mean())
        txt = f"{ALL_PROMPTS[cat_idx]}#{rank}({score:.2f})"
        col = hex2bgr(TAB40[rank % len(TAB40)])
        cv2.putText(img_bgr, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img_bgr, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, col, 1, cv2.LINE_AA)
    cv2.imwrite(out_png, img_bgr)
    return img_bgr


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
    print("[dump] loading SAM3 ...")
    sam3_model = build_sam3_image_model(
        bpe_path="/root/autodl-tmp/v3dsam/third_party/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        checkpoint_path="/root/autodl-tmp/v3dsam/third_party/sam3/sam3.pt", device="cuda")
    processor = Sam3Processor(sam3_model, confidence_threshold=0.1)
    dummy_geo = sam3_model._get_dummy_prompt(num_prompts=1)
    print("[dump] SAM3 loaded")

    for k, ii in enumerate(idxs):
        p = img_paths[ii]
        inst, img = sam3_instance_masks(p, sam3_model, processor, dummy_geo)
        inst = [x for x in inst if x[2] >= args.min_score]
        print(f"\n[dump] frame {k} (kf_idx {ii}) {os.path.basename(p)}: "
              f"{len(inst)} instance masks")
        from collections import Counter
        cc = Counter(ALL_PROMPTS[c] for c, _, _ in inst)
        for name, cnt in sorted(cc.items(), key=lambda x: -x[1]):
            print(f"   {name:<14} {cnt}")
        img_bgr = cv2.imread(p)
        out_png = os.path.join(args.out_dir, f"mask_frame{k}_kf{ii}.png")
        draw_instances(img_bgr, inst, out_png)
        print(f"   -> {out_png}")

    print("[dump] DONE")


if __name__ == "__main__":
    main()
