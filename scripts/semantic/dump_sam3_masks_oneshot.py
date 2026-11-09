#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAM3 一次性多类别实例分割（官方 batched API + 跨类别 argmax 去重）。

一次 forward 喂全部 50 类：pred_logits=(50,200,1) → squeeze 成 (50,200)，
对 200 个 object query 做「跨类别 argmax」，每个 query 归属它得分最高的一个类别，
于是同一物体不再被几十个 query 重复命中（book 从 50~110 → ~3-5）。

对比：
  dump_sam3_masks.py           循环 50 次单类别 + 无去重（book 50~110）
  dump_sam3_masks_dedup.py     循环 50 次 + 同类别 IoU 去重（book 仍 ~40）
  本脚本                        一次 50 类 + 跨类别 argmax（真正降重）

Run on AutoDL (sgs env):
  /root/miniconda3/envs/sgs/bin/python dump_sam3_masks_oneshot.py \
    --semantic /root/autodl-tmp/ablation/scannet/scene0000_00/kf_semantic.npz \
    --n_frames 3 --out_dir /root/autodl-tmp/sam3_mask_oneshot
"""
import os, sys, argparse
import numpy as np, torch, torch.nn.functional as F
import cv2
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/v3dsam")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/sam3")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/vggt")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/perception_models")
os.environ['XFORMERS_DISABLED'] = '1'

from sam3 import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.transforms.basic_for_api import (
    ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI)
from sam3.train.data.sam3_image_dataset import (
    InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint)

ALL_PROMPTS = [
    "wall","floor","ceiling","door","window","stairs","railing",
    "bed","cabinet","table","sofa",
    "sink","toilet","bathtub","mirror","lamp",
    "television","laptop","keyboard","mouse","phone","refrigerator","oven","microwave",
    "chair","plant","vase","picture","curtain","pillow","clock",
    "book","bottle","cup","bowl","plate","bag","box","person",
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


def build_transform():
    return ComposeAPI(transforms=[
        RandomResizeAPI(sizes=1008, max_size=1008, square=True,
                        consistent_transform=False),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def make_datapoint(pil_img):
    w, h = pil_img.size
    dp = Datapoint(find_queries=[], images=[])
    dp.images = [SAMImage(data=pil_img, objects=[], size=[h, w])]
    for c in range(N_cats):
        dp.find_queries.append(FindQueryLoaded(
            query_text=ALL_PROMPTS[c],
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=c,
                original_image_id=c,
                original_category_id=c,
                original_size=[h, w],
                object_id=0,
                frame_index=0,
            ),
        ))
    return dp


def oneshot_masks(img_path, model, transform, device, min_score=0.1):
    """一次 forward 预测全部 50 类，跨类别 argmax 后返回 [(cat_idx, mask, score), ...]。"""
    img = Image.open(img_path).convert("RGB")
    w0, h0 = img.size
    dp = make_datapoint(img)
    dp = transform(dp)
    batch = collate([dp], dict_key="dummy")["dummy"]
    batch = copy_data_to_device(batch, device, non_blocking=True)
    with torch.inference_mode():
        output = model(batch)
    out = list(output)[0]  # 单 stage，LAST_STEP_PER_STAGE

    if os.environ.get("ONESHOT_DEBUG"):
        for k in ("pred_logits", "pred_masks", "pred_boxes", "presence_logit_dec"):
            v = out.get(k)
            print(f"  [shape] {k}: {tuple(v.shape) if v is not None else None}")

    logits = out["pred_logits"]
    if logits.dim() == 4:
        logits = logits[-1]
    logits = logits.squeeze(-1)                       # (50, 200)
    probs = logits.sigmoid()
    presence = out.get("presence_logit_dec")
    if presence is not None:
        p = presence.sigmoid()
        if p.dim() == 4:
            p = p[-1]
        # presence=(N_cats,1)：每类一个"该类是否存在"分数，逐类广播乘到 query 概率
        if p.dim() == 2 and p.shape[-1] == 1 and p.shape[0] == probs.shape[0]:
            probs = probs * p

    best_score, best_cat = probs.max(0)               # (200,), (200,)

    masks = out["pred_masks"]
    if masks.dim() == 5:
        masks = masks[-1]                             # (50, 200, H, W)
    q = torch.arange(masks.shape[1], device=masks.device)
    sel_masks = masks[best_cat, q]                    # (200, H, W) 每 query 取其最佳类别的 mask
    sel_masks = F.interpolate(sel_masks.unsqueeze(1).float(), (h0, w0),
                              mode="bilinear", align_corners=False).squeeze(1)
    sel_masks = (sel_masks.sigmoid() > 0.5).cpu().numpy().astype(np.uint8)

    # 类别并集 + 连通域：把同一类别的所有 query 并成一张 mask，闭运算粘合碎片间小缝，
    # 再取连通域，每个连通域 = 一个实例。吸收「热门类别被几十个 query 切成碎片」问题。
    keep = (best_score > min_score).cpu().numpy()
    cat = best_cat.cpu().numpy().astype(int)
    score = best_score.cpu().numpy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    instances = []
    for c in range(N_cats):
        idx = np.where(keep & (cat == c))[0]
        if len(idx) == 0:
            continue
        union = sel_masks[idx].max(0)
        union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, kernel)
        n_lab, lab = cv2.connectedComponents(union, connectivity=8)
        sc = float(score[idx].max())
        for comp in range(1, n_lab):
            m = (lab == comp)
            if int(m.sum()) < 300:
                continue
            instances.append((c, m, sc))
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
    ap.add_argument("--min_score", type=float, default=0.2)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sem = np.load(args.semantic, allow_pickle=True)
    img_paths = [str(p) for p in sem["img_paths"]]
    n = len(img_paths)
    idxs = np.linspace(0, n - 1, args.n_frames).astype(int)

    print("[oneshot] loading SAM3 ...")
    model = build_sam3_image_model(
        bpe_path="/root/autodl-tmp/v3dsam/third_party/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        checkpoint_path="/root/autodl-tmp/v3dsam/third_party/sam3/sam3.pt",
        device="cuda")
    transform = build_transform()
    device = torch.device("cuda")
    print("[oneshot] SAM3 loaded")

    for k, ii in enumerate(idxs):
        p = img_paths[ii]
        inst, img = oneshot_masks(p, model, transform, device, args.min_score)
        print(f"\n[oneshot] frame {k} (kf_idx {ii}) {os.path.basename(p)}: "
              f"{len(inst)} instance masks")
        from collections import Counter
        cc = Counter(ALL_PROMPTS[c] for c, _, _ in inst)
        for name, cnt in sorted(cc.items(), key=lambda x: -x[1]):
            print(f"   {name:<14} {cnt}")
        img_bgr = cv2.imread(p)
        out_png = os.path.join(args.out_dir, f"oneshot_frame{k}_kf{ii}.png")
        draw_regions(img_bgr, inst, out_png)
        print(f"   -> {out_png}")

    print("[oneshot] DONE")


if __name__ == "__main__":
    main()
