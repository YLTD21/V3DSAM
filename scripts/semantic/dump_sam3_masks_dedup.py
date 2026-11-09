#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAM3 实例 mask 去重 + 实心区域可视化（验证「语义切边界」第一层）。

问题：SAM3 forward_grounding 对单个类别输出的 pred_masks 是大量重复预测
（同一本书被预测几十次，mask 几乎重叠堆叠）。本脚本在同类别内做 IoU 图聚类
去重：IoU >= iou_thresh 的 mask 归为同一实例，取并集得到一个干净的实例区域。

输出：每帧一张「实例区域图」——每个实例一个纯色实心区域 + 轮廓 + "#id 类别"，
颜色按全局实例 id（跨类别）。对比 dump_sam3_masks.py 的堆叠版本看效果。

Run on AutoDL (sgs env):
  /root/miniconda3/envs/sgs/bin/python dump_sam3_masks_dedup.py \
    --semantic /root/autodl-tmp/ablation/scannet/scene0000_00/kf_semantic.npz \
    --n_frames 3 --out_dir /root/autodl-tmp/sam3_mask_dedup
"""
import os, sys, argparse
import numpy as np, torch, torch.nn.functional as F
import cv2
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

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


def sam3_raw_masks(img_path, sam3_model, processor, dummy_geo, min_score=0.1):
    """逐类别跑 grounding，返回所有 raw mask（含重复）。"""
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
            keep = out_probs > min_score
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
    return instances, img


def dedup(instances, iou_thresh=0.5):
    """同类别内 IoU 图聚类去重。返回 [(cat_idx, merged_mask, score, n_src), ...]，
    每个分量 = 一个实例，merged_mask = 分量内所有 mask 的并集。"""
    by_cat = {}
    for cat_idx, m, s in instances:
        by_cat.setdefault(cat_idx, []).append((m, s))
    out = []
    for cat_idx, items in by_cat.items():
        n = len(items)
        if n == 1:
            m, s = items[0]
            out.append((cat_idx, m, s, 1))
            continue
        H, W = items[0][0].shape
        M = np.stack([m.reshape(-1) for m, _ in items]).astype(np.float32)
        area = M.sum(1)
        inter = M @ M.T
        union = area[:, None] + area[None, :] - inter
        iou = inter / np.maximum(union, 1e-6)
        adj = iou >= iou_thresh
        n_cc, labels = connected_components(csr_matrix(adj), directed=False)
        for cc in range(n_cc):
            idx = np.where(labels == cc)[0]
            merged = M[idx].max(0).reshape(H, W).astype(bool)
            sc = max(items[i][1] for i in idx)
            out.append((cat_idx, merged, sc, len(idx)))
    return out


def draw_regions(img_bgr, instances, out_png):
    """每个实例一个纯色实心区域 + 轮廓 + '#id 类别' 标签。"""
    H, W = img_bgr.shape[:2]
    overlay = img_bgr.astype(np.float32)
    for rank, (cat_idx, mask, score, n_src) in enumerate(instances):
        col = hex2bgr(TAB40[rank % len(TAB40)])
        overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.array(col, np.float32)
    img_bgr = np.clip(overlay, 0, 255).astype(np.uint8)
    for rank, (cat_idx, mask, score, n_src) in enumerate(instances):
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
    ap.add_argument("--iou_thresh", type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sem = np.load(args.semantic, allow_pickle=True)
    img_paths = [str(p) for p in sem["img_paths"]]
    n = len(img_paths)
    idxs = np.linspace(0, n - 1, args.n_frames).astype(int)

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    print("[dedup] loading SAM3 ...")
    sam3_model = build_sam3_image_model(
        bpe_path="/root/autodl-tmp/v3dsam/third_party/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        checkpoint_path="/root/autodl-tmp/v3dsam/third_party/sam3/sam3.pt", device="cuda")
    processor = Sam3Processor(sam3_model, confidence_threshold=0.1)
    dummy_geo = sam3_model._get_dummy_prompt(num_prompts=1)
    print("[dedup] SAM3 loaded")

    for k, ii in enumerate(idxs):
        p = img_paths[ii]
        raw, img = sam3_raw_masks(p, sam3_model, processor, dummy_geo, args.min_score)
        inst = dedup(raw, iou_thresh=args.iou_thresh)
        inst.sort(key=lambda x: -x[2])
        # 面积过滤：太小当噪声
        inst = [x for x in inst if int(x[1].sum()) >= 200]
        print(f"\n[dedup] frame {k} (kf_idx {ii}) {os.path.basename(p)}: "
              f"{len(raw)} raw masks -> {len(inst)} instances")
        from collections import Counter
        cc = Counter(ALL_PROMPTS[c] for c, _, _, _ in inst)
        for name, cnt in sorted(cc.items(), key=lambda x: -x[1]):
            print(f"   {name:<14} {cnt}")
        img_bgr = cv2.imread(p)
        out_png = os.path.join(args.out_dir, f"dedup_frame{k}_kf{ii}.png")
        draw_regions(img_bgr, inst, out_png)
        print(f"   -> {out_png}")

    print("[dedup] DONE")


if __name__ == "__main__":
    main()
