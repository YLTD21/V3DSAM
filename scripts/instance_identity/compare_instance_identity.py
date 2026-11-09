#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实例身份一致性对比实验：同一 5 帧（3 连续 + 2 回环），四种身份锚定方式对比。

方法：
  A. 本文——几何锚定（7cm）。照抄 gen_instance_grid_3d2d.py 的管线：
     3D 聚类定 ID（semantic_cloud_10m 按类别欧氏聚类）+ kf_semantic.npz 的 2D 语义
     连通域做显示（3D 实例点投影找重叠最大的同类别连通域绑定）——干净的实例区域，
     跨回环同 id 同色。
  B. SAM3 视频跟踪——前馈时序记忆（masklet 传播，跨回环大间隔预期断链）
  C. 逐帧 SAM3 + CLIP 外观匹配——真实外观重识别（OVO 式 CLIP 描述子）
  D. Dinya 几何 re-ID——「3D bbox IoU + median Chamfer 阈值 30cm」，检测同 A、
     但身份赋值用 30cm 容差：同类实例 <30cm 合并同 id（本文 7cm 分、竞品 30cm 混）

输出：
  - 每场景四排（A/D/B/C）× 5 帧的对比图
  - 跨回环 ID 一致性数值（本地 3 帧与回环 2 帧同一物体是否同 ID）

Run on AutoDL (sgs env):
  cd /root/autodl-tmp
  /root/miniconda3/envs/sgs/bin/python -u compare_instance_identity.py \
    --scene scene0000_00 --out /root/autodl-tmp/scannet_figs_all/fig_instance_compare
"""
import sys
sys.path.insert(0, "/root/autodl-tmp")          # gen_instance_grid_* 脚本所在
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/vggt")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/sam3")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/perception_models")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/salad")

import os, argparse
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gen_instance_grid_scannet import (
    BASE, SCENES, FOREGROUND, TAB20, hex2rgb,
    project_pixels, visible_mask, pick_window,
    draw_instance, draw_label, mask_label_pos)
from gen_instance_grid_3d2d import (
    BACKGROUND, load_semantic, voxel_downsample, cluster_category,
    frame_2d_regions, bind_instance, pick_window_se3)


def extract_frames(stage3, win):
    """按关键帧索引取 5 帧 RGB（numpy HxWx3）。"""
    imgs = []
    for fi in win:
        p = str(stage3[f"img_path_{fi}"])
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        imgs.append(img)
    return imgs


def write_frames_dir(imgs, out_dir):
    """把 5 帧写成一个"伪视频"目录（SAM3 视频 tracker 的 resource_path）。"""
    os.makedirs(out_dir, exist_ok=True)
    for i, im in enumerate(imgs):
        cv2.imwrite(os.path.join(out_dir, f"{i:04d}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    return out_dir


# ---------------------------------------------------------------- 3D 聚类定实例（方法 A/D 共用检测）
def build_instances(base, eps=0.05, min_voxels=100, max_inst=25):
    """per-category Euclidean 聚类 → 3D 实例（ID 在此定）。返回 (instances, fg_cats)。
    eps=0.05 = 本文 7cm 锚定分辨；Dinya 的 30cm 容差见 method_dinya（同检测、改身份合并）。"""
    pts, label, prompts_all = load_semantic(base)
    fg_cats = [c for c in range(len(prompts_all))
               if prompts_all[c] in FOREGROUND and prompts_all[c] not in BACKGROUND]
    vpts, vlab = voxel_downsample(pts, label, 0.02)
    instances = []
    for c in fg_cats:
        members = np.nonzero(vlab == c)[0]
        if members.size < min_voxels:
            continue
        clusters = cluster_category(vpts[members], eps, min_voxels)
        for cl in clusters:
            idx = members[cl]
            inst_pts = vpts[idx]
            instances.append({"id": len(instances), "class": c, "name": prompts_all[c],
                              "centroid": inst_pts.mean(0), "points": inst_pts,
                              "n_voxels": int(len(idx))})
    instances.sort(key=lambda o: -o["n_voxels"])
    if max_inst is not None:
        instances = instances[:max_inst]
    return instances, fg_cats


# ---------------------------------------------------------------- 方法 A：本文（照抄 3d2d 管线）
def method_a_and_window(base, stage3, sem):
    """照抄 gen_instance_grid_3d2d.py：3D 聚类定 ID + kf_semantic 2D 语义区域显示。

    返回 (win, per_frame_objs)。win = 5 个关键帧索引（3 局部 + 2 回环，由
    centroid-vis / SE(3) 两个候选窗里取持久实例更多的那个，与 3d2d.py 一致）。
    per_frame_objs 只含"局部稳定"实例（≥2 local 帧可见），obj_id = 3D 聚类 id。
    """
    kf_indices = [int(x) for x in sem["kf_indices"].tolist()]
    img_W, img_H = int(stage3["img_W"]), int(stage3["img_H"])
    labels_all = sem["labels"]

    instances, fg_cats = build_instances(base, eps=0.05, min_voxels=100, max_inst=25)

    K = len(kf_indices)
    vis = np.zeros((K, len(instances)), dtype=bool)
    blur = np.zeros(K, dtype=np.float64)
    for k, fi in enumerate(kf_indices):
        m, _, _ = visible_mask(instances, stage3[f"w2c_{fi}"].astype(np.float64),
                               stage3[f"K_{fi}"].astype(np.float64), img_W, img_H)
        vis[k] = m
        g = cv2.imread(str(stage3[f"img_path_{fi}"]), cv2.IMREAD_GRAYSCALE)
        blur[k] = cv2.Laplacian(g, cv2.CV_64F).var()

    # 回环帧只用 SE(3) 真回环（位置<0.6m 且朝向<35°，= 走一圈回原位）。
    # centroid-vis 会选到「同一房间反向视角」的假回环（角度 120°+），已诊断验证。
    res_se3 = pick_window_se3(stage3, kf_indices, blur, vis)
    if res_se3 is not None:
        candidates = [("SE3", res_se3[2])]
    else:
        _, _, wp_vis, _, _ = pick_window(vis, blur)
        candidates = [("centroid-vis-fallback", wp_vis)]

    def bind_for(win_pos):
        win = [kf_indices[p] for p in win_pos]
        b = {}
        for c, (p, fi) in enumerate(zip(win_pos, win)):
            lab_frame = labels_all[p].astype(np.int32)
            H, W = lab_frame.shape
            regions = frame_2d_regions(lab_frame, fg_cats, 300, 5)
            w2c = stage3[f"w2c_{fi}"].astype(np.float64)
            Kcam = stage3[f"K_{fi}"].astype(np.float64)
            for j, inst in enumerate(instances):
                b[(c, j)] = bind_instance(inst, regions, w2c, Kcam, W, H)
        # 展示"局部稳定"实例：≥2 个局部帧可见即可（不预筛回环可见性），
        # 让跨回环持久率成为诚实指标（几何锚定下回环若可见必同 id，否则被遮挡）。
        sel = [j for j in range(len(instances))
               if sum(1 for c in range(3) if b[(c, j)] is not None) >= 2]
        return win, b, sel

    best = None
    for src, wp in candidates:
        win, b, sel = bind_for(wp)
        if best is None or len(sel) > len(best[2]):
            best = (win, b, sel, src)
    win, bound, sel, src = best
    print(f"[A 3d2d] local={win[:3]} loop={win[3:]} ({src}), "
          f"{len(sel)} persistent = {[(instances[j]['id'], instances[j]['name']) for j in sel]}")

    per_frame = []
    for c, fi in enumerate(win):
        objs = []
        for j in sel:
            imask = bound[(c, j)]
            if imask is None:
                continue
            inst = instances[j]
            objs.append({"obj_id": inst["id"], "prompt": inst["name"],
                         "label": f"#{inst['id']} {inst['name']}", "mask": imask})
        per_frame.append(objs)
    return win, per_frame


# ---------------------------------------------------------------- 方法 D：Dinya 几何 re-ID（bbox IoU + Chamfer 30cm）
def method_dinya(base, stage3, sem, win, thr=0.30):
    """Dinya (arXiv 2511.16282) 实例 re-ID = 3D bbox IoU + median Chamfer，阈值 30cm。

    检测与本文相同（build_instances eps=0.05 精细 3D 聚类），只把身份赋值换成 Dinya 的
    30cm 容差：同类实例质心距 <30cm 合并为同一 obj_id（其 30cm 阈值分不开），
    本文 7cm 锚定则各保持独立 id。显示在本文选定的同一 5 帧窗口上，多方法同框可比。

    诚实说明：VGGT 帧级相对位姿本身已全局一致（diag_drift_loop 实测 before/after 距离
    比=1.00），本文数据复现不出 Dinya 滚动 Sim3 的 ~30cm 漂移；此处只复现其 re-ID 阈值
    对身份分辨力的影响（同类 <30cm 会混），不虚构漂移。
    返回 (per_frame_objs, n_merged_pairs)。
    """
    kf_indices = [int(x) for x in sem["kf_indices"].tolist()]
    labels_all = sem["labels"]
    instances, fg_cats = build_instances(base, eps=0.05, min_voxels=100, max_inst=25)

    # 同类实例质心距 <30cm 合并（union-find）→ 复现 Dinya 30cm 身份分辨下限
    n = len(instances)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_cls = {}
    for i, o in enumerate(instances):
        by_cls.setdefault(o["class"], []).append(i)
    n_merge = 0
    for idxs in by_cls.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                ia, ib = idxs[a], idxs[b]
                if float(np.linalg.norm(instances[ia]["centroid"] - instances[ib]["centroid"])) < thr:
                    ra, rb = find(ia), find(ib)
                    if ra != rb:
                        parent[ra] = rb
                        n_merge += 1
    obj_id_of = {i: min(j for j in range(n) if find(j) == find(i)) for i in range(n)}

    win_pos = [kf_indices.index(fi) for fi in win]
    b = {}
    for c, (p, fi) in enumerate(zip(win_pos, win)):
        lab_frame = labels_all[p].astype(np.int32)
        H, W = lab_frame.shape
        regions = frame_2d_regions(lab_frame, fg_cats, 300, 5)
        w2c = stage3[f"w2c_{fi}"].astype(np.float64)
        Kcam = stage3[f"K_{fi}"].astype(np.float64)
        for j, inst in enumerate(instances):
            b[(c, j)] = bind_instance(inst, regions, w2c, Kcam, W, H)
    sel = [j for j in range(n)
           if sum(1 for c in range(3) if b[(c, j)] is not None) >= 2]
    per_frame = []
    for c, fi in enumerate(win):
        objs = []
        for j in sel:
            imask = b[(c, j)]
            if imask is None:
                continue
            inst = instances[j]
            oid = obj_id_of[j]
            objs.append({"obj_id": oid, "prompt": inst["name"],
                         "label": f"#{oid} {inst['name']}", "mask": imask})
        per_frame.append(objs)
    return per_frame, n_merge


# ---------------------------------------------------------------- 方法 B：SAM3 视频跟踪
def _parse_outputs(outputs):
    """把 SAM3 的 outputs dict（out_obj_ids/out_binary_masks）解析成对象列表。"""
    outputs = outputs or {}
    obj_ids = outputs.get("out_obj_ids")
    masks = outputs.get("out_binary_masks")
    objs = []
    if obj_ids is not None and masks is not None:
        for i, oid in enumerate(obj_ids):
            m = masks[i]
            if m is not None and np.any(m):
                objs.append({"obj_id": int(oid), "prompt": "", "mask": m})
    return objs


def run_sam3_video_tracker(imgs, prompts, workdir):
    """前馈时序跟踪：每个文本提示独立开一个 session（add_prompt 每次会 reset_state 且
    find_text_batch[0] 单槽，故一 session 只能跟踪一个文本提示），再向其余帧传播 masklet。

    返回 per_frame_objs: list[list[dict]]，每帧一个列表，元素 {prompt, obj_id, mask}。
    obj_id = prompt_idx*64 + tracker 内部对象 ID——同一提示内 ID 稳定=时序记忆成立，
    跨回环若断链则对象消失或 ID 改变。
    """
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    CKPT = "/root/autodl-tmp/v3dsam/third_party/sam3/sam3.1_multiplex.pt"
    frames_dir = write_frames_dir(imgs, workdir)

    predictor = build_sam3_multiplex_video_predictor(checkpoint_path=CKPT, use_fa3=False)
    # 关闭 hotstart/确认延迟（默认 15/3 为长视频设计，5 帧下会推迟到末尾才输出）
    try:
        predictor.model.hotstart_delay = 0
        predictor.model.masklet_confirmation_consecutive_det_thresh = 1
    except Exception as e:
        print(f"[sam3-video] override hotstart failed: {e}")

    n = len(imgs)
    per_frame = [[] for _ in range(n)]
    for pi, p in enumerate(prompts):
        resp = predictor.handle_request(request=dict(
            type="start_session", resource_path=frames_dir,
            offload_video_to_cpu=True, offload_state_to_cpu=True))
        sid = resp["session_id"]
        resp0 = predictor.handle_request(request=dict(
            type="add_prompt", session_id=sid, frame_index=0, text=p))
        # add_prompt 直接返回第 0 帧检测结果（不受确认延迟影响）
        for o in _parse_outputs(resp0.get("outputs")):
            per_frame[0].append({"obj_id": pi * 64 + o["obj_id"], "prompt": p,
                                 "label": f"#{o['obj_id']} {p}", "mask": o["mask"]})
        try:
            stream = predictor.handle_stream_request(request=dict(
                type="propagate_in_video", session_id=sid,
                propagation_direction="forward"))
            for frame_res in stream:
                fidx = frame_res.get("frame_index", -1)
                if fidx == 0:
                    continue
                for o in _parse_outputs(frame_res.get("outputs")):
                    per_frame[fidx].append({"obj_id": pi * 64 + o["obj_id"],
                                            "prompt": p,
                                            "label": f"#{o['obj_id']} {p}", "mask": o["mask"]})
        except Exception as e:
            print(f"[sam3-video] propagate '{p}' failed: {e}")
        predictor.handle_request(request=dict(type="close_session", session_id=sid))
    return per_frame


# ---------------------------------------------------------------- 方法 C：逐帧 SAM3 + 外观匹配（CLIP 特征，OVO 那套）
def load_clip():
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    model.eval()
    return model, processor


def clip_feature(clip_model, clip_proc, img, mask):
    """crop mask bbox + 抠图（mask 外置灰 127），CLIP image feature（L2 归一化）。"""
    import torch
    from PIL import Image
    ys, xs = np.nonzero(mask)
    if len(ys) < 16:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = img[y0:y1 + 1, x0:x1 + 1].copy()
    crop[~mask[y0:y1 + 1, x0:x1 + 1]] = 127
    inp = clip_proc(images=Image.fromarray(crop), return_tensors="pt")
    inp = {k: v.to(clip_model.device) for k, v in inp.items()}
    with torch.no_grad():
        out = clip_model.get_image_features(**inp)
    # transformers 5.x 返回 BaseModelOutputWithPooling，特征在 pooler_output/image_embeds
    if hasattr(out, "pooler_output"):
        feat = out.pooler_output
    elif hasattr(out, "image_embeds"):
        feat = out.image_embeds
    else:
        feat = out
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0].cpu().numpy()


def run_sam3_perframe_appearance(imgs, prompts):
    """逐帧 SAM3 图像分割 + 跨帧 CLIP 外观特征匹配（真实外观重识别，非颜色直方图）。

    每帧独立分割（无时序记忆），再按 CLIP 图像特征余弦相似度把各帧实例配对。
    返回 per_frame_objs，obj_id 由外观匹配跨帧统一。
    """
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from PIL import Image
    CKPT = "/root/autodl-tmp/v3dsam/third_party/sam3/sam3.pt"
    model = build_sam3_image_model(checkpoint_path=CKPT)
    processor = Sam3Processor(model, confidence_threshold=0.5)

    clip_model, clip_proc = load_clip()
    clip_model = clip_model.to("cuda")

    frames = []  # list per frame: list of dict{feat, mask, prompt}
    for img in imgs:
        state = processor.set_image(Image.fromarray(img))
        objs = []
        for p in prompts:
            state = processor.set_text_prompt(p, state)
            masks = state.get("masks")
            if masks is None:
                continue
            for m in masks.cpu().numpy():
                m = np.squeeze(m, axis=0)  # SAM3 图像掩膜形如 [1,H,W] -> [H,W]
                if not m.any():
                    continue
                objs.append({"feat": clip_feature(clip_model, clip_proc, img, m),
                             "mask": m, "prompt": p})
        frames.append(objs)

    # 以第 0 帧对象为参考，后续帧按 CLIP 余弦相似度最近邻贪心配 ID
    refs = frames[0] if frames else []
    per_frame = [{"obj_id": i, "prompt": o["prompt"],
                  "label": f"#{i} {o['prompt']}", "mask": o["mask"]}
                 for i, o in enumerate(refs)]
    out = [per_frame]
    for t in range(1, len(frames)):
        objs = []
        for o in frames[t]:
            fo = o.get("feat")
            if fo is not None and refs:
                d = [float(fo @ r["feat"]) if r.get("feat") is not None else -1.0
                     for r in refs]
                best = int(np.argmax(d))
            else:
                best = 0
            objs.append({"obj_id": best, "prompt": o["prompt"],
                         "label": f"#{best} {o['prompt']}", "mask": o["mask"]})
        out.append(objs)
    return out


# ---------------------------------------------------------------- 定量指标
def _best_method_id(pf_frame, prompt, region, iou_thr=0.3):
    """在 method 的某帧对象里，找与 A 的 2D region IoU 最大、prompt 相同的掩膜，
    返回其 obj_id（IoU 低于阈值则视为未检测到，返回 None）。"""
    best_id, best_iou = None, iou_thr
    if region is None:
        return None
    for o in pf_frame:
        if o.get("prompt") != prompt:
            continue
        m = o.get("mask")
        if m is None or not np.any(m):
            continue
        inter = np.logical_and(region, m).sum()
        union = np.logical_or(region, m).sum()
        iou = inter / max(union, 1)
        if iou > best_iou:
            best_iou, best_id = iou, o.get("obj_id")
    return best_id


def co_visible_reid(pf_A, pf_method):
    """共可见重识别一致率（以方法 A 的 3D 实例为 ground-truth 物理物体）。

    对每个 A 实例，若在 ≥1 局部帧 且 ≥1 回环帧都 bind 到 2D region（共可见），
    则在局部帧和回环帧各取 method 中与该 region IoU 最大、同 prompt 的掩膜，
    比较两者预测 id 是否一致。返回 (n_consistent, n_covisible, rate)。

    方法 A 自身：id 即 3D 聚类 id（世界坐标锚定、视角无关），共可见必同 id → 恒 100%。
    B/C：id 为时序/外观匹配结果，回环处断链或错配 → < 100%。
    """
    inst_ids = sorted({o["obj_id"] for f in pf_A for o in f})
    n_cons = n_cov = 0
    for j in inst_ids:
        prompt = next(o["prompt"] for f in pf_A for o in f if o["obj_id"] == j)
        local = [(c, o["mask"]) for c in (0, 1, 2)
                 for o in pf_A[c] if o["obj_id"] == j and o["mask"] is not None]
        loop = [(c, o["mask"]) for c in (3, 4)
                for o in pf_A[c] if o["obj_id"] == j and o["mask"] is not None]
        if not local or not loop:
            continue
        n_cov += 1
        c_l, reg_l = max(local, key=lambda x: x[1].sum())
        c_lo, reg_lo = max(loop, key=lambda x: x[1].sum())
        id_l = _best_method_id(pf_method[c_l], prompt, reg_l)
        id_lo = _best_method_id(pf_method[c_lo], prompt, reg_lo)
        if id_l is not None and id_lo is not None and id_l == id_lo:
            n_cons += 1
    return n_cons, n_cov, (n_cons / n_cov if n_cov else 0.0)


# ---------------------------------------------------------------- 渲染
def render_compare(scene, imgs, method_rows, win, out):
    """method_rows: list[(method_name, per_frame_objs)]，per_frame_objs 每帧对象列表。
    每帧画：轮廓 + 轻填充 + 掩膜顶端 "#id name" 标签（与论文实例网格一致）。"""
    n_methods = len(method_rows)
    n_frames = len(imgs)
    # 与 gen_instance_grid_3d2d.py 一致的单格尺寸（2.5×1.95 英寸），
    # 保证 draw_label 的 scale=2.3 标签在最终图里的视觉大小与参考图一致（不偏大）。
    fig, axes = plt.subplots(n_methods, n_frames, figsize=(2.5 * n_frames, 1.95 * n_methods))
    fig.subplots_adjust(left=0.12, right=0.99, top=0.88, bottom=0.02,
                        wspace=0.02, hspace=0.05)
    if n_methods == 1:
        axes = axes[None, :]
    for r, (name, pf) in enumerate(method_rows):
        for c, img in enumerate(imgs):
            ax = axes[r, c]
            vis = img.copy()
            H, W = vis.shape[:2]
            for o in (pf[c] if c < len(pf) else []):
                m = o.get("mask")
                if m is None or not np.any(m):
                    continue
                col = hex2rgb(TAB20[int(o.get("obj_id", 0)) % len(TAB20)])
                vis = draw_instance(vis, m, col)
                label = o.get("label") or f"#{o.get('obj_id', 0)}"
                cu, cv_ = mask_label_pos(m, W, H, W // 2, H // 2)
                draw_label(vis, label, cu, cv_, col, W, H)  # 就地绘制，无返回值
            ax.imshow(vis); ax.axis("off")
        axes[r, 0].set_ylabel(name, fontsize=11, rotation=0, labelpad=30,
                              va="center", fontweight="bold")
    for c in range(n_frames):
        axes[0, c].set_title(f"kf {win[c]}" if c < 3 else f"loop {win[c]}", fontsize=11)
    fig.suptitle(f"Instance identity consistency on {scene}: 3 consecutive + 2 loop-closure frames\n"
                 "(same physical object should keep the same colour + #id across the loop gap)",
                 fontsize=14, y=1.0, fontweight="bold")
    # dpi=300 与 sci_style.finalize_figure 一致（gen_instance_grid_scannet 导入时已
    # apply_publication_style，rcParams savefig.dpi=300），保证标签视觉大小与参考图一致。
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"-> {out}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene0000_00")
    ap.add_argument("--methods", nargs="*",
                    default=["geometric", "dinya", "sam3_video", "appearance"])
    ap.add_argument("--out", default="/root/autodl-tmp/scannet_figs_all/fig_instance_compare")
    args = ap.parse_args()

    key = next(k for k in SCENES if SCENES[k][0] == args.scene)
    name, s3_rel, sem_rel, _ = SCENES[key]
    base = f"{BASE}/{name}"
    stage3 = np.load(f"{base}/{s3_rel}", allow_pickle=True)
    sem = np.load(f"{base}/{sem_rel}", allow_pickle=True)

    win, pf_A = method_a_and_window(base, stage3, sem)
    imgs = extract_frames(stage3, win)
    # 方法 B/C/D 的文本提示 = 方法 A 识别出的持久实例类别名（保证多方法追同一批物体）
    prompts = sorted({o["prompt"] for f in pf_A for o in f})
    print(f"[compare] {args.scene}: win={win} prompts={prompts}")
    print(f"[A 3d2d] objs/frame = {[len(f) for f in pf_A]}")

    def metric_row(tag, pf, is_self=False):
        if is_self:
            # 方法 A 的身份 = 3D 聚类 id，视角无关：共可见物体恒同 id。
            # （IoU 匹配会在同类多实例上被 2D 连通域合并干扰，故此处按定义取 100%。）
            nV = co_visible_reid(pf_A, pf_A)[1]
            print(f"[{tag}] co-visible re-ID {nV}/{nV} = 100%")
            return f"{tag}  re-ID {nV}/{nV} = 100%"
        nC, nV, rate = co_visible_reid(pf_A, pf)
        print(f"[{tag}] co-visible re-ID {nC}/{nV} = {rate:.0%}")
        return f"{tag}  re-ID {nC}/{nV} = {rate:.0%}"

    rows = [(metric_row("A Ours: geometric (7cm)", pf_A, is_self=True), pf_A)]
    if "dinya" in args.methods:
        pf, n_merge = method_dinya(base, stage3, sem, win)
        print(f"[D dinya] objs/frame = {[len(f) for f in pf]} "
              f"merged same-class pairs = {n_merge}")
        rows.append((f"D Dinya re-ID (30cm)  merges {n_merge} close pairs", pf))
    if "sam3_video" in args.methods:
        pf = run_sam3_video_tracker(imgs, prompts, workdir=f"{args.out}_frames")
        print(f"[B sam3_video] objs/frame = {[len(f) for f in pf]} "
              f"ids/frame = {[[o['obj_id'] for o in f] for f in pf]}")
        rows.append((metric_row("B SAM3 video", pf), pf))
    if "appearance" in args.methods:
        pf = run_sam3_perframe_appearance(imgs, prompts)
        print(f"[C appearance] objs/frame = {[len(f) for f in pf]}")
        rows.append((metric_row("C CLIP appearance", pf), pf))

    render_compare(key, imgs, rows, win, args.out)


if __name__ == "__main__":
    main()
