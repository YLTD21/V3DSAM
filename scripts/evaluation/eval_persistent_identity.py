#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""持久实例身份统一评估：4 方法同 GT matching + frame set。

方法（同一套 GT = ScanNet instance-filt，同一关键帧集）：
  V3DSAM (geometric)   3D 语义实例点（semantic_cloud_10m）投影 -> 凸包掩膜，跨帧 ID 由世界坐标锚定
  SAM3 Tracking        前馈时序 masklet 传播（SAM3 视频跟踪，跨回环大间隔预期断链）
  CLIP Association     逐帧 SAM3 连通域（kf_semantic）+ CLIP 外观特征跨帧关联
  OV baseline          逐帧 SAM3 检测（kf_semantic 连通域），无跨帧身份（帧局部 ID）

指标（MOT 口径，同一 (GT实例 × 可见帧) 对上）：
  IDC              ID Consistency = 匹配 ID == 众数 ID 的帧占比
  total IDS        ID Switches 总数
  mean IDS/obj     total IDS / GT 实例数
  total FM         Fragmentation = 可见但未匹配帧总数
  mean FM/obj      total FM / GT 实例数
  ReID-R           Long-term Re-ID Recall = 首末可见帧跨度>=25% 场景固定序列长度时，
                   首末都匹配上且 predicted ID 相同的占比；分母只由 GT 决定（无条件）
  GT Coverage      匹配率 = matched pairs / visible pairs
  Class Consistency 匹配对里类别一致的占比

输出：persistent_metrics_per_scene.csv + persistent_metrics_overall.csv

Run on AutoDL (sgs env):
  cd /root/autodl-tmp
  /root/miniconda3/envs/sgs/bin/python -u eval_persistent_identity.py \
    --methods v3dsam ov_baseline --max_frames 200          # smoke
  /root/miniconda3/envs/sgs/bin/python -u eval_persistent_identity.py   # 全量
"""
import os, sys, argparse, pickle, gzip
import numpy as np, cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scannet_labels import NYU40, map_gt
from instance_gt_common import (
    FOREGROUND_NYU, FG_LIST, PROMPT2NYU,
    project_pixels, convex_hull_mask, match_frame, match_frame_full, compute_metrics,
    compute_mota, compute_idf1, compute_4d_pq)

ABL = "/root/autodl-tmp/ablation/scannet"
DAT = "/root/autodl-tmp/v3dsam/datasets/scannet"
INST = "/root/autodl-tmp/gt_inst_filt"
SAM3_DIR = "/root/autodl-tmp/v3dsam/third_party/sam3"
SAM3_VIDEO_CKPT = f"{SAM3_DIR}/sam3.1_multiplex.pt"
SCENES = ["scene0000_00", "scene0059_00", "scene0106_00", "scene0207_00"]


# ---------------------------------------------------------------- GT 加载
def load_gt_instances(scene, kf_indices, min_px=50):
    inst_dir = f"{INST}/{scene}"
    sem_dir = f"{DAT}/{scene}/semantic_id"
    px_cls, frames = {}, {}
    for fi in kf_indices:
        inst = cv2.imread(f"{inst_dir}/{fi}.png", cv2.IMREAD_UNCHANGED)
        if inst is None:
            continue
        sem = map_gt(np.array(Image.open(f"{sem_dir}/{fi}.png")))
        if sem.shape != inst.shape:
            sem = cv2.resize(sem.astype(np.float32), (inst.shape[1], inst.shape[0]),
                             interpolation=cv2.INTER_NEAREST).astype(np.int16)
        for iid in np.unique(inst):
            if iid == 0:
                continue
            m = inst == iid
            if int(m.sum()) < min_px:
                continue
            iid = int(iid)
            frames.setdefault(iid, []).append(fi)
            cls, cnt = np.unique(sem[m], return_counts=True)
            d = px_cls.setdefault(iid, {})
            for c, n in zip(cls, cnt):
                if int(c) >= 0:
                    d[int(c)] = d.get(int(c), 0) + int(n)
    inst_class = {}
    for iid, d in px_cls.items():
        inst_class[iid] = max(d, key=d.get) if d else -1
    return inst_class, frames


# ---------------------------------------------------------------- V3DSAM（几何）
def load_v3dsam(scene, min_pts=30, subdir="semantic_cloud_10m"):
    """从 {subdir}/instance_clustered.ply + instance_legend.txt 加载 3D 实例。
    subdir="semantic_cloud_gt" 即 GT 几何版（对称消融）。"""
    base = f"{ABL}/{scene}"
    ply_path = f"{base}/{subdir}/instance_clustered.ply"
    legend_path = f"{base}/{subdir}/instance_legend.txt"
    pts, labels = [], []
    header = True
    with open(ply_path) as f:
        for line in f:
            if header:
                if line.strip() == "end_header":
                    header = False
                continue
            p = line.split()
            if len(p) < 7:
                continue
            pts.append((float(p[0]), float(p[1]), float(p[2])))
            labels.append(int(p[6]))
    pts = np.array(pts, np.float64)
    labels = np.array(labels, np.int64)
    id2name = {}
    for line in open(legend_path):
        parts = line.split()
        if len(parts) >= 2:
            try:
                id2name[int(parts[0].lstrip('#'))] = parts[1]
            except ValueError:
                pass
    objs = []
    for iid in np.unique(labels):
        if iid < 0:
            continue
        name = id2name.get(int(iid), "")
        nyu = PROMPT2NYU.get(name, -1)
        if nyu not in FOREGROUND_NYU:
            continue
        p = pts[labels == iid]
        if len(p) < min_pts:
            continue
        objs.append({"id": int(iid), "nyu": nyu, "name": name, "points": p})
    objs.sort(key=lambda o: -len(o["points"]))
    return objs


def v3dsam_masks_frame(objs, w2c, K, W, H, min_px=40, dilate=3):
    out = []
    for o in objs:
        m = convex_hull_mask(o["points"], w2c, K, W, H, min_px=min_px)
        if m is not None:
            m = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8),
                           iterations=dilate).astype(bool)
        out.append((o["id"], o["nyu"], m))
    return out


# ---------------------------------------------------------------- OVO-SLAM（几何，官方 vanilla mapper 输出）
def load_ovo(scene, exp, min_pts=30):
    """读 OVO 官方输出 ovo_map.ckpt（xyz + obj_ids，世界坐标）+ estimated_c2w.npy。
    obj_ids 为纯几何实例 ID（无语义类别，nyu=-1）；身份=几何。"""
    import torch
    base = f"/root/autodl-tmp/OVO/data/output/ScanNet/{exp}/{scene}"
    ckpt = torch.load(f"{base}/ovo_map.ckpt", map_location="cpu", weights_only=False)
    mp = ckpt["map_params"]
    xyz = np.asarray(mp["xyz"], np.float64)
    obj_ids = np.asarray(mp["obj_ids"]).ravel()
    c2w = torch.load(f"{base}/estimated_c2w.npy", map_location="cpu", weights_only=False)
    objs = []
    for oid in np.unique(obj_ids):
        if int(oid) < 0:
            continue
        p = xyz[obj_ids == oid]
        if len(p) < min_pts:
            continue
        objs.append({"id": int(oid), "nyu": -1, "points": p})
    objs.sort(key=lambda o: -len(o["points"]))
    return objs, c2w


def load_gt_poses(scene, kf_indices):
    """ScanNet GT 位姿（原始 c2w，DAT/pose/{fi}.txt）。返回 {fi: 4x4 c2w}。"""
    poses = {}
    for fi in kf_indices:
        T = np.loadtxt(f"{DAT}/{scene}/pose/{fi}.txt")
        if np.isfinite(T).all():
            poses[fi] = np.asarray(T, np.float64)
    return poses


def ovo_masks_frame(objs, c2w, fi, K, W, H, min_px=40, dilate=3):
    """OVO 3D 实例点 + 自身 GT c2w 位姿 -> 相机平面凸包掩膜（与 V3DSAM 几何法同协议）。"""
    # ScanNet GT pose 存在 -inf 帧（如 scene0059_00 的 1460–1479），OVO track_camera 已跳过、
    # c2w 里无该帧，故此类关键帧 OVO 无法投影，按"无预测"返回空（由调用方计入 0 匹配）。
    if fi not in c2w:
        return []
    c2w_f = np.asarray(c2w[fi], np.float64)
    if not np.isfinite(c2w_f).all():
        return []
    w2c = np.linalg.inv(c2w_f)
    out = []
    for o in objs:
        m = convex_hull_mask(o["points"], w2c, K, W, H, min_px=min_px)
        if m is not None:
            m = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8),
                           iterations=dilate).astype(bool)
        out.append((o["id"], o["nyu"], m))
    return out


# ---------------------------------------------------------------- OV baseline（逐帧无身份）
def load_ov_state(sem, W, H):
    prompts = [str(p) for p in sem["prompts"].tolist()]
    labels = sem["labels"]  # (K, H, W) int16
    fg_cats = [c for c, p in enumerate(prompts) if PROMPT2NYU.get(p, -1) in FOREGROUND_NYU]
    return {"labels": labels, "fg_cats": fg_cats, "prompts": prompts, "W": W, "H": H}


def ov_masks_frame(state, p, min_px=50):
    """逐帧连通域，帧局部 ID = 类别*10000 + 连通域序号（跨帧不链接）。"""
    labels = state["labels"][p]
    out = []
    for c in state["fg_cats"]:
        mask_c = (labels == c)
        if int(mask_c.sum()) < min_px:
            continue
        n, comp = cv2.connectedComponents(mask_c.astype(np.uint8))
        for ci in range(1, n):
            m = comp == ci
            if int(m.sum()) < min_px:
                continue
            out.append((c * 10000 + ci, PROMPT2NYU[state["prompts"][c]], m.astype(bool)))
    return out


# ---------------------------------------------------------------- CLIP Association（逐帧检测 + 外观关联）
def _clip_features_batch(model, proc, img, masks):
    """对一帧的多个 mask 批量提取 CLIP 图像特征（L2 归一化）。返回 list[feat or None]。"""
    import torch
    crops, valid = [], []
    for m in masks:
        ys, xs = np.nonzero(m)
        if len(ys) < 16:
            crops.append(None); valid.append(False); continue
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        crop = img[y0:y1 + 1, x0:x1 + 1].copy()
        crop[~m[y0:y1 + 1, x0:x1 + 1]] = 127
        crops.append(Image.fromarray(crop)); valid.append(True)
    if not any(valid):
        return [None] * len(masks)
    inp = proc(images=[c for c, v in zip(crops, valid) if v], return_tensors="pt")
    inp = {k: v.to(model.device) for k, v in inp.items()}
    with torch.no_grad():
        out = model.get_image_features(**inp)
    if hasattr(out, "pooler_output"):
        out = out.pooler_output
    elif hasattr(out, "image_embeds"):
        out = out.image_embeds
    out = out / out.norm(dim=-1, keepdim=True)
    feats = out.cpu().numpy()
    result, fi = [], 0
    for v in valid:
        if v:
            result.append(feats[fi]); fi += 1
        else:
            result.append(None)
    return result


def run_clip_association(sem, stage3, kf_indices, args):
    """逐帧连通域（同 OV）+ CLIP 外观特征顺序贪心关联。返回 {fi: [(id, nyu, mask)]}。"""
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                           local_files_only=True).to("cuda").eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",
                                              local_files_only=True)

    prompts = [str(p) for p in sem["prompts"].tolist()]
    labels = sem["labels"]
    fg_cats = [c for c, p in enumerate(prompts) if PROMPT2NYU.get(p, -1) in FOREGROUND_NYU]

    tracks = []          # [{feat: np.ndarray or None}]，id = list index
    masks_by_fi = {}
    thr = args.clip_thr

    for p, fi in enumerate(kf_indices):
        img = cv2.cvtColor(cv2.imread(str(stage3[f"img_path_{fi}"])), cv2.COLOR_BGR2RGB)
        lab = labels[p].astype(np.int32)
        comps = []
        for c in fg_cats:
            mc = (lab == c)
            if int(mc.sum()) < args.min_px:
                continue
            n, comp = cv2.connectedComponents(mc.astype(np.uint8))
            for ci in range(1, n):
                m = comp == ci
                if int(m.sum()) < args.min_px:
                    continue
                comps.append((m.astype(bool), PROMPT2NYU.get(prompts[c], -1)))
        feats = _clip_features_batch(clip_model, clip_proc, img, [m for m, _ in comps])
        frame_objs = []
        for (m, nyu), feat in zip(comps, feats):
            if feat is None:
                oid = len(tracks)
                tracks.append({"feat": None})
            else:
                best_id, best_sim = -1, thr
                for idx, t in enumerate(tracks):
                    if t["feat"] is None:
                        continue
                    s = float(feat @ t["feat"])
                    if s > best_sim:
                        best_sim, best_id = s, idx
                if best_id < 0:
                    oid = len(tracks)
                    tracks.append({"feat": feat})
                else:
                    oid = best_id  # track 特征保持首帧锚定，不滑动平均（避免漂移污染）
            frame_objs.append((oid, nyu, m))
        masks_by_fi[fi] = frame_objs
        if (p + 1) % 100 == 0:
            print(f"  [clip] {p+1}/{len(kf_indices)} frames, {len(tracks)} tracks")
    return masks_by_fi


# ---------------------------------------------------------------- SAM3 Tracking（前馈时序）
def _parse_outputs(outputs):
    outputs = outputs or {}
    obj_ids = outputs.get("out_obj_ids")
    masks = outputs.get("out_binary_masks")
    objs = []
    if obj_ids is not None and masks is not None:
        for i, oid in enumerate(obj_ids):
            m = masks[i]
            if m is not None and np.any(m):
                objs.append({"obj_id": int(oid), "mask": m})
    return objs


def run_sam3_tracking(sem, stage3, kf_indices, args, workdir):
    """SAM3 视频跟踪前向传播。每个前景 prompt 一个 session，返回 {fi: [(id, nyu, mask)]}。"""
    import gc
    import torch
    sys.path.insert(0, "/root/autodl-tmp/v3dsam")
    sys.path.insert(0, SAM3_DIR)
    sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/vggt")
    sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/perception_models")
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    prompts = [str(p) for p in sem["prompts"].tolist()]
    fg_prompts = [p for p in prompts if PROMPT2NYU.get(p, -1) in FOREGROUND_NYU]

    # 帧目录：符号链接到已有 color JPEG（零磁盘、瞬时）
    os.makedirs(workdir, exist_ok=True)
    for p, fi in enumerate(kf_indices):
        dst = f"{workdir}/{p}.jpg"
        if not os.path.exists(dst):
            try:
                os.symlink(str(stage3[f"img_path_{fi}"]), dst)
            except OSError:
                cv2.imwrite(dst, cv2.imread(str(stage3[f"img_path_{fi}"])))

    def _build():
        pr = build_sam3_multiplex_video_predictor(checkpoint_path=SAM3_VIDEO_CKPT,
                                                  use_fa3=False)
        try:
            pr.model.hotstart_delay = 0
            pr.model.masklet_confirmation_consecutive_det_thresh = 1
        except Exception as e:
            print(f"[sam3-track] override hotstart failed: {e}")
        return pr

    # SAM3 close_session 释放不彻底，长序列多 prompt 会累积 ~0.4GiB/prompt 致 OOM；
    # 周期性重建 predictor 以彻底释放 GPU 显存。
    REBUILD_EVERY = 5
    predictor = _build()

    masks_by_fi = {fi: [] for fi in kf_indices}
    for pi, p in enumerate(fg_prompts):
        if pi > 0 and pi % REBUILD_EVERY == 0:
            del predictor
            gc.collect()
            torch.cuda.empty_cache()
            predictor = _build()
            print(f"[sam3-track] rebuilt predictor before prompt {pi+1}")
        nyu = PROMPT2NYU.get(p, -1)
        resp = predictor.handle_request(request=dict(
            type="start_session", resource_path=workdir,
            offload_video_to_cpu=True, offload_state_to_cpu=True))
        sid = resp["session_id"]
        resp0 = predictor.handle_request(request=dict(
            type="add_prompt", session_id=sid, frame_index=0, text=p))
        for o in _parse_outputs(resp0.get("outputs")):
            masks_by_fi[kf_indices[0]].append((pi * 64 + o["obj_id"], nyu, o["mask"]))
        try:
            stream = predictor.handle_stream_request(request=dict(
                type="propagate_in_video", session_id=sid,
                propagation_direction="forward"))
            for frame_res in stream:
                fidx = frame_res.get("frame_index", -1)
                if fidx <= 0 or fidx >= len(kf_indices):
                    continue
                for o in _parse_outputs(frame_res.get("outputs")):
                    masks_by_fi[kf_indices[fidx]].append(
                        (pi * 64 + o["obj_id"], nyu, o["mask"]))
        except Exception as e:
            print(f"[sam3-track] propagate '{p}' failed: {e}")
        try:
            predictor.handle_request(request=dict(type="close_session", session_id=sid))
        except Exception as e:
            print(f"[sam3-track] close_session failed: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[sam3-track] prompt {pi+1}/{len(fg_prompts)} '{p}' done")
    return masks_by_fi


# ---------------------------------------------------------------- 两阶段缓存
def _pred_cache_path(cache_dir, scene, method):
    return f"{cache_dir}/pred_{scene}__{method}.pkl.gz"


def _pack_masks(masks_by_fi):
    """bool 掩膜 -> bit-packed，8× 缩小落盘体积（避免 12G 磁盘被全分辨率掩膜撑爆）。"""
    packed = {}
    for fi, objs in masks_by_fi.items():
        packed[fi] = [(iid, nyu, None if m is None else (np.packbits(m.ravel()), m.shape))
                      for (iid, nyu, m) in objs]
    return packed


def _unpack_masks(packed):
    masks = {}
    for fi, objs in packed.items():
        masks[fi] = []
        for iid, nyu, item in objs:
            if item is None:
                masks[fi].append((iid, nyu, None))
            else:
                p, shape = item
                n = shape[0] * shape[1]
                masks[fi].append((iid, nyu,
                                  np.unpackbits(p)[:n].reshape(shape).astype(bool)))
    return masks


def _load_or_generate_masks(cache_dir, scene, method, force, generate_fn):
    """Stage A：方法逐帧掩膜缓存。命中直接读盘，未命中跑模型推理一次并落盘。"""
    path = _pred_cache_path(cache_dir, scene, method)
    if (not force) and os.path.exists(path):
        print(f"[cache] load pred {scene}/{method}")
        with gzip.open(path, "rb") as f:
            return _unpack_masks(pickle.load(f))
    masks = generate_fn()
    os.makedirs(cache_dir, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(_pack_masks(masks), f, protocol=4)
    print(f"[cache] save pred {scene}/{method}")
    return masks


def _eval_cache_path(cache_dir, scene, method):
    return f"{cache_dir}/eval_{scene}__{method}_v2.pkl"


# ---------------------------------------------------------------- 方法注册表
def build_method_state(name, scene, sem, stage3, args):
    img_W, img_H = int(stage3["img_W"]), int(stage3["img_H"])
    kf_indices = [int(x) for x in sem["kf_indices"].tolist()]
    if args.max_frames > 0:
        kf_indices = kf_indices[:args.max_frames]
    if name == "v3dsam":
        return {"kind": "v3dsam", "objs": load_v3dsam(scene), "W": img_W, "H": img_H}
    if name == "v3dsam_noclass":
        return {"kind": "v3dsam", "objs": load_v3dsam(scene, subdir="semantic_cloud_noclass"),
                "W": img_W, "H": img_H}
    if name == "v3dsam_gt":
        return {"kind": "v3dsam_gt", "objs": load_v3dsam(scene, subdir="semantic_cloud_gt"),
                "gt_poses": load_gt_poses(scene, kf_indices), "W": img_W, "H": img_H}
    if name == "ov_baseline":
        return {"kind": "ov", **load_ov_state(sem, img_W, img_H)}
    if name == "clip_association":
        masks = _load_or_generate_masks(
            args.cache_dir, scene, name, args.force,
            lambda: run_clip_association(sem, stage3, kf_indices, args))
        return {"kind": "precomputed", "masks": masks, "W": img_W, "H": img_H}
    if name == "sam3_tracking":
        workdir = f"/root/autodl-tmp/sam3_track_frames/{scene}"
        masks = _load_or_generate_masks(
            args.cache_dir, scene, name, args.force,
            lambda: run_sam3_tracking(sem, stage3, kf_indices, args, workdir))
        return {"kind": "precomputed", "masks": masks, "W": img_W, "H": img_H}
    if name == "ovo":
        objs, c2w = load_ovo(scene, args.ovo_exp)
        return {"kind": "ovo", "objs": objs, "c2w": c2w, "W": img_W, "H": img_H}
    raise NotImplementedError(name)


def method_masks_frame(name, state, fi, stage3):
    W, H = state["W"], state["H"]
    if state["kind"] == "precomputed":
        return state["masks"].get(fi, [])
    if name in ("v3dsam", "v3dsam_noclass"):
        w2c = stage3[f"w2c_{fi}"].astype(np.float64)
        K = stage3[f"K_{fi}"].astype(np.float64)
        return v3dsam_masks_frame(state["objs"], w2c, K, W, H)
    if name == "v3dsam_gt":
        if fi not in state["gt_poses"]:
            return []
        w2c = np.linalg.inv(state["gt_poses"][fi]).astype(np.float64)
        K = stage3[f"K_{fi}"].astype(np.float64)
        return v3dsam_masks_frame(state["objs"], w2c, K, W, H)
    if name == "ov_baseline":
        return ov_masks_frame(state, state["_p_idx"])
    if name == "ovo":
        K = stage3[f"K_{fi}"].astype(np.float64)
        return ovo_masks_frame(state["objs"], state["c2w"], fi, K, W, H)
    raise NotImplementedError(name)


# ---------------------------------------------------------------- 单场景评估
def eval_method(scene, name, stage3, sem, kf_indices, img_W, img_H, seq_len,
                fg_insts, gt_visible, M, args):
    """单方法评估：Stage B 缓存 tracks+stats，命中则直接重算指标（模型不重跑）。"""
    epath = _eval_cache_path(args.cache_dir, scene, name)
    if (not args.force) and os.path.exists(epath):
        with open(epath, "rb") as f:
            d = pickle.load(f)
        tracks, stats = d["tracks"], d["stats"]
        print(f"[cache] load eval {scene}/{name}")
    else:
        state = build_method_state(name, scene, sem, stage3, args)
        tracks = {iid: {} for iid in gt_visible}
        stats = {"n_vis": 0, "n_match": 0, "n_cls_agree": 0,
                 "n_pred": 0, "n_fp": 0, "tube_iou": {}, "assoc": {}, "pred_ids": set()}
        for p, fi in enumerate(kf_indices):
            inst = cv2.imread(f"{INST}/{scene}/{fi}.png", cv2.IMREAD_UNCHANGED)
            if inst is None:
                continue
            gt_masks = []
            for iid in np.unique(inst):
                if iid == 0 or int(iid) not in fg_insts:
                    continue
                m = (inst == iid)
                if int(m.sum()) < args.min_px:
                    continue
                gt_masks.append((int(iid), m.astype(bool)))
            gt_nyu = {iid: fg_insts[iid] for iid, _ in gt_masks}
            n_vis_frame = len(gt_masks)

            if name == "ov_baseline":
                state["_p_idx"] = p
            ams = method_masks_frame(name, state, fi, stage3)
            inst_masks = [(iid, m) for (iid, nyu, m) in ams
                          if m is not None and m.shape == (img_H, img_W)]
            inst_nyu = {iid: nyu for (iid, nyu, m) in ams
                        if m is not None and m.shape == (img_H, img_W)}
            res, inter, gt_area, inst_area = match_frame_full(
                gt_masks, inst_masks, inst_nyu, gt_nyu,
                cov_thr=args.cov_thr, use_class=False)
            for iid in gt_visible:
                tracks[iid][fi] = res.get(iid)
            matched_pred = {v for v in res.values() if v is not None}
            stats["n_vis"] += n_vis_frame
            stats["n_match"] += len(matched_pred)
            stats["n_cls_agree"] += sum(
                1 for gid, mid in res.items()
                if mid is not None and inst_nyu.get(mid, -1) == gt_nyu.get(gid, -1))

            # ---- 官方跟踪/全景指标派生量（MOTA/IDF1/4D-PQ）----
            gt_id_list = [gid for gid, _ in gt_masks]
            pred_id_list = [iid for iid, _ in inst_masks]
            stats["n_pred"] += len(inst_masks)
            stats["n_fp"] += len(inst_masks) - len(matched_pred)
            stats["pred_ids"].update(pred_id_list)

            gt_idx = {gid: i for i, gid in enumerate(gt_id_list)}
            pred_idx = {iid: j for j, iid in enumerate(pred_id_list)}
            denom_iou = gt_area[:, None] + inst_area[None, :] - inter
            with np.errstate(divide="ignore", invalid="ignore"):
                iou = np.where(denom_iou > 0, inter / np.maximum(denom_iou, 1), 0.0)

            # 4D-PQ：matched 对累加 tube-IoU（共可见帧均值）
            for gid, mid in res.items():
                if mid is None:
                    continue
                v = iou[gt_idx[gid], pred_idx[mid]]
                key = (gid, mid)
                s0, c0 = stats["tube_iou"].get(key, (0.0, 0))
                stats["tube_iou"][key] = (s0 + v, c0 + 1)

            # IDF1：全对 IoU>0.5 累加 association 帧数
            for i, j in np.argwhere(iou > 0.5):
                key = (gt_id_list[i], pred_id_list[j])
                stats["assoc"][key] = stats["assoc"].get(key, 0) + 1

            if (p + 1) % 100 == 0:
                print(f"  [{scene}/{name}] {p+1}/{len(kf_indices)} frames")
        os.makedirs(args.cache_dir, exist_ok=True)
        with open(epath, "wb") as f:
            pickle.dump({"tracks": tracks, "stats": stats}, f, protocol=4)
        print(f"[cache] save eval {scene}/{name}")

    m = compute_metrics(gt_visible, tracks, revisit_gap_frac=args.revisit_frac,
                        seq_len=seq_len)
    s = stats

    # ---- 官方跟踪/全景指标（MOTA / IDF1 / 4D-PQ）----
    mota = compute_mota(s["n_vis"], s["n_fp"], m["FM"], m["IDS"])
    idf1, idtp = compute_idf1(s["assoc"], s["n_vis"], s["n_pred"])
    pq, pq_tp, pq_fp, pq_fn, pq_sum_iou = compute_4d_pq(
        s["tube_iou"], sorted(gt_visible.keys()), sorted(s["pred_ids"]))

    row = {
        "scene": scene, "method": name,
        "IDC": m["IDC"],
        "total_IDS": m["IDS"],
        "mean_IDS_per_gt_obj": m["IDS"] / M if M else 0.0,
        "total_fragmentation": m["FM"],
        "mean_frag_per_gt_obj": m["FM"] / M if M else 0.0,
        "ReID_recall": m["ReID_R"],
        "GT_instance_coverage": s["n_match"] / max(1, s["n_vis"]),
        "class_consistency": s["n_cls_agree"] / max(1, s["n_match"]),
        "MOTA": mota,
        "IDF1": idf1,
        "4D_PQ": pq,
        # 聚合精确分子分母（不写进 CSV，仅用于 overall）
        "_IDC_num": m["IDC_num"], "_IDC_den": m["IDC_den"],
        "_ReID_num": m["ReID_num"], "_ReID_den": m["ReID_den"],
        "_n_gt_objects": M,
        "_n_vis_pairs": s["n_vis"],
        "_n_match_pairs": s["n_match"],
        "_n_cls_agree": s["n_cls_agree"],
        "_n_fp": s["n_fp"], "_n_fm": m["FM"], "_n_ids": m["IDS"],
        "_idtp": idtp, "_n_pred": s["n_pred"],
        "_pq_tp": pq_tp, "_pq_fp": pq_fp, "_pq_fn": pq_fn, "_pq_sum_iou": pq_sum_iou,
    }
    print(f"  [{scene}/{name}] IDC={row['IDC']:.3f} IDS={row['total_IDS']} "
          f"FM={row['total_fragmentation']} ReID-R={row['ReID_recall']:.3f} "
          f"cov={row['GT_instance_coverage']:.3f} cls={row['class_consistency']:.3f} "
          f"MOTA={mota:.3f} IDF1={idf1:.3f} 4D-PQ={pq:.3f} (gt_obj={M})")
    return row


def eval_scene(scene, args):
    base = f"{ABL}/{scene}"
    stage3 = np.load(f"{base}/_stage3_input_{args.stage3_suffix}.npz", allow_pickle=True)
    sem = np.load(f"{base}/kf_semantic.npz", allow_pickle=True)
    kf_indices = [int(x) for x in sem["kf_indices"].tolist()]
    if args.max_frames > 0:
        kf_indices = kf_indices[:args.max_frames]
    img_W, img_H = int(stage3["img_W"]), int(stage3["img_H"])
    seq_len = max(kf_indices) - min(kf_indices) + 1

    inst_class, inst_frames = load_gt_instances(scene, kf_indices, min_px=args.min_px)
    fg_insts = {iid: c for iid, c in inst_class.items() if c in FOREGROUND_NYU}
    M = len(fg_insts)
    gt_visible = {iid: sorted(frames) for iid, frames in inst_frames.items()
                  if iid in fg_insts}
    print(f"\n[{scene}] GT 前景实例={M}, kf={len(kf_indices)}, seq_len={seq_len}")

    return [eval_method(scene, name, stage3, sem, kf_indices, img_W, img_H,
                        seq_len, fg_insts, gt_visible, M, args)
            for name in args.methods]


# ---------------------------------------------------------------- 汇总 + 写 CSV
METRIC_KEYS = ["IDC", "total_IDS", "mean_IDS_per_gt_obj", "total_fragmentation",
               "mean_frag_per_gt_obj", "ReID_recall", "GT_instance_coverage",
               "class_consistency", "MOTA", "IDF1", "4D_PQ"]
INT_KEYS = {"total_IDS", "total_fragmentation"}


def _fmt(k, v):
    return str(int(round(v))) if k in INT_KEYS else f"{v:.4f}"


def _agg_metrics(a):
    """由聚合的分子分母重算各指标（MOTA/IDF1/4D-PQ 须全局聚合，不能简单平均）。"""
    idc = a["IDC_num"] / a["IDC_den"] if a["IDC_den"] else 0.0
    ids = a["IDS"]
    mean_ids = ids / a["n_gt"] if a["n_gt"] else 0.0
    fm = a["FM"]
    mean_fm = fm / a["n_gt"] if a["n_gt"] else 0.0
    reid = a["ReID_num"] / a["ReID_den"] if a["ReID_den"] else 0.0
    cov = a["n_match"] / a["n_vis"] if a["n_vis"] else 0.0
    cls = a["n_cls"] / a["n_match"] if a["n_match"] else 0.0
    mota = compute_mota(a["n_vis"], a["n_fp"], a["FM"], a["IDS"])
    idf1 = (2.0 * a["idtp"] / (a["n_vis"] + a["n_pred"])
            if (a["n_vis"] + a["n_pred"]) else 0.0)
    pq_denom = a["pq_tp"] + 0.5 * a["pq_fp"] + 0.5 * a["pq_fn"]
    pq = a["pq_sum_iou"] / pq_denom if pq_denom else 0.0
    return [idc, ids, mean_ids, fm, mean_fm, reid, cov, cls, mota, idf1, pq]


def write_csvs(all_rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    header = ["scene", "method"] + METRIC_KEYS
    with open(f"{out_dir}/persistent_metrics_per_scene.csv", "w") as f:
        f.write(",".join(header) + "\n")
        for r in all_rows:
            f.write(",".join([r["scene"], r["method"]] +
                             [_fmt(k, r[k]) for k in METRIC_KEYS]) + "\n")

    methods = sorted({r["method"] for r in all_rows})
    agg = {}
    for r in all_rows:
        m = r["method"]
        a = agg.setdefault(m, {"IDC_num": 0, "IDC_den": 0, "ReID_num": 0, "ReID_den": 0,
                               "IDS": 0, "FM": 0, "n_gt": 0, "n_vis": 0,
                               "n_match": 0, "n_cls": 0,
                               "n_fp": 0, "idtp": 0, "n_pred": 0,
                               "pq_tp": 0, "pq_fp": 0, "pq_fn": 0, "pq_sum_iou": 0.0})
        a["IDC_num"] += r["_IDC_num"]; a["IDC_den"] += r["_IDC_den"]
        a["ReID_num"] += r["_ReID_num"]; a["ReID_den"] += r["_ReID_den"]
        a["IDS"] += r["total_IDS"]; a["FM"] += r["total_fragmentation"]
        a["n_gt"] += r["_n_gt_objects"]
        a["n_vis"] += r["_n_vis_pairs"]
        a["n_match"] += r["_n_match_pairs"]
        a["n_cls"] += r["_n_cls_agree"]
        a["n_fp"] += r["_n_fp"]
        a["idtp"] += r["_idtp"]
        a["n_pred"] += r["_n_pred"]
        a["pq_tp"] += r["_pq_tp"]; a["pq_fp"] += r["_pq_fp"]; a["pq_fn"] += r["_pq_fn"]
        a["pq_sum_iou"] += r["_pq_sum_iou"]

    with open(f"{out_dir}/persistent_metrics_overall.csv", "w") as f:
        f.write(",".join(["method"] + METRIC_KEYS) + "\n")
        for m in methods:
            a = agg[m]
            vals = _agg_metrics(a)
            f.write(",".join([m] + [_fmt(k, v) for k, v in zip(METRIC_KEYS, vals)]) + "\n")

    # 论文对比表（markdown，overall 聚合）
    md = ["# Table 4-3  持久实例身份（4 场景聚合，GT = ScanNet instance-filt）\n",
          "| 方法 | IDC ↑ | IDS ↓ | FM ↓ | ReID-R ↑ | Cov ↑ | MOTA ↑ | IDF1 ↑ | 4D-PQ ↑ |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in methods:
        a = agg[m]
        vals = _agg_metrics(a)
        idc, ids, _, fm, _, reid, cov, _, mota, idf1, pq = vals
        md.append(f"| {m} | {idc:.3f} | {ids} | {fm} | {reid:.3f} | {cov:.3f} "
                  f"| {mota:.3f} | {idf1:.3f} | {pq:.3f} |")
    md.append("\n> 口径：GT=ScanNet 官方 2D instance-filt；匹配=逐帧几何覆盖(Hungarian, "
              "cov≥0.30, use_class=False)；MOTA=1-(FP+FN+IDSW)/GT；IDF1=轨迹级匈牙利；"
              "4D-PQ=类无关 thing tube 贪心匹配(IoU>0.5)，身份=几何。OVO 无类标签。")
    with open(f"{out_dir}/comparison_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("\n[overall]")
    for m in methods:
        a = agg[m]
        vals = _agg_metrics(a)
        s = "  " + m + ": "
        s += "  ".join(f"{k}={_fmt(k, v)}" for k, v in zip(METRIC_KEYS, vals))
        print(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="", help="空=全部 4 场景")
    ap.add_argument("--methods", nargs="+",
                    default=["v3dsam", "ov_baseline", "sam3_tracking", "clip_association"])
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--cov_thr", type=float, default=0.30)
    ap.add_argument("--revisit_frac", type=float, default=0.25)
    ap.add_argument("--min_px", type=int, default=50)
    ap.add_argument("--clip_thr", type=float, default=0.85,
                    help="CLIP 关联余弦相似度阈值")
    ap.add_argument("--cache_dir", default="/root/autodl-tmp/persistent_cache",
                    help="两阶段缓存目录（pred 掩膜 + eval tracks）")
    ap.add_argument("--ovo_exp", default="ovo_smoke",
                    help="OVO 实验名（output/ScanNet/{ovo_exp}/{scene}/ovo_map.ckpt）")
    ap.add_argument("--stage3_suffix", default="after",
                    help="stage3 npz 后缀：after=含回环校验 / before=无回环校验（消融）")
    ap.add_argument("--force", action="store_true",
                    help="忽略缓存，重跑模型推理与匹配")
    ap.add_argument("--out", default="/root/autodl-tmp/persistent_metrics")
    args = ap.parse_args()

    scenes = [args.scene] if args.scene else SCENES
    all_rows = []
    for scene in scenes:
        try:
            all_rows.extend(eval_scene(scene, args))
            if all_rows:
                write_csvs(all_rows, args.out)  # 每场景增量落盘，长任务崩溃可恢复
        except FileNotFoundError as e:
            print(f"[eval] SKIP {scene} (缺数据): {e}")

    if not all_rows:
        print("[eval] 无场景可评估"); return
    print("\n[eval] DONE")


if __name__ == "__main__":
    main()
