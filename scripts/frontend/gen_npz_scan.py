#!/usr/bin/env python3
"""V3DSAM ScanNet 语义回环校验 + npz 导出（单次 Stage1）。

合并 gen_npz_semantic.py（npz 导出）+ semantic_3d_check.py（逐像素 3D 一致性）
+ semantic_object_check.py（物体级一致性），Stage1 只跑一次：

  1. Stage1（VGGT-SLAM）检测回环 → solver（BEFORE 态）
  2. 导出 BEFORE npz + 位姿
  3. 对每个回环：SAM3 掩码（存 loop_figures/loop{k}_masks.npz）
     → 逐像素 3D 语义一致性 cons_avg（阈值 0.10）
     → 物体级 match_ratio（阈值 0.25，正交加固）
  4. 判别假回环：cons_avg < 0.10 → 拒（cons 不可用时退回物体级 match_ratio < 0.25）
  5. remove_loops + 重优化 → 导出 AFTER npz + 位姿 + loop_log.json

判别阈值来自 TUM 全 9 序列 15 回环验证：逐像素假≤0.078/真≥0.152，物体级假≤0.158/真≥0.371。
不修改用户 v3dsam 源码，仅 import 其模块。
"""
import os, sys, re, glob, time, argparse, json, gc
import numpy as np, cv2, torch
import torch.nn.functional as F
from PIL import Image
import gtsam
from gtsam.symbol_shorthand import X

sys.path.insert(0, "/root/autodl-tmp/v3dsam")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/vggt")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/sam3")
os.environ['XFORMERS_DISABLED'] = '1'

from vggt_slam_official.solver import Solver
import vggt_slam_official.slam_utils as slam_utils
from vggt.models.vggt import VGGT
from vggt_slam.slam_utils import decompose_camera
from vggt_slam.semantic_loop_verify import cluster_objects_by_class

ALL_PROMPTS = [
    "wall","floor","ceiling","door","window","stairs","railing",
    "bed","cabinet","bookshelf","shelf","counter","desk","table","sofa","couch",
    "sink","toilet","bathtub","mirror","lamp","light",
    "television","monitor","laptop","keyboard","mouse","phone","refrigerator","oven","microwave",
    "chair","plant","vase","picture","painting","curtain","pillow","clock",
    "book","bottle","cup","bowl","plate","bag","backpack","box","bin","basket","person",
]
N_cats = len(ALL_PROMPTS)


# ---------------------------------------------------------------- SAM3
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


# ---------------------------------------------------------------- graph / 导出
def remove_loops(solver, lc_ids):
    false_nodes = set()
    for lc_num in lc_ids:
        false_nodes.add(X(lc_num))
        false_nodes.add(X(lc_num + 1))
    new_graph = gtsam.NonlinearFactorGraph()
    removed = 0
    for i in range(solver.graph.graph.size()):
        f = solver.graph.graph.at(i)
        if set(f.keys()) & false_nodes:
            removed += 1
        else:
            new_graph.add(f)
    solver.graph.graph = new_graph
    solver.graph.optimize()
    return removed


def build_pointcloud(solver, max_points):
    pts_list, rgb_list = [], []
    for sm in solver.map.ordered_submaps_by_key():
        if sm.get_lc_status():
            continue
        pts_list.append(sm.get_points_in_world_frame(solver.graph))
        rgb_list.append(sm.get_points_colors())
    pts = np.concatenate(pts_list).astype(np.float32)
    rgb = np.concatenate(rgb_list).astype(np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    n = min(len(pts), len(rgb))
    target = min(n, max_points)
    idx = np.random.RandomState(42).choice(n, target, replace=False)
    return pts[idx].astype(np.float32), rgb[idx].astype(np.float32)


def export_npz(solver, names, vggt_kf_names, pts_gs, rgb_gs, img_H_full, img_W_full,
               vggt_H, vggt_W, out_path):
    scale_x = img_W_full / vggt_W
    scale_y = img_H_full / vggt_H
    _all_c2w = solver.map.get_all_cam_matricies(solver.graph, False)
    _all_proj = solver.map.get_all_cam_matricies(solver.graph, True)
    _submap_lookup = {}
    _counter = 0
    for sm in solver.map.ordered_submaps_by_key():
        if sm.get_lc_status():
            continue
        for _fi in range(len(sm.poses)):
            if _counter >= len(_all_c2w):
                break
            if sm.img_names and _fi < len(sm.img_names):
                _bn = os.path.basename(sm.img_names[_fi])
                _depth = sm.get_frame_pointcloud(_fi)
                _d = _depth[..., 2].astype(np.float32) if (_depth is not None and _depth.ndim == 3) else None
                _submap_lookup[_bn] = (_all_c2w[_counter], _all_proj[_counter], _d)
            _counter += 1
    save = {"img_W": img_W_full, "img_H": img_H_full, "pts_gs": pts_gs, "rgb_gs": rgb_gs}
    _n_exported = 0
    vggt_kf_export_indices = []
    for _orig_path in names:
        _bn = os.path.basename(_orig_path)
        if _bn in _submap_lookup:
            _c2w, _proj, _d = _submap_lookup[_bn]
        else:
            _orig_num = int(re.search(r'\d+', _bn).group())
            _best_dist = float('inf'); _best = None
            for _sbn, _sv in _submap_lookup.items():
                _snum = int(re.search(r'\d+', _sbn).group())
                _dnum = abs(_orig_num - _snum)
                if _dnum < _best_dist:
                    _best_dist = _dnum; _best = _sv
            if _best is None:
                continue
            _c2w, _proj, _d = _best
        _K_vggt, _, _, _ = decompose_camera(_proj)
        _K_full = _K_vggt.copy()
        _K_full[0, 0] *= scale_x; _K_full[0, 2] *= scale_x
        _K_full[1, 1] *= scale_y; _K_full[1, 2] *= scale_y
        _w2c = np.linalg.inv(_c2w)
        save[f"K_{_n_exported}"] = _K_full.astype(np.float32)
        save[f"w2c_{_n_exported}"] = _w2c.astype(np.float32)
        save[f"img_path_{_n_exported}"] = _orig_path
        if _d is not None:
            save[f"depth_{_n_exported}"] = _d.astype(np.float32)
        if _orig_path in vggt_kf_names:
            vggt_kf_export_indices.append(_n_exported)
        _n_exported += 1
    save["vggt_kf_indices"] = np.array(vggt_kf_export_indices, dtype=np.int32)
    np.savez_compressed(out_path, **save)
    return _n_exported


# ---------------------------------------------------------------- 逐像素 3D 一致性
def resize_mask(m, H, W):
    if m.shape[1] == H and m.shape[2] == W:
        return m
    t = torch.from_numpy(m.astype(np.uint8))
    t = F.interpolate(t.unsqueeze(0).float(), (H, W), mode="nearest").squeeze(0)
    return t.numpy().astype(np.uint8)


def frame_data(solver, sid, fi):
    sm = solver.map.get_submap(sid)
    pts = sm.get_frame_pointcloud(fi)          # (H, W, 3) camera frame
    K = sm.proj_mats[fi][:3, :3]
    T = solver.graph.get_homography(sid + fi)  # (4,4) cam2world
    return pts, K, T


def transform_and_project(P_cam_D, T_D, T_Q, K_Q):
    N = P_cam_D.shape[0]
    ones = np.ones((N, 1))
    X_w = (T_D @ np.hstack([P_cam_D, ones]).T).T
    X_w = X_w[:, :3] / X_w[:, 3:4]
    X_Q = (np.linalg.inv(T_Q) @ np.hstack([X_w, ones]).T).T
    X_Q = X_Q[:, :3] / X_Q[:, 3:4]
    uv = (K_Q @ X_Q.T).T
    u = uv[:, 0] / uv[:, 2]
    v = uv[:, 1] / uv[:, 2]
    return X_Q[:, 2], u, v


def consistency(label_D, label_Q, pts_D, T_D, T_Q, K_Q, HQ, WQ, max_pts=60000):
    ys, xs = np.nonzero(label_D >= 0)
    if len(xs) == 0:
        return 0.0, 0
    if len(xs) > max_pts:
        sel = np.random.default_rng(0).choice(len(xs), max_pts, replace=False)
        ys, xs = ys[sel], xs[sel]
    labs = label_D[ys, xs]
    P = pts_D[ys, xs].reshape(-1, 3).astype(np.float64)
    z, u, v = transform_and_project(P, T_D, T_Q, K_Q)
    ok = (z > 0.1) & (u >= 0) & (u < WQ) & (v >= 0) & (v < HQ)
    if ok.sum() < 50:
        return 0.0, int(ok.sum())
    uu = np.clip(u[ok].astype(int), 0, WQ - 1)
    vv = np.clip(v[ok].astype(int), 0, HQ - 1)
    agree = (label_Q[vv, uu] == labs[ok])
    return float(agree.mean()), int(ok.sum())


# ---------------------------------------------------------------- 物体级一致性
def frame_objects(pts, label, T, min_points=80, cluster_eps=0.3, max_pts=120000):
    H, W = label.shape
    ys, xs = np.nonzero(label >= 0)
    if len(xs) == 0:
        return []
    if len(xs) > max_pts:
        sel = np.random.default_rng(0).choice(len(xs), max_pts, replace=False)
        ys, xs = ys[sel], xs[sel]
    labs = label[ys, xs]
    P = pts[ys, xs].reshape(-1, 3).astype(np.float64)
    ok = np.isfinite(P).all(1) & (P[:, 2] > 0.1)
    P, labs = P[ok], labs[ok]
    if len(P) == 0:
        return []
    ones = np.ones((len(P), 1))
    W_ = (T @ np.hstack([P, ones]).T).T
    W_ = W_[:, :3] / W_[:, 3:4]
    return cluster_objects_by_class(W_, labs, min_points=min_points, cluster_eps=cluster_eps)


def project_world_to_cam(c_world, T_Q):
    ones = np.ones((len(c_world), 1))
    X = (np.linalg.inv(T_Q) @ np.hstack([c_world, ones]).T).T
    return X[:, :3] / X[:, 3:4]


def object_consistency(objs_D, objs_Q, T_D, T_Q, match_thr=0.5):
    nD, nQ = len(objs_D), len(objs_Q)
    if nD == 0 or nQ == 0:
        return 0.0, float('inf'), 0, 0
    cD_world = np.array([o['centroid'] for o in objs_D])
    cQ_world = np.array([o['centroid'] for o in objs_Q])
    cD_qcam = project_world_to_cam(cD_world, T_Q)
    cQ_qcam = project_world_to_cam(cQ_world, T_Q)
    clsD = np.array([o['class'] for o in objs_D])
    clsQ = np.array([o['class'] for o in objs_Q])
    matched_dq = 0; resid_dq = []
    for i in range(nD):
        if cD_qcam[i, 2] < 0.1:
            continue
        cand = np.where(clsQ == clsD[i])[0]
        if len(cand) == 0:
            continue
        d = np.linalg.norm(cQ_qcam[cand] - cD_qcam[i], axis=1)
        j = cand[int(np.argmin(d))]
        if d.min() < match_thr:
            matched_dq += 1
            resid_dq.append(float(d.min()))
    matched_qd = 0; resid_qd = []
    cQ_dcam = project_world_to_cam(cQ_world, T_D)
    cD_dcam = project_world_to_cam(cD_world, T_D)
    for j in range(nQ):
        if cQ_dcam[j, 2] < 0.1:
            continue
        cand = np.where(clsD == clsQ[j])[0]
        if len(cand) == 0:
            continue
        d = np.linalg.norm(cD_dcam[cand] - cQ_dcam[j], axis=1)
        if d.min() < match_thr:
            matched_qd += 1
            resid_qd.append(float(d.min()))
    ratio = (matched_dq + matched_qd) / (nD + nQ)
    resid = float(np.median(resid_dq + resid_qd)) if (resid_dq or resid_qd) else float('inf')
    return ratio, resid, matched_dq, matched_qd


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--submap_size", type=int, default=32)
    ap.add_argument("--min_disparity", type=float, default=50)
    ap.add_argument("--lc_thres", type=float, default=0.95)
    ap.add_argument("--max_loops", type=int, default=1)
    ap.add_argument("--max_points", type=int, default=10000000)
    ap.add_argument("--cons_thr", type=float, default=0.10, help="逐像素 3D 一致性阈值")
    ap.add_argument("--object_thr", type=float, default=0.25, help="物体级 match_ratio 阈值")
    ap.add_argument("--match_thr", type=float, default=0.5, help="物体级质心匹配距离(m)")
    ap.add_argument("--reject_lc_ids", type=str, default="")
    ap.add_argument("--skip_sam3", action="store_true")
    args = ap.parse_args()
    force_reject = {int(x) for x in args.reject_lc_ids.split(",") if x.strip() != ""}
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    names = sorted(glob.glob(os.path.join(args.image_dir, "*")))
    names = [f for f in names if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    names = slam_utils.sort_images_by_number(names)
    print(f"[gen_npz_scan] images={len(names)} S={args.submap_size} lc={args.lc_thres}")

    # ── Stage 1 ──
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(
        "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"))
    model.eval(); model = model.to(torch.bfloat16).to(device)
    solver = Solver(init_conf_threshold=25.0, lc_thres=args.lc_thres)
    image_names_subset = []
    vggt_kf_names = set()
    detected_loops = []
    t0 = time.time()
    for name in names:
        img = cv2.imread(name)
        if solver.flow_tracker.compute_disparity(img, args.min_disparity, False):
            image_names_subset.append(name)
            vggt_kf_names.add(name)
        if len(image_names_subset) == args.submap_size + 1 or name == names[-1]:
            predictions = solver.run_predictions(image_names_subset, model, args.max_loops, None, None)
            for loop in predictions.get("detected_loops", []):
                detected_loops.append({
                    "query_sid": int(loop.query_submap_id),
                    "query_fi": int(loop.query_submap_frame),
                    "detected_sid": int(loop.detected_submap_id),
                    "detected_fi": int(loop.detected_submap_frame),
                    "score": float(loop.similarity_score),
                })
            solver.add_points(predictions)
            solver.graph.optimize()
            image_names_subset = image_names_subset[-1:]
    print(f"[gen_npz_scan] Stage1 done {time.time()-t0:.1f}s, {len(detected_loops)} loops, {len(vggt_kf_names)} kf")
    del model
    torch.cuda.empty_cache()

    lc_submaps = [sm for sm in solver.map.ordered_submaps_by_key() if sm.get_lc_status()]
    for k, lp in enumerate(detected_loops):
        lp["lc_num"] = int(lc_submaps[k].get_id()) if k < len(lc_submaps) else None

    pts_gs, rgb_gs = build_pointcloud(solver, args.max_points)

    _first_sm = [sm for sm in solver.map.ordered_submaps_by_key() if not sm.get_lc_status()][0]
    _first_depth = _first_sm.get_frame_pointcloud(0)
    vggt_H, vggt_W = _first_depth.shape[:2] if _first_depth is not None else (392, 518)
    _first_img = Image.open(names[0])
    img_H_full, img_W_full = _first_img.size[1], _first_img.size[0]

    before_npz = os.path.join(args.output_dir, "_stage3_input_before.npz")
    n_exported = export_npz(solver, names, vggt_kf_names, pts_gs, rgb_gs,
                            img_H_full, img_W_full, vggt_H, vggt_W, before_npz)
    before_poses = os.path.join(args.output_dir, "stage1_poses_before.txt")
    solver.map.write_poses_to_file(before_poses, solver.graph, give_camera_mat=True)
    print(f"[gen_npz_scan] BEFORE npz {n_exported} frames -> {before_npz}")

    fig_dir = os.path.join(args.output_dir, "loop_figures")
    os.makedirs(fig_dir, exist_ok=True)

    if not args.skip_sam3 and detected_loops:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        print("[gen_npz_scan] loading SAM3...")
        sam3_model = build_sam3_image_model(
            bpe_path="/root/autodl-tmp/v3dsam/third_party/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            checkpoint_path="/root/autodl-tmp/v3dsam/third_party/sam3/sam3.pt", device="cuda")
        processor = Sam3Processor(sam3_model, confidence_threshold=0.1)
        dummy_geo = sam3_model._get_dummy_prompt(num_prompts=1)

        for k, lp in enumerate(detected_loops):
            try:
                q_sm = solver.map.get_submap(lp["query_sid"])
                d_sm = solver.map.get_submap(lp["detected_sid"])
                q_img = q_sm.get_img_names_at_index(lp["query_fi"])
                d_img = d_sm.get_img_names_at_index(lp["detected_fi"])
                mq = sam3_frame_mask(q_img, sam3_model, processor, dummy_geo)
                md = sam3_frame_mask(d_img, sam3_model, processor, dummy_geo)
                np.savez_compressed(os.path.join(fig_dir, f"loop{k}_masks.npz"),
                                    mask_q=mq, mask_d=md)

                # 类别重叠（参考，预期失效）
                cls_q = set(c for c in range(N_cats) if mq[c].sum() > 500)
                cls_d = set(c for c in range(N_cats) if md[c].sum() > 500)
                overlap = len(cls_q & cls_d) / min(len(cls_q), len(cls_d)) if (cls_q and cls_d) else 0.0

                # 逐像素 3D 一致性
                pts_Q, K_Q, T_Q = frame_data(solver, lp["query_sid"], lp["query_fi"])
                pts_D, K_D, T_D = frame_data(solver, lp["detected_sid"], lp["detected_fi"])
                HQ, WQ = pts_Q.shape[:2]; HD, WD = pts_D.shape[:2]
                mq_r = resize_mask(mq, HQ, WQ); md_r = resize_mask(md, HD, WD)
                label_Q = mq_r.argmax(0).astype(np.int16); label_Q[~mq_r.any(0)] = -1
                label_D = md_r.argmax(0).astype(np.int16); label_D[~md_r.any(0)] = -1
                c_dq, n_dq = consistency(label_D, label_Q, pts_D, T_D, T_Q, K_Q, HQ, WQ)
                c_qd, n_qd = consistency(label_Q, label_D, pts_Q, T_Q, T_D, K_D, HD, WD)
                cons_avg = 0.5 * (c_dq + c_qd)

                # 物体级一致性
                objs_D = frame_objects(pts_D, label_D, T_D)
                objs_Q = frame_objects(pts_Q, label_Q, T_Q)
                ratio, resid, mdq, mqd = object_consistency(objs_D, objs_Q, T_D, T_Q,
                                                            match_thr=args.match_thr)

                lp["overlap"] = overlap
                lp["cons_dq"] = c_dq; lp["cons_qd"] = c_qd; lp["cons_avg"] = cons_avg
                lp["n_dq"] = n_dq; lp["n_qd"] = n_qd
                lp["n_obj_D"] = len(objs_D); lp["n_obj_Q"] = len(objs_Q)
                lp["match_ratio"] = ratio; lp["resid_med"] = resid
                print(f"  [loop lc={lp['lc_num']}] score={lp['score']:.3f} overlap={overlap:.2f} "
                      f"cons_avg={cons_avg:.3f} (n={n_dq}/{n_qd}) match_ratio={ratio:.3f} "
                      f"(objs {len(objs_D)}/{len(objs_Q)})")
            except Exception as e:
                lp["cons_avg"] = None; lp["match_ratio"] = None; lp["overlap"] = None
                print(f"  [loop lc={lp['lc_num']}] SAM3/一致性失败: {e}")
        del sam3_model, processor
        gc.collect()
        torch.cuda.empty_cache()
    else:
        for lp in detected_loops:
            lp["cons_avg"] = None; lp["match_ratio"] = None; lp["overlap"] = None

    # ── 判别假回环 ──
    false_lc_ids = []
    for lp in detected_loops:
        cons = lp.get("cons_avg")
        ratio = lp.get("match_ratio")
        reject = False
        if lp.get("lc_num") in force_reject:
            reject = True
        elif cons is not None:
            reject = cons < args.cons_thr
        elif ratio is not None:
            reject = ratio < args.object_thr
        lp["verdict"] = "reject" if reject else "keep"
        if reject and lp.get("lc_num") is not None:
            false_lc_ids.append(lp["lc_num"])
    print(f"[gen_npz_scan] rejecting {len(false_lc_ids)} loops: lc_ids={false_lc_ids}")

    if false_lc_ids:
        removed = remove_loops(solver, false_lc_ids)
        print(f"[gen_npz_scan] removed {removed} factors, re-optimized")

    after_npz = os.path.join(args.output_dir, "_stage3_input_after.npz")
    n_exported2 = export_npz(solver, names, vggt_kf_names, pts_gs, rgb_gs,
                             img_H_full, img_W_full, vggt_H, vggt_W, after_npz)
    after_poses = os.path.join(args.output_dir, "stage1_poses_after.txt")
    solver.map.write_poses_to_file(after_poses, solver.graph, give_camera_mat=True)
    print(f"[gen_npz_scan] AFTER npz {n_exported2} frames -> {after_npz}")

    with open(os.path.join(args.output_dir, "loop_log.json"), "w") as f:
        json.dump({"loops": detected_loops, "reject_lc_ids": false_lc_ids,
                   "cons_thr": args.cons_thr, "object_thr": args.object_thr}, f, indent=2)
    print(f"[gen_npz_scan] DONE. before={before_npz} after={after_npz}")


if __name__ == "__main__":
    main()
