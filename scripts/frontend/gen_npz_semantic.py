#!/usr/bin/env python3
"""V3DSAM semantic loop filter + npz export (BEFORE/AFTER), for the
semantic→PSNR ablation.  Run Stage1 (VGGT-SLAM), verify loops with SAM3,
reject false loops (remove_loops + re-optimize), and export two npz:
  _stage3_input_before.npz  (raw, with false loops)
  _stage3_input_after.npz   (semantic-corrected)
Both feed v38/v40_stage3_splatslam.py.  Mirrors gen_stage3_npz.py + semantic_filter.py."""
import os, sys, re, glob, time, argparse, json, shutil, gc
import numpy as np, cv2, torch
from PIL import Image
import gtsam
from gtsam.symbol_shorthand import X

sys.path.insert(0, "/root/autodl-tmp/v3dsam")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/vggt")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/sam3")
os.environ['XFORMERS_DISABLED'] = '1'

import torch.nn.functional as F
from vggt_slam_official.solver import Solver
import vggt_slam_official.slam_utils as slam_utils
from vggt.models.vggt import VGGT
from vggt_slam.slam_utils import decompose_camera

ALL_PROMPTS = [
    "wall","floor","ceiling","door","window","stairs","railing",
    "bed","cabinet","bookshelf","shelf","counter","desk","table","sofa","couch",
    "sink","toilet","bathtub","mirror","lamp","light",
    "television","monitor","laptop","keyboard","mouse","phone","refrigerator","oven","microwave",
    "chair","plant","vase","picture","painting","curtain","pillow","clock",
    "book","bottle","cup","bowl","plate","bag","backpack","box","bin","basket","person",
]
N_cats = len(ALL_PROMPTS)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--submap_size", type=int, default=16)
    ap.add_argument("--min_disparity", type=float, default=50)
    ap.add_argument("--lc_thres", type=float, default=0.95)
    ap.add_argument("--max_loops", type=int, default=1)
    ap.add_argument("--min_overlap", type=float, default=0.3)
    ap.add_argument("--max_points", type=int, default=10000000)
    ap.add_argument("--reject_lc_ids", type=str, default="")
    ap.add_argument("--skip_sam3", action="store_true")
    args = ap.parse_args()
    force_reject = {int(x) for x in args.reject_lc_ids.split(",") if x.strip() != ""}
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    names = sorted(glob.glob(os.path.join(args.image_dir, "*")))
    names = [f for f in names if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    names = slam_utils.sort_images_by_number(names)
    print(f"[gen_npz_semantic] images={len(names)} S={args.submap_size} lc={args.lc_thres}")

    # ── Stage 1 ──
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(
        "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"))
    model.eval()
    model = model.to(torch.bfloat16).to(device)
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
    print(f"[gen_npz_semantic] Stage1 done {time.time()-t0:.1f}s, {len(detected_loops)} loops, {len(vggt_kf_names)} kf")
    del model
    torch.cuda.empty_cache()

    lc_submaps = [sm for sm in solver.map.ordered_submaps_by_key() if sm.get_lc_status()]
    for k, lp in enumerate(detected_loops):
        lp["lc_num"] = int(lc_submaps[k].get_id()) if k < len(lc_submaps) else None

    # ── point cloud (pose-independent) ──
    pts_gs, rgb_gs = build_pointcloud(solver, args.max_points)

    # image resolution
    _first_sm = [sm for sm in solver.map.ordered_submaps_by_key() if not sm.get_lc_status()][0]
    _first_depth = _first_sm.get_frame_pointcloud(0)
    vggt_H, vggt_W = _first_depth.shape[:2] if _first_depth is not None else (392, 518)
    _first_img = Image.open(names[0])
    img_H_full, img_W_full = _first_img.size[1], _first_img.size[0]

    # ── export BEFORE (raw) ──
    before_npz = os.path.join(args.output_dir, "_stage3_input_before.npz")
    n_exported = export_npz(solver, names, vggt_kf_names, pts_gs, rgb_gs,
                            img_H_full, img_W_full, vggt_H, vggt_W, before_npz)
    before_poses = os.path.join(args.output_dir, "stage1_poses_before.txt")
    solver.map.write_poses_to_file(before_poses, solver.graph, give_camera_mat=True)
    print(f"[gen_npz_semantic] BEFORE npz {n_exported} frames -> {before_npz}")

    # ── SAM3 semantic verification ──
    if not args.skip_sam3 and detected_loops:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        print("[gen_npz_semantic] loading SAM3...")
        sam3_model = build_sam3_image_model(
            bpe_path="/root/autodl-tmp/v3dsam/third_party/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            checkpoint_path="/root/autodl-tmp/v3dsam/third_party/sam3/sam3.pt", device="cuda")
        processor = Sam3Processor(sam3_model, confidence_threshold=0.1)
        dummy_geo = sam3_model._get_dummy_prompt(num_prompts=1)
        for lp in detected_loops:
            try:
                q_sm = solver.map.get_submap(lp["query_sid"])
                d_sm = solver.map.get_submap(lp["detected_sid"])
                q_img = q_sm.get_img_names_at_index(lp["query_fi"])
                d_img = d_sm.get_img_names_at_index(lp["detected_fi"])
                mq = sam3_frame_mask(q_img, sam3_model, processor, dummy_geo)
                md = sam3_frame_mask(d_img, sam3_model, processor, dummy_geo)
                cls_q = set(c for c in range(N_cats) if mq[c].sum() > 500)
                cls_d = set(c for c in range(N_cats) if md[c].sum() > 500)
                overlap = len(cls_q & cls_d) / min(len(cls_q), len(cls_d)) if (cls_q and cls_d) else 0.0
                lp["overlap"] = overlap
                print(f"  [loop lc={lp['lc_num']}] score={lp['score']:.3f} overlap={overlap:.2f}")
            except Exception as e:
                lp["overlap"] = None
                print(f"  [loop lc={lp['lc_num']}] SAM3 failed: {e}")
        del sam3_model, processor
        gc.collect()
        torch.cuda.empty_cache()
    else:
        for lp in detected_loops:
            lp["overlap"] = None

    # ── decide false loops ──
    false_lc_ids = []
    for lp in detected_loops:
        ov = lp.get("overlap")
        reject = (ov is not None and ov < args.min_overlap)
        if lp.get("lc_num") in force_reject:
            reject = True
        lp["verdict"] = "reject" if reject else ("keep" if ov is not None else "unknown")
        if reject and lp.get("lc_num") is not None:
            false_lc_ids.append(lp["lc_num"])
    print(f"[gen_npz_semantic] rejecting {len(false_lc_ids)} loops: lc_ids={false_lc_ids}")

    if false_lc_ids:
        removed = remove_loops(solver, false_lc_ids)
        print(f"[gen_npz_semantic] removed {removed} factors, re-optimized")

    # ── export AFTER (corrected) ──
    after_npz = os.path.join(args.output_dir, "_stage3_input_after.npz")
    n_exported2 = export_npz(solver, names, vggt_kf_names, pts_gs, rgb_gs,
                             img_H_full, img_W_full, vggt_H, vggt_W, after_npz)
    after_poses = os.path.join(args.output_dir, "stage1_poses_after.txt")
    solver.map.write_poses_to_file(after_poses, solver.graph, give_camera_mat=True)
    print(f"[gen_npz_semantic] AFTER npz {n_exported2} frames -> {after_npz}")

    with open(os.path.join(args.output_dir, "loop_log.json"), "w") as f:
        json.dump({"loops": detected_loops, "reject_lc_ids": false_lc_ids}, f, indent=2)
    print(f"[gen_npz_semantic] DONE. before={before_npz} after={after_npz}")


if __name__ == "__main__":
    main()
