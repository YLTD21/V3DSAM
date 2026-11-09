import os, sys, glob, time, argparse, subprocess, shlex, json, shutil, gc
import numpy as np, cv2, torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/v3dsam")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/vggt")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/sam3")

from vggt_slam_official.solver import Solver
from vggt.models.vggt import VGGT
from gtsam.symbol_shorthand import X
import gtsam

EVO = "/root/miniconda3/envs/sgs/bin/evo_ape"
device = "cuda"

ALL_PROMPTS = [
    "wall","floor","ceiling","door","window","stairs","railing",
    "bed","cabinet","bookshelf","shelf","counter","desk","table","sofa","couch",
    "sink","toilet","bathtub","mirror","lamp","light",
    "television","monitor","laptop","keyboard","mouse","phone","refrigerator","oven","microwave",
    "chair","plant","vase","picture","painting","curtain","pillow","clock",
    "book","bottle","cup","bowl","plate","bag","backpack","box","bin","basket","person",
]
N_cats = len(ALL_PROMPTS)


def run_ate(gt, est):
    r = subprocess.run(f"{EVO} tum {shlex.quote(gt)} {shlex.quote(est)} -as --no_warnings",
                       shell=True, capture_output=True, text=True)
    rmse = None
    for line in r.stdout.splitlines():
        if line.strip().startswith("rmse"):
            rmse = float(line.split()[-1])
    return rmse, r.stdout


def run_stage1(image_dir, submap_size, min_disparity, lc_thres, max_loops, out):
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(
        "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"))
    model.eval()
    model = model.to(torch.bfloat16).to(device)

    solver = Solver(init_conf_threshold=25.0, lc_thres=lc_thres)
    names = sorted(glob.glob(os.path.join(image_dir, "*")))
    detected_loop_pairs = []
    image_names_subset = []
    t0 = time.time()
    for name in names:
        img = cv2.imread(name)
        if solver.flow_tracker.compute_disparity(img, min_disparity, False):
            image_names_subset.append(name)
        if len(image_names_subset) == submap_size + 1 or name == names[-1]:
            predictions = solver.run_predictions(image_names_subset, model, max_loops, None, None)
            for loop in predictions.get("detected_loops", []):
                detected_loop_pairs.append({
                    "query_sid": int(loop.query_submap_id),
                    "query_fi": int(loop.query_submap_frame),
                    "detected_sid": int(loop.detected_submap_id),
                    "detected_fi": int(loop.detected_submap_frame),
                    "score": float(loop.similarity_score),
                })
            solver.add_points(predictions)
            solver.graph.optimize()
            image_names_subset = image_names_subset[-1:]
    print(f"[semantic_filter] Stage1 done {time.time()-t0:.1f}s, {len(detected_loop_pairs)} loops, {len(names)} imgs")

    os.makedirs(out, exist_ok=True)
    before = os.path.join(out, "stage1_poses_before.txt")
    solver.map.write_poses_to_file(before, solver.graph, give_camera_mat=True)

    lc_submaps = [sm for sm in solver.map.ordered_submaps_by_key() if sm.get_lc_status()]
    print(f"[semantic_filter] {len(lc_submaps)} lc submaps vs {len(detected_loop_pairs)} loops")
    for k, lp in enumerate(detected_loop_pairs):
        lp["lc_num"] = int(lc_submaps[k].get_id()) if k < len(lc_submaps) else None

    return solver, detected_loop_pairs, before


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
    return cat_mask, (h0, w0)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--submap_size", type=int, default=32)
    ap.add_argument("--min_disparity", type=float, default=50)
    ap.add_argument("--lc_thres", type=float, default=0.95)
    ap.add_argument("--max_loops", type=int, default=1)
    ap.add_argument("--min_overlap", type=float, default=0.3)
    ap.add_argument("--skip_sam3", action="store_true")
    ap.add_argument("--reject_lc_ids", type=str, default="",
                    help="comma-separated lc_nums to force-reject (overrides overlap)")
    args = ap.parse_args()
    force_reject = {int(x) for x in args.reject_lc_ids.split(",") if x.strip() != ""}

    solver, loops, before_poses = run_stage1(
        args.image_dir, args.submap_size, args.min_disparity, args.lc_thres,
        args.max_loops, args.out)
    rmse_before, _ = run_ate(args.gt, before_poses)
    print(f"[semantic_filter] ATE before = {rmse_before}")

    if not args.skip_sam3 and loops:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        print("[semantic_filter] loading SAM3...")
        sam3_model = build_sam3_image_model(
            bpe_path="/root/autodl-tmp/v3dsam/third_party/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            checkpoint_path="/root/autodl-tmp/v3dsam/third_party/sam3/sam3.pt", device="cuda")
        processor = Sam3Processor(sam3_model, confidence_threshold=0.1)
        dummy_geo = sam3_model._get_dummy_prompt(num_prompts=1)

        fig_dir = os.path.join(args.out, "loop_figures")
        os.makedirs(fig_dir, exist_ok=True)
        for k, lp in enumerate(loops):
            q_sm = solver.map.get_submap(lp["query_sid"])
            d_sm = solver.map.get_submap(lp["detected_sid"])
            q_img = q_sm.get_img_names_at_index(lp["query_fi"])
            d_img = d_sm.get_img_names_at_index(lp["detected_fi"])
            try:
                mq, _ = sam3_frame_mask(q_img, sam3_model, processor, dummy_geo)
                md, _ = sam3_frame_mask(d_img, sam3_model, processor, dummy_geo)
                cls_q = set(c for c in range(N_cats) if mq[c].sum() > 500)
                cls_d = set(c for c in range(N_cats) if md[c].sum() > 500)
                overlap = len(cls_q & cls_d) / min(len(cls_q), len(cls_d)) if (cls_q and cls_d) else 0.0
                lp["overlap"] = overlap
                lp["classes_q"] = sorted(cls_q)
                lp["classes_d"] = sorted(cls_d)
                shutil.copy(q_img, os.path.join(fig_dir, f"loop{k}_query.png"))
                shutil.copy(d_img, os.path.join(fig_dir, f"loop{k}_detected.png"))
                np.savez_compressed(os.path.join(fig_dir, f"loop{k}_masks.npz"), mask_q=mq, mask_d=md)
                print(f"[loop {k}] score={lp['score']:.3f} overlap={overlap:.2f} "
                      f"q_classes={len(cls_q)} d_classes={len(cls_d)}")
            except Exception as e:
                lp["overlap"] = None
                print(f"[loop {k}] SAM3 failed: {e}")
        del sam3_model, processor
        gc.collect()
        torch.cuda.empty_cache()
    else:
        for lp in loops:
            lp["overlap"] = None

    false_lc_ids = []
    for lp in loops:
        ov = lp.get("overlap")
        reject = (ov is not None and ov < args.min_overlap)
        if lp.get("lc_num") in force_reject:
            reject = True
        lp["verdict"] = "reject" if reject else ("keep" if ov is not None else "unknown")
        if reject and lp.get("lc_num") is not None:
            false_lc_ids.append(lp["lc_num"])

    print(f"[semantic_filter] rejecting {len(false_lc_ids)} loops: lc_ids={false_lc_ids}")
    if false_lc_ids:
        removed = remove_loops(solver, false_lc_ids)
        print(f"[semantic_filter] removed {removed} factors, re-optimized")

    after = os.path.join(args.out, "stage1_poses_after.txt")
    solver.map.write_poses_to_file(after, solver.graph, give_camera_mat=True)
    rmse_after, _ = run_ate(args.gt, after)
    print(f"[semantic_filter] ATE after = {rmse_after}")

    log_path = os.path.join(args.out, "loop_log.json")
    with open(log_path, "w") as f:
        json.dump({"loops": loops, "rmse_before": rmse_before, "rmse_after": rmse_after}, f, indent=2)
    print(f"[semantic_filter] saved {log_path}")
    print(f"RESULT rmse_before={rmse_before} rmse_after={rmse_after}")


if __name__ == "__main__":
    main()
