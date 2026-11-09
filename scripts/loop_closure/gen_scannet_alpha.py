#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重跑 VGGT-SLAM stage1（仅回环检测）为 ScanNet 提取 αmatch = image_match_ratio。

不跑 SAM3、不导出 npz，只复用 gen_npz_scan.py 的 stage1 前半段，并用 AlphaCapture
包装 VGGT，在每次 compute_similarity=True 时记录 image_match_ratio。max_loops=1 保证
每个被接受的回环恰好触发一次 compute_similarity，故 alphas 与 detected_loops 一一对应。

输出：ablation/scannet/{scene}/loop_log_alpha.json（字段与已有 loop_log.json 同名，
即 query_sid/query_fi/detected_sid/detected_fi/lc_num/score，外加 alpha）。

Run:
  /root/miniconda3/envs/sgs/bin/python gen_scannet_alpha.py \
    --scenes scene0000_00 scene0059_00 scene0106_00 scene0207_00
"""
import os, sys, glob, time, json, argparse, gc
import numpy as np, cv2, torch

sys.path.insert(0, "/root/autodl-tmp/v3dsam")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/vggt")
os.environ['XFORMERS_DISABLED'] = '1'

from vggt_slam_official.solver import Solver
import vggt_slam_official.slam_utils as slam_utils
from vggt.models.vggt import VGGT

BASE = "/root/autodl-tmp"


class AlphaCapture:
    """包装 VGGT，每次 compute_similarity=True 记录 image_match_ratio（αmatch）。"""
    def __init__(self, model):
        self._m = model
        self.alphas = []

    def __call__(self, *a, **k):
        out = self._m(*a, **k)
        if k.get("compute_similarity", False):
            self.alphas.append(float(out["image_match_ratio"]))
        return out

    def __getattr__(self, n):
        return getattr(self._m, n)


def run_scene(image_dir, submap_size, min_disparity, lc_thres, max_loops):
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(
        "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"))
    model.eval()
    model = model.to(torch.bfloat16).to("cuda")
    cap = AlphaCapture(model)
    solver = Solver(init_conf_threshold=25.0, lc_thres=lc_thres)

    names = sorted(glob.glob(os.path.join(image_dir, "*")))
    names = [f for f in names if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    names = slam_utils.sort_images_by_number(names)

    loops = []
    subset = []
    t0 = time.time()
    for name in names:
        img = cv2.imread(name)
        if solver.flow_tracker.compute_disparity(img, min_disparity, False):
            subset.append(name)
        if len(subset) == submap_size + 1 or name == names[-1]:
            pred = solver.run_predictions(subset, cap, max_loops, None, None)
            for loop in pred.get("detected_loops", []):
                loops.append({
                    "query_sid": int(loop.query_submap_id),
                    "query_fi": int(loop.query_submap_frame),
                    "detected_sid": int(loop.detected_submap_id),
                    "detected_fi": int(loop.detected_submap_frame),
                    "score": float(loop.similarity_score),
                })
            solver.add_points(pred)
            solver.graph.optimize()
            subset = subset[-1:]

    lc_submaps = [sm for sm in solver.map.ordered_submaps_by_key() if sm.get_lc_status()]
    for k, lp in enumerate(loops):
        lp["lc_num"] = int(lc_submaps[k].get_id()) if k < len(lc_submaps) else None

    print(f"[alpha] {image_dir}: {len(loops)} loops, {len(cap.alphas)} alphas, "
          f"{time.time()-t0:.1f}s", flush=True)
    return loops, cap.alphas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+",
                    default=["scene0000_00", "scene0059_00", "scene0106_00", "scene0207_00"])
    ap.add_argument("--submap_size", type=int, default=32)
    ap.add_argument("--min_disparity", type=float, default=50)
    ap.add_argument("--lc_thres", type=float, default=0.95)
    ap.add_argument("--max_loops", type=int, default=1)
    args = ap.parse_args()

    for sc in args.scenes:
        image_dir = f"{BASE}/v3dsam/datasets/scannet/{sc}/color"
        out_dir = f"{BASE}/ablation/scannet/{sc}"
        loops, alphas = run_scene(image_dir, args.submap_size, args.min_disparity,
                                  args.lc_thres, args.max_loops)
        for k, lp in enumerate(loops):
            lp["alpha"] = alphas[k] if k < len(alphas) else None
        if len(alphas) != len(loops):
            print(f"[warn] {sc}: alpha count {len(alphas)} != loops {len(loops)}", flush=True)
        out = os.path.join(out_dir, "loop_log_alpha.json")
        with open(out, "w") as f:
            json.dump({"loops": loops, "lc_thres": args.lc_thres,
                       "note": "alpha = image_match_ratio (VGGT-SLAM 2.0 official amatch)"},
                      f, indent=2)
        print(f"[alpha] saved {out}  (n={len(loops)})", flush=True)
        del loops, alphas
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
