#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Table 4-3 核心评估（instance-filt GT 版）：GT 物理实例 + 各方法逐帧实例掩膜 -> IDC/IDS/FM/ReID-R。

非循环性：
  - GT 实例 = ScanNet 官方人工 2D 实例标注（instance-filt/，跨帧一致的 instance id）。
    任何方法的预测 ID 都不参与 GT 身份定义。
  - GT 掩膜 = instance-filt PNG 中 (pixel == instance_id) 的直接像素掩膜（精确，无凸包近似）。
  - GT 类 = 该 instance 所有可见像素的 NYU40 类多数投票（semantic_id + map_gt）。
  - 方法掩膜 = 方法 3D 实例点投影（自身位姿 + K）膨胀，或方法直接输出的 2D 掩膜。
  - 两者在同一图像平面逐帧按「同类 + 覆盖率」贪心匹配，得到每个 GT 实例在各可见帧
    匹配到的方法实例 ID 序列。

指标（MOT 标准，同一 (GT实例 × 可见帧) 对上）：
  IDC  ID Consistency = 匹配 ID == 众数 ID 的帧占比（1=完美）
  IDS  ID Switches   = 连续两可见帧匹配 ID 改变次数（全场景求和）
  FM   Fragmentation = 可见但未匹配到任何方法掩膜的帧数
  ReID-R Loop Re-ID Recall = 首末可见帧时间跨度 >= 25% 序列长度时，首末匹配 ID 相同的占比

Run on AutoDL (sgs env):
  cd /root/autodl-tmp
  /root/miniconda3/envs/sgs/bin/python -u eval_instance_identity_gt.py --scene scene0000_00
"""
import os, sys, argparse, json
import numpy as np, cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scannet_labels import NYU40, map_gt
from instance_gt_common import (
    FOREGROUND_NYU, FG_LIST, PROMPT2NYU,
    project_pixels, convex_hull_mask, match_frame, compute_metrics)

ABL = "/root/autodl-tmp/ablation/scannet"
DAT = "/root/autodl-tmp/v3dsam/datasets/scannet"
INST = "/root/autodl-tmp/gt_inst_filt"   # instance-filt 关键帧 PNG（本地提取上传）

SCENES = ["scene0000_00", "scene0059_00", "scene0106_00", "scene0207_00"]


# ---------------------------------------------------------------- GT 加载
def load_gt_instances(scene, kf_indices, min_px=50):
    """读 instance-filt + semantic_id，返回 (inst_class, inst_frames)。
    inst_class: {instance_id: nyu40}；inst_frames: {instance_id: [可见帧号]}。
    instance id 跨帧一致（ScanNet 官方 2D 实例从 3D 实例投影）。"""
    inst_dir = f"{INST}/{scene}"
    sem_dir = f"{DAT}/{scene}/semantic_id"
    px_cls = {}    # {iid: {nyu40: count}}
    frames = {}    # {iid: [fi]}

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


# ---------------------------------------------------------------- 方法 A（几何）加载
def load_instance_cloud(base):
    pts, ids = [], []
    with open(f"{base}/semantic_instances/instance_cloud.ply") as f:
        for i, line in enumerate(f):
            if i < 11:
                continue
            p = line.split()
            pts.append((float(p[0]), float(p[1]), float(p[2])))
            ids.append(int(p[6]))
    return np.array(pts, np.float64), np.array(ids, np.int64)


def load_method_a(base):
    inst = json.load(open(f"{base}/semantic_instances/instances.json"))
    voted = json.load(open(f"{base}/semantic_instances/voted_instances.json"))
    prompts = [str(p) for p in inst["prompts"]]
    vc = {int(o["id"]): int(o["voted_class"]) for o in voted["instances"]}
    objs = []
    for o in inst["instances"]:
        i = int(o["id"])
        if i not in vc:
            continue
        name = prompts[vc[i]]
        nyu = PROMPT2NYU.get(name, -1)
        if nyu not in FOREGROUND_NYU:
            continue
        if int(o.get("n_vis", 0)) < 3:
            continue
        objs.append({"id": i, "nyu": nyu, "name": name, "n_vis": int(o.get("n_vis", 0))})
    objs.sort(key=lambda o: -o["n_vis"])
    pts_all, ids_all = load_instance_cloud(base)
    id2pts = {o["id"]: pts_all[ids_all == o["id"]] for o in objs}
    return objs, id2pts


def method_a_masks_frame(objs, id2pts, w2c, K, W, H, min_px=40, dilate=3):
    out = []
    for o in objs:
        p = id2pts[o["id"]]
        m = convex_hull_mask(p, w2c, K, W, H, min_px=min_px)
        if m is not None:
            m = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8),
                           iterations=dilate).astype(bool)
        out.append((o["id"], o["nyu"], m))
    return out


# ---------------------------------------------------------------- 单场景评估
def eval_scene(scene, args):
    base = f"{ABL}/{scene}"
    stage3 = np.load(f"{base}/_stage3_input_after.npz", allow_pickle=True)
    sem = np.load(f"{base}/kf_semantic.npz", allow_pickle=True)
    kf_indices = [int(x) for x in sem["kf_indices"].tolist()]
    if args.max_frames > 0:
        kf_indices = kf_indices[:args.max_frames]
    img_W, img_H = int(stage3["img_W"]), int(stage3["img_H"])

    inst_class, inst_frames = load_gt_instances(scene, kf_indices, min_px=args.min_px)
    # 只保留前景类实例
    fg_insts = {iid: c for iid, c in inst_class.items() if c in FOREGROUND_NYU}
    M = len(fg_insts)
    print(f"\n[{scene}] GT 前景实例={M} (总实例={len(inst_class)}), kf={len(kf_indices)}")
    dist = {}
    for c in fg_insts.values():
        dist[c] = dist.get(c, 0) + 1
    print(f"  GT class dist: { {NYU40[c]: n for c, n in sorted(dist.items())} }")

    objsA, id2pts = load_method_a(base)
    print(f"  method A instances={len(objsA)}")

    # GT 可见帧（仅前景实例），用于指标
    gt_visible = {iid: sorted(frames) for iid, frames in inst_frames.items()
                  if iid in fg_insts}
    match_tracks = {iid: {} for iid in gt_visible}
    n_vis_frame = 0
    n_match_frame = 0
    n_class_agree = 0

    for k, fi in enumerate(kf_indices):
        K = stage3[f"K_{fi}"].astype(np.float64)
        inst = cv2.imread(f"{INST}/{scene}/{fi}.png", cv2.IMREAD_UNCHANGED)
        if inst is None:
            continue
        # GT 掩膜：直接读 instance-filt
        gt_masks = []
        for iid in np.unique(inst):
            if iid == 0 or int(iid) not in fg_insts:
                continue
            m = (inst == iid)
            if int(m.sum()) < args.min_px:
                continue
            gt_masks.append((int(iid), m.astype(bool)))
        gt_nyu = {iid: fg_insts[iid] for iid, _ in gt_masks}

        # 方法 A 掩膜
        w2c_a = stage3[f"w2c_{fi}"].astype(np.float64)
        ams = method_a_masks_frame(objsA, id2pts, w2c_a, K, img_W, img_H)
        inst_masks = [(iid, m) for (iid, nyu, m) in ams if m is not None]
        inst_nyu = {iid: nyu for (iid, nyu, m) in ams}

        res = match_frame([(iid, m) for (iid, m) in gt_masks],
                          inst_masks, inst_nyu, gt_nyu, cov_thr=args.cov_thr,
                          use_class=args.use_class)
        for iid in gt_visible:
            if iid in res:
                match_tracks[iid][fi] = res[iid]
            else:
                match_tracks[iid][fi] = None

        n_vis_frame += len(gt_masks)
        n_match_frame += sum(1 for v in res.values() if v is not None)
        n_class_agree += sum(1 for gid, mid in res.items()
                             if mid is not None
                             and inst_nyu.get(mid, -1) == gt_nyu.get(gid, -1))

        if (k + 1) % 100 == 0:
            print(f"  [{scene}] {k+1}/{len(kf_indices)} frames")

    m = compute_metrics(gt_visible, match_tracks, revisit_gap_frac=args.revisit_frac)
    m["scene"] = scene
    m["n_gt_objects"] = M
    m["n_method_instances"] = len(objsA)
    m["n_vis_pairs"] = n_vis_frame
    m["n_match_pairs"] = n_match_frame
    m["n_class_agree"] = n_class_agree
    m["class_agree"] = n_class_agree / max(1, n_match_frame)
    print(f"  [{scene}] vis_pairs={n_vis_frame} match_pairs={n_match_frame} "
          f"match_rate={n_match_frame/max(1,n_vis_frame):.3f} "
          f"class_agree={m['class_agree']:.3f}")
    print(f"  [{scene}] IDC={m['IDC']:.3f} ({m['IDC_num']}/{m['IDC_den']}) "
          f"IDS={m['IDS']} FM={m['FM']} "
          f"ReID_R={m['ReID_R']:.3f} ({m['ReID_num']}/{m['ReID_den']})")
    return m


# ---------------------------------------------------------------- 汇总 + 写表
def aggregate(rows):
    agg = {"IDC_num": 0, "IDC_den": 0, "ReID_num": 0, "ReID_den": 0,
           "IDS": 0, "FM": 0, "n_class_agree": 0, "n_match_pairs": 0}
    for r in rows:
        agg["IDC_num"] += r["IDC_num"]; agg["IDC_den"] += r["IDC_den"]
        agg["ReID_num"] += r["ReID_num"]; agg["ReID_den"] += r["ReID_den"]
        agg["IDS"] += r["IDS"]; agg["FM"] += r["FM"]
        agg["n_class_agree"] += r.get("n_class_agree", 0)
        agg["n_match_pairs"] += r.get("n_match_pairs", 0)
    agg["IDC"] = agg["IDC_num"] / agg["IDC_den"] if agg["IDC_den"] else 0.0
    agg["ReID_R"] = agg["ReID_num"] / agg["ReID_den"] if agg["ReID_den"] else 0.0
    agg["class_agree"] = (agg["n_class_agree"] / agg["n_match_pairs"]
                          if agg["n_match_pairs"] else 0.0)
    return agg


def write_table(method_rows, out_dir, cov_thr, revisit_frac, use_class=False):
    os.makedirs(out_dir, exist_ok=True)
    scenes = [r["scene"] for r in method_rows[0][1]]

    csv_lines = ["scene,method,IDC,IDC_num,IDC_den,IDS,FM,ReID_R,ReID_num,ReID_den,"
                 "class_agree"]
    for name, rows in method_rows:
        for r in rows:
            csv_lines.append(
                f"{r['scene']},{name},{r['IDC']:.4f},{r['IDC_num']},{r['IDC_den']},"
                f"{r['IDS']},{r['FM']},{r['ReID_R']:.4f},{r['ReID_num']},{r['ReID_den']},"
                f"{r['class_agree']:.4f}")
    with open(f"{out_dir}/table4_3.csv", "w") as f:
        f.write("\n".join(csv_lines) + "\n")

    md = ["# Table 4-3  实例身份一致性（GT = ScanNet instance-filt，非循环）\n",
          "| 方法 | IDC ↑ | IDS ↓ | FM ↓ | ReID-R ↑ | 类一致率 |",
          "|---|---|---|---|---|---|"]
    for name, rows in method_rows:
        a = aggregate(rows)
        md.append(f"| {name} | {a['IDC']:.3f} ({a['IDC_num']}/{a['IDC_den']}) "
                  f"| {a['IDS']} | {a['FM']} "
                  f"| {a['ReID_R']:.3f} ({a['ReID_num']}/{a['ReID_den']}) "
                  f"| {a['class_agree']:.3f} |")
    md.append("\n> 口径：GT=ScanNet 官方 2D instance-filt（跨帧一致实例 id）；"
              "类=语义多数投票（仅用于 GT 前景过滤）；匹配=逐帧%s覆盖率≥%.2f；"
              "ReID-R=首末可见帧跨度≥%.0f%%序列长度。\n"
              % ("同类+几何" if use_class else "几何", cov_thr, revisit_frac * 100))
    with open(f"{out_dir}/table4_3.md", "w") as f:
        f.write("\n".join(md) + "\n")

    print("\n" + "\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="", help="空=全部 4 场景")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--cov_thr", type=float, default=0.30)
    ap.add_argument("--revisit_frac", type=float, default=0.25)
    ap.add_argument("--min_px", type=int, default=50)
    ap.add_argument("--use_class", action="store_true",
                    help="匹配时要求同类（默认关闭=纯几何覆盖，实例身份=几何）")
    ap.add_argument("--out", default="/root/autodl-tmp/table4_3")
    args = ap.parse_args()

    scenes = [args.scene] if args.scene else SCENES
    rows = []
    for scene in scenes:
        try:
            rows.append(eval_scene(scene, args))
        except FileNotFoundError as e:
            print(f"[eval] SKIP {scene} (缺方法输出): {e}")

    if not rows:
        print("[eval] 无场景可评估"); return
    write_table([("V3DSAM (geometric)", rows)], args.out, args.cov_thr,
                args.revisit_frac, args.use_class)
    print("\n[eval] DONE")


if __name__ == "__main__":
    main()
