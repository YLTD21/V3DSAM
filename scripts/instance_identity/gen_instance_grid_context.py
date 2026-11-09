#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3×5 持久实例矩阵 —— SAM3 oneshot 2D 实例 + 跨帧 3D 关联 + 70/30 邻域类别精修。

在 gen_instance_grid_oneshot.py 的基础上加「上下文类别精修」：每个 3D 实例的最终
类别不再单纯取 SAM3 的多数投票，而是：

    score[c] = w_sam3 * p_sam3[c]  +  w_ctx * agree[c]

其中
    p_sam3[c] = 该实例所有 2D mask 的 SAM3 类别得分（按类别累加后归一化）—— 类别自身证据
    agree[c]  = 该实例 3D 邻域（质心 KD-tree 半径 radius 内）里类别为 c 的邻居占比 —— 上下文一致性

w_sam3 = 0.7, w_ctx = 0.3 为默认（70/30）。邻居项是「一致性先验」：孤岛误分被邻域
拉回，但不重猜类别；无邻居时退化为纯 SAM3 证据。

同一脚本输出**两行对比**：上行 = 纯 SAM3（w_ctx=0 基线），下行 = 70/30 精修。

Run on AutoDL (sgs env):
  cd /root/autodl-tmp
  /root/miniconda3/envs/sgs/bin/python -u gen_instance_grid_context.py \
    --scenes scene0000_00 --out /root/autodl-tmp/scannet_figs_all/fig_instance_grid_context
"""
import os, sys, argparse
from collections import Counter
import numpy as np, torch, cv2

sys.path.insert(0, "/root/autodl-tmp/v3dsam")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/vggt")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/sam3")
sys.path.insert(0, "/root/autodl-tmp/v3dsam/third_party/perception_models")
os.environ['XFORMERS_DISABLED'] = '1'

from sam3 import build_sam3_image_model
from dump_sam3_masks_oneshot import ALL_PROMPTS, N_cats, build_transform, oneshot_masks
from gen_instance_grid_scannet import (
    TAB20, hex2rgb, draw_instance, draw_label, mask_label_pos,
    load_objs, load_instance_cloud, visible_mask, pick_window, SCENES, BASE)
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from sci_style import finalize_figure
except Exception:
    def finalize_figure(fig, out):
        fig.savefig(out + ".png", dpi=200, bbox_inches="tight", facecolor="white")
        print(f"-> {out}.png")

BACKGROUND = {"wall", "floor", "ceiling"}


def backproject(pts, stage3, frames, instances_per_frame, min_pts=10):
    """每个 2D 实例反投影得到 pts 里的 3D 点索引（z-buffer 最近点）。"""
    pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
    entries = []
    for fi in frames:
        if fi not in instances_per_frame:
            continue
        w2c = stage3[f"w2c_{fi}"].astype(np.float64)
        K = stage3[f"K_{fi}"].astype(np.float64)
        cam = (w2c @ pts_h.T).T
        z = cam[:, 2]
        front = z > 0.05
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        u = fx * cam[:, 0] / np.maximum(z, 1e-6) + cx
        v = fy * cam[:, 1] / np.maximum(z, 1e-6) + cy
        insts = instances_per_frame[fi]
        H, W = insts[0][1].shape
        ui = np.floor(u).astype(np.int64)
        vi = np.floor(v).astype(np.int64)
        inb = front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        inb_idx = np.nonzero(inb)[0]
        ui_in = ui[inb_idx]; vi_in = vi[inb_idx]; zi_in = z[inb_idx]
        flat = vi_in.astype(np.int64) * W + ui_in
        order = np.lexsort((zi_in, flat))
        sflat = flat[order]
        first = np.concatenate(([True], sflat[1:] != sflat[:-1]))
        surf_pidx = inb_idx[order][first]
        surf_ui = ui_in[order][first]; surf_vi = vi_in[order][first]
        for li, (cat, mask, score) in enumerate(insts):
            sel = mask[surf_vi, surf_ui]
            pidx = surf_pidx[sel]
            if len(pidx) < min_pts:
                continue
            entries.append({"fi": fi, "li": li, "cat": int(cat),
                            "score": float(score), "pidx": pidx})
    return entries


def associate(entries, pts, dist_thresh=0.3):
    """跨帧关联（同类别 + 3D 质心距离），返回 [{cat, centroid, members:[entry idx]}]。"""
    n = len(entries)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    cents = np.empty((n, 3), dtype=np.float64)
    for i, e in enumerate(entries):
        cents[i] = pts[e["pidx"]].mean(axis=0)

    # 跨帧几何关联，不 gate 类别：同一物理物体跨视角可能被 SAM3 标成不同类别
    #（chair/table 正反脸），只有几何（3D 质心距离）才能定身份。同帧不合并
    #（同帧每个 2D mask 已是 SAM3 分开的独立物体）。
    for a in range(n):
        for b in range(a + 1, n):
            if entries[a]["fi"] == entries[b]["fi"]:
                continue
            if np.linalg.norm(cents[a] - cents[b]) < dist_thresh:
                union(a, b)

    comp = {}
    for i in range(n):
        comp.setdefault(find(i), []).append(i)

    instances = []
    for members in comp.values():
        cat = Counter(entries[m]["cat"] for m in members).most_common(1)[0][0]
        instances.append({"cat": cat, "members": members})
    return instances


def p_sam3_dist(inst, entries, n_cats):
    """该实例的 SAM3 类别证据分布：按类别累加 2D mask 得分，归一化。"""
    dist = np.zeros(n_cats, dtype=np.float64)
    for m in inst["members"]:
        e = entries[m]
        dist[e["cat"]] += max(e["score"], 0.0)
    s = dist.sum()
    if s > 0:
        dist /= s
    return dist


def refine_category(instances, entries, pts, n_cats, w_sam3, w_ctx, radius):
    """70/30 邻域精修：score[c] = w_sam3*p_sam3[c] + w_ctx*agree[c]。

    agree[c] = 质心半径 radius 内邻居（排除自身）中类别为 c 的比例。无邻居 -> 退化为纯 SAM3。
    返回每个实例的 (final_cat, p_sam3, agree)。"""
    cents = np.array([pts[np.concatenate([entries[m]["pidx"] for m in inst["members"]])].mean(0)
                      for inst in instances]) if instances else np.zeros((0, 3))
    tree = cKDTree(cents) if len(cents) else None
    out = []
    for i, inst in enumerate(instances):
        ps = p_sam3_dist(inst, entries, n_cats)
        agree = np.zeros(n_cats, dtype=np.float64)
        if tree is not None and w_ctx > 0:
            nbr = tree.query_ball_point(cents[i], radius)
            nbr = [j for j in nbr if j != i]
            if nbr:
                cnt = Counter(instances[j]["cat"] for j in nbr)
                tot = sum(cnt.values())
                for c, k in cnt.items():
                    agree[c] = k / tot
        score = w_sam3 * ps + w_ctx * agree
        final = int(score.argmax())
        out.append((final, ps, agree))
    return out


def select_persistent(instances, finals, win, entries, pts, prompts, max_inst):
    local_set = set(win[:3]); loop_set = set(win[3:])
    keep = []
    for i, inst in enumerate(instances):
        name = prompts[inst["cat"]]
        if name in BACKGROUND:
            continue
        frames = {entries[m]["fi"] for m in inst["members"]}
        nl = sum(1 for fi in frames if fi in local_set)
        nlp = sum(1 for fi in frames if fi in loop_set)
        if nl >= 2 and nlp >= 1:
            keep.append(i)
    keep.sort(key=lambda i: -sum(1 for _ in instances[i]["members"]))
    keep = keep[:max_inst]
    return keep


def draw_row(ax, win, keep_idx, instances, finals, entries, instances_per_frame,
             stage3, prompts, row_title, is_top):
    ax[0].set_ylabel(f"{row_title}\n{len(keep_idx)} persistent",
                     fontsize=11, rotation=0, labelpad=30,
                     va="center", color="#272727", fontweight="bold")
    for c, fi in enumerate(win):
        img = cv2.imread(str(stage3[f"img_path_{fi}"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        for rank, i in enumerate(keep_idx):
            if fi not in instances_per_frame:
                continue
            # 找到该实例在此帧的 2D mask（取该帧里属于此实例的成员）
            li = None
            for m in instances[i]["members"]:
                if entries[m]["fi"] == fi:
                    li = entries[m]["li"]
                    break
            if li is None:
                continue
            imask = instances_per_frame[fi][li][1]
            final_cat = finals[i][0]
            colrgb = hex2rgb(TAB20[rank % len(TAB20)])
            img = draw_instance(img, imask, colrgb)
        for rank, i in enumerate(keep_idx):
            li = None
            for m in instances[i]["members"]:
                if entries[m]["fi"] == fi:
                    li = entries[m]["li"]
                    break
            if li is None:
                continue
            imask = instances_per_frame[fi][li][1]
            final_cat = finals[i][0]
            colrgb = hex2rgb(TAB20[rank % len(TAB20)])
            lu, lv = mask_label_pos(imask, W, H, 0, 0)
            draw_label(img, f"#{rank} {prompts[final_cat]}", lu, lv, colrgb, W, H)
        ax[c].imshow(img)
        if is_top:
            ax[c].set_title(f"kf {fi}" if c < 3 else f"loop {fi}", fontsize=11, color="#272727")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=["scene0000_00"])
    ap.add_argument("--n_frames", type=int, default=5)
    ap.add_argument("--min_score", type=float, default=0.3)
    ap.add_argument("--dist_thresh", type=float, default=0.3)
    ap.add_argument("--w_sam3", type=float, default=0.7)
    ap.add_argument("--w_ctx", type=float, default=0.3)
    ap.add_argument("--radius", type=float, default=0.8)
    ap.add_argument("--max_pts", type=int, default=1000000)
    ap.add_argument("--max_inst", type=int, default=25)
    ap.add_argument("--out", default="/root/autodl-tmp/scannet_figs_all/fig_instance_grid_context")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    scene_keys = [k for k in SCENES if SCENES[k][0] in args.scenes]
    if not scene_keys:
        scene_keys = [k for k in SCENES if any(s in k for s in args.scenes)]
    if not scene_keys:
        scene_keys = list(SCENES.keys())
    print(f"[context] scenes = {scene_keys}")

    print("[context] loading SAM3 ...")
    model = build_sam3_image_model(
        bpe_path="/root/autodl-tmp/v3dsam/third_party/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        checkpoint_path="/root/autodl-tmp/v3dsam/third_party/sam3/sam3.pt",
        device="cuda")
    transform = build_transform()
    device = torch.device("cuda")
    print("[context] SAM3 loaded")

    n = args.n_frames
    n_rows = 2  # 上行基线 w_ctx=0，下行 70/30
    fig, axes = plt.subplots(n_rows, n, figsize=(2.35 * n, 3.1 * n_rows))
    for r in range(n_rows):
        for c in range(n):
            axes[r, c].axis("off")

    for scene in scene_keys:
        name, s3_rel, sem_rel, inst_rel = SCENES[scene]
        base = f"{BASE}/{name}"
        stage3 = np.load(f"{base}/{s3_rel}", allow_pickle=True)
        sem = np.load(f"{base}/{sem_rel}", allow_pickle=True)
        kf_indices = [int(x) for x in sem["kf_indices"].tolist()]
        img_W, img_H = int(stage3["img_W"]), int(stage3["img_H"])

        # 选帧（复用旧 pick_window）
        objs = load_objs(base, inst_rel)
        pts_all, ids_all = load_instance_cloud(base)
        id2pts = {o["id"]: pts_all[ids_all == o["id"]] for o in objs}
        K = len(kf_indices)
        vis = np.zeros((K, len(objs)), dtype=bool)
        blur = np.zeros(K, dtype=np.float64)
        for k, fi in enumerate(kf_indices):
            m, _, _ = visible_mask(objs, stage3[f"w2c_{fi}"].astype(np.float64),
                                   stage3[f"K_{fi}"].astype(np.float64), img_W, img_H)
            vis[k] = m
            g = cv2.imread(str(stage3[f"img_path_{fi}"]), cv2.IMREAD_GRAYSCALE)
            blur[k] = cv2.Laplacian(g, cv2.CV_64F).var()
        _, _, win_pos, _, _ = pick_window(vis, blur)
        win = [kf_indices[p] for p in win_pos]
        print(f"[context] {scene}: local={win[:3]} loop={win[3:]}")

        # oneshot 2D 实例
        instances_per_frame = {}
        for fi in win:
            p = str(stage3[f"img_path_{fi}"])
            insts, _ = oneshot_masks(p, model, transform, device, args.min_score)
            instances_per_frame[fi] = insts
            print(f"[context] {scene} kf{fi}: {len(insts)} 2D instances")

        # 反投影 + 关联
        pts = stage3["pts_gs"].astype(np.float64)
        if len(pts) > args.max_pts:
            idx = np.random.RandomState(0).choice(len(pts), args.max_pts, replace=False)
            idx.sort()
            pts = pts[idx]
        entries = backproject(pts, stage3, win, instances_per_frame)
        instances = associate(entries, pts, dist_thresh=args.dist_thresh)
        print(f"[context] {scene}: {len(entries)} 2D entries -> {len(instances)} 3D instances")

        # 基线（纯 SAM3）+ 70/30 精修
        finals_base = refine_category(instances, entries, pts, N_cats,
                                      w_sam3=1.0, w_ctx=0.0, radius=args.radius)
        finals_ctx = refine_category(instances, entries, pts, N_cats,
                                     w_sam3=args.w_sam3, w_ctx=args.w_ctx, radius=args.radius)

        # 打印邻域精修前后差异
        n_flip = sum(1 for i in range(len(instances))
                     if finals_base[i][0] != finals_ctx[i][0]
                     and ALL_PROMPTS[instances[i]["cat"]] not in BACKGROUND)
        print(f"[context] {scene}: {n_flip} instances flipped by neighborhood "
              f"(w_ctx {args.w_sam3}/{args.w_ctx})")
        for i in range(len(instances)):
            c0, c1 = finals_base[i][0], finals_ctx[i][0]
            if c0 != c1 and ALL_PROMPTS[instances[i]["cat"]] not in BACKGROUND:
                print(f"    [{ALL_PROMPTS[c0]:<10} -> {ALL_PROMPTS[c1]:<10}] "
                      f"p_sam3={finals_ctx[i][1].max():.2f} "
                      f"agree_new={finals_ctx[i][2][c1]:.2f}")

        keep_base = select_persistent(instances, finals_base, win, entries, pts,
                                      ALL_PROMPTS, args.max_inst)
        keep_ctx = select_persistent(instances, finals_ctx, win, entries, pts,
                                     ALL_PROMPTS, args.max_inst)
        print(f"[context] {scene}: persistent base={len(keep_base)} ctx={len(keep_ctx)}")

        draw_row(axes[0], win, keep_base, instances, finals_base, entries,
                 instances_per_frame, stage3, ALL_PROMPTS, "w_ctx=0 (SAM3)", is_top=True)
        draw_row(axes[1], win, keep_ctx, instances, finals_ctx, entries,
                 instances_per_frame, stage3, ALL_PROMPTS,
                 f"{args.w_sam3:.0%}/{args.w_ctx:.0%} context", is_top=False)

    fig.suptitle(
        "Persistent instance identity on ScanNet — SAM3 oneshot 2D masks + cross-frame 3D association\n"
        "top: pure SAM3 category vote;  bottom: 70% SAM3 category + 30% 3D-neighbourhood category consistency",
        fontsize=14, y=1.02)
    finalize_figure(fig, args.out)
    print(f"[context] -> {args.out}.png")


if __name__ == "__main__":
    main()
