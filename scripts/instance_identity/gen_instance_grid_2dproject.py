#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3×5 持久实例矩阵 —— 逐帧 2D 投影聚类 + 跨帧 3D 关联（含「同类别最近邻 3D 距离」补救）。

纯 3D 欧氏聚类的问题：遮挡把同一物体在 3D 里切出缝（椅子座/腿、沙发正反面），
聚类就碎。本脚本反过来：把「定身份」挪到 2D 投影里做——

  1. 逐帧投影：用该帧位姿（w2c/K）把全局语义点云投到图上，z-buffer 取最近点，
     得到一张 2D 语义图（每像素 = 类别）。
  2. 2D 聚类：对每个前景类别在 2D 语义图上做连通域 → 每帧的干净实例掩膜。
  3. 跨帧关联：两帧同类别实例满足任一条件 → union-find 合并成同一 3D 实例：
       (a) 质心距离 < dist_thresh；
       (b) 共享 3D 点 >= min_overlap；
       (c) 最近邻 3D 距离补救：较小实例有 >= min_nn 个点落在较大实例 nn_thresh 内
           （修「正/反面不共享点、但两段紧邻」的断裂，如床的头板+床垫）。

默认跑单场景，输出**两行对比**：上行 = 无最近邻补救，下行 = 加补救。

Run on AutoDL (sgs env):
  cd /root/autodl-tmp
  /root/miniconda3/envs/sgs/bin/python -u gen_instance_grid_2dproject.py
"""
import os, argparse
from collections import Counter
from itertools import combinations
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

try:
    from sci_style import apply_publication_style, finalize_figure
    apply_publication_style(font_size=8, axes_linewidth=1)
except Exception:
    def finalize_figure(fig, out):
        fig.savefig(out + ".png", dpi=200, bbox_inches="tight", facecolor="white")
        print(f"-> {out}.png")

from gen_instance_grid_scannet import (
    visible_mask, pick_window, draw_instance, draw_label, mask_label_pos,
    hex2rgb, TAB20, FOREGROUND, SCENES, BASE)

BACKGROUND = {"wall", "floor", "ceiling"}


def load_semantic(base):
    d = np.load(f"{base}/semantic_cloud_10m/semantic_cloud.npz", allow_pickle=True)
    pts = d["pts"].astype(np.float64)
    label = d["label"].astype(np.int32)
    prompts = [str(p) for p in d["prompts"]]
    return pts, label, prompts


def project_frame(pts, label, w2c, K, W, H):
    """z-buffer 投影全量点，返回 (cat_map[H,W] int32, surf_idx[H,W] int64)。"""
    ph = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
    cam = (w2c @ ph.T).T
    z = cam[:, 2]
    front = z > 0.05
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * cam[:, 0] / np.maximum(z, 1e-6) + cx
    v = fy * cam[:, 1] / np.maximum(z, 1e-6) + cy
    ui = np.floor(u).astype(np.int64)
    vi = np.floor(v).astype(np.int64)
    inb = front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    idx = np.nonzero(inb)[0]
    flat = vi[idx] * W + ui[idx]
    zi = z[idx]
    order = np.lexsort((zi, flat))
    sflat = flat[order]
    first = np.empty(len(sflat), dtype=bool)
    first[0] = True
    if len(first) > 1:
        first[1:] = sflat[1:] != sflat[:-1]
    surf = idx[order][first]
    pix = sflat[first]
    flat_cat = np.full(H * W, -1, np.int32)
    flat_cat[pix] = label[surf]
    flat_surf = np.full(H * W, -1, np.int64)
    flat_surf[pix] = surf
    return flat_cat.reshape(H, W), flat_surf.reshape(H, W)


def frame_2d_instances(cat_map, surf_idx, fg_cats, min_area, close_kernel):
    """对一帧的 2D 语义图按前景类别做连通域 → 2D 实例列表。"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    entries = []
    for c in fg_cats:
        binmap = (cat_map == c).astype(np.uint8)
        if int(binmap.sum()) < min_area:
            continue
        binmap = cv2.morphologyEx(binmap, cv2.MORPH_CLOSE, kernel)
        n_lab, lab = cv2.connectedComponents(binmap, connectivity=8)
        for comp in range(1, n_lab):
            m = lab == comp
            if int(m.sum()) < min_area:
                continue
            pid = surf_idx[m]
            pid = pid[pid >= 0]
            if len(pid) < 20:
                continue
            entries.append({"cat": int(c), "mask": m, "pidx": pid})
    return entries


def nn_count_from_trees(trees, ptarr, a, b, thresh):
    """较小实例的点落在较大实例 thresh 内的个数（最近邻 3D 距离补救）。"""
    if thresh <= 0:
        return 0
    if len(ptarr[a]) <= len(ptarr[b]):
        d, _ = trees[b].query(ptarr[a], k=1)
    else:
        d, _ = trees[a].query(ptarr[b], k=1)
    return int((d <= thresh).sum())


def associate(entries, pts, dist_thresh, min_overlap, nn_thresh, min_nn, prompts):
    """跨帧关联：同类别 + (质心近 或 共享点多 或 最近邻距离近) → union-find。"""
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

    cents = np.array([pts[e["pidx"]].mean(0) for e in entries])
    sets = [set(e["pidx"].tolist()) for e in entries]
    ptarr = [pts[e["pidx"]] for e in entries]
    trees = [cKDTree(p) for p in ptarr] if nn_thresh > 0 else None

    for a in range(n):
        for b in range(a + 1, n):
            if entries[a]["cat"] != entries[b]["cat"]:
                continue
            if entries[a]["fi"] == entries[b]["fi"]:
                continue
            d = float(np.linalg.norm(cents[a] - cents[b]))
            ov = len(sets[a] & sets[b])
            nn = nn_count_from_trees(trees, ptarr, a, b, nn_thresh) if trees else 0
            if d < dist_thresh or ov >= min_overlap or nn >= min_nn:
                if nn >= min_nn and d >= dist_thresh and ov < min_overlap:
                    print(f"      [nn-saved] {prompts[entries[a]['cat']]} "
                          f"nn={nn} d={d:.2f}m ov={ov}")
                union(a, b)

    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)

    instances_3d = []
    for members in comps.values():
        cat = Counter(entries[m]["cat"] for m in members).most_common(1)[0][0]
        per_frame = {entries[m]["fi"]: {"mask": entries[m]["mask"],
                                        "pidx": entries[m]["pidx"]}
                     for m in members}
        instances_3d.append({"cat": cat, "per_frame": per_frame})
    return instances_3d


def compute_entries(base, stage3, sem, pts, label, prompts, fg_cats, args):
    """选帧 + 逐帧投影 2D 聚类，返回 (win, entries)。"""
    # 选帧：每个前景类别一个伪实例（质心=类别点均值）
    pseudo = []
    for c in fg_cats:
        m = label == c
        if int(m.sum()) < 100:
            continue
        pseudo.append({"centroid": pts[m].mean(0), "name": prompts[c]})
    img_W, img_H = int(stage3["img_W"]), int(stage3["img_H"])
    kf_indices = [int(x) for x in sem["kf_indices"].tolist()]
    K = len(kf_indices)
    vis = np.zeros((K, len(pseudo)), dtype=bool)
    blur = np.zeros(K, dtype=np.float64)
    for k, fi in enumerate(kf_indices):
        vm, _, _ = visible_mask(pseudo, stage3[f"w2c_{fi}"].astype(np.float64),
                                stage3[f"K_{fi}"].astype(np.float64), img_W, img_H)
        vis[k] = vm
        g = cv2.imread(str(stage3[f"img_path_{fi}"]), cv2.IMREAD_GRAYSCALE)
        blur[k] = cv2.Laplacian(g, cv2.CV_64F).var()
    _, _, win_pos, _, _ = pick_window(vis, blur)
    win = [kf_indices[p] for p in win_pos]
    print(f"[2dproj] {base.split('/')[-1]}: local kf={win[:3]}, loop kf={win[3:]}")

    entries = []
    for fi in win:
        w2c = stage3[f"w2c_{fi}"].astype(np.float64)
        Kcam = stage3[f"K_{fi}"].astype(np.float64)
        cat_map, surf_idx = project_frame(pts, label, w2c, Kcam, img_W, img_H)
        fe = frame_2d_instances(cat_map, surf_idx, fg_cats, args.min_area, args.close_kernel)
        for e in fe:
            e["fi"] = fi
        entries.extend(fe)
        print(f"[2dproj] kf{fi}: {len(fe)} 2D instances "
              f"{dict(Counter(prompts[e['cat']] for e in fe))}")

    # 关联前诊断：每类别 2D 实例（跨帧）的 3D 质心空间范围
    pre = {}
    for e in entries:
        pre.setdefault(prompts[e["cat"]], []).append(pts[e["pidx"]].mean(0))
    for nm, cs in pre.items():
        cs = np.array(cs)
        if len(cs) < 2:
            continue
        mx = max(np.linalg.norm(a - b) for a, b in combinations(cs, 2))
        print(f"   [pre] {nm:<10} x{len(cs)} spread={mx:.2f}m")
    return win, entries


def select_persistent(instances_3d, win, pts, prompts, max_inst):
    local_set = set(win[:3]); loop_set = set(win[3:])
    keep = []
    for it in instances_3d:
        name = prompts[it["cat"]]
        if name in BACKGROUND:
            continue
        pf = it["per_frame"]
        nl = sum(1 for fi in pf if fi in local_set)
        nlp = sum(1 for fi in pf if fi in loop_set)
        if nl >= 2 and nlp >= 1:
            allp = np.concatenate([pf_["pidx"] for pf_ in pf.values()])
            keep.append({"cat": it["cat"], "name": name, "per_frame": pf,
                         "n_pts": int(len(np.unique(allp))),
                         "n_local": nl, "n_loop": nlp})
    keep.sort(key=lambda x: -x["n_pts"])
    keep = keep[:max_inst]
    for i, it in enumerate(keep):
        it["id"] = i
    return keep


def print_diag(instances_3d, pts, prompts, tag):
    by_name = {}
    for it in instances_3d:
        allp = np.concatenate([pf["pidx"] for pf in it["per_frame"].values()])
        by_name.setdefault(prompts[it["cat"]], []).append(pts[allp].mean(0))
    print(f"   [{tag}] {len(instances_3d)} 3D instances")
    for nm, cs in by_name.items():
        if len(cs) < 2:
            continue
        cs = np.array(cs)
        ds = sorted(np.linalg.norm(a - b) for a, b in combinations(cs, 2))
        print(f"      {nm:<10} x{len(cs)} min_pair={ds[0]:.2f}m "
              f"({', '.join(f'{d:.2f}' for d in ds[:5])})")


def draw_row(axes, scene, win, keep, stage3, prompts, row_title, is_top):
    axes[0].set_ylabel(f"{scene}\n{row_title}\n{len(keep)} persistent",
                       fontsize=11, rotation=0, labelpad=30,
                       va="center", color="#272727", fontweight="bold")
    for c, fi in enumerate(win):
        ax = axes[c]
        img = cv2.imread(str(stage3[f"img_path_{fi}"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        for it in keep:
            if fi not in it["per_frame"]:
                continue
            imask = it["per_frame"][fi]["mask"]
            colrgb = hex2rgb(TAB20[it["id"] % len(TAB20)])
            img = draw_instance(img, imask, colrgb)
        for it in keep:
            if fi not in it["per_frame"]:
                continue
            imask = it["per_frame"][fi]["mask"]
            colrgb = hex2rgb(TAB20[it["id"] % len(TAB20)])
            lu, lv = mask_label_pos(imask, W, H, 0, 0)
            draw_label(img, f"#{it['id']} {it['name']}", lu, lv, colrgb, W, H)
        ax.imshow(img)
        if is_top:
            ax.set_title(f"kf {fi}" if c < 3 else f"loop {fi}", fontsize=11, color="#272727")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=["scene0000 (bedroom)"])
    ap.add_argument("--min_area", type=int, default=300)
    ap.add_argument("--close_kernel", type=int, default=5)
    ap.add_argument("--dist_thresh", type=float, default=0.25)
    ap.add_argument("--min_overlap", type=int, default=10)
    ap.add_argument("--nn_thresh", type=float, default=0.12)
    ap.add_argument("--min_nn", type=int, default=15)
    ap.add_argument("--max_inst", type=int, default=25)
    ap.add_argument("--out", default="/root/autodl-tmp/scannet_figs_all/fig_instance_grid_2dproj_compare")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    scene_keys = [k for k in SCENES if k.split()[0] in args.scenes or k in args.scenes]
    if not scene_keys:
        scene_keys = [k for k in SCENES if any(s in k for s in args.scenes)]
    if not scene_keys:
        scene_keys = list(SCENES.keys())
    print(f"[2dproj] scenes = {scene_keys}")

    n_frames = 5
    n_rows = 2  # 上行 = 无补救，下行 = 加最近邻补救
    fig, axes = plt.subplots(n_rows, n_frames, figsize=(2.35 * n_frames, 3.1 * n_rows))
    for r in range(n_rows):
        for c in range(n_frames):
            axes[r, c].axis("off")

    for scene in scene_keys:
        name, s3_rel, sem_rel, _ = SCENES[scene]
        base = f"{BASE}/{name}"
        stage3 = np.load(f"{base}/{s3_rel}", allow_pickle=True)
        sem = np.load(f"{base}/{sem_rel}", allow_pickle=True)
        pts, label, prompts = load_semantic(base)
        print(f"\n[2dproj] {scene}: {len(pts):,} points")

        fg_cats = [c for c in range(len(prompts))
                   if prompts[c] in FOREGROUND and prompts[c] not in BACKGROUND]
        win, entries = compute_entries(base, stage3, sem, pts, label, prompts, fg_cats, args)
        print(f"[2dproj] {scene}: {len(entries)} 2D entries")

        # 上行：无最近邻补救
        print(f"\n[2dproj] {scene} -- variant A (w/o nn remedy) --")
        inst_no = associate(entries, pts, args.dist_thresh, args.min_overlap, 0.0, args.min_nn, prompts)
        print_diag(inst_no, pts, prompts, "A")
        keep_no = select_persistent(inst_no, win, pts, prompts, args.max_inst)
        print(f"   [A] {len(keep_no)} persistent = "
              f"{[(it['id'], it['name'], it['n_pts']) for it in keep_no]}")

        # 下行：加最近邻补救
        print(f"\n[2dproj] {scene} -- variant B (+ nn remedy) --")
        inst_nn = associate(entries, pts, args.dist_thresh, args.min_overlap, args.nn_thresh, args.min_nn, prompts)
        print_diag(inst_nn, pts, prompts, "B")
        keep_nn = select_persistent(inst_nn, win, pts, prompts, args.max_inst)
        print(f"   [B] {len(keep_nn)} persistent = "
              f"{[(it['id'], it['name'], it['n_pts']) for it in keep_nn]}")

        draw_row(axes[0], scene, win, keep_no, stage3, prompts, "w/o nn remedy", is_top=True)
        draw_row(axes[1], scene, win, keep_nn, stage3, prompts, "+ nn remedy", is_top=False)

    fig.suptitle(
        "Persistent instance identity on ScanNet — per-frame 2D projected clustering + cross-frame 3D association\n"
        "top row: centroid + shared-point association only;  bottom row: + same-category nearest-neighbor 3D distance remedy",
        fontsize=14, y=1.02)
    finalize_figure(fig, args.out)
    print(f"[2dproj] -> {args.out}.png")


if __name__ == "__main__":
    main()
