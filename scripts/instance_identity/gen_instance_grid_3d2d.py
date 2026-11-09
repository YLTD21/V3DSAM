#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3×5 持久实例矩阵 —— 3D 聚类定 ID + stage2 语义 2D 区域做显示。

用户定方向：
  - ID：3D 点云聚类（语义点云按类别欧氏聚类）→ 几何定身份，跨视角一致。
  - 显示：stage2 已分割好的 2D 语义掩膜（kf_semantic.npz 的 labels 按类别连通域）
          → 干净的实例区域，不再是 3D 稀疏点投影成的"密密麻麻小点"。
  - 绑定：3D 实例点投影到 2D，找重叠最大的同类别连通域，把该实例 id 赋给它。

数据源：semantic_cloud_10m/semantic_cloud.npz（pts+label+prompts，与 kf_semantic
        labels 的类别索引完全一致）+ _stage3_input_after.npz（w2c/K/img_path）
        + kf_semantic.npz（labels 逐帧语义图，用于 2D 连通域显示）。

Run on AutoDL (sgs env):
  cd /root/autodl-tmp
  /root/miniconda3/envs/sgs/bin/python -u gen_instance_grid_3d2d.py \
    --scenes scene0000_00 --out /root/autodl-tmp/scannet_figs_all/fig_instance_grid_3d2d
"""
import os, argparse
from collections import Counter
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
    project_pixels, visible_mask, pick_window, draw_instance, draw_label,
    mask_label_pos, hex2rgb, TAB20, FOREGROUND, SCENES, BASE)

BACKGROUND = {"wall", "floor", "ceiling"}


def load_semantic(base):
    d = np.load(f"{base}/semantic_cloud_10m/semantic_cloud.npz", allow_pickle=True)
    pts = d["pts"].astype(np.float64)
    label = d["label"].astype(np.int32)
    prompts = [str(p) for p in d["prompts"]]
    return pts, label, prompts


def voxel_downsample(pts, label, voxel=0.02):
    origin = pts.min(axis=0)
    q = np.floor((pts - origin) / voxel).astype(np.int64)
    keys, inverse = np.unique(q, axis=0, return_inverse=True)
    n_vox = len(keys)
    cnt = np.bincount(inverse, minlength=n_vox).astype(np.float64)
    vpts = np.empty((n_vox, 3), np.float64)
    for d in range(3):
        vpts[:, d] = np.bincount(inverse, weights=pts[:, d], minlength=n_vox) / cnt
    lab_shift = label + 1
    n_lab = int(lab_shift.max()) + 1
    flat = inverse * n_lab + lab_shift
    cnt_lab = np.bincount(flat, minlength=n_vox * n_lab).reshape(n_vox, n_lab)
    non_unlab = cnt_lab[:, 1:]
    vlab = non_unlab.argmax(1).astype(np.int32)
    vlab[non_unlab.max(1) == 0] = -1
    return vpts, vlab


def cluster_category(sub_pts, eps, min_voxels):
    n = len(sub_pts)
    if n < min_voxels:
        return []
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree = cKDTree(sub_pts)
    pairs = tree.query_pairs(r=eps, output_type="ndarray")
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return [members for members in comps.values() if len(members) >= min_voxels]


def frame_2d_regions(lab_frame, fg_cats, min_area=300, close_kernel=5):
    """对一帧语义标签图按前景类别做 2D 连通域 → [{cat, mask}]。"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    regions = []
    for c in fg_cats:
        binmap = (lab_frame == c).astype(np.uint8)
        if int(binmap.sum()) < min_area:
            continue
        binmap = cv2.morphologyEx(binmap, cv2.MORPH_CLOSE, kernel)
        n_lab, lab = cv2.connectedComponents(binmap, connectivity=8)
        for comp in range(1, n_lab):
            m = lab == comp
            if int(m.sum()) < min_area:
                continue
            regions.append({"cat": int(c), "mask": m})
    return regions


def bind_instance(inst, regions, w2c, K, W, H, min_proj=30, min_ov=30):
    """3D 实例投影到帧，找重叠最大的同类别 2D 连通域，返回其 mask（或 None）。"""
    u, v, z, front = project_pixels(inst["points"], w2c, K)
    ok = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if int(ok.sum()) < min_proj:
        return None
    ui = u[ok].astype(np.int64)
    vi = v[ok].astype(np.int64)
    best_mask = None
    best_ov = min_ov - 1
    for r in regions:
        if r["cat"] != inst["class"]:
            continue
        ov = int(r["mask"][vi, ui].sum())
        if ov > best_ov:
            best_ov = ov
            best_mask = r["mask"]
    return best_mask if best_ov >= min_ov else None


def _rot_angle(R):
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def pick_window_se3(stage3, kf_indices, blur, vis, n_local=3, n_loop=2,
                    gap_min=None, pos_thr=0.6, ang_deg=35.0):
    """选 3 局部 + 2 回环：只在 SE(3) 真回环（位置+朝向都近）里挑，按质心可见
    持久实例数打分（与 pick_window 一致），清晰度做 tiebreaker。位置+朝向约束
    能排除假回环（同一位置反向视角，跨房间），避免 scene0207 选到另一间房。"""
    K = len(kf_indices)
    blur = np.asarray(blur, dtype=np.float64)
    cents = np.zeros((K, 3), dtype=np.float64)
    Rots = np.zeros((K, 3, 3), dtype=np.float64)
    for k, fi in enumerate(kf_indices):
        w2c = stage3[f"w2c_{fi}"].astype(np.float64)
        R, t = w2c[:3, :3], w2c[:3, 3]
        cents[k] = -R.T @ t
        Rots[k] = R
    if gap_min is None:
        gap_min = max(10, int(0.20 * K))
    ang_thr = np.deg2rad(ang_deg)

    best = None  # (n_persist, clarity, win_pos)
    for i in range(K):
        best_j = None  # (pos, j)
        for j in range(K):
            if abs(i - j) < gap_min:
                continue
            dp = float(np.linalg.norm(cents[i] - cents[j]))
            if dp >= pos_thr:
                continue
            if _rot_angle(Rots[j] @ Rots[i].T) >= ang_thr:
                continue
            if best_j is None or dp < best_j[0]:
                best_j = (dp, j)
        if best_j is None:
            continue
        j = best_j[1]
        s = min(max(i - 1, 0), K - n_local)
        local_pos = list(range(s, s + n_local))
        l0 = min(max(j - 1, 0), K - n_loop)
        loop_pos = list(range(l0, l0 + n_loop))
        if set(local_pos) & set(loop_pos):
            continue
        vis_local = vis[local_pos].sum(axis=0)
        local_objs = vis_local >= 2
        vis_loop = vis[loop_pos].sum(axis=0)
        n_persist = int((local_objs & (vis_loop >= 1)).sum())
        clarity = float(np.mean(blur[local_pos]) + np.mean(blur[loop_pos]))
        if best is None or n_persist > best[0] or (n_persist == best[0] and clarity > best[1]):
            best = (n_persist, clarity, local_pos + loop_pos)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=["scene0000_00"])
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--min_voxels", type=int, default=100)
    ap.add_argument("--min_area", type=int, default=300)
    ap.add_argument("--close_kernel", type=int, default=5)
    ap.add_argument("--max_inst", type=int, default=25)
    ap.add_argument("--out", default="/root/autodl-tmp/scannet_figs_all/fig_instance_grid_3d2d")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    scene_keys = [k for k in SCENES if SCENES[k][0] in args.scenes]
    if not scene_keys:
        scene_keys = list(SCENES.keys())
    print(f"[3d2d] scenes = {scene_keys}")

    n_frames = 5
    fig, axes = plt.subplots(len(scene_keys), n_frames,
                             figsize=(2.5 * n_frames, 1.95 * len(scene_keys)))
    if len(scene_keys) == 1:
        axes = axes[None, :]
    fig.subplots_adjust(left=0.04, right=0.99, top=0.90, bottom=0.02,
                        wspace=0.02, hspace=0.05)
    for r in range(axes.shape[0]):
        for c in range(axes.shape[1]):
            axes[r, c].axis("off")

    for r, scene in enumerate(scene_keys):
        name, s3_rel, sem_rel, _ = SCENES[scene]
        base = f"{BASE}/{name}"
        stage3 = np.load(f"{base}/{s3_rel}", allow_pickle=True)
        sem = np.load(f"{base}/{sem_rel}", allow_pickle=True)
        kf_indices = [int(x) for x in sem["kf_indices"].tolist()]
        img_W, img_H = int(stage3["img_W"]), int(stage3["img_H"])
        labels_all = sem["labels"]  # (K, H, W)

        pts, label, prompts = load_semantic(base)
        print(f"\n[3d2d] {scene}: {len(pts):,} points, {len(prompts)} cats")

        vpts, vlab = voxel_downsample(pts, label, args.voxel)
        print(f"[3d2d] voxel {args.voxel:.2f}m -> {len(vpts):,} voxel pts")

        fg_cats = [c for c in range(len(prompts))
                   if prompts[c] in FOREGROUND and prompts[c] not in BACKGROUND]

        # 3D 聚类 → 实例（ID 在此定）
        instances = []
        for c in fg_cats:
            members = np.nonzero(vlab == c)[0]
            if members.size < args.min_voxels:
                continue
            clusters = cluster_category(vpts[members], args.eps, args.min_voxels)
            for cl in clusters:
                idx = members[cl]
                inst_pts = vpts[idx]
                instances.append({
                    "id": len(instances),
                    "class": c,
                    "name": prompts[c],
                    "centroid": inst_pts.mean(0),
                    "points": inst_pts,
                    "n_voxels": int(len(idx)),
                })
        instances.sort(key=lambda o: -o["n_voxels"])
        instances = instances[:args.max_inst]
        print(f"[3d2d] {len(instances)} 3D instances: "
              f"{dict(Counter(o['name'] for o in instances))}")

        # 选帧
        K = len(kf_indices)
        vis = np.zeros((K, len(instances)), dtype=bool)
        blur = np.zeros(K, dtype=np.float64)
        for k, fi in enumerate(kf_indices):
            m, _, _ = visible_mask(instances, stage3[f"w2c_{fi}"].astype(np.float64),
                                   stage3[f"K_{fi}"].astype(np.float64), img_W, img_H)
            vis[k] = m
            g = cv2.imread(str(stage3[f"img_path_{fi}"]), cv2.IMREAD_GRAYSCALE)
            blur[k] = cv2.Laplacian(g, cv2.CV_64F).var()
        # 两个候选窗口：质心可见性（pick_window）+ SE(3) 真回环（pick_window_se3）。
        # 各自算绑定，取持久实例更多的那个 —— 既保住 3 个老场景的高持久数，
        # 又能修 scene0207（质心可见性会选到假回环/跨房间，SE(3) 能选到真回环）。
        _, _, wp_vis, _, _ = pick_window(vis, blur)
        res_se3 = pick_window_se3(stage3, kf_indices, blur, vis)
        candidates = [("centroid-vis", wp_vis)]
        if res_se3 is not None and res_se3[2] != wp_vis:
            candidates.append(("SE3", res_se3[2]))

        def bind_for(win_pos):
            win = [kf_indices[p] for p in win_pos]
            b = {}
            for c, (p, fi) in enumerate(zip(win_pos, win)):
                lab_frame = labels_all[p].astype(np.int32)
                H, W = lab_frame.shape
                regions = frame_2d_regions(lab_frame, fg_cats, args.min_area, args.close_kernel)
                w2c = stage3[f"w2c_{fi}"].astype(np.float64)
                Kcam = stage3[f"K_{fi}"].astype(np.float64)
                for j, inst in enumerate(instances):
                    b[(c, j)] = bind_instance(inst, regions, w2c, Kcam, W, H)
            sel = [j for j in range(len(instances))
                   if sum(1 for c in range(3) if b[(c, j)] is not None) >= 2
                   and sum(1 for c in range(3, 5) if b[(c, j)] is not None) >= 1]
            return win, b, sel

        best = None
        for src, wp in candidates:
            win, b, sel = bind_for(wp)
            if best is None or len(sel) > len(best[2]):
                best = (win, b, sel, src)
        win, bound, sel, loop_src = best
        print(f"[3d2d] {scene}: local={win[:3]} loop={win[3:]} ({loop_src})")
        print(f"[3d2d] {scene}: {len(sel)} persistent = "
              f"{[(instances[j]['id'], instances[j]['name']) for j in sel]}")

        axes[r, 0].set_ylabel(f"{scene}\n{len(sel)} persistent",
                              fontsize=10, rotation=0, labelpad=10,
                              va="center", color="#272727", fontweight="bold")

        for c, fi in enumerate(win):
            ax = axes[r, c]
            img = cv2.imread(str(stage3[f"img_path_{fi}"]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]
            for j in sel:
                imask = bound[(c, j)]
                if imask is None:
                    continue
                colrgb = hex2rgb(TAB20[instances[j]["id"] % len(TAB20)])
                img = draw_instance(img, imask, colrgb)
            for j in sel:
                imask = bound[(c, j)]
                if imask is None:
                    continue
                colrgb = hex2rgb(TAB20[instances[j]["id"] % len(TAB20)])
                lu, lv = mask_label_pos(imask, W, H, 0, 0)
                draw_label(img, f"#{instances[j]['id']} {instances[j]['name']}",
                           lu, lv, colrgb, W, H)
            ax.imshow(img)
            if r == 0:
                ax.set_title(f"kf {fi}" if c < 3 else f"loop {fi}", fontsize=11, color="#272727")

    fig.suptitle(
        "Persistent instance identity on ScanNet — 3D-clustered ID + stage-2 semantic 2D regions\n"
        "(instance ID from per-category Euclidean clustering of the semantic cloud; "
        "display region = matched 2D semantic connected component)",
        fontsize=14, y=1.02)
    finalize_figure(fig, args.out)
    print(f"[3d2d] -> {args.out}.png")


if __name__ == "__main__":
    main()
