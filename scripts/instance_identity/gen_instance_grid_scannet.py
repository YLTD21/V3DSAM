#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ScanNet 版持久实例身份图（3 场景 x 5 帧）——真实实例点投影掩膜（v3）。

与旧版（语义类别连通域当实例掩膜）的差异：
  1. 掩膜不再用「类别标签图的连通域」近似，而是把 instance_cloud.ply 里该实例的
     真实 3D 点（逐点 instance_id，与 instances.json 的 id 一一对应）投影到每帧，
     膨胀成 2D 掩膜——掩膜 = 该实例真实可见区域，跨帧天然一致（同 id 同点集）。
  2. 每个实例画掩膜边界（轮廓）+ 极轻填充，标签 "#id 类别" 放在掩膜顶端中心。
  3. 颜色按 instance id 分配（40 色调色板取模），同一物理物体在 3 个局部帧 +
     2 个回环帧里保持同样颜色 + #id + 类别，作为持久实例身份的证据。

数据源：_stage3_input_after.npz（w2c/K/img_path）+ kf_semantic.npz（仅 kf_indices 选帧）
        + semantic_instances/{instances,voted_instances}.json + instance_cloud.ply。

Run on AutoDL (sgs env):  python gen_instance_grid_scannet.py
"""
import os, json, argparse
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sci_style import apply_publication_style, finalize_figure
    apply_publication_style(font_size=8, axes_linewidth=1)
except Exception:
    def finalize_figure(fig, out):
        fig.savefig(out + ".png", dpi=200, bbox_inches="tight", facecolor="white")
        fig.savefig(out + ".pdf", bbox_inches="tight")
        print(f"-> {out}.png / .pdf")


FOREGROUND = {"chair","desk","table","television","monitor","plant","picture",
              "cabinet","lamp","keyboard","laptop","sofa","couch","bed","dresser",
              "night stand","bookshelf",
              "book","bottle","cup","phone","mouse","clock","shelf","bin","bag",
              "pillow","vase","box","backpack","bowl","plate","mirror","sink",
              "counter","stairs","door","window","curtain","refrigerator","tv"}

# scene -> (name, stage3_rel, semantic_rel, instance_dir_rel)
BASE = "/root/autodl-tmp/ablation/scannet"
SCENES = {
    "scene0000 (bedroom)": ("scene0000_00", "_stage3_input_after.npz", "kf_semantic.npz", "semantic_instances"),
    "scene0059 (living)":  ("scene0059_00", "_stage3_input_after.npz", "kf_semantic.npz", "semantic_instances"),
    "scene0106 (living)":  ("scene0106_00", "_stage3_input_after.npz", "kf_semantic.npz", "semantic_instances"),
    "scene0207 (bathroom)": ("scene0207_00", "_stage3_input_after.npz", "kf_semantic.npz", "semantic_instances"),
}

# 40 个高区分度颜色（tab20 + tab20b），按 instance id 取模 -> 同一实例跨帧同色
TAB20 = ["#1f77b4","#aec7e8","#ff7f0e","#ffbb78","#2ca02c","#98df8a","#d62728",
         "#ff9896","#9467bd","#c5b0d5","#8c564b","#c49c94","#e377c2","#f7b6d2",
         "#7f7f7f","#c7c7c7","#bcbd22","#dbdb8d","#17becf","#9edae5",
         "#393b79","#5254a3","#6b6ecf","#9c9ede","#637939","#8ca252","#b5cf6b",
         "#cedb9c","#8c6d31","#bd9e39","#e7ba52","#e7cb94","#843c39","#ad494a",
         "#d6616b","#e7969c","#7b4173","#a55194","#ce6dbd","#de9ed6"]


def hex2rgb(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def project_pixels(pts, w2c, K):
    ph = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
    cam = (w2c @ ph.T).T
    z = cam[:, 2]
    front = z > 0.05
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = np.where(front, fx * cam[:, 0] / np.maximum(z, 1e-6) + cx, -9999.0)
    v = np.where(front, fy * cam[:, 1] / np.maximum(z, 1e-6) + cy, -9999.0)
    return u, v, z, front


def load_objs(base, inst_dir):
    inst = json.load(open(f"{base}/{inst_dir}/instances.json"))
    voted = json.load(open(f"{base}/{inst_dir}/voted_instances.json"))
    prompts = [str(p) for p in inst["prompts"]]
    vc = {int(o["id"]): int(o["voted_class"]) for o in voted["instances"]}
    objs = []
    for o in inst["instances"]:
        i = int(o["id"])
        if i not in vc:
            continue
        c = int(vc[i])
        if prompts[c] not in FOREGROUND:
            continue
        if int(o.get("n_vis", 0)) < 3:
            continue
        objs.append({"id": i, "class": c, "name": prompts[c],
                     "centroid": np.array(o["centroid"], np.float64),
                     "n_vis": int(o.get("n_vis", 0))})
    objs.sort(key=lambda o: -o["n_vis"])
    return objs


def load_instance_cloud(base):
    """读 instance_cloud.ply（100万点，逐点 instance_id，与 instances.json 的 id 一一对应）。"""
    pts, ids = [], []
    with open(f"{base}/semantic_instances/instance_cloud.ply") as f:
        for i, line in enumerate(f):
            if i < 11:  # 跳过 ply 头（11 行）
                continue
            p = line.split()
            pts.append((float(p[0]), float(p[1]), float(p[2])))
            ids.append(int(p[6]))
    return np.array(pts, np.float64), np.array(ids, np.int64)


def visible_mask(objs, w2c, K, W, H):
    if not objs:
        return np.zeros(0, dtype=bool), np.zeros(0), np.zeros(0)
    u, v, z, front = project_pixels(np.array([o["centroid"] for o in objs]), w2c, K)
    return front & (u >= 0) & (u < W) & (v >= 0) & (v < H), u, v


def instance_mask_points(pts, w2c, K, W, H, min_px=40, dilate=3):
    """把实例的真实 3D 点（instance_cloud.ply 中 instance_id 对应的点）投影到该帧，
    得到 2D 掩膜。这样掩膜 = 该实例真实可见区域，跨帧天然一致（同 instance_id 同点集）。
    返回 (imask, visible)：imask=None 表示该实例在此帧投影点过少（视为不可见）。"""
    u, v, z, front = project_pixels(pts, w2c, K)
    ok = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if ok.sum() < min_px:
        return None, False
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[v[ok].astype(np.int64), u[ok].astype(np.int64)] = 1
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=dilate)
    return mask.astype(bool), True


def draw_instance(img, imask, colrgb):
    """掩膜边界：极轻填充 + 粗轮廓（标准实例分割可视化样式）。"""
    overlay = img.astype(np.float32)
    overlay[imask] = 0.85 * overlay[imask] + 0.15 * np.array(colrgb, np.float32)
    img = np.clip(overlay, 0, 255).astype(np.uint8)
    cnts, _ = cv2.findContours(imask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, colrgb, 3, cv2.LINE_AA)
    return img


def mask_label_pos(imask, W, H, fallback_u, fallback_v):
    """标签锚点 = 掩膜顶端中心（沿掩膜上边界），避免堆叠物体的标签重叠。"""
    ys, xs = np.nonzero(imask)
    if len(ys) == 0:
        return fallback_u, fallback_v
    ytop = int(ys.min())
    xrow = xs[ys == ytop]
    return int(xrow.mean()), ytop


def draw_label(img, text, cu, cv_, colrgb, W, H):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 1.7, 3
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    tx = cu + 16
    ty = cv_ - 10  # 放在掩膜顶端之上
    x0, y0 = tx - 4, ty - th - 4
    x1, y1 = tx + tw + 4, ty + base + 4
    if x1 > W - 2:
        x0 -= (x1 - (W - 2)); x1 = W - 2
    if x0 < 2:
        x0 = 2
    if y0 < 2:
        y0 = 2; y1 = y0 + (th + base + 8)
    if y1 > H - 2:
        y1 = H - 2; y0 = y1 - (th + base + 8)
    tx, ty = x0 + 4, y0 + th + 4
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0), (x1, y1), colrgb, 3, cv2.LINE_AA)
    cv2.putText(img, text, (tx, ty), font, scale, (0, 0, 0), thick, cv2.LINE_AA)


def pick_window(vis, blur=None, n_local=3, n_loop=2, gap_min=None):
    K = vis.shape[0]
    if blur is None:
        blur = np.zeros(K)
    blur = np.asarray(blur, dtype=np.float64)
    if gap_min is None:
        gap_min = max(10, int(0.20 * K))
    best = None
    for s in range(K - n_local + 1):
        vis_local = vis[s:s + n_local].sum(axis=0)
        local_objs = vis_local >= 2
        if local_objs.sum() == 0:
            continue
        scores = np.zeros(K, dtype=int)
        for k in range(K):
            if s - gap_min < k < s + n_local - 1 + gap_min:
                continue
            scores[k] = int((vis[k] & local_objs).sum())
        # loop 帧：可见性得分优先，同分按清晰度（Laplacian 方差）降序
        cand = [k for k in range(K) if scores[k] > 0]
        cand.sort(key=lambda k: (-int(scores[k]), -float(blur[k])))
        loop = cand[:n_loop]
        if len(loop) < n_loop:
            continue
        vis_loop = vis[loop].sum(axis=0)
        n_persist = int((local_objs & (vis_loop >= 1)).sum())
        clarity = float(np.mean(blur[s:s + n_local])) + float(np.mean(blur[loop]))
        if best is None or n_persist > best[0] or (n_persist == best[0] and clarity > best[1]):
            win_pos = list(range(s, s + n_local)) + loop
            best = (n_persist, clarity, win_pos, vis_local, vis_loop)
    if best is None:
        s = 0
        vis_local = vis[s:s + n_local].sum(axis=0)
        vis_loop = vis[s + n_local:s + n_local + n_loop].sum(axis=0)
        win_pos = list(range(s, s + n_local + n_loop))
        best = (int(((vis_local >= 2) & (vis_loop >= 1)).sum()), 0.0, win_pos, vis_local, vis_loop)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_frames", type=int, default=5)
    ap.add_argument("--out", default="/root/autodl-tmp/scannet_figs_all/fig_instance_grid_scannet")
    ap.add_argument("--max_objs", type=int, default=25)
    args = ap.parse_args()

    scenes = list(SCENES.keys())
    n = args.n_frames
    fig, axes = plt.subplots(len(scenes), n, figsize=(2.35 * n, 3.1 * len(scenes)))
    if len(scenes) == 1:
        axes = axes[None, :]
    for r in range(axes.shape[0]):
        for c in range(axes.shape[1]):
            axes[r, c].axis("off")

    scene_metrics = {}
    for r, scene in enumerate(scenes):
        name, s3_rel, sem_rel, inst_rel = SCENES[scene]
        base = f"{BASE}/{name}"
        stage3 = np.load(f"{base}/{s3_rel}", allow_pickle=True)
        sem = np.load(f"{base}/{sem_rel}", allow_pickle=True)
        kf_indices = [int(x) for x in sem["kf_indices"].tolist()]
        img_W, img_H = int(stage3["img_W"]), int(stage3["img_H"])
        objs = load_objs(base, inst_rel)
        objs = objs[:args.max_objs]
        print(f"[grid] {scene}: {len(objs)} foreground instances, {len(kf_indices)} keyframes")

        # 真实逐点实例分割（instance_cloud.ply）按 instance id 建点集
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

        score, clarity, win_pos, vis_local, vis_loop = pick_window(vis, blur)
        win = [kf_indices[p] for p in win_pos]

        # 在选中的 5 帧里，逐实例把真实 3D 点投影成 2D 掩膜（一次计算，供可见性判定与绘制共用）
        masks = {}
        for c, fi in enumerate(win):
            w2c = stage3[f"w2c_{fi}"].astype(np.float64)
            Kcam = stage3[f"K_{fi}"].astype(np.float64)
            for j in range(len(objs)):
                imask, _ = instance_mask_points(id2pts[objs[j]["id"]], w2c, Kcam, img_W, img_H)
                masks[(c, j)] = imask

        # 跨回环持久实例：≥2 个 local 帧 + ≥1 个 loop 帧都有可见点
        sel = [j for j in range(len(objs))
               if sum(1 for c in range(3) if masks[(c, j)] is not None) >= 2
               and sum(1 for c in range(3, 5) if masks[(c, j)] is not None) >= 1]
        print(f"[grid] {scene}: local kf={win[:3]}, loop kf={win[3:]}, "
              f"n_persist={score}, clarity={clarity:.1f}, selected {len(sel)}/{len(objs)} objs "
              f"= {[(objs[i]['id'], objs[i]['name']) for i in sel]}")

        # 质心-掩膜命中：实例质心投影是否落在自身投影点掩膜内（几何自一致性）
        hits = {}
        for c, fi in enumerate(win):
            w2c = stage3[f"w2c_{fi}"].astype(np.float64)
            Kcam = stage3[f"K_{fi}"].astype(np.float64)
            uc, vc, _, front_c = project_pixels(
                np.array([o["centroid"] for o in objs]), w2c, Kcam)
            for j in range(len(objs)):
                imask = masks[(c, j)]
                if imask is None:
                    hits[(c, j)] = False
                    continue
                inb = (bool(front_c[j]) and 0 <= uc[j] < img_W and 0 <= vc[j] < img_H)
                hits[(c, j)] = inb and bool(imask[int(vc[j]), int(uc[j])])

        vis_pairs = [(c, j) for c in range(5) for j in sel if masks[(c, j)] is not None]
        hit_rate = (sum(1 for (c, j) in vis_pairs if hits[(c, j)]) / len(vis_pairs)
                    if vis_pairs else 0.0)
        consistent = sum(1 for j in sel
                         if any(hits[(c, j)] for c in range(3) if masks[(c, j)] is not None)
                         and any(hits[(c, j)] for c in range(3, 5) if masks[(c, j)] is not None))
        id_consistency = consistent / len(sel) if sel else 0.0
        scene_metrics[scene] = (hit_rate, id_consistency)
        print(f"[grid] {scene}: centroid-mask hit={hit_rate:.1%}, "
              f"cross-loop id consistency={id_consistency:.1%} ({consistent}/{len(sel)})")

        axes[r, 0].set_ylabel(f"{scene}\n{len(sel)}/{len(objs)} tracked\n"
                              f"hit {hit_rate:.0%} / loop {id_consistency:.0%}",
                              fontsize=11, rotation=0, labelpad=24,
                              va="center", color="#272727", fontweight="bold")

        for c, fi in enumerate(win):
            ax = axes[r, c]
            img = cv2.imread(str(stage3[f"img_path_{fi}"]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]

            # 先画掩膜边界（所有选中实例），再画标签（顶层）
            for j in sel:
                imask = masks[(c, j)]
                if imask is None:
                    continue
                colrgb = hex2rgb(TAB20[objs[j]["id"] % len(TAB20)])
                img = draw_instance(img, imask, colrgb)
            for j in sel:
                imask = masks[(c, j)]
                if imask is None:
                    continue
                colrgb = hex2rgb(TAB20[objs[j]["id"] % len(TAB20)])
                lu, lv = mask_label_pos(imask, W, H, 0, 0)
                draw_label(img, f"#{objs[j]['id']} {objs[j]['name']}", lu, lv, colrgb, W, H)

            ax.imshow(img)
            if r == 0:
                ax.set_title(f"kf {fi}" if c < 3 else f"loop {fi}", fontsize=11, color="#272727")

    metrics_line = "   |   ".join(
        f"{scene.split()[0]}: hit {h:.0%}, cross-loop {c:.0%}"
        for scene, (h, c) in scene_metrics.items())
    fig.suptitle(
        "Persistent instance identity on ScanNet: 3 scenes — 3 overlapping keyframes + 2 loop-closure frames\n"
        "(each instance = its projected 3D instance points; same object keeps the same colour + #id + category across the loop)\n"
        f"centroid-mask hit rate & cross-loop ID consistency — {metrics_line}",
        fontsize=14, y=1.02)

    finalize_figure(fig, args.out)
    print(f"[grid] -> {args.out}.png")


if __name__ == "__main__":
    main()
