#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Table 4-3 共享工具：前景类映射、投影、GT/方法掩膜生成、指标计算。"""
import numpy as np, cv2

from scannet_labels import NYU40

# NYU40 可计数前景物体类（实例身份对象域）
FOREGROUND_NYU = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18,
                  22, 23, 24, 28, 31, 33, 34, 36}
FG_LIST = sorted(FOREGROUND_NYU)

# 开放词汇 prompt -> NYU40 id（PROMPT2NYU + FOREGROUND 里未覆盖的补全）
PROMPT2NYU = {
    "wall":0, "floor":1, "ceiling":21, "door":7, "window":8, "stairs":37, "railing":37,
    "bed":3, "cabinet":2, "bookshelf":9, "shelf":14, "counter":11, "desk":13, "table":6,
    "sofa":5, "couch":5, "sink":33, "toilet":32, "bathtub":35, "mirror":18, "lamp":34, "light":34,
    "television":24, "monitor":24, "laptop":39, "keyboard":39, "mouse":39, "phone":39,
    "refrigerator":23, "oven":38, "microwave":38, "chair":4, "plant":39, "vase":39,
    "picture":10, "painting":10, "curtain":15, "pillow":17, "clock":39, "book":22,
    "bottle":39, "cup":39, "bowl":39, "plate":39, "bag":36, "backpack":36, "box":28,
    "bin":39, "basket":39, "person":30,
    "dresser":16, "night stand":31, "shelves":14, "tv":24,
}


def project_pixels(pts, w2c, K):
    """3D 点 -> 相机平面 (u,v,z,front)。w2c 是 4x4 world2cam。"""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    ph = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
    cam = (w2c @ ph.T).T
    z = cam[:, 2]
    front = z > 0.05
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = np.where(front, fx * cam[:, 0] / np.maximum(z, 1e-6) + cx, -9999.0)
    v = np.where(front, fy * cam[:, 1] / np.maximum(z, 1e-6) + cy, -9999.0)
    return u, v, z, front


def convex_hull_mask(pts, w2c, K, W, H, min_px=15):
    """3D 点投影 -> 凸包填充掩膜（对稀疏点也稳定）。返回 bool mask 或 None。"""
    u, v, z, front = project_pixels(pts, w2c, K)
    ok = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if int(ok.sum()) < min_px:
        return None
    pts2d = np.stack([u[ok], v[ok]], axis=1).astype(np.int32)
    hull = cv2.convexHull(pts2d)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool)


def match_frame(gt_masks, inst_masks, inst_nyu, gt_nyu, cov_thr=0.3, use_class=True):
    """一帧内的匹配：gt_masks=[(gt_id, mask)]，inst_masks=[(inst_id, mask)]。
    one-to-one Hungarian 最大化 GT-normalized coverage S(g,p)=|g∩p|/|g|，
    仅接受 S>=cov_thr 的分配（低于阈 = 未匹配）。避免一个 predicted instance
    同时解释多个 GT。use_class=True 时只考虑同类实例；实例身份=几何时应设
    False（SAM3 开放词汇分类噪声会致同类硬过滤丢失大量匹配）。
    返回 {gt_id: inst_id or None}。"""
    from scipy.optimize import linear_sum_assignment
    n_gt, n_inst = len(gt_masks), len(inst_masks)
    res = {gid: None for gid, _ in gt_masks}
    if n_gt == 0 or n_inst == 0:
        return res
    S = np.zeros((n_gt, n_inst), dtype=np.float64)
    for i, (gid, gmask) in enumerate(gt_masks):
        area = int(gmask.sum())
        if area == 0:
            continue
        for j, (iid, imask) in enumerate(inst_masks):
            if use_class and inst_nyu.get(iid, -1) != gt_nyu.get(gid, -1):
                continue
            S[i, j] = int(np.logical_and(gmask, imask).sum()) / area
    row_ind, col_ind = linear_sum_assignment(-S)
    for i, j in zip(row_ind, col_ind):
        if S[i, j] >= cov_thr:
            res[gt_masks[i][0]] = inst_masks[j][0]
    return res


def match_frame_full(gt_masks, inst_masks, inst_nyu, gt_nyu, cov_thr=0.3, use_class=True):
    """同 match_frame，但额外返回 (inter, gt_area, inst_area) 供 IoU/MOTA/IDF1/4D-PQ 派生。

    inter[i,j] = |gt_i ∩ inst_j|（use_class=False 或同类时为真交集，否则 0）
    gt_area[i] = |gt_i|, inst_area[j] = |inst_j|
    IoU(i,j) = inter[i,j] / (gt_area[i] + inst_area[j] - inter[i,j])
    """
    from scipy.optimize import linear_sum_assignment
    n_gt, n_inst = len(gt_masks), len(inst_masks)
    inter = np.zeros((n_gt, n_inst), dtype=np.int64)
    gt_area = np.zeros(n_gt, dtype=np.int64)
    inst_area = np.zeros(n_inst, dtype=np.int64)
    for j, (iid, imask) in enumerate(inst_masks):
        inst_area[j] = int(imask.sum())
    S = np.zeros((n_gt, n_inst), dtype=np.float64)
    res = {gid: None for gid, _ in gt_masks}
    for i, (gid, gmask) in enumerate(gt_masks):
        a = int(gmask.sum())
        gt_area[i] = a
        if a == 0:
            continue
        for j, (iid, imask) in enumerate(inst_masks):
            if use_class and inst_nyu.get(iid, -1) != gt_nyu.get(gid, -1):
                continue
            n_inter = int(np.logical_and(gmask, imask).sum())
            inter[i, j] = n_inter
            S[i, j] = n_inter / a
    row_ind, col_ind = linear_sum_assignment(-S)
    for i, j in zip(row_ind, col_ind):
        if S[i, j] >= cov_thr:
            res[gt_masks[i][0]] = inst_masks[j][0]
    return res, inter, gt_area, inst_area


def compute_mota(n_vis, n_fp, n_fm, n_ids):
    """MOTA = 1 - (FP + FN + IDSW) / GT。GT=可见 GT 实例帧数(n_vis)；FN=fragmentation(n_fm)。"""
    gt = max(1, n_vis)
    return 1.0 - (n_fp + n_fm + n_ids) / gt


def compute_idf1(assoc, n_gt_dets, n_pred_dets):
    """IDF1 = 2·IDTP / (n_gt_dets + n_pred_dets)（Ristani 2016，轨迹级匈牙利最大化匹配数）。

    assoc: {(gt_id, pred_id): n_frames_with_IoU>thr}（跨序列累计）。
    返回 (idf1, idtp)。
    """
    from scipy.optimize import linear_sum_assignment
    if not assoc:
        return 0.0, 0
    gt_ids = sorted({g for g, _ in assoc})
    pred_ids = sorted({p for _, p in assoc})
    gi = {g: i for i, g in enumerate(gt_ids)}
    pi = {p: j for j, p in enumerate(pred_ids)}
    C = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    for (g, p), c in assoc.items():
        C[gi[g], pi[p]] = c
    row, col = linear_sum_assignment(-C)
    idtp = int(sum(C[i, j] for i, j in zip(row, col)))
    denom = n_gt_dets + n_pred_dets
    return (2.0 * idtp / denom) if denom else 0.0, idtp


def compute_4d_pq(tube_iou, gt_ids, pred_ids, match_thr=0.5):
    """VIPER 4D-PQ（类无关 thing tube 匹配，身份=几何）。

    tube_iou: {(gt_id, pred_id): (iou_sum, iou_cnt)}；gt_ids/pred_ids = 全部 tube id 集合。
    tube IoU = 共可见帧内逐帧 IoU 的均值；贪心降序匹配（IoU>match_thr）。
    返回 (pq, tp, fp, fn, sum_matched_iou)。
    """
    pairs = []
    for (g, p), (s, c) in tube_iou.items():
        iou = s / c if c else 0.0
        if iou > match_thr:
            pairs.append((iou, g, p))
    pairs.sort(key=lambda x: x[0], reverse=True)
    used_gt, used_pred = set(), set()
    tp = 0
    sum_iou = 0.0
    for iou, g, p in pairs:
        if g in used_gt or p in used_pred:
            continue
        used_gt.add(g)
        used_pred.add(p)
        tp += 1
        sum_iou += iou
    fp = len([p for p in pred_ids if p not in used_pred])
    fn = len([g for g in gt_ids if g not in used_gt])
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = sum_iou / denom if denom else 0.0
    return pq, tp, fp, fn, sum_iou


def compute_metrics(gt_visible, match_tracks, revisit_gap_frac=0.25, seq_len=None):
    """gt_visible: {gt_id: sorted frame indices (按时间)}。
    match_tracks: {gt_id: {frame: inst_id or None}}。
    seq_len: 场景固定序列长度（raw frame 数 = max(kf)-min(kf)+1），用于长期跨度阈值。
    None 时回退旧口径 frames[-1]+1。
    返回 dict(IDC, IDS, FM, ReID_R, 各项分子分母)。"""
    n_idc_num = n_idc_den = 0
    n_ids = 0
    n_fm = 0
    n_reid_num = n_reid_den = 0

    for gid, frames in gt_visible.items():
        tr = match_tracks[gid]
        ids_seq = [tr.get(f, None) for f in frames]
        # 众数 id（忽略 None）
        nonnull = [i for i in ids_seq if i is not None]
        mode = max(set(nonnull), key=nonnull.count) if nonnull else None
        # IDC
        for i in ids_seq:
            n_idc_den += 1
            if i is not None and i == mode:
                n_idc_num += 1
        # IDS：连续两帧（都非 None）id 改变
        for a, b in zip(ids_seq, ids_seq[1:]):
            if a is not None and b is not None and a != b:
                n_ids += 1
        # FM：可见但未匹配（track 中断）
        for i in ids_seq:
            if i is None:
                n_fm += 1
        # ReID-R（Long-term Re-ID Recall，无条件）：
        # 分母 = GT 长期可见实例（跨度 >= revisit_gap_frac * 序列长度），只由 GT 决定；
        # 分子 = 首末可见帧都匹配上且 predicted ID 相同。漏检/端点失配/ID switch 均计失败。
        if len(frames) >= 2:
            span = frames[-1] - frames[0]
            thr = (revisit_gap_frac * (seq_len - 1) if seq_len is not None
                   else revisit_gap_frac * max(1, frames[-1] + 1))
            if span >= thr:
                n_reid_den += 1
                if (ids_seq[0] is not None and ids_seq[-1] is not None
                        and ids_seq[0] == ids_seq[-1]):
                    n_reid_num += 1

    idc = n_idc_num / n_idc_den if n_idc_den else 0.0
    reid = n_reid_num / n_reid_den if n_reid_den else 0.0
    return {
        "IDC": idc, "IDC_num": n_idc_num, "IDC_den": n_idc_den,
        "IDS": n_ids, "FM": n_fm,
        "ReID_R": reid, "ReID_num": n_reid_num, "ReID_den": n_reid_den,
    }
