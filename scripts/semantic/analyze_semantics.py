import os, glob, json, numpy as np

ROOTS = {
    "room": "/root/autodl-tmp/semantic_out/room",
    "360":  "/root/autodl-tmp/semantic_out/360",
}

def centroid(m):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()], dtype=float)


def layout_consistency(mq, md):
    """Semantic Layout Score: align object centroids via least-squares similarity
    (translation + uniform scale), return RMS residual (normalized)."""
    qc = [c for c in range(mq.shape[0]) if mq[c].sum() > 500]
    dc = [c for c in range(md.shape[0]) if md[c].sum() > 500]
    shared = [c for c in qc if c in dc]
    if len(shared) < 3:
        return None, len(shared)
    P = np.array([centroid(mq[c]) for c in shared])
    D = np.array([centroid(md[c]) for c in shared])
    # normalize coords to unit image
    H, W = mq.shape[1], mq.shape[2]
    P = P / np.array([W, H]); D = D / np.array([W, H])
    # least-squares similarity: D -> P  (uniform scale s, translation t)
    # unknown [s, tx, ty], rows: [Dx,1,0] and [Dy,0,1]
    n = len(D)
    A = np.zeros((2 * n, 3))
    A[0:n, 0] = D[:, 0]; A[0:n, 1] = 1.0
    A[n:, 0] = D[:, 1]; A[n:, 2] = 1.0
    b = np.concatenate([P[:, 0], P[:, 1]])
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    s, tx, ty = x
    pred = s * D + np.array([tx, ty])
    resid = np.linalg.norm(pred - P, axis=1)
    rms = float(np.sqrt((resid ** 2).mean()))
    return rms, len(shared)


def main():
    for seq, root in ROOTS.items():
        log = json.load(open(os.path.join(root, "loop_log.json")))
        print(f"\n===== {seq} =====")
        for k, lp in enumerate(log["loops"]):
            mz = os.path.join(root, "loop_figures", f"loop{k}_masks.npz")
            if not os.path.exists(mz):
                print(f"loop{k}: NO MASK")
                continue
            d = np.load(mz)
            mq, md = d["mask_q"], d["mask_d"]
            qc = set(c for c in range(mq.shape[0]) if mq[c].sum() > 500)
            dc = set(c for c in range(md.shape[0]) if md[c].sum() > 500)
            inter = qc & dc
            union = qc | dc
            jacc = len(inter) / len(union) if union else 0.0
            dice = 2 * len(inter) / (len(qc) + len(dc)) if (qc or dc) else 0.0
            rms, nshared = layout_consistency(mq, md)
            print(f"loop{k}: score={lp['score']:.3f} "
                  f"overlap(min)={lp['overlap']:.2f} jacc={jacc:.2f} dice={dice:.2f} "
                  f"|q|={len(qc)} |d|={len(dc)} shared={len(inter)} "
                  f"layout_rms={rms:.4f} (n={nshared})")


if __name__ == "__main__":
    main()
