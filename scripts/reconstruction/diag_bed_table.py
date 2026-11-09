#!/usr/bin/env python3
"""Compare per-category pixel counts old (kf_semantic.npz) vs new (kf_semantic_v2.npz)."""
import numpy as np

FOCUS = {"bed", "table", "desk", "cabinet", "dresser", "night stand", "sofa", "shelf", "bookshelf"}
BASE = "/root/autodl-tmp/ablation/scannet"

for scene in ["scene0000_00", "scene0059_00", "scene0106_00"]:
    print(f"\n========== {scene} ==========")
    for tag, fn in [("old", "kf_semantic.npz"), ("new", "kf_semantic_v2.npz")]:
        d = np.load(f"{BASE}/{scene}/{fn}", allow_pickle=True)
        labels = d["labels"]
        prompts = [str(p) for p in d["prompts"]]
        lab = labels.ravel()
        lab = lab[lab >= 0]
        cnt = np.bincount(lab.astype(np.int64), minlength=len(prompts))
        total = int(cnt.sum())
        print(f"\n  --- {tag} ({fn}): {total:,} labeled px, {len(prompts)} cats ---")
        # focus categories
        for c in range(len(prompts)):
            if prompts[c] in FOCUS and cnt[c] > 0:
                print(f"    {prompts[c]:<12} idx={c:<3} {cnt[c]:>12,}  ({cnt[c]/total*100:.1f}%)")
        # top 8
        order = np.argsort(-cnt)[:8]
        top = ", ".join(f"{prompts[c]}:{cnt[c]/total*100:.1f}%" for c in order if cnt[c] > 0)
        print(f"    [top8] {top}")
