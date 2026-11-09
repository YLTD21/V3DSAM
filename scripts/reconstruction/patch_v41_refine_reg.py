#!/usr/bin/env python3
"""Standalone patch: restore REGULARIZATION to Splat-SLAM's final_refine (2区 lever).

Root cause (reconstruction-blob-collapse-rootcause.md): final_refine runs 26000
iters of UNREGULARIZED pure-RGB overfit.  Floating blobs (unobserved regions /
depth gaps) grow opacity -> 1 and are never pruned, so held-out NVS collapses
(~9 dB) while train keyframes overfit (~23 dB).

This patch restores the CANONICAL 3DGS regularizer that is missing from
final_refine: periodic opacity reset (opacity -> 0.01, forcing floaters to
re-earn opacity) + periodic low-opacity prune (removes floaters that cannot
re-earn it).  Defaults are OFF (0) so baseline behaviour is unchanged; enable
via v40 flags.

Changes (idempotent, same pattern as patch_v40_semantic_gate.py):
  1. mapper.py final_refine: local counter refine_i + reset/prune hook (in no_grad).
  2. v40_stage3_scaled.py: --refine_opacity_reset_interval / --refine_prune_interval
     / --refine_prune_opacity -> cfg["mapping"]["Training"].

Run on AutoDL (sgs env):
  /root/miniconda3/envs/sgs/bin/python patch_v41_refine_reg.py
"""
MAP = "/root/autodl-tmp/Splat-SLAM-main/src/mapper.py"
V40 = "/root/autodl-tmp/v40_stage3_scaled.py"


def apply(s, old, new, label):
    if new in s:
        print(f"[patch] {label}: already patched (skip)")
        return s
    if old not in s:
        raise SystemExit(f"[patch] ANCHOR NOT FOUND in {label}: {old[:80]!r}")
    n = s.count(old)
    if n != 1:
        print(f"[patch] WARNING {label}: anchor found {n}x (expected 1)")
    s = s.replace(old, new, 1)
    print(f"[patch] {label}: applied")
    return s


# ── 1. mapper.py: local refine counter ──
s = open(MAP, encoding="utf-8").read()
s = apply(s,
    "        for _ in tqdm(range(iters)):",
    "        for refine_i in tqdm(range(iters)):",
    "mapper.refine_counter")

# ── 2. mapper.py: reset + prune hook inside final_refine no_grad ──
old_block = '''            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)'''
new_block = old_block + '''
                # ── Regularization: periodic opacity reset + low-opacity prune ──
                refine_reset_interval = self.config["mapping"]["Training"].get("refine_opacity_reset_interval", 0)
                refine_prune_interval = self.config["mapping"]["Training"].get("refine_prune_interval", 0)
                refine_prune_opacity = self.config["mapping"]["Training"].get("refine_prune_opacity", 0.005)
                if refine_prune_interval and (refine_i + 1) % refine_prune_interval == 0:
                    self.gaussians.prune_points((self.gaussians.get_opacity < refine_prune_opacity).squeeze())
                if refine_reset_interval and (refine_i + 1) % refine_reset_interval == 0:
                    self.gaussians.reset_opacity()'''
s = apply(s, old_block, new_block, "mapper.refine_reg_hook")
open(MAP, "w", encoding="utf-8").write(s)

# ── 3. v40: argparse flags ──
s = open(V40, encoding="utf-8").read()
s = apply(s,
    'parser.add_argument("--semantic_edges", type=str, default="", help="path to semantic_edges.npz for edge-gated opacity decay")\n',
    'parser.add_argument("--semantic_edges", type=str, default="", help="path to semantic_edges.npz for edge-gated opacity decay")\n'
    'parser.add_argument("--refine_opacity_reset_interval", type=int, default=0, help="final_refine: reset opacity every N iters (0=off)")\n'
    'parser.add_argument("--refine_prune_interval", type=int, default=0, help="final_refine: prune low-opacity every N iters (0=off)")\n'
    'parser.add_argument("--refine_prune_opacity", type=float, default=0.005, help="final_refine: prune opacity below this")\n',
    "v40.flags")

# ── 4. v40: cfg override ──
s = apply(s,
    'cfg["mapping"]["BA"] = True\n',
    'cfg["mapping"]["BA"] = True\n'
    'if args.refine_opacity_reset_interval:\n'
    '    cfg["mapping"]["Training"]["refine_opacity_reset_interval"] = args.refine_opacity_reset_interval\n'
    '    print(f"  Refine reg: opacity reset every {args.refine_opacity_reset_interval} iters")\n'
    'if args.refine_prune_interval:\n'
    '    cfg["mapping"]["Training"]["refine_prune_interval"] = args.refine_prune_interval\n'
    '    cfg["mapping"]["Training"]["refine_prune_opacity"] = args.refine_prune_opacity\n'
    '    print(f"  Refine reg: prune opacity<{args.refine_prune_opacity} every {args.refine_prune_interval} iters")\n',
    "v40.override")
open(V40, "w", encoding="utf-8").write(s)

print("[patch_v41_refine_reg] DONE: mapper.py + v40_stage3_scaled.py")
