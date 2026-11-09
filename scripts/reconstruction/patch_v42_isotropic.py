#!/usr/bin/env python3
"""Standalone patch: restore the isotropic scale regularizer to final_refine.

Diagnosis (2区, algorithm-level): Splat-SLAM's `map()` loop regularizes the
Gaussian scale every iteration:
      scaling = self.gaussians.get_scaling
      isotropic_loss = |scaling - scaling.mean(dim=1)|   # penalize elongated Gaussians
      loss_mapping += 10 * isotropic_loss.mean()          # weight 10
but `final_refine()` computes `scaling` and then drops straight into
`loss_mapping.backward()` with NO isotropic term (mapper.py:719-720).  This is
the ONE regularizer map() uses that final_refine silently discards — needle-like
/ elongated floaters are free to grow during the 26000-iter pure-RGB overfit,
which is the collapse mechanism.

This patch restores it behind a flag (default 0 = off, baseline unchanged).
Enables sweeping isotropic weight in isolation and in combination with the
opacity-reset + prune patch (v41).

Changes (idempotent):
  1. mapper.py final_refine: add configurable isotropic loss before backward().
  2. v40_stage3_scaled.py: --refine_isotropic_weight flag -> Training.

Run on AutoDL (sgs env):
  /root/miniconda3/envs/sgs/bin/python patch_v42_isotropic.py
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


# ── 1. mapper.py: isotropic loss in final_refine ──
s = open(MAP, encoding="utf-8").read()
s = apply(s,
    "            scaling = self.gaussians.get_scaling\n"
    "            loss_mapping.backward()\n",
    "            scaling = self.gaussians.get_scaling\n"
    "            if self.config[\"mapping\"][\"Training\"].get(\"refine_isotropic_weight\", 0) > 0:\n"
    "                isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))\n"
    "                loss_mapping += self.config[\"mapping\"][\"Training\"][\"refine_isotropic_weight\"] * isotropic_loss.mean()\n"
    "            loss_mapping.backward()\n",
    "mapper.isotropic")
open(MAP, "w", encoding="utf-8").write(s)

# ── 2. v40: argparse flag ──
s = open(V40, encoding="utf-8").read()
s = apply(s,
    'parser.add_argument("--refine_prune_opacity", type=float, default=0.005, help="final_refine: prune opacity below this")\n',
    'parser.add_argument("--refine_prune_opacity", type=float, default=0.005, help="final_refine: prune opacity below this")\n'
    'parser.add_argument("--refine_isotropic_weight", type=float, default=0.0, help="final_refine: isotropic scale-reg weight (0=off, map() uses 10)")\n',
    "v40.flag")

# ── 3. v40: cfg override ──
s = apply(s,
    'if args.refine_prune_interval:\n'
    '    cfg["mapping"]["Training"]["refine_prune_interval"] = args.refine_prune_interval\n'
    '    cfg["mapping"]["Training"]["refine_prune_opacity"] = args.refine_prune_opacity\n'
    '    print(f"  Refine reg: prune opacity<{args.refine_prune_opacity} every {args.refine_prune_interval} iters")\n',
    'if args.refine_prune_interval:\n'
    '    cfg["mapping"]["Training"]["refine_prune_interval"] = args.refine_prune_interval\n'
    '    cfg["mapping"]["Training"]["refine_prune_opacity"] = args.refine_prune_opacity\n'
    '    print(f"  Refine reg: prune opacity<{args.refine_prune_opacity} every {args.refine_prune_interval} iters")\n'
    'if args.refine_isotropic_weight:\n'
    '    cfg["mapping"]["Training"]["refine_isotropic_weight"] = args.refine_isotropic_weight\n'
    '    print(f"  Refine reg: isotropic scale-reg weight={args.refine_isotropic_weight}")\n',
    "v40.override")
open(V40, "w", encoding="utf-8").write(s)

print("[patch_v42_isotropic] DONE: mapper.py + v40_stage3_scaled.py")
