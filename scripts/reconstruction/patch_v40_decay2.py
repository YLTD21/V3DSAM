#!/usr/bin/env python3
"""Extend v40 with decay-rate/sensitivity + refine_iters knobs (standalone patch)."""
SRC = "/root/autodl-tmp/v40_stage3_scaled.py"
s = open(SRC, encoding="utf-8").read()

# ── Patch 1: argparse flags after --opacity_decay ──
old1 = 'parser.add_argument("--opacity_decay", action="store_true", help="Enable StereoGS gradient-aware opacity decay in final_refine")\n'
new1 = old1 + (
    'parser.add_argument("--opacity_decay_rate", type=float, default=0.995, help="min_decay_rate for gradient-aware opacity decay")\n'
    'parser.add_argument("--opacity_decay_sensitivity", type=float, default=1.0, help="sensitivity for gradient-aware opacity decay")\n'
    'parser.add_argument("--refine_iters", type=int, default=-1, help="override final_refine_iters (-1 = config default 26000)")\n'
)
assert s.count(old1) == 1, f"patch1 anchor count={s.count(old1)}"
s = s.replace(old1, new1)

# ── Patch 2: override block (replace the opacity_decay override added earlier) ──
old2 = '''if args.opacity_decay:
    cfg["mapping"]["Training"]["opacity_decay"] = True
    print("  Opacity decay: ON (StereoGS gradient-aware, min_decay_rate=0.995)")
cfg["mapping"]["BA"] = True'''
new2 = '''if args.opacity_decay:
    cfg["mapping"]["Training"]["opacity_decay"] = True
    cfg["mapping"]["Training"]["opacity_decay_rate"] = args.opacity_decay_rate
    cfg["mapping"]["Training"]["opacity_decay_sensitivity"] = args.opacity_decay_sensitivity
    print(f"  Opacity decay: ON (rate={args.opacity_decay_rate}, sensitivity={args.opacity_decay_sensitivity})")
if args.refine_iters > 0:
    cfg["mapping"]["final_refine_iters"] = args.refine_iters
    print(f"  refine_iters override: {args.refine_iters}")
cfg["mapping"]["BA"] = True'''
assert s.count(old2) == 1, f"patch2 anchor count={s.count(old2)}"
s = s.replace(old2, new2)

open(SRC, "w", encoding="utf-8").write(s)
print(f"[patch_v40_decay2] patched {SRC} ({len(s)} bytes)")
