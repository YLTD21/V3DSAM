#!/usr/bin/env python3
"""Add --opacity_decay flag to v40_stage3_scaled.py (standalone patch).

Enables the ALREADY-PORTED StereoGS gradient-aware opacity decay that sits
disabled in Splat-SLAM's final_refine loop (mapper.py:721 -> adaptive_opacity_decay).
No change to v3dsam source; only v40 (our standalone stage3 script) is touched.
"""
SRC = "/root/autodl-tmp/v40_stage3_scaled.py"
s = open(SRC, encoding="utf-8").read()

# ── Patch 1: argparse flag after --depth_mode ──
old1 = 'parser.add_argument("--depth_mode", type=str, default="raw_norm", help="raw_norm (v38) | raw_none | scaled_none")\n'
new1 = old1 + 'parser.add_argument("--opacity_decay", action="store_true", help="Enable StereoGS gradient-aware opacity decay in final_refine")\n'
assert s.count(old1) == 1, f"patch1 anchor count={s.count(old1)}"
s = s.replace(old1, new1)

# ── Patch 2: config override right before cfg["mapping"]["BA"] ──
old2 = 'cfg["mapping"]["BA"] = True\n'
new2 = ('if args.opacity_decay:\n'
        '    cfg["mapping"]["Training"]["opacity_decay"] = True\n'
        '    print("  Opacity decay: ON (StereoGS gradient-aware, min_decay_rate=0.995)")\n'
        'cfg["mapping"]["BA"] = True\n')
assert s.count(old2) == 1, f"patch2 anchor count={s.count(old2)}"
s = s.replace(old2, new2)

open(SRC, "w", encoding="utf-8").write(s)
print(f"[patch_v40_opacity] patched {SRC} ({len(s)} bytes)")
print(f"[patch_v40_opacity] --opacity_decay flag + Training.opacity_decay override added")
