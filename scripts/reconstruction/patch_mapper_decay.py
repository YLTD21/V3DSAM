#!/usr/bin/env python3
"""Patch Splat-SLAM's mapper.py so opacity decay rate/sensitivity come from config.

Current: mapper.py:722 calls self.gaussians.adaptive_opacity_decay() with the
hardcoded defaults (0.995, 1.0) -- which over-decay surfaces over 26000 iters.
This lets the config drive min_decay_rate / sensitivity so we can sweep a gentler
rate. NOTE: this edits third-party Splat-SLAM-main (NOT the user's v3dsam source).
"""
SRC = "/root/autodl-tmp/Splat-SLAM-main/src/mapper.py"
s = open(SRC, encoding="utf-8").read()

old = '''            if self.config["mapping"]["Training"].get("opacity_decay", False):
                self.gaussians.adaptive_opacity_decay()'''
new = '''            if self.config["mapping"]["Training"].get("opacity_decay", False):
                self.gaussians.adaptive_opacity_decay(
                    min_decay_rate=self.config["mapping"]["Training"].get("opacity_decay_rate", 0.995),
                    sensitivity=self.config["mapping"]["Training"].get("opacity_decay_sensitivity", 1.0))'''
assert s.count(old) == 1, f"anchor count={s.count(old)}"
s = s.replace(old, new)

open(SRC, "w", encoding="utf-8").write(s)
print(f"[patch_mapper_decay] patched {SRC}")
