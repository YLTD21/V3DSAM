#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从官方 scannetv2-labels.combined.tsv 生成 raw_id -> nyu40id(0-based) 映射。
ScanNet 语义 PNG (label/label-filt/semantic_id) 里存的是 raw id 列，不是 0-39 的 nyu40id。
"""
import csv, json

TSV = r"E:\scannet\OpenDataLab___ScanNet_v2\raw\scannetv2-labels.combined.tsv"
OUT = r"E:\0722v3dsam\paper\scannet_raw2nyu.json"

raw2nyu = {}
with open(TSV, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f, delimiter="\t")
    for row in r:
        rid = int(row["id"])
        nyu = int(row["nyu40id"])          # 1..40
        raw2nyu[rid] = nyu - 1             # 0..39

print("unique raw ids:", len(raw2nyu))
print("raw id range:", min(raw2nyu), "-", max(raw2nyu))
print("nyu40id(0-based) range:", min(raw2nyu.values()), "-", max(raw2nyu.values()))
for rid in [0, 1, 2, 3, 4, 5, 6, 7, 9, 21, 48]:
    print(f"  raw {rid:4d} -> nyu40id {raw2nyu.get(rid, 'UNANNOTATED/unknown')}")

from collections import Counter
c = Counter(raw2nyu.values())
print("nyu40id coverage (0-based):", sorted(c.keys()))

with open(OUT, "w") as f:
    json.dump({str(k): v for k, v in sorted(raw2nyu.items())}, f)
print("wrote", OUT, "with", len(raw2nyu), "entries")
