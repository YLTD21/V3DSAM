# V3DSAM

**Open-Vocabulary Semantic Dense SLAM via Semantic-Geometric Loop Verification and Geometry-Anchored Instance Identity**

<p align="center">
  <img alt="License" src="https://img.shields.io/badge/license-BSD--2--Clause-blue.svg">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10+-blue.svg">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Linux%20(CUDA)-lightgrey.svg">
</p>

> 中文摘要：V3DSAM 是一个纯单目前馈的开放词汇语义稠密 SLAM 系统，把逐字官方的 VGGT-SLAM 位姿前端、
> SAM3 开放词汇语义分割、3DGS 稠密重建串成一条流水线，并在此基础上提出两个贡献：①语义-几何三维一致性
> 回环校验；②几何锚定的持久实例身份。本仓库是该工作的最终代码，用于复现全部实验数字。

---

## Overview

V3DSAM is a **feed-forward monocular SLAM** system. Unlike conventional optimization-based SLAM,
its front-end performs no online bundle adjustment and its loop-closure backend is trained-free.
Three components are chained:

| Stage | Component | Role |
| --- | --- | --- |
| 1 | **VGGT-SLAM 2.0** | Feed-forward pose front-end, verbatim official (consumed as-is) |
| 2 | **SAM3** | Open-vocabulary semantic segmentation |
| 3 | **3DGS (Splat-SLAM)** | Dense Gaussian reconstruction backend |

On this base, V3DSAM contributes two results that address two concrete failure modes of
feed-forward SLAM:

### Contribution 1 — Semantic-geometric 3D-consistency loop verification

**Problem.** A feed-forward front-end detects loop candidates by *appearance* (SALAD retrieval)
and *geometric* fit (αmatch). In "same room, reversed viewpoint" scenes, a false loop can look
appearance-similar, geometrically plausible, and class-overlap-similar — none of these signals
separate it from a true loop.

**Method.** Back-project SAM3 category masks into 3D, transform loop-candidate views by their
estimated relative pose, then compare per-pixel semantic labels **bi-directionally and
symmetrically**, reinforced by an **object-level** consistency criterion (τ ≈ 0.25). A true loop
has the same objects at the same 3D locations; a false loop does not.

**Result.** Zero false positives where αmatch's native threshold reports 4, with 12.5 ms
overhead per candidate.

### Contribution 2 — Geometry-anchored persistent instance identity

**Problem.** Semantic segmentation tells you *what* an object is (category), not *which* one it
is (instance). Feed-forward tracking (SAM3 video) breaks across large loop gaps, and per-frame
appearance matching (CLIP) is too weak.

**Method.** Anchor **instance identity in geometry**: cluster SAM3-masked points by category in
the SL(4) globally-consistent world frame (cluster `eps = 0.05`, ≈7 cm), then bind each 3D
instance to its overlapping 2D semantic region. The same physical object keeps the same `#id`
across loop closures because its world-frame position is stable.

**Result.** ≈90% re-identification rate with ≈7 cm anchor stability — 4× tighter than a rolling
Sim(3) baseline (≈30 cm).

---

## Key results

### Loop verification (Contribution 1)

| Metric | Result |
| --- | --- |
| Head-to-head (15 candidates: 11 true + 4 false) | **P = 1.0 / R = 1.0** (TP 11 / FP 0 / TN 4 / FN 0) |
| αmatch native threshold (−0.95) | P = 0.733 (FP = 4) |
| Discrimination margin | false c ≤ 0.078 vs true c ≥ 0.152 (≈2× gap); P = R = 1.0 for τ ∈ (0.078, 0.152) |
| Noise robustness | P = 1.0 at 0.2 noise; P = 1.0 / R = 0.818 at 0.5 noise |
| Trajectory gain (ATE, m) | room 0.809 → **0.074** (11×); 360 0.188 → **0.0488** |
| Compute overhead | **12.5 ms / candidate** |

### Instance identity (Contribution 2)

| Metric | Result |
| --- | --- |
| Re-identification — geometric / category-gated / appearance | room **89.5%** / 59.2% / 2.6%; 360 **86.6%** / 52.2% / 1.5% |
| Anchor stability (median centroid displacement, half-view split) | **≈7 cm** vs rolling-Sim(3) ≈30 cm |
| Persistent instance matrix (4 scenes × 5 frames) | scene0000 = 7 / 0059 = 9 / 0106 = 12 / 0207 = 3 |

### ScanNet long sequences

| Metric | Result |
| --- | --- |
| False-loop rate (4 scenes, 31 candidates) | **6.5%** (2/31) |
| scene0207 false-loop rejection | appearance 0.947 but consistency 0.0 → ATE 0.109 → **0.089** |
| mIoU (200 frames) | 0.119 / pixel-acc 0.477 (structural classes 0.5–0.87) |

---

## Repository structure

```
V3DSAM/
├── main.py                        # VGGT-SLAM demo entry (+ open-set query menu)
├── vggt_slam/                     # active front-end package (SL(4) manifold)
│   ├── solver.py  submap.py  slam_utils.py  graph.py  map.py  loop_closure.py …
│   ├── semantic_loop_verify.py    #   cluster_objects_by_class helper (verbatim)
│   └── pipeline.py  stage{2,3,4}_*.py   # VGGT-SLAM → 3DGS reconstruction pipeline
├── vggt_slam_official/            # verbatim upstream VGGT-SLAM 2.0 reference
├── scripts/                       # experiments + evaluation, grouped by function
│   ├── frontend/                  #   stage-1 front-end + npz generation + diagnostics
│   ├── semantic/                  #   SAM3 segmentation + semantic cloud / keyframe / instance
│   ├── loop_closure/              #   Contribution 1: verification, αmatch, ablations
│   ├── instance_identity/         #   Contribution 2: identity, re-ID, matrices
│   ├── reconstruction/            #   3DGS backend, opacity decay, floater pruning, NVS
│   ├── evaluation/                #   mIoU / instance GT evaluation / overhead
│   ├── paper/                     #   figure generation + md→docx conversion
│   └── run/                       #   shell launchers (incl. AutoDL)
├── evals/                         # TUM / EuRoC ATE evaluation
├── tools/                         # mIoU / 7-Scenes / 3DGS export utilities
├── setup.sh  setup.py             # base deps + third_party (vggt/sam3/perception_models/salad)
├── environment_sgs.yml  requirements.txt
└── data/README.md                 # dataset & intermediate-artifact conventions
```

`third_party/` is **not** bundled — `setup.sh` clones `vggt` (VGGT_SPARK), `sam3`,
`perception_models`, and `salad` into it.

---

## Installation

```bash
# 1) Base deps + third_party (vggt / sam3 / perception_models / salad)
bash setup.sh

# 2) VGGT-1B weights are pulled at runtime from HuggingFace
#    (facebook/VGGT-1B/resolve/main/model.pt)
```

Requirements: Python 3.10+, PyTorch 2.x + CUDA (RTX 4090 / A100 recommended). Exact conda
environment: [`environment_sgs.yml`](environment_sgs.yml).

---

## Reproduction

The scripts were run on AutoDL with the repo at `/root/autodl-tmp/v3dsam` (the `sys.path`
entries in each script assume this). Adjust the paths in the [Configuration](#configuration)
section for your machine.

### 0. Datasets

| Dataset | Use | Convention |
| --- | --- | --- |
| TUM-RGBD | front-end ATE, loop verification (9 `fr1` sequences) | `datasets/tum/{seq}/` |
| ScanNet | long-sequence loop, mIoU, persistent instances (4 scenes) | `datasets/scannet/{scene}/color` |

See [`data/README.md`](data/README.md) for the large intermediate artifacts generated (not
distributed).

### 1. Front-end + loop verification (Contribution 1)

`gen_npz_scan.py` is the single entry point — it runs the VGGT-SLAM stage-1 front-end, detects
loops, applies **per-pixel** 3D consistency (`semantic_3d_check.py`, τ = 0.10) **and**
**object-level** consistency (`semantic_object_check.py`, τ = 0.25), rejects false loops, and
re-optimizes:

```bash
python scripts/frontend/gen_npz_scan.py --scene scene0000_00
```

The two criteria can also be run standalone, and ablations / head-to-head tables are generated
with the scripts in `scripts/loop_closure/` (e.g. `gen_e1_e3.py`, `gen_scannet_alpha.py`).

### 2. Instance identity (Contribution 2)

3D clustering (eps = 0.05) to fix instance IDs + 2D-mask binding:

```bash
python scripts/instance_identity/gen_instance_grid_3d2d.py --scenes scene0000_00
```

Re-identification head-to-head (geometric vs SAM3-video vs CLIP-appearance vs Dinya baseline):

```bash
python scripts/instance_identity/compare_instance_identity.py --scene scene0000_00
```

### 3. Instance evaluation (Table 4-3) & mIoU

```bash
# smoke test
python scripts/evaluation/eval_persistent_identity.py --methods v3dsam ov_baseline --max_frames 200
# full run
python scripts/evaluation/eval_persistent_identity.py

python scripts/evaluation/measure_miou.py ...
```

### 4. Front-end ATE

```bash
bash evals/eval_tum.sh   # TUM-RGBD ATE (see the script for sequence list)
```

### 5. Reconstruction (3DGS) + NVS

```bash
python vggt_slam/pipeline.py --image_dir datasets/416 --stage all --num_iters 3000
# batch launchers: scripts/run/*.sh
```

> Exact argument values and AutoDL specifics are in each script's docstring.

---

## Configuration

A few absolute paths from our environment are **not portable** — replace them before running:

| Location | Ours (default) | Override |
| --- | --- | --- |
| `main.py` CLIP checkpoint | `/media/yml/share/VGGT-SLAM-3DGS/models/PE-Core-B16-224.pt` | env `PE_CLIP_CKPT` |
| `vggt_slam/pipeline.py` diff-gaussian-rasterization | `/media/yml/share/v3dsam/diff-gaussian-rasterization-modified` | edit `sys.path` |
| most `scripts/**` (`sys.path`, `--out`) | `/root/autodl-tmp/v3dsam` | edit `sys.path` / `--out` |

Scripts inside a package use `os.path.dirname(os.path.abspath(__file__))` for relative imports;
the AutoDL scripts expect the repo cloned at `/root/autodl-tmp/v3dsam`.

---

## Reproducibility & honesty notes

1. **No single "loop-verification" runtime.** Contribution-1 numbers are the output of
   `gen_npz_scan.py` (front-end + candidate poses) → `semantic_3d_check.py` (per-pixel) +
   `semantic_object_check.py` (object-level). An earlier `semantic_loop_verify.py` *runtime* was
   a stub; only its `cluster_objects_by_class` helper is kept and used verbatim.
2. **`vggt_slam/` vs `vggt_slam_official/`.** `vggt_slam_official/solver.py` is the verbatim
   upstream `Solver` main loop; it reuses the shared core modules from `vggt_slam` (a reverse
   import), so the two packages are not fully separable. `setup.py` installs both.
3. **Honest reporting.** Appearance-based re-ID (~2%) is reported as a *weak baseline* (the L2
   appearance score is itself the retrieval signal — circular). Category identity = semantics,
   instance identity = geometry. The dense-reconstruction (NVS) collapse is a structural
   negative result of the feed-forward setting, not a method contribution.

All reported numbers are produced by the scripts in this repo from the datasets above.

---

## Citation

If you use V3DSAM, please cite the base system and, once available, V3DSAM itself:

```bibtex
@inproceedings{maggio2025vggtslam,
  title   = {{VGGT-SLAM}: Dense {RGB} {SLAM} Optimized on the {SL(4)} Manifold},
  author  = {Maggio, Dominic and Lim, Hyungtae and Carlone, Luca},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year    = {2025}
}

@misc{v3dsam2026,
  title   = {{V3DSAM}: Open-Vocabulary Semantic Dense {SLAM} via Semantic-Geometric
             Loop Verification and Geometry-Anchored Instance Identity},
  author  = {{V3DSAM Authors}},
  year    = {2026},
  note    = {under submission}
}
```

---

## License

V3DSAM is distributed under the [BSD 2-Clause License](LICENSE), retaining the upstream
VGGT-SLAM copyright (`Copyright (c) 2025, MIT-SPARK`). Third-party components are installed
separately and retain their own licenses — see the table in [`LICENSE`](LICENSE).
