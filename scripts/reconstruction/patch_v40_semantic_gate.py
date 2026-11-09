#!/usr/bin/env python3
"""Standalone patch: wire SEMANTIC-EDGE-GATED opacity decay (Lever 1) into the
Splat-SLAM backend + v40 driver, without touching the user's v3dsam source.

Splat-SLAM is third-party and was already patched for plain StereoGS opacity
decay. This patch adds a *semantic gate* so the decay only hits Gaussians whose
projected centre lands on a semantic boundary, sparing the textureless interior
of walls/floors.

Changes:
  1. gaussian_model.adaptive_opacity_decay(..., gate=None)
        decay_factor = 1 - penalty * gate * exp(-sensitivity * normalized_grad)
        gate=1 (boundary) -> full StereoGS decay; gate=0 (interior) -> spared.
  2. mapper.py:
        - module global SEMANTIC_EDGES = {frame_idx: (H,W) cuda tensor}
        - final_refine builds random_viewpoint_edge_stack in parallel with the
          viewpoint stack; in the inner loop, computes per-Gaussian gate from
          viewspace_point_tensor (means2D pixel coords) + visibility_filter.
  3. v40_stage3_scaled.py: --semantic_edges flag -> load edge npz -> fill
     src.mapper.SEMANTIC_EDGES.

Idempotent-ish: each replacement is applied only if the anchor is present and
the new text is not already there. Run on AutoDL:

  /root/miniconda3/envs/sgs/bin/python patch_v40_semantic_gate.py
"""
import sys

GMO = "/root/autodl-tmp/Splat-SLAM-main/thirdparty/gaussian_splatting/scene/gaussian_model.py"
MAP = "/root/autodl-tmp/Splat-SLAM-main/src/mapper.py"
V40 = "/root/autodl-tmp/v40_stage3_scaled.py"


def apply(s, old, new, label):
    if new in s:
        print(f"[patch] {label}: already patched (skip)")
        return s
    if old not in s:
        raise SystemExit(f"[patch] ANCHOR NOT FOUND in {label}: {old[:80]!r}")
    n = s.count(old)
    s = s.replace(old, new, 1)
    print(f"[patch] {label}: applied ({n} occurrence(s))")
    return s


# ── 1. gaussian_model.py: gate param ──
s = open(GMO, encoding="utf-8").read()
s = apply(s,
    "def adaptive_opacity_decay(self, min_decay_rate=0.995, sensitivity=1.0):",
    "def adaptive_opacity_decay(self, min_decay_rate=0.995, sensitivity=1.0, gate=None):",
    "gaussian_model.signature")
s = apply(s,
    "        penalty_strength = 1.0 - min_decay_rate\n"
    "        decay_factor = 1.0 - penalty_strength * torch.exp(-sensitivity * normalized_grad)\n",
    "        penalty_strength = 1.0 - min_decay_rate\n"
    "        if gate is not None:\n"
    "            penalty_strength = penalty_strength * gate.view(-1, 1)\n"
    "        decay_factor = 1.0 - penalty_strength * torch.exp(-sensitivity * normalized_grad)\n",
    "gaussian_model.gate")
open(GMO, "w", encoding="utf-8").write(s)

# ── 2. mapper.py: module global + edge stack + gate ──
s = open(MAP, encoding="utf-8").read()
s = apply(s,
    "from thirdparty.monogs.utils.camera_utils import Camera\n",
    "from thirdparty.monogs.utils.camera_utils import Camera\n\n"
    "SEMANTIC_EDGES = {}  # frame_idx -> (H_out, W_out) float32 cuda tensor (semantic boundaries)\n",
    "mapper.global")
s = apply(s,
    "        random_viewpoint_stack = []\n"
    "        frames_to_optimize = self.config[\"mapping\"][\"Training\"][\"pose_window\"]\n"
    "\n"
    "        for cam_idx, viewpoint in self.viewpoints.items():\n"
    "            random_viewpoint_stack.append(viewpoint)\n",
    "        random_viewpoint_stack = []\n"
    "        random_viewpoint_edge_stack = []\n"
    "        frames_to_optimize = self.config[\"mapping\"][\"Training\"][\"pose_window\"]\n"
    "\n"
    "        vid2frame = dict(zip(self.video_idxs, self.keyframe_idxs))\n"
    "        for cam_idx, viewpoint in self.viewpoints.items():\n"
    "            random_viewpoint_stack.append(viewpoint)\n"
    "            random_viewpoint_edge_stack.append(SEMANTIC_EDGES.get(vid2frame.get(cam_idx, -1)))\n"
    "        use_semantic_gate = any(e is not None for e in random_viewpoint_edge_stack)\n",
    "mapper.edge_stack")
s = apply(s,
    "            scaling = self.gaussians.get_scaling\n"
    "            loss_mapping.backward()\n"
    "            if self.config[\"mapping\"][\"Training\"].get(\"opacity_decay\", False):\n"
    "                self.gaussians.adaptive_opacity_decay(\n"
    "                    min_decay_rate=self.config[\"mapping\"][\"Training\"].get(\"opacity_decay_rate\", 0.995),\n"
    "                    sensitivity=self.config[\"mapping\"][\"Training\"].get(\"opacity_decay_sensitivity\", 1.0))\n",
    "            scaling = self.gaussians.get_scaling\n"
    "            loss_mapping.backward()\n"
    "            if self.config[\"mapping\"][\"Training\"].get(\"opacity_decay\", False):\n"
    "                gate = None\n"
    "                if use_semantic_gate:\n"
    "                    edge = random_viewpoint_edge_stack[rand_idx]\n"
    "                    if edge is not None:\n"
    "                        W = int(viewpoint.image_width)\n"
    "                        H = int(viewpoint.image_height)\n"
    "                        u = viewspace_point_tensor[:, 0].long().clamp(0, W - 1)\n"
    "                        v = viewspace_point_tensor[:, 1].long().clamp(0, H - 1)\n"
    "                        gate = edge[v, u] * visibility_filter.float()\n"
    "                self.gaussians.adaptive_opacity_decay(\n"
    "                    min_decay_rate=self.config[\"mapping\"][\"Training\"].get(\"opacity_decay_rate\", 0.995),\n"
    "                    sensitivity=self.config[\"mapping\"][\"Training\"].get(\"opacity_decay_sensitivity\", 1.0),\n"
    "                    gate=gate)\n",
    "mapper.gate_hook")
open(MAP, "w", encoding="utf-8").write(s)

# ── 3. v40: --semantic_edges flag + load + set module global ──
s = open(V40, encoding="utf-8").read()
s = apply(s,
    'parser.add_argument("--opacity_decay_sensitivity", type=float, default=1.0, help="sensitivity for gradient-aware opacity decay")\n',
    'parser.add_argument("--opacity_decay_sensitivity", type=float, default=1.0, help="sensitivity for gradient-aware opacity decay")\n'
    'parser.add_argument("--semantic_edges", type=str, default="", help="path to semantic_edges.npz for edge-gated opacity decay")\n',
    "v40.flag")
s = apply(s,
    "mapper = Mapper(fake_slam, pipe)\n"
    "mapper.save_dir = args.output_dir\n"
    "os.makedirs(args.output_dir, exist_ok=True)\n",
    "mapper = Mapper(fake_slam, pipe)\n"
    "mapper.save_dir = args.output_dir\n"
    "os.makedirs(args.output_dir, exist_ok=True)\n"
    "\n"
    "# ── Load semantic edge maps (frame_idx -> cuda tensor) for edge-gated decay ──\n"
    "import src.mapper as _mapper_mod\n"
    "if args.semantic_edges:\n"
    "    _ed = np.load(args.semantic_edges, allow_pickle=True)\n"
    "    _ekf = [int(x) for x in _ed[\"kf_indices\"].tolist()]\n"
    "    _eH = int(_ed[\"H_out\"]); _eW = int(_ed[\"W_out\"])\n"
    "    _mapper_mod.SEMANTIC_EDGES.clear()\n"
    "    for _i, _fi in enumerate(_ekf):\n"
    "        _mapper_mod.SEMANTIC_EDGES[_fi] = torch.from_numpy(_ed[\"edges\"][_i].astype(np.float32)).to(\"cuda\")\n"
    "    print(f\"  Semantic edges: {len(_mapper_mod.SEMANTIC_EDGES)} frame_idx-keyed ({_eH}x{_eW}) loaded\")\n",
    "v40.load")
open(V40, "w", encoding="utf-8").write(s)

print("[patch_v40_semantic_gate] DONE: gaussian_model.py + mapper.py + v40_stage3_scaled.py")
