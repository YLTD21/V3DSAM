# 数据目录说明

本目录用于存放**数据集与中间结果**的约定与来源说明。**原始数据与超大中间产物不随仓库分发**
（已由根目录 `.gitignore` 排除），需按下方路径在本地 / AutoDL 复现。

## 数据集

| 数据集 | 用途 | 来源 / 目录约定 |
| --- | --- | --- |
| TUM-RGBD | 前端 ATE、回环校验（9 个 `fr1` 序列） | `datasets/tum/{seq}/` |
| ScanNet | 长序列回环、mIoU、持久实例（4 场景） | `datasets/scannet/{scene}/color` |
| VGGT-1B 权重 | 前端模型 | HuggingFace `facebook/VGGT-1B/resolve/main/model.pt` |
| SAM3 权重 | 开放词汇语义 | 按 SAM3 官方仓库路径 |
| Perception Encoder CLIP 权重 | 开放集语义检索 | 环境变量 `PE_CLIP_CKPT`（见 `main.py`） |

## 大体积中间产物（不随仓库分发）

| 目录 | 大小 | 说明 |
| --- | --- | --- |
| `semantic_cloud_10m/` | ~1.75 GB | 语义点云 npz（`pts`+`label`+`prompts`），每场景 ~440 MB |
| `scannet_all/` | ~301 MB | ScanNet 全量中间结果 |
| `_scannet_upload/*.tar` | ~1.3 GB | 打包上传的原始色图/位姿 |

这些文件由 `scripts/semantic/gen_semantic_cloud_10m.py` 与 `scripts/frontend/gen_npz_scan.py`
生成，路径约定见各脚本头部 docstring。所有实验数字均由对应脚本从这些数据重新生成。
