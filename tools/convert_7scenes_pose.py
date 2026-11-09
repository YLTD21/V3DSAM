#!/usr/bin/env python3
"""
Convert 7-Scenes pose.txt (4x4 matrix) to TUM format for evo ATE evaluation.

7-Scenes pose.txt format (4x4):
    R11 R12 R13 t1
    R21 R22 R23 t2
    R31 R32 R33 t3
    0   0   0   1

TUM format:
    timestamp tx ty tz qx qy qz qw

Usage:
    python convert_7scenes_pose.py \
        --seq_dir /path/to/7scenes/chess/seq-02 \
        --output chess_seq02_gt_tum.txt
"""
import numpy as np, os, glob, re, argparse
from scipy.spatial.transform import Rotation


def convert_seq(seq_dir, output_path):
    pose_files = sorted(glob.glob(os.path.join(seq_dir, "frame-*.pose.txt")))
    if not pose_files:
        print(f"ERROR: no pose files found in {seq_dir}")
        return False

    n = 0
    with open(output_path, 'w') as f:
        for p in pose_files:
            # 帧号 = frame-XXXXXX.pose.txt 中的数字
            m = re.search(r'frame-(\d+)\.pose\.txt', os.path.basename(p))
            if not m:
                continue
            frame_idx = int(m.group(1))
            T = np.loadtxt(p)  # 4x4 matrix
            R = T[:3, :3]
            t = T[:3, 3]
            q = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
            # TUM格式用图像文件名做时间戳（与stage1_poses.txt对齐）
            # 这里用帧号，与VGGT-SLAM输出的图像命名保持一致
            f.write(f"{frame_idx} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                    f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")
            n += 1

    print(f"Converted {n} poses → {output_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert_seq(args.seq_dir, args.output)
