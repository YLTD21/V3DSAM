#!/bin/bash
# ═══════════════════════════════════════════════════════════
# 7-Scenes ATE 评估 —— 严格遵循官方 TestSplit 协议
#
# 协议：
#   1. 每个场景跑全部官方测试 seq（见 TEST_SEQS）
#   2. 每个 seq 计算 Sim(3) 对齐后的 ATE-RMSE（evo_ape -as）
#   3. 场景 ATE = 该场景所有测试 seq 的 ATE 平均
#   4. 7-Scenes 整体 ATE = 7 个场景 ATE 的平均
#
# 用法（AutoDL 上运行）：
#   bash tools/run_7scenes_ate.sh /root/autodl-tmp/v3dsam/datasets/7-scence
# ═══════════════════════════════════════════════════════════
set -u

DATASET_DIR="${1:-/root/autodl-tmp/v3dsam/datasets/7-scence}"
V3DSAM_DIR="/root/autodl-tmp/v3dsam"
RESULTS_FILE="experiments/7scenes_ate_results.csv"

cd "$V3DSAM_DIR"
echo "scene,seq,ate_rmse" > "$RESULTS_FILE"

# ═══ 官方 TestSplit 定义 ═══
declare -A TEST_SEQS
TEST_SEQS[chess]="seq-02 seq-03"
TEST_SEQS[fire]="seq-01 seq-02"
TEST_SEQS[heads]="seq-01 seq-02"
TEST_SEQS[office]="seq-02 seq-03 seq-04"
TEST_SEQS[pumpkin]="seq-02 seq-03 seq-06"
TEST_SEQS[redkitchen]="seq-01 seq-02 seq-05"
TEST_SEQS[stairs]="seq-01 seq-02"

# ═══ 每个场景的累计 ATE（用于取平均）═══
declare -A SCENE_ATE_SUM
declare -A SCENE_ATE_CNT

for scene in chess fire heads office pumpkin redkitchen stairs; do
    SCENE_ATE_SUM[$scene]=0
    SCENE_ATE_CNT[$scene]=0
    echo ""
    echo "════════════════ $scene ════════════════"

    for seq in ${TEST_SEQS[$scene]}; do
        SEQ_DIR="$DATASET_DIR/$scene/$seq"
        if [ ! -d "$SEQ_DIR" ]; then
            echo "  ✗ 缺失 $SEQ_DIR，跳过"
            continue
        fi

        OUT="experiments/7scenes_${scene}_${seq}"
        echo "  ── $seq ──"

        # Step 1: Stage1 (VGGT-SLAM)，生成 stage1_poses.txt
        if [ ! -f "$OUT/stage1_poses.txt" ]; then
            conda run -n sgs python vggt_slam/v38_full_pipeline.py \
                --image_dir "$SEQ_DIR" \
                --output_dir "$OUT" \
                --submap_size 16 --min_disparity 50 --lc_thres 0.95 --max_loops 1 \
                --no_viz --skip_sam3 2>&1 | grep -E "Images|Keyframes|Error"
        else
            echo "    Stage1 已存在，跳过"
        fi

        # Step 2: 转换 GT pose.txt → TUM 格式
        GT_TUM="$OUT/gt_tum.txt"
        conda run -n sgs python tools/convert_7scenes_pose.py \
            --seq_dir "$SEQ_DIR" \
            --output "$GT_TUM" >/dev/null 2>&1

        # Step 3: 计算 Sim(3) 对齐后的 ATE
        ATE=$(conda run -n sgs evo_ape tum "$GT_TUM" "$OUT/stage1_poses.txt" \
              -as --no_warnings 2>/dev/null | grep -oP 'rmse\s+\K[\d.]+' | head -1)

        if [ -z "$ATE" ]; then
            ATE="N/A"
            echo "    ATE 计算失败"
        else
            echo "    ATE = $ATE m"
            SCENE_ATE_SUM[$scene]=$(echo "${SCENE_ATE_SUM[$scene]} + $ATE" | bc)
            SCENE_ATE_CNT[$scene]=$((SCENE_ATE_CNT[$scene] + 1))
        fi

        echo "$scene,$seq,$ATE" >> "$RESULTS_FILE"
    done

    # 场景平均
    CNT=${SCENE_ATE_CNT[$scene]}
    if [ "$CNT" -gt 0 ]; then
        SCENE_AVG=$(echo "scale=4; ${SCENE_ATE_SUM[$scene]} / $CNT" | bc)
        echo "  ▶ $scene 场景平均 ATE = $SCENE_AVG m (${CNT}个seq)"
        echo "$scene,ALL,$SCENE_AVG" >> "$RESULTS_FILE"
    fi
done

# ═══ 7-Scenes 整体平均 ═══
echo ""
echo "════════════════ 7-Scenes 整体 ════════════════"
TOTAL_SUM=0
TOTAL_CNT=0
for scene in chess fire heads office pumpkin redkitchen stairs; do
    CNT=${SCENE_ATE_CNT[$scene]}
    if [ "$CNT" -gt 0 ]; then
        SCENE_AVG=$(echo "scale=4; ${SCENE_ATE_SUM[$scene]} / $CNT" | bc)
        TOTAL_SUM=$(echo "scale=4; $TOTAL_SUM + $SCENE_AVG" | bc)
        TOTAL_CNT=$((TOTAL_CNT + 1))
    fi
done

if [ "$TOTAL_CNT" -gt 0 ]; then
    FINAL_ATE=$(echo "scale=4; $TOTAL_SUM / $TOTAL_CNT" | bc)
    echo "  7-Scenes 整体 ATE (${TOTAL_CNT}个场景平均) = $FINAL_ATE m"
    echo "7SCENES,ALL_SCENES,$FINAL_ATE" >> "$RESULTS_FILE"
fi

echo ""
echo "════════════════ 结果汇总 ════════════════"
cat "$RESULTS_FILE"
