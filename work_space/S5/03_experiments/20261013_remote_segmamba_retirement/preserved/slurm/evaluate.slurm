#!/bin/bash
# =============================================================================
# SLURM 作业脚本: SegMamba BraTS 2026 独立评估
# 适用集群: 华东师范大学超算八期
# 使用 CPU 分区运行官方 Panoptica 评估 (不需要 GPU)
# =============================================================================
# 使用方式:
#   sbatch slurm/evaluate.slurm
#   EXP_NAMES=exp_ce_dice sbatch slurm/evaluate.slurm
#   EXP_NAMES="exp_ce_dice exp_dice_focal" sbatch slurm/evaluate.slurm
# =============================================================================

# ---------------------------------------------------------------------------
# SLURM 资源申请 (CPU 即可, Panoptica 不需要 GPU)
# ---------------------------------------------------------------------------
#SBATCH -J brats_eval
#SBATCH -p cpu_96G
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

# ---------------------------------------------------------------------------
# 用户可配置变量
# ---------------------------------------------------------------------------
WORK_DIR="${WORK_DIR:-/public/home/zqchen/SegMamba-main}"
CONDA_ENV="${CONDA_ENV:-segmamba}"

# 要评估的实验列表 (空格分隔), 设为 "all" 评估所有已有预测结果的实验
EXP_NAMES="${EXP_NAMES:-all}"

# 原始数据 (ground truth) 目录
RAW_DATA_DIR="${RAW_DATA_DIR:-/public/home/zqchen/data/}"

# ---------------------------------------------------------------------------
# 环境初始化
# ---------------------------------------------------------------------------
echo "=============================================="
echo "SLURM Job ID:    ${SLURM_JOB_ID}"
echo "Job Name:        ${SLURM_JOB_NAME}"
echo "Partition:       ${SLURM_JOB_PARTITION}"
echo "Start Time:      $(date)"
echo "=============================================="
echo ""

# 使用 conda 环境的绝对路径 Python (最稳妥, 避开 PATH 和 Shell 初始化问题)
ENV_PYTHON="/public/home/zqchen/.conda/envs/${CONDA_ENV}/bin/python"

echo "Python: $($ENV_PYTHON --version)"
echo ""

cd "$WORK_DIR" || {
    echo "ERROR: 工作目录不存在: $WORK_DIR"
    exit 1
}

mkdir -p logs/slurm logs/experiments

# ---------------------------------------------------------------------------
# 确定要评估的实验列表
# ---------------------------------------------------------------------------
if [[ "$EXP_NAMES" == "all" ]]; then
    # 自动发现所有已有预测结果的实验
    EXP_NAMES=""
    for pred_dir in results/*/predictions/; do
        if [[ -d "$pred_dir" ]]; then
            exp_name=$(basename "$(dirname "$pred_dir")")
            # 跳过 default
            if [[ "$exp_name" != "default" ]]; then
                EXP_NAMES="$EXP_NAMES $exp_name"
            fi
        fi
    done
fi

if [[ -z "$EXP_NAMES" ]]; then
    echo "ERROR: 没有找到可评估的实验。"
    echo "请先运行预测，或手动指定 EXP_NAMES。"
    exit 1
fi

echo "待评估实验: $EXP_NAMES"
echo "原始数据:   $RAW_DATA_DIR"
echo ""

# ---------------------------------------------------------------------------
# 逐个实验评估
# ---------------------------------------------------------------------------
TOTAL=$(echo "$EXP_NAMES" | wc -w)
CURRENT=0
FAILED=""

for exp_name in $EXP_NAMES; do
    CURRENT=$((CURRENT + 1))
    echo "=============================================="
    echo "[$CURRENT/$TOTAL] 评估: $exp_name — $(date '+%H:%M:%S')"
    echo "=============================================="

    PRED_DIR="./results/${exp_name}/predictions"

    if [[ ! -d "$PRED_DIR" ]]; then
        echo "WARNING: 预测目录不存在: $PRED_DIR, 跳过"
        FAILED="$FAILED $exp_name(无预测)"
        continue
    fi

    PRED_COUNT=$(ls "$PRED_DIR"/*.nii.gz 2>/dev/null | wc -l)
    echo "预测文件: $PRED_COUNT 个"

    $ENV_PYTHON 5_compute_metrics_brats2026.py \
        --config "configs/${exp_name}.yaml" \
        --raw_data_dir "$RAW_DATA_DIR" \
        --output_dir "./results/${exp_name}/metrics" \
        2>&1 | tee "logs/experiments/${exp_name}/metrics.log"

    EXIT_CODE=${PIPESTATUS[0]}
    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "✓ $exp_name 评估完成"
    else
        echo "✗ $exp_name 评估失败 (exit code: $EXIT_CODE)"
        FAILED="$FAILED $exp_name(exit=$EXIT_CODE)"
    fi
    echo ""
done

# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
echo "=============================================="
echo "评估汇总 — $(date)"
echo "=============================================="
echo "总计: $TOTAL 个实验"

if [[ -n "$FAILED" ]]; then
    echo "失败: $FAILED"
fi

echo ""
echo "评估结果目录:"
for exp_name in $EXP_NAMES; do
    csv_path="./results/${exp_name}/metrics/segmentation_metrics.csv"
    if [[ -f "$csv_path" ]]; then
        echo "  ✓ $exp_name"
        echo "  --- $exp_name 均值 ---"
        grep "mean," "$csv_path"
    else
        echo "  ✗ $exp_name"
    fi
done
