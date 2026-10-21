#!/bin/bash
# =============================================================================
# SLURM 作业脚本: SegMamba BraTS 2026 单实验训练
# 适用集群: 华东师范大学超算八期 (A100 GPU)
# =============================================================================
# 使用方式:
#   sbatch slurm/train_single.slurm
#   EXP_NAME=exp_ce_dice sbatch slurm/train_single.slurm
#   RESUME=true EXP_NAME=exp_ce_dice sbatch slurm/train_single.slurm
# =============================================================================

# ---------------------------------------------------------------------------
# SLURM 资源申请
# ---------------------------------------------------------------------------
#SBATCH -J segmamba_train
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 72:00:00
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

# ---------------------------------------------------------------------------
# 用户可配置变量 (也可通过环境变量覆盖)
# ---------------------------------------------------------------------------
# 工作目录 — 修改为你的实际路径
WORK_DIR="${WORK_DIR:-/public/home/zqchen/SegMamba-main}"
# Conda 环境名
CONDA_ENV="${CONDA_ENV:-segmamba}"
# 实验名称 (exp_ce_dice / exp_dice_focal / exp_ce_focal / exp1_larger_patch / ...)
EXP_NAME="${EXP_NAME:-exp_ce_dice}"
# 是否跳过训练 (仅做预测+评估): "true" 或 "false"
SKIP_TRAIN="${SKIP_TRAIN:-false}"
# 是否断点续训: "true" 或 "false"
RESUME="${RESUME:-false}"

# ---------------------------------------------------------------------------
# 环境初始化
# ---------------------------------------------------------------------------
echo "=============================================="
echo "SLURM Job ID:    ${SLURM_JOB_ID}"
echo "Job Name:        ${SLURM_JOB_NAME}"
echo "Partition:       ${SLURM_JOB_PARTITION}"
echo "Node:            ${SLURM_NODELIST}"
echo "GPUs:            ${SLURM_GPUS}"
echo "Start Time:      $(date)"
echo "=============================================="
echo ""

# 加载 CUDA 运行时 (Python 用绝对路径, 不需要 conda module)
module load compiler/cuda/12.1

# 使用 conda 环境的绝对路径 Python (最稳妥, 避开 PATH 和 Shell 初始化问题)
ENV_PYTHON="/public/home/zqchen/.conda/envs/${CONDA_ENV}/bin/python"

echo "Python: $($ENV_PYTHON --version)"
echo "CUDA:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# ---------------------------------------------------------------------------
# 切换到工作目录
# ---------------------------------------------------------------------------
cd "$WORK_DIR" || {
    echo "ERROR: 工作目录不存在: $WORK_DIR"
    echo "请修改脚本中的 WORK_DIR 变量。"
    exit 1
}

# 创建日志目录
mkdir -p logs/slurm logs/experiments

# ---------------------------------------------------------------------------
# 前置检查
# ---------------------------------------------------------------------------
CONFIG_FILE="configs/${EXP_NAME}.yaml"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: 配置文件不存在: $CONFIG_FILE"
    echo "可用的实验:"
    ls configs/*.yaml 2>/dev/null | sed 's|configs/||;s|\.yaml||'
    exit 1
fi

if [[ ! -d "./data/fullres/train" ]]; then
    echo "ERROR: 预处理数据不存在: ./data/fullres/train"
    echo "请先在登录节点运行:"
    echo "  python 1_rename_mri_data.py"
    echo "  python 2_preprocessing_mri.py"
    exit 1
fi

echo "实验: $EXP_NAME"
echo "配置: $CONFIG_FILE"
echo ""

# ---------------------------------------------------------------------------
# 运行实验
# ---------------------------------------------------------------------------
if [[ "$SKIP_TRAIN" == "true" ]]; then
    echo "[模式] 跳过训练，仅预测 + 评估"
else
    echo "[模式] 完整流程: 训练 + 预测 + 评估"
fi
echo ""

EXP_LOG_DIR="logs/experiments/${EXP_NAME}"
mkdir -p "$EXP_LOG_DIR"

# --- Step 1: 训练 ---
if [[ "$SKIP_TRAIN" != "true" ]]; then
    echo "=============================================="
    echo "[1/3] 训练模型 — $(date '+%H:%M:%S')"
    if [[ "$RESUME" == "true" ]]; then
        echo "[模式] 断点续训 (--resume)"
    fi
    echo "=============================================="
    TRAIN_CMD="$ENV_PYTHON 3_train_brats2026.py --config $CONFIG_FILE"
    if [[ "$RESUME" == "true" ]]; then
        TRAIN_CMD="$TRAIN_CMD --resume"
    fi
    $TRAIN_CMD 2>&1 | tee "$EXP_LOG_DIR/train.log"
    TRAIN_EXIT=${PIPESTATUS[0]}
    if [[ $TRAIN_EXIT -ne 0 ]]; then
        echo "ERROR: 训练失败 (exit code: $TRAIN_EXIT)"
        exit $TRAIN_EXIT
    fi
    echo "训练完成 — $(date '+%H:%M:%S')"
else
    echo "[1/3] 跳过训练"
fi
echo ""

# --- Step 2: 预测 ---
echo "=============================================="
echo "[2/3] 生成预测 — $(date '+%H:%M:%S')"
echo "=============================================="
$ENV_PYTHON 4_predict_brats2026.py --config "$CONFIG_FILE" 2>&1 | tee "$EXP_LOG_DIR/predict.log"
PRED_EXIT=${PIPESTATUS[0]}
if [[ $PRED_EXIT -ne 0 ]]; then
    echo "ERROR: 预测失败 (exit code: $PRED_EXIT)"
    exit $PRED_EXIT
fi
echo "预测完成 — $(date '+%H:%M:%S')"
echo ""

# --- Step 3: 评估 (官方 Panoptica 管线) ---
echo "=============================================="
echo "[3/3] 计算指标 — $(date '+%H:%M:%S')"
echo "=============================================="
$ENV_PYTHON 5_compute_metrics_brats2026.py --config "$CONFIG_FILE" 2>&1 | tee "$EXP_LOG_DIR/metrics.log"
METRICS_EXIT=${PIPESTATUS[0]}

echo ""
echo "=============================================="
echo "实验完成 — $(date)"
echo "=============================================="

if [[ $METRICS_EXIT -eq 0 ]]; then
    echo ""
    echo "输出文件:"
    echo "  全部产出: ./results/${EXP_NAME}/"
    echo "    ├── model/          模型权重"
    echo "    ├── visualizations/ 训练曲线"
    echo "    ├── predictions/    预测 NIfTI"
    echo "    └── metrics/        评估指标"
    echo "  运行日志: $EXP_LOG_DIR/"
else
    echo "WARNING: 指标计算步骤有错误 (exit code: $METRICS_EXIT)"
    echo "训练和预测已完成，可手动运行评估:"
    echo "  python 5_compute_metrics_brats2026.py --config $CONFIG_FILE"
fi

exit $METRICS_EXIT