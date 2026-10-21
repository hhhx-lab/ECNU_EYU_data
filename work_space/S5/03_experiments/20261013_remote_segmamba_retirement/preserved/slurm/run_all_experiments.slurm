#!/bin/bash
# =============================================================================
# SLURM 作业脚本: SegMamba BraTS 2026 全部实验批量运行
# 适用集群: 华东师范大学超算八期 (A100 GPU)
# =============================================================================
# 使用方式:
#   sbatch slurm/run_all_experiments.slurm
#   SKIP_TRAIN=true sbatch slurm/run_all_experiments.slurm
# =============================================================================

# ---------------------------------------------------------------------------
# SLURM 资源申请
# ---------------------------------------------------------------------------
#SBATCH -J segmamba_all
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 7-00:00:00
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

# ---------------------------------------------------------------------------
# 用户可配置变量
# ---------------------------------------------------------------------------
WORK_DIR="${WORK_DIR:-/public/home/zqchen/SegMamba-main}"
CONDA_ENV="${CONDA_ENV:-segmamba}"
SKIP_TRAIN="${SKIP_TRAIN:-false}"

# ---------------------------------------------------------------------------
# 环境初始化
# ---------------------------------------------------------------------------
echo "=============================================="
echo "SLURM Job ID:    ${SLURM_JOB_ID}"
echo "Job Name:        ${SLURM_JOB_NAME}"
echo "Partition:       ${SLURM_JOB_PARTITION}"
echo "Node:            ${SLURM_NODELIST}"
echo "Start Time:      $(date)"
echo "=============================================="
echo ""

# 加载 CUDA 运行时 (Python 用绝对路径, 不需要 conda module)
module load compiler/cuda/12.1

# 使用 conda 环境的绝对路径 Python (最稳妥, 避开 PATH 和 Shell 初始化问题)
ENV_PYTHON="/public/home/zqchen/.conda/envs/${CONDA_ENV}/bin/python"
export ENV_PYTHON

echo "Python: $($ENV_PYTHON --version)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

cd "$WORK_DIR" || {
    echo "ERROR: 工作目录不存在: $WORK_DIR"
    exit 1
}

mkdir -p logs/slurm logs/experiments

# ---------------------------------------------------------------------------
# 前置检查
# ---------------------------------------------------------------------------
if [[ ! -d "./data/fullres/train" ]]; then
    echo "ERROR: 预处理数据不存在: ./data/fullres/train"
    exit 1
fi

DATA_COUNT=$(ls ./data/fullres/train/*.npz 2>/dev/null | wc -l)
echo "预处理数据: $DATA_COUNT 个 npz 文件"
echo ""

# ---------------------------------------------------------------------------
# 运行全部实验
# ---------------------------------------------------------------------------
ARGS="--log-dir ./logs/experiments"
if [[ "$SKIP_TRAIN" == "true" ]]; then
    ARGS="$ARGS --skip-train"
fi

echo "命令: bash run_all_experiments.sh $ARGS"
echo "开始时间: $(date)"
echo ""

bash run_all_experiments.sh $ARGS
EXIT_CODE=$?

echo ""
echo "=============================================="
echo "全部实验结束 — $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================================="
echo ""
echo "查看日志:  logs/experiments/"
echo "评估结果:  evaluation_results/"

exit $EXIT_CODE