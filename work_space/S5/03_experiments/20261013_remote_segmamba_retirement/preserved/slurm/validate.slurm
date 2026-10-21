#!/bin/bash
# =============================================================================
# SLURM 作业脚本: BraTS 2026 Validation Set 预测
# 适用集群: 华东师范大学超算八期 (V100 GPU)
# =============================================================================
# 使用方式:
#   sbatch slurm/validate.slurm                                    # 预测全部实验
#   EXP_NAMES=exp_dice_focal sbatch slurm/validate.slurm           # 预测指定实验
#   EXP_NAMES=exp_dice_focal,exp_ce_dice300epoch sbatch slurm/validate.slurm
# =============================================================================

# ---------------------------------------------------------------------------
# SLURM 资源申请
# ---------------------------------------------------------------------------
#SBATCH -J brats_validate
#SBATCH -p v100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

# ---------------------------------------------------------------------------
# 用户可配置变量
# ---------------------------------------------------------------------------
WORK_DIR="${WORK_DIR:-/public/home/zqchen/SegMamba-main}"
CONDA_ENV="${CONDA_ENV:-segmamba}"
# 实验列表: 逗号分隔, "all" 为全部
EXP_NAMES="${EXP_NAMES:-all}"

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

module load compiler/cuda/12.1

ENV_PYTHON="/public/home/zqchen/.conda/envs/${CONDA_ENV}/bin/python"
echo "Python: $($ENV_PYTHON --version)"
echo ""

cd "$WORK_DIR" || { echo "ERROR: 工作目录不存在: $WORK_DIR"; exit 1; }
mkdir -p logs/slurm

# ---------------------------------------------------------------------------
# 前置检查
# ---------------------------------------------------------------------------
if [[ ! -d "./data/validation" ]]; then
    echo "ERROR: 验证集目录不存在: ./data/validation"
    echo "请先用 WinSCP 上传验证集数据。"
    echo "目录结构: data/validation/BraTS-MET-XXXXX-XXX/t1c.nii.gz ..."
    exit 1
fi

# ---------------------------------------------------------------------------
# 运行预测
# ---------------------------------------------------------------------------
echo "实验: $EXP_NAMES"
echo ""

if [[ "$EXP_NAMES" == "all" ]]; then
    $ENV_PYTHON predict_validation.py --all
else
    $ENV_PYTHON predict_validation.py --exp "$EXP_NAMES"
fi

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "验证集预测完成 — $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================================="

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "预测结果:"
    if [[ "$EXP_NAMES" == "all" ]]; then
        for d in results/*/validation_predictions/; do
            if [[ -d "$d" ]]; then
                count=$(ls "$d"/*.nii.gz 2>/dev/null | wc -l)
                echo "  $d  ($count 个文件)"
            fi
        done
    else
        IFS=',' read -ra EXPS <<< "$EXP_NAMES"
        for exp in "${EXPS[@]}"; do
            exp=$(echo "$exp" | xargs)
            d="results/${exp}/validation_predictions/"
            if [[ -d "$d" ]]; then
                count=$(ls "$d"/*.nii.gz 2>/dev/null | wc -l)
                echo "  $d  ($count 个文件)"
            fi
        done
    fi
fi

exit $EXIT_CODE
