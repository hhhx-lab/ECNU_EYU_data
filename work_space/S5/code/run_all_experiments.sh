#!/bin/bash
# =============================================================================
# SegMamba BraTS 2026 - 运行全部实验脚本
# =============================================================================
# 使用方式:
#   bash run_all_experiments.sh              # 运行全部实验
#   bash run_all_experiments.sh --skip-train  # 跳过训练，只做预测和指标计算
#   bash run_all_experiments.sh exp_ce_dice   # 只运行指定实验
# =============================================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 实验列表
EXPERIMENTS=(
    "exp_ce_dice"
    "exp_dice_focal"
    "exp_ce_focal"
    "exp1_larger_patch"
    "exp2_adamw_cosine"
    "exp3_deeper_model"
    "exp4_no_augmentation"
    "exp5_high_overlap_infer"
)

# 默认参数
SKIP_TRAIN=false
SINGLE_EXP=""
LOG_DIR="./logs/experiments"

# ---------------------------------------------------------------------------
# 解析命令行参数
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash run_all_experiments.sh [options] [experiment_name]"
            echo ""
            echo "Options:"
            echo "  --skip-train       跳过训练，只运行预测和指标计算"
            echo "  --log-dir DIR      日志保存目录 (默认: ./logs/experiments)"
            echo "  -h, --help         显示帮助信息"
            echo ""
            echo "Examples:"
            echo "  bash run_all_experiments.sh                # 运行全部实验"
            echo "  bash run_all_experiments.sh exp_ce_dice     # 只运行 exp_ce_dice"
            echo "  bash run_all_experiments.sh --skip-train   # 跳过训练步骤"
            exit 0
            ;;
        *)
            # 检查是否是已知的实验名
            if [[ " ${EXPERIMENTS[@]} " =~ " $1 " ]]; then
                SINGLE_EXP="$1"
            else
                echo -e "${RED}Error: Unknown argument: $1${NC}"
                exit 1
            fi
            shift
            ;;
    esac
done

# 创建日志目录
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# 输出配置信息
# ---------------------------------------------------------------------------
echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  SegMamba BraTS 2026 实验运行器${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

if [[ -n "$SINGLE_EXP" ]]; then
    echo -e "${YELLOW}运行单个实验: $SINGLE_EXP${NC}"
    EXPERIMENTS=("$SINGLE_EXP")
else
    echo -e "${YELLOW}实验数量: ${#EXPERIMENTS[@]}${NC}"
fi

if [[ "$SKIP_TRAIN" == true ]]; then
    echo -e "${YELLOW}模式: 跳过训练 (仅预测和指标计算)${NC}"
else
    echo -e "${YELLOW}模式: 完整流程 (训练 + 预测 + 指标计算)${NC}"
fi

echo -e "${YELLOW}日志目录: $LOG_DIR${NC}"
echo ""

# ---------------------------------------------------------------------------
# 运行单个实验的函数
# ---------------------------------------------------------------------------
run_experiment() {
    local exp_name=$1
    local config_file="configs/${exp_name}.yaml"
    local exp_log_dir="$LOG_DIR/${exp_name}"
    local start_time=$(date +%s)

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  实验: $exp_name${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    # 创建实验日志目录
    mkdir -p "$exp_log_dir"

    # 检查配置文件是否存在
    if [[ ! -f "$config_file" ]]; then
        echo -e "${RED}Error: 配置文件不存在: $config_file${NC}"
        return 1
    fi

    # Step 1: 训练
    if [[ "$SKIP_TRAIN" == false ]]; then
        echo -e "${GREEN}[1/3] 训练模型...${NC}"
        echo -e "${YELLOW}命令: python 3_train_brats2026.py --config $config_file${NC}"

        if python 3_train_brats2026.py --config "$config_file" 2>&1 | tee "$exp_log_dir/train.log"; then
            echo -e "${GREEN}✓ 训练完成${NC}"
        else
            echo -e "${RED}✗ 训练失败，请查看日志: $exp_log_dir/train.log${NC}"
            return 1
        fi
        echo ""
    else
        echo -e "${YELLOW}[1/3] 跳过训练${NC}"
    fi

    # Step 2: 预测
    echo -e "${GREEN}[2/3] 生成预测...${NC}"
    echo -e "${YELLOW}命令: python 4_predict_brats2026.py --config $config_file${NC}"

    if python 4_predict_brats2026.py --config "$config_file" 2>&1 | tee "$exp_log_dir/predict.log"; then
        echo -e "${GREEN}✓ 预测完成${NC}"
    else
        echo -e "${RED}✗ 预测失败，请查看日志: $exp_log_dir/predict.log${NC}"
        return 1
    fi
    echo ""

    # Step 3: 计算指标
    echo -e "${GREEN}[3/3] 计算指标...${NC}"
    echo -e "${YELLOW}命令: python 5_compute_metrics_brats2026.py --config $config_file${NC}"

    if python 5_compute_metrics_brats2026.py --config "$config_file" 2>&1 | tee "$exp_log_dir/metrics.log"; then
        echo -e "${GREEN}✓ 指标计算完成${NC}"
    else
        echo -e "${RED}✗ 指标计算失败，请查看日志: $exp_log_dir/metrics.log${NC}"
        return 1
    fi
    echo ""

    # 计算耗时
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  实验 $exp_name 完成!${NC}"
    echo -e "${GREEN}  耗时: ${hours}h ${minutes}m ${seconds}s${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    return 0
}

# ---------------------------------------------------------------------------
# 主循环：运行所有实验
# ---------------------------------------------------------------------------
TOTAL_EXP=${#EXPERIMENTS[@]}
CURRENT_EXP=0
SUCCESS_COUNT=0
FAILED_EXPS=()

for exp_name in "${EXPERIMENTS[@]}"; do
    CURRENT_EXP=$((CURRENT_EXP + 1))
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  进度: $CURRENT_EXP / $TOTAL_EXP${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    if run_experiment "$exp_name"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAILED_EXPS+=("$exp_name")
    fi
done

# ---------------------------------------------------------------------------
# 汇总报告
# ---------------------------------------------------------------------------
echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  实验汇总报告${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo -e "总实验数: ${TOTAL_EXP}"
echo -e "成功: ${GREEN}${SUCCESS_COUNT}${NC}"
echo -e "失败: ${RED}$((TOTAL_EXP - SUCCESS_COUNT))${NC}"

if [[ ${#FAILED_EXPS[@]} -gt 0 ]]; then
    echo ""
    echo -e "${RED}失败的实验:${NC}"
    for failed_exp in "${FAILED_EXPS[@]}"; do
        echo -e "  - $failed_exp"
    done
    echo ""
    echo -e "${YELLOW}请查看失败实验的日志文件了解详情:${NC}"
    for failed_exp in "${FAILED_EXPS[@]}"; do
        echo -e "  $LOG_DIR/$failed_exp/"
    done
fi

echo ""
echo -e "${GREEN}全部实验运行完成!${NC}"
echo -e "${YELLOW}日志文件保存在: $LOG_DIR${NC}"
echo ""

# 输出结果文件位置
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  输出文件位置${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo "模型权重:  ./logs/segmamba_<实验名>/model/"
echo "预测结果:  ./prediction_results/segmamba_brats2026_<实验名>/"
echo "指标文件:  ./prediction_results/result_metrics_<实验名>/"
echo ""
