# Diffusion augmentation V3：NYU Greene Slurm 入口

更新日期：2026-07-15

本目录只负责从 `seg` 学习并生成四模态病灶影像，不是缺失 T2W 填补线。

脚本风格对齐：

```text
work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN/slurm/*_v2_nyu.slurm
```

正式代码入口：

```text
work_space/G1/code/BraTS_2023_2024_solutions-main 3/Segmentation_Tasks/GliGAN
```

## Scripts

| Script | GPU | Purpose |
|---|---:|---|
| `00_smoke_v3_nyu.slurm` | 1 | 小尺寸端到端 GPU 回归 |
| `00_preflight_crop64_v3_nyu.slurm` | 1 | 正式 `64^3/batch` + 宽网络一步训练预检 |
| `01_prepare_dataset_v3_nyu.slurm` | 0 | 依据 G2 master manifest 建 823/103 软链接视图 + lesion CSV |
| `02_train_4modal_v3_nyu.slurm` | 4 | 单节点四卡并行训练 `t1c/t1n/t2w/t2f` |
| `03_eval_4modal_v3_nyu.slurm` | 1 | 固定 103 val 四模态 whole-brain 评估 |
| `04_generate_visual_v3_nyu.slurm` | 1 | 单病例可视化，不是批量 production 入口 |

## Before Submit

1. 仓库放在 NYU scratch：`/scratch/bf2260/ECNU_EYU_data`（可用 `PROJ` 覆盖）。
2. G2 source manifest 存在：
   `work_space/G2/results/manifests/g1_v2_source_manifest.csv`
3. 原始数据与 corrected seg 可按 manifest 相对路径解析（相对 `PROJ`）。
4. 创建/激活 Conda 环境，默认名 `g1_diffusion_v3`：

```bash
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate g1_diffusion_v3
# 若已有兼容环境：export CONDA_ENV=brats 或 g1_diffusion_v2
```

5. 日志目录：

```bash
mkdir -p /scratch/bf2260/ECNU_EYU_data/logs
```

## Recommended Submit

```bash
export PROJ=/scratch/bf2260/ECNU_EYU_data
export RUN_ROOT=${PROJ}/runs/g1_diffusion_v3
mkdir -p "${PROJ}/logs" "${RUN_ROOT}"
cd "${PROJ}/work_space/G1/code/BraTS_2023_2024_solutions-main 3/Segmentation_Tasks/GliGAN"

SMOKE_JOB=$(sbatch --parsable slurm/00_smoke_v3_nyu.slurm)
PREFLIGHT_JOB=$(sbatch --parsable slurm/00_preflight_crop64_v3_nyu.slurm)
PREP_JOB=$(sbatch --parsable slurm/01_prepare_dataset_v3_nyu.slurm)
TRAIN_JOB=$(sbatch --parsable \
  --dependency=afterok:${SMOKE_JOB}:${PREFLIGHT_JOB}:${PREP_JOB} \
  slurm/02_train_4modal_v3_nyu.slurm)
EVAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} \
  slurm/03_eval_4modal_v3_nyu.slurm)
sbatch --dependency=afterok:${TRAIN_JOB} slurm/04_generate_visual_v3_nyu.slurm

printf 'smoke=%s preflight=%s prep=%s train=%s eval=%s\n' \
  "${SMOKE_JOB}" "${PREFLIGHT_JOB}" "${PREP_JOB}" "${TRAIN_JOB}" "${EVAL_JOB}"
```

只跑生产链时，可跳过 smoke/preflight：

```bash
PREP_JOB=$(sbatch --parsable slurm/01_prepare_dataset_v3_nyu.slurm)
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${PREP_JOB} \
  slurm/02_train_4modal_v3_nyu.slurm)
sbatch --dependency=afterok:${TRAIN_JOB} slurm/03_eval_4modal_v3_nyu.slurm
```

## Default Production Parameters

```text
crop_size=64
batch_size=8          # NYU 默认；OOM 时 BATCH_SIZE=4
network_channels=48,96,192,384
network_strides=2,2,2
normalization=zscore
sigma_data=1.0
noise_schedule=edm
small_lesion_weight=3.0
patient_balance_mode=sqrt
num_steps=100000
USE_COMPILE=1         # NYU 默认可开；失败则 USE_COMPILE=0
AUTO_RESUME=1
```

训练脚本在**一个节点 4 张 GPU**上各起一个模态进程，不使用 ECNU 式 array 四任务。

## Paths

默认 run 根目录：

```text
/scratch/bf2260/ECNU_EYU_data/runs/g1_diffusion_v3/
  DataSet/
  splits/current/lesions.csv
  checkpoints/brats2026_diffusion_v3_edm_zscore/<modality>/weights/
  eval/
  visual/
  PREPARED.ok
```

## Useful Overrides

```bash
# 换账号路径
PROJ=/scratch/$USER/ECNU_EYU_data sbatch slurm/01_prepare_dataset_v3_nyu.slurm

# 显存不够
BATCH_SIZE=4 USE_COMPILE=0 sbatch slurm/02_train_4modal_v3_nyu.slurm

# 快速评估
MAX_CASES=20 EVAL_MODE=patch sbatch slurm/03_eval_4modal_v3_nyu.slurm

# 指定可视化病例
CASE_ID=BraTS-MET-00004-000 sbatch slurm/04_generate_visual_v3_nyu.slurm
```

若集群要求特定 GPU 分区/约束（例如 A100），在脚本顶部自行补：

```bash
#SBATCH --partition=...
#SBATCH --constraint=...
```

默认请求是 Greene 通用写法：`#SBATCH --gres=gpu:N`。

## Pass Markers

日志必须出现：

```text
SMOKE_TEST_PASS
CROP64_BATCH4_EAGER_PASS
PREPARE_DATASET_PASS
TRAIN_MODALITY_PASS modality=<modality>
EVALUATION_PASS metrics=<path>/metrics.json
```

任一上游非零退出时，`afterok` 依赖不会启动下游。

## G2 Handoff

G1 输出是病灶区域 raw diffusion 结果。批量生成后走：

```bash
python work_space/G2/code/g2_v2_compose_augmentation.py \
  --v2-output-root /path/to/v3_flat_output \
  --source-manifest work_space/G2/results/manifests/g1_v2_source_manifest.csv \
  --output-run-root /path/to/g2_composed/v3_run_id

python work_space/G2/code/g2_synthetic_raw_intake_qc.py \
  --synthetic-run-root /path/to/g2_composed/v3_run_id \
  --generation-mode full_generation
```

G2 接口名里的 `v2` 是 augmentation 契约名，不代表继续使用 `main 2` 训练代码。
