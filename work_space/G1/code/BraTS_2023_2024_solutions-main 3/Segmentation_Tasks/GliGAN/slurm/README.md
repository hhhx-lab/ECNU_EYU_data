# Diffusion augmentation V3：ECNU Slurm 入口

更新日期：2026-07-15

本目录只负责从 `seg` 学习并生成四模态病灶影像，不是缺失 T2W 填补线。

## 固定顺序

| 顺序 | 脚本 | 资源 | 作用 |
|---:|---|---|---|
| 0A | `00_smoke_v3_ecnu.slurm` | 1 x A100（生产提交覆盖默认分区） | 小尺寸端到端 GPU 回归 |
| 0B | `00_preflight_crop64_v3_ecnu.slurm` | 1 x V100 | 正式 `64^3/batch=4/宽网络` 显存预检 |
| 1 | `01_prepare_dataset_v3_ecnu.slurm` | CPU | 依据 G2 master manifest 建 823/103 数据视图并生成 lesion CSV |
| 2 | `02_train_4modal_v3_ecnu.slurm` | 4 个单卡 A100 array task | 独立训练 `t1c/t1n/t2w/t2f` |
| 3 | `03_eval_4modal_v3_ecnu.slurm` | 1 x A100 | 固定 103 例 val 的四模态 whole-brain 评估 |

## 推荐提交

```bash
export PROD_ROOT=/public/home/${USER}/g1_diffusion_v3_production_20260715
export SMOKE_ROOT=/public/home/${USER}/g1_diffusion_v3_smoke_20260715
mkdir -p "${PROD_ROOT}/logs" "${SMOKE_ROOT}/logs"
cd "${PROD_ROOT}/code/Segmentation_Tasks/GliGAN"

SMOKE_JOB=$(sbatch --parsable -p a100 slurm/00_smoke_v3_ecnu.slurm)
PREFLIGHT_JOB=$(sbatch --parsable slurm/00_preflight_crop64_v3_ecnu.slurm)
PREP_JOB=$(sbatch --parsable slurm/01_prepare_dataset_v3_ecnu.slurm)
TRAIN_JOB=$(sbatch --parsable \
  --dependency=afterok:${SMOKE_JOB}:${PREFLIGHT_JOB}:${PREP_JOB} \
  slurm/02_train_4modal_v3_ecnu.slurm)
EVAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} \
  slurm/03_eval_4modal_v3_ecnu.slurm)
printf 'smoke=%s preflight=%s prep=%s train=%s eval=%s\n' \
  "${SMOKE_JOB}" "${PREFLIGHT_JOB}" "${PREP_JOB}" "${TRAIN_JOB}" "${EVAL_JOB}"
```

前三项并行执行，训练只在三项全部成功后启动。smoke/preflight 默认使用提交命令所在的 `SLURM_SUBMIT_DIR`，因此必须先进入上面的 production 代码目录。

`02_train_4modal_v3_ecnu.slurm` 是一个文件、四个 array task：

```text
0=t1c, 1=t1n, 2=t2w, 3=t2f
```

每个 task 独占一张 A100，独立排队和失败重跑。ECNU `a100` 节点每台只有 2 张 GPU，因此不要申请同节点 4 卡。

## 默认生产参数

```text
crop_size=64
batch_size=4
network_channels=48,96,192,384
network_strides=2,2,2
normalization=zscore
sigma_data=1.0
noise_schedule=edm
small_lesion_weight=3.0
patient_balance_mode=sqrt
num_steps=100000
```

ECNU 系统 GCC 4.8 缺少 Triton 所需 `stdatomic.h`，所以 `USE_COMPILE=0` 是服务器已验证默认值。禁止直接打开 `torch.compile`；只有加载并验证新 GCC 后才可显式传 `USE_COMPILE=1`。

## 断点续训

脚本默认 `AUTO_RESUME=1`，会按数值选择当前模态最大的 `diffusion_<step>.pt`。只重跑一个模态：

```bash
sbatch --array=2 slurm/02_train_4modal_v3_ecnu.slurm
```

## 通过标记

日志必须出现：

```text
PREPARE_DATASET_PASS
TRAIN_MODALITY_PASS modality=<modality>
EVALUATION_PASS metrics=<path>/metrics.json
```

任一任务非零退出时，下游 `afterok` 依赖不会启动。
