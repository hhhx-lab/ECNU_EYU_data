# G1 V3 ECNU Slurm 执行说明

本目录只用于最新 V3 缺失 T2W 填补线。模型学习
`t1n + t1c + t2f -> t2w`；扩散样本增强是另一条 G1 代码线，不在这里。

## 1. 固定路径与环境

```bash
PROJECT_ROOT=/public/home/${USER}/projects/ECNU_EYU_data
CODE_DIR=${PROJECT_ROOT}/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v3
SOURCE_DATA_DIR=${PROJECT_ROOT}/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v2/data

cd "${CODE_DIR}"
mkdir -p logs
```

- Conda 环境：服务器已有 `segmamba`。
- CUDA 模块：`compiler/cuda/12.1`。
- GPU：单张 `NVIDIAA100-PCIE-40GB`。
- 不在作业里安装依赖，不新建环境。
- Slurm 分配的物理 GPU 在作业内映射为 `cuda:0`。

## 2. 七个阶段

| 阶段 | 脚本 | 是否现在提交 | 输入 | 输出 |
|---|---|---|---|---|
| 0 | `00_smoke.slurm` | 是 | V3 代码、`segmamba`、预训练 VAE | CUDA/依赖/模型前向检查 |
| 1 | `01_adopt_prepared_data.slurm` | 是，依赖 0 | V2 已验证的数据链接和固定 split | V3 独立 metadata，复用 NIfTI 链接 |
| 2 | `02_finetune_vae.slurm` | 是，依赖 1 | 823 train、103 val、预训练 VAE | patch 微调、全 val 对比、VAE 选择 |
| 3 | `03_encode_and_prepare_aux.slurm` | VAE 人工验收后 | 被选中的 VAE、1030 完整病例 | 4120 latents、mask、lesion weight、channel weight |
| 4 | `04_train_models.slurm` | 阶段 3 后，两个 job 并行 | train latents 和辅助权重 | EncDec 或 BBDM checkpoint |
| 5 | `05_evaluate_val.slurm` | 两模型都完成后 | 103 val、两模型 checkpoint | 完整 val 指标、重建 NIfTI、运行清单 |
| 6 | `06_infer_missing_t2w.slurm` | val 人工批准后 | 265 个真缺 T2W 病例 | 每病例完整五文件，交给 G2 QC |

阶段 2 与 3 之间、阶段 5 与 6 之间都有人工门。不要把七个阶段一次性串完。

## 3. 当前应提交的作业

```bash
cd "${CODE_DIR}"
mkdir -p logs

SMOKE_JOB=$(sbatch --parsable slurm/00_smoke.slurm)
ADOPT_JOB=$(G1_V3_SOURCE_DATA_DIR="${SOURCE_DATA_DIR}" \
  sbatch --parsable --dependency=afterok:${SMOKE_JOB} \
  slurm/01_adopt_prepared_data.slurm)
VAE_JOB=$(sbatch --parsable --dependency=afterok:${ADOPT_JOB} \
  slurm/02_finetune_vae.slurm)

printf 'SMOKE=%s ADOPT=%s VAE=%s\n' "${SMOKE_JOB}" "${ADOPT_JOB}" "${VAE_JOB}"
```

阶段 1 只复用 V2 已验证的 `input/`、`input_inference/` 和 split metadata；
不会复制 40GB NIfTI，也不会复用 V2 latent、VAE 输出或模型 checkpoint。预期：

- 完整病例 CSV：1030。
- train / val / locked test：823 / 103 / 104。
- 真缺 T2W 推理病例：265。
- 排除非法标签：`BraTS-MET-01094-002`。
- 使用 corrected label：`BraTS-MET-01184-002`。

监控：

```bash
squeue -u ${USER}
tail -f logs/g1v3_smoke_${SMOKE_JOB}.out
tail -f logs/g1v3_adopt_${ADOPT_JOB}.out
tail -f logs/g1v3_vae_${VAE_JOB}.out
```

## 4. VAE 默认方案与验收

阶段 2 固定执行：

1. 用预训练 VAE 对全部 103 个 val 建立完整体积 baseline。
2. 全部 823 个 train 参与训练。
3. 每病例同步裁剪四模态与 seg，patch 为 `128×128×96`。
4. 80% 均匀选择一个肿瘤连通域中心；20% 随机脑区。
5. BF16、关闭 activation checkpointing、最多 3 epoch。
6. 每 epoch 固定 20 个 val 快速验证，tumor MSE patience=2。
7. 最后重新跑全部 103 val、四模态，共 412 行 delta。

输出：

```text
training/vae_finetuned/run_<jobid>/
├── best_model.pt
├── baseline_metrics.csv
├── finetuned_metrics.csv
├── delta_metrics.csv
├── vae_selection.json
├── finetune_config.json
├── training_history.csv
├── quick_val_subjects.json
└── comparison_samples/*.png
```

自动门槛：mean `delta_tumor_SSIM >= 0.03` 且 mean
`delta_whole_SSIM >= -0.005`。未达标时 `selected_weights` 自动指向原始 VAE，
不是强行采用微调权重。操作者仍需查看 5 张最差病例图：

```bash
VAE_RUN_DIR=$(head -n 1 training/vae_finetuned/latest_run.txt)
python -m json.tool "${VAE_RUN_DIR}/vae_selection.json"
ls -lh "${VAE_RUN_DIR}/comparison_samples"
```

## 5. VAE 验收后生成训练输入

确认 VAE 选择后：

```bash
ENCODE_JOB=$(sbatch --parsable slurm/03_encode_and_prepare_aux.slurm)
echo "ENCODE_JOB=${ENCODE_JOB}"
```

阶段 3 会先删除 V3 自己的旧 latent/mask/lesion-weight，再统一重建。成功条件：

- 1030 个 latent 目录、4120 个 `*_latent.npy`。
- 1030 个 latent attention mask。
- 1030 个 latent lesion-weight mask。
- `channel_weights.json` 明确记录 823 个 train 病例。
- `data/selected_vae.json` 固定本轮 VAE。

## 6. 并行训练 EncDec 与 BBDM

阶段 3 成功后，两模型独立申请一张 A100，可并行：

```bash
ENDEC_JOB=$(TRAIN_TARGET=endec sbatch --parsable \
  --dependency=afterok:${ENCODE_JOB} slurm/04_train_models.slurm)
BBDM_JOB=$(TRAIN_TARGET=bbdm sbatch --parsable \
  --dependency=afterok:${ENCODE_JOB} slurm/04_train_models.slurm)
printf 'ENDEC=%s BBDM=%s\n' "${ENDEC_JOB}" "${BBDM_JOB}"
```

默认完全按 V3 README 推荐值：

| 模型 | batch | steps | seg loss | weight decay | BBDM s |
|---|---:|---:|---|---:|---:|
| EncDec | 12 | 67000 | 开 | 0.0 | - |
| BBDM | 8 | 201000 | 开 | 0.0 | 0.01 |

OOM 时先取消对应 job，只降低对应 batch 后重提。不要同时改 `s` 和 weight decay。
恢复单个模型：

```bash
TRAIN_TARGET=bbdm \
G1_RESUME_CHECKPOINT=/absolute/path/model_50000.pt \
G1_MODEL_RUN_ID=bbdm_resume_50000 \
sbatch slurm/04_train_models.slurm
```

## 7. Val 与最终缺失 T2W 推理

两模型完成后：

```bash
EVAL_JOB=$(sbatch --parsable \
  --dependency=afterok:${ENDEC_JOB}:${BBDM_JOB} \
  slurm/05_evaluate_val.slurm)
echo "EVAL_JOB=${EVAL_JOB}"
```

阶段 5 必须成功处理全部 103 val，输出：

```text
data/evaluation/val/run_<jobid>/
├── metrics.csv
├── evaluation_run.json
└── synthesized/*.nii.gz
```

检查 `metrics.csv`、均值和重建图像后，显式批准同一 validation run：

```bash
VAL_APPROVED=1 sbatch slurm/06_infer_missing_t2w.slurm
```

阶段 6 从阶段 5 的 `evaluation_run.json` 读取确切的 VAE、EncDec、BBDM
checkpoint 和 `s`，不会改用其他权重。它不启用额外 brain-mask 模型。成功输出：

```text
data/output/run_<jobid>/<case_id>/
├── <case_id>-t1n.nii.gz
├── <case_id>-t1c.nii.gz
├── <case_id>-t2w.nii.gz
├── <case_id>-t2f.nii.gz
├── <case_id>-seg.nii.gz
└── intermediate_<case_id>/
```

必须正好 265 个 case、265 个生成 T2W。该 run 目录才是 G2 QC 输入。

## 8. 可覆盖参数

```bash
# VAE OOM 时只先降这个值
VAE_BATCH_SIZE=1 sbatch slurm/02_finetune_vae.slurm

# BBDM s 粗调；每个 s 必须独立训练、独立 val
TRAIN_TARGET=bbdm G1_BBDM_S=0.005 G1_MODEL_RUN_ID=bbdm_s0005 \
  sbatch slurm/04_train_models.slurm

# 只有观察到过拟合后才考虑 weight decay
TRAIN_TARGET=bbdm G1_WEIGHT_DECAY=1e-5 G1_MODEL_RUN_ID=bbdm_wd1e5 \
  sbatch slurm/04_train_models.slurm
```

代码参数拼写保留上游接口 `ensamble`，不要在命令中写 `ensemble`。
