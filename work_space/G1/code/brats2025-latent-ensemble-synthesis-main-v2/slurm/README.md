# G1 V2 Slurm 执行说明

适用服务器：华东师范大学超算八期集群。代码任务是用完整四模态病例学习 `t1n + t1c + t2f -> t2w`，再对真正缺失 T2W 的病例重建 T2W。

## 1. 六个阶段

| 阶段 | 脚本 | 资源 | 输入 | 主要输出 |
|---|---|---|---|---|
| 0 | `00_a100_smoke.slurm` | `a100` | 环境、代码、预训练 VAE | CUDA/MAISI/权重加载确认 |
| 1 | `01_prepare_data.slurm` | `cpu_96G` | raw Task1 数据 | 链接、非法标签过滤、patient-grouped split、完整性检查 |
| 2 | `02_finetune_vae.slurm` | `a100` | train/val CSV、预训练 VAE | VAE 微调权重、val 对比指标、选择结论 |
| 3 | `03_encode_latents.slurm` | `a100` | 阶段 2 选中的 VAE | 全量 latent、attention masks、channel weights |
| 4 | `04_train_models.slurm` | `a100` | train latent | EncDec/BBDM checkpoints |
| 5 | `05_evaluate_val.slurm` | `a100` | val split、两模型 checkpoint | val 指标和重建 NIfTI |
| 6 | `06_infer_missing_t2w.slurm` | `a100` | 真正缺 T2W 病例 | `data/output/<case_id>-t2w.nii.gz` |

阶段 2 结束后必须停下来检查 VAE 结果，不能直接把阶段 3 自动接上。微调改变 latent space，只有确定使用哪个 VAE 后，才能用阶段 3 对全部病例统一重编码。

## 2. 默认服务器路径

```bash
PROJECT_ROOT=/public/home/${USER}/projects/ECNU_EYU_data
CODE_DIR=${PROJECT_ROOT}/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v2
RAW_DATA_ROOT=${HOME}/data
```

当前服务器 `${HOME}/data` 共识别 `1296` 个病例：`265` 个缺失或伪 T2W 病例进入 `input_inference`，其余完整病例进入训练候选。应用 corrected labels 并排除 `BraTS-MET-01094-002` 后，最终 metadata 为 `1030` 例，即 `823 train / 103 val / 104 locked test`。

## 3. 先运行 VAE 阶段

进入代码目录并保证 `logs/` 在提交前存在：

```bash
cd /public/home/${USER}/projects/ECNU_EYU_data/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v2
mkdir -p logs
```

提交数据准备：

```bash
PREP_JOB=$(sbatch --parsable slurm/01_prepare_data.slurm)
echo "PREP_JOB=${PREP_JOB}"
```

`01` 会先生成 metadata CSV，优先 corrected label，自动排除无可用修正版的非法标签病例，再按 patient group 固定切分。默认 seed 为 `42`，目标比例为 `80% train / 10% val / 10% locked test`。同一 `BraTS-MET-xxxxx` 下的所有记录只能进入同一个 split。

当前服务器准备结果应为 `1030` 例进入 CSV，其中 `823 train / 103 val / 104 locked test`。其中 `BraTS-MET-01094-002` 因非法标签且无可用修正版被排除，`BraTS-MET-01184-002` 使用 corrected label。若服务器结果不同，先检查 raw data、缺失 T2W 识别和 corrected labels，不要直接开训。

数据准备成功后再提交 VAE 微调：

```bash
VAE_JOB=$(sbatch --parsable --dependency=afterok:${PREP_JOB} slurm/02_finetune_vae.slurm)
echo "VAE_JOB=${VAE_JOB}"
```

阶段 2 严格执行 README 1.2 的三步：在全部 `103` 个 val 上建立预训练 VAE baseline；用全部 `823` 个 train 做肿瘤中心 patch 微调；最后在全部 `103` 个 val 上做完整体积对比。默认 patch 为 `128×128×96`，`80%` 均匀选肿瘤连通域、`20%` 随机脑区，四模态同步裁剪，最多 `3` epochs，BF16，关闭梯度检查点。每个 epoch 固定验证 `20` 个 val 病例并按 tumor MSE early stop，locked test 不参与训练、调参或模型选择。

首次运行先做 20 病例 benchmark，不发布选择结果：

```bash
VAE_EPOCHS=1 VAE_MAX_TRAIN_SUBJECTS=20 VAE_MAX_VAL_SUBJECTS=2 \
VAE_QUICK_VAL_SUBJECTS=2 PUBLISH_VAE_SELECTION=0 \
VAE_OUTPUT_DIR=training/vae_finetuned/benchmark_${USER}_$(date +%Y%m%d_%H%M%S) \
sbatch slurm/02_finetune_vae.slurm
```

确认日志中显存无 OOM、`optimizer_steps > 0`、`best_model.pt` 和 `vae_selection.json` 均生成，再按 benchmark 的训练段耗时估算 `823×3/20`。预计总任务不超过 `18` 小时后，直接提交默认正式任务：

```bash
sbatch slurm/02_finetune_vae.slurm
```

监控：

```bash
squeue -u ${USER}
tail -f logs/g1v2_prep_${PREP_JOB}.out
tail -f logs/g1v2_vae_${VAE_JOB}.out
```

## 4. VAE 验收门

阶段 2 输出目录默认是：

```text
training/vae_finetuned/run_<jobid>/
```

必须检查：

```text
vae_selection.json
baseline_metrics.csv
finetuned_metrics.csv
delta_metrics.csv
training_history.csv
quick_val_subjects.json
finetune_config.json
comparison_samples/*.png
best_model.pt
```

自动选择门槛：

- mean `delta_tumor_SSIM >= 0.03`
- mean `delta_whole_SSIM >= -0.005`

`vae_selection.json` 的 `selected_weights` 是阶段 3 唯一采用的权重。如果未达门槛，会保留原始预训练 VAE；不会因为已经花时间微调就强行使用变差的模型。

查看结论：

```bash
VAE_RUN_DIR=$(head -n 1 training/vae_finetuned/latest_run.txt)
python -m json.tool "${VAE_RUN_DIR}/vae_selection.json"
```

## 5. VAE 验收后继续

确认阶段 2 的指标和 5 张最差病例对比图后，提交统一重编码：

```bash
ENCODE_JOB=$(sbatch --parsable slurm/03_encode_latents.slurm)
echo "ENCODE_JOB=${ENCODE_JOB}"
```

EncDec 和 BBDM 可在阶段 3 成功后并行训练：

```bash
ENDEC_JOB=$(TRAIN_TARGET=endec sbatch --parsable --dependency=afterok:${ENCODE_JOB} slurm/04_train_models.slurm)
BBDM_JOB=$(TRAIN_TARGET=bbdm sbatch --parsable --dependency=afterok:${ENCODE_JOB} slurm/04_train_models.slurm)
echo "ENDEC_JOB=${ENDEC_JOB} BBDM_JOB=${BBDM_JOB}"
```

两个训练 job 都成功后再验证 ensemble：

```bash
EVAL_JOB=$(sbatch --parsable \
  --dependency=afterok:${ENDEC_JOB}:${BBDM_JOB} \
  slurm/05_evaluate_val.slurm)
echo "EVAL_JOB=${EVAL_JOB}"
```

确认 `data/eval_metrics_val.csv` 和 `data/eval_synthesized_val/` 后，且确实已经补充缺 T2W 数据，才运行：

```bash
sbatch slurm/06_infer_missing_t2w.slurm
```

## 6. 可覆盖参数

```bash
RAW_DATA_ROOT=/path/to/raw \
SPLIT_SEED=42 \
VAL_FRACTION=0.10 \
TEST_FRACTION=0.10 \
G1_V2_CONDA_ENV=segmamba \
sbatch slurm/01_prepare_data.slurm

VAE_EPOCHS=3 \
VAE_BATCH_SIZE=2 \
VAE_VAL_INTERVAL=1 \
VAE_SAVE_INTERVAL=1 \
VAE_PATCH_SIZE="128 128 96" \
VAE_TUMOR_PATCH_PROBABILITY=0.8 \
VAE_QUICK_VAL_SUBJECTS=20 \
VAE_EARLY_STOPPING_PATIENCE=2 \
sbatch slurm/02_finetune_vae.slurm
```

若 A100 上 VAE 微调 OOM，先将 `VAE_BATCH_SIZE=1`。这个参数是每次 optimizer step 累积的病例数，每个病例包含 4 个模态。

## 7. 关键约束

1. `data/data_csv.csv` 中 `val` 必须非空；`01` 会自动生成并验证，不手改 CSV。
2. 非法标签病例先写入 `g1_v2_label_filter_report.csv` 再排除，禁止静默改标签值。
3. 同一 patient group 不得跨 train/val/test。
4. VAE 微调只读 train；模型选择只看 val；locked test 保留到最终内部复核。
5. 一旦选择了微调 VAE，原 VAE 产生的 latents 全部作废；`03` 会清空并按过滤后 CSV 白名单重编码。
6. EncDec、BBDM、验证和推理都通过 `G1_VAE_WEIGHTS` 读取同一个已选 VAE。
7. 代码真实参数拼写是 `ensamble`，不要擅自改成 `ensemble`。
