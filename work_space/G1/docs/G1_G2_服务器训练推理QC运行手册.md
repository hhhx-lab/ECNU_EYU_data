# G1-G2 服务器训练、生成、QC 总运行手册

更新日期：2026-07-16

## 1. 当前只有两条正式产线

| 产线 | 当前代码 | 数据含义 | 模型专用手册 |
|---|---|---|---|
| 缺失 T2W V3 | `work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v3` | 修复原病例 T2W | `slurm/README.md` |
| Diffusion augmentation V3 | `work_space/G1/code/BraTS_2023_2024_solutions-main 3/Segmentation_Tasks/GliGAN` | 新增四模态 augmentation | `README_DIFFUSION.md`、`slurm/README.md` |

旧 `brats2025-latent-ensemble-synthesis-main-v2` 不再是缺失 T2W 正式训练入口。历史代码和结果可保留追溯，但新任务使用 V3。

## 2. 两条线不能互换

### 2.1 缺失 T2W V3

```text
真实完整病例
  -> patient-group train/val/test
  -> VAE 选择
  -> EncDec/BBDM 训练
  -> 固定 val 选择
  -> 265 个 fake/broken T2W 病例 completion
  -> G2 V3 intake/QC
```

V3 保留原病例身份，只替换 T2W，不增加 synthetic 病例数。

### 2.2 Diffusion augmentation V3

```text
823 个 authentic master-train source
  -> 四模态 diffusion 训练
  -> 按 seg 生成四模态病灶区域
  -> G2 composer 恢复完整病例
  -> G2 full-generation intake/QC
  -> accepted synthetic 只追加 train
```

V2 建立新的 synthetic 病例，不能覆盖原病例，也不能使用 val/test source。

## 3. 全队固定数据口径

| 数据层 | train | val | test | 合计 |
|---|---:|---:|---:|---:|
| master | 1035 | 130 | 130 | 1295 |
| authentic-T2W real-only | 823 | 103 | 104 | 1030 |
| V3 completion 目标 | 212 | 27 | 26 | 265 |

唯一身份与 split 文件：

```text
work_space/G2/results/manifests/nnunet_case_mapping_master.csv
work_space/G2/results/manifests/nnunet_case_mapping_realonly.csv
work_space/G2/results/manifests/g1_v2_source_manifest.csv
work_space/G2/results/splits/splits_master_train_val_test.json
work_space/G2/results/splits/splits_final_train_val_test.json
```

同一 `BraTS-MET-xxxxx` 患者组不得跨 split。

## 4. 服务器路径原则

不要在文档中写死个人用户名。登录后先设置：

```bash
export PROJECT_ROOT=/path/to/ECNU_EYU_data
cd "${PROJECT_ROOT}"
```

原始大数据放在服务器稳定数据盘，并通过软链接或脚本映射到 `work_space/G1/data/raw`。不要把 NIfTI、checkpoint、latent、预处理缓存提交到 Git。

所有 Python 环境使用独立 Conda 环境，不使用系统 Python，不运行 `sudo pip`，不在 Slurm 作业中临时安装依赖。

## 5. G2 数据索引预检

服务器 raw data 和 corrected labels 就位后，先运行：

```bash
python work_space/G2/code/g2_build_realonly_from_raw.py \
  --data-root /path/to/MICCAI-LH-BraTS2025-MET-Challenge-Training \
  --corrected-labels-root /path/to/MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels \
  --fail-if-no-valid-cases
```

必须核对：

1. master 1295、real-only 1030、completion 265。
2. real-only split 为 823/103/104。
3. master split 为 1035/130/130。
4. patient-group overlap 为 0。
5. `BraTS-MET-01094-002` 因非法标签被排除。
6. corrected label 按清单优先使用。
7. V2 allowed source 为 823。

数量不符时停止模型任务，先确认 raw root、fake T2W 清单和 corrected labels。

## 6. 缺失 T2W V3 操作顺序

进入：

```bash
cd "${PROJECT_ROOT}/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v3"
mkdir -p logs
```

严格按 `slurm/README.md` 执行七个阶段：

| 阶段 | 脚本 | 关键门 |
|---:|---|---|
| 0 | `00_smoke.slurm` | 环境、CUDA、模型前向通过 |
| 1 | `01_adopt_prepared_data.slurm` | 复用已验证数据身份和 fixed split |
| 2 | `02_finetune_vae.slurm` | 全 train patch 微调，完整 val 选择 |
| 3 | `03_encode_and_prepare_aux.slurm` | VAE 人工批准后统一重编码 |
| 4 | `04_train_models.slurm` | EncDec/BBDM 可并行，各一张 GPU |
| 5 | `05_evaluate_val.slurm` | 固定 103 val，原生空间输出与 spatial/geometry audit |
| G2 gate | `01_g2_v3_paired_quality.slurm` + `02_g2_v3_s2_teacher.slurm` | 配对 QC、冻结 teacher、montage 人工复核 |
| 6 | `06_infer_missing_t2w.slurm` | `FINAL_GATE.json=approve_stage6` 后处理 265 例 |

阶段 2 到 3、阶段 5 到 6 是人工门，不能一次性无条件串完。
正式 raw NIfTI 不再做固定体素中心裁切；统一使用 `t1n/t1c/t2f + seg`
确定前景中心和自适应 isotropic spacing，并将生成 T2W 直接恢复到原生 shape/affine。

### 6.1 V3 Stage 5 到 G2 的必要输出

```text
data/evaluation/val/run_<jobid>/
  metrics.csv
  spatial_audit.csv
  evaluation_run.json
  synthesized/*.nii.gz
  geometry_audit/geometry_audit.csv
  geometry_audit/geometry_audit.json
```

`spatial_audit.csv` 必须正好 103 行且 ID 与 `metrics.csv` 完全一致；
foreground/lesion outside count 必须全为 0。G2 paired QC 会将上述条件作为写输出前的硬门。

在项目根目录对同一 Stage 5 run 并行提交：

```bash
PROJECT_ROOT="${PROJECT_ROOT}" G1_V3_VAL_RUN_DIR=/absolute/path/to/run_<jobid> \
  sbatch work_space/G2/slurm/01_g2_v3_paired_quality.slurm
PROJECT_ROOT="${PROJECT_ROOT}" G1_V3_VAL_RUN_DIR=/absolute/path/to/run_<jobid> \
  sbatch work_space/G2/slurm/02_g2_v3_s2_teacher.slurm
```

两者技术完成后仍需按 `review_index.csv` 复核 montage。最终
`FINAL_GATE.json` 的 `run_id` 必须对应 Stage 5，`decision` 必须为
`approve_stage6`，否则 `06_infer_missing_t2w.slurm` 直接拒绝。

V3 阶段 6 必须输出：

```text
data/output/run_<id>/
  inference_run.json
  generation_config.json
  generation_log.jsonl
  synthetic_generation_manifest.csv
  <case_id>/
    <case_id>-t1n.nii.gz
    <case_id>-t1c.nii.gz
    <case_id>-t2w.nii.gz
    <case_id>-t2f.nii.gz
    <case_id>-seg.nii.gz
```

run metadata 必须记录 VAE、EncDec、BBDM checkpoint、`bbdm_s`、seed、source CSV 和 validation run。

## 7. V3 进入 G2

从项目根运行：

```bash
python work_space/G2/code/g2_v3_completion_intake.py \
  --completion-run-root /absolute/path/to/v3/data/output/run_<id> \
  --data-root /absolute/path/to/2026_task1_data
```

`--data-root` 指向同时包含 Training、corrected-labels 和可选 Validation
目录的原始数据根目录。G2 用它解析版本化 mapping 中的
`work_space/G1/data/raw/...` 路径；不复制 40GB 数据，也不要省略后改用
completion 输出自身充当 source。服务器已经在项目内物化相同 raw 路径时可省略。

G2 检查：

1. 265 个 source 都在 master fake/broken 清单。
2. 保留原病例 ID 和 nnU-Net ID。
3. t1n/t1c/t2f 与 source 未改变。
4. seg 与 corrected source seg 未改变。
5. T2W 几何、数值、脑区、病灶信号和 z 连续性合理。
6. train completion 和 val/test completion 使用不同 release 角色。

无 `g2_approval_manifest.csv` 时技术通过病例仍为 pending。

## 8. Diffusion augmentation V3 操作顺序（NYU Greene）

进入：

```bash
export PROJECT_ROOT=/scratch/bf2260/ECNU_EYU_data
cd "${PROJECT_ROOT}/work_space/G1/code/BraTS_2023_2024_solutions-main 3/Segmentation_Tasks/GliGAN"
mkdir -p "${PROJECT_ROOT}/logs"
```

按 `slurm/README.md` 执行：

1. `00_smoke_v3_nyu.slurm`
2. `00_preflight_crop64_v3_nyu.slurm`
3. `01_prepare_dataset_v3_nyu.slurm`
4. `02_train_4modal_v3_nyu.slurm`（单节点 4 GPU，每模态一卡）
5. `03_eval_4modal_v3_nyu.slurm`
6. 可选：`04_generate_visual_v3_nyu.slurm`

正式 source 只来自 `g1_v2_source_manifest.csv` 的 823 个 allowed train；103 个 val 只评估。

V2 raw 输出特性：

1. 内部 crop 默认 `64³`。
2. 最终文件 shape 与 source label 相同。
3. 生成区外为 0。
4. affine 为单位阵。
5. 不输出 seg。

因此 V2 raw 不能直接给 S1-S5。

## 9. Diffusion augmentation V3 进入 G2

正式 raw root 必须带 `generation_config.json`，其中必须记录 run ID、seed、checkpoint、source manifest、`sampling_method`、`sampling_steps`、`eta` 和 `crop_size`，然后运行：

```bash
python work_space/G2/code/g2_v2_compose_augmentation.py \
  --v2-output-root /absolute/path/to/v2_raw \
  --source-manifest work_space/G2/results/manifests/g1_v2_source_manifest.csv \
  --output-run-root /absolute/path/to/g2_composed/v2_run_id

python work_space/G2/code/g2_synthetic_raw_intake_qc.py \
  --synthetic-run-root /absolute/path/to/g2_composed/v2_run_id \
  --generation-mode full_generation
```

composer 会恢复 source 强度和 geometry、复制 corrected seg、保留非 generation support 区域并生成 config/log/manifest。任一病例 composition 失败时脚本返回非 0，不得跳过错误直接训练。输出目录非空时默认拒绝；只有确认整轮重建时才加 `--overwrite`，该选项会先清空整个 composed run。

G2 脚本名中的 `v2` 是 augmentation 接口版本名；当前生产模型代码已经升级为 `main 3`，不要因此改回旧模型目录。

## 10. G2 审批

人工或 teacher 复核后，在对应 run root 写：

```csv
synthetic_raw_id,approved_for_training,approved_for_evaluation,reviewer,reason,cleared_review_reasons
```

审批规则：

1. V2 augmentation：只允许 `approved_for_training=True`。
2. V3 master train completion：允许 `approved_for_training=True`。
3. V3 master val/test completion：只允许 `approved_for_evaluation=True`。
4. `cleared_review_reasons` 只填已人工排除的 `tiny_ratio_high` / `z_discontinuity`，多项用分号分隔。
5. 审批后重新运行 intake，让脚本派生 accepted manifest。

禁止直接编辑 accepted CSV。

## 11. 发布给 S1-S5

先运行 `manifest-only` 做源文件预检，再物化：

```bash
python work_space/G2/code/g2_materialize_nnunet_dataset.py \
  --output-root /path/to/nnUNet_raw \
  --case-folder-root /path/to/case_folders \
  --dataset-id 262 \
  --dataset-name BraTS2026_MET_RealSynth \
  --dataset-profile real-synth \
  --synthetic-accepted-manifest /path/to/v3_accepted_training.csv \
  --synthetic-accepted-manifest /path/to/v3_accepted_evaluation.csv \
  --synthetic-accepted-manifest /path/to/v2_accepted.csv \
  --mode symlink \
  --clean-output \
  --run-nnunet-integrity
```

V3 的两份 manifest 都必须传入：training 文件覆盖 master train completion，evaluation 文件覆盖原 val/test completion。漏传后者时 materializer 会因 locked val/test 缺 T2W 而停止。上面的 `--clean-output` 只应在已检查同路径 manifest-only 预检结果、确认可删除该轻量预检目录后使用。

固定通道：

```text
0000=t1n
0001=t1c
0002=t2w
0003=t2f
```

输出隔离：

```text
imagesTr/labelsTr = train + val
imagesTs/labelsTs = locked test
case_folders/train|val|test = S1/S3/S4/S5 视图
```

S2 使用 nnU-Net view；S1/S3/S4/S5 使用按 split 分区的 case-folder view，S5 再运行自己的 fixed-split preprocessing。

## 12. 最终验收

1. G2 全部技术硬门通过。
2. pending 全部完成审批或明确拒绝。
3. synthetic augmentation 在 val/test 中为 0。
4. locked test 不出现在 imagesTr/labelsTr。
5. G2 integrity 与 nnU-Net integrity 均通过。
6. 在当前 patient-group master split 上建立 paired real-only baseline。
7. 比较 real-only、V3 completion、V2 augmentation 和 V3+V2。
8. 使用官方 lesionwise DSC/NSD 与 small-instance F1 收口。

历史使用不同 split 的 checkpoint 可以保留，但不能作为当前 G2 数据的严格 paired comparator。

## 13. 故障停止条件

出现以下情况立即停止后续 job：

1. 数据数量或 split 与预期不符。
2. 同患者跨 split。
3. checkpoint、seed、manifest 或 log 缺失。
4. NIfTI 不可读或 geometry 不一致。
5. V3 spatial audit 病例不齐或任一 foreground/lesion outside count 非 0。
6. V3 改动了 t1n/t1c/t2f/seg。
7. V2 非生成区被改动。
8. nnU-Net integrity 失败。
9. 输出目录非空但操作者无法确认内容来源。

不要通过手改 CSV、关闭错误检查或复用未知旧目录继续运行。
