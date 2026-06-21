# G1 diffusion augmentation 服务器训练手册

更新日期：2026-06-21
适用对象：负责在服务器上跑 G1 diffusion augmentation 线的操作者。

## 0. 先分清两条 G1 线

G1 现在有两条完全不同的线，不要混着跑：

| 代码目录 | 任务 | 输入 | 输出 | 当前用途 |
|---|---|---|---|---|
| `work_space/G1/code/brats2025-latent-ensemble-synthesis-main` | 缺失模态填补 | `t1n/t1c/t2f/seg`，目标补 `t2w` | `work_space/G1/data/output/<case_id>/<case_id>-t2w.nii.gz` | 先补 fake/broken T2W 病例 |
| `work_space/G1/code/BraTS_2023_2024_solutions-main/Segmentation_Tasks/GliGAN` | diffusion augmentation | `seg` 条件，训练时用完整 `t1n/t1c/t2w/t2f/seg` | 每个 synthetic case 的 `t1c/t1n/t2w/t2f/seg` | 后续做样本增强 |

这份手册只讲第二条线。

## 1. 共享目录约定

原始数据只挂一次：

```text
work_space/G1/data/raw/
```

如果要做本地镜像缓存，优先用：

```text
work_space/G1/data/diffusion_cache/
```

不要再用历史缓存目录作为默认入口。所有缓存都应按本轮脚本重新生成。

## 2. 先确认前置文件

训练和缓存构建都依赖 G2 已经跑完的真实数据清单：

```text
work_space/G2/results/manifests/real_train_manifest.csv
work_space/G2/results/manifests/nnunet_case_mapping_realonly.csv
work_space/G2/results/qc/official_fake_t2w_cases_by_gzip_header_2026-06-15.csv
```

如果需要补充校验，还会用到：

```text
work_space/G2/results/splits/splits_final_train_val_test.json
```

如果这些文件没刷新，先停，不要直接开始训练。

## 3. 环境

建议单独 Conda 或 uv 环境，不要混系统 Python、Homebrew Python 或 `sudo pip`。

常见依赖：

```text
torch
monai
nibabel
numpy
pandas
scipy
tqdm
matplotlib
```

## 4. 先建立可选的 diffusion 缓存

如果服务器想先做一份共享缓存，用这个脚本：

```bash
cd work_space/G1/code/BraTS_2023_2024_solutions-main/Segmentation_Tasks/GliGAN
python scripts/prepare_diffusion_dataset.py \
  --dataset-root ../../../../data/diffusion_cache \
  --mode symlink
```

这一步会：

1. 从 G2 的 `real_train_manifest.csv` 中筛出当前可训练病例。
2. 只保留 `BraTS-MET-*`、`final_qc_pass=True`、四模态加 `seg` 都完整的病例。
3. 自动排除 fake/broken T2W 病例。
4. 每次重建前先清空旧缓存，避免历史产物和旧 smoke test 混进来。

如果你不需要缓存，直接跑 `csv_creator.py` 也可以。

## 5. 创建训练 CSV

在 `GliGAN` 目录下运行：

```bash
python src/train/csv_creator.py \
  --dataset BRATS_2026 \
  --datadir ../../../../data/raw \
  --logdir brats2026_diffusion \
  --require_met True
```

这一步会递归扫描 `work_space/G1/data/raw/` 下的 `BraTS-MET-*` 病例，只写入：

1. `t1n/t1c/t2w/t2f/seg` 齐全的病例。
2. 肿瘤 bbox 三个方向都不超过 96 的病例。
3. 不是 fake/broken T2W 的病例。

输出：

```text
../../Checkpoint/brats2026_diffusion/brats2026_diffusion.csv
../../Checkpoint/brats2026_diffusion/brats2026_diffusion_skipped.csv
```

如果 `skipped.csv` 里出现 `missing:t2w`，这是正常行为，说明缺 T2W 的病例被自动跳过了。

## 6. 训练四个模态

四个模态要分别训练：

```bash
python src/train/tumour_main_diffusion.py \
  --dataset BRATS_2026 \
  --modality t1c \
  --logdir brats2026_diffusion \
  --batch_size 2 \
  --generator_type SwinUNETR \
  --num_steps 100000 \
  --noise_schedule edm \
  --sampling_method edm_heun
```

把 `--modality t1c` 分别替换成：

```text
t1n
t2w
t2f
```

训练后权重会在：

```text
../../Checkpoint/brats2026_diffusion/<modality>/weights/
```

## 7. 快速自检

先拿一个真实 `seg` 跑单病例推理：

```bash
CASE_ID=BraTS-MET-00001-000
python src/infer/generate_from_label.py \
  --label_path ../../../../data/raw/$CASE_ID/$CASE_ID-seg.nii.gz \
  --diffusion_ckpt_dir ../../Checkpoint/brats2026_diffusion \
  --dataset BRATS_2026 \
  --output_dir /path/to/output_brats2026_diffusion_v1 \
  --generator_type SwinUNETR \
  --noise_schedule edm \
  --sampling_method edm_heun \
  --sampling_steps 18 \
  --modality all
```

这一步会输出：

```text
<case_id>-t1c.nii.gz
<case_id>-t1n.nii.gz
<case_id>-t2w.nii.gz
<case_id>-t2f.nii.gz
<case_id>-seg.nii.gz
```

注意：

1. `--dataset BRATS_2026` 只接受 `BraTS-MET-*` label。
2. `seg` 是条件输入和输出的一部分，不是要生成的 T2W。
3. 输出目录不要放进 Git 工作区。

## 8. 交给 G2 的输出

diffusion augmentation 的 G2 接收命令是：

```bash
python work_space/G2/code/g2_synthetic_raw_intake_qc.py \
  --synthetic-run-root /path/to/output_brats2026_diffusion_v1 \
  --synthetic-run-id g1_diffusion_augmentation_v1 \
  --generation-mode full_generation \
  --refresh-templates
```

G2 会按 full_generation 口径生成：

```text
synthetic_generation_manifest_*.csv
synthetic_candidate_manifest_*.csv
synthetic_accepted_manifest_*.csv
synthetic_rejected_manifest_*.csv
synthetic_normalized_mapping_*.csv
qc_metrics_*.csv
diffusion_quality_metrics_*.csv
qc_case_review_*.csv
qc_batch_summary_*.json
G2_synthetic_data_quality_report_*.md
```

## 9. 不要做的事

1. 不要把历史缓存目录当成默认入口。
2. 不要把历史病例或旧 smoke test 混进本轮训练。
3. 不要把缺 T2W 的病例手工塞进训练 CSV。
4. 不要把 `Validation/` 混进训练 CSV。
5. 不要把大体积 NIfTI 和 checkpoint 提交进 Git。

## 10. 最后检查清单

跑完后至少确认：

1. `work_space/G1/data/raw/` 只挂了一份原始数据。
2. `csv_creator.py` 只写入 `BraTS-MET-*` 完整病例。
3. `brats2026_diffusion_skipped.csv` 里能看到被跳过的缺模态病例。
4. 四个模态都训练过，权重都在各自目录下。
5. 单病例推理能输出 `t1c/t1n/t2w/t2f/seg`。
6. G2 intake 用 `--generation-mode full_generation`，而不是 completion。
