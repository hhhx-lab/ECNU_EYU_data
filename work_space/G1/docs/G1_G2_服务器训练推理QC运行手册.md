# G1 T2W 缺失模态填补服务器运行手册

更新日期：2026-06-21
适用对象：负责在服务器上跑 G1 缺失模态填补线的操作者。

## 0. 先记住范围

这份手册只管 **T2W 缺失模态填补**，不是完整的 diffusion synthetic augmentation。

当前流程是：

1. G2 先给出真实训练清单、fake T2W 清单和固定 train/val/test 划分。
2. G1 用 `prepare_g1_t2w_data.py` 自动把数据摆到 `work_space/G1/data/input/` 和 `work_space/G1/data/input_inference/`。
3. `preprocess.py` 只保留完整四模态训练病例，缺 `t2w` 的病例不会进 `data_csv.csv`。
4. `mark_val_split_from_g2.py` 按 G2 固定划分写入 `train/val/test`。
5. `main.py` 只处理 `t1n/t1c/t2f/seg`，并生成 `t2w`。
6. G2 再接收 G1 输出，做 QC、accepted/rejected 和后续物化。

## 1. 共享目录约定

只保留一份原始数据挂载：

```text
work_space/G1/data/raw/
```

工作区数据目录：

```text
work_space/G1/data/input/
work_space/G1/data/input_inference/
work_space/G1/data/output/
work_space/G1/data/latents/
work_space/G1/data/data_csv.csv
work_space/G1/data/data_csv_skipped_subjects.csv
work_space/G1/data/g1_data_placement_manifest.csv
```

规则很简单：

1. `raw/` 只读，只挂一次，不复制大体积 NIfTI。
2. `input/` 只放完整四模态训练病例。
3. `input_inference/` 只放需要补 `t2w` 的病例，`t2w` 必须缺省。
4. `output/` 只放 G1 生成结果，不要混旧结果。

## 2. 先确认前置文件

G1 completion 依赖 G2 先生成这些文件：

```text
work_space/G2/results/manifests/real_train_manifest.csv
work_space/G2/results/manifests/nnunet_case_mapping_realonly.csv
work_space/G2/results/splits/splits_final_train_val_test.json
work_space/G2/results/qc/official_fake_t2w_cases_by_gzip_header_2026-06-15.csv
```

如果 `official_fake_t2w_cases_by_gzip_header_2026-06-15.csv` 不在，脚本会回退读：

```text
work_space/G2/results/qc/official_t2w_gzip_header_audit_2026-06-15.csv
```

如果这些文件缺失、路径还是旧机路径，先停，不要手工改病例目录，直接让 G2 刷新 audit。

## 3. 环境

建议只用 Conda 或 uv，不要混系统 Python、Homebrew Python 或 `sudo pip`。

需要的关键资源：

```text
work_space/G1/code/brats2025-latent-ensemble-synthesis-main/weights/vae/autoencoder_epoch273.pt
```

先在 G1 代码目录跑：

```bash
cd work_space/G1/code/brats2025-latent-ensemble-synthesis-main
python test_vae.py
```

只有 VAE 权重加载成功，才继续后面的步骤。

## 4. 第一步：自动摆放 G1 数据

在 G1 completion 代码目录运行：

```bash
cd work_space/G1/code/brats2025-latent-ensemble-synthesis-main
python prepare_g1_t2w_data.py --data-root ../../data --mode symlink --clean --overwrite
```

这一步会：

1. 读取 G2 的真实训练 manifest 和 fake T2W 清单。
2. 把完整四模态病例放进 `work_space/G1/data/input/`。
3. 把需要补 `t2w` 的病例放进 `work_space/G1/data/input_inference/`。
4. 对 inference 病例 **不链接 `t2w`**，即使原始目录里曾经有坏文件也不会保留。
5. 写出 `work_space/G1/data/g1_data_placement_manifest.csv`。

如果源路径缺失，脚本会直接报错。不要跳过这个错误继续跑。

## 5. 第二步：G1 预处理

```bash
python preprocess.py
```

这一步会：

1. 扫描 `work_space/G1/data/input/`。
2. 只保留 `t1n/t1c/t2w/t2f` 齐全的病例。
3. 缺 `t2w` 或缺其他训练必需模态的病例自动跳过。
4. 编码 latent。
5. 生成 `work_space/G1/data/data_csv.csv`。
6. 如果有跳过病例，生成 `work_space/G1/data/data_csv_skipped_subjects.csv`。

正式流程里不要手工改 `data_csv.csv`。

## 6. 第三步：写入固定 train/val/test

```bash
python mark_val_split_from_g2.py
```

这一步会按 G2 的固定划分把 `data_csv.csv` 的 `split` 列改成 `train/val/test`。

如果出现 unmatched，先停，不要使用 `--allow-unmatched-as-train` 作为正式训练方案。

## 7. 第四步：生成辅助文件

### 7.1 肿瘤掩码

```bash
python generate_attmask.py
```

用途：给 BBDM 的 loss 计算肿瘤掩码。

### 7.2 通道权重

```bash
python compute_weights.py
```

用途：重新计算当前数据集的通道权重。

这两个步骤都依赖 `seg` 存在。

## 8. 第五步：训练

### 8.1 EncDec

```bash
python training_endec.py
```

### 8.2 BBDM

```bash
python training_bbdm.py
```

调参顺序建议：

1. 先调 BBDM 的 `s`。
2. 再看 `weight_decay`。
3. 最后再动其他次要参数。

## 9. 第六步：用 Validation 做自检

`evaluate.py` 只要求完整四模态，不要求 `seg`。官方 `Validation/` 正适合做生成质量 sanity check。

```bash
python evaluate.py \
  --input_dir ../../data/raw/Validation \
  --synthesis_type ensamble \
  --gpu_id 0 \
  --verbose \
  --save_csv ../../data/eval_validation_metrics.csv \
  --save_output
```

注意：

1. `ensamble` 是代码里的真实拼写，不要改成 `ensemble`。
2. 这里不要指向 `input_inference/`，那里面本来就没有 `t2w`。
3. 只看生成结果是不是有明显空白、错位、截断、强噪声。

## 10. 第七步：推理生成 T2W

`input_inference/` 的正确结构是：

```text
work_space/G1/data/input_inference/<case_id>/
  <case_id>-t1n.nii.gz
  <case_id>-t1c.nii.gz
  <case_id>-t2f.nii.gz
  <case_id>-seg.nii.gz
```

其中没有 `t2w` 是正常状态。

推理命令：

```bash
python main.py \
  --input_dir ../../data/input_inference \
  --output_dir ../../data/output \
  --synthesis_type ensamble \
  --gpu_id 0 \
  --verbose \
  --compute_bmask
```

输出结构：

```text
work_space/G1/data/output/<case_id>/
  <case_id>-t1n.nii.gz
  <case_id>-t1c.nii.gz
  <case_id>-t2f.nii.gz
  <case_id>-seg.nii.gz
  <case_id>-t2w.nii.gz
```

前四个是镜像源文件，最后一个是生成结果。

## 11. 交给 G2 的输出

G2 接收命令：

```bash
python work_space/G2/code/g2_synthetic_raw_intake_qc.py \
  --synthetic-run-root work_space/G1/data/output \
  --synthetic-run-id g1_t2w_completion_v1 \
  --generation-mode completion \
  --refresh-templates
```

G2 会生成：

```text
work_space/G2/results/manifests/synthetic_generation_manifest_g1_t2w_completion_v1.csv
work_space/G2/results/manifests/synthetic_candidate_manifest_g1_t2w_completion_v1.csv
work_space/G2/results/manifests/synthetic_accepted_manifest_g1_t2w_completion_v1.csv
work_space/G2/results/manifests/synthetic_rejected_manifest_g1_t2w_completion_v1.csv
work_space/G2/results/manifests/synthetic_normalized_mapping_g1_t2w_completion_v1.csv
work_space/G2/results/qc/qc_metrics_g1_t2w_completion_v1.csv
work_space/G2/results/qc/diffusion_quality_metrics_g1_t2w_completion_v1.csv
work_space/G2/results/qc/qc_case_review_g1_t2w_completion_v1.csv
work_space/G2/results/qc/qc_batch_summary_g1_t2w_completion_v1.json
work_space/G2/results/reports/G2_synthetic_data_quality_report_g1_t2w_completion_v1.md
```

## 12. 最后检查清单

跑完后至少确认：

1. `work_space/G1/data/raw/` 只挂了一份原始数据。
2. `work_space/G1/data/input/` 里全是完整四模态病例。
3. `work_space/G1/data/input_inference/` 里没有 `t2w`。
4. `work_space/G1/data/data_csv.csv` 里没有缺 `t2w` 的病例。
5. `mark_val_split_from_g2.py` 已执行，`split` 列有 `train/val/test`。
6. `work_space/G1/data/output/` 里能找到生成的 `t2w`。
7. `evaluate.py` 对 `Validation/` 跑通。
8. G2 intake 能产出 accepted/rejected/QC/report 文件。
