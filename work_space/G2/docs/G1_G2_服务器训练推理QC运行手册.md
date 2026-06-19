# G1 / G2 服务器运行手册

更新日期：2026-06-19
适用对象：拿到原始 BraTS Task1 数据、要在服务器上把 G1 训练/推理跑起来，并让 G2 完成 QC、accepted/rejected 判定和 nnU-Net 导出的同学。

## 0. 你先记住一句话

不要手动删坏掉的 `t2w`，不要手动改 `data/data_csv.csv`，不要把系统 Python 和 Conda/uv 混起来。
这套流程已经改成：

1. G2 先从原始 Task1 数据生成真实数据侧清单和 fake T2W 清单。
2. G2 自动生成固定 train/val/test 划分：829/207/259。
3. G1 自动把训练集和推理集摆好。
4. G1 预处理时，缺 `t2w` 的训练病例自动跳过，不进 `data_csv.csv`。
5. G1 推理时，只要求 `t1n/t1c/t2f/seg`，`t2w` 缺失是正常状态。
6. G2 接收 G1 输出后做 QC，给出 `accepted / rejected / ablation_only / needs_regeneration`。

---

## 1. 目录约定

建议把原始数据放在一个只读路径，例如：

```text
/path/to/2026的task1以及数据
```

里面通常至少有三块：

```text
MICCAI-LH-BraTS2025-MET-Challenge-Training/
Validation/
MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels/
```

G1 代码目录：

```text
work_space/G1/code/brats2025-latent-ensemble-synthesis-main
```

G2 代码和结果目录：

```text
work_space/G2/code
work_space/G2/results
```

### 1.1 手上的三个原始数据集分别怎么用

服务器同学通常只会拿到下面三个目录。不要拆散病例文件，也不要把 corrected labels 直接混进 Training 目录；先让 G2 audit 脚本统一识别和覆盖。

| 原始目录 | 里面有什么 | 谁使用 | 用途 |
|---|---|---|---|
| `MICCAI-LH-BraTS2025-MET-Challenge-Training/` | 每个病例一个文件夹，通常包含 `t1n/t1c/t2w/t2f/seg` | G2 audit、G1 training | 生成真实训练 manifest、训练 G1 EncDec/BBDM、生成 attention mask 和 channel weights。 |
| `Validation/` | 每个病例一个文件夹，通常包含 `t1n/t1c/t2w/t2f`，没有 `seg` | G2 audit、G1 evaluate | 只做官方 validation 结构检查和 G1 合成质量 sanity check；不要混入主训练集。 |
| `MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels/` | 少量 corrected `*-seg.nii.gz` | G2 audit | 自动覆盖 Training 中对应病例的原始 seg，形成最终 `real_train_manifest.csv`。 |

最重要的原则：

1. `Training/` 是 G1 训练的来源，但 fake/broken T2W 病例会被自动分到 `data/input_inference/`。
2. `Validation/` 不能作为 synthetic source，也不能混进主训练集；它可以拿来跑 `evaluate.py` 做图像合成 sanity check。
3. `corrected-labels/` 只由 G2 audit 读取；不要手工复制到病例文件夹里。
4. 旧 manifest 里有本机绝对路径，换服务器后必须重新跑 G2 audit，不要直接搬旧 CSV 当成路径真相。

---

## 2. 环境建议

建议只用 Conda 或 uv，不要混系统 Python / Homebrew Python。

### 2.1 推荐环境

```bash
conda create -n brats2026_t2w python=3.10
conda activate brats2026_t2w
```

### 2.2 常用依赖

G1 / G2 共同会用到：

```text
torch
monai
nibabel
numpy<2
pandas
scipy
tqdm
tensorboard
```

如果要跑 `--compute_bmask`，还可能需要 `TotalSegmentator`。

### 2.3 必备模型权重

G1 的 VAE 权重不进 Git，需要在服务器上单独放好：

```text
work_space/G1/code/brats2025-latent-ensemble-synthesis-main/weights/vae/autoencoder_epoch273.pt
```

放好后先在 G1 目录验证：

```bash
cd work_space/G1/code/brats2025-latent-ensemble-synthesis-main
python test_vae.py
```

只有看到 `VAE loaded OK`，再继续跑 `preprocess.py`、训练和推理。

---

## 3. 第一步：先把 G2 基线清出来

这一步的目的是生成：

1. 真实训练清单
2. 真实验证清单
3. G1 可用 source 清单
4. nnU-Net 映射
5. fixed train/val/test split
6. fake T2W 清单

命令：

```bash
TASK1_ROOT=/path/to/2026的task1以及数据

python work_space/G2/code/g2_pretraining_audit.py \
  --data-root "$TASK1_ROOT" \
  --results-root work_space/G2/results \
  --force
```

### 3.1 这一步会生成什么

重点看这些文件：

```text
work_space/G2/results/manifests/real_train_manifest.csv
work_space/G2/results/manifests/real_validation_manifest.csv
work_space/G2/results/manifests/g1_gligan_source_cases_v1.csv
work_space/G2/results/manifests/nnunet_case_mapping_realonly.csv
work_space/G2/results/splits/splits_final_fold0_realval.json
work_space/G2/results/splits/splits_final_train_val_test.json
work_space/G2/results/splits/splits_final_train_val_test_membership.csv
work_space/G2/results/qc/official_fake_t2w_cases_by_gzip_header_2026-06-15.csv
work_space/G2/results/qc/official_t2w_gzip_header_audit_2026-06-15.csv
```

### 3.2 你要检查什么

1. `real_train_manifest.csv` 里路径是否指向你服务器上的原始数据。
2. `g1_gligan_source_cases_v1.csv` 是否已经生成。
3. `official_fake_t2w_cases_by_gzip_header_2026-06-15.csv` 是否存在。
4. `splits_final_train_val_test.json` 是否存在。
5. `splits_final_train_val_test_membership.csv` 是否存在。

如果你只换了挂载路径，但病例 ID 没变，这一步就够用了。

如果后续把 `--results-root` 改成别的位置，那么 G1 的 `prepare_g1_t2w_data.py` 也要显式传入同一套 manifest 路径；为了减少出错，建议服务器第一版先使用默认 `work_space/G2/results`。

### 3.3 当前固定划分口径

当前 G2 正式划分文件是：

```text
work_space/G2/results/splits/splits_final_train_val_test.json
```

划分原则：

1. `splits_final_fold0_realval.json` 是历史 two-way fold，原来是 1036 train / 259 val。
2. 现在把历史 `259 val` 锁定为内部 `test`，后续不训练、不调参、不作为 synthetic source。
3. 从历史 `1036 train` 中再按稳定 hash 切出 `207 val`，用于调参和 dev 评估。
4. 剩余 `829 train` 作为真实训练池和 synthetic source 池。

所以 G2/S1/S2 的正式口径是：

```text
train = 829
val   = 207
test  = 259
```

G1 因为只能用完整真实 T2W 病例训练，投影到 G1 的 `data/data_csv.csv` 后是：

```text
train = 660
val   = 160
test  = 210
```

另外 265 个 fake/broken T2W 病例只进入 `data/input_inference/` 做 T2W 补全，不进入 G1 的 `data/data_csv.csv`。

如果只想重生成划分文件，不想重跑完整 audit，可以执行：

```bash
python work_space/G2/code/g2_create_train_val_test_split.py
```

---

## 4. 第二步：自动摆放 G1 数据

这一步把 G1 需要的数据自动分到：

```text
work_space/G1/code/brats2025-latent-ensemble-synthesis-main/data/input/
work_space/G1/code/brats2025-latent-ensemble-synthesis-main/data/input_inference/
```

命令：

```bash
cd work_space/G1/code/brats2025-latent-ensemble-synthesis-main
python prepare_g1_t2w_data.py --mode symlink --clean --overwrite
```

### 4.1 这个脚本做什么

1. 从 G2 的真实训练 manifest 和 fake T2W 清单里找病例。
2. 把完整四模态病例放进 `data/input/`。
3. 把需要补 `t2w` 的病例放进 `data/input_inference/`。
4. 推理目录里**自动不放 `t2w`**。
5. 写出 `data/g1_data_placement_manifest.csv`。

### 4.2 运行后你应该看到

```text
data/input/<case_id>/
data/input_inference/<case_id>/
data/g1_data_placement_manifest.csv
```

### 4.3 如果没有 fake T2W 清单

脚本会先找：

```text
work_space/G2/results/qc/official_fake_t2w_cases_by_gzip_header_2026-06-15.csv
```

如果这个文件不存在，它会退回到：

```text
work_space/G2/results/qc/official_t2w_gzip_header_audit_2026-06-15.csv
```

然后自动筛 `t2w_is_fake_by_gzip_header=True` 的病例。

---

## 5. 第三步：G1 预处理

命令：

```bash
python preprocess.py
```

### 5.1 这一步会做什么

1. 扫描 `data/input/`
2. 只保留 `t1n/t1c/t2w/t2f` 四模态齐全病例
3. 缺 `t2w` 的病例自动跳过
4. 编码成 latent
5. 生成 `data/data_csv.csv`
6. 生成 `data/data_csv_skipped_subjects.csv`

### 5.2 结果判断

你应该看到：

```text
data/latents/<case_id>/*.npy
data/data_csv.csv
data/data_csv_skipped_subjects.csv
```

如果某病例缺 `t2w`，它应该只出现在 `data_csv_skipped_subjects.csv`，不应该进 `data_csv.csv`。

---

## 6. 第四步：写入固定 train/val/test

命令：

```bash
python mark_val_split_from_g2.py
```

### 6.1 这一步做什么

它会按 G2 的 fixed train/val/test split：

1. 优先读取 `splits_final_train_val_test.json`
2. 如果新文件不存在，才回退读取旧的 `splits_final_fold0_realval.json`
3. 读取 `nnunet_case_mapping_realonly.csv`
4. 把 `data/data_csv.csv` 里对应病例改成 `train`、`val` 或 `test`
5. 其余无法匹配但已经进入 CSV 的病例保守维持 `train`

### 6.2 为什么不用手工改

因为手工改容易把 ID 改错，也容易把应该进 `val/test` 的病例漏掉。现在这步是自动的。

### 6.3 正常输出数量

如果使用当前 2026 Task1 数据和 G2 默认 split，G1 完整真实 T2W 子集应看到：

```text
train=660
val=160
test=210
```

如果你的数量不同，先检查：

1. `prepare_g1_t2w_data.py` 是否使用了同一套 G2 manifest。
2. `preprocess.py` 是否跳过了缺 `t2w` 或 fake T2W 病例。
3. `splits_final_train_val_test.json` 是否来自当前这套数据。

---

## 7. 第五步：生成训练需要的辅助文件

### 7.1 肿瘤掩码

```bash
python generate_attmask.py
```

用途：给 BBDM 的 loss 计算肿瘤掩码。

### 7.2 通道权重

```bash
python compute_weights.py
```

用途：算 `channel_importance_weights`，再把结果填回 `training_bbdm.py`。

---

## 8. 第六步：训练

### 8.1 EncDec

```bash
python training_endec.py
```

### 8.2 BBDM

```bash
python training_bbdm.py
```

### 8.3 你主要看什么

1. `bb_scheduler.s`，这是最先调的超参数。
2. `weight_decay`，放在 `s` 后面再看。
3. `channel_importance_weights`，要跟当前数据集重新算。
4. `batch_size`，显存不够就降。

### 8.4 训练后 sanity check：用 Validation 跑一次 evaluate

`Validation/` 通常没有 `seg`，但有 `t1n/t1c/t2w/t2f`，所以可以用来检查模型生成的 T2W 与真实 T2W 的图像相似度。它不进入训练，只做 sanity check：

```bash
TASK1_ROOT=/path/to/2026的task1以及数据

python evaluate.py \
  --input_dir "$TASK1_ROOT/Validation" \
  --synthesis_type ensamble \
  --gpu_id 0 \
  --verbose \
  --save_csv data/eval_validation_metrics.csv \
  --save_output
```

输出：

```text
data/eval_validation_metrics.csv
data/eval_synthesized/<case_id>-t2w.nii.gz
```

判断方法：

1. 先看 terminal 里的 SSIM / PSNR / MSE / MAE 汇总是否异常。
2. 再抽查 `data/eval_synthesized/` 的 NIfTI，确认没有明显空白、截断、强噪声或方向错位。
3. 这一步只是 G1 自检，不等价于官方 Task1 segmentation leaderboard。

---

## 9. 第七步：G1 推理

### 9.1 推理输入

```text
data/input_inference/<case_id>/
  <case_id>-t1n.nii.gz
  <case_id>-t1c.nii.gz
  <case_id>-t2f.nii.gz
  <case_id>-seg.nii.gz
```

注意：这里**不要放 `t2w`**。

### 9.2 推理命令

```bash
python main.py \
  --input_dir data/input_inference \
  --output_dir data/output \
  --synthesis_type ensamble \
  --gpu_id 0 \
  --verbose \
  --compute_bmask
```

### 9.3 推理输出

```text
data/output/<case_id>/
  <case_id>-t1n.nii.gz
  <case_id>-t1c.nii.gz
  <case_id>-t2f.nii.gz
  <case_id>-seg.nii.gz
  <case_id>-t2w.nii.gz
```

其中前四个一般是源文件软链接，最后一个是 G1 生成的 T2W。

如果服务器输出盘比较大，可以把 `--output_dir` 指到外部路径，例如 `/scratch/<user>/g1_t2w_output`。后面 G2 intake 的 `--synthetic-run-root` 必须跟这个输出路径一致。

---

## 10. 第八步：G2 接收 G1 输出

命令：

```bash
python work_space/G2/code/g2_synthetic_raw_intake_qc.py \
  --synthetic-run-root work_space/G1/code/brats2025-latent-ensemble-synthesis-main/data/output \
  --synthetic-run-id g1_t2w_completion_v1 \
  --refresh-templates
```

### 10.1 这一步会生成

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

### 10.2 你看什么结果

1. `accepted_for_training`
2. `accepted_for_ablation_only`
3. `rejected`
4. `needs_regeneration`

### 10.3 completion 口径

G1 的 `data/output/<case_id>/` 会被 G2 识别成：

1. `source_case_id = <case_id>`
2. `label_kind = completion`
3. `label_index = 0`

如果 source 是 internal val、internal test 或官方 validation，G2 不会把它直接塞进主训练集，但会保留为受控消融或复查记录。

---

## 11. 第九步：物化 nnU-Net 数据集

默认只物化 `accepted_for_training=True` 的样本。

```bash
python work_space/G2/code/g2_materialize_nnunet_dataset.py \
  --output-root /path/to/nnUNet_raw \
  --synthetic-accepted-manifest work_space/G2/results/manifests/synthetic_accepted_manifest_g1_t2w_completion_v1.csv \
  --dataset-id 261 \
  --dataset-name BraTS2026_MET_RealSynth_G1V1 \
  --channel-order g2_official \
  --mode symlink
```

如果要把 `accepted_for_ablation_only=True` 也一起放进去做受控消融，再加：

```bash
--include-ablation-only
```

### 11.1 通道顺序

```text
0000 = t1n
0001 = t1c
0002 = t2w
0003 = t2f
```

---

## 12. 常见问题

### 12.1 为什么训练 CSV 里没有缺 T2W 病例

因为 `preprocess.py` 已经自动跳过了它们。这是现在的正确行为。

### 12.2 为什么推理目录里没有 T2W

因为 `t2w` 是要生成的目标模态，正常就不应该放进去。

### 12.3 为什么 G2 会把某些 completion 样本标成 ablation_only

因为这些病例可能来自 internal val、internal test 或官方 validation。它们可以做质量分析，但不应该混进主训练集。

### 12.4 为什么 `accepted_for_ablation_only` 默认不进主训练集

因为现在物化脚本默认只收 `accepted_for_training=True`。要做消融时再显式加 `--include-ablation-only`。

---

## 13. 最后检查清单

跑完后至少确认：

1. `data/input/` 里都是完整四模态病例。
2. `data/input_inference/` 里没有 `t2w`。
3. `data/data_csv.csv` 没有缺 `t2w` 的病例。
4. `mark_val_split_from_g2.py` 已执行，并且 `data/data_csv.csv` 中存在 `train/val/test` 三类 split。
5. G1 推理输出出现在 `data/output/<case_id>/`。
6. G2 能生成 accepted/rejected/QC/报告文件。
7. nnU-Net 物化时没有把 ablation-only 样本误塞进主训练集。
