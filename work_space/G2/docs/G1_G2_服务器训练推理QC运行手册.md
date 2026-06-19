# G1 T2W 缺失模态填补服务器运行手册

更新日期：2026-06-19
适用对象：负责先把当前 G1 代码跑通的操作者。当前目标是先完成 `t1n+t1c+t2f(+seg) -> t2w` 的缺失模态填补训练与推理；G2 的 QC、accepted/rejected 判定和 nnU-Net 导出属于 G1 输出完成后的后续阶段。

## 0. 你先记住一句话

当前 G1 代码是 **T2W 缺失模态填补**，不是完整的 diffusion synthetic augmentation。操作者现在先跑 G1 completion；G2 只提供前置清单和固定划分，等 G1 产出 `data/output/<case_id>/<case_id>-t2w.nii.gz` 后再接收和 QC。

不要手动删坏掉的 `t2w`，不要手动改 `data/data_csv.csv`，不要把系统 Python 和 Conda/uv 混起来。这套流程已经改成：

1. G2 已生成或刷新真实数据清单、fake T2W 清单和固定 train/val/test 划分。
2. G1 用 `prepare_g1_t2w_data.py` 自动摆放 `data/input/` 和 `data/input_inference/`。
3. G1 训练只使用完整真实 T2W 病例；缺失或 fake T2W 病例不进 `data/data_csv.csv`。
4. G1 推理只要求 `t1n/t1c/t2f/seg`，`data/input_inference/` 中没有 `t2w` 是正确状态。
5. G2 后续接收 G1 输出，做 QC、accepted/rejected、ablation-only 和 nnU-Net 物化；这不是操作者当前第一步要跑的内容。

当前两条线要分清：

1. **已在跑的线：T2W completion / imputation**。当前 G1 代码就是这条线，先补齐 fake/broken T2W 病例。
2. **后续还需要的线：diffusion synthetic augmentation**。这才对应“用 diffusion 替代往年 GAN/GliGAN 做数据增强”，目前不能把当前 completion 代码当成完整 augmentation pipeline。

操作者当前只需要跑这一串：

```bash
cd work_space/G1/code/brats2025-latent-ensemble-synthesis-main

python test_vae.py
python prepare_g1_t2w_data.py --mode symlink --clean --overwrite
python preprocess.py
python mark_val_split_from_g2.py
python generate_attmask.py
python compute_weights.py

python training_endec.py
python training_bbdm.py

python main.py \
  --input_dir data/input_inference \
  --output_dir data/output \
  --synthesis_type ensamble \
  --gpu_id 0 \
  --verbose \
  --compute_bmask
```

如果 `prepare_g1_t2w_data.py` 报 manifest 路径找不到或源 NIfTI 不存在，先停下来让 G2 操作者刷新服务器路径下的 `work_space/G2/results/`，不要手工改病例目录。

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

服务器操作者通常只会拿到下面三个目录。不要拆散病例文件，也不要把 corrected labels 直接混进 Training 目录；由 G2 audit 脚本统一识别和覆盖。操作者当前只需要消费 G2 已经生成好的清单；如果清单里的路径不是服务器路径，再让 G2 操作者刷新 audit。

| 原始目录 | 里面有什么 | 谁使用 | 用途 |
|---|---|---|---|
| `MICCAI-LH-BraTS2025-MET-Challenge-Training/` | 每个病例一个文件夹，通常包含 `t1n/t1c/t2w/t2f/seg` | G2 audit、G1 training | 生成真实训练 manifest、训练 G1 EncDec/BBDM、生成 attention mask 和 channel weights。 |
| `Validation/` | 每个病例一个文件夹，通常包含 `t1n/t1c/t2w/t2f`，没有 `seg` | G2 audit、G1 evaluate | 只做官方 validation 结构检查和 G1 合成质量 sanity check；不要混入主训练集。 |
| `MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels/` | 少量 corrected `*-seg.nii.gz` | G2 audit | 自动覆盖 Training 中对应病例的原始 seg，形成最终 `real_train_manifest.csv`。 |

最重要的原则：

1. `Training/` 是 G1 completion 训练的来源，但 fake/broken T2W 病例会被自动分到 `data/input_inference/`。
2. `Validation/` 不能混进主训练集；它可以拿来跑 `evaluate.py` 做图像合成 sanity check。
3. `corrected-labels/` 只由 G2 audit 读取；不要手工复制到病例文件夹里。
4. 旧 manifest 里有本机绝对路径，换服务器后必须刷新 G2 audit 或改用服务器路径的 manifest，不要直接搬旧 CSV 当成路径真相。

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

## 3. 第零步：确认 G2 前置清单

操作者当前不是先跑完整 G2 QC，而是先确认 G1 completion 需要的前置清单存在、路径有效。当前 G1 代码实际依赖的文件是：

```text
work_space/G2/results/manifests/real_train_manifest.csv
work_space/G2/results/manifests/nnunet_case_mapping_realonly.csv
work_space/G2/results/splits/splits_final_train_val_test.json
work_space/G2/results/qc/official_fake_t2w_cases_by_gzip_header_2026-06-15.csv
```

其中：

1. `prepare_g1_t2w_data.py` 使用 `real_train_manifest.csv` 和 fake T2W 清单，自动生成 `data/input/` 与 `data/input_inference/`。
2. `mark_val_split_from_g2.py` 使用 `splits_final_train_val_test.json` 和 `nnunet_case_mapping_realonly.csv`，自动把 `data/data_csv.csv` 改成 `train/val/test`。
3. `g1_gligan_source_cases_v1.csv` 是后续 synthetic augmentation 线会用的 source 清单，不是当前 T2W completion 必需文件。

如果这些文件已经存在，并且 CSV 里的原始数据路径能在服务器上访问，操作者可以直接跳到第 4 节。

如果路径仍指向本机 Mac，或者清单缺失，再由 G2 操作者刷新 audit：

命令：

```bash
TASK1_ROOT=/path/to/2026的task1以及数据

python work_space/G2/code/g2_pretraining_audit.py \
  --data-root "$TASK1_ROOT" \
  --results-root work_space/G2/results \
  --force
```

### 3.1 刷新 audit 后会生成什么

当前 T2W completion 重点看这些文件：

```text
work_space/G2/results/manifests/real_train_manifest.csv
work_space/G2/results/manifests/nnunet_case_mapping_realonly.csv
work_space/G2/results/splits/splits_final_train_val_test.json
work_space/G2/results/splits/splits_final_train_val_test_membership.csv
work_space/G2/results/qc/official_fake_t2w_cases_by_gzip_header_2026-06-15.csv
work_space/G2/results/qc/official_t2w_gzip_header_audit_2026-06-15.csv
```

下面这些是后续 G2 报告或 synthetic augmentation 会用到的文件，不是操作者当前 completion 必跑项：

```text
work_space/G2/results/manifests/real_validation_manifest.csv
work_space/G2/results/manifests/g1_gligan_source_cases_v1.csv
work_space/G2/results/splits/splits_final_fold0_realval.json
```

### 3.2 你要检查什么

1. `real_train_manifest.csv` 里路径是否指向你服务器上的原始数据。
2. `official_fake_t2w_cases_by_gzip_header_2026-06-15.csv` 是否存在，并且包含 265 个 fake/broken T2W 病例。
3. `splits_final_train_val_test.json` 是否存在。
4. `nnunet_case_mapping_realonly.csv` 是否存在。
5. `splits_final_train_val_test_membership.csv` 是否存在，方便人工核对。

如果你只换了挂载路径，但病例 ID 没变，刷新 audit 这一步就够用了。

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
4. 剩余 `829 train` 作为真实训练池；后续 synthetic augmentation source 也必须只从允许训练来源里取，但这不是当前 completion 代码的任务。

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

## 4. 操作者第一步：自动摆放 G1 数据

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

当前代码会使用 `configs.py` 里的路径约定：

```text
PATH_INPUT = data/input
PATH_INPUT_INFERENCE = data/input_inference
PATH_OUTPUT = data/output
MISSING_MODALITY = t2w
AVAILABLE_MODALITIES = t1n, t1c, t2f
```

### 4.1 这个脚本做什么

1. 从 G2 的真实训练 manifest 和 fake T2W 清单里找病例。
2. 把完整四模态病例放进 `data/input/`。
3. 把需要补 `t2w` 的病例放进 `data/input_inference/`。
4. 推理目录里**自动不放 `t2w`**。
5. 写出 `data/g1_data_placement_manifest.csv`。

如果 manifest 里的源 NIfTI 路径不存在，脚本会默认停下来，并在 `data/g1_data_placement_manifest.csv` 里写清楚缺哪个病例、哪个模态。不要绕过这个错误继续训练；这通常说明服务器路径还没刷新，或原始数据没有挂载完整。

当前默认口径应是：

```text
data/input/           1030 个完整真实 T2W 病例
data/input_inference/ 265 个需要重建 T2W 的病例
```

`data/input_inference/<case_id>/` 中如果出现 `*-t2w.nii.gz`，说明摆放不符合当前 G1 推理口径，要先修正再推理。

### 4.2 运行后你应该看到

```text
data/input/<case_id>/
data/input_inference/<case_id>/
data/g1_data_placement_manifest.csv
```

抽查一个训练病例：

```text
data/input/<case_id>/
  <case_id>-t1n.nii.gz
  <case_id>-t1c.nii.gz
  <case_id>-t2w.nii.gz
  <case_id>-t2f.nii.gz
  <case_id>-seg.nii.gz
```

抽查一个推理病例：

```text
data/input_inference/<case_id>/
  <case_id>-t1n.nii.gz
  <case_id>-t1c.nii.gz
  <case_id>-t2f.nii.gz
  <case_id>-seg.nii.gz
```

推理病例这里没有 `t2w` 是正确的，因为当前 G1 `main.py` 的真实输入是 `t1n/t1c/t2f/seg`，目标输出才是 `t2w`。

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

## 5. 操作者第二步：G1 预处理

命令：

```bash
python preprocess.py
```

### 5.1 这一步会做什么

1. 扫描 `data/input/`
2. 只保留 `t1n/t1c/t2w/t2f` 四模态齐全病例
3. 缺 `t2w` 或缺其他训练必需模态的病例自动跳过
4. 编码成 latent
5. 生成 `data/data_csv.csv`
6. 如果存在被跳过病例，则生成 `data/data_csv_skipped_subjects.csv`

### 5.2 结果判断

你应该看到：

```text
data/latents/<case_id>/*.npy
data/data_csv.csv
data/data_csv_skipped_subjects.csv  # 只有出现 skipped subject 时才一定存在
```

如果某病例缺 `t2w`，它不应该进 `data_csv.csv`。正常情况下，前一步已经把 fake/broken T2W 病例放进 `data/input_inference/`，所以 `data/input/` 本身应主要是完整四模态病例。

---

## 6. 操作者第三步：写入固定 train/val/test

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
5. 如果 CSV 里出现 G2 mapping/split 无法匹配的病例，默认直接报错停止

### 6.2 为什么不用手工改

因为手工改容易把 ID 改错，也容易把应该进 `val/test` 的病例漏掉。现在这步是自动的。

如果确实是在做临时诊断，想恢复旧的“未知病例保守归 train”行为，才加：

```bash
python mark_val_split_from_g2.py --allow-unmatched-as-train
```

正式训练不要使用这个参数。

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

## 7. 操作者第四步：生成训练需要的辅助文件

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

## 8. 操作者第五步：训练

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

## 9. 操作者第六步：G1 推理

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

注意：当前 G1 代码里的参数值拼写就是 `ensamble`，不要改成 `ensemble`。

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

## 10. G2 后续第一步：接收 G1 输出

这一节不是操作者当前第一阶段必须执行的内容。操作者当前只需要把 G1 completion 跑到 `data/output/<case_id>/<case_id>-t2w.nii.gz` 产出稳定。等这批 T2W 生成完成后，再由 G2 接收、QC 和写报告。

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

## 11. G2 后续第二步：物化 nnU-Net 数据集

这一节也不属于操作者当前任务。只有当 G2 已经完成 completion 输出的 QC，并生成 `synthetic_accepted_manifest_*` 后，才进入 nnU-Net 物化。

默认只物化 `accepted_for_training=True` 的样本。对当前 G1 completion 输出，物化脚本默认执行的是 **replace fake T2W**，不是 append duplicate case：

1. 完整真实 T2W 病例照常进入 nnU-Net。
2. fake/broken T2W 病例如果有通过 QC 的 completion 输出，就用生成的 `t2w` 替换原始 fake/broken `t2w`。
3. fake/broken T2W 病例如果没有通过 QC 的 completion 输出，默认不进入主物化数据集。
4. 非 completion 的 diffusion augmentation 样本才作为额外 synthetic case 追加。

`official_fake_t2w_cases_by_gzip_header_2026-06-15.csv` 是这一步的硬前置；如果这个清单缺失，物化脚本会默认停止，避免原始 fake/broken T2W 混进主训练。

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

如果团队专门要做“原始官方 fake T2W 不替换”的对照实验，才显式加：

```bash
--include-unreplaced-fake-t2w
```

正式主训练数据不要加这个参数。

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

### 12.5 操作者现在要不要跑 G2

操作者当前只需要确认 G2 前置清单存在并且路径有效，然后跑 G1 completion。G2 audit、QC、accepted/rejected、nnU-Net 物化都属于后续阶段。

如果服务器上的 `real_train_manifest.csv` 路径不对，才需要 G2 操作者刷新 audit；这不是 G1 completion 算法本身的一部分。

### 12.6 当前 G1 代码是不是完整 synthetic augmentation

不是。当前 `brats2025-latent-ensemble-synthesis-main` 是缺失 T2W 模态填补代码，核心任务是从 `t1n/t1c/t2f` 生成 `t2w`。

完整数据增强还需要另一条 diffusion synthetic augmentation 代码线。那条线后续应生成可追溯 synthetic cases，再交给 G2 做 manifest、QC、accepted/rejected 和下游消融。

---

## 13. 最后检查清单

操作者跑完 G1 completion 后至少确认：

1. `data/input/` 里都是完整四模态病例。
2. `data/input_inference/` 里没有 `t2w`。
3. `data/data_csv.csv` 没有缺 `t2w` 的病例。
4. `mark_val_split_from_g2.py` 已执行，并且 `data/data_csv.csv` 中存在 `train/val/test` 三类 split。
5. G1 推理输出出现在 `data/output/<case_id>/`。
6. `data/output/<case_id>/` 中包含源 `t1n/t1c/t2f/seg` 和生成的 `t2w`。
7. 推理日志里没有大量 `skipped incomplete subjects`。

G2 后续接收时再确认：

1. G2 能生成 accepted/rejected/QC/报告文件。
2. completion 输出没有 NaN/Inf、空图、常数图、shape/spacing/affine 不一致等硬错误。
3. internal val/test 来源的 completion 样本没有误进主训练集。
4. nnU-Net 物化时没有把 ablation-only 样本误塞进主训练集。
5. fake/broken T2W 病例没有以原始 fake T2W 进入主物化数据；要么被 accepted completion 替换，要么被跳过。
