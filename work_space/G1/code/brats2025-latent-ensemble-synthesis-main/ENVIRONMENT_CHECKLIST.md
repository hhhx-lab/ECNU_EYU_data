# G1 环境清单

更新日期：2026-06-18

## 1. 适用范围

这份清单只用于新 G1 代码的环境交接，不要求在本机完整跑项目。

当前 G1 代码口径：

1. 训练集放在 `data/input/`，必须是完整四模态加 `seg`。
2. 推理集放在 `data/input_inference/`，只放 `t1n/t1c/t2f/seg`，**不要放 T2W**。
3. 任务是缺失模态补全，目标是生成 `t2w`。

## 2. 推荐 Conda 环境

建议环境名：

```bash
conda create -n g1_t2w_bbdm python=3.10
```

原则：

1. 只用 Conda 环境，不混系统 Python。
2. 不用 `sudo pip`。
3. 不要把依赖装进 Homebrew Python。

## 3. 必需依赖

### 3.1 核心依赖

```text
torch
torchvision
monai
nibabel
numpy<2
pandas
Pillow
tqdm
tensorboard
```

### 3.2 推荐版本范围

按当前 README 的口径：

1. `python >= 3.10`
2. `torch >= 2.0.0, < 2.8.0`
3. `monai >= 1.4.0`
4. `nibabel >= 5.0.0`
5. `numpy >= 1.24.0, < 2.0.0`
6. `pandas >= 2.0.0`

### 3.3 可选依赖

只在需要时安装：

1. `TotalSegmentator`：只有启用 `--compute_bmask` 时才需要。
2. `nnunetv2`：如果训练机上要做进一步分割/格式对接才需要。

## 4. 硬件要求

### 4.1 训练/推理机

1. 有 GPU 更好。
2. 如果要用 Flash Attention，建议 Ampere 或更新架构。
3. CUDA 环境建议 `11.6+`，更稳妥是 `12.x`。

### 4.2 本机

本机只适合：

1. 改 CSV。
2. 生成 manifest。
3. 做数据摆放脚本的 `manifest-only` 检查。
4. 看小样本结果。

不建议在本机做：

1. 全量 `preprocess.py`。
2. 大规模训练。
3. 大规模推理。

## 5. 必备文件

### 5.1 模型权重

需要存在：

```text
weights/vae/autoencoder_epoch273.pt
```

这个权重文件不跟 Git 提交，需要在服务器上单独放到上述位置。放好后先运行：

```bash
python test_vae.py
```

看到 `VAE loaded OK` 后，再继续 `preprocess.py`、训练和推理。

### 5.2 数据目录

训练集：

```text
data/input/<case_id>/
  <case_id>-t1n.nii.gz
  <case_id>-t1c.nii.gz
  <case_id>-t2w.nii.gz
  <case_id>-t2f.nii.gz
  <case_id>-seg.nii.gz
```

推理集：

```text
data/input_inference/<case_id>/
  <case_id>-t1n.nii.gz
  <case_id>-t1c.nii.gz
  <case_id>-t2f.nii.gz
  <case_id>-seg.nii.gz
```

注意：

1. 推理集里不要保留坏掉的 T2W。
2. 文件后缀要和 README 里一致。
3. `case_id` 命名要统一，不要混不同批次写法。

G1 推理输出：

```text
data/output/<case_id>/
  <case_id>-t1n.nii.gz
  <case_id>-t1c.nii.gz
  <case_id>-t2f.nii.gz
  <case_id>-seg.nii.gz
  <case_id>-t2w.nii.gz
```

其中 `t1n/t1c/t2f/seg` 是源输入软链接，`t2w` 是模型生成结果。这个目录可以直接交给 G2 的 raw intake/QC 脚本。

## 6. G2 对接文件

建议同时准备这几份 G2 输出：

1. `work_space/G2/results/manifests/real_train_manifest.csv`
2. `work_space/G2/results/qc/official_fake_t2w_cases_by_gzip_header_2026-06-15.csv`
3. `work_space/G2/results/splits/splits_final_fold0_realval.json`
4. `work_space/G2/results/manifests/nnunet_case_mapping_realonly.csv`

对应脚本：

1. `prepare_g1_t2w_data.py`
2. `mark_val_split_from_g2.py`

## 7. 推荐执行顺序

### 7.1 数据摆放

先准备目录：

```bash
python prepare_g1_t2w_data.py --mode symlink --clean --overwrite
```

该脚本会默认：

1. 把完整真实 T2W 病例放进 `data/input/`
2. 把 fake T2W 病例放进 `data/input_inference/`
3. 在 `data/input_inference` 中去掉 T2W

### 7.2 预处理

```bash
python preprocess.py
```

作用：

1. 扫描 `data/input/`
2. 做归一化和裁剪
3. 编码成 latent
4. 生成 `data/data_csv.csv`

### 7.3 写回固定验证集

```bash
python mark_val_split_from_g2.py
```

作用：

1. 读取 G2 fixed fold0
2. 把 `data/data_csv.csv` 中对应病例标成 `val`
3. 其余标成 `train`

### 7.4 生成肿瘤掩码

```bash
python generate_attmask.py
```

前提：

1. 有 `seg` 文件。
2. 需要用于 BBDM loss。

### 7.5 计算通道权重

```bash
python compute_weights.py
```

输出的 `channel_importance_weights` 需要填回 `training_bbdm.py`。

### 7.6 训练

```bash
python training_endec.py
python training_bbdm.py
```

### 7.7 推理

```bash
python main.py --synthesis_type ensamble --gpu_id 0 --verbose
```

### 7.8 评估

```bash
python evaluate.py --gpu_id 0 --verbose
```

## 8. 关键超参数

优先级：

1. `bb_scheduler.s`
2. `weight_decay`
3. `channel_importance_weights`
4. `sample_step`
5. `lr`

建议：

1. 先调 `s`
2. 再调 `weight_decay`
3. 最后再细调别的参数

## 9. 运行前检查

建议至少确认：

1. `python --version` 是 3.10 系列
2. `import torch, monai, nibabel, pandas, numpy` 没报错
3. `weights/vae/autoencoder_epoch273.pt` 存在
4. `data/input/` 和 `data/input_inference/` 路径结构正确
5. `data/input_inference/` 没有 `t2w`
6. `data/data_csv.csv` 里 `split=train/val` 已统一

## 10. 不需要做的事

1. 不要在本机强行跑完整训练。
2. 不要在本机把大数据复制一份到项目里。
3. 不要把 `T2W` 的坏文件留在推理目录里。
4. 不要手工一个个改 `train/val`，用脚本做。
