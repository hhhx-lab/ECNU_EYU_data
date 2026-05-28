# BraTS 2023 本地 Conda 环境、运行结果与权重路径记录

## 1. 本次工作结论

本次只完成环境搭建、依赖修正、代码链路 smoke test 和权重路径定位；没有下载作者发布的 39.8 GB 权重包。

已完成：

1. 创建独立 Conda 环境 `brats2023_seg`。
2. 修正 `requirements_seg.txt`，避免安装错误的旧版 `nnunet`。
3. 修正 `Segmentation_Tasks/mednext/setup.py` 的包发现错误。
4. 跑通 2023 inference 代码中不依赖真实权重的最小数据链路。
5. 找到作者官方发布页、GitHub 仓库和大权重 zip 的路径。

未完成：

1. 没有下载 `BraTS_2023_2024_code_with_weights.zip`。
2. 没有执行真实冠军模型推理，因为本地缺少 `checkpoint_final.pth`。

## 2. 环境结论

已创建独立 Conda 环境：

```bash
conda activate brats2023_seg
```

环境位置：

```text
/Users/hwaigc/miniforge3/envs/brats2023_seg
```

本机为 macOS arm64。当前 PyTorch 状态：

```text
torch 2.5.1
cuda False
mps True
```

因此本机适合做依赖检查、数据格式转换、后处理 smoke test、小规模 CPU/MPS 调试；冠军级完整推理/训练建议放到 NVIDIA GPU 机器上运行。

## 3. requirements_seg.txt 修正

已更新：

```text
往年文章和code/2023/requirements_seg.txt
```

修正点：

1. `nnunet` 不应直接安装。
   - 该仓库实际使用本地 `Segmentation_Tasks/nnUNet_install` 中的 `nnunetv2`。
   - 直接 `pip install nnunet` 容易装到旧版 nnU-Net，污染环境。

2. 缺少 `connected-components-3d`。
   - `BraTS2023_inference/infer_low_disk.py` 使用了 `import cc3d`。
   - 正确安装包名是 `connected-components-3d`。

3. `pathlib` 不需要安装。
   - Python 3.11 标准库已经包含 `pathlib`。
   - 单独安装 PyPI 的 `pathlib` 属于旧兼容包，没必要。

4. `mednext/setup.py` 有包发现配置错误。
   - 原始写法查找 `mednextv1` 包。
   - 但源码目录实际是 `nnunet_mednext`。
   - 已修正为：

```python
packages=find_namespace_packages(include=["nnunet_mednext", "nnunet_mednext.*"])
```

更新后的安装方式：

```bash
cd "往年文章和code/2023"
conda activate brats2023_seg
python -m pip install -r requirements_seg.txt
```

## 4. 本地安装命令

```bash
conda create -n brats2023_seg python=3.11 -y
conda activate brats2023_seg

python -m pip install --upgrade pip

python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  monai nilearn nibabel matplotlib einops tqdm SimpleITK \
  connected-components-3d

python -m pip install -e "往年文章和code/2023/Segmentation_Tasks/nnUNet_install"
python -m pip install -e "往年文章和code/2023/Segmentation_Tasks/mednext"
```

安装后检查：

```bash
python -c "import torch, monai, nibabel, SimpleITK, cc3d, nnunetv2, nnunet_mednext; print('imports_ok')"
python -m pip check
```

当前结果：

```text
imports_ok
No broken requirements found.
```

## 5. 运行结果整理

已写入并运行：

```text
tmp/brats2023_smoke/run_smoke.py
```

执行命令：

```bash
conda run -n brats2023_seg python tmp/brats2023_smoke/run_smoke.py
```

该测试覆盖：

1. 生成 1 个 BraTS2023 风格假样本。
2. 调用 `convert_data_step`，把 BraTS case 转为 nnU-Net `_0000` 到 `_0003` 输入格式。
3. 构造假概率图 `.npz`，调用 `convert_prob_to_label` 生成分割标签。
4. 调用 `thresholding_step` 执行连通域体积阈值后处理。
5. 调用 `convert_back_BraTS_step` 回写到 BraTS2023 标签约定。

当前 smoke test 输出：

```text
SMOKE_OK
converted_files=[
  'BraTS-Smoke-00001_0000.nii.gz',
  'BraTS-Smoke-00001_0001.nii.gz',
  'BraTS-Smoke-00001_0002.nii.gz',
  'BraTS-Smoke-00001_0003.nii.gz'
]
labels=[0, 1, 2, 3]
```

结果解释：

1. `converted_files` 说明 BraTS2023 输入四模态已正确转换成 nnU-Net 推理格式：
   - `_0000` 对应 `t1c`
   - `_0001` 对应 `t1n`
   - `_0002` 对应 `t2f`
   - `_0003` 对应 `t2w`

2. `labels=[0, 1, 2, 3]` 说明后处理和 BraTS 标签回写链路保留了背景与三个 2023 glioma 标签。

3. 本 smoke test 没有验证真实模型精度，只验证依赖、NIfTI I/O、格式转换、连通域阈值、标签回写这些工程链路。

4. `python -m pip check` 输出 `No broken requirements found.`，说明当前环境没有显式 Python 包依赖冲突。

## 6. 作者与官方权重路径

作者官方发布信息：

```text
名称：Faking_it team! BraTS submissions.
主要作者：Ferreira, André
Zenodo DOI：10.5281/zenodo.14001262
Zenodo 页面：https://zenodo.org/records/14001262
GitHub 仓库：https://github.com/andre-fs-ferreira/BraTS_2023_2024_solutions
权重文件：BraTS_2023_2024_code_with_weights.zip
文件大小：39.8 GB
md5：024709c75f1246622c300dfd89fdebd2
```

直接下载路径仅记录，不要默认执行：

```text
https://zenodo.org/records/14001262/files/BraTS_2023_2024_code_with_weights.zip?download=1
```

## 7. 本地权重应放置目录

本地 `往年文章和code/2023/Segmentation_Tasks/nnUNet/nnUNet_results` 目录目前只有结构文件，例如：

```text
dataset.json
plans.json
debug.json
progress.png
```

没有实际推理所需权重：

```text
checkpoint_final.pth
checkpoint_best.pth
```

所以目前已经确认的是：

1. 环境可用。
2. 关键依赖可导入。
3. 2023 inference 代码中不依赖权重的数据转换、概率图合成、后处理、标签回写路径已跑通。

完整冠军推理还需要从作者 Zenodo 下载对应权重，并按代码期望目录放入 `nnUNet_results`。

2023 默认推理脚本 `BraTS2023_inference/main.py` 使用：

```python
ensemble_code = 'rGB_rGL_rGS'
```

因此最少需要补齐以下 3 个模型目录，每个目录下需要 `fold_0` 到 `fold_4` 的真实 checkpoint：

```text
往年文章和code/2023/Segmentation_Tasks/nnUNet/nnUNet_results/
└── Dataset232_BraTS_2023_rGANs/
    ├── nnUNetTrainer__nnUNetPlans__3d_fullres/
    │   ├── dataset.json
    │   ├── plans.json
    │   ├── fold_0/checkpoint_final.pth
    │   ├── fold_1/checkpoint_final.pth
    │   ├── fold_2/checkpoint_final.pth
    │   ├── fold_3/checkpoint_final.pth
    │   └── fold_4/checkpoint_final.pth
    ├── nnUNetTrainer_SwinUNETR__nnUNetPlans__3d_fullres_SwinUNETR/
    │   ├── dataset.json
    │   ├── plans.json
    │   ├── fold_0/checkpoint_final.pth
    │   ├── fold_1/checkpoint_final.pth
    │   ├── fold_2/checkpoint_final.pth
    │   ├── fold_3/checkpoint_final.pth
    │   └── fold_4/checkpoint_final.pth
    └── nnUNetTrainerBN_BS5_RBT_DS_BD_PS__nnUNetPlans__3d_fullres_BN_BS5_RBT_DS_BD_PS/
        ├── dataset.json
        ├── plans.json
        ├── fold_0/checkpoint_final.pth
        ├── fold_1/checkpoint_final.pth
        ├── fold_2/checkpoint_final.pth
        ├── fold_3/checkpoint_final.pth
        └── fold_4/checkpoint_final.pth
```

说明：

1. 上面的 `dataset.json` 和 `plans.json` 本地已有。
2. `fold_0` 到 `fold_4` 的目录已补齐为空目录。
3. `checkpoint_final.pth` 不应手工伪造，必须来自作者权重包或自己训练得到的结果。
4. 如果后续把 `ensemble_code` 改成全量 `'GB_GL_GS_RB_RL_RS_rGB_rGL_rGS'`，还需要补齐 `Dataset233_BraTS_2023_Naida` 和 `Dataset236_BraTS_2023_GANs` 对应目录。
