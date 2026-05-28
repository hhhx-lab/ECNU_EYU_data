# BraTS 2025 MET 第一名 Docker 镜像源码查看记录

## 基本信息

镜像：

```text
brainles/brats25_met_gli-team:latest
```

本地拉取结果：

```text
Digest: sha256:f83d914d4d7a9d316f7605abaf03fa6adbcf0e86d840d93276e86ad65b2a2a82
Image ID: f83d914d4d7a
Image size shown by docker image ls: 13GB
```

镜像配置：

```text
WorkingDir: /workspace
Cmd: ["python", "main.py"]
nnUNet_raw=/raw
nnUNet_preprocessed=/preprocessed
nnUNet_results=/results
PYTORCH_VERSION=2.7.1
```

## 导出的文件

已导出轻量源码和配置文件，不导出 820MB 模型权重：

```text
docker_source/brats25_met_gli-team/
├── workspace/main.py
├── raw/Dataset101_submission/dataset.json
├── preprocessed/Dataset101_submission/
│   ├── dataset.json
│   ├── dataset_fingerprint.json
│   └── plans.json
└── results/Dataset101_submission/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres/
    ├── dataset.json
    ├── dataset_fingerprint.json
    └── plans.json
```

镜像内真实权重位置：

```text
/results/Dataset101_submission/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres/fold_all/checkpoint_final.pth
```

权重大小约：

```text
820475671 bytes
```

## main.py 做了什么

`/workspace/main.py` 是一个很薄的推理 wrapper，不包含训练代码。

核心逻辑：

1. 读取容器挂载的 `/input`。
2. 创建 `/input_tmp`。
3. 把 BraTS-MET 四模态文件名转换成 nnU-Net 输入名。
4. 调用 `nnUNetv2_predict`。
5. 删除临时目录和非 `.nii.gz` 输出。

文件名映射：

```python
filename_mapping = {
    "-t1n.nii.gz": "_0000.nii.gz",
    "-t1c.nii.gz": "_0001.nii.gz",
    "-t2w.nii.gz": "_0002.nii.gz",
    "-t2f.nii.gz": "_0003.nii.gz",
}
```

推理命令：

```bash
nnUNetv2_predict \
  -i /input_tmp \
  -o /output \
  -d Dataset101_submission \
  -tr nnUNetTrainer \
  -p nnUNetResEncUNetXLPlans \
  -c 3d_fullres \
  -f all \
  -chk checkpoint_final.pth
```

## 结论

这个 Docker 镜像里没有看到完整训练源码，也没有看到复杂自定义 pipeline 源码。能看到的是：

1. 一个推理入口 `main.py`。
2. nnU-Net 数据集配置、fingerprint 和 plans。
3. 一个 `fold_all/checkpoint_final.pth` 权重。

因此它更像是：

```text
nnU-Net v2 ResEncUNet XL 推理封装 + 已训练好的 fold_all 权重
```

不是完整可复现实验源码。

对今年任务有用的信息主要是：

1. 2025 第一名至少在提交镜像中使用 `nnUNetResEncUNetXLPlans`。
2. 训练/推理通道顺序是 `t1n, t1c, t2w, t2f`。
3. 使用 `3d_fullres` 配置。
4. 使用 `fold_all` 单模型 checkpoint，而不是 5-fold ensemble。
