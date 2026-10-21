# S2 Original E Docker Reserve

状态：`experimental_unvalidated`。

这是 BraTS 2026 Task 1 原 E 的隔离 Docker 储备上下文。它绑定已经完成179例技术验证的冻结权重：

- trainer：`nnUNetTrainerBraTS2026RCFocalCompletionFineTune`
- epoch：`200`
- checkpoint SHA256：`4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267`
- PyTorch/CUDA：`2.7.1` / `12.8`

该目录不会改写原 checkpoint，也不包含 Synapse 凭据、登录、推送或提交动作。跳过的 Reference、Development 96、独立 holdout 和 Gate-0/1A/1B/2 均不得视为通过。

## 容器契约

- 从只读 `/input` 发现 `BraTS-MET-xxxxx-xxx` 病例目录。
- 验证每例 `t1n/t1c/t2w/t2f` 四个 NIfTI 文件。
- 在容器临时目录中映射为 nnU-Net `0000-0003` 通道。
- 使用冻结推理 shim，仅加载 segmentation checkpoint。
- 验证输出覆盖、标签集合、shape、spacing 和 affine。
- 仅把平铺的 `<case-id>.nii.gz` 写入 `/output`。

## 静态验证

使用项目约定的 Conda Python 执行：

```bash
python validate_context.py
python -m unittest discover -s tests -v
```

`STATIC_CONTEXT_SHA256.json` 排他记录该储备上下文中全部静态文件的 SHA256；验收器会同时验证其覆盖集合和逐文件哈希。checkpoint 保持在原目录，由独立冻结 SHA256 绑定，不写入静态上下文。

## 本地构建

构建会下载 CUDA/PyTorch 基础镜像和 Python wheel，因此必须在获得下载授权后执行：

```bash
cp .env.example .env
./build_local.sh
```

构建目标固定为 `linux/amd64`。脚本会创建一次性、仅含 `checkpoint_final.pth` 的 BuildKit named context，并把 checkpoint 放在最后的内容层；原 `fold_0` 中的历史预测不会进入构建上下文。后续换 checkpoint 时可以复用前面的依赖缓存。

## GPU 离线测试

必须在带 Docker、NVIDIA Container Toolkit 和 NVIDIA GPU 的 x86_64 Linux 主机运行。H20 正式训练主机当前没有容器运行时，本地 ARM64 Mac 也不能完成 GPU 验收。

```bash
docker compose run --rm model
```

正式测试必须保持 `--network none`、输入只读、输出目录为空，并检查完整病例覆盖。未经另行明确授权，不得推送到 Synapse。
