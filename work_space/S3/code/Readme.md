# Swin UNETR for BraTS 2025 MET Challenge

实现 Swin UNETR 模型，用于脑转移瘤分割。在验证阶段会输出Dice系数以及小病灶（<275 mm³）检测的 F1‑score 和 AUC（类别 1,3,4）。

## 文件说明

| 文件 | 作用 |
|------|------|
| `main.py` | 训练入口脚本，解析参数，调用数据加载和训练流程（一般无需修改）。 |
| `trainer.py` | 训练/验证核心逻辑。包含训练循环、验证循环、Dice 计算、小病灶检测指标（F1/AUC）计算、模型保存。 |
| `make_split.py` | 生成训练/验证集划分 JSON 文件。将数据目录下所有病例随机分为 80% 训练（`fold=1`）和 20% 验证（`fold=0`），输出 `full_split.json`。 |
| `utils/data_utils.py` | 数据加载工具。适配原始数据名称，根据 JSON 中的 `fold` 字段自动划分训练集和验证集。 |


## 环境配置

创建 Conda 环境并安装依赖：

```bash
conda create -n brats_swin python=3.9 -y
conda activate brats_swin
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install monai[all]==1.3.0 nibabel einops timm tensorboardX ml-collections scipy tqdm
```

# 环境配置
从 BraTS 2025 MET Challenge 官网 下载 MICCAI-LH-BraTS2025-MET-Challenge-Training.zip，解压得到文件夹 MICCAI-LH-BraTS2025-MET-Challenge-Training/
确保每个病例文件夹内的文件命名格式如下：

```bash
BraTS-MET-xxxxx-xxx-t1n.nii.gz
BraTS-MET-xxxxx-xxx-t1c.nii.gz
BraTS-MET-xxxxx-xxx-t2w.nii.gz
BraTS-MET-xxxxx-xxx-t2f.nii.gz   
BraTS-MET-xxxxx-xxx-seg.nii.gz
```
将所有 BraTS-MET-* 文件夹放在同一顶层目录中。如果存在 UCSD - Training 等子目录，请将其中的 BraTS-MET-* 文件夹移出到顶层，并删除空目录。

# 生成全量数据训练/验证集划分

运行 make_split.py 脚本，它会自动扫描数据目录，随机划分 80% 病例为训练集（fold=1），20% 为验证集（fold=0），并生成 full_split.json。

```bash
python make_split.py
```

生成的 full_split.json 内容示例：

```json
{
  "training": [
    {"image": "BraTS-MET-00001-000", "label": "BraTS-MET-00001-000", "fold": 1},
    {"image": "BraTS-MET-00002-000", "label": "BraTS-MET-00002-000", "fold": 1},
    ...
    {"image": "BraTS-MET-00100-000", "label": "BraTS-MET-00100-000", "fold": 0}
  ]
}
```
注意：fold=1 为训练集，fold=0 为验证集。脚本使用随机种子 42，保证可复现。


# 全量数据训练（200 epochs）
使用上一步生成的 full_split.json 启动全量训练。
```bash
nohup python main.py \
    --data_dir /path/to/MICCAI-LH-BraTS2025-MET-Challenge-Training \
    --json_list ./full_split.json \
    --logdir ./exp_full \
    --max_epochs 200 \
    --batch_size 2 \
    --val_every 10 \
    --out_channels 5 \
    --gpu 0 > train_full.log 2>&1 &
```

实时查看训练日志：
```bash
tail -f train_full.log
```

重要参数说明
```
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | 必填 | 数据根目录（包含所有 BraTS-MET-* 文件夹） |
| `--json_list` | 必填 | 划分 JSON 文件路径（如 `full_split.json`） |
| `--logdir` | `./runs` | 保存模型和日志的目录 |
| `--max_epochs` | `200` | 总训练轮数 |
| `--batch_size` | `2` | 批大小（24GB 显存建议 2，若显存不足可改为 1） |
| `--val_every` | `10` | 每多少个 epoch 验证一次 |
| `--out_channels` | `5` | 输出通道数（固定为 5，对应背景 + 4 类标签） |
| `--gpu` | `0` | GPU 编号 |
```


# 输出结果
训练过程中，每 --val_every 个 epoch 会输出：
验证损失

Dice 系数（整体平均）

小病灶检测指标（类别 1,3,4）：
F1‑score（基于多个 IoU 阈值：0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9）
AUC（F1 曲线下面积）

最佳模型（验证 Dice 最高）会被保存为 {logdir}/best_model_dice.pth。

# 常见问题
1. FileNotFoundError: No such file or directory:
检查 --data_dir 路径是否正确。
确认病例文件夹内文件命名是否与代码中的模态列表 ["t1n","t1c","t2w","t2f"] 一致。官方数据即为该格式，无需修改。
确保 JSON 文件中的 image 和 label 字段只包含病例名（如 BraTS-MET-00001-000），不包含路径。

2. 验证 Dice 始终为 0
正常现象，初期模型未收敛。继续训练几十个 epoch 后应该会逐渐上升。

3. 检测指标（F1/AUC）始终为 0
可能验证集中没有小病灶（体积 < 275 mm³）。检查数据分布，或训练更多 epoch 后模型预测出小病灶时指标会变为非零。
