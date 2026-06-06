# Swin UNETR for BraTS 2025 MET Challenge

本项目实现 Swin UNETR 模型，用于脑转移瘤分割。在验证阶段会自动输出Dice 系数以及小病灶（<275 mm³）检测的 F1‑score 和 AUC（类别 1,3,4）。

## 文件说明

| 文件 | 作用 |
|------|------|
| `main.py` | 训练入口脚本，解析参数，调用数据加载和训练流程（一般无需修改）。 |
| `trainer.py` | 训练/验证核心逻辑。包含训练循环、验证循环、Dice 计算、小病灶检测指标（F1/AUC）计算、模型保存。 |
| `make_split.py` | 生成训练/验证集划分 JSON 文件。将数据目录下所有病例随机分为 80% 训练（`fold=1`）和 20% 验证（`fold=0`），输出 `full_split.json`。 |
| `utils/data_utils.py` | 数据加载工具。适配 BraTS 长名格式（`BraTS-MET-xxxxx-xxx-t1n.nii.gz` 等），根据 JSON 中的 `fold` 字段自动划分训练集和验证集。 |
| `utils/__init__.py` | 标识 `utils` 为 Python 包（空文件）。 |

## 环境配置

创建 Conda 环境并安装依赖：

```bash
conda create -n brats_swin python=3.9 -y
conda activate brats_swin
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install monai[all]==1.3.0 nibabel einops timm tensorboardX ml-collections scipy tqdm

## 数据准备

从 BraTS 2025 MET Challenge 官网 下载 MICCAI-LH-BraTS2025-MET-Challenge-Training.zip，解压得到文件夹 MICCAI-LH-BraTS2025-MET-Challenge-Training/
确保每个病例文件夹内的文件命名格式如下：

BraTS-MET-xxxxx-xxx-t1n.nii.gz
BraTS-MET-xxxxx-xxx-t1c.nii.gz
BraTS-MET-xxxxx-xxx-t2w.nii.gz
BraTS-MET-xxxxx-xxx-t2f.nii.gz   
BraTS-MET-xxxxx-xxx-seg.nii.gz

将所有 BraTS-MET-* 文件夹放在同一顶层目录中。如果存在 UCSD - Training 等子目录，请将其中的 BraTS-MET-* 文件夹移出到顶层，并删除空目录。

## 生成训练/验证集划分
运行以下脚本，自动将全部病例随机划分为 80% 训练（fold=1）和 20% 验证（fold=0），生成 full_split.json：
python make_split.py


##全量数据训练（200 epochs）
使用上一步生成的 full_split.json 启动全量训练。
bash
nohup python main.py \
    --data_dir /path/to/MICCAI-LH-BraTS2025-MET-Challenge-Training \
    --json_list ./full_split.json \
    --logdir ./exp_full \
    --max_epochs 200 \
    --batch_size 2 \
    --val_every 10 \
    --out_channels 5 \
    --gpu 0 > train_full.log 2>&1 &

实时查看训练日志：
bash
tail -f train_full.log

重要参数说明
参数	默认值	说明
--data_dir	必填	数据根目录（包含所有 BraTS-MET-* 文件夹）
--json_list	必填	划分 JSON 文件路径（如 full_split.json）
--logdir	./runs	保存模型和日志的目录
--max_epochs	200	总训练轮数
--batch_size	2	批大小（24GB 显存建议 2，若显存不足可改为 1）
--val_every	10	每多少个 epoch 验证一次
--out_channels	5	输出通道数（固定为 5，对应背景 + 4 类标签）
--gpu	0	GPU 编号
