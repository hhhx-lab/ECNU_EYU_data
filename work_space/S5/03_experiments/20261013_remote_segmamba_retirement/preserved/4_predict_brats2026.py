"""
Prediction script for BraTS 2026 Brain Metastases Challenge
"""
import argparse
import numpy as np
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
set_determinism(123)
import os
import glob
import yaml
from light_training.prediction import Predictor
import SimpleITK as sitk


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SegMamba BraTS 2026 Prediction')
    parser.add_argument('-c', '--config', type=str, default='',
                        help='Path to YAML config file')
    return parser.parse_known_args()[0]


def load_config(config_path):
    """从 YAML 文件加载配置"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from: {config_path}")
        return config or {}
    return {}


def get_config_value(config, *keys, default=None):
    """从嵌套配置中获取值，支持多级 key"""
    value = config
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return default
        if value is None:
            return default
    return value


# 加载配置
args = parse_args()
config = load_config(args.config) if args.config else {}

# 从 config 文件名提取实验标识
config_name = ""
if args.config:
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    print(f"Config name: {config_name}")

data_dir = get_config_value(config, 'data_dir', default="./data/fullres/train")
env = "pytorch"
max_epoch = 10
batch_size = 1
num_gpus = 1
device = get_config_value(config, 'device', default="cuda:0")
patch_size = get_config_value(config, 'roi_size', default=[128, 128, 128])
# 如果 config 中未指定 logdir，则自动从文件名生成
if 'logdir' in config:
    logdir = config['logdir']
else:
    logdir = f"./results/{config_name}" if config_name else "./results/default"

# 预测结果也保存到统一目录 (放在 if/else 外面!)
pred_save_path = os.path.join(logdir, "predictions")
os.makedirs(pred_save_path, exist_ok=True)


class BraTS2026Predictor(Trainer):

    LABEL_NETC = 1
    LABEL_SNFH = 2
    LABEL_ET   = 3
    LABEL_RC   = 4

    def __init__(self, env_type, max_epochs, batch_size, device="cpu",
                 val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750,
                 training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device,
                         val_every, num_gpus, logdir,
                         master_ip, master_port, training_script)
        self.patch_size = patch_size
        self.augmentation = False

        # ✅ 模型只加载一次
        self.model, self.predictor, self.save_path = self._load_model()

    def _load_model(self):
        from model_segmamba.segmamba import SegMamba

        model = SegMamba(
            in_chans=4,
            out_chans=5,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384]
        )

        # ✅ 自动找到best_model文件（文件名带有dice数值）
        model_dir = f"{logdir}/model/"
        suffix_pattern = f"best_model_{config_name}_*.pt" if config_name else "best_model_*.pt"
        candidates = glob.glob(os.path.join(model_dir, suffix_pattern))

        # 如果没找到带suffix的，尝试不带suffix的（兼容旧模型）
        if not candidates:
            candidates = glob.glob(os.path.join(model_dir, "best_model_*.pt"))

        if not candidates:
            raise FileNotFoundError(f"No best_model found in {model_dir}")
        model_path = sorted(candidates)[-1]  # 取最新的
        print(f"Loading model from: {model_path}")

        new_sd = self.filte_state_dict(torch.load(model_path, map_location="cpu"))
        model.load_state_dict(new_sd)
        model.eval()

        window_infer = SlidingWindowInferer(
            roi_size=patch_size,
            sw_batch_size=2,
            overlap=0.5,
            progress=True,
            mode="gaussian"
        )
        predictor = Predictor(window_infer=window_infer, mirror_axes=[0, 1, 2])

        # 保存到统一目录
        save_path = pred_save_path

        return model, predictor, save_path

    def convert_labels(self, labels):
        """Per-label multi-hot encoding, batch维度 (B,1,D,H,W) -> (B,4,D,H,W)"""
        netc = (labels == self.LABEL_NETC)
        snfh = (labels == self.LABEL_SNFH)
        et   = (labels == self.LABEL_ET)
        rc   = (labels == self.LABEL_RC)
        return torch.cat([netc, snfh, et, rc], dim=1).float()

    def convert_labels_dim0(self, labels):
        """Per-label multi-hot encoding, 单样本 (1,D,H,W) -> (4,D,H,W)"""
        netc = (labels == self.LABEL_NETC)
        snfh = (labels == self.LABEL_SNFH)
        et   = (labels == self.LABEL_ET)
        rc   = (labels == self.LABEL_RC)
        return torch.cat([netc, snfh, et, rc], dim=0).float()

    def convert_labels_dim0_back(self, labels):
        """
        Multi-hot (4,D,H,W) -> label index (1,D,H,W)
        优先级: SNFH最低，RC最高（后写覆盖前写）
        """
        result = torch.zeros_like(labels[0])
        result[labels[1] > 0.5] = self.LABEL_SNFH  # 最低优先级
        result[labels[0] > 0.5] = self.LABEL_NETC
        result[labels[2] > 0.5] = self.LABEL_ET
        result[labels[3] > 0.5] = self.LABEL_RC    # 最高优先级
        return result.unsqueeze(0)

    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]
        properties = batch["properties"]
        label = self.convert_labels(label)
        return image, label, properties

    def cal_metric(self, gt, pred):
        """处理边界情况的Dice计算"""
        if pred.sum() > 0 and gt.sum() > 0:
            return dice(pred, gt)
        elif gt.sum() == 0 and pred.sum() == 0:
            return 1.0
        else:
            return 0.0

    def validation_step(self, batch):
        image, label, properties = self.get_input(batch)

        # 推理: model_output 和 label 此刻都在 1mm 重采样空间
        model_output = self.predictor.maybe_mirror_and_predict(
            image, self.model, device=device
        )

        # ---- 算指标 (在 1mm 空间, pred 与 label shape 一致) ----
        model_output_metric = model_output.argmax(dim=0)[None]   # (1, D, H, W)
        model_output_metric = self.convert_labels_dim0(model_output_metric)  # (4, D, H, W)

        label_np = label[0]  # (4, D, H, W)
        dices = []
        for i in range(4):
            d = self.cal_metric(
                label_np[i].cpu().numpy(),
                model_output_metric[i].cpu().numpy()
            )
            dices.append(d)

        print(
            f"NETC: {dices[0]:.4f} | SNFH: {dices[1]:.4f} | "
            f"ET: {dices[2]:.4f} | RC: {dices[3]:.4f}"
        )

        # ---- 保存: 先 resample 回原始间距 → 还原裁剪区域 ----
        model_output = self.predictor.predict_raw_probability(
            model_output, properties=properties
        )  # resample 到 shape_after_cropping_before_resample
        model_output_save = model_output.argmax(dim=0).cpu().numpy().astype(np.uint8)
        del model_output

        # 还原裁剪区域
        model_output_save = self.predictor.predict_noncrop_probability(
            model_output_save, properties
        )  # 扩回 shape_before_cropping

        # 安全解包 collate 包裹的值
        raw_spacing = list(self._unwrap_collate(properties['spacing']))
        case_name = str(self._unwrap_collate(properties['name']))

        # 方向矩阵: 优先从 properties 读, 兜底从原始 seg.nii.gz 读
        direction = properties.get('direction', None)
        origin = properties.get('origin', None)
        if direction is not None:
            direction = self._unwrap_collate(direction)
        if origin is not None:
            origin = self._unwrap_collate(origin)

        if direction is None or origin is None:
            raw_seg = f"/public/home/zqchen/data/{case_name}/seg.nii.gz"
            if os.path.exists(raw_seg):
                ref = sitk.ReadImage(raw_seg)
                if direction is None:
                    direction = ref.GetDirection()
                if origin is None:
                    origin = ref.GetOrigin()
            else:
                print(f"[WARN] 找不到原始标注文件用于方向/原点: {raw_seg}", flush=True)

        self.predictor.save_to_nii(
            model_output_save,
            raw_spacing=raw_spacing,
            direction=direction,
            origin=origin,
            case_name=case_name,
            save_dir=self.save_path
        )

        return dices

    @staticmethod
    def _unwrap_collate(val):
        """DataLoader collate 可能把标量值包在 list 里 (batch 维度), 安全解包."""
        if isinstance(val, (list, tuple)) and len(val) == 1:
            return val[0]
        return val

    def filte_state_dict(self, sd):
        if "module" in sd:
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k
            new_sd[new_k] = v
        del sd
        return new_sd


if __name__ == "__main__":
    predictor = BraTS2026Predictor(
        env_type=env,
        max_epochs=max_epoch,
        batch_size=batch_size,
        device=device,
        logdir="",
        val_every=1,
        num_gpus=num_gpus,
        master_port=17751,
        training_script=__file__
    )

    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(data_dir)
    predictor.validation_single_gpu(test_ds)