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
    logdir = f"./logs/segmamba_{config_name}" if config_name else "./logs/segmamba_brats2026"


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

        # 保存路径包含实验标识
        suffix_str = f"_{config_name}" if config_name else "_default"
        save_path = f"./prediction_results/segmamba_brats2026{suffix_str}"
        os.makedirs(save_path, exist_ok=True)

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

        model_output = self.predictor.maybe_mirror_and_predict(
            image, self.model, device=device
        )
        model_output = self.predictor.predict_raw_probability(
            model_output, properties=properties
        )

        model_output = model_output.argmax(dim=0)[None]
        model_output = self.convert_labels_dim0(model_output)

        label = label[0]
        dices = []
        for i in range(4):
            d = self.cal_metric(
                label[i].cpu().numpy(),
                model_output[i].cpu().numpy()
            )
            dices.append(d)

        print(
            f"NETC: {dices[0]:.4f} | SNFH: {dices[1]:.4f} | "
            f"ET: {dices[2]:.4f} | RC: {dices[3]:.4f}"
        )

        # 保存预测结果 (转换为numpy数组)
        model_output_for_save = self.convert_labels_dim0_back(model_output)
        model_output_for_save = model_output_for_save.cpu().numpy()
        model_output_for_save = self.predictor.predict_noncrop_probability(model_output_for_save, properties)
        model_output_for_save = np.squeeze(model_output_for_save)  # shape从 (240,155,1,1,240) → (240,155,240)
        self.predictor.save_to_nii(
        model_output_for_save,  # ✅ 现在是还原后的完整尺寸
        raw_spacing=[1, 1, 1],
        case_name=properties['name'][0],
        save_dir=self.save_path
        )

        return dices

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