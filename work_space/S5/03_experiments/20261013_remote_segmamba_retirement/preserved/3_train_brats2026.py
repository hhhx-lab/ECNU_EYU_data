import argparse
import glob
import re
import numpy as np
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.files_helper import save_new_model_and_delete_last
from light_training.utils.lr_scheduler import (get_constant_schedule_with_warmup,
                                                get_cosine_schedule_with_warmup,
                                                get_polynomial_decay_schedule_with_warmup)
from torch.utils.tensorboard import SummaryWriter
import os
import yaml
import matplotlib.pyplot as plt
from loss_functions import get_loss_function, CombinedLoss
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SegMamba BraTS 2026 Training')
    parser.add_argument('-c', '--config', type=str, default='',
                        help='Path to YAML config file')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from the latest checkpoint (auto-find in model dir)')
    parser.add_argument('--resume_from', type=str, default='',
                        help='Explicit path to checkpoint .pt file for resuming')
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

# ---------------------------------------------------------------------------
# 配置参数 (可从 YAML 文件覆盖)
# ---------------------------------------------------------------------------
data_dir = get_config_value(config, 'data_dir', default="./data/fullres/train")
# 如果 config 中未指定 logdir，则自动从文件名生成
if 'logdir' in config:
    logdir = config['logdir']
else:
    logdir = f"./results/{config_name}" if config_name else "./results/default"
augmentation = get_config_value(config, 'augmentation', default=True)
max_epoch = get_config_value(config, 'max_epoch', default=10)
batch_size = get_config_value(config, 'batch_size', default=2)
val_every = get_config_value(config, 'val_every', default=2)
num_gpus = get_config_value(config, 'num_gpus', default=1)
device = get_config_value(config, 'device', default="cuda:0")
seed = get_config_value(config, 'seed', default=123)
roi_size = get_config_value(config, 'roi_size', default=[128, 128, 128])

# 早停配置
early_stopping_enabled = get_config_value(config, 'early_stopping', 'enabled', default=False)
early_stopping_patience = get_config_value(config, 'early_stopping', 'patience', default=10)
early_stopping_min_delta = get_config_value(config, 'early_stopping', 'min_delta', default=0.001)

# 小病灶检测阈值
SMALL_LESION_THRESHOLD_MM3 = get_config_value(
    config, 'evaluation', 'small_lesion_threshold_mm3', default=27.0
)

# 模型配置
in_chans = get_config_value(config, 'model', 'in_chans', default=4)
out_chans = get_config_value(config, 'model', 'out_chans', default=5)
depths = get_config_value(config, 'model', 'depths', default=[2, 2, 2, 2])
feat_size = get_config_value(config, 'model', 'feat_size', default=[48, 96, 192, 384])

# 优化器配置
optimizer_type = get_config_value(config, 'optimizer', 'type', default="SGD")
lr = get_config_value(config, 'optimizer', 'lr', default=1e-2)
weight_decay = get_config_value(config, 'optimizer', 'weight_decay', default=3e-5)
momentum = get_config_value(config, 'optimizer', 'momentum', default=0.99)
nesterov = get_config_value(config, 'optimizer', 'nesterov', default=True)

# 学习率调度器
scheduler_type = get_config_value(config, 'scheduler', 'type', default="poly")

# 推理配置
inference_roi_size = get_config_value(config, 'inference', 'roi_size', default=roi_size)
sw_batch_size = get_config_value(config, 'inference', 'sw_batch_size', default=1)
inference_overlap = get_config_value(config, 'inference', 'overlap', default=0.5)

# 损失函数配置
loss_type = get_config_value(config, 'loss_type', default="ce_dice")
loss_config = {
    'loss_type': loss_type,
    'loss': get_config_value(config, 'loss', default={})
}

# 设置随机种子
set_determinism(seed)

# 创建必要的目录
model_save_path = os.path.join(logdir, "model")
viz_save_path = os.path.join(logdir, "visualizations")
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(viz_save_path, exist_ok=True)


class BraTS2026Trainer(Trainer):
    """
    Trainer for BraTS 2026 Brain Metastases Challenge

    Segmentation评估: per-label Dice (NETC, SNFH, ET, RC)
    Detection评估: 病灶级别 F1，病灶 = ET ∪ NETC ∪ RC
    """

    LABEL_NETC = 1
    LABEL_SNFH = 2
    LABEL_ET   = 3
    LABEL_RC   = 4

    def __init__(self, env_type, max_epochs, batch_size, device="cpu",
                 val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750,
                 training_script="train.py", exp_suffix=""):

        super().__init__(env_type, max_epochs, batch_size, device,
                         val_every, num_gpus, logdir,
                         master_ip, master_port, training_script)

        self.window_infer = SlidingWindowInferer(
            roi_size=inference_roi_size, sw_batch_size=sw_batch_size, overlap=inference_overlap
        )
        self.augmentation = augmentation
        self.exp_suffix = exp_suffix

        from model_segmamba.segmamba import SegMamba

        self.model = SegMamba(
            in_chans=in_chans,
            out_chans=out_chans,
            depths=depths,
            feat_size=feat_size
        )

        self.patch_size = inference_roi_size
        self.best_mean_dice = 0.0

        # 创建损失函数
        self.loss_fn, loss_info = get_loss_function(config)
        print(f"Loss function: {loss_type}")
        print(f"  CE weight: {loss_info.get('ce_weight', 0):.1f}")
        print(f"  Dice weight: {loss_info.get('dice_weight', 0):.1f}")
        print(f"  Focal weight: {loss_info.get('focal_weight', 0):.1f}")

        self.train_loss_history = []
        self.train_ce_history = []
        self.train_dice_history = []
        self.train_focal_history = []
        self.val_dice_history = {
            'netc': [], 'snfh': [], 'et': [], 'rc': [], 'mean': []
        }
        self.epoch_history = []

        # 根据配置选择优化器
        if optimizer_type.upper() == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr, weight_decay=weight_decay,
                momentum=momentum, nesterov=nesterov
            )
        elif optimizer_type.upper() == "ADAM":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr, weight_decay=weight_decay
            )
        elif optimizer_type.upper() == "ADAMW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        self.scheduler_type = scheduler_type

        # 早停
        self.early_stopping = early_stopping_enabled
        self.es_patience = early_stopping_patience
        self.es_min_delta = early_stopping_min_delta
        self.es_best_metric = 0.0
        self.es_counter = 0
        self.early_stop = False

    # ------------------------------------------------------------------
    # 标签转换
    # ------------------------------------------------------------------

    def convert_labels(self, labels):
        """
        Per-label multi-hot encoding for segmentation evaluation.

        输入: (B, 1, D, H, W) long tensor，值为 0~4
        输出: (B, 4, D, H, W) float tensor
             通道顺序: [NETC, SNFH, ET, RC]
        """
        netc = (labels == self.LABEL_NETC)   # Label 1 only
        snfh = (labels == self.LABEL_SNFH)   # Label 2 only
        et   = (labels == self.LABEL_ET)     # Label 3 only
        rc   = (labels == self.LABEL_RC)     # Label 4 only

        return torch.cat([netc, snfh, et, rc], dim=1).float()

    def convert_labels_for_detection(self, labels):
        """
        病灶mask，用于detection评估。
        病灶 = ET ∪ NETC ∪ RC（不含SNFH水肿区）

        输入: (B, 1, D, H, W)
        输出: (B, 1, D, H, W) float
        """
        lesion = (
            (labels == self.LABEL_ET)   |
            (labels == self.LABEL_NETC) |
            (labels == self.LABEL_RC)
        )
        return lesion.float()

    # ------------------------------------------------------------------
    # 训练 (override 父类以支持早停)
    # ------------------------------------------------------------------

    def train(self, train_dataset, val_dataset=None):
        """Override 父类 train()，加入 early stopping 支持。"""
        print(f"augmentation: {self.augmentation}")
        assert self.patch_size is not None, "please define the patch_size"

        set_determinism(42 + self.local_rank)
        if self.model is not None:
            para = sum([np.prod(list(p.size())) for p in self.model.parameters()])
            if self.local_rank == 0:
                print(f"model parameters is {para / 1000 / 1000}M ")

        self.global_step = 0
        if self.env_type == "pytorch":
            if self.model is not None:
                self.model.to(self.device)
            os.makedirs(self.logdir, exist_ok=True)
            self.writer = SummaryWriter(self.logdir)
        elif self.ddp:
            if self.local_rank == 0:
                os.makedirs(self.logdir, exist_ok=True)
                self.writer = SummaryWriter(self.logdir)
            else:
                self.writer = None
            if self.model is not None:
                self.model.cuda(self.local_rank)
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=True)
        else:
            print("not support env_type")
            exit(0)

        self.train_loader, self.val_loader = self.get_multi_processor_loader(
            train_dataset, val_dataset)
        self.max_steps = self.max_epochs * len(self.train_loader)
        print(f"step number is {self.max_steps}")

        # --- scheduler setup (同父类) ---
        if self.scheduler_type == "cosine_with_warmup":
            if self.warmup == 0.0:
                self.warmup = 0.1
            warmup_steps = self.max_steps * self.warmup
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps,
                num_training_steps=self.max_steps)
        elif self.scheduler_type == "constant_with_warmup":
            if self.warmup == 0.0:
                self.warmup = 0.1
            warmup_steps = self.max_steps * self.warmup
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps)
        elif self.scheduler_type == "poly_with_warmup":
            if self.warmup == 0.0:
                self.warmup = 0.1
            warmup_steps = self.max_steps * self.warmup
            self.scheduler = get_polynomial_decay_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps,
                num_training_steps=self.max_steps)
        elif self.scheduler_type == "poly":
            from light_training.utils.lr_scheduler import PolyLRScheduler
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.scheduler = PolyLRScheduler(self.optimizer, initial_lr=lr,
                                              max_steps=self.max_steps)

        if self.early_stopping:
            print(f"[EarlyStopping] Enabled — patience={self.es_patience}, "
                  f"min_delta={self.es_min_delta}")

        for epoch in range(0, self.max_epochs):
            self.epoch = epoch
            if self.ddp:
                torch.distributed.barrier()
            self.train_epoch(epoch)
            if (self.epoch + 1) % self.val_every == 0:
                self.validate()
                if self.early_stop:
                    print(f"[EarlyStopping] Training stopped at epoch {self.epoch}")
                    break
            if self.model is not None:
                self.model.train()

    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]
        label = label[:, 0].long()   # (B, D, H, W)
        return image, label

    def training_step(self, batch):
        image, label = self.get_input(batch)
        pred = self.model(image)

        loss_output = self.loss_fn(pred, label)

        # CombinedLoss 返回 (total_loss, loss_dict)
        if isinstance(loss_output, tuple):
            total_loss, loss_dict = loss_output
            self.log("training_loss", total_loss, step=self.global_step)
            self.log("training_ce_loss", loss_dict.get('ce', 0), step=self.global_step)
            self.log("training_dice_loss", loss_dict.get('dice', 0), step=self.global_step)
            self.log("training_focal_loss", loss_dict.get('focal', 0), step=self.global_step)

            self.train_loss_history.append(total_loss.item())
            self.train_ce_history.append(loss_dict.get('ce', 0))
            self.train_dice_history.append(loss_dict.get('dice', 0))
            self.train_focal_history.append(loss_dict.get('focal', 0))
        else:
            total_loss = loss_output
            self.log("training_loss", total_loss, step=self.global_step)
            self.train_loss_history.append(total_loss.item())

        return total_loss

    # ------------------------------------------------------------------
    # 验证
    # ------------------------------------------------------------------

    def cal_metric(self, gt, pred):
        """
        计算单个类别的 Dice score。
        返回: float
        """
        if pred.sum() > 0 and gt.sum() > 0:
            return dice(pred, gt)
        elif gt.sum() == 0 and pred.sum() == 0:
            return 1.0
        else:
            return 0.0

    def validation_step(self, batch):
        image, label = self.get_input(batch)

        output = self.window_infer(image, self.model)   # (B, 5, D, H, W)
        output = output.argmax(dim=1, keepdim=True)     # (B, 1, D, H, W)

        # per-label multi-hot
        output_onehot = self.convert_labels(output)     # (B, 4, D, H, W)
        label_onehot  = self.convert_labels(label[:, None])

        output_np = output_onehot.cpu().numpy()
        label_np  = label_onehot.cpu().numpy()

        # Segmentation dice: [NETC, SNFH, ET, RC]
        seg_dices = []
        for i in range(4):
            d = self.cal_metric(label_np[:, i], output_np[:, i])
            seg_dices.append(d)

        return seg_dices

    def validation_end(self, val_outputs):
        dices = np.array(val_outputs)   # (N_cases, 4)

        netc_dice = dices[:, 0].mean()
        snfh_dice = dices[:, 1].mean()
        et_dice   = dices[:, 2].mean()
        rc_dice   = dices[:, 3].mean()
        mean_dice = dices.mean()

        print(
            f"NETC: {netc_dice:.4f} | "
            f"SNFH: {snfh_dice:.4f} | "
            f"ET: {et_dice:.4f} | "
            f"RC: {rc_dice:.4f} | "
            f"Mean: {mean_dice:.4f}"
        )

        self.log("netc_dice", netc_dice, step=self.epoch)
        self.log("snfh_dice", snfh_dice, step=self.epoch)
        self.log("et_dice",   et_dice,   step=self.epoch)
        self.log("rc_dice",   rc_dice,   step=self.epoch)
        self.log("mean_dice", mean_dice, step=self.epoch)

        # Track history
        self.epoch_history.append(self.epoch)
        self.val_dice_history['netc'].append(netc_dice)
        self.val_dice_history['snfh'].append(snfh_dice)
        self.val_dice_history['et'].append(et_dice)
        self.val_dice_history['rc'].append(rc_dice)
        self.val_dice_history['mean'].append(mean_dice)

        # Save visualizations every validation
        self._save_visualizations()

        # 保存最优模型
        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            suffix = f"_{self.exp_suffix}" if self.exp_suffix else ""
            save_new_model_and_delete_last(
                self.model,
                os.path.join(model_save_path, f"best_model{suffix}_{mean_dice:.4f}.pt"),
                delete_symbol="best_model"
            )

        suffix = f"_{self.exp_suffix}" if self.exp_suffix else ""
        save_new_model_and_delete_last(
            self.model,
            os.path.join(model_save_path, f"final_model{suffix}_{mean_dice:.4f}.pt"),
            delete_symbol="final_model"
        )

        # 每100epoch存一个checkpoint
        if (self.epoch + 1) % 100 == 0:
            torch.save(
                self.model.state_dict(),
                os.path.join(model_save_path, f"ckpt_ep{self.epoch}{suffix}_{mean_dice:.4f}.pt")
            )

        print(f"Best mean dice so far: {self.best_mean_dice:.4f}")

        # 早停检查
        if self.early_stopping:
            if mean_dice > self.es_best_metric + self.es_min_delta:
                self.es_best_metric = mean_dice
                self.es_counter = 0
            else:
                self.es_counter += 1
                print(f"[EarlyStopping] No improvement for {self.es_counter}/{self.es_patience} validations "
                      f"(best: {self.es_best_metric:.4f}, current: {mean_dice:.4f})")
                if self.es_counter >= self.es_patience:
                    self.early_stop = True
                    print(f"[EarlyStopping] Triggered at epoch {self.epoch}!")

    def _save_visualizations(self):
        """Save loss curve and validation metrics visualizations."""
        epochs = self.epoch_history
        if len(epochs) == 0:
            return

        # 添加实验标识后缀
        suffix = f"_{self.exp_suffix}" if self.exp_suffix else ""
        colors = {
            'netc': '#E63946', 'snfh': '#457B9D',
            'et': '#2A9D8F', 'rc': '#E9C46A', 'mean': '#1D3557'
        }

        # Subsample for smoother curve display
        step_interval = max(1, len(self.train_loss_history) // 500)
        display_steps = list(range(0, len(self.train_loss_history), step_interval))

        # 1. Loss Curve
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        display_loss = []
        for i in display_steps:
            display_loss.append(np.mean(self.train_loss_history[i:i+step_interval]))

        axes[0].plot(display_steps, display_loss, color='#E63946', linewidth=1.5)
        axes[0].set_xlabel('Training Step', fontsize=11)
        axes[0].set_ylabel('Loss', fontsize=11)
        axes[0].set_title('Training Loss Curve', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_facecolor('#F8F9FA')

        # 2. Validation Dice Scores
        for label_name, color in colors.items():
            linestyle = '-' if label_name == 'mean' else '--'
            linewidth = 2.0 if label_name == 'mean' else 1.2
            axes[1].plot(epochs, self.val_dice_history[label_name],
                         label=label_name.upper(), color=color,
                         linestyle=linestyle, linewidth=linewidth)

        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Dice Score', fontsize=11)
        axes[1].set_title('Validation Dice Scores', fontsize=13, fontweight='bold')
        axes[1].legend(loc='lower right', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.05])
        axes[1].set_facecolor('#F8F9FA')

        plt.tight_layout()
        plt.savefig(os.path.join(viz_save_path, f'training_curves{suffix}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # 3. Per-class Dice Bar Chart (latest epoch)
        fig, ax = plt.subplots(figsize=(8, 5))
        class_names = ['NETC', 'SNFH', 'ET', 'RC']
        latest_dices = [self.val_dice_history[k][-1] for k in ['netc', 'snfh', 'et', 'rc']]
        bar_colors = [colors['netc'], colors['snfh'], colors['et'], colors['rc']]

        bars = ax.bar(class_names, latest_dices, color=bar_colors, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, latest_dices):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Dice Score', fontsize=11)
        ax.set_title(f'Validation Dice per Class (Epoch {epochs[-1]})',
                     fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1.15])
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_facecolor('#F8F9FA')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_save_path, f'dice_per_class{suffix}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # 4. Loss Components Breakdown (if available)
        if len(self.train_ce_history) > 0 and any(x > 0 for x in self.train_ce_history):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            # CE Loss
            if any(x > 0 for x in self.train_ce_history):
                display_ce = []
                for i in display_steps:
                    display_ce.append(np.mean(self.train_ce_history[i:i+step_interval]))
                axes[0].plot(display_steps, display_ce, color='#E63946', linewidth=1.5)
                axes[0].set_xlabel('Training Step', fontsize=10)
                axes[0].set_ylabel('CE Loss', fontsize=10)
                axes[0].set_title('Cross Entropy Loss', fontsize=12, fontweight='bold')
                axes[0].grid(True, alpha=0.3)
                axes[0].set_facecolor('#F8F9FA')

            # Dice Loss
            if any(x > 0 for x in self.train_dice_history):
                display_dice = []
                for i in display_steps:
                    display_dice.append(np.mean(self.train_dice_history[i:i+step_interval]))
                axes[1].plot(display_steps, display_dice, color='#457B9D', linewidth=1.5)
                axes[1].set_xlabel('Training Step', fontsize=10)
                axes[1].set_ylabel('Dice Loss', fontsize=10)
                axes[1].set_title('Dice Loss', fontsize=12, fontweight='bold')
                axes[1].grid(True, alpha=0.3)
                axes[1].set_facecolor('#F8F9FA')

            # Focal Loss
            if any(x > 0 for x in self.train_focal_history):
                display_focal = []
                for i in display_steps:
                    display_focal.append(np.mean(self.train_focal_history[i:i+step_interval]))
                axes[2].plot(display_steps, display_focal, color='#2A9D8F', linewidth=1.5)
                axes[2].set_xlabel('Training Step', fontsize=10)
                axes[2].set_ylabel('Focal Loss', fontsize=10)
                axes[2].set_title('Focal Loss', fontsize=12, fontweight='bold')
                axes[2].grid(True, alpha=0.3)
                axes[2].set_facecolor('#F8F9FA')

            plt.suptitle(f'Loss Components Breakdown (Loss Type: {loss_type})', fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_save_path, f'loss_components{suffix}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

        print(f"Visualizations saved to {viz_save_path}")


if __name__ == "__main__":
    trainer = BraTS2026Trainer(
        env_type="pytorch",
        max_epochs=max_epoch,
        batch_size=batch_size,
        device=device,
        logdir=logdir,
        val_every=val_every,
        num_gpus=num_gpus,
        master_port=17759,
        training_script=__file__,
        exp_suffix=config_name
    )

    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(data_dir)

    # ------------------------------------------------------------------
    # 断点续训: 加载已有模型权重
    # ------------------------------------------------------------------
    if args.resume or args.resume_from:
        if args.resume_from:
            ckpt_path = args.resume_from
            if not os.path.exists(ckpt_path):
                print(f"ERROR: Checkpoint not found: {ckpt_path}")
                exit(1)
        else:
            # 自动查找最新的 checkpoint (优先 final_model, 其次 best_model)
            suffix = f"_{config_name}" if config_name else ""
            candidates = []
            for prefix in [f"final_model{suffix}", f"best_model{suffix}"]:
                candidates = sorted(glob.glob(
                    os.path.join(model_save_path, f"{prefix}_*.pt")
                ))
                if candidates:
                    break
            if not candidates:
                print(f"ERROR: No checkpoint found in {model_save_path}/")
                print("  Use --resume_from <path> to specify a checkpoint file.")
                exit(1)
            ckpt_path = candidates[-1]  # 取 dice 分数最高的（文件名排序）

        print(f"[Resume] Loading checkpoint: {ckpt_path}")
        trainer.load_state_dict(ckpt_path)

        # 尝试从文件名中恢复 best_mean_dice
        match = re.search(r'_(\d+\.\d+)\.pt$', ckpt_path)
        if match:
            restored_dice = float(match.group(1))
            trainer.best_mean_dice = restored_dice
            print(f"[Resume] Restored best_mean_dice = {restored_dice:.4f}")

        print(f"[Resume] Model loaded. Training will run for {max_epoch} epochs.")
        print(f"[Resume] Note: optimizer/scheduler are freshly initialized.")

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)