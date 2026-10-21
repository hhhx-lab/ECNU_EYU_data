"""
损失函数模块
支持多种损失函数及其组合，针对 BraTS 2026 医学分割任务优化。

损失函数选择建议:
- CE + Dice:    标准做法，稳定提升 DSC，适合大多数情况
- CE + Focal:   类别极度不平衡时（如 RC 体积极小），Focal 处理小目标效果好
- Dice + Focal: 小病灶多的情况下效果更好
- Focal:        单独使用效果不如组合稳定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss
    适用于类别不平衡的分割任务，直接优化 Dice 系数。
    """

    def __init__(self, smooth=1e-5, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, ...) logits
            target: (B, ...) long tensor (class indices)
        Returns:
            loss: scalar
        """
        num_classes = pred.shape[1]
        # 安全检查：确保标签在合法范围
        target = target.clamp(0, num_classes - 1)

        pred_soft = F.softmax(pred, dim=1)

        if target.dim() > 1:
            target = target.squeeze(1)
        target_onehot = F.one_hot(target.long(), num_classes).permute(0, -1, *range(1, target.dim())).float()

        # Flatten
        pred_flat = pred_soft.view(pred.shape[0], num_classes, -1)
        target_flat = target_onehot.view(target_onehot.shape[0], num_classes, -1)

        intersection = (pred_flat * target_flat).sum(dim=-1)
        union = pred_flat.sum(dim=-1) + target_flat.sum(dim=-1)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice

        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss
    专门处理类别极度不平衡的问题，降低简单样本的权重，
    关注难样本和小目标。
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', num_classes=5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, ...) logits
            target: (B, ...) long tensor
        Returns:
            loss: scalar
        """
        target = target.clamp(0, self.num_classes - 1)
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    带权重的 Cross Entropy Loss
    """

    def __init__(self, weight=None, reduction='mean', num_classes=5):
        super().__init__()
        if weight is not None:
            self.register_buffer('weight', torch.tensor(weight, dtype=torch.float))
        else:
            self.weight = None
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, pred, target):
        target = target.clamp(0, self.num_classes - 1)
        return F.cross_entropy(pred, target, weight=self.weight, reduction=self.reduction)


class CombinedLoss(nn.Module):
    """
    组合损失函数，支持 CE/Dice/Focal 的任意组合和权重调整。

    Args:
        ce_weight: CE 损失权重
        dice_weight: Dice 损失权重
        focal_weight: Focal 损失权重
        focal_alpha: Focal loss alpha 参数
        focal_gamma: Focal loss gamma 参数
        ce_class_weights: CE 类别权重列表 [背景, NETC, SNFH, ET, RC]
        num_classes: 类别数 (默认 5: BG + NETC + SNFH + ET + RC)
    """

    def __init__(
        self,
        ce_weight=1.0,
        dice_weight=1.0,
        focal_weight=0.0,
        focal_alpha=1.0,
        focal_gamma=2.0,
        ce_class_weights=None,
        num_classes=5,
    ):
        super().__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.num_classes = num_classes

        # 解析类别权重
        if ce_class_weights is not None:
            weights = ce_class_weights
        else:
            weights = None

        # CE Loss — 权重在 __init__ 时创建好 (避免每个 forward 新建 tensor)
        if weights is not None:
            assert len(weights) == num_classes, \
                f"ce_class_weights 长度 ({len(weights)}) != num_classes ({num_classes})"
            self.ce_fn = WeightedCrossEntropyLoss(
                weight=weights, reduction='mean', num_classes=num_classes
            )
        else:
            self.ce_fn = WeightedCrossEntropyLoss(
                weight=None, reduction='mean', num_classes=num_classes
            )

        self.dice_fn = DiceLoss()
        self.focal_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, num_classes=num_classes)

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, D, H, W) logits
            target: (B, D, H, W) long tensor, 值域 [0, C-1]
        Returns:
            total_loss: scalar
            loss_dict: dict with individual loss values
        """
        # 安全检查：防止标签越界导致 CUDA device-side assert
        target = target.clamp(0, self.num_classes - 1)

        total_loss = 0.0
        loss_dict = {}

        if self.ce_weight > 0:
            ce_loss = self.ce_fn(pred, target)
            total_loss += self.ce_weight * ce_loss
            loss_dict['ce'] = ce_loss.item()

        if self.dice_weight > 0:
            dice_loss = self.dice_fn(pred, target)
            total_loss += self.dice_weight * dice_loss
            loss_dict['dice'] = dice_loss.item()

        if self.focal_weight > 0:
            focal_loss = self.focal_fn(pred, target)
            total_loss += self.focal_weight * focal_loss
            loss_dict['focal'] = focal_loss.item()

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict


def get_loss_function(config):
    """
    根据配置创建损失函数

    Args:
        config: dict, 包含损失函数配置

    Returns:
        loss_fn: CombinedLoss 实例
        loss_info: dict, 包含损失权重信息
    """
    loss_type = config.get('loss_type', 'ce_dice')

    loss_cfg = config.get('loss', {})

    ce_weight = loss_cfg.get('ce_weight', 1.0)
    dice_weight = loss_cfg.get('dice_weight', 1.0)
    focal_weight = loss_cfg.get('focal_weight', 0.0)
    num_classes = config.get('out_chans', 5)

    # 根据 loss_type 自动设置权重
    if loss_type == 'ce_dice':
        focal_weight = 0.0
    elif loss_type == 'ce_focal':
        dice_weight = 0.0
    elif loss_type == 'dice_focal':
        ce_weight = 0.0
    elif loss_type == 'focal':
        ce_weight = 0.0
        dice_weight = 0.0
    elif loss_type == 'ce':
        dice_weight = 0.0
        focal_weight = 0.0
    elif loss_type == 'dice':
        ce_weight = 0.0
        focal_weight = 0.0

    focal_alpha = loss_cfg.get('focal_alpha', 1.0)
    focal_gamma = loss_cfg.get('focal_gamma', 2.0)
    ce_class_weights = loss_cfg.get('ce_class_weights', None)

    loss_fn = CombinedLoss(
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        focal_weight=focal_weight,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        ce_class_weights=ce_class_weights,
        num_classes=num_classes,
    )

    loss_info = {
        'ce_weight': ce_weight,
        'dice_weight': dice_weight,
        'focal_weight': focal_weight,
        'loss_type': loss_type,
    }

    return loss_fn, loss_info
