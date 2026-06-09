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
        pred_soft = F.softmax(pred, dim=1)

        # Convert target to one-hot
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

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, ...) logits
            target: (B, ...) long tensor
        Returns:
            loss: scalar
        """
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
    根据类别频率自动计算权重，处理类别不平衡。
    """

    def __init__(self, weight=None, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, pred, target):
        weight = self.weight.to(pred.device) if self.weight is not None else None
        return F.cross_entropy(pred, target, weight=weight, reduction=self.reduction)


class CombinedLoss(nn.Module):
    """
    组合损失函数，支持 CE/Dice/Focal 的任意组合和权重调整。

    Args:
        losses: dict, e.g. {'ce': 1.0, 'dice': 1.0, 'focal': 0.5}
        ce_weight: CE 损失权重（背景权重降低，因为肿瘤区域更重要）
        dice_weight: Dice 损失权重
        focal_weight: Focal 损失权重
        focal_alpha: Focal loss alpha 参数
        focal_gamma: Focal loss gamma 参数
        ce_class_weights: CE 类别权重列表 [背景, NETC, SNFH, ET, RC]
    """

    def __init__(
        self,
        ce_weight=1.0,
        dice_weight=1.0,
        focal_weight=0.0,
        focal_alpha=1.0,
        focal_gamma=2.0,
        ce_class_weights=None,
        use_background_weight=False
    ):
        super().__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        # CE Loss - 背景权重降低（如果启用）
        self.ce_class_weights = None  # 将在 forward 中动态处理
        if use_background_weight and ce_class_weights is None:
            self.ce_class_weights = [0.1, 1.0, 0.5, 1.5, 2.0]  # [BG, NETC, SNFH, ET, RC]
        elif ce_class_weights is not None:
            self.ce_class_weights = ce_class_weights

        # CE Loss 选择
        if self.ce_class_weights is not None:
            self.ce_fn = WeightedCrossEntropyLoss(reduction='mean')
        else:
            self.ce_fn = nn.CrossEntropyLoss(reduction='mean')

        self.dice_fn = DiceLoss()
        self.focal_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, D, H, W) logits
            target: (B, D, H, W) long tensor
        Returns:
            total_loss: scalar
            loss_dict: dict with individual loss values
        """
        total_loss = 0.0
        loss_dict = {}

        if self.ce_weight > 0:
            if self.ce_class_weights is not None:
                ce_loss = F.cross_entropy(
                    pred, target,
                    weight=torch.tensor(self.ce_class_weights, device=pred.device),
                    reduction='mean'
                )
            else:
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
    loss_type = config.get('loss_type', 'ce_dice')  # 默认 CE + Dice

    loss_cfg = config.get('loss', {})

    ce_weight = loss_cfg.get('ce_weight', 1.0)
    dice_weight = loss_cfg.get('dice_weight', 1.0)
    focal_weight = loss_cfg.get('focal_weight', 0.0)

    # 根据 loss_type 自动设置权重
    if loss_type == 'ce_dice':
        ce_weight = max(ce_weight, 1.0) if ce_weight == 1.0 and dice_weight == 1.0 else ce_weight
        dice_weight = max(dice_weight, 1.0) if ce_weight == 1.0 and dice_weight == 1.0 else dice_weight
        focal_weight = 0.0
    elif loss_type == 'ce_focal':
        ce_weight = max(ce_weight, 1.0) if ce_weight == 1.0 and focal_weight == 0.0 else ce_weight
        dice_weight = 0.0
        focal_weight = max(focal_weight, 1.0) if focal_weight == 0.0 else focal_weight
    elif loss_type == 'dice_focal':
        ce_weight = 0.0
        dice_weight = max(dice_weight, 1.0) if dice_weight == 1.0 else dice_weight
        focal_weight = max(focal_weight, 0.5) if focal_weight == 0.0 else focal_weight
    elif loss_type == 'focal':
        ce_weight = 0.0
        dice_weight = 0.0
        focal_weight = max(focal_weight, 1.0) if focal_weight == 0.0 else focal_weight
    elif loss_type == 'ce':
        ce_weight = max(ce_weight, 1.0)
        dice_weight = 0.0
        focal_weight = 0.0
    elif loss_type == 'dice':
        ce_weight = 0.0
        dice_weight = max(dice_weight, 1.0)
        focal_weight = 0.0

    focal_alpha = loss_cfg.get('focal_alpha', 1.0)
    focal_gamma = loss_cfg.get('focal_gamma', 2.0)

    ce_class_weights = loss_cfg.get('ce_class_weights', None)
    use_background_weight = loss_cfg.get('use_background_weight', False)

    loss_fn = CombinedLoss(
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        focal_weight=focal_weight,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        ce_class_weights=ce_class_weights,
        use_background_weight=use_background_weight
    )

    loss_info = {
        'ce_weight': ce_weight,
        'dice_weight': dice_weight,
        'focal_weight': focal_weight,
        'loss_type': loss_type
    }

    return loss_fn, loss_info
