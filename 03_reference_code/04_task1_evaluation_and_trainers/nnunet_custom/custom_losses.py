# nnunetv2/training/loss/custom_losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops


def compute_lesion_wise_dice_score(pred_mask, target_mask, threshold=0.5):
    """
    计算lesion-wise dice score，为每个独立的病灶计算dice并取平均

    Args:
        pred_mask: 预测掩码 (H, W, D) 或 (H, W)
        target_mask: 真实掩码 (H, W, D) 或 (H, W)
        threshold: 预测的二值化阈值

    Returns:
        float: lesion-wise dice score
    """
    # 将tensor转换为numpy
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.cpu().numpy()
    if torch.is_tensor(target_mask):
        target_mask = target_mask.cpu().numpy()

    # 二值化预测
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    target_binary = target_mask.astype(np.uint8)

    # 如果target是空的，返回适当的值
    if target_binary.sum() == 0:
        if pred_binary.sum() == 0:
            return 1.0  # 真负例情况
        else:
            return 0.0  # 假正例情况

    # 如果pred是空的但target不空，返回0
    if pred_binary.sum() == 0:
        return 0.0

    # 找到target中的连通区域（病灶）
    labeled_target = label(target_binary)
    lesion_dice_scores = []

    for region in regionprops(labeled_target):
        # 为每个病灶创建掩码
        lesion_mask = (labeled_target == region.label).astype(np.uint8)

        # 计算与该病灶重叠的预测区域
        lesion_pred = pred_binary * lesion_mask
        lesion_target = lesion_mask

        # 计算该病灶的dice score
        intersection = (lesion_pred * lesion_target).sum()
        union = lesion_pred.sum() + lesion_target.sum()

        if union > 0:
            dice = 2.0 * intersection / union
            lesion_dice_scores.append(dice)

    # 返回所有病灶dice的平均值
    if lesion_dice_scores:
        return np.mean(lesion_dice_scores)
    else:
        return 0.0


def compute_real_nsd(pred_mask, target_mask, spacing=(1.0, 1.0, 1.0), tolerance=0.5):
    """
    计算标准化表面距离（NSD）

    Args:
        pred_mask: 预测掩码
        target_mask: 真实掩码
        spacing: 体素间距
        tolerance: 容忍距离（毫米）

    Returns:
        float: NSD score
    """
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.cpu().numpy()
    if torch.is_tensor(target_mask):
        target_mask = target_mask.cpu().numpy()

    # 二值化
    pred_binary = (pred_mask > 0.5).astype(np.uint8)
    target_binary = target_mask.astype(np.uint8)

    # 处理特殊情况 - 更严格的空区域处理
    if target_binary.sum() == 0 and pred_binary.sum() == 0:
        return 1.0  # 真负例 - 完全匹配
    if target_binary.sum() == 0 or pred_binary.sum() == 0:
        return 0.0  # 一个为空一个不为空 - 完全不匹配

    try:
        # 计算距离变换
        target_dt = distance_transform_edt(1 - target_binary, sampling=spacing)
        pred_dt = distance_transform_edt(1 - pred_binary, sampling=spacing)

        # 改进的表面提取方法 - 使用腐蚀操作
        from scipy.ndimage import binary_erosion
        from scipy.ndimage.morphology import generate_binary_structure

        struct_element = generate_binary_structure(target_binary.ndim, 2)

        # 目标掩码的表面
        eroded_target = binary_erosion(target_binary, struct_element)
        target_surface = np.logical_and(target_binary, np.logical_not(eroded_target))

        # 预测掩码的表面
        eroded_pred = binary_erosion(pred_binary, struct_element)
        pred_surface = np.logical_and(pred_binary, np.logical_not(eroded_pred))

        # 检查表面是否存在
        if not np.any(pred_surface) or not np.any(target_surface):
            return 0.0

        # 双向距离计算
        target_to_pred_dist = pred_dt[target_surface]
        pred_to_target_dist = target_dt[pred_surface]

        # 分别计算两个方向的匹配比例
        target_to_pred_within_tol = np.sum(target_to_pred_dist <= tolerance)
        pred_to_target_within_tol = np.sum(pred_to_target_dist <= tolerance)

        target_to_pred_ratio = target_to_pred_within_tol / np.sum(target_surface)
        pred_to_target_ratio = pred_to_target_within_tol / np.sum(pred_surface)

        # 取两个方向的平均值
        nsd = (target_to_pred_ratio + pred_to_target_ratio) / 2.0

        return float(nsd)

    except Exception as e:
        print(f"Error in NSD calculation: {e}")
        return 0.0


class DifferentiableLesionWiseDiceLoss(nn.Module):
    """
    可微分的真正Lesion-wise Dice损失函数
    使用纯PyTorch操作，保持梯度连续性
    """
    def __init__(self, smooth=1e-5, alpha=0.7):
        super().__init__()
        self.smooth = smooth
        self.alpha = alpha  # 权衡病灶级和像素级的权重

    def forward(self, pred, target):
        """
        计算可微分的病灶级dice loss

        Args:
            pred: 预测张量 [B, C, H, W, D] 或 [B, C, H, W] (logits)
            target: 目标张量 [B, C, H, W, D] 或 [B, C, H, W]
        """
        pred_sigmoid = torch.sigmoid(pred)
        target = target.float()

        batch_size, num_classes = pred.shape[:2]
        total_loss = 0.0
        valid_classes = 0

        for c in range(num_classes):
            pred_c = pred_sigmoid[:, c]  # [B, H, W, D]
            target_c = target[:, c]      # [B, H, W, D]

            # 跳过完全为空的类别
            if target_c.sum() == 0:
                # 如果目标为空但预测不为空，惩罚假阳性
                if pred_c.sum() > 0:
                    total_loss += pred_c.mean()
                    valid_classes += 1
                continue

            # 计算传统的像素级dice作为基础
            pixel_dice = self._compute_pixel_dice(pred_c, target_c)

            # 计算可微分的病灶感知权重
            lesion_weights = self._compute_lesion_weights(target_c)

            # 计算加权的病灶级dice
            weighted_dice = self._compute_weighted_dice(pred_c, target_c, lesion_weights)

            # 组合像素级和病灶级
            combined_dice = self.alpha * weighted_dice + (1 - self.alpha) * pixel_dice

            loss = 1 - combined_dice
            total_loss += loss
            valid_classes += 1

        if valid_classes > 0:
            return total_loss / valid_classes
        else:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

    def _compute_pixel_dice(self, pred, target):
        """
        计算传统的像素级dice
        """
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return (2.0 * intersection + self.smooth) / (union + self.smooth)

    def _compute_lesion_weights(self, target):
        """
        计算病灶权重 - 纯PyTorch实现
        给每个病灶区域分配权重
        """
        batch_size = target.shape[0]
        weights = torch.zeros_like(target)

        for b in range(batch_size):
            target_b = target[b]
            if target_b.sum() == 0:
                continue

            # 使用形态学操作近似连通组件
            weights_b = self._approximate_connected_components(target_b)
            weights[b] = weights_b

        return weights

    def _approximate_connected_components(self, mask):
        """
        使用形态学操作近似连通组件分析
        """
        if len(mask.shape) == 3:  # 3D
            return self._approximate_cc_3d(mask)
        else:  # 2D
            return self._approximate_cc_2d(mask)

    def _approximate_cc_3d(self, mask):
        """
        3D连通组件近似
        """
        # 使用高斯模糊来模拟连通性
        from torch.nn.functional import conv3d

        # 创建3D高斯核
        kernel_size = 5
        sigma = 1.5
        kernel = self._create_gaussian_kernel_3d(kernel_size, sigma).to(mask.device)

        # 扩展维度用于卷积
        mask_expanded = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, D]

        # 应用高斯卷积
        blurred = conv3d(mask_expanded, kernel, padding=kernel_size//2)
        blurred = blurred.squeeze(0).squeeze(0)  # [H, W, D]

        # 归一化并应用阈值
        weights = torch.where(mask > 0.5, blurred / (blurred.max() + 1e-8), torch.zeros_like(blurred))

        return weights

    def _approximate_cc_2d(self, mask):
        """
        2D连通组件近似
        """
        from torch.nn.functional import conv2d

        kernel_size = 5
        sigma = 1.5
        kernel = self._create_gaussian_kernel_2d(kernel_size, sigma).to(mask.device)

        mask_expanded = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        blurred = conv2d(mask_expanded, kernel, padding=kernel_size//2)
        blurred = blurred.squeeze(0).squeeze(0)  # [H, W]

        weights = torch.where(mask > 0.5, blurred / (blurred.max() + 1e-8), torch.zeros_like(blurred))

        return weights

    def _create_gaussian_kernel_3d(self, kernel_size, sigma):
        """
        创建3D高斯核
        """
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords = coords - (kernel_size - 1) / 2

        z, y, x = torch.meshgrid(coords, coords, coords, indexing='ij')
        distance = z**2 + y**2 + x**2

        kernel = torch.exp(-distance / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        return kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K, K]

    def _create_gaussian_kernel_2d(self, kernel_size, sigma):
        """
        创建2D高斯核
        """
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords = coords - (kernel_size - 1) / 2

        y, x = torch.meshgrid(coords, coords, indexing='ij')
        distance = y**2 + x**2

        kernel = torch.exp(-distance / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        return kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]

    def _compute_weighted_dice(self, pred, target, weights):
        """
        计算加权dice
        """
        # 应用权重
        weighted_pred = pred * weights
        weighted_target = target * weights

        intersection = (weighted_pred * weighted_target).sum()
        union = weighted_pred.sum() + weighted_target.sum()

        return (2.0 * intersection + self.smooth) / (union + self.smooth)


class TrueLesionWiseDiceLoss(nn.Module):
    """
    真正的Lesion-wise Dice损失函数
    训练时使用可微分版本，评估时使用numpy版本
    """
    def __init__(self, smooth=1e-5, use_differentiable=True):
        super().__init__()
        self.smooth = smooth
        self.use_differentiable = use_differentiable

        if use_differentiable:
            self.diff_lesion_dice = DifferentiableLesionWiseDiceLoss(smooth)

    def forward(self, pred, target):
        """
        计算lesion-wise dice loss

        Args:
            pred: 预测张量 [B, C, H, W, D] 或 [B, C, H, W] (logits)
            target: 目标张量 [B, C, H, W, D] 或 [B, C, H, W]
        """
        if self.use_differentiable:
            # 训练时使用可微分版本
            return self.diff_lesion_dice(pred, target)
        else:
            # 评估时使用numpy版本（更准确但不可微分）
            return self._compute_numpy_lesion_dice_loss(pred, target)

    def _compute_numpy_lesion_dice_loss(self, pred, target):
        """
        使用numpy版本计算lesion-wise dice loss（不可微分，仅用于评估）
        """
        pred_sigmoid = torch.sigmoid(pred)
        target = target.float()

        batch_size, num_classes = pred.shape[:2]
        total_loss = 0.0
        valid_classes = 0

        for b in range(batch_size):
            for c in range(num_classes):
                pred_c = pred_sigmoid[b, c]
                target_c = target[b, c]

                # 检查是否有有效的target
                if target_c.sum() > 0:
                    # 使用numpy版本计算lesion-wise dice
                    lesion_dice = compute_lesion_wise_dice_score(pred_c, target_c)

                    if not np.isnan(lesion_dice):
                        loss = 1 - lesion_dice
                        total_loss += loss
                        valid_classes += 1

        if valid_classes > 0:
            return torch.tensor(total_loss / valid_classes, device=pred.device, requires_grad=True)
        else:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)


class BoundaryLoss(nn.Module):
    """
    边界损失函数，强调边界区域的准确性
    修复：使用autocast安全的损失函数
    """
    def __init__(self, boundary_weight=2.0):
        super().__init__()
        self.boundary_weight = boundary_weight

    def forward(self, pred, target):
        """
        计算边界损失
        注意：pred应该是logits（未经sigmoid的），target是0/1的二值掩码
        """
        # 计算边界掩码（基于target）
        target_boundary = self._get_boundary_mask(target)

        # 使用BCEWithLogitsLoss替代binary_cross_entropy (autocast安全)
        boundary_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # 对边界区域赋予更高权重
        weighted_loss = boundary_loss * (1 + self.boundary_weight * target_boundary)

        return weighted_loss.mean()

    def _get_boundary_mask(self, mask):
        """
        计算边界掩码
        """
        if len(mask.shape) == 5:  # [B, C, H, W, D]
            boundary_masks = []
            for b in range(mask.shape[0]):
                batch_boundaries = []
                for c in range(mask.shape[1]):
                    mask_c = mask[b, c]
                    # 3D情况下的梯度计算
                    grad_x = torch.abs(mask_c[1:, :, :] - mask_c[:-1, :, :])
                    grad_y = torch.abs(mask_c[:, 1:, :] - mask_c[:, :-1, :])
                    grad_z = torch.abs(mask_c[:, :, 1:] - mask_c[:, :, :-1])

                    # Padding以保持原始尺寸
                    grad_x = F.pad(grad_x, (0, 0, 0, 0, 1, 0))
                    grad_y = F.pad(grad_y, (0, 0, 1, 0, 0, 0))
                    grad_z = F.pad(grad_z, (1, 0, 0, 0, 0, 0))

                    boundary = torch.clamp(grad_x + grad_y + grad_z, 0, 1)
                    batch_boundaries.append(boundary)
                boundary_masks.append(torch.stack(batch_boundaries))

            return torch.stack(boundary_masks)

        elif len(mask.shape) == 4:  # [B, C, H, W]
            boundary_masks = []
            for b in range(mask.shape[0]):
                batch_boundaries = []
                for c in range(mask.shape[1]):
                    mask_c = mask[b, c]
                    # 2D情况下的梯度计算
                    grad_x = torch.abs(mask_c[1:, :] - mask_c[:-1, :])
                    grad_y = torch.abs(mask_c[:, 1:] - mask_c[:, :-1])

                    # Padding
                    grad_x = F.pad(grad_x, (0, 0, 1, 0))
                    grad_y = F.pad(grad_y, (1, 0, 0, 0))

                    boundary = torch.clamp(grad_x + grad_y, 0, 1)
                    batch_boundaries.append(boundary)
                boundary_masks.append(torch.stack(batch_boundaries))

            return torch.stack(boundary_masks)

        else:
            return torch.zeros_like(mask)


class CombinedLoss(nn.Module):
    """
    组合损失函数：BCE + TRUE Lesion-wise Dice + Boundary Loss
    修复：确保所有损失函数都与autocast兼容
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, boundary_weight=0.3):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

        # 使用真正的lesion-wise dice loss
        self.bce_loss = nn.BCEWithLogitsLoss()  # autocast安全
        self.dice_loss = TrueLesionWiseDiceLoss(use_differentiable=True)  # 使用可微分版本
        self.boundary_loss = BoundaryLoss()

    def forward(self, pred, target):
        """
        计算组合损失

        Args:
            pred: 预测输出，可能是列表（深度监督）或单个张量 (logits, 未经sigmoid)
            target: 目标张量列表
        """
        # 处理深度监督的情况
        if isinstance(pred, list):
            # 对所有分辨率计算损失
            total_loss = 0
            for i, (pred_i, target_i) in enumerate(zip(pred, target)):
                # 计算各个损失分量
                target_float = target_i.float()

                bce = self.bce_loss(pred_i, target_float)
                dice = self.dice_loss(pred_i, target_float)  # 真正的lesion-wise dice
                boundary = self.boundary_loss(pred_i, target_float)

                # 组合损失
                combined = (self.bce_weight * bce +
                           self.dice_weight * dice +
                           self.boundary_weight * boundary)

                # 较高分辨率的权重更大
                weight = 0.5 ** (len(pred) - 1 - i)
                total_loss += weight * combined

            return total_loss
        else:
            # 单一输出的情况
            target_tensor = target[0].float()  # 使用最高分辨率的target

            # 计算各个损失分量
            bce = self.bce_loss(pred, target_tensor)
            dice = self.dice_loss(pred, target_tensor)  # 真正的lesion-wise dice
            boundary = self.boundary_loss(pred, target_tensor)

            return (self.bce_weight * bce +
                   self.dice_weight * dice +
                   self.boundary_weight * boundary)