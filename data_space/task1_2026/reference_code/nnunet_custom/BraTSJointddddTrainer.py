# nnunetv2/training/nnUNetTrainer/BraTSJointTrainerFixed.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, List
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops


def compute_lesion_wise_dice_score_numpy(pred, target, threshold=0.5):
    """计算lesion-wise dice score (仅用于评估)"""
    try:
        if torch.is_tensor(pred):
            pred_np = pred.detach().cpu().numpy()
        else:
            pred_np = pred

        if torch.is_tensor(target):
            target_np = target.detach().cpu().numpy()
        else:
            target_np = target

        pred_binary = (pred_np > threshold).astype(np.uint8)
        target_binary = target_np.astype(np.uint8)

        if target_binary.sum() == 0:
            return 1.0 if pred_binary.sum() == 0 else 0.0

        if pred_binary.sum() == 0:
            return 0.0

        labeled_target = label(target_binary)
        lesion_dice_scores = []

        for region in regionprops(labeled_target):
            lesion_mask = (labeled_target == region.label).astype(np.uint8)
            lesion_pred = pred_binary * lesion_mask
            lesion_target = lesion_mask

            intersection = (lesion_pred * lesion_target).sum()
            union = lesion_pred.sum() + lesion_target.sum()

            if union > 0:
                dice = 2.0 * intersection / union
                lesion_dice_scores.append(dice)

        if lesion_dice_scores:
            return np.mean(lesion_dice_scores)
        else:
            return 0.0
    except:
        return float('nan')


class SimplifiedLesionDiceLoss(nn.Module):
    """简化版病灶级Dice损失"""
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        target = target.float()

        batch_size, num_classes = pred.shape[:2]
        total_loss = 0.0
        valid_classes = 0

        for c in range(num_classes):
            pred_c = pred_sigmoid[:, c]
            target_c = target[:, c]

            # 跳过空类别
            if target_c.sum() == 0:
                if pred_c.sum() > 0:
                    # 轻微惩罚假阳性
                    total_loss += 0.1 * pred_c.mean()
                    valid_classes += 1
                continue

            # 计算标准Dice
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

            loss = 1 - dice
            total_loss += loss
            valid_classes += 1

        if valid_classes > 0:
            return total_loss / valid_classes
        else:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)


class SimplifiedBoundaryLoss(nn.Module):
    """简化版边界损失"""
    def __init__(self, boundary_weight=2.0):
        super().__init__()
        self.boundary_weight = boundary_weight

    def forward(self, pred, target):
        target = target.float()

        # 计算边界掩码
        boundary_mask = self._compute_boundary_mask(target)

        # 基础BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # 边界加权
        boundary_weights = 1 + self.boundary_weight * boundary_mask
        weighted_loss = bce_loss * boundary_weights

        return weighted_loss.mean()

    def _compute_boundary_mask(self, mask):
        """计算边界掩码"""
        if len(mask.shape) == 5:  # [B, C, H, W, D]
            return self._compute_boundary_3d(mask)
        elif len(mask.shape) == 4:  # [B, C, H, W]
            return self._compute_boundary_2d(mask)
        else:
            return torch.zeros_like(mask)

    def _compute_boundary_3d(self, mask):
        """3D边界计算"""
        boundaries = []
        for b in range(mask.shape[0]):
            batch_boundaries = []
            for c in range(mask.shape[1]):
                mask_c = mask[b, c]

                # 计算梯度
                grad_x = torch.abs(mask_c[1:, :, :] - mask_c[:-1, :, :])
                grad_y = torch.abs(mask_c[:, 1:, :] - mask_c[:, :-1, :])
                grad_z = torch.abs(mask_c[:, :, 1:] - mask_c[:, :, :-1])

                # Padding
                grad_x = F.pad(grad_x, (0, 0, 0, 0, 1, 0))
                grad_y = F.pad(grad_y, (0, 0, 1, 0, 0, 0))
                grad_z = F.pad(grad_z, (1, 0, 0, 0, 0, 0))

                # 组合边界
                boundary = torch.clamp(grad_x + grad_y + grad_z, 0, 1)
                batch_boundaries.append(boundary)
            boundaries.append(torch.stack(batch_boundaries))

        return torch.stack(boundaries)

    def _compute_boundary_2d(self, mask):
        """2D边界计算"""
        boundaries = []
        for b in range(mask.shape[0]):
            batch_boundaries = []
            for c in range(mask.shape[1]):
                mask_c = mask[b, c]

                grad_x = torch.abs(mask_c[1:, :] - mask_c[:-1, :])
                grad_y = torch.abs(mask_c[:, 1:] - mask_c[:, :-1])

                grad_x = F.pad(grad_x, (0, 0, 1, 0))
                grad_y = F.pad(grad_y, (1, 0, 0, 0))

                boundary = torch.clamp(grad_x + grad_y, 0, 1)
                batch_boundaries.append(boundary)
            boundaries.append(torch.stack(batch_boundaries))

        return torch.stack(boundaries)


class JointTrainingLoss(nn.Module):
    """Joint Training损失函数"""
    def __init__(self, bce_weight=0.3, dice_weight=0.5, boundary_weight=0.2, rc_label=4):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = SimplifiedLesionDiceLoss()
        self.boundary_loss = SimplifiedBoundaryLoss()

        self.rc_label = rc_label

    def forward(self, output, target):
        """计算组合损失"""
        if isinstance(target, list):
            return self._handle_deep_supervision(output, target)
        else:
            return self._handle_single_output(output, target)

    def _handle_deep_supervision(self, output_list, target_list):
        """处理deep supervision的情况"""
        total_loss = 0

        for out, tgt in zip(output_list, target_list):
            loss = self._handle_single_output(out, tgt)
            total_loss += loss

        return total_loss / len(output_list)

    def _handle_single_output(self, output, target):
        """处理单个输出的损失计算"""
        # 检查batch中是否包含RC标签
        has_rc_in_batch = torch.any(target == self.rc_label)

        if not has_rc_in_batch:
            # 整个batch都是Pre数据（没有RC标签）
            return self._compute_pre_loss(output, target)
        else:
            # batch中包含Post数据，需要逐样本处理
            return self._compute_mixed_batch_loss(output, target)

    def _compute_pre_loss(self, output, target):
        """计算Pre数据的损失（没有RC标签）"""
        if output.shape[1] <= self.rc_label:
            # 如果网络输出通道数不包含RC通道，直接计算损失
            return self._compute_combined_loss(output, target)

        # 修改输出：将RC通道设为背景概率（使用温和的抑制）
        output_modified = output.clone()
        output_modified[:, self.rc_label, ...] = output[:, 0, ...] - 5.0  # 减少抑制强度

        return self._compute_combined_loss(output_modified, target)

    def _compute_mixed_batch_loss(self, output, target):
        """处理混合batch（同时包含Pre和Post数据）"""
        batch_size = output.shape[0]
        total_loss = 0

        for i in range(batch_size):
            # 提取单个样本
            single_output = output[i:i+1]
            single_target = target[i:i+1]

            # 检查当前样本是否包含RC标签
            has_rc = torch.any(single_target == self.rc_label)

            if has_rc:
                # Post数据：正常计算损失
                sample_loss = self._compute_combined_loss(single_output, single_target)
            else:
                # Pre数据：抑制RC通道
                sample_loss = self._compute_pre_loss(single_output, single_target)

            total_loss += sample_loss

        return total_loss / batch_size

    def _compute_combined_loss(self, output, target):
        """计算组合损失"""
        # 将target转换为one-hot格式
        if len(target.shape) == len(output.shape) - 1:
            target_onehot = torch.zeros_like(output)
            for c in range(output.shape[1]):
                target_onehot[:, c] = (target == c).float()
        else:
            target_onehot = target.float()

        # 计算各个损失组件
        bce = self.bce_loss(output, target_onehot)
        dice = self.dice_loss(output, target_onehot)
        boundary = self.boundary_loss(output, target_onehot)

        # 使用成功的权重组合
        total_loss = (
            self.bce_weight * bce +
            self.dice_weight * dice +
            self.boundary_weight * boundary
        )

        return total_loss


class BraTSJointddddTrainer(nnUNetTrainer):
    """修复版BraTS Joint Training Trainer"""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.rc_label = 4

        # 使用成功的权重配置
        self.bce_weight = 0.3
        self.dice_weight = 0.5
        self.boundary_weight = 0.2

        self.print_to_log_file("Initializing Fixed BraTS Joint Trainer...")
        self.print_to_log_file(f"RC label value: {self.rc_label}")
        self.print_to_log_file(f"Loss weights - BCE: {self.bce_weight}, Dice: {self.dice_weight}, Boundary: {self.boundary_weight}")

    def _build_loss(self):
        """构建修复版损失函数"""
        joint_loss = JointTrainingLoss(
            bce_weight=self.bce_weight,
            dice_weight=self.dice_weight,
            boundary_weight=self.boundary_weight,
            rc_label=self.rc_label
        )

        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0
        weights = weights / weights.sum()

        final_loss = DeepSupervisionWrapper(joint_loss, weights)

        if self.device != torch.device('cpu'):
            final_loss = final_loss.to(self.device)

        self.print_to_log_file(f"Built joint loss with {len(weights)} supervision levels")
        self.print_to_log_file(f"Deep supervision weights: {weights}")

        return final_loss

    def train_step(self, batch: dict) -> dict:
        """训练步骤"""
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(self.device.type, enabled=True):
            output = self.network(data)

            if self.loss is None:
                self.loss = self._build_loss()

            l = self.loss(output, target)

        self.grad_scaler.scale(l).backward()
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        """验证步骤"""
        # 使用标准nnUNet验证步骤
        result = super().validation_step(batch)

        # 添加调试信息
        if not hasattr(self, '_debug_printed'):
            target = batch['target']
            has_rc = torch.any(target == self.rc_label) if not isinstance(target, list) else any(torch.any(t == self.rc_label) for t in target)
            self.print_to_log_file(f"DEBUG: Validation batch contains RC labels: {has_rc}")
            self._debug_printed = True

        return result

    def initialize(self):
        """初始化时构建自定义损失"""
        super().initialize()
        self.loss = self._build_loss()

    def on_train_start(self):
        """训练开始时的设置"""
        super().on_train_start()
        if self.loss is None:
            self.loss = self._build_loss()
        self.print_to_log_file("=" * 60)
        self.print_to_log_file("FIXED BRATS JOINT TRAINER:")
        self.print_to_log_file(f"RC Label: {self.rc_label}")
        self.print_to_log_file("Pre data: labels 0,1,2,3 (no RC)")
        self.print_to_log_file("Post data: labels 0,1,2,3,4 (with RC)")
        self.print_to_log_file(f"Loss: BCE({self.bce_weight}) + Dice({self.dice_weight}) + Boundary({self.boundary_weight})")
        self.print_to_log_file("KEY FIXES:")
        self.print_to_log_file("✓ 简化损失函数实现")
        self.print_to_log_file("✓ 减少RC抑制强度 (-5.0 而非 -10.0)")
        self.print_to_log_file("✓ 保留joint training功能")
        self.print_to_log_file("✓ 更稳定的梯度流")
        self.print_to_log_file("=" * 60)