# nnunetv2/training/nnUNetTrainer/nnUNetTrainerCustom.py

import numpy as np
import torch
from typing import Union, Tuple, List
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.custom_losses import (
    CombinedLoss,
    compute_lesion_wise_dice_score,
    compute_real_nsd
)


class nnUNetTrainerCustom(nnUNetTrainer):
    """
    Custom nnUNet trainer with TRUE lesion-wise dice and enhanced NSD evaluation
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        # 初始化评估指标存储
        self.real_nsd_values = []
        self.fast_nsd_values = []

        self.print_to_log_file("Initializing Custom Trainer with TRUE Lesion-wise Dice Loss...")

    def _build_loss(self):
        """
        构建自定义损失函数：BCE + TRUE Lesion-wise Dice + Boundary Loss
        """
        # 使用真正的lesion-wise dice loss
        self.loss = CombinedLoss(
            bce_weight=0.5,
            dice_weight=0.5,  # 真正的lesion-wise dice
            boundary_weight=0.3
        )

        # 确保loss在正确的设备上
        if self.device != torch.device('cpu'):
            self.loss = self.loss.to(self.device)

        self.print_to_log_file("Built TRUE Lesion-wise Dice Loss with differentiable implementation")

    def train_step(self, batch: dict) -> dict:
        """
        修改训练步骤以使用真正的lesion-wise dice loss
        """
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = [target.to(self.device, non_blocking=True)]

        self.optimizer.zero_grad(set_to_none=True)

        # 使用PyTorch的自动混合精度 (autocast)
        with torch.autocast(self.device.type, enabled=True):
            output = self.network(data)
            # 确保损失函数已初始化
            if self.loss is None:
                self._build_loss()
            l = self.loss(output, target)

        # 使用正确的GradScaler进行反向传播
        self.grad_scaler.scale(l).backward()
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        """
        修改验证步骤：使用numpy版本计算真实的lesion-wise dice进行评估
        """
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target_list = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target_list = [target.to(self.device, non_blocking=True)]

        target_onehot = target_list[-1]

        with torch.no_grad():
            output = self.network(data)

            if isinstance(output, list):
                output_for_metrics = output[-1]
            else:
                output_for_metrics = output

            predicted_segmentation_onehot = torch.sigmoid(output_for_metrics)

            if self.loss is None:
                self._build_loss()
            l = self.loss(output, target_list)

            # 为父类计算所需指标
            predicted_binary = (predicted_segmentation_onehot > 0.5).float()
            axes = tuple(range(2, len(predicted_binary.shape)))
            tp_hard = torch.sum(predicted_binary * target_onehot, axes)
            fp_hard = torch.sum(predicted_binary * (1 - target_onehot), axes)
            fn_hard = torch.sum((1 - predicted_binary) * target_onehot, axes)

        # 初始化结果字典
        result = {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp_hard.detach().cpu().numpy(),
            'fp_hard': fp_hard.detach().cpu().numpy(),
            'fn_hard': fn_hard.detach().cpu().numpy()
        }

        # 添加调试信息（仅第一次）
        if not hasattr(self, '_debug_printed'):
            print(f"DEBUG: Using TRUE Lesion-wise Dice Loss")
            print(f"DEBUG: predicted_segmentation_onehot.shape: {predicted_segmentation_onehot.shape}")
            print(f"DEBUG: target_onehot.shape: {target_onehot.shape}")
            self._debug_printed = True

        num_classes = predicted_segmentation_onehot.shape[1]

        # 计算TRUE lesion-wise dice scores - 使用numpy版本确保准确性
        lesion_dice_scores = []
        for c in range(num_classes):
            pred_c = predicted_segmentation_onehot[:, c]
            target_c = target_onehot[:, c]

            if target_c.sum() > 0 or pred_c.sum() > 0:
                batch_dice_scores = []
                for b in range(pred_c.shape[0]):
                    # 使用numpy版本计算真正的lesion-wise dice
                    lesion_dice = compute_lesion_wise_dice_score(pred_c[b], target_c[b])
                    batch_dice_scores.append(lesion_dice)

                valid_scores = [score for score in batch_dice_scores if not np.isnan(score)]
                if valid_scores:
                    lesion_dice_scores.append(np.mean(valid_scores))
                else:
                    lesion_dice_scores.append(float('nan'))
            else:
                lesion_dice_scores.append(float('nan'))

        result['lesion_dice_scores'] = lesion_dice_scores

        # 计算快速NSD scores
        fast_nsd_scores = []
        for c in range(num_classes):
            pred_c = predicted_segmentation_onehot[:, c]
            target_c = target_onehot[:, c]

            if target_c.sum() > 0 or pred_c.sum() > 0:
                fast_nsd = self._compute_boundary_iou(pred_c, target_c)
                fast_nsd_scores.append(fast_nsd)
            else:
                fast_nsd_scores.append(float('nan'))

        result['fast_nsd_scores'] = fast_nsd_scores

        # 每20个epoch计算真实NSD
        if self.current_epoch % 20 == 0:
            real_nsd_scores = []
            spacing = self.configuration_manager.spacing
            for c in range(num_classes):
                pred_c = predicted_segmentation_onehot[:, c]
                target_c = target_onehot[:, c]

                if target_c.sum() > 0 or pred_c.sum() > 0:
                    batch_nsd = []
                    for b in range(pred_c.shape[0]):
                        real_nsd = compute_real_nsd(
                            pred_c[b], target_c[b],
                            spacing=spacing,
                            tolerance=0.5
                        )
                        batch_nsd.append(real_nsd)
                    real_nsd_scores.append(np.mean(batch_nsd))
                else:
                    real_nsd_scores.append(float('nan'))

            result['real_nsd_scores'] = real_nsd_scores

        return result

    def _compute_boundary_iou(self, pred, target):
        """
        计算边界IoU作为快速NSD近似
        """
        pred_boundary = self._get_boundary_mask(pred)
        target_boundary = self._get_boundary_mask(target)

        intersection = (pred_boundary * target_boundary).sum()
        union = pred_boundary.sum() + target_boundary.sum() - intersection

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        return float(intersection / union)

    def _get_boundary_mask(self, mask):
        """
        提取边界掩码
        """
        if len(mask.shape) == 4:  # [B, H, W, D] or [B, H, W]
            boundary_masks = []
            for b in range(mask.shape[0]):
                mask_b = mask[b]
                if len(mask_b.shape) == 3:  # 3D
                    grad_x = torch.abs(mask_b[1:, :, :] - mask_b[:-1, :, :])
                    grad_y = torch.abs(mask_b[:, 1:, :] - mask_b[:, :-1, :])
                    grad_z = torch.abs(mask_b[:, :, 1:] - mask_b[:, :, :-1])

                    grad_x = torch.nn.functional.pad(grad_x, (0, 0, 0, 0, 1, 0))
                    grad_y = torch.nn.functional.pad(grad_y, (0, 0, 1, 0, 0, 0))
                    grad_z = torch.nn.functional.pad(grad_z, (1, 0, 0, 0, 0, 0))

                    boundary = torch.clamp(grad_x + grad_y + grad_z, 0, 1)
                else:  # 2D
                    grad_x = torch.abs(mask_b[1:, :] - mask_b[:-1, :])
                    grad_y = torch.abs(mask_b[:, 1:] - mask_b[:, :-1])

                    grad_x = torch.nn.functional.pad(grad_x, (0, 0, 1, 0))
                    grad_y = torch.nn.functional.pad(grad_y, (1, 0, 0, 0))

                    boundary = torch.clamp(grad_x + grad_y, 0, 1)

                boundary_masks.append(boundary)

            return torch.stack(boundary_masks)
        else:
            return mask

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        """
        验证epoch结束时的处理
        """
        super().on_validation_epoch_end(val_outputs)

        # 处理TRUE lesion-wise dice
        lesion_dice_scores = []
        for val_output in val_outputs:
            if 'lesion_dice_scores' in val_output:
                lesion_dice_scores.append(val_output['lesion_dice_scores'])

        if lesion_dice_scores:
            lesion_dice_mean = np.nanmean(lesion_dice_scores, axis=0)
            overall_lesion_dice = np.nanmean(lesion_dice_mean)

            self.print_to_log_file(f'TRUE Lesion-wise Dice: {overall_lesion_dice:.4f}')
            self.print_to_log_file(f'TRUE Lesion-wise Dice by region: {[f"{x:.4f}" if not np.isnan(x) else "nan" for x in lesion_dice_mean]}')

        # 处理快速NSD
        fast_nsd_scores = []
        for val_output in val_outputs:
            if 'fast_nsd_scores' in val_output:
                fast_nsd_scores.append(val_output['fast_nsd_scores'])

        if fast_nsd_scores:
            fast_nsd_mean = np.nanmean(fast_nsd_scores, axis=0)
            overall_fast_nsd = np.nanmean(fast_nsd_mean)
            self.fast_nsd_values.append(overall_fast_nsd)

            self.print_to_log_file(f'Fast NSD (Boundary-IoU): {overall_fast_nsd:.4f}')
            self.print_to_log_file(f'Fast NSD by region: {[f"{x:.4f}" if not np.isnan(x) else "nan" for x in fast_nsd_mean]}')

        # 处理真实NSD (每20个epoch)
        if self.current_epoch % 20 == 0:
            real_nsd_scores = []
            for val_output in val_outputs:
                if 'real_nsd_scores' in val_output:
                    real_nsd_scores.append(val_output['real_nsd_scores'])

            if real_nsd_scores:
                real_nsd_mean = np.nanmean(real_nsd_scores, axis=0)
                overall_real_nsd = np.nanmean(real_nsd_mean)
                self.real_nsd_values.append(overall_real_nsd)

                self.print_to_log_file(f'=== REAL NSD EVALUATION (Epoch {self.current_epoch}) ===')
                self.print_to_log_file(f'Real NSD: {overall_real_nsd:.4f}')
                self.print_to_log_file(f'Real NSD by region: {[f"{x:.4f}" if not np.isnan(x) else "nan" for x in real_nsd_mean]}')
                self.print_to_log_file(f'=====================================')

    def on_train_epoch_end(self, train_outputs=None):
        """
        训练epoch结束时的处理
        """
        super().on_train_epoch_end(train_outputs)

        if hasattr(self, 'fast_nsd_values') and len(self.fast_nsd_values) > 0:
            self.print_to_log_file(f'Using TRUE lesion-wise dice + boundary-enhanced loss')

        if (self.current_epoch + 1) % 20 == 0:
            self.print_to_log_file(f'Next real NSD evaluation at epoch {self.current_epoch + 1}')

    def initialize(self):
        """
        初始化时构建自定义损失
        """
        super().initialize()
        self._build_loss()

    def on_train_start(self):
        """
        训练开始时的设置
        """
        super().on_train_start()
        if self.loss is None:
            self._build_loss()
        self.print_to_log_file("="*60)
        self.print_to_log_file("TRUE LESION-WISE DICE TRAINER CONFIGURATION:")
        self.print_to_log_file("- Loss: BCE(0.5) + TRUE Lesion-wise Dice(0.5) + Boundary(0.3)")
        self.print_to_log_file("- Training: Differentiable lesion-wise dice (maintains gradients)")
        self.print_to_log_file("- Evaluation: Numpy lesion-wise dice (accurate)")
        self.print_to_log_file("- Fast NSD: Boundary IoU (every epoch)")
        self.print_to_log_file("- Real NSD: Surface distance (every 20 epochs)")
        self.print_to_log_file("- Key improvement: TRUE lesion-level optimization")
        self.print_to_log_file("="*60)