"""
Batchgenerators Transform — 包装 OnTheFlyTumourAugmenter，使其可插入 nnU-Net 的 transform 链。

该 transform 应在 get_training_transforms() 中作为**第一个 transform**（SpatialTransform 之前），
确保扩散模型看到的是未经旋转/翻转/缩放的原始方向数据。
生成的合成肿瘤会被后续的空间和强度增强一起变换，提高鲁棒性。
"""

from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np


class OnTheFlyTumourTransform(AbstractTransform):
    """
    将 OnTheFlyTumourAugmenter 包装为 batchgenerators AbstractTransform，
    使其可以插入 nnU-Net 的 Compose transform 链。

    __call__ 方法接收并返回 data_dict = {'data': ..., 'seg': ..., 'properties': ..., 'keys': ...}。
    data 形状: (B, C, D, H, W) numpy float32
    seg  形状: (B, 1, D, H, W) numpy int16
    """

    def __init__(self, augmenter, data_key="data", seg_key="seg"):
        """
        参数:
            augmenter: OnTheFlyTumourAugmenter 实例（模型和标签池已初始化）
            data_key:  str, data_dict 中数据张量的键名（默认 "data"）
            seg_key:   str, data_dict 中分割标签的键名（默认 "seg"）
        """
        self.augmenter = augmenter
        self.data_key = data_key
        self.seg_key = seg_key

    def __call__(self, **data_dict):
        """
        对 batch 中的每个样本依次调用 augmenter.augment_sample()。

        参数:
            **data_dict: 包含 'data' 和 'seg' 的字典
                data: (B, C, D, H, W) numpy float32 — 多模态 MRI（z-score 归一化）
                seg:  (B, 1, D, H, W) numpy int16 — 分割标签（-1=padding, 0=bg, 1/2/3/4=肿瘤类）

        返回:
            data_dict: 修改后的字典（data 和 seg 被增强）
        """
        data = data_dict[self.data_key]   # (B, C, D, H, W)
        seg = data_dict[self.seg_key]      # (B, 1, D, H, W)

        # 逐个样本处理
        for b in range(data.shape[0]):
            # 提取单样本（保持 channel 维度）
            data_b = data[b:b + 1]  # (1, C, D, H, W)
            seg_b = seg[b:b + 1]    # (1, 1, D, H, W)

            # 调用增强器
            data_b_aug, seg_b_aug, modified = self.augmenter.augment_sample(
                data_b[0],  # (C, D, H, W) — 去掉 batch 维度
                seg_b[0],   # (1, D, H, W)
            )

            if modified:
                # 将增强后的数据写回 batch
                data[b] = data_b_aug
                seg[b] = seg_b_aug

        data_dict[self.data_key] = data
        data_dict[self.seg_key] = seg
        return data_dict
