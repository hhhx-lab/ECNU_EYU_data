"""
nnU-Net Trainer 子类 — 在训练过程中启用 BraTS 2025 On-the-Fly 肿瘤数据增强。

使用方法:
    export DIFFUSION_CKPT_DIR="Segmentation_Tasks/GliGAN/Checkpoint/brats_2024"
    export LABEL_POOL_GLOB="nnUNet_raw/Dataset232_BraTS/imagesTr/*_seg.nii.gz"
    nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerOnTheFly

与原 GAN 版本的区别:
    - 原版本: 离线预生成所有合成病例 → 保存 .nii.gz → 加入训练集 → nnU-Net 训练
    - 新版本: 每个 batch 动态生成合成肿瘤 → 直接送入 nnU-Net 训练

所有改动均通过子类化实现，不修改 nnU-Net 源代码。
"""

import os
import glob
import sys
from typing import List, Union, Tuple

import numpy as np
import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.data_augmentation.custom_transforms.on_the_fly_tumour import (
    OnTheFlyTumourTransform,
)
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

# 将 GliGAN 路径加入 sys.path，以便导入 OnTheFlyTumourAugmenter
# 路径: nnUNet_install/nnunetv2/... → Segmentation_Tasks/GliGAN/src/infer/
_gligan_infer_dir = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "..", "..",
    "GliGAN", "src", "infer"
)
if _gligan_infer_dir not in sys.path:
    sys.path.insert(0, _gligan_infer_dir)

from on_the_fly_augmentation import OnTheFlyTumourAugmenter


class nnUNetTrainerOnTheFly(nnUNetTrainer):
    """
    BraTS 2025 On-the-Fly 增强 Trainer。

    初始化时加载扩散模型和标签池到 GPU，训练时在每个 batch 的 transform 阶段
    动态插入合成肿瘤。

    关键设计:
        - 覆盖 get_training_transforms() 为实例方法，在最前面插入 OnTheFlyTumourTransform
        - 覆盖 get_dataloaders() 强制 SingleThreadedAugmenter（避免多进程 CUDA fork）
        - 通过环境变量 DIFFUSION_CKPT_DIR 和 LABEL_POOL_GLOB 配置模型和标签池路径
        - 扩散模型加载到 GPU，transform 在主线程运行，GPU 推理安全
    """

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        """
        初始化 Trainer 和 OnTheFlyTumourAugmenter。

        增强器配置通过以下环境变量控制:
            DIFFUSION_CKPT_DIR: 扩散模型 checkpoint 根目录（必需）
            LABEL_POOL_GLOB:    标签池 .nii.gz 文件的 glob 模式（必需）
            DATASET_TYPE:       数据集类型，默认 "BRATS_2023"
            SAMPLING_STEPS:     DDPM 加速采样步数，默认 50
            N_STEPS:            扩散总步数，默认 1000
            BETA_SCHEDULE:      beta schedule，默认 "cosine"
        """
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)

        # ---- 从环境变量读取配置 ----
        diffusion_ckpt_dir = os.environ.get(
            "DIFFUSION_CKPT_DIR",
            os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "..", "..", "..",
                "GliGAN", "Checkpoint", "brats_2024"
            )
        )
        label_pool_glob = os.environ.get(
            "LABEL_POOL_GLOB",
            os.path.join(
                os.environ.get("nnUNet_raw", "nnUNet_raw"),
                "Dataset*_BraTS*", "imagesTr", "*_seg.nii.gz"
            )
        )
        dataset_type = os.environ.get("DATASET_TYPE", "BRATS_2023")
        sampling_steps = int(os.environ.get("SAMPLING_STEPS", "50"))
        sampling_method = os.environ.get("SAMPLING_METHOD", "ddpm")  # 'ddpm' 或 'ddim'
        eta = float(os.environ.get("DDIM_ETA", "0.0"))               # DDIM 随机性系数
        n_steps = int(os.environ.get("N_STEPS", "1000"))
        beta_schedule = os.environ.get("BETA_SCHEDULE", "cosine")

        # ---- 查找标签池文件 ----
        label_pool_paths = sorted(glob.glob(label_pool_glob))
        if not label_pool_paths:
            raise FileNotFoundError(
                f"[nnUNetTrainerOnTheFly] 未找到标签池文件！\n"
                f"  LABEL_POOL_GLOB = {label_pool_glob}\n"
                f"  请设置环境变量 LABEL_POOL_GLOB 指向训练集的 *_seg.nii.gz 文件。"
            )

        print(f"[nnUNetTrainerOnTheFly] 配置:")
        print(f"  DIFFUSION_CKPT_DIR = {diffusion_ckpt_dir}")
        print(f"  LABEL_POOL_GLOB    = {label_pool_glob}")
        print(f"  label_pool 大小     = {len(label_pool_paths)} 个标签文件")
        print(f"  DATASET_TYPE       = {dataset_type}")
        print(f"  SAMPLING_METHOD    = {sampling_method}")
        print(f"  SAMPLING_STEPS     = {sampling_steps}")
        print(f"  DDIM_ETA           = {eta}")
        print(f"  device             = {device}")

        # ---- 初始化增强器 ----
        # 如果 device 是 CPU，强制使用 CPU；否则优先使用 CUDA
        aug_device = "cuda" if (
            device.type == "cuda" and torch.cuda.is_available()
        ) else "cpu"

        self.augmenter = OnTheFlyTumourAugmenter(
            diffusion_ckpt_dir=diffusion_ckpt_dir,
            label_pool_paths=label_pool_paths,
            dataset_type=dataset_type,
            n_steps=n_steps,
            beta_schedule=beta_schedule,
            sampling_steps=sampling_steps,
            sampling_method=sampling_method,
            eta=eta,
            device=aug_device,
        )

    def get_training_transforms(
        self,
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: dict,
        deep_supervision_scales: Union[List, Tuple],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        order_resampling_data: int = 3,
        order_resampling_seg: int = 1,
        border_val_seg: int = -1,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> AbstractTransform:
        """
        覆盖为实例方法（父类是 @staticmethod），在父类 transform 链最前面插入
        OnTheFlyTumourTransform。

        注意:
            - 覆盖为实例方法后，get_dataloaders() 中通过
              self.get_training_transforms(...) 调用时 Python 会正确绑定 self。
            - OnTheFlyTumourTransform 放在 SpatialTransform 之前，
              确保扩散模型看到的是未经旋转/翻转的原始方向数据。
        """
        # ---- 获取父类的 transform 链 ----
        # 父类的 get_training_transforms 是 @staticmethod，直接调用（不传 self）
        parent_transforms = nnUNetTrainer.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales,
            mirror_axes, do_dummy_2d_data_aug,
            order_resampling_data=order_resampling_data,
            order_resampling_seg=order_resampling_seg,
            border_val_seg=border_val_seg,
            use_mask_for_norm=use_mask_for_norm,
            is_cascaded=is_cascaded,
            foreground_labels=foreground_labels,
            regions=regions,
            ignore_label=ignore_label,
        )

        # ---- 拆开 Compose，在最前面插入 OnTheFlyTumourTransform ----
        # parent_transforms 是 Compose 对象，包含 transform 列表
        original_list = list(parent_transforms.transforms)

        # 找到 NumpyToTensor 的位置，在其之前插入（我们的 transform 操作 numpy 数组）
        insert_idx = len(original_list)
        for i, t in enumerate(original_list):
            if isinstance(t, NumpyToTensor):
                insert_idx = i
                break

        # 在最前面插入 OnTheFlyTumourTransform（data 仍在 numpy 阶段）
        new_list = [OnTheFlyTumourTransform(self.augmenter)] + original_list
        # 注意: 这样 NumpyToTensor 仍然在最后，OnTheFlyTumourTransform 在第一个

        return Compose(new_list)

    def get_dataloaders(self):
        """
        覆盖 get_dataloaders，强制使用 SingleThreadedAugmenter（单线程增强器），
        避免多进程 CUDA fork 问题。

        原理:
            - 原 GAN 版本使用 multiprocessing(spawn) 让子进程各自初始化 CUDA
            - 这里改为 SingleThreadedAugmenter，transform 在主线程运行，GPU 安全
            - 数据 I/O 的多进程预取通过 nnU-Net 的 dataloader num_workers 实现，
              与 transform 分离，不受影响
        """
        # 临时覆盖 get_allowed_n_proc_DA 为 0，强制使用单线程增强器
        _original_get_allowed = get_allowed_n_proc_DA

        # 用 monkey-patch 方式覆盖（nnUNetTrainer.get_dataloaders 内部调用此函数）
        import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as _trainer_module
        _trainer_module.get_allowed_n_proc_DA = lambda: 0

        try:
            result = super().get_dataloaders()
        finally:
            # 恢复原始函数
            _trainer_module.get_allowed_n_proc_DA = _original_get_allowed

        return result
