"""
On-the-Fly 肿瘤数据增强器 — BraTS 2025 风格动态增强。

替代原先离线预生成所有合成病例的流程（main_random_label_random_dataset_generator_multiprocess.py），
改为在 nnU-Net 训练过程中对每个 batch 动态执行以下五步：

    Step 1: 以 60% 概率选中某个样本进行增强
    Step 2: 从训练集标签池中随机借入一个其他病人的真实标签
    Step 3: 对借入标签进行 SNFH→ET→NETC 级联类替换（每步 70% 概率）
    Step 4: 基于 SNFH 是否被移除进行差分缩放
    Step 5: 以 40% 概率插入第 2 个肿瘤；通过扩散模型 inpainting 生成肿瘤外观

使用方法（命令行，与 GAN 版本相似）:
    python on_the_fly_augmentation.py \\
        --diffusion_ckpt_dir ../../Checkpoint/brats2026_diffusion \\
        --label_pool_csv ../../Checkpoint/brats2026_diffusion/brats2026_diffusion.csv \\
        --dataset BRATS_2026 \\
        --sampling_steps 50 \\
        --device cuda

或在 nnU-Net trainer 中导入 OnTheFlyTumourAugmenter 类直接使用。
"""

import os
import sys
import argparse
import glob

import numpy as np
import torch
import nibabel as nib

# 确保可以导入同目录下的 diffusion_inference_utils
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from diffusion_inference_utils import (
    make_diffusion_coefficients,
    sample_tumour_diffusion_inpaint,
    rescale_to_mm1,
    rescale_from_mm1,
    correct_background_vec,
    correct_label_vec,
    get_inten_coord_vec,
    linear_interpolation_vec,
    add_gaussian_noise_tumour_vec,
)

# 从 src.networks 导入扩散网络和标签修改器
from src.networks.DiffusionNetwork import get_diffusion_network
from src.utils.label_modifier import modify_borrowed_label


class OnTheFlyTumourAugmenter:
    """
    加载扩散模型 + 标签池，提供单样本 on-the-fly 增强。

    初始化时加载所有模型到指定设备（GPU 推荐），将训练集所有标签预加载到内存。
    每次调用 augment_sample() 对单个样本执行 BraTS 2025 五步增强流程。

    注意：
        - 所有模型必须在初始化前完成训练，并保存为以下格式的 checkpoint：
          {diffusion_ckpt_dir}/{modality}/weights/diffusion_{iter}.pt
        - 标签池文件为 .nii.gz 格式的整型分割标签，从 nnU-Net raw 数据集中提取
        - 模型在 __init__ 中加载到设备，整个生命周期内不移动
    """

    def __init__(
        self,
        diffusion_ckpt_dir,
        label_pool_paths,
        dataset_type="BRATS_2026",
        n_steps=1000,
        beta_schedule="cosine",
        sampling_steps=50,
        sampling_method='ddpm',      # 'ddpm' 或 'ddim'
        eta=0.0,                     # DDIM 随机性系数 (method='ddpm' 时忽略)
        device="cuda",
        generator_type="SwinUNETR",
        feature_size=48,
        use_checkpoint=False,
        in_channels_tumour=4,
        out_channels_tumour=1,
    ):
        """
        参数:
            diffusion_ckpt_dir: str, 扩散模型 checkpoint 根目录
                               结构：{dir}/{modality}/weights/diffusion_{iter}.pt
            label_pool_paths:   list[str], 标签池中所有 .nii.gz 文件的路径列表
            dataset_type:       str, "BRATS_2026"（当前 2026 MET 线）或旧版 "BRATS_2023"/"BRATS_2024"
            n_steps:            int, 扩散总步数（默认 1000）
            beta_schedule:      str, beta schedule 类型（默认 "cosine"）
            sampling_steps:     int, 加速采样步数（DDIM 模式生效，默认 50）
            sampling_method:    str, 采样算法 'ddpm' 或 'ddim'
            eta:                float, DDIM 随机性系数 (0=确定性, 1≈DDPM)
            device:             str, "cuda" 或 "cpu"
            generator_type:     str, 网络骨干类型（"SwinUNETR" / "AttentionUnet" / "Unet"）
            feature_size:       int, SwinUNETR 特征维度
            use_checkpoint:     bool, 是否使用梯度检查点（推理时无效但保留接口兼容）
            in_channels_tumour: int, 肿瘤扩散模型输入通道数（= scan_channels + label_channels）
            out_channels_tumour:int, 肿瘤扩散模型输出通道数（= 1）
        """
        self.device = device
        self.dataset_type = dataset_type
        self.n_steps = n_steps
        self.beta_schedule = beta_schedule
        self.sampling_steps = sampling_steps if sampling_steps > 0 else n_steps
        self.sampling_method = sampling_method
        self.eta = eta
        self.in_channels_tumour = in_channels_tumour
        self.out_channels_tumour = out_channels_tumour

        # ---- 构建伪 args 对象（复用 get_diffusion_network 工厂函数）----
        class _Args:
            pass
        self._args = _Args()
        self._args.generator_type = generator_type
        self._args.feature_size = feature_size
        self._args.use_checkpoint = use_checkpoint
        self._args.in_channels = in_channels_tumour
        self._args.out_channels = out_channels_tumour

        # ---- 加载 4 模态扩散模型 ----
        print(f"[OnTheFly] 加载扩散模型到 {device}...")
        self.models = {}  # key: modality_name → model
        self._load_diffusion_models(diffusion_ckpt_dir)

        # ---- 构建扩散系数 ----
        print(f"[OnTheFly] 构建扩散系数 (schedule={beta_schedule}, n_steps={n_steps})...")
        betas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, alphas_bar = make_diffusion_coefficients(
            n_steps=n_steps,
            beta_schedule=beta_schedule,
            device=device,
        )
        self.betas = betas
        self.alphas_bar_sqrt = alphas_bar_sqrt
        self.one_minus_alphas_bar_sqrt = one_minus_alphas_bar_sqrt
        self.alphas_bar = alphas_bar  # DDIM 所需完整累积 α 序列

        # ---- 加载标签池 ----
        print(f"[OnTheFly] 加载标签池 ({len(label_pool_paths)} 个标签文件)...")
        self.label_pool = self._load_label_pool(label_pool_paths)
        print(f"[OnTheFly] 初始化完成。标签池大小: {len(self.label_pool)}")

    def _load_diffusion_models(self, ckpt_dir):
        """
        从指定目录加载 4 个模态的扩散模型 checkpoint。

        checkpoint 命名约定: {ckpt_dir}/{modality}/weights/diffusion_{iter}.pt
        其中 modality ∈ {t1c, t1, t2, flair}
        """
        modalities = ["t1c", "t1n", "t2w", "t2f"]
        for modality in modalities:
            weights_dir = os.path.join(ckpt_dir, modality, "weights")
            if not os.path.isdir(weights_dir):
                raise FileNotFoundError(
                    f"[OnTheFly] 找不到 {modality} 的权重目录: {weights_dir}"
                )
            # 找到该目录下最新的 diffusion checkpoint
            ckpt_files = sorted(glob.glob(os.path.join(weights_dir, "diffusion_*.pt")))
            if not ckpt_files:
                raise FileNotFoundError(
                    f"[OnTheFly] 在 {weights_dir} 中找不到 diffusion_*.pt checkpoint"
                )
            ckpt_path = ckpt_files[-1]  # 使用最新的 checkpoint
            print(f"  [{modality}] 加载: {ckpt_path}")

            # 创建模型并加载权重
            model = get_diffusion_network(self._args, n_steps=self.n_steps)
            ckpt = torch.load(ckpt_path, map_location=torch.device(self.device))
            if "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"])
            else:
                model.load_state_dict(ckpt)
            model.to(self.device)
            model.eval()
            self.models[modality] = model

    def _load_label_pool(self, label_paths):
        """
        将标签池的所有 .nii.gz 文件加载到内存中。

        每个标签文件在内存中保存为 int16 numpy 数组。
        对于大数据集（>100 个样本），可改为 on-demand 加载。
        """
        pool = []
        for path in label_paths:
            try:
                label_data = nib.load(path).get_fdata().astype(np.int16)
                pool.append(label_data)
            except Exception as e:
                print(f"  [警告] 无法加载标签 {path}: {e}，跳过")
        if not pool:
            raise RuntimeError("[OnTheFly] 标签池为空！请检查 label_pool_paths 参数")
        return pool

    def augment_sample(self, data_4ch, seg, rng=None):
        """
        对单个样本执行 BraTS 2025 on-the-fly 增强。

        输入:
            data_4ch: np.ndarray, 形状 (4, D, H, W)
                      4 个 MRI 模态 (t1c, t1, t2, flair)，z-score 归一化
            seg:      np.ndarray, 形状 (1, D, H, W)
                      原始分割标签，值域 {-1(padding), 0(bg), 1(NETC), 2(SNFH), 3(ET), 4(RC)}

        返回:
            data_4ch:     np.ndarray, 增强后的数据（形状不变）
            seg:          np.ndarray, 合并了合成肿瘤标签的分割图
            was_modified: bool, 是否实际进行了增强
        """
        if rng is None:
            rng = np.random

        D, H, W = data_4ch.shape[1], data_4ch.shape[2], data_4ch.shape[3]

        # ---- 快速检查：patch 是否足够大 ----
        if D < 96 or H < 96 or W < 96:
            return data_4ch, seg, False

        # ---- Step 1: 60% 概率选中 ----
        if rng.uniform() >= 0.6:
            return data_4ch, seg, False

        num_inserted = 0

        # ---- 借入标签 + 修改 + 插入 ----
        for tumour_idx in range(2):  # 最多 2 个肿瘤
            if tumour_idx == 1:
                # 第二个肿瘤：40% 概率
                if rng.uniform() >= 0.4:
                    break

            # ---- Step 2: 从标签池随机借入一个标签 ----
            pool_idx = rng.randint(0, len(self.label_pool))
            borrowed_label = self.label_pool[pool_idx]

            # ---- Step 3+4: 修改借入标签 + 差分缩放 ----
            modified_label, meta = modify_borrowed_label(borrowed_label, rng)

            # ---- 将修改后的标签适配到目标 patch ----
            # 提取肿瘤 bounding box，裁剪后 pad 到 96³
            tumour_crop_96 = self._prepare_label_for_insertion(modified_label, rng)

            # ---- Step 5: 在目标数据中找空位，扩散 inpainting 插入 ----
            success = self._insert_one_tumour(
                data_4ch, seg, tumour_crop_96, meta, rng
            )
            if success:
                num_inserted += 1

        was_modified = (num_inserted > 0)
        return data_4ch, seg, was_modified

    def _prepare_label_for_insertion(self, modified_label, rng):
        """
        将修改后的借入标签裁剪到其肿瘤 bounding box，然后 pad 到 96³。

        参数:
            modified_label: np.ndarray (H, W, D)，修改后的整型标签
        返回:
            np.ndarray (96, 96, 96)，肿瘤居中、周围填 0 的标签立方体
        """
        # 找到非零区域的 bounding box
        non_zero = (modified_label != 0)
        if not np.any(non_zero):
            # 没有肿瘤区域，返回全零的 96³
            return np.zeros((96, 96, 96), dtype=np.int16)

        coords = np.where(non_zero)
        z_min, z_max = coords[0].min(), coords[0].max() + 1
        y_min, y_max = coords[1].min(), coords[1].max() + 1
        x_min, x_max = coords[2].min(), coords[2].max() + 1

        # 裁剪
        crop = modified_label[z_min:z_max, y_min:y_max, x_min:x_max].astype(np.int16)

        # 如果任一维度 > 96，等比缩放至 96³ 以内
        max_dim = max(crop.shape)
        if max_dim > 92:  # 留 4 体素边距
            scale = 92.0 / max_dim
            from scipy.ndimage import zoom as ndimage_zoom
            new_shape = np.maximum(np.round(np.array(crop.shape) * scale), 1).astype(int)
            zoom_factors = tuple(new_shape / np.array(crop.shape))
            crop = ndimage_zoom(crop.astype(np.float32), zoom_factors, order=0)
            crop = np.round(crop).astype(np.int16)

        # Pad 到 96³，肿瘤居中
        result = np.zeros((96, 96, 96), dtype=np.int16)
        z_start = (96 - crop.shape[0]) // 2
        y_start = (96 - crop.shape[1]) // 2
        x_start = (96 - crop.shape[2]) // 2

        z_end = z_start + crop.shape[0]
        y_end = y_start + crop.shape[1]
        x_end = x_start + crop.shape[2]

        result[z_start:z_end, y_start:y_end, x_start:x_end] = crop
        return result

    def _insert_one_tumour(self, data_4ch, seg, tumour_label_96, meta, rng):
        """
        在目标数据中找一个不重叠的位置，用扩散 inpainting 插入合成肿瘤。

        参数:
            data_4ch:        np.ndarray (4, D, H, W)，4 模态 MRI 数据
            seg:             np.ndarray (1, D, H, W)，分割标签
            tumour_label_96: np.ndarray (96, 96, 96)，要插入的肿瘤标签
            meta:            dict，来自 modify_borrowed_label 的元信息
            rng:             numpy random generator

        返回:
            bool: 是否成功插入
        """
        D, H, W = data_4ch.shape[1], data_4ch.shape[2], data_4ch.shape[3]

        # 搜索插入位置：在 96³ 窗口内无现有肿瘤、无 padding
        centre = self._find_insertion_center(data_4ch, seg, tumour_label_96, rng)
        if centre is None:
            return False  # 没有足够空间

        cx, cy, cz = centre  # 96³ 立方体的中心坐标

        # 定义 96³ 区域的边界
        half = 48
        z0, z1 = max(cz - half, 0), min(cz + half, D)
        y0, y1 = max(cy - half, 0), min(cy + half, H)
        x0, x1 = max(cx - half, 0), min(cx + half, W)

        # 如果边界不足 96，调整（尽量保持 96³）
        if z1 - z0 < 96:
            shift = min(cz, D - cz)
            z0 = max(0, cz - shift)
            z1 = min(D, z0 + 96)
            z0 = max(0, z1 - 96)
        if y1 - y0 < 96:
            shift = min(cy, H - cy)
            y0 = max(0, cy - shift)
            y1 = min(H, y0 + 96)
            y0 = max(0, y1 - 96)
        if x1 - x0 < 96:
            shift = min(cx, W - cx)
            x0 = max(0, cx - shift)
            x1 = min(W, x0 + 96)
            x0 = max(0, x1 - 96)

        # 从 96³ 肿瘤标签中裁剪对应区域（标签可能不满 96³ 但 pad 是 96³ 居中的）
        tz0 = (96 - (z1 - z0)) // 2
        ty0 = (96 - (y1 - y0)) // 2
        tx0 = (96 - (x1 - x0)) // 2
        tz1, ty1, tx1 = tz0 + (z1 - z0), ty0 + (y1 - y0), tx0 + (x1 - x0)
        tumour_label_crop = tumour_label_96[tz0:tz1, ty0:ty1, tx0:tx1]

        # 对每个模态执行扩散 inpainting
        modality_keys = ["t1c", "t1n", "t2w", "t2f"]
        for ch_idx, modality in enumerate(modality_keys):
            # 提取目标 96³ patch
            target_patch = data_4ch[ch_idx, z0:z1, y0:y1, x0:x1].copy()

            # Pad 到精确 96³（如果需要）
            pad_z = 96 - target_patch.shape[0]
            pad_y = 96 - target_patch.shape[1]
            pad_x = 96 - target_patch.shape[2]
            patch_96 = np.pad(target_patch,
                              ((0, pad_z), (0, pad_y), (0, pad_x)),
                              mode='constant', constant_values=0)

            label_pad_z = 96 - tumour_label_crop.shape[0]
            label_pad_y = 96 - tumour_label_crop.shape[1]
            label_pad_x = 96 - tumour_label_crop.shape[2]
            label_96 = np.pad(tumour_label_crop,
                              ((0, label_pad_z), (0, label_pad_y), (0, label_pad_x)),
                              mode='constant', constant_values=0)

            # 归一化：z-score → [-1, 1]（min-max 缩放，与 GAN 版本一致）
            patch_normed, arr_min, arr_max = rescale_to_mm1(patch_96)

            # 加高斯噪声到肿瘤区域
            patch_noisy, noise_mask = add_gaussian_noise_tumour_vec(patch_normed, label_96)

            # 转换为 torch tensor，送入扩散模型
            noisy_tensor = torch.from_numpy(patch_noisy).float().unsqueeze(0).unsqueeze(0)
            label_tensor = torch.from_numpy(
                self._convert_label_to_multichannel(label_96)
            ).float().unsqueeze(0)

            noisy_tensor = noisy_tensor.to(self.device)
            label_tensor = label_tensor.to(self.device)

            # 扩散 inpainting 生成
            generated = sample_tumour_diffusion_inpaint(
                model=self.models[modality],
                noisy_scan=noisy_tensor,
                label_cond=label_tensor,
                n_steps=self.n_steps,
                betas=self.betas,
                alphas_bar_sqrt=self.alphas_bar_sqrt,
                one_minus_alphas_bar_sqrt=self.one_minus_alphas_bar_sqrt,
                device=self.device,
                method=self.sampling_method,
                sampling_steps=self.sampling_steps,
                eta=self.eta,
                alphas_bar=self.alphas_bar,
            )

            # 转回 numpy
            generated_np = generated.squeeze(0).squeeze(0).cpu().numpy()  # (96, 96, 96)

            # 后处理：背景修正
            generated_corrected = correct_background_vec(patch_normed, generated_np)

            # 强度校正：用参照点的线性插值映射回原始值域
            untouch_x, untouch_y, untouch_z = get_inten_coord_vec(
                patch_normed, label_96, noise_mask
            )
            generated_final = linear_interpolation_vec(
                generated_corrected, patch_96,
                untouch_x, untouch_y, untouch_z
            )

            # 去掉 padding，恢复到目标 patch 的实际尺寸
            generated_cropped = generated_final[:target_patch.shape[0],
                                                :target_patch.shape[1],
                                                :target_patch.shape[2]]

            # 写回 data_4ch
            data_4ch[ch_idx, z0:z1, y0:y1, x0:x1] = generated_cropped

        # 更新 seg：将肿瘤标签合并到分割图
        # 只修改非 padding 区域（seg != -1）
        seg_region = seg[0, z0:z1, y0:y1, x0:x1]
        valid_mask = (seg_region != -1)
        tumour_valid = tumour_label_crop[:seg_region.shape[0],
                                          :seg_region.shape[1],
                                          :seg_region.shape[2]].copy()
        tumour_valid[~valid_mask] = 0
        seg[0, z0:z1, y0:y1, x0:x1][tumour_valid != 0] = tumour_valid[tumour_valid != 0]

        return True

    def _find_insertion_center(self, data_4ch, seg, tumour_label_96, rng, max_attempts=50):
        """
        在目标 patch 中搜索一个 96³ 区域，要求该区域内无现有肿瘤、无 padding。

        参数:
            data_4ch:        (4, D, H, W)
            seg:             (1, D, H, W)
            tumour_label_96: (96, 96, 96) 要插入的肿瘤标签（用于检查尺寸）
            rng:             numpy random generator
            max_attempts:    最大尝试次数

        返回:
            (cx, cy, cz) 或 None
        """
        D, H, W = data_4ch.shape[1], data_4ch.shape[2], data_4ch.shape[3]

        # 构建"有效区域"掩码：脑内 且 无现有肿瘤（seg==0）且 非 padding（seg!=-1）
        # 用 T1ce（第 0 通道）判断脑内：z-score 归一化后，脑内组织值 > 背景
        is_brain = (data_4ch[0] != 0)
        no_tumour = (seg[0] == 0)
        valid_mask = is_brain & no_tumour  # (D, H, W)

        half = 48
        for _ in range(max_attempts):
            # 随机采样中心点（只在有效区域内）
            valid_indices = np.where(valid_mask)
            if len(valid_indices[0]) == 0:
                return None

            idx = rng.randint(0, len(valid_indices[0]))
            cz = valid_indices[0][idx]
            cy = valid_indices[1][idx]
            cx = valid_indices[2][idx]

            # 确保以 (cx, cy, cz) 为中心的 96³ 区域完全在边界内且无肿瘤无 padding
            z0 = cz - half
            z1 = cz + half
            y0 = cy - half
            y1 = cy + half
            x0 = cx - half
            x1 = cx + half

            if z0 < 0 or z1 > D or y0 < 0 or y1 > H or x0 < 0 or x1 > W:
                continue

            # 检查该 96³ 区域内是否有肿瘤或 padding
            if np.all(valid_mask[z0:z1, y0:y1, x0:x1]):
                return (cx, cy, cz)

        return None

    def _convert_label_to_multichannel(self, label):
        """
        将单通道整型标签转换为多通道标签（与 GAN 推理中的 LABEL_TRANSFORM 一致）。

        BraTS 2023 (3 通道):
            Channel 0: TC  (NETC + ET) = label==1 OR label==3
            Channel 1: WT  (NETC + SNFH + ET) = label==1 OR label==2 OR label==3
            Channel 2: ET  = label==3

        BraTS 2024/2026 (4 通道):
            同上，额外 Channel 3: RC = label==4

        参数:
            label: np.ndarray (H, W, D)，整型标签 {0,1,2,3,4}
        返回:
            np.ndarray (C, H, W, D)，多通道标签
        """
        if self.dataset_type in {"BRATS_2024", "BRATS_2026"}:
            n_channels = 4
            result = np.zeros((n_channels,) + label.shape, dtype=np.float32)
            result[0] = ((label == 1) | (label == 3)).astype(np.float32)  # TC
            result[1] = ((label == 1) | (label == 2) | (label == 3)).astype(np.float32)  # WT
            result[2] = (label == 3).astype(np.float32)  # ET
            result[3] = (label == 4).astype(np.float32)  # RC
        else:  # Legacy BRATS_2023 or BRATS_GOAT_2024 support only.
            n_channels = 3
            result = np.zeros((n_channels,) + label.shape, dtype=np.float32)
            result[0] = ((label == 1) | (label == 3)).astype(np.float32)  # TC
            result[1] = ((label == 1) | (label == 2) | (label == 3)).astype(np.float32)  # WT
            result[2] = (label == 3).astype(np.float32)  # ET
        return result


# =========================================================================
# 命令行入口：用于独立测试 on-the-fly 增强器
# 用法（与原 GAN 推理命令相似）:
#   cd Segmentation_Tasks/GliGAN/src/infer
#   python on_the_fly_augmentation.py \
#       --diffusion_ckpt_dir ../../Checkpoint/brats2026_diffusion \
#       --label_pool_csv ../../Checkpoint/brats2026_diffusion/brats2026_diffusion.csv \
#       --dataset BRATS_2026 \
#       --sampling_steps 50
# =========================================================================

def __main__():
    parser = argparse.ArgumentParser(
        description="On-the-Fly 肿瘤增强器 - BraTS 2025 风格动态增强"
    )
    parser.add_argument("--diffusion_ckpt_dir", type=str, required=True,
                        help="扩散模型 checkpoint 根目录（结构: {dir}/{modality}/weights/diffusion_*.pt）")
    parser.add_argument("--label_pool_csv", type=str, default="",
                        help="标签池 CSV 文件路径。CSV 中需包含 'label' 列。如为空，使用 --label_pool_dir")
    parser.add_argument("--label_pool_dir", type=str, default="",
                        help="标签池目录（备用，glob 所有 *_seg.nii.gz 文件）")
    parser.add_argument("--dataset", type=str, default="BRATS_2026",
                        help="数据集类型: BRATS_2026 / BRATS_2023 / BRATS_2024 / BRATS_GOAT_2024")
    parser.add_argument("--sampling_steps", type=int, default=50,
                        help="DDPM 加速采样步数（0 = 使用完整 1000 步）")
    parser.add_argument("--n_steps", type=int, default=1000,
                        help="扩散总步数")
    parser.add_argument("--beta_schedule", type=str, default="cosine",
                        help="Beta schedule 类型")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备 (cuda / cpu)")
    parser.add_argument("--test_case", type=str, default="",
                        help="测试：对单个 nii.gz 路径执行增强并保存结果")
    parser.add_argument("--test_save_dir", type=str, default="./on_the_fly_test_output",
                        help="测试输出目录")
    args = parser.parse_args()

    # 构建标签池路径列表
    label_paths = []
    if args.label_pool_csv:
        import pandas as pd
        df = pd.read_csv(args.label_pool_csv)
        if args.dataset == "BRATS_2026" and "id" in df.columns:
            non_met = [str(case_id) for case_id in df["id"].astype(str).tolist() if not str(case_id).startswith("BraTS-MET-")]
            if non_met:
                preview = ", ".join(non_met[:5])
                raise ValueError(f"BRATS_2026 on-the-fly augmentation refuses non-MET labels in CSV: {preview}")
        if "label" in df.columns:
            label_paths = df["label"].tolist()
        else:
            raise ValueError(f"CSV 文件中没有 'label' 列: {args.label_pool_csv}")
    elif args.label_pool_dir:
        label_paths = sorted(glob.glob(os.path.join(args.label_pool_dir, "*_seg.nii.gz")))
        if args.dataset == "BRATS_2026":
            label_paths = [p for p in label_paths if "BraTS-MET-" in os.path.basename(p)]
    else:
        raise ValueError("必须提供 --label_pool_csv 或 --label_pool_dir")

    print(f"标签池: {len(label_paths)} 个标签文件")

    # 初始化增强器
    augmenter = OnTheFlyTumourAugmenter(
        diffusion_ckpt_dir=args.diffusion_ckpt_dir,
        label_pool_paths=label_paths,
        dataset_type=args.dataset,
        n_steps=args.n_steps,
        beta_schedule=args.beta_schedule,
        sampling_steps=args.sampling_steps,
        device=args.device,
    )

    # 测试模式：对单个病例执行增强
    if args.test_case:
        import os as _os
        _os.makedirs(args.test_save_dir, exist_ok=True)

        print(f"\n[测试] 加载测试病例: {args.test_case}")
        test_data = nib.load(args.test_case).get_fdata()
        # 假设 4 模态：目录下通常有 t1c, t1, t2, flair .nii.gz 文件
        test_dir = os.path.dirname(args.test_case)
        test_basename = os.path.basename(args.test_case).replace("-t1c.nii.gz", "")
        modalities = ["t1c", "t1n", "t2w", "t2f"]
        data_4ch = []
        for mod in modalities:
            mod_path = os.path.join(test_dir, f"{test_basename}-{mod}.nii.gz")
            if os.path.exists(mod_path):
                data_4ch.append(nib.load(mod_path).get_fdata().astype(np.float32))
            else:
                print(f"  [警告] 找不到 {mod_path}，用 t1c 代替")
                data_4ch.append(test_data.astype(np.float32))

        data_4ch = np.stack(data_4ch, axis=0)  # (4, D, H, W)

        # 构造假 seg（全零，无肿瘤）
        seg = np.zeros((1,) + data_4ch.shape[1:], dtype=np.int16)

        print(f"data_4ch shape: {data_4ch.shape}, seg shape: {seg.shape}")

        rng = np.random.RandomState(42)
        data_out, seg_out, modified = augmenter.augment_sample(data_4ch, seg, rng)
        print(f"增强完成: modified={modified}")

        # 保存结果
        if modified:
            for mod_idx, mod in enumerate(modalities):
                save_path = os.path.join(args.test_save_dir,
                                         f"{test_basename}-{mod}_aug.nii.gz")
                feat = nib.Nifti1Image(data_out[mod_idx], np.eye(4))
                nib.save(feat, save_path)
                print(f"  已保存: {save_path}")
            seg_save = os.path.join(args.test_save_dir,
                                    f"{test_basename}-seg_aug.nii.gz")
            nib.save(nib.Nifti1Image(seg_out[0].astype(np.int16), np.eye(4)), seg_save)
            print(f"  已保存: {seg_save}")
    else:
        print("增强器初始化成功。使用 --test_case 参数测试增强效果。")
        print("或在 nnU-Net trainer 中导入 OnTheFlyTumourAugmenter 类使用。")


if __name__ == "__main__":
    __main__()
