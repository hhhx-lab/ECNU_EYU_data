"""
借入标签修改器 — 实现 BraTS 2025 的 "Adapt Tumor Classes" + "Scale" 两步操作。

对从训练集中借入的其他病人的真实标签进行修改，使其多样化，避免过拟合。

BraTS 标签值定义:
    - 0 = 背景 (Background)
    - 1 = NETC (Necrotic and non-enhancing tumor core / 坏死和非增强肿瘤核心)
    - 2 = SNFH (Edema, infiltrating tumor, post-treatment changes / 水肿)
    - 3 = ET  (Active/enhancing tumor / 增强肿瘤)
    - 4 = RC  (Resection cavity / 切除腔，仅 BraTS 2024)

修改流程（对应 BraTS 2025 论文第 4 页）:
    Step 3 (Adapt Tumor Classes):
        - 以 70% 概率将 SNFH(2) 改为 ET(3)
        - 以 70% 概率将原始 ET(3) 改为 NETC(1)
        - 注意：SNFH 改来的 ET 不参与第二步的 ET→NETC 转换
    Step 4 (Scale):
        - 如果 SNFH 被移除: 缩放系数 ∈ [0.1, 0.3]
        - 如果 SNFH 保留:   缩放系数 ∈ [0.3, 0.8]
        - 以肿瘤中心为锚点，对各维度按缩放系数缩小
"""

import numpy as np
from scipy.ndimage import zoom as ndimage_zoom


def modify_borrowed_label(label, rng=None):
    """
    对借入的真实标签执行 BraTS 2025 的 Adapt Tumor Classes + Scale。

    参数:
        label: np.ndarray, 整型标签, 形状 (H, W, D)
               值域 {0, 1, 2, 3, 4}
               1=NETC, 2=SNFH, 3=ET, 4=RC
        rng:   numpy.random.Generator 或 None，用于可复现的随机性

    返回:
        modified_label: np.ndarray, 修改后的整型标签（与输入形状相同）
        meta: dict, 包含以下字段:
              - "snfh_removed": bool, SNFH 是否被移除（用于决定缩放系数）
              - "scale_factor": float, 缩放系数
              - "snfh_to_et": bool, SNFH→ET 是否实际发生
              - "et_to_netc": bool, ET→NETC 是否实际发生
    """
    if rng is None:
        rng = np.random

    # 深拷贝，避免修改原始标签
    modified = np.copy(label)

    # ---- Step 3: Adapt Tumor Classes ----

    # 3a: 先记录"原始"的 ET 位置（在 SNFH 可能变为 ET 之前）
    # 因为我们要确保 SNFH 改来的 ET 不参与后面的 ET→NETC 转换
    is_et_original = (modified == 3)

    # 3b: 以 70% 概率将 SNFH(2) 改为 ET(3)
    snfh_to_et = rng.uniform() < 0.7
    snfh_removed = False  # 默认 SNFH 未被移除
    if snfh_to_et:
        is_snfh = (modified == 2)
        if np.any(is_snfh):
            modified[is_snfh] = 3  # SNFH → ET
            snfh_removed = True   # SNFH 被移除，影响后续缩放系数

    # 3c: 以 70% 概率将"原始 ET"(3) 改为 NETC(1)
    # 使用步骤 3a 记录的原始 ET 掩码，避免改变步骤 3b 中 SNFH 新转来的 ET
    et_to_netc = rng.uniform() < 0.7
    if et_to_netc:
        if np.any(is_et_original):
            modified[is_et_original] = 1  # 原始 ET → NETC

    # ---- Step 4: Scale (差分缩放) ----
    # 根据 SNFH 是否被移除，选择不同的缩放系数范围
    if snfh_removed:
        scale_factor = rng.uniform(0.1, 0.3)
    else:
        scale_factor = rng.uniform(0.3, 0.8)

    # 对非零区域进行缩放：以肿瘤中心为锚点，按 scale_factor 缩小各维度
    non_zero_mask = (modified != 0)
    if np.any(non_zero_mask):
        modified = _scale_label_region(modified, scale_factor, non_zero_mask)

    meta = {
        "snfh_removed": snfh_removed,
        "scale_factor": scale_factor,
        "snfh_to_et": snfh_to_et,
        "et_to_netc": et_to_netc,
    }
    return modified, meta


def _scale_label_region(label, scale_factor, mask=None):
    """
    对标签中的非零区域做中心缩放。

    将非零区域裁剪出来，按 scale_factor 缩小（各维度等比缩放），
    然后放回原始空间中的相同中心位置。

    参数:
        label:       整型标签 (H, W, D)
        scale_factor: 缩放系数，0.3 表示缩小到 30% 大小
        mask:         非零区域的布尔掩码（可选，如果不提供则计算 label != 0）

    返回:
        np.ndarray: 缩放后的标签（非零区域被缩小，其余为 0）
    """
    if mask is None:
        mask = (label != 0)

    if not np.any(mask):
        return label  # 没有肿瘤区域，直接返回

    # 找到非零区域的 bounding box
    coords = np.where(mask)
    z_min, z_max = coords[0].min(), coords[0].max() + 1
    y_min, y_max = coords[1].min(), coords[1].max() + 1
    x_min, x_max = coords[2].min(), coords[2].max() + 1

    # 裁剪出包含肿瘤的子体积
    crop = label[z_min:z_max, y_min:y_max, x_min:x_max].astype(np.float32)
    crop_shape = np.array(crop.shape)

    # 按 scale_factor 缩放各维度
    # 至少保留 1 个体素，避免完全消失
    new_shape = np.maximum(np.round(crop_shape * scale_factor).astype(int), 1)

    # 使用 scipy.ndimage.zoom 进行缩放
    # zoom 的缩放因子是 new_shape / old_shape
    zoom_factors = new_shape / crop_shape
    zoomed = ndimage_zoom(crop, zoom_factors, order=0)  # order=0 = 最近邻插值（保持标签为整数）

    # 将缩放后的肿瘤放回原 bounding box 的中心位置
    result = np.zeros_like(label)
    z_center, y_center, x_center = (
        (z_min + z_max) // 2,
        (y_min + y_max) // 2,
        (x_min + x_max) // 2,
    )

    z_start = z_center - zoomed.shape[0] // 2
    y_start = y_center - zoomed.shape[1] // 2
    x_start = x_center - zoomed.shape[2] // 2

    z_end = z_start + zoomed.shape[0]
    y_end = y_start + zoomed.shape[1]
    x_end = x_start + zoomed.shape[2]

    # 确保不越界
    z_start = max(0, z_start)
    y_start = max(0, y_start)
    x_start = max(0, x_start)
    z_end = min(label.shape[0], z_end)
    y_end = min(label.shape[1], y_end)
    x_end = min(label.shape[2], x_end)

    # 裁剪 zoomed 以匹配可用空间
    zoomed_z_start = max(0, -(z_center - zoomed.shape[0] // 2))
    zoomed_y_start = max(0, -(y_center - zoomed.shape[1] // 2))
    zoomed_x_start = max(0, -(x_center - zoomed.shape[2] // 2))
    zoomed_z_end = zoomed_z_start + (z_end - z_start)
    zoomed_y_end = zoomed_y_start + (y_end - y_start)
    zoomed_x_end = zoomed_x_start + (x_end - x_start)

    # 四舍五入恢复为整型标签
    result[z_start:z_end, y_start:y_end, x_start:x_end] = np.round(
        zoomed[zoomed_z_start:zoomed_z_end,
               zoomed_y_start:zoomed_y_end,
               zoomed_x_start:zoomed_x_end]
    ).astype(np.int16)

    return result


# ---- 测试代码 ----
if __name__ == "__main__":
    """快速测试标签修改逻辑"""
    rng = np.random.RandomState(42)

    # 构造一个假的借入标签：肿瘤在中心区域
    fake_label = np.zeros((96, 96, 96), dtype=np.int16)
    # SNFH 区域（水肿，标签=2）
    fake_label[30:66, 30:66, 30:66] = 2
    # ET 核心（增强肿瘤，标签=3）
    fake_label[40:56, 40:56, 40:56] = 3
    # NETC 坏死核心（标签=1）
    fake_label[44:52, 44:52, 44:52] = 1

    print(f"原始标签: SNFH={np.sum(fake_label==2)}, ET={np.sum(fake_label==3)}, NETC={np.sum(fake_label==1)}")

    modified, meta = modify_borrowed_label(fake_label, rng)

    print(f"修改后标签: SNFH={np.sum(modified==2)}, ET={np.sum(modified==3)}, NETC={np.sum(modified==1)}")
    print(f"元信息: snfh_removed={meta['snfh_removed']}, scale_factor={meta['scale_factor']:.3f}")
    print(f"        snfh_to_et={meta['snfh_to_et']}, et_to_netc={meta['et_to_netc']}")
    print("测试通过！")
