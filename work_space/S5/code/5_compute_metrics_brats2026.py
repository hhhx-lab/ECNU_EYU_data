"""
Metrics computation script for BraTS 2026 Brain Metastases Challenge
Metrics:
- Segmentation: DSC, NSD (only for lesions > 275 mm³)
- Detection: F1, AUC over multiple F1 scores at different detection thresholds
"""
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from monai.utils import set_determinism
import os
import numpy as np
import SimpleITK as sitk
from medpy import metric
import argparse
import yaml
from tqdm import tqdm
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.metrics import auc
set_determinism(123)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SegMamba BraTS 2026 Metrics')
    parser.add_argument('-c', '--config', type=str, default='',
                        help='Path to YAML config file')
    parser.add_argument("--pred_name", type=str, default="")
    parser.add_argument("--nsd_threshold", type=float, default=1.0)
    parser.add_argument("--lesion_volume_threshold", type=float, default=275.0)
    return parser.parse_known_args()


def load_config(config_path):
    """从 YAML 文件加载配置"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from: {config_path}")
        return config or {}
    return {}


args, _ = parse_args()
config = load_config(args.config) if args.config else {}

# 从 config 文件名提取实验标识
config_name = ""
if args.config:
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    print(f"Config name: {config_name}")

# pred_name 默认从 config_name 派生
if args.pred_name:
    pred_name = args.pred_name
elif config_name:
    pred_name = f"segmamba_brats2026_{config_name}"
else:
    pred_name = "segmamba_brats2026_default"

LABEL_NETC = 1
LABEL_SNFH = 2
LABEL_ET   = 3
LABEL_RC   = 4

VOXEL_SPACING = [1.0, 1.0, 1.0]
LESION_VOLUME_THRESHOLD = 275.0

# ✅ 多阈值：overlap比例从0.1到0.9，共9个阈值
DETECTION_OVERLAP_THRESHOLDS = np.arange(0.1, 1.0, 0.1)


# ------------------------------------------------------------------
# 标签转换
# ------------------------------------------------------------------

def convert_labels_to_multilabel(labels):
    return {
        'netc':   (labels == LABEL_NETC).astype(np.int32),
        'snfh':   (labels == LABEL_SNFH).astype(np.int32),
        'et':     (labels == LABEL_ET).astype(np.int32),
        'rc':     (labels == LABEL_RC).astype(np.int32),
        'lesion': ((labels == LABEL_ET) |
                   (labels == LABEL_NETC) |
                   (labels == LABEL_RC)).astype(np.int32),
    }


# ------------------------------------------------------------------
# 连通域工具
# ------------------------------------------------------------------

def get_connected_components(volume):
    labeled_array, num_features = ndimage.label(volume)
    return labeled_array, num_features


def get_lesion_volumes(volume, voxel_spacing=VOXEL_SPACING):
    labeled_array, num_features = get_connected_components(volume)
    voxel_vol = np.prod(voxel_spacing)
    lesions = []
    for i in range(1, num_features + 1):
        voxel_count = np.sum(labeled_array == i)
        lesions.append((i, voxel_count * voxel_vol))
    return labeled_array, lesions


# ------------------------------------------------------------------
# 分割指标
# ------------------------------------------------------------------

def cal_dice_hd95(gt, pred, voxel_spacing=VOXEL_SPACING):
    if pred.sum() > 0 and gt.sum() > 0:
        d = metric.binary.dc(pred, gt)
        try:
            hd = metric.binary.hd95(pred, gt, voxelspacing=voxel_spacing)
        except Exception:
            hd = np.inf
        return d, hd
    elif gt.sum() == 0 and pred.sum() == 0:
        return 1.0, 0.0
    else:
        return 0.0, np.inf


def get_surface_points(binary_mask, voxel_spacing=VOXEL_SPACING):
    if binary_mask.sum() == 0:
        return np.array([]).reshape(0, 3)
    eroded = ndimage.binary_erosion(binary_mask)
    boundary = binary_mask.astype(int) - eroded.astype(int)
    points = np.array(np.where(boundary > 0)).T.astype(float)
    points *= np.array(voxel_spacing)
    return points


def cal_nsd(gt, pred, voxel_spacing=VOXEL_SPACING, tolerance=1.0):
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    try:
        pred_surface = get_surface_points(pred, voxel_spacing)
        gt_surface   = get_surface_points(gt,   voxel_spacing)
        if len(pred_surface) == 0 or len(gt_surface) == 0:
            return 0.0
        pred_to_gt = cdist(pred_surface, gt_surface).min(axis=1)
        gt_to_pred = cdist(gt_surface, pred_surface).min(axis=1)
        within_pred = np.sum(pred_to_gt <= tolerance)
        within_gt   = np.sum(gt_to_pred <= tolerance)
        return (within_pred + within_gt) / (len(pred_surface) + len(gt_surface))
    except Exception:
        return 0.0


def each_case_segmentation_metric(gt_array, pred_array,
                                   voxel_spacing=VOXEL_SPACING,
                                   nsd_threshold=1.0,
                                   volume_threshold=LESION_VOLUME_THRESHOLD):
    gt_ml   = convert_labels_to_multilabel(gt_array)
    pred_ml = convert_labels_to_multilabel(pred_array)

    class_names = ['netc', 'snfh', 'et', 'rc']
    metrics = np.zeros((4, 3))

    for i, name in enumerate(class_names):
        gt   = gt_ml[name]
        pred = pred_ml[name]

        gt_vol = np.sum(gt) * np.prod(voxel_spacing)
        if gt_vol < volume_threshold and gt_vol > 0:
            metrics[i] = [np.nan, np.nan, np.nan]
            continue

        d, hd = cal_dice_hd95(gt, pred, voxel_spacing)
        nsd   = cal_nsd(gt, pred, voxel_spacing, tolerance=nsd_threshold)
        metrics[i] = [d, hd, nsd]

    return metrics


# ------------------------------------------------------------------
# 检测指标
# ✅ 新增：基于overlap比例阈值的检测，用于计算AUC
# ------------------------------------------------------------------

def calculate_detection_metrics_at_threshold(gt_lesion, pred_lesion,
                                              overlap_threshold=0.1,
                                              volume_threshold=LESION_VOLUME_THRESHOLD):
    """
    在指定overlap比例阈值下计算检测指标。

    匹配条件：
    - 病灶体积 >= volume_threshold mm³
    - pred与gt的IoU >= overlap_threshold
    """
    gt_labeled,   gt_lesions   = get_lesion_volumes(gt_lesion)
    pred_labeled, pred_lesions = get_lesion_volumes(pred_lesion)

    gt_lesions   = [(i, v) for i, v in gt_lesions   if v >= volume_threshold]
    pred_lesions = [(i, v) for i, v in pred_lesions if v >= volume_threshold]

    matched_gt = set()
    tp, fp = 0, 0

    for pred_id, pred_vol in pred_lesions:
        pred_mask = (pred_labeled == pred_id)
        best_iou = 0.0
        best_gt_id = None

        for gt_id, gt_vol in gt_lesions:
            if gt_id in matched_gt:
                continue
            gt_mask = (gt_labeled == gt_id)

            intersection = np.sum(pred_mask & gt_mask)
            union = np.sum(pred_mask | gt_mask)
            iou = intersection / union if union > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id

        # ✅ 用IoU阈值判断是否匹配成功
        if best_gt_id is not None and best_iou >= overlap_threshold:
            tp += 1
            matched_gt.add(best_gt_id)
        else:
            fp += 1

    fn = len(gt_lesions) - len(matched_gt)
    return tp, fp, fn


def calculate_f1_score(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


def calculate_auc_over_f1(all_detection_metrics_per_threshold):
    """
    ✅ 计算多阈值F1曲线的AUC。

    输入: dict {threshold: [(tp, fp, fn), ...]}
    输出: AUC值, 每个阈值对应的F1列表
    """
    thresholds = sorted(all_detection_metrics_per_threshold.keys())
    f1_scores = []

    for thresh in thresholds:
        cases = all_detection_metrics_per_threshold[thresh]
        total_tp = sum(x[0] for x in cases)
        total_fp = sum(x[1] for x in cases)
        total_fn = sum(x[2] for x in cases)
        f1, _, _ = calculate_f1_score(total_tp, total_fp, total_fn)
        f1_scores.append(f1)

    # 用梯形法则计算AUC，x轴是阈值，y轴是F1
    auc_score = auc(thresholds, f1_scores)

    # 归一化到[0,1]（除以阈值范围）
    auc_score_normalized = auc_score / (thresholds[-1] - thresholds[0])

    return auc_score_normalized, dict(zip(thresholds, f1_scores))


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------

def main():
    data_dir     = "./data/fullres/train"
    raw_data_dir = "./data/raw_data/MICCAI-LH-BraTS2025-MET-Challenge-TrainingData/"
    results_root = "prediction_results"

    _, _, test_ds = get_train_val_test_loader_from_train(data_dir)
    print(f"Number of test cases: {len(test_ds)}")

    num_cases   = len(test_ds)
    num_classes = 4

    all_seg_metrics = np.full((num_cases, num_classes, 3), np.nan)

    # ✅ 每个阈值单独存一组检测结果
    all_detection_per_threshold = {
        t: [] for t in DETECTION_OVERLAP_THRESHOLDS
    }

    for ind, batch in enumerate(tqdm(test_ds, total=num_cases)):
        properties = batch["properties"]
        case_name  = properties["name"]

        gt_itk   = sitk.ReadImage(os.path.join(raw_data_dir, case_name, "seg.nii.gz"))
        gt_array = sitk.GetArrayFromImage(gt_itk).astype(np.int32)

        pred_itk   = sitk.ReadImage(f"./{results_root}/{pred_name}/{case_name}.nii.gz")
        pred_array = sitk.GetArrayFromImage(pred_itk).astype(np.int32)

        # 分割指标
        seg_metrics = each_case_segmentation_metric(
            gt_array, pred_array,
            voxel_spacing=VOXEL_SPACING,
            nsd_threshold=args.nsd_threshold,
            volume_threshold=args.lesion_volume_threshold,
        )
        all_seg_metrics[ind] = seg_metrics

        # ✅ 在每个overlap阈值下计算检测指标
        gt_ml   = convert_labels_to_multilabel(gt_array)
        pred_ml = convert_labels_to_multilabel(pred_array)

        for thresh in DETECTION_OVERLAP_THRESHOLDS:
            tp, fp, fn = calculate_detection_metrics_at_threshold(
                gt_ml['lesion'], pred_ml['lesion'],
                overlap_threshold=thresh,
                volume_threshold=args.lesion_volume_threshold,
            )
            all_detection_per_threshold[thresh].append((tp, fp, fn))

    # ------------------------------------------------------------------
    # 输出分割结果
    # ------------------------------------------------------------------
    result_metrics_dir = f"./{results_root}/result_metrics_{config_name}"
    os.makedirs(result_metrics_dir, exist_ok=True)
    np.save(f"{result_metrics_dir}/{pred_name}_segmentation.npy", all_seg_metrics)

    print("\n" + "=" * 60)
    print("BraTS 2026 Segmentation Metrics (DSC / HD95 / NSD)")
    print(f"(Lesions < {args.lesion_volume_threshold}mm³ excluded)")
    print("=" * 60)

    class_names = ['NETC', 'SNFH', 'ET', 'RC']
    mean_dice = np.nanmean(all_seg_metrics[:, :, 0], axis=0)
    mean_hd95 = np.nanmean(all_seg_metrics[:, :, 1], axis=0)
    mean_nsd  = np.nanmean(all_seg_metrics[:, :, 2], axis=0)

    print(f"{'Class':>6}  {'DSC':>8}  {'HD95(mm)':>10}  {'NSD':>8}")
    print("-" * 40)
    for i, name in enumerate(class_names):
        print(f"{name:>6}  {mean_dice[i]:>8.4f}  {mean_hd95[i]:>10.2f}  {mean_nsd[i]:>8.4f}")
    print("-" * 40)
    print(f"{'Mean':>6}  {np.nanmean(mean_dice):>8.4f}  "
          f"{np.nanmean(mean_hd95):>10.2f}  {np.nanmean(mean_nsd):>8.4f}")

    # ------------------------------------------------------------------
    # ✅ 输出检测结果：多阈值F1 + AUC
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BraTS 2026 Detection Metrics")
    print(f"(Volume threshold: {args.lesion_volume_threshold}mm³)")
    print("=" * 60)

    auc_score, f1_per_threshold = calculate_auc_over_f1(all_detection_per_threshold)

    print(f"\n{'Threshold':>12}  {'F1':>8}  {'Precision':>10}  {'Recall':>8}")
    print("-" * 45)
    for thresh in sorted(f1_per_threshold.keys()):
        cases = all_detection_per_threshold[thresh]
        total_tp = sum(x[0] for x in cases)
        total_fp = sum(x[1] for x in cases)
        total_fn = sum(x[2] for x in cases)
        f1, precision, recall = calculate_f1_score(total_tp, total_fp, total_fn)
        print(f"{thresh:>12.1f}  {f1:>8.4f}  {precision:>10.4f}  {recall:>8.4f}")

    print("-" * 45)
    print(f"\nAUC over F1 scores: {auc_score:.4f}")

    np.save(f"{result_metrics_dir}/{pred_name}_detection.npy",
            np.array([(t, f1_per_threshold[t]) for t in sorted(f1_per_threshold.keys())]))

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Mean DSC        : {np.nanmean(mean_dice):.4f}")
    print(f"Mean NSD        : {np.nanmean(mean_nsd):.4f}")
    print(f"F1 (IoU=0.1)    : {f1_per_threshold[0.1]:.4f}")
    print(f"AUC over F1     : {auc_score:.4f}")


if __name__ == "__main__":
    main()