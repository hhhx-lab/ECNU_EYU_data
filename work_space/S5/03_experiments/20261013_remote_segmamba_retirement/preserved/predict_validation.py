"""
BraTS 2026 Validation Set Prediction
=====================================
对官方验证集 (data/validation/) 运行预测, 支持选择不同实验的最佳模型。

用法:
  python predict_validation.py --exp exp_dice_focal
  python predict_validation.py --exp exp_dice_focal,exp_ce_dice300epoch
  python predict_validation.py --all           # 所有实验
"""
import argparse
import os
import sys
import glob
import numpy as np
import torch
import yaml
import SimpleITK as sitk
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism

set_determinism(123)

# 数据路径
VALIDATION_RAW = "./data/validation"
VALIDATION_PREPROCESSED = "./data/validation_preprocessed"
RESULTS_DIR = "./results"

# 所有可用实验 (自动发现 results/ 下含 model/best_model_*.pt 的实验)
def discover_experiments():
    exps = []
    for model_path in sorted(glob.glob(f"{RESULTS_DIR}/*/model/best_model_*.pt")):
        exp = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
        if exp not in exps:
            exps.append(exp)
    return exps


def rename_validation_files():
    """重命名验证集文件: BraTS-MET-XXXXX-XXX-t1n.nii.gz → t1n.nii.gz"""
    if not os.path.isdir(VALIDATION_RAW):
        print(f"ERROR: 验证集目录不存在: {VALIDATION_RAW}")
        sys.exit(1)
    for case_name in sorted(os.listdir(VALIDATION_RAW)):
        case_dir = os.path.join(VALIDATION_RAW, case_name)
        if not os.path.isdir(case_dir):
            continue
        for fname in os.listdir(case_dir):
            if "-" not in fname:
                continue
            new_name = fname.split("-")[-1]
            old_path = os.path.join(case_dir, fname)
            new_path = os.path.join(case_dir, new_name)
            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                print(f"  重命名: {fname} → {new_name}")


def preprocess_validation():
    """对验证集做预处理 (1mm 重采样, 无 seg)."""
    from light_training.preprocessing.preprocessors.preprocessor_mri import MultiModalityPreprocessor

    if not os.path.isdir(VALIDATION_RAW):
        print(f"ERROR: 验证集目录不存在: {VALIDATION_RAW}")
        sys.exit(1)

    preprocessor = MultiModalityPreprocessor(
        base_dir=os.path.dirname(VALIDATION_RAW) + "/",
        image_dir=os.path.basename(VALIDATION_RAW),
        data_filenames=["t2w.nii.gz", "t2f.nii.gz", "t1n.nii.gz", "t1c.nii.gz"],
        seg_filename="",   # 验证集无标注
    )

    os.makedirs(VALIDATION_PREPROCESSED, exist_ok=True)

    preprocessor.run(
        output_spacing=[1.0, 1.0, 1.0],
        output_dir=VALIDATION_PREPROCESSED,
        all_labels=[1, 2, 3, 4],
    )
    count = len(glob.glob(f"{VALIDATION_PREPROCESSED}/*.npz"))
    print(f"验证集预处理完成: {count} 个 npz 文件")
    return count


def predict_experiment(exp_name, device="cuda:0"):
    """用指定实验的最佳模型预测验证集."""
    from light_training.dataloading.dataset import get_test_loader_from_test
    from light_training.prediction import Predictor
    from model_segmamba.segmamba import SegMamba

    # 找模型
    model_dir = f"{RESULTS_DIR}/{exp_name}/model/"
    candidates = sorted(glob.glob(f"{model_dir}/best_model_*.pt"))
    if not candidates:
        print(f"ERROR: 找不到模型 {model_dir}/best_model_*.pt")
        return None
    model_path = candidates[-1]
    print(f"\n{'='*60}")
    print(f"实验: {exp_name}")
    print(f"模型: {model_path}")
    print(f"{'='*60}")

    # 加载模型
    model = SegMamba(in_chans=4, out_chans=5, depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384])
    sd = torch.load(model_path, map_location="cpu")
    if "module" in sd:
        sd = sd["module"]
    new_sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(new_sd)
    model.to(device)
    model.eval()

    # 推理器
    window_infer = SlidingWindowInferer(
        roi_size=[128, 128, 128],
        sw_batch_size=2,
        overlap=0.5,
        progress=True,
        mode="gaussian",
    )
    predictor = Predictor(window_infer=window_infer, mirror_axes=[0, 1, 2])

    # 保存目录
    save_dir = f"{RESULTS_DIR}/{exp_name}/validation_predictions"
    os.makedirs(save_dir, exist_ok=True)

    # 数据加载
    ds = get_test_loader_from_test(VALIDATION_PREPROCESSED)
    from monai.data import DataLoader
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

    def _unwrap(val):
        """DataLoader collate 可能把标量包在 list 里, 安全解包."""
        if isinstance(val, (list, tuple)) and len(val) == 1 and not isinstance(val[0], (list, tuple, np.ndarray)):
            return val[0]
        return val

    count = 0
    for batch in loader:
        image = batch["data"].to(device)
        # 安全解包 properties (collate 可能包裹了标量值)
        raw_props = batch["properties"]
        properties = {}
        for k, v in raw_props.items():
            properties[k] = _unwrap(v)

        with torch.no_grad():
            with torch.autocast("cuda", enabled=True):
                pred = predictor.maybe_mirror_and_predict(image, model, device=device)

        # Resample 回原始间距
        pred = predictor.predict_raw_probability(pred, properties=properties)
        pred = pred.argmax(dim=0).cpu().numpy().astype(np.uint8)

        # 还原裁剪
        pred = predictor.predict_noncrop_probability(pred, properties)

        # 方向 & origin
        case_name = str(_unwrap(raw_props['name']))

        raw_spacing = list(properties['spacing'])

        # 从原始验证集文件读方向信息
        direction = properties.get('direction', None)
        origin = properties.get('origin', None)
        if direction is None or origin is None:
            ref_path = os.path.join(VALIDATION_RAW, case_name, "t1c.nii.gz")
            if os.path.exists(ref_path):
                ref = sitk.ReadImage(ref_path)
                if direction is None:
                    direction = ref.GetDirection()
                if origin is None:
                    origin = ref.GetOrigin()

        # 确保 direction/origin 是 SimpleITK 接受的 flat list
        if direction is not None:
            direction = [float(x) for x in direction]
        if origin is not None:
            origin = [float(x) for x in origin]

        predictor.save_to_nii(
            pred, raw_spacing=raw_spacing,
            direction=direction, origin=origin,
            case_name=case_name, save_dir=save_dir,
        )
        count += 1

    print(f"✓ {exp_name}: 预测完成, {count} 个文件 → {save_dir}")
    return save_dir


def parse_args():
    p = argparse.ArgumentParser(description="BraTS 2026 Validation Set Prediction")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--exp", type=str, default="",
                   help="实验名 (逗号分隔多个), 如 exp_dice_focal,exp_ce_dice300epoch")
    g.add_argument("--all", action="store_true", help="所有可用实验")
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Step 1: 重命名 (如需要)
    print("[1/3] 重命名验证集文件...")
    rename_validation_files()

    # Step 2: 预处理 (已有 npz 则跳过)
    existing_npz = glob.glob(f"{VALIDATION_PREPROCESSED}/*.npz")
    if existing_npz:
        print(f"\n[2/3] 预处理已有 {len(existing_npz)} 个 npz 文件, 跳过")
    else:
        print("\n[2/3] 预处理验证集 (1mm 重采样)...")
        n = preprocess_validation()
        if n == 0:
            print("ERROR: 预处理未生成文件")
            sys.exit(1)

    # Step 3: 确定实验列表
    if args.all:
        experiments = discover_experiments()
    else:
        experiments = [e.strip() for e in args.exp.split(",") if e.strip()]

    if not experiments:
        print("ERROR: 没有选择实验")
        sys.exit(1)

    print(f"\n[3/3] 预测: {len(experiments)} 个实验")
    for exp in experiments:
        predict_experiment(exp, device=args.device)

    print(f"\n{'='*60}")
    print("全部完成! 输出目录:")
    for exp in experiments:
        print(f"  results/{exp}/validation_predictions/")
