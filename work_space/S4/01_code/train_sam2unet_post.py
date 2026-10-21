import argparse
import csv
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# PLAN2 CHANGE: import the two-head model constants so post training stays
# aligned with sam2unet_model.py instead of assuming a single 5-channel head.
from sam2unet_model import BRATS_MET_NUM_CLASSES, SAM2UNet3D


# ============================================================
# Constants
# ============================================================

# PLAN2 CHANGE: final reporting is limited to the four foreground regions and
# their average Dice; no segmentation NIfTI prediction images are written.
CLASS_NAMES = ["NETC", "SNFH", "ET", "RC"]
MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
MAIN_IGNORE_INDEX = 4
RC_LABEL = 4


# ============================================================
# Runtime Configuration
# ============================================================

def parse_triplet(value, name):
    """Parse comma-separated 3D sizes such as 96,96,96."""
    try:
        parsed = tuple(int(part.strip()) for part in value.split(","))
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"{name} must look like 96,96,96") from exc
    if len(parsed) != 3 or any(v <= 0 for v in parsed):
        raise argparse.ArgumentTypeError(f"{name} must contain three positive integers")
    return parsed


def parse_float_list(value):
    """Parse comma-separated RC thresholds."""
    try:
        parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    except Exception as exc:
        raise argparse.ArgumentTypeError("threshold list must look like 0.15,0.20,0.30") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("threshold list must not be empty")
    return parsed


def get_env_int(name, default):
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def get_env_float(name, default):
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def build_arg_parser():
    # PLAN2 CHANGE: all server-specific paths and main run knobs come from CLI
    # args or environment variables; no local TrainingData path is hard-coded.
    parser = argparse.ArgumentParser(
        description="Train SAM2-UNet Plan2 with a 4-class main head and binary RC head."
    )
    parser.add_argument("--train_dir", default=os.environ.get("SAM2UNET_TRAIN_DIR"))
    parser.add_argument(
        "--fixed_split_root",
        default=os.environ.get("SAM2UNET_FIXED_SPLIT_ROOT"),
        help="G2 case-folder root containing train/val/test subdirectories.",
    )
    parser.add_argument("--save_dir", default=os.environ.get("SAM2UNET_SAVE_DIR"))
    parser.add_argument("--epochs", type=int, default=get_env_int("SAM2UNET_EPOCHS", 400))
    parser.add_argument("--crop_size", type=lambda s: parse_triplet(s, "crop_size"),
                        default=parse_triplet(os.environ.get("SAM2UNET_CROP_SIZE", "96,96,96"), "crop_size"))
    parser.add_argument("--batch_size", type=int, default=get_env_int("SAM2UNET_BATCH_SIZE", 1))
    parser.add_argument("--accumulation_steps", type=int,
                        default=get_env_int("SAM2UNET_ACCUMULATION_STEPS", 1))
    parser.add_argument("--num_workers", type=int, default=get_env_int("SAM2UNET_NUM_WORKERS", 4))
    parser.add_argument("--prefetch_factor", type=int, default=get_env_int("SAM2UNET_PREFETCH_FACTOR", 2))
    parser.add_argument("--checkpoint_interval", type=int,
                        default=get_env_int("SAM2UNET_CHECKPOINT_INTERVAL", 10))
    parser.add_argument("--debug_case_limit", type=int,
                        default=os.environ.get("SAM2UNET_DEBUG_CASE_LIMIT"))

    parser.add_argument("--split_ratio", type=float, default=get_env_float("SAM2UNET_SPLIT_RATIO", 0.8))
    parser.add_argument("--split_seed", type=int, default=get_env_int("SAM2UNET_SPLIT_SEED", 2025))
    parser.add_argument("--warmup_epochs", type=int, default=get_env_int("SAM2UNET_WARMUP_EPOCHS", 30))
    parser.add_argument("--rc_thresholds", type=parse_float_list,
                        default=parse_float_list(os.environ.get(
                            "SAM2UNET_RC_THRESHOLDS", "0.15,0.20,0.25,0.30,0.35,0.40,0.50"
                        )))

    parser.add_argument("--lr", type=float, default=get_env_float("SAM2UNET_LR", 1e-4))
    parser.add_argument("--main_head_lr", type=float, default=get_env_float("SAM2UNET_MAIN_HEAD_LR", 1e-4))
    parser.add_argument("--rc_head_lr", type=float, default=get_env_float("SAM2UNET_RC_HEAD_LR", 3e-4))
    parser.add_argument("--weight_decay", type=float, default=get_env_float("SAM2UNET_WEIGHT_DECAY", 1e-5))

    parser.add_argument("--main_loss_weight", type=float,
                        default=get_env_float("SAM2UNET_MAIN_LOSS_WEIGHT", 1.0))
    parser.add_argument("--rc_phase_main_loss_weight", type=float,
                        default=get_env_float("SAM2UNET_RC_PHASE_MAIN_LOSS_WEIGHT", 0.3))
    parser.add_argument("--rc_loss_weight", type=float,
                        default=get_env_float("SAM2UNET_RC_LOSS_WEIGHT", 4.0))
    parser.add_argument("--rc_focal_alpha", type=float,
                        default=get_env_float("SAM2UNET_RC_FOCAL_ALPHA", 0.75))
    parser.add_argument("--rc_focal_gamma", type=float,
                        default=get_env_float("SAM2UNET_RC_FOCAL_GAMMA", 2.0))
    parser.add_argument("--rc_tversky_alpha", type=float,
                        default=get_env_float("SAM2UNET_RC_TVERSKY_ALPHA", 0.3))
    parser.add_argument("--rc_tversky_beta", type=float,
                        default=get_env_float("SAM2UNET_RC_TVERSKY_BETA", 0.7))

    parser.add_argument("--feature_size", type=int, default=get_env_int("SAM2UNET_FEATURE_SIZE", 48))
    parser.add_argument("--depths", type=int, default=get_env_int("SAM2UNET_DEPTHS", 4))
    parser.add_argument("--num_heads", type=int, default=get_env_int("SAM2UNET_NUM_HEADS", 4))
    parser.add_argument("--window_size", type=lambda s: parse_triplet(s, "window_size"),
                        default=parse_triplet(os.environ.get("SAM2UNET_WINDOW_SIZE", "4,4,4"), "window_size"))
    parser.add_argument("--dropout_rate", type=float, default=get_env_float("SAM2UNET_DROPOUT_RATE", 0.2))
    parser.add_argument("--no_attention", action="store_true")

    parser.add_argument("--sliding_window_batch_size", type=int,
                        default=get_env_int("SAM2UNET_SW_BATCH_SIZE", 1))
    parser.add_argument("--sliding_window_overlap", type=float,
                        default=get_env_float("SAM2UNET_SW_OVERLAP", 0.5))
    parser.add_argument("--no_resume", action="store_true")
    return parser


def build_config(args):
    if not args.train_dir and not args.fixed_split_root:
        raise ValueError(
            "Missing data input. Pass --fixed_split_root/SAM2UNET_FIXED_SPLIT_ROOT "
            "for formal G2 runs, or --train_dir/SAM2UNET_TRAIN_DIR for exploratory runs."
        )
    if not args.save_dir:
        raise ValueError("Missing save_dir. Pass --save_dir or set SAM2UNET_SAVE_DIR.")

    fixed_split_root = Path(args.fixed_split_root).resolve() if args.fixed_split_root else None
    if fixed_split_root is not None:
        missing = [name for name in ("train", "val", "test") if not (fixed_split_root / name).is_dir()]
        if missing:
            raise ValueError(f"fixed_split_root is missing split directories {missing}: {fixed_split_root}")
        train_dir = fixed_split_root
    else:
        train_dir = Path(args.train_dir).resolve()
        if not train_dir.exists():
            raise ValueError(f"train_dir does not exist: {train_dir}")

    config = vars(args).copy()
    config["train_dir"] = str(train_dir)
    config["fixed_split_root"] = str(fixed_split_root) if fixed_split_root else ""
    config["save_dir"] = str(Path(args.save_dir))
    config["debug_case_limit"] = int(args.debug_case_limit) if args.debug_case_limit else None
    config["use_attention"] = not args.no_attention
    config["resume"] = not args.no_resume
    config["in_channels"] = len(MODALITIES)
    config["out_channels"] = BRATS_MET_NUM_CLASSES
    config["class_names"] = CLASS_NAMES
    return config


# ============================================================
# Data Discovery, Statistics, And RC-Stratified Split
# ============================================================

def find_case_dirs(train_dir, limit=None):
    """Find case folders containing the four modalities and segmentation label."""
    train_dir = Path(train_dir)
    case_dirs = []
    skipped = []
    for case_dir in sorted(path for path in train_dir.iterdir() if path.is_dir()):
        required = [case_dir / f"{case_dir.name}-{modality}.nii.gz" for modality in MODALITIES]
        required.append(case_dir / f"{case_dir.name}-seg.nii.gz")
        missing = [path.name for path in required if not path.exists()]
        if missing:
            skipped.append({"case": case_dir.name, "missing": missing})
            continue
        case_dirs.append(case_dir)
        if limit is not None and len(case_dirs) >= limit:
            break

    if skipped:
        print(f"Skipped {len(skipped)} incomplete case directories.")
    if not case_dirs:
        raise ValueError(
            f"No valid cases found in {train_dir}. Expected each case to contain "
            "{case}-t1n/t1c/t2w/t2f/seg.nii.gz."
        )
    return case_dirs


def scan_label_statistics(train_dir, save_dir, limit=None):
    # PLAN2 CHANGE: scan labels before training, record RC-positive cases, and
    # write the exact statistics files requested by plan2.
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    case_dirs = find_case_dirs(train_dir, limit=limit)
    records = []
    invalid_labels = {}

    for case_dir in tqdm(case_dirs, desc="Scanning label statistics"):
        seg_path = case_dir / f"{case_dir.name}-seg.nii.gz"
        seg = nib.load(str(seg_path)).get_fdata().astype(np.int16)
        labels, counts = np.unique(seg, return_counts=True)
        label_counts = {str(label): 0 for label in range(BRATS_MET_NUM_CLASSES)}
        # PLAN2 CHANGE: some released BraTS-MET labels contain values outside
        # 0..4; skip those cases instead of stopping the whole post-training run.
        case_invalid_labels = []
        for label, count in zip(labels.tolist(), counts.tolist()):
            if label < 0 or label >= BRATS_MET_NUM_CLASSES:
                case_invalid_labels.append(int(label))
                continue
            label_counts[str(label)] = int(count)

        if case_invalid_labels:
            invalid_labels[case_dir.name] = sorted(set(case_invalid_labels))
            continue

        rc_mask = seg == RC_LABEL
        rc_positive = bool(rc_mask.any())
        rc_bbox = None
        rc_bbox_size = None
        rc_center = None
        if rc_positive:
            coords = np.argwhere(rc_mask)
            bbox_min = coords.min(axis=0)
            bbox_max = coords.max(axis=0)
            rc_bbox = {"min": bbox_min.astype(int).tolist(), "max": bbox_max.astype(int).tolist()}
            rc_bbox_size = (bbox_max - bbox_min + 1).astype(int).tolist()
            rc_center = np.round(coords.mean(axis=0)).astype(int).tolist()

        records.append({
            "case": case_dir.name,
            "path": str(case_dir),
            "label_counts": label_counts,
            "rc_positive": rc_positive,
            "rc_bbox": rc_bbox,
            "rc_bbox_size": rc_bbox_size,
            "rc_center": rc_center,
        })

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    skipped_invalid_cases = [
        {"case": case_name, "invalid_labels": labels}
        for case_name, labels in sorted(invalid_labels.items())
    ]
    # PLAN2 CHANGE: persist skipped invalid-label cases so training output makes
    # the automatic data exclusion explicit and auditable.
    if skipped_invalid_cases:
        print(
            f"Skipped {len(skipped_invalid_cases)} cases with labels outside 0..4: "
            + ", ".join(item["case"] for item in skipped_invalid_cases)
        )
        with open(save_dir / "skipped_invalid_label_cases.json", "w") as f:
            json.dump(skipped_invalid_cases, f, indent=2)
        with open(save_dir / "skipped_invalid_label_cases.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case", "invalid_labels"])
            writer.writeheader()
            for item in skipped_invalid_cases:
                writer.writerow({
                    "case": item["case"],
                    "invalid_labels": item["invalid_labels"],
                })

    if not records:
        raise ValueError("No valid cases remain after skipping invalid-label segmentation files.")

    summary = {
        "num_cases": len(records),
        "num_skipped_invalid_label_cases": len(skipped_invalid_cases),
        "num_rc_positive_cases": int(sum(record["rc_positive"] for record in records)),
        "num_rc_negative_cases": int(sum(not record["rc_positive"] for record in records)),
        "label_order": {"0": "background", "1": "NETC", "2": "SNFH", "3": "ET", "4": "RC"},
        "skipped_invalid_label_cases": skipped_invalid_cases,
        "cases": records,
    }
    with open(save_dir / "label_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(save_dir / "rc_case_list.csv", "w", newline="") as f:
        fieldnames = [
            "case", "rc_positive", "rc_voxels", "rc_bbox_min", "rc_bbox_max",
            "rc_bbox_size", "rc_center",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "case": record["case"],
                "rc_positive": record["rc_positive"],
                "rc_voxels": record["label_counts"][str(RC_LABEL)],
                "rc_bbox_min": record["rc_bbox"]["min"] if record["rc_bbox"] else None,
                "rc_bbox_max": record["rc_bbox"]["max"] if record["rc_bbox"] else None,
                "rc_bbox_size": record["rc_bbox_size"],
                "rc_center": record["rc_center"],
            })
    return records


def load_g2_fixed_split_records(fixed_split_root, save_dir, limit=None):
    root = Path(fixed_split_root)
    train_records = scan_label_statistics(root / "train", Path(save_dir) / "g2_train_audit", limit=limit)
    val_records = scan_label_statistics(root / "val", Path(save_dir) / "g2_val_audit", limit=limit)
    test_case_dirs = find_case_dirs(root / "test", limit=limit)

    train_ids = {record["case"] for record in train_records}
    val_ids = {record["case"] for record in val_records}
    test_ids = {path.name for path in test_case_dirs}
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("G2 fixed split contains overlapping case IDs")

    split_json = {
        "source": "g2_case_folder_fixed_split",
        "fixed_split_root": str(root),
        "train_cases": sorted(train_ids),
        "val_cases": sorted(val_ids),
        "test_cases": sorted(test_ids),
    }
    save_dir = Path(save_dir)
    with (save_dir / "g2_fixed_split.json").open("w") as handle:
        json.dump(split_json, handle, indent=2)
    with (save_dir / "g2_fixed_split.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "split"])
        writer.writeheader()
        for split_name, case_ids in (
            ("train", sorted(train_ids)),
            ("val", sorted(val_ids)),
            ("test", sorted(test_ids)),
        ):
            for case_id in case_ids:
                writer.writerow({"case": case_id, "split": split_name})
    print(
        f"G2 fixed split: {len(train_records)} train / {len(val_records)} val / "
        f"{len(test_ids)} locked test"
    )
    return train_records, val_records


def rc_stratified_split(records, split_ratio, seed, save_dir):
    # PLAN2 CHANGE: replace ordinary random split with RC-positive/negative
    # stratification so rare RC cases appear in validation whenever possible.
    rng = np.random.default_rng(seed)
    positives = [record for record in records if record["rc_positive"]]
    negatives = [record for record in records if not record["rc_positive"]]

    def split_group(group):
        if not group:
            return [], []
        shuffled = [group[i] for i in rng.permutation(len(group))]
        if len(shuffled) == 1:
            return shuffled, []
        n_train = int(round(len(shuffled) * split_ratio))
        n_train = min(max(n_train, 1), len(shuffled) - 1)
        return shuffled[:n_train], shuffled[n_train:]

    pos_train, pos_val = split_group(positives)
    neg_train, neg_val = split_group(negatives)
    train_records = sorted(pos_train + neg_train, key=lambda item: item["case"])
    val_records = sorted(pos_val + neg_val, key=lambda item: item["case"])

    if not val_records:
        raise ValueError("Validation split is empty. Use more cases or a smaller split_ratio.")
    if positives and not pos_val:
        print("Warning: only one RC-positive case was available, so validation has no RC-positive case.")

    split_json = {
        "split_ratio": split_ratio,
        "split_seed": seed,
        "train_cases": [record["case"] for record in train_records],
        "val_cases": [record["case"] for record in val_records],
        "train_rc_positive_cases": [record["case"] for record in train_records if record["rc_positive"]],
        "val_rc_positive_cases": [record["case"] for record in val_records if record["rc_positive"]],
    }
    save_dir = Path(save_dir)
    with open(save_dir / "train_val_split_rc_stratified.json", "w") as f:
        json.dump(split_json, f, indent=2)

    with open(save_dir / "train_val_split_rc_stratified.csv", "w", newline="") as f:
        fieldnames = ["case", "split", "rc_positive", "rc_voxels"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, split_records in (("train", train_records), ("val", val_records)):
            for record in split_records:
                writer.writerow({
                    "case": record["case"],
                    "split": split_name,
                    "rc_positive": record["rc_positive"],
                    "rc_voxels": record["label_counts"][str(RC_LABEL)],
                })

    print(
        f"RC-stratified split: {len(train_records)} train / {len(val_records)} val "
        f"({len(pos_train)} RC+ train, {len(pos_val)} RC+ val)"
    )
    return train_records, val_records


# ============================================================
# Datasets And Crop Strategy
# ============================================================

class BraTSPatchDataset(Dataset):
    # PLAN2 CHANGE: task-aware crops replace pure random crops.
    # main phase: 70% lesion-aware crops, 30% random crops.
    # rc phase: 80% RC-centered crops, 20% hard-negative lesion crops.
    def __init__(self, records, crop_size=(96, 96, 96), phase="main", samples_per_epoch=None):
        if not records:
            raise ValueError(f"No records provided for {phase} dataset.")
        self.records = list(records)
        self.crop_size = tuple(crop_size)
        self.phase = phase
        self.samples_per_epoch = samples_per_epoch or len(self.records)
        self.rc_positive_records = [record for record in self.records if record["rc_positive"]]
        self.hard_negative_records = [
            record for record in self.records
            if sum(int(record["label_counts"][str(label)]) for label in (1, 2, 3)) > 0
        ]

    def __len__(self):
        return self.samples_per_epoch

    def _load_case(self, record):
        case_dir = Path(record["path"])
        modalities = [
            nib.load(str(case_dir / f"{case_dir.name}-{modality}.nii.gz")).get_fdata().astype(np.float32)
            for modality in MODALITIES
        ]
        image = np.stack([self._normalize(modality) for modality in modalities], axis=0)
        seg = nib.load(str(case_dir / f"{case_dir.name}-seg.nii.gz")).get_fdata().astype(np.int64)
        return image, seg

    @staticmethod
    def _normalize(image):
        mean = image.mean()
        std = image.std()
        return (image - mean) / (std + 1e-8) if std > 0 else image

    def _pad_if_needed(self, image, seg):
        d, h, w = seg.shape
        cd, ch, cw = self.crop_size
        pad_d = max(0, cd - d)
        pad_h = max(0, ch - h)
        pad_w = max(0, cw - w)
        if pad_d == 0 and pad_h == 0 and pad_w == 0:
            return image, seg
        image = np.pad(image, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
                       mode="constant", constant_values=0)
        seg = np.pad(seg, ((0, pad_d), (0, pad_h), (0, pad_w)),
                     mode="constant", constant_values=0)
        return image, seg

    def _crop_from_start(self, image, seg, start):
        cd, ch, cw = self.crop_size
        ds, hs, ws = start
        return (
            image[:, ds:ds + cd, hs:hs + ch, ws:ws + cw],
            seg[ds:ds + cd, hs:hs + ch, ws:ws + cw],
        )

    def _random_start(self, shape):
        d, h, w = shape
        cd, ch, cw = self.crop_size
        return (
            np.random.randint(0, d - cd + 1) if d > cd else 0,
            np.random.randint(0, h - ch + 1) if h > ch else 0,
            np.random.randint(0, w - cw + 1) if w > cw else 0,
        )

    def _start_from_center(self, center, shape):
        d, h, w = shape
        cd, ch, cw = self.crop_size
        center = np.asarray(center, dtype=np.int64)
        starts = center - np.asarray(self.crop_size, dtype=np.int64) // 2
        max_starts = np.asarray([d - cd, h - ch, w - cw], dtype=np.int64)
        return tuple(np.minimum(np.maximum(starts, 0), max_starts).astype(int).tolist())

    def _crop_random(self, image, seg):
        image, seg = self._pad_if_needed(image, seg)
        return self._crop_from_start(image, seg, self._random_start(seg.shape))

    def _crop_around_mask(self, image, seg, mask, avoid_rc=False, attempts=20):
        image, seg = self._pad_if_needed(image, seg)
        coords = np.argwhere(mask)
        if coords.size == 0:
            return self._crop_random(image, seg)
        for _ in range(attempts):
            center = coords[np.random.randint(0, len(coords))]
            crop_image, crop_seg = self._crop_from_start(
                image, seg, self._start_from_center(center, seg.shape)
            )
            if not avoid_rc or not (crop_seg == RC_LABEL).any():
                return crop_image, crop_seg
        return crop_image, crop_seg

    def _choose_record(self, idx):
        if self.phase == "rc":
            if self.rc_positive_records and np.random.random() < 0.8:
                return self.rc_positive_records[np.random.randint(0, len(self.rc_positive_records))], "rc_center"
            if self.hard_negative_records:
                return self.hard_negative_records[np.random.randint(0, len(self.hard_negative_records))], "hard_negative"
        return self.records[idx % len(self.records)], "main"

    def __getitem__(self, idx):
        record, crop_kind = self._choose_record(idx)
        image, seg = self._load_case(record)

        if self.phase == "main":
            if np.random.random() < 0.7:
                image, seg = self._crop_around_mask(image, seg, seg > 0)
            else:
                image, seg = self._crop_random(image, seg)
        elif crop_kind == "rc_center":
            image, seg = self._crop_around_mask(image, seg, seg == RC_LABEL)
        elif crop_kind == "hard_negative":
            image, seg = self._crop_around_mask(image, seg, (seg > 0) & (seg < RC_LABEL), avoid_rc=True)
        else:
            image, seg = self._crop_random(image, seg)

        return (
            torch.from_numpy(np.ascontiguousarray(image)).float(),
            torch.from_numpy(np.ascontiguousarray(seg)).long(),
        )


class BraTSFullVolumeDataset(Dataset):
    # PLAN2 CHANGE: full-volume dataset is used only for Dice validation, not
    # for writing segmentation images.
    def __init__(self, records):
        self.records = list(records)
        if not self.records:
            raise ValueError("No validation records provided.")

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _normalize(image):
        mean = image.mean()
        std = image.std()
        return (image - mean) / (std + 1e-8) if std > 0 else image

    def __getitem__(self, idx):
        record = self.records[idx]
        case_dir = Path(record["path"])
        modalities = [
            nib.load(str(case_dir / f"{case_dir.name}-{modality}.nii.gz")).get_fdata().astype(np.float32)
            for modality in MODALITIES
        ]
        image = np.stack([self._normalize(modality) for modality in modalities], axis=0)
        seg = nib.load(str(case_dir / f"{case_dir.name}-seg.nii.gz")).get_fdata().astype(np.int64)
        return (
            torch.from_numpy(np.ascontiguousarray(image)).float(),
            torch.from_numpy(np.ascontiguousarray(seg)).long(),
            record["case"],
        )


# ============================================================
# Losses And Metrics
# ============================================================

class MainDiceLoss(nn.Module):
    # PLAN2 CHANGE: RC voxels stay as ignore_index=4 for the main task instead
    # of being converted to background.
    def __init__(self, smooth=1.0, num_classes=4, ignore_index=MAIN_IGNORE_INDEX):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        target = target.squeeze(1) if target.dim() == 5 else target
        valid = target != self.ignore_index
        target_for_one_hot = target.clone()
        target_for_one_hot[~valid] = 0
        target_one_hot = F.one_hot(target_for_one_hot, num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
        valid = valid.unsqueeze(1).float()
        probs = torch.softmax(logits, dim=1) * valid
        target_one_hot = target_one_hot * valid

        dice_scores = []
        for label in range(1, self.num_classes):
            pred_flat = probs[:, label].reshape(probs.size(0), -1)
            target_flat = target_one_hot[:, label].reshape(target_one_hot.size(0), -1)
            intersection = (pred_flat * target_flat).sum(dim=1)
            union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
            dice_scores.append((2.0 * intersection + self.smooth) / (union + self.smooth))
        return 1.0 - torch.stack(dice_scores, dim=1).mean()


class FocalBCELoss(nn.Module):
    # PLAN2 CHANGE: dedicated RC focal BCE for rare binary RC supervision.
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.exp(-bce)
        alpha_factor = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)
        return (alpha_factor * (1.0 - pt).pow(self.gamma) * bce).mean()


class TverskyLoss(nn.Module):
    # PLAN2 CHANGE: Tversky emphasizes RC recall by default beta > alpha.
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.size(0), -1)
        target = target.reshape(target.size(0), -1)
        true_pos = (probs * target).sum(dim=1)
        false_pos = (probs * (1.0 - target)).sum(dim=1)
        false_neg = ((1.0 - probs) * target).sum(dim=1)
        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
        )
        return 1.0 - tversky.mean()


class Plan2Losses(nn.Module):
    # PLAN2 CHANGE: combine Dice+CE for main labels and FocalBCE+Tversky for RC.
    def __init__(self, config):
        super().__init__()
        self.main_dice = MainDiceLoss()
        self.main_ce = nn.CrossEntropyLoss(ignore_index=MAIN_IGNORE_INDEX)
        self.rc_focal = FocalBCELoss(config["rc_focal_alpha"], config["rc_focal_gamma"])
        self.rc_tversky = TverskyLoss(config["rc_tversky_alpha"], config["rc_tversky_beta"])

    def forward(self, outputs, target):
        main_logits = outputs["main_logits"]
        rc_logit = outputs["rc_logit"]
        target = target.squeeze(1) if target.dim() == 5 else target
        target_rc = (target == RC_LABEL).float().unsqueeze(1)
        main_loss = self.main_dice(main_logits, target) + self.main_ce(main_logits, target)
        rc_loss = self.rc_focal(rc_logit, target_rc) + self.rc_tversky(rc_logit, target_rc)
        return main_loss, rc_loss


def dice_score_volume(prediction, target, num_classes=BRATS_MET_NUM_CLASSES):
    scores = []
    for label in range(1, num_classes):
        pred_mask = (prediction == label).float()
        target_mask = (target == label).float()
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        scores.append(((2.0 * intersection + 1e-5) / (union + 1e-5)).item())
    return scores


def class_dice_dict(dice_values):
    return {f"dice_{name}": float(value) for name, value in zip(CLASS_NAMES, dice_values)}


def outputs_to_prediction(outputs, rc_threshold):
    return SAM2UNet3D.logits_to_label_map(
        outputs["main_logits"],
        outputs["rc_logit"],
        rc_threshold=rc_threshold,
    )


# ============================================================
# Sliding Window Validation
# ============================================================

def combine_probs_to_prediction(main_probs, rc_prob, rc_threshold):
    main_prediction = torch.argmax(main_probs, dim=0)
    prediction = main_prediction.clone()
    prediction[rc_prob > rc_threshold] = RC_LABEL
    return prediction.long()


def sliding_window_predict_probs(model, image, patch_size, overlap, device, batch_size):
    # PLAN2 CHANGE: sliding-window validation aggregates main probabilities and
    # RC probabilities separately, then thresholds RC to produce final labels.
    model.eval()
    _, depth, height, width = image.shape
    pd, ph, pw = patch_size
    step_d = max(1, int(pd * (1.0 - overlap)))
    step_h = max(1, int(ph * (1.0 - overlap)))
    step_w = max(1, int(pw * (1.0 - overlap)))

    pad_d = max(0, pd - depth)
    pad_h = max(0, ph - height)
    pad_w = max(0, pw - width)
    total_d, total_h, total_w = depth + pad_d, height + pad_h, width + pad_w
    if (total_d - pd) % step_d != 0:
        pad_d += step_d - ((total_d - pd) % step_d)
    if (total_h - ph) % step_h != 0:
        pad_h += step_h - ((total_h - ph) % step_h)
    if (total_w - pw) % step_w != 0:
        pad_w += step_w - ((total_w - pw) % step_w)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h, 0, pad_d))

    _, padded_d, padded_h, padded_w = image.shape
    main_output = torch.zeros((4, padded_d, padded_h, padded_w), device=device)
    rc_output = torch.zeros((padded_d, padded_h, padded_w), device=device)
    count = torch.zeros((padded_d, padded_h, padded_w), device=device)

    sigma = 0.125
    coords = [torch.linspace(-1, 1, size, device=device) for size in patch_size]
    grid = torch.meshgrid(*coords, indexing="ij")
    gaussian = torch.exp(-(grid[0] ** 2 + grid[1] ** 2 + grid[2] ** 2) / (2 * sigma ** 2))
    gaussian = gaussian / gaussian.max()

    positions = [
        (d, h, w)
        for d in range(0, padded_d - pd + 1, step_d)
        for h in range(0, padded_h - ph + 1, step_h)
        for w in range(0, padded_w - pw + 1, step_w)
    ]

    with torch.no_grad():
        for index in tqdm(range(0, len(positions), batch_size), desc="Sliding Window", leave=False):
            batch_positions = positions[index:index + batch_size]
            patches = torch.stack([
                image[:, d:d + pd, h:h + ph, w:w + pw] for d, h, w in batch_positions
            ]).to(device)

            with autocast(enabled=device.type == "cuda"):
                outputs = model(patches, return_dict=True)
                main_probs = torch.softmax(outputs["main_logits"].float(), dim=1)
                rc_probs = torch.sigmoid(outputs["rc_logit"].float())[:, 0]

            for patch_index, (d, h, w) in enumerate(batch_positions):
                main_output[:, d:d + pd, h:h + ph, w:w + pw] += main_probs[patch_index] * gaussian
                rc_output[d:d + pd, h:h + ph, w:w + pw] += rc_probs[patch_index] * gaussian
                count[d:d + pd, h:h + ph, w:w + pw] += gaussian

    count = count.clamp(min=1e-8)
    main_output = main_output / count.unsqueeze(0)
    rc_output = rc_output / count
    return (
        main_output[:, :depth, :height, :width],
        rc_output[:depth, :height, :width],
    )


def write_validation_outputs(summary_rows, per_case_rows, config):
    # PLAN2 CHANGE: validation output files contain only four foreground Dice
    # scores and the average Dice; no segmentation label maps are saved.
    save_dir = Path(config["save_dir"])

    threshold_path = save_dir / "validation_rc_threshold_sweep.csv"
    with open(threshold_path, "w", newline="") as f:
        fieldnames = ["epoch", "rc_threshold"] + [f"dice_{name}" for name in CLASS_NAMES] + ["dice_average"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([
            {key: row[key] for key in fieldnames}
            for row in summary_rows
        ])

    best_row = max(summary_rows, key=lambda row: row["combined_score"])
    public_best_row = {
        key: value for key, value in best_row.items()
        if key in (["epoch", "rc_threshold", "dice_average"] + [f"dice_{name}" for name in CLASS_NAMES])
    }

    summary_csv_path = save_dir / "validation_dice_summary.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(public_best_row.keys()))
        writer.writeheader()
        writer.writerow(public_best_row)

    summary_json_path = save_dir / "validation_dice_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(public_best_row, f, indent=2)

    per_case_path = save_dir / "validation_dice_per_case.csv"
    with open(per_case_path, "w", newline="") as f:
        fieldnames = ["case", "rc_threshold"] + [f"dice_{name}" for name in CLASS_NAMES] + ["dice_average"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([
            row for row in per_case_rows if row["rc_threshold"] == public_best_row["rc_threshold"]
        ])

    # PLAN2 CHANGE: append one row per validation epoch and redraw the Dice
    # curve image requested for epoch-vs-NETC/SNFH/ET/RC Dice tracking.
    history_path, plot_path = update_validation_dice_history(public_best_row, config)
    print(f"Saved validation Dice history: {history_path}")
    if plot_path is not None:
        print(f"Saved validation Dice curve: {plot_path}")

    return best_row, public_best_row


def update_validation_dice_history(validation_row, config):
    # PLAN2 CHANGE: keep a persistent validation Dice history so resumed runs
    # can continue drawing the epoch-Dice figure without losing earlier points.
    save_dir = Path(config["save_dir"])
    history_path = save_dir / "validation_dice_history.csv"
    fieldnames = ["epoch", "rc_threshold"] + [f"dice_{name}" for name in CLASS_NAMES] + ["dice_average"]

    history_rows = []
    if history_path.exists():
        with open(history_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            history_rows = [row for row in reader if row.get("epoch")]

    current_epoch = int(validation_row["epoch"])
    history_rows = [
        row for row in history_rows
        if int(float(row["epoch"])) != current_epoch
    ]
    history_rows.append({key: validation_row[key] for key in fieldnames})
    history_rows.sort(key=lambda row: int(float(row["epoch"])))

    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)

    plot_path = plot_validation_dice_history(history_rows, save_dir)
    return history_path, plot_path


def plot_validation_dice_history(history_rows, save_dir):
    # PLAN2 CHANGE: generate a PNG curve of epoch vs four foreground Dice
    # scores; matplotlib is imported lazily so environments without it can
    # still train and write CSV summaries.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib is not installed; skipped validation Dice curve PNG.")
        return None

    if not history_rows:
        return None

    epochs = [int(float(row["epoch"])) for row in history_rows]
    plot_path = save_dir / "validation_dice_by_epoch.png"

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for name in CLASS_NAMES:
        values = [float(row[f"dice_{name}"]) for row in history_rows]
        ax.plot(epochs, values, marker="o", linewidth=1.8, label=name)

    average_values = [float(row["dice_average"]) for row in history_rows]
    ax.plot(epochs, average_values, marker="s", linewidth=2.2,
            linestyle="--", color="black", label="Average")

    ax.set_title("Validation Dice by Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def validate_full_volume(model, val_dataset, config, device, epoch):
    summary_accumulators = {
        threshold: {"dice_sum": np.zeros(len(CLASS_NAMES), dtype=np.float64), "case_count": 0}
        for threshold in config["rc_thresholds"]
    }
    per_case_rows = []

    print("\nRunning full-volume validation...")
    for image, label, case_name in tqdm(val_dataset, desc="Validation Cases"):
        image = image.to(device)
        label_cpu = label.cpu()
        main_probs, rc_prob = sliding_window_predict_probs(
            model,
            image,
            patch_size=config["crop_size"],
            overlap=config["sliding_window_overlap"],
            device=device,
            batch_size=config["sliding_window_batch_size"],
        )
        for threshold in config["rc_thresholds"]:
            prediction = combine_probs_to_prediction(main_probs, rc_prob, threshold).cpu()
            dice = dice_score_volume(prediction, label_cpu)
            summary_accumulators[threshold]["dice_sum"] += np.asarray(dice)
            summary_accumulators[threshold]["case_count"] += 1
            per_case_rows.append({
                "case": case_name,
                "rc_threshold": threshold,
                **class_dice_dict(dice),
                "dice_average": float(np.mean(dice)),
            })

    summary_rows = []
    for threshold, accumulator in summary_accumulators.items():
        avg_dice = accumulator["dice_sum"] / max(accumulator["case_count"], 1)
        main_mean = float(np.mean(avg_dice[:3]))
        rc_dice = float(avg_dice[3])
        row = {
            "epoch": epoch,
            "rc_threshold": threshold,
            **class_dice_dict(avg_dice),
            "dice_average": float(np.mean(avg_dice)),
            "main_dice_average": main_mean,
            "combined_score": 0.4 * main_mean + 0.6 * rc_dice,
        }
        summary_rows.append(row)

    best_row, public_best_row = write_validation_outputs(summary_rows, per_case_rows, config)
    print_validation_summary(public_best_row)
    return best_row


def print_validation_summary(row):
    print("\nValidation Dice Summary")
    print(f"RC threshold: {row['rc_threshold']}")
    for name in CLASS_NAMES:
        print(f"Dice {name}: {row[f'dice_{name}']:.4f}")
    print(f"Dice Average: {row['dice_average']:.4f}")


# ============================================================
# Training
# ============================================================

def build_loader(dataset, config, shuffle):
    loader_kwargs = {
        "batch_size": config["batch_size"],
        "shuffle": shuffle,
        "num_workers": config["num_workers"],
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    if config["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = config["prefetch_factor"]
        loader_kwargs["persistent_workers"] = True
    return DataLoader(dataset, **loader_kwargs)


def build_model(config, device):
    # PLAN2 CHANGE: instantiate SAM2UNet3D in dict-return mode so training can
    # consume main_logits and rc_logit separately.
    model = SAM2UNet3D(
        spatial_size=config["crop_size"],
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        feature_size=config["feature_size"],
        depths=config["depths"],
        num_heads=config["num_heads"],
        window_size=tuple(config["window_size"]),
        dropout_rate=config["dropout_rate"],
        use_attention=config["use_attention"],
        return_dict=True,
    ).to(device)
    print(f"Total parameters: {model.get_num_parameters():,}")
    return model


def build_optimizer(model, config):
    # PLAN2 CHANGE: optimizer uses a higher learning rate for rc_head while the
    # backbone/decoder and main_head keep the plan's base LR.
    main_head_ids = {id(param) for param in model.main_head.parameters()}
    rc_head_ids = {id(param) for param in model.rc_head.parameters()}
    backbone_params = [
        param for param in model.parameters()
        if id(param) not in main_head_ids and id(param) not in rc_head_ids
    ]
    param_groups = [
        {"params": backbone_params, "lr": config["lr"]},
        {"params": model.main_head.parameters(), "lr": config["main_head_lr"]},
        {"params": model.rc_head.parameters(), "lr": config["rc_head_lr"]},
    ]
    try:
        return optim.AdamW(param_groups, weight_decay=config["weight_decay"], fused=True)
    except Exception:
        return optim.AdamW(param_groups, weight_decay=config["weight_decay"])


def train_phase(model, loader, losses, optimizer, scaler, device, epoch, phase, config):
    model.train()
    total_loss = 0.0
    total_main_loss = 0.0
    total_rc_loss = 0.0
    total_dice = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Epoch {epoch} [{phase}]")
    for batch_index, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=device.type == "cuda"):
            outputs = model(images, return_dict=True)
            main_loss, rc_loss = losses(outputs, labels)
            if phase == "main":
                loss = config["main_loss_weight"] * main_loss
            else:
                loss = config["rc_phase_main_loss_weight"] * main_loss + config["rc_loss_weight"] * rc_loss
            scaled_loss = loss / config["accumulation_steps"]

        scaler.scale(scaled_loss).backward()
        should_step = (
            (batch_index + 1) % config["accumulation_steps"] == 0
            or (batch_index + 1) == len(loader)
        )
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().item())
        total_main_loss += float(main_loss.detach().item())
        total_rc_loss += float(rc_loss.detach().item())
        with torch.no_grad():
            prediction = outputs_to_prediction(outputs, config["rc_thresholds"][3])
            dice = dice_score_volume(prediction.detach().cpu(), labels.detach().cpu())
            total_dice += np.asarray(dice)

        pbar.set_postfix({
            "loss": f"{loss.detach().item():.4f}",
            "main_loss": f"{main_loss.detach().item():.4f}",
            "rc_loss": f"{rc_loss.detach().item():.4f}",
            "avg_dice": f"{np.mean(dice):.4f}",
        })

    steps = max(len(loader), 1)
    return {
        "loss": total_loss / steps,
        "main_loss": total_main_loss / steps,
        "rc_loss": total_rc_loss / steps,
        "dice": (total_dice / steps).tolist(),
    }


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, history, best_scores, config):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "history": history,
        "best_scores": best_scores,
        "config": config,
    }, path)


def maybe_resume(model, optimizer, scheduler, scaler, config, device):
    start_epoch = 1
    best_scores = {"main": -1.0, "rc": -1.0, "combined": -1.0}
    history = []
    checkpoint_path = Path(config["save_dir"]) / "latest_checkpoint.pth"
    if config["resume"] and checkpoint_path.exists():
        # PLAN2 CHANGE: resume includes the new separate best model scores.
        print(f"Resuming from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint.get("history", history)
        best_scores = checkpoint.get("best_scores", best_scores)
    return start_epoch, best_scores, history


def update_best_models(validation_row, model, optimizer, scheduler, scaler, epoch, history, best_scores, config):
    main_score = float(validation_row["main_dice_average"])
    rc_score = float(validation_row["dice_RC"])
    combined_score = float(validation_row["combined_score"])
    save_dir = Path(config["save_dir"])

    if main_score > best_scores["main"]:
        best_scores["main"] = main_score
        save_checkpoint(save_dir / "best_main_model.pth", model, optimizer, scheduler,
                        scaler, epoch, history, best_scores, config)
        print(f"Saved best_main_model.pth: main Dice={main_score:.4f}")

    if rc_score > best_scores["rc"]:
        best_scores["rc"] = rc_score
        save_checkpoint(save_dir / "best_rc_model.pth", model, optimizer, scheduler,
                        scaler, epoch, history, best_scores, config)
        print(f"Saved best_rc_model.pth: RC Dice={rc_score:.4f}")

    if combined_score > best_scores["combined"]:
        best_scores["combined"] = combined_score
        save_checkpoint(save_dir / "best_combined_model.pth", model, optimizer, scheduler,
                        scaler, epoch, history, best_scores, config)
        print(f"Saved best_combined_model.pth: combined score={combined_score:.4f}")


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    if config["fixed_split_root"]:
        train_records, val_records = load_g2_fixed_split_records(
            config["fixed_split_root"],
            save_dir,
            limit=config["debug_case_limit"],
        )
    else:
        records = scan_label_statistics(config["train_dir"], save_dir, limit=config["debug_case_limit"])
        train_records, val_records = rc_stratified_split(
            records,
            split_ratio=config["split_ratio"],
            seed=config["split_seed"],
            save_dir=save_dir,
        )

    main_dataset = BraTSPatchDataset(train_records, crop_size=config["crop_size"], phase="main")
    rc_dataset = BraTSPatchDataset(train_records, crop_size=config["crop_size"], phase="rc")
    val_dataset = BraTSFullVolumeDataset(val_records)
    main_loader = build_loader(main_dataset, config, shuffle=True)
    rc_loader = build_loader(rc_dataset, config, shuffle=True)

    model = build_model(config, device)
    losses = Plan2Losses(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    scaler = GradScaler(enabled=device.type == "cuda")
    start_epoch, best_scores, history = maybe_resume(model, optimizer, scheduler, scaler, config, device)

    for epoch in range(start_epoch, config["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        main_metrics = train_phase(
            model, main_loader, losses, optimizer, scaler, device, epoch, "main", config
        )
        rc_metrics = None
        if epoch > config["warmup_epochs"]:
            rc_metrics = train_phase(
                model, rc_loader, losses, optimizer, scaler, device, epoch, "rc", config
            )
        else:
            print(f"Warmup epoch {epoch}: skipped RC phase.")

        scheduler.step()
        epoch_record = {
            "epoch": epoch,
            "main_phase": main_metrics,
            "rc_phase": rc_metrics,
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_main_head": optimizer.param_groups[1]["lr"],
            "lr_rc_head": optimizer.param_groups[2]["lr"],
        }
        history.append(epoch_record)

        save_checkpoint(save_dir / "latest_checkpoint.pth", model, optimizer, scheduler,
                        scaler, epoch, history, best_scores, config)
        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        should_validate = epoch % config["checkpoint_interval"] == 0 or epoch == config["epochs"]
        if should_validate:
            validation_row = validate_full_volume(model, val_dataset, config, device, epoch)
            update_best_models(validation_row, model, optimizer, scheduler, scaler,
                               epoch, history, best_scores, config)
            save_checkpoint(save_dir / "latest_checkpoint.pth", model, optimizer, scheduler,
                            scaler, epoch, history, best_scores, config)

    print("\nTraining completed.")
    print(f"Best main Dice: {best_scores['main']:.4f}")
    print(f"Best RC Dice: {best_scores['rc']:.4f}")
    print(f"Best combined score: {best_scores['combined']:.4f}")


# ============================================================
# Main
# ============================================================

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    config = build_config(args)
    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 70)
    print("SAM2-UNET PLAN2 POST TRAINING")
    print("=" * 70)
    print(f"Training data: {config['train_dir']}")
    print(f"G2 fixed split: {config['fixed_split_root'] or 'disabled (exploratory RC split)'}")
    print(f"Save dir:      {config['save_dir']}")
    print(f"Epochs:        {config['epochs']} (warmup={config['warmup_epochs']})")
    print(f"Crop size:     {config['crop_size']}")
    print(f"Batch:         {config['batch_size']} x {config['accumulation_steps']}")
    print(f"Workers:       {config['num_workers']}")
    print(f"Dice output:   NETC/SNFH/ET/RC + average only")
    # PLAN2 CHANGE: validation now also writes validation_dice_by_epoch.png
    # for the requested epoch-vs-Dice visualization.
    print(f"Dice plot:     validation_dice_by_epoch.png")
    print(f"Seg images:    disabled")
    print("=" * 70)
    train(config)


if __name__ == "__main__":
    main()
