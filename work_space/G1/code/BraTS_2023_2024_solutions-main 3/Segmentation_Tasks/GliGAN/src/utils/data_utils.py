import os
import csv
import torch
import numpy as np
from monai.data import CSVDataset, CacheDataset, DataLoader, Dataset, DistributedSampler, SmartCacheDataset, load_decathlon_datalist
from monai.data.utils import pad_list_data_collate
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    ScaleIntensityd,
    CopyItemsd,
    CropForegroundd,
    SpatialCropd,
    ToTensord,
    ResizeWithPadOrCropd,
)
import warnings
from src.utils.gaussian_noise_tumour_extended import GaussianNoiseTumourExtended
from src.utils.gaussian_noise_tumour import GaussianNoiseTumour


def _load_csv_records(csv_path, col_names, col_types, split="train"):
    """Load CSV rows and enforce the requested train/validation boundary."""
    split = split or "train"
    valid_splits = {"train", "val", "all"}
    if split not in valid_splits:
        raise ValueError(f"--split must be one of {sorted(valid_splits)}, got: {split}")

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        missing_columns = [name for name in col_names if name not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"CSV is missing required columns: {missing_columns}")
        if split != "all" and "split" not in reader.fieldnames:
            raise ValueError(
                f"CSV has no 'split' column; refusing to load split='{split}': {csv_path}")
        rows = [
            row for row in reader
            if split == "all" or row.get("split", "").strip() == split
        ]

    if not rows:
        raise ValueError(f"No CSV rows available for split='{split}' in {csv_path}")

    records = []
    for row in rows:
        record = {}
        for name in col_names:
            value = row[name]
            converter = col_types.get(name, {}).get("type") if name in col_types else None
            if converter is not None:
                value = converter(value)
            record[name] = value
        records.append(record)
    return records


def get_loader(args):
    loader_workers = int(getattr(args, "loader_workers", 0))
    CSV_PATH = None
    if args.csv_path == "":
        csv_dir = os.path.join(getattr(args, "checkpoint_root", "../../Checkpoint"), args.logdir)
        if os.path.isdir(csv_dir):
            for file_name in os.listdir(csv_dir):
                if file_name.endswith("csv"):
                    CSV_PATH = os.path.join(csv_dir, file_name)
                    break
    else:
        CSV_PATH = args.csv_path
    if CSV_PATH is None:
        raise FileNotFoundError(
            f"No CSV file found. Provide --csv_path or create it under the checkpoint root")
    print(f"CSV_PATH: {CSV_PATH}")

    crop_size = getattr(args, "crop_size", 64)

    modality_to_scan = {"t1c": "scan_t1c", "t1n": "scan_t1n", "t2w": "scan_t2w", "t2f": "scan_t2f"}
    if args.modality not in modality_to_scan:
        raise ValueError(f"Unsupported modality: {args.modality}")
    scan_name = modality_to_scan[args.modality]
    col_names = [scan_name, 'label', 'center_x', 'center_y', 'center_z', 'x_extreme_min', 'x_extreme_max', 'y_extreme_min', 'y_extreme_max', 'z_extreme_min', 'z_extreme_max', 'x_size', 'y_size', 'z_size', 'n_voxels', 'patient_n_crops']
    col_types = {'center_x': {'type': int}, 'center_y': {'type': int}, 'center_z': {'type': int}, 'x_extreme_min': {'type': int}, 'x_extreme_max': {'type': int}, 'y_extreme_min': {'type': int}, 'y_extreme_max': {'type': int}, 'z_extreme_min': {'type': int}, 'z_extreme_max': {'type': int}, 'x_size': {'type': int}, 'y_size': {'type': int}, 'z_size': {'type': int}, 'n_voxels': {'type': int}, 'patient_n_crops': {'type': int}}
    print(f"Scan Modality: {scan_name}")
    split = getattr(args, "split", "train")
    print(f"CSV split: {split}")

    normalization = getattr(args, "normalization", "minmax")
    print(f"Normalization: {normalization}")

    if args.dataset=="BRATS_2023" or args.dataset=="BRATS_GOAT_2024":
        if args.dataset=="BRATS_2023":
            print(f"Using dataset: BRATS_2023")
        else:
             print(f"Using dataset: BRATS_GOAT_2024")
        from src.utils.convert_to_multi_channel_based_on_brats_classes import ConvertToMultiChannelBasedOnBratsGliomaClasses2023d as LABEL_TRANSFORM
        if int(args.in_channels)!=4:
            print("YOU WILL HAVE AN ERROR IN THE DATA LOADER. Change in_channels to 4")
    elif args.dataset=="BRATS_2024":
        print(f"Using dataset: BRATS_2024")
        from src.utils.convert_to_multi_channel_based_on_brats_classes import ConvertToMultiChannelBasedOnBratsGliomaPosTreatClasses2024d as LABEL_TRANSFORM
        if int(args.in_channels)!=5:
            print("YOU WILL HAVE AN ERROR IN THE DATA LOADER. Change in_channels to 5")
    elif args.dataset=="BRATS_2024_MENINGIOMA":
        print(f"Using dataset: BRATS_2024_MENINGIOMA")
        from src.utils.convert_to_multi_channel_based_on_brats_classes import ConvertToMultiChannelBasedOnBratsMeningiomaClasses2024d as LABEL_TRANSFORM
        if int(args.in_channels)!=2:
            print("YOU WILL HAVE AN ERROR IN THE DATA LOADER. Change in_channels to 2")
    else:
        raise ValueError("The dataset must be from BraTS: BRATS_GOAT_2024, BRATS_2024, BRATS_2023 or BRATS_2024_MENINGIOMA")

    if args.noise_type=="gaussian_extended":
        print("Using Gaussian noise with noise in the surrounding tissue")
        train_transforms = Compose(
                    [
                        LoadImaged(keys=[scan_name, 'label'], image_only=False),
                        EnsureChannelFirstd(keys=[scan_name, "label"]),
                        EnsureTyped(keys=[scan_name, "label"]),
                        # TODO uncomment if not found a solution around
                        #ResizeWithPadOrCropd(  # TODO: In principle this is not need for the Brats2023 and BratsGOAT2024, however the Brats2024 glioma requires this (original shape 182, 218, 182)...
                        #    keys=[scan_name, 'label'],
                        #    spatial_size=(240,240,155),
                        #    mode="constant",
                        #    value=0,
                        #    lazy=False,
                        #),
                        LABEL_TRANSFORM(keys="label"),
                        GaussianNoiseTumourExtended(keys=scan_name, normalization=normalization, target_size=crop_size),
                        ToTensord(keys=[scan_name, f'{scan_name}_crop', f'{scan_name}_crop_pad', f'{scan_name}_noisy', 'label', 'label_crop_pad', 'center_x', 'center_y', 'center_z', 'x_extreme_min', 'x_extreme_max', 'y_extreme_min', 'y_extreme_max', 'z_extreme_min', 'z_extreme_max', 'x_size', 'y_size', 'z_size', 'n_voxels', 'effective_n_voxels', 'patient_n_crops']),
                    ]
                )

    elif args.noise_type=="gaussian_tumour":
        print("Using Gaussian noise only in the tumour zone")
        train_transforms = Compose(
                    [
                        LoadImaged(keys=[scan_name, 'label'], image_only=False),
                        EnsureChannelFirstd(keys=[scan_name, "label"]),
                        EnsureTyped(keys=[scan_name, "label"]),
                        # TODO uncomment if not found a solution around
                        #ResizeWithPadOrCropd( # In principle this is not need for the Brats2023 and BratsGOAT2024, however the Brats2024 glioma requires this (original shape 182, 218, 182)...
                        #    keys=[scan_name, 'label'],
                        #    spatial_size=(240,240,155),
                        #    mode="constant",
                        #    value=0,
                        #    lazy=False,
                        #),
                        LABEL_TRANSFORM(keys="label"),
                        GaussianNoiseTumour(keys=scan_name, normalization=normalization, target_size=crop_size),
                        ToTensord(keys=[scan_name, f'{scan_name}_crop', f'{scan_name}_crop_pad', f'{scan_name}_noisy', 'label', 'label_crop_pad', 'center_x', 'center_y', 'center_z', 'x_extreme_min', 'x_extreme_max', 'y_extreme_min', 'y_extreme_max', 'z_extreme_min', 'z_extreme_max', 'x_size', 'y_size', 'z_size', 'n_voxels', 'effective_n_voxels', 'patient_n_crops']),
                    ]
                )

    train_records = _load_csv_records(CSV_PATH, col_names, col_types, split=split)
    print(f"Number of {split} lesion crops: {len(train_records)}")

    # The crop offset and tumour noise are stochastic. CacheDataset would freeze
    # both transforms after the first pass, so use a plain Dataset deliberately.
    train_ds = Dataset(data=train_records, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        num_workers=loader_workers,
        drop_last=False,
        shuffle=(split == "train"),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(loader_workers > 0),
        collate_fn=pad_list_data_collate,
    )
    print(f'Dataset training: number of batches: {len(train_loader)}')
    if len(train_loader) == 0:
        raise RuntimeError("DataLoader contains zero batches")
    print("Leaving the data loader. Good luck!")
    return train_loader
