import os
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
from src.utils.gaussian_noise_tumour import GaussianNoiseTumour, CropLesionROId
import pandas as pd
from src.utils.splits import select_split

def resolve_csv_path(args):
    """[1mm] Training CSV: --csv_path, else Checkpoint/<logdir>/<logdir>.csv, else the only *.csv there.

    The original took the first *.csv returned by os.listdir (arbitrary order), so a backup CSV in the
    same folder could silently be used instead."""
    if args.csv_path:
        return args.csv_path
    csv_dir = f"Checkpoint/{args.logdir}"
    default = os.path.join(csv_dir, f"{args.logdir}.csv")
    if os.path.isfile(default):
        return default
    found = sorted(f for f in os.listdir(csv_dir) if f.endswith(".csv")) if os.path.isdir(csv_dir) else []
    if len(found) == 1:
        return os.path.join(csv_dir, found[0])
    raise FileNotFoundError(f"Expected {default} (or exactly one *.csv in {csv_dir}, found {found}). "
                            f"Build it with src/train/csv_creator.py or pass --csv_path.")


def get_loader(args):
    NUM_WORKERS = int(args.num_workers)
    CSV_PATH = resolve_csv_path(args)
    print(f"CSV_PATH: {CSV_PATH}")

    crop_size = getattr(args, "crop_size", 64)

    modality_to_scan = {"t1c": "scan_t1c", "t1n": "scan_t1n", "t2w": "scan_t2w", "t2f": "scan_t2f"}
    scan_name = modality_to_scan[args.modality]
    col_names = [scan_name, 'label', 'center_x', 'center_y', 'center_z', 'x_extreme_min', 'x_extreme_max', 'y_extreme_min', 'y_extreme_max', 'z_extreme_min', 'z_extreme_max', 'x_size', 'y_size', 'z_size', 'n_voxels', 'patient_n_crops']
    col_types = {'center_x': {'type': int}, 'center_y': {'type': int}, 'center_z': {'type': int}, 'x_extreme_min': {'type': int}, 'x_extreme_max': {'type': int}, 'y_extreme_min': {'type': int}, 'y_extreme_max': {'type': int}, 'z_extreme_min': {'type': int}, 'z_extreme_max': {'type': int}, 'x_size': {'type': int}, 'y_size': {'type': int}, 'z_size': {'type': int}, 'n_voxels': {'type': int}, 'patient_n_crops': {'type': int}}
    print(f"Scan Modality: {scan_name}")

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

    elif args.noise_type=="gaussian_tumour" and not getattr(args, "no_roi_cache", False):
        # [1mm] Cache only a lesion ROI per row (CropLesionROId) and re-draw the random offset /
        # noise every iteration (GaussianNoiseTumour is Randomizable). The multi-channel label
        # conversion is applied to the cropped window inside GaussianNoiseTumour.
        print("Using Gaussian noise only in the tumour zone (ROI cache, per-iteration random offset)")
        train_transforms = Compose(
                    [
                        LoadImaged(keys=[scan_name, 'label'], image_only=False),
                        EnsureChannelFirstd(keys=[scan_name, "label"]),
                        EnsureTyped(keys=[scan_name, "label"]),
                        CropLesionROId(keys=[scan_name, 'label'], target_size=crop_size),
                        GaussianNoiseTumour(keys=scan_name, normalization=normalization, target_size=crop_size,
                                            label_transform=LABEL_TRANSFORM(keys="label").converter),
                        ToTensord(keys=[f'{scan_name}_crop', f'{scan_name}_crop_pad', f'{scan_name}_noisy', 'label_crop_pad', 'center_x', 'center_y', 'center_z', 'x_extreme_min', 'x_extreme_max', 'y_extreme_min', 'y_extreme_max', 'z_extreme_min', 'z_extreme_max', 'x_size', 'y_size', 'z_size', 'n_voxels', 'effective_n_voxels', 'patient_n_crops'], track_meta=False),  # plain tensors: much faster collate, values unchanged
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

    # [1mm] Row selection: hold out the cases the user put in the val set (csv_creator --val_patients /
    # --val_list; without them every row is split == "train"). The t2w model also drops t2w_valid == 0
    # rows, which only exist if csv_creator got --t2w_invalid_list. --use_all_rows uses every CSV row
    # (original behaviour: val rows are trained on too).
    df = pd.read_csv(CSV_PATH)
    n_total = len(df)
    df = select_split(df, "all" if getattr(args, "use_all_rows", False) else "train")
    if args.modality == "t2w" and "t2w_valid" in df.columns:
        n_before = len(df)
        df = df[df["t2w_valid"] == 1]
        print(f"[t2w] dropped {n_before - len(df)} rows with t2w_valid == 0")
    df = df.reset_index(drop=True)
    print(f"CSV rows: {n_total} total -> {len(df)} used for training "
          f"({df['patient_id'].nunique() if 'patient_id' in df.columns else '?'} patients)")
    if len(df) == 0:
        raise RuntimeError("No training rows left after filtering - check the 'split' / 't2w_valid' columns.")
    args.n_train_rows = len(df)  # [1mm] recorded in the checkpoint's data_identity
    if "patient_n_crops" not in df.columns:
        df["patient_n_crops"] = df.groupby("patient_id")["patient_id"].transform("size")
    train_CSVdataset = CSVDataset(src=df, col_names=col_names, col_types=col_types)
    print(f"Number of training images: {len(train_CSVdataset)}")
    cache_rate = float(getattr(args, "cache_rate", 1.0))
    warnings.warn(f"CacheDataset cache_rate={cache_rate}. Reduce --cache_rate if the process runs out of memory.")

    train_ds = CacheDataset(
        data=train_CSVdataset,
        transform=train_transforms,
        cache_rate=cache_rate,
        copy_cache=False,
        progress=True,
        num_workers=NUM_WORKERS,
    )
    # num_workers=0 by default: the original code forced 0 because with the legacy (non-ROI) pipeline
    # DataLoader worker processes re-ran the cached transforms. With the ROI pipeline the random part
    # runs after the cache, so --loader_workers > 0 is safe on Linux (fork) and speeds up batching.
    loader_workers = int(getattr(args, "loader_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), num_workers=loader_workers,
                              persistent_workers=loader_workers > 0, drop_last=False, shuffle=True,
                              collate_fn=pad_list_data_collate)
    print(f'Dataset training: number of batches: {len(train_loader)}')
    print("Leaving the data loader. Good luck!") 
    return train_loader
