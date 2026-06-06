import os
import json
import torch
import numpy as np
import nibabel as nib
from monai.transforms import (
    Compose, RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    NormalizeIntensityd, RandScaleIntensityd, RandShiftIntensityd,
    ToTensord
)
from monai.data import DataLoader, Dataset, CacheDataset
import warnings
warnings.filterwarnings("ignore")

class NibabelLoader:
    def __init__(self, keys, image_only=True):
        self.keys = keys if isinstance(keys, list) else [keys]
        self.image_only = image_only

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            fpath = d[key]
            if isinstance(fpath, list):
                arrays = []
                for fp in fpath:
                    img = nib.load(fp)
                    arr = img.get_fdata().astype(np.float32)
                    arrays.append(arr)
                stacked = np.stack(arrays, axis=0)  # (C, H, W, D)
                d[key] = stacked
            else:
                img = nib.load(fpath)
                arr = img.get_fdata().astype(np.float32)
                if key == "label":
                    # 将非法标签值（>=5）映射为背景（0）
                    arr = (arr * (arr < 5)).astype(np.int64)
                    d[key] = arr
                else:
                    d[key] = arr
        return d

def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)
    tr = []
    val = []
    for d in json_data.get("training", []):
        if isinstance(d["image"], str):
            case_id = d["image"]
            modalities = ["t1", "t1ce", "t2", "flair"]
            d["image"] = [os.path.join(basedir, case_id, f"{m}.nii.gz") for m in modalities]
            d["label"] = os.path.join(basedir, case_id, "seg.nii.gz")
        if d.get("fold", 0) == fold:
            val.append(d)
        else:
            tr.append(d)
    return tr, val

def get_loader(args):
    data_dir = args.data_dir
    datalist_json = args.json_list
    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=args.fold)
    
    roi_size = [args.roi_x, args.roi_y, args.roi_z]
    
    train_transform = Compose([
        NibabelLoader(keys=["image", "label"], image_only=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size,
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
        RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
    ])
    
    val_transform = Compose([
        NibabelLoader(keys=["image", "label"], image_only=True),
        ToTensord(keys=["image", "label"]),
    ])
    
    train_ds = CacheDataset(data=train_files, transform=train_transform, cache_rate=1.0, num_workers=args.workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    
    val_ds = CacheDataset(data=validation_files, transform=val_transform, cache_rate=1.0, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    return train_loader, val_loader
