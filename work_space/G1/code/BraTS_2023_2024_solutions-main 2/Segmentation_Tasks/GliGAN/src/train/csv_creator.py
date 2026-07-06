"""CSV creator with connected-component analysis for multi-lesion tumours.

Each connected component in the label becomes an independent CSV row,
with its own bbox, centre of mass, and lesion volume. Adjacent lesions
(centroid distance < merge_dist voxels) are merged into a single crop.

Supports grouped train/val split: all lesions of the same patient go to
the same split. Use --val_patients to specify validation patient IDs.
"""

import os
import csv
from collections import Counter
import numpy as np
from scipy import ndimage
import nibabel as nib
import argparse


def connected_component_analysis(mask_binary):
    """Return list of (cc_mask, centroid_voxel, bbox) per connected component."""
    structure = np.ones((3, 3, 3), dtype=np.int16)
    labeled, n_cc = ndimage.label(mask_binary, structure=structure)
    components = []
    for cc_id in range(1, n_cc + 1):
        cc_mask = (labeled == cc_id)
        coords = np.argwhere(cc_mask)
        centroid = coords.mean(axis=0)  # (z, y, x)
        z_min, z_max = coords[:, 0].min(), coords[:, 0].max() + 1
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max() + 1
        x_min, x_max = coords[:, 2].min(), coords[:, 2].max() + 1
        components.append({
            'cc_mask': cc_mask,
            'centroid': (round(centroid[2]), round(centroid[1]), round(centroid[0])),  # (x, y, z)
            'bbox': (x_min, x_max, y_min, y_max, z_min, z_max),
            'n_voxels': int(cc_mask.sum()),
        })
    return components


def merge_nearby_lesions(components, merge_dist=16, crop_size=64):
    """Merge lesions whose centroid distance < merge_dist voxels.
    Does NOT merge if the merged bbox would exceed crop_size in any dimension."""
    if len(components) <= 1:
        return components
    merged = []
    used = [False] * len(components)
    for i, ci in enumerate(components):
        if used[i]:
            continue
        group = [ci]
        used[i] = True
        for j, cj in enumerate(components):
            if used[j]:
                continue
            dx = ci['centroid'][0] - cj['centroid'][0]
            dy = ci['centroid'][1] - cj['centroid'][1]
            dz = ci['centroid'][2] - cj['centroid'][2]
            if np.sqrt(dx*dx + dy*dy + dz*dz) < merge_dist:
                # Check if merging would exceed crop_size
                x_mins_tmp = [g['bbox'][0] for g in group] + [cj['bbox'][0]]
                x_maxs_tmp = [g['bbox'][1] for g in group] + [cj['bbox'][1]]
                y_mins_tmp = [g['bbox'][2] for g in group] + [cj['bbox'][2]]
                y_maxs_tmp = [g['bbox'][3] for g in group] + [cj['bbox'][3]]
                z_mins_tmp = [g['bbox'][4] for g in group] + [cj['bbox'][4]]
                z_maxs_tmp = [g['bbox'][5] for g in group] + [cj['bbox'][5]]
                merged_x = max(x_maxs_tmp) - min(x_mins_tmp)
                merged_y = max(y_maxs_tmp) - min(y_mins_tmp)
                merged_z = max(z_maxs_tmp) - min(z_mins_tmp)
                if merged_x <= crop_size and merged_y <= crop_size and merged_z <= crop_size:
                    group.append(cj)
                    used[j] = True
        if len(group) == 1:
            merged.append(ci)
        else:
            x_mins = [g['bbox'][0] for g in group]
            x_maxs = [g['bbox'][1] for g in group]
            y_mins = [g['bbox'][2] for g in group]
            y_maxs = [g['bbox'][3] for g in group]
            z_mins = [g['bbox'][4] for g in group]
            z_maxs = [g['bbox'][5] for g in group]
            x_min, x_max = min(x_mins), max(x_maxs)
            y_min, y_max = min(y_mins), max(y_maxs)
            z_min, z_max = min(z_mins), max(z_maxs)
            n_voxels = sum(g['n_voxels'] for g in group)
            cx = round(np.mean([g['centroid'][0] for g in group]))
            cy = round(np.mean([g['centroid'][1] for g in group]))
            cz = round(np.mean([g['centroid'][2] for g in group]))
            merged.append({
                'cc_mask': group[0]['cc_mask'],
                'centroid': (cx, cy, cz),
                'bbox': (x_min, x_max, y_min, y_max, z_min, z_max),
                'n_voxels': n_voxels,
            })
    return merged


def tile_oversized_lesion(cc, mask_binary, crop_size=64, stride=56):
    """Split an oversized lesion into overlapping crop_size^3 tiles.

    Each tile covers part of the lesion with overlap between adjacent tiles.
    Empty tiles (no tumour voxels) are skipped. Per-tile n_voxels is computed
    from the binary mask so small-lesion loss weighting works correctly.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = cc['bbox']
    sx, sy, sz = x_max - x_min, y_max - y_min, z_max - z_min

    xs = list(range(0, sx, stride))
    ys = list(range(0, sy, stride))
    zs = list(range(0, sz, stride))

    for tx in xs:
        for ty in ys:
            for tz in zs:
                gx_min = x_min + tx
                gy_min = y_min + ty
                gz_min = z_min + tz
                gx_max = min(gx_min + crop_size, x_max)
                gy_max = min(gy_min + crop_size, y_max)
                gz_max = min(gz_min + crop_size, z_max)

                # Per-tile n_voxels from binary mask (z, y, x)
                tile_n = int(mask_binary[gz_min:gz_max, gy_min:gy_max, gx_min:gx_max].sum())
                if tile_n == 0:
                    continue

                yield {
                    'centroid': ((gx_min + gx_max) // 2,
                                 (gy_min + gy_max) // 2,
                                 (gz_min + gz_max) // 2),
                    'bbox': (gx_min, gx_max, gy_min, gy_max, gz_min, gz_max),
                    'n_voxels': tile_n,
                }


def get_training_dict(args, datadir):
    training = []
    if not os.path.isdir(datadir):
        raise FileNotFoundError(f"Data directory not found: {datadir}")

    for sub_dir in sorted(os.listdir(datadir)):
        case_dir = os.path.join(datadir, sub_dir)
        if sub_dir.startswith(".") or not os.path.isdir(case_dir):
            continue

        images = []
        label = None
        for file in sorted(os.listdir(case_dir)):
            file_path = os.path.join(case_dir, file)
            if file.startswith(".") or not os.path.isfile(file_path):
                continue
            if file.endswith("seg.nii.gz") or file.endswith(args.seg_ending):
                label = file_path
            elif file.endswith((".nii", ".nii.gz")):
                images.append(file_path)
            else:
                print(f"  [WARN] Ignoring non-NIfTI file: {file_path}")
        if label is not None:
            dict_entry = {"image": images, "label": label}
            training.append(dict_entry)
    training_dict = {"training": training}
    return training_dict


def modal_paths(args, mask_path):
    scan_path_t1c = None
    scan_path_t2w = None
    scan_path_t2f = None
    scan_path_t1n = None
    for scan_paths in mask_path['image']:
        if args.t1c_ending in scan_paths:
            scan_path_t1c = scan_paths
        elif args.t2w_ending in scan_paths:
            scan_path_t2w = scan_paths
        elif args.t2f_ending in scan_paths:
            scan_path_t2f = scan_paths
        elif args.t1n_ending in scan_paths:
            scan_path_t1n = scan_paths
    label_path = mask_path['label']
    return scan_path_t1c, scan_path_t2w, scan_path_t2f, scan_path_t1n, label_path


def create_csv(args, DATASET_NAME, CSV_PATH, DATADIR):
    """Create CSV with one row per connected component (lesion)."""
    header = ['patient_id', 'lesion_id', 'scan_t1c', 'scan_t2w', 'scan_t2f', 'scan_t1n',
              'label', 'center_x', 'center_y', 'center_z',
              'x_extreme_min', 'x_extreme_max', 'y_extreme_min', 'y_extreme_max',
              'z_extreme_min', 'z_extreme_max', 'x_size', 'y_size', 'z_size',
              'n_voxels', 'patient_n_crops', 'split']

    training_dict = get_training_dict(args, DATADIR)
    training = training_dict['training']

    # Build val patient ID set
    val_patient_ids = set()
    if args.val_patients:
        val_patient_ids = {pid.strip() for pid in args.val_patients.split(',') if pid.strip()}

    crop_size = args.crop_size
    merge_dist = args.merge_dist
    total_lesions = 0
    skipped_missing_modalities = 0
    rows = []
    patient_id_idx = header.index('patient_id')
    patient_n_crops_idx = header.index('patient_n_crops')

    for mask_path in training:
        # Extract patient ID
        if ("brats" in DATASET_NAME.lower()) and ("goat" in DATASET_NAME.lower()) and ("2024" in DATASET_NAME.lower()):
            patient_id = os.path.basename(os.path.dirname(mask_path['label']))[-5:]
        elif ("brats" in DATASET_NAME.lower()) and ("2023" in DATASET_NAME.lower()) and ("goat" not in DATASET_NAME.lower()) and ("meningioma" not in DATASET_NAME.lower()):
            patient_id = os.path.basename(os.path.dirname(mask_path['label']))[-9:]
        elif ("brats" in DATASET_NAME.lower()) and ("2024" in DATASET_NAME.lower()) and ("goat" not in DATASET_NAME.lower()) and ("meningioma" not in DATASET_NAME.lower()):
            patient_id = os.path.basename(os.path.dirname(mask_path['label']))[-9:]
        elif ("brats" in DATASET_NAME.lower()) and ("meningioma" in DATASET_NAME.lower()):
            patient_id = os.path.basename(os.path.dirname(mask_path['label']))[-6:]
        else:
            raise ValueError("Unknown dataset")

        # Determine split
        split = "val" if patient_id in val_patient_ids else "train"

        scan_path_t1c, scan_path_t2w, scan_path_t2f, scan_path_t1n, label_path = modal_paths(args, mask_path)
        required_paths = {
            "t1c": scan_path_t1c,
            "t2w": scan_path_t2w,
            "t2f": scan_path_t2f,
            "t1n": scan_path_t1n,
            "seg": label_path,
        }
        missing = [name for name, path in required_paths.items() if path is None]
        if missing:
            skipped_missing_modalities += 1
            print(f"  [WARN] Case {patient_id}: missing {', '.join(missing)}, skipping")
            continue

        print(f"Doing case ID: {patient_id} ({split})")
        mask = nib.load(mask_path['label'])
        mask_data = np.asarray(mask.get_fdata())

        # Binary mask for CC analysis (any non-zero label = tumour)
        mask_binary = mask_data > 0.5

        if not np.any(mask_binary):
            print(f"  Case {patient_id}: no tumour found, skipping")
            continue

        # Connected component analysis + merge
        cc_list = connected_component_analysis(mask_binary)
        cc_list = merge_nearby_lesions(cc_list, merge_dist=merge_dist, crop_size=crop_size)
        print(f"  Found {len(cc_list)} lesion(s) (after merging, dist < {merge_dist} vox)")

        for idx, cc in enumerate(cc_list):
            x_min, x_max, y_min, y_max, z_min, z_max = cc['bbox']
            x_size = x_max - x_min
            y_size = y_max - y_min
            z_size = z_max - z_min

            oversized = (x_size > crop_size or y_size > crop_size or z_size > crop_size)
            if oversized:
                tiles = list(tile_oversized_lesion(cc, mask_binary, crop_size))
                print(f"    Lesion {idx}: bbox ({x_size},{y_size},{z_size}) > {crop_size}, "
                      f"split into {len(tiles)} tiles")
                for ti, tile in enumerate(tiles):
                    tx_min, tx_max, ty_min, ty_max, tz_min, tz_max = tile['bbox']
                    tcx, tcy, tcz = tile['centroid']
                    total_lesions += 1
                    row = [patient_id, f"{patient_id}_cc{idx}_t{ti}",
                           scan_path_t1c, scan_path_t2w, scan_path_t2f, scan_path_t1n,
                           label_path, tcx, tcy, tcz,
                           tx_min, tx_max, ty_min, ty_max, tz_min, tz_max,
                           tx_max - tx_min, ty_max - ty_min, tz_max - tz_min,
                           tile['n_voxels'], 0, split]
                    rows.append(row)
            else:
                cx, cy, cz = cc['centroid']
                total_lesions += 1
                row = [patient_id, f"{patient_id}_cc{idx}",
                       scan_path_t1c, scan_path_t2w, scan_path_t2f, scan_path_t1n,
                       label_path, cx, cy, cz,
                       x_min, x_max, y_min, y_max, z_min, z_max,
                       x_size, y_size, z_size, cc['n_voxels'],
                       0, split]
                rows.append(row)

    patient_crop_counts = Counter(row[patient_id_idx] for row in rows)
    for row in rows:
        row[patient_n_crops_idx] = patient_crop_counts[row[patient_id_idx]]

    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nDone. {total_lesions} lesions saved to {CSV_PATH}")
    if skipped_missing_modalities:
        print(f"Skipped {skipped_missing_modalities} case(s) because required modalities were missing.")
    return training


def __main__():
    parser = argparse.ArgumentParser(description="CSV creator with multi-lesion support")
    parser.add_argument("--logdir", default="test", type=str, help="Directory to save the CSV")
    parser.add_argument("--dataset", type=str, help="Dataset name. E.g. Brats_2023, Brats_2024")
    parser.add_argument("--datadir", type=str, help="Complete or relative path of the dataset")
    parser.add_argument("--debug", default="True", type=str, help="Show debug output")
    parser.add_argument("--csv_path", default="", type=str, help="Path to the CSV")
    parser.add_argument("--seg_ending", default="seg.nii.gz", type=str, help="Seg file ending")
    parser.add_argument("--t1n_ending", default="t1n.nii.gz", type=str, help="t1n file ending")
    parser.add_argument("--t1c_ending", default="t1c.nii.gz", type=str, help="t1c file ending")
    parser.add_argument("--t2w_ending", default="t2w.nii.gz", type=str, help="t2w file ending")
    parser.add_argument("--t2f_ending", default="t2f.nii.gz", type=str, help="t2f file ending")
    parser.add_argument("--crop_size", default=64, type=int,
                        help="Target crop/pad size (64=default, 96=glioma)")
    parser.add_argument("--merge_dist", default=16, type=int,
                        help="Merge lesions closer than this distance (voxels)")
    parser.add_argument("--val_patients", default="", type=str,
                        help="Comma-separated list of patient IDs for validation set")
    args = parser.parse_args()

    HOME_DIR = f'../../Checkpoint/{args.logdir}'
    if not os.path.exists(HOME_DIR):
        os.makedirs(HOME_DIR)
        print(f"Directory {HOME_DIR} created")
    else:
        print(f"Directory {HOME_DIR} already exists")
    if not os.path.exists(f"{HOME_DIR}/debug"):
        os.makedirs(f"{HOME_DIR}/debug")
        print(f"Directory {HOME_DIR}/debug created")
    else:
        print(f"Directory {HOME_DIR}/debug already exists")

    if args.csv_path == "":
        CSV_PATH = f'../../Checkpoint/{args.logdir}/{args.logdir}.csv'
    else:
        CSV_PATH = args.csv_path
    print(f"CSV_PATH: {CSV_PATH}")

    training = create_csv(args=args, DATASET_NAME=args.dataset, CSV_PATH=CSV_PATH, DATADIR=args.datadir)

    if args.debug == "True":
        print("####################################")
        print(f"Output for debug")
        print(f"Number of patients in training dict: {len(training)}")
        import pandas as pd
        df = pd.read_csv(CSV_PATH)
        print(f"Number of rows (lesions) in CSV: {len(df)}")
        if len(df) == 0:
            print("No valid rows in CSV; skip sample visualization.")
            return
        train_rows = (df['split'] == 'train').sum()
        val_rows = (df['split'] == 'val').sum()
        print(f"Train lesions: {train_rows}, Val lesions: {val_rows}")
        print(f"\n### First 5 rows ###")
        print(df.head())
        print(f"\n### Size distribution ###")
        for label, lo, hi in [("Tiny (<27mm^3)", 0, 27), ("Small", 27, 100),
                                ("Medium", 100, 1000), ("Large", 1000, 1e9)]:
            print(f"  {label}: {((df['n_voxels'] > lo) & (df['n_voxels'] <= hi)).sum()}")
        print(f"\n### Sample paths ###")
        print(f"t1c: {df['scan_t1c'][0]}")
        print(f"label: {df['label'][0]}")
        print(f"center: ({df['center_x'][0]}, {df['center_y'][0]}, {df['center_z'][0]})")

        # Visualize a sample
        import matplotlib.pyplot as plt
        def visualize_sample(idx, slice_num, types=('scan_t1c', 'scan_t2w', 'scan_t2f', 'scan_t1n')):
            plt.figure(figsize=(16, 5))
            for i, t in enumerate(types, 1):
                data = nib.load(df[t][idx])
                data = np.asarray(data.get_fdata())
                plt.subplot(1, 4, i)
                plt.imshow(data[:, :, slice_num], cmap='gray')
                plt.title(f'{t}', fontsize=16)
                plt.axis('off')
            plt.suptitle(f'idx: {idx}, patient: {df["patient_id"][idx]}, lesion: {df["lesion_id"][idx]}', fontsize=14)
            plt.savefig(f"{HOME_DIR}/debug/sample_{slice_num}.png", format='png')
            plt.close()

        if ("brats" in args.dataset.lower()) and ("meningioma" in args.dataset.lower()):
            types = ['scan_t1c']
        else:
            types = ('scan_t1c', 'scan_t2w', 'scan_t2f', 'scan_t1n')
        for s in range(5):
            visualize_sample(idx=0, slice_num=100 + s * 5, types=types)


if __name__ == "__main__":
    __main__()
    print("Finished!")
