#!/usr/bin/env python3
"""Generate G2 pre-training manifests, QC summaries, and templates.

This script reads the external BraTS-MET data in-place and writes small
CSV/JSON/Markdown artifacts under work_space/G2/results. It does not copy
NIfTI volumes into the repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage


LABELS = {0: "background", 1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
MODALITIES = {
    "t1n": "t1n",
    "t1c": "t1c",
    "t2w": "t2w",
    "t2f": "t2f",
}
GLIGAN_KEYS = {
    "scan_t1ce": "t1c",
    "scan_t2": "t2w",
    "scan_flair": "t2f",
    "scan_t1": "t1n",
}
RUN_DATE = datetime.now().strftime("%Y-%m-%d")


def as_posix(path: Path | str | None) -> str:
    return "" if path is None else str(path)


def ensure_dirs(results_root: Path) -> dict[str, Path]:
    dirs = {
        "manifests": results_root / "manifests",
        "stats": results_root / "stats",
        "qc": results_root / "qc",
        "splits": results_root / "splits",
        "reports": results_root / "reports",
        "nnunet_raw": results_root / "nnunet_raw" / "Dataset260_BraTS2026_MET_RealOnly",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    (dirs["nnunet_raw"] / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dirs["nnunet_raw"] / "labelsTr").mkdir(parents=True, exist_ok=True)
    return dirs


def write_readme_files(results_root: Path, dirs: dict[str, Path]) -> None:
    readmes = {
        results_root / "README.md": "# G2 Results\n\n本目录保存 G2 在模型训练完成前已经能在本机完成的小型结果文件。NIfTI 大数据、nnU-Net 预处理结果和生成模型输出不复制到本仓库。\n",
        dirs["manifests"] / "README.md": "# Manifests\n\n保存真实数据、corrected labels、GliGAN 兼容 source CSV、nnU-Net 映射表等小型清单。\n",
        dirs["stats"] / "README.md": "# Stats\n\n保存真实 label/lesion 分布和 synthetic target distribution。\n",
        dirs["qc"] / "README.md": "# QC\n\n保存 synthetic data 质量控制规则和指标模板。\n",
        dirs["splits"] / "README.md": "# Splits\n\n保存固定真实验证 fold，供 real-only 和 real+synth 消融复用。\n",
        dirs["reports"] / "README.md": "# Reports\n\n保存路径检查、数据 QC、执行总结和团队沟通文档源稿。\n",
        dirs["nnunet_raw"] / "README.md": "# Dataset260_BraTS2026_MET_RealOnly\n\n本目录当前只保存 `dataset.json` 和映射说明，不复制或软链接全量 NIfTI。需要正式训练时，由 S1/S2 根据 `manifests/nnunet_case_mapping_realonly.csv` 在训练机器上物化数据集并运行 nnU-Net 预处理。\n",
    }
    for path, text in readmes.items():
        path.write_text(text, encoding="utf-8")


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def human_size(num: int) -> str:
    value = float(num)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num} B"


def nifti_meta(path: Path) -> dict[str, object]:
    img = nib.load(str(path))
    header = img.header
    affine = np.asarray(img.affine, dtype=np.float64)
    affine_hash = hashlib.sha256(np.round(affine, 6).tobytes()).hexdigest()[:16]
    return {
        "shape": tuple(int(v) for v in img.shape[:3]),
        "spacing": tuple(float(v) for v in header.get_zooms()[:3]),
        "dtype": str(header.get_data_dtype()),
        "affine_hash": affine_hash,
        "affine": affine,
    }


def find_case_dirs(root: Path) -> list[Path]:
    case_dirs: list[Path] = []
    for path in root.rglob("BraTS-MET-*"):
        if path.is_dir():
            case_dirs.append(path)
    return sorted(case_dirs, key=lambda p: p.name)


def find_modality_files(case_dir: Path, include_seg: bool) -> dict[str, Path | None]:
    files: dict[str, Path | None] = {}
    for mod in MODALITIES:
        matches = sorted(case_dir.glob(f"*-{mod}.nii.gz"))
        files[mod] = matches[0] if matches else None
    if include_seg:
        matches = sorted(case_dir.glob("*-seg.nii.gz"))
        files["seg"] = matches[0] if matches else None
    return files


def unique_label_values(path: Path) -> tuple[list[int | float], bool, str]:
    try:
        img = nib.load(str(path))
        arr = np.asanyarray(img.dataobj)
        finite = bool(np.isfinite(arr).all())
        unique = np.unique(arr)
        values: list[int | float] = []
        for item in unique.tolist():
            if isinstance(item, float) and item.is_integer():
                values.append(int(item))
            else:
                values.append(item)
        return values, finite, ""
    except Exception as exc:  # noqa: BLE001
        return [], False, f"{type(exc).__name__}: {exc}"


def bbox_and_center(mask: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return [0, 0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0]
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    center = np.rint(coords.mean(axis=0)).astype(int)
    size = maxs - mins
    return [int(mins[0]), int(maxs[0]), int(mins[1]), int(maxs[1]), int(mins[2]), int(maxs[2])], [int(center[0]), int(center[1]), int(center[2])], [int(size[0]), int(size[1]), int(size[2])]


def scan_training(train_root: Path) -> pd.DataFrame:
    rows = []
    for case_dir in find_case_dirs(train_root):
        case_id = case_dir.name
        files = find_modality_files(case_dir, include_seg=True)
        row: dict[str, object] = {
            "case_id": case_id,
            "split_source": "train_ucsd" if "UCSD - Training" in case_dir.parts else "train_top_level",
            "case_dir": as_posix(case_dir),
        }
        metas: dict[str, dict[str, object]] = {}
        reasons = []
        for mod in ["t1n", "t1c", "t2w", "t2f", "seg"]:
            path = files.get(mod)
            row[f"{'raw_seg' if mod == 'seg' else mod}_path"] = as_posix(path)
            row[f"has_{mod}"] = bool(path and path.exists())
            if not path:
                reasons.append(f"missing_{mod}")
                continue
            try:
                meta = nifti_meta(path)
                metas[mod] = meta
                row[f"shape_{mod}"] = "x".join(map(str, meta["shape"]))
                row[f"spacing_{mod}"] = ",".join(f"{v:.6g}" for v in meta["spacing"])
                row[f"affine_hash_{mod}"] = meta["affine_hash"]
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"read_error_{mod}:{type(exc).__name__}")
                row[f"shape_{mod}"] = ""
                row[f"spacing_{mod}"] = ""
                row[f"affine_hash_{mod}"] = ""
        row["image_dtypes"] = ";".join(sorted({str(metas[m]["dtype"]) for m in ["t1n", "t1c", "t2w", "t2f"] if m in metas}))
        row["label_dtype"] = str(metas["seg"]["dtype"]) if "seg" in metas else ""
        if all(mod in metas for mod in ["t1n", "t1c", "t2w", "t2f", "seg"]):
            shapes = {metas[m]["shape"] for m in metas}
            spacings = {tuple(round(float(v), 6) for v in metas[m]["spacing"]) for m in metas}
            affines = {metas[m]["affine_hash"] for m in metas}
            if len(shapes) != 1:
                reasons.append("shape_mismatch")
            if len(spacings) != 1:
                reasons.append("spacing_mismatch")
            if len(affines) != 1:
                reasons.append("affine_hash_mismatch_warning")
        label_values, label_finite, label_error = ([], False, "missing_seg")
        if files.get("seg"):
            label_values, label_finite, label_error = unique_label_values(files["seg"])  # type: ignore[arg-type]
        if label_error:
            reasons.append(f"label_read_error:{label_error}")
        illegal = [v for v in label_values if v not in LABELS]
        if illegal:
            reasons.append(f"illegal_label_values:{illegal}")
        row["labels_present"] = ";".join(map(str, label_values))
        row["has_nan_or_inf"] = not label_finite
        row["image_nan_inf_check"] = "deferred_full_volume_io_36GB"
        blocking_reasons = [
            r for r in reasons
            if not r.startswith("illegal_label_values") and not r.endswith("_warning")
        ]
        row["basic_qc_pass"] = len(blocking_reasons) == 0
        row["basic_qc_reason"] = "pass" if not reasons else ";".join(reasons)
        rows.append(row)
    return pd.DataFrame(rows)


def scan_validation(validation_root: Path) -> pd.DataFrame:
    rows = []
    for case_dir in find_case_dirs(validation_root):
        case_id = case_dir.name
        files = find_modality_files(case_dir, include_seg=False)
        row: dict[str, object] = {"case_id": case_id, "case_dir": as_posix(case_dir)}
        metas: dict[str, dict[str, object]] = {}
        reasons = []
        for mod in ["t1n", "t1c", "t2w", "t2f"]:
            path = files.get(mod)
            row[f"{mod}_path"] = as_posix(path)
            row[f"has_{mod}"] = bool(path and path.exists())
            if not path:
                reasons.append(f"missing_{mod}")
                continue
            try:
                metas[mod] = nifti_meta(path)
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"read_error_{mod}:{type(exc).__name__}")
        if metas:
            row["shape"] = ";".join(sorted({"x".join(map(str, meta["shape"])) for meta in metas.values()}))
            row["spacing"] = ";".join(sorted({",".join(f"{v:.6g}" for v in meta["spacing"]) for meta in metas.values()}))
            row["affine_hash"] = ";".join(sorted({str(meta["affine_hash"]) for meta in metas.values()}))
            row["image_dtypes"] = ";".join(sorted({str(meta["dtype"]) for meta in metas.values()}))
        else:
            row["shape"] = ""
            row["spacing"] = ""
            row["affine_hash"] = ""
            row["image_dtypes"] = ""
        if len({meta["shape"] for meta in metas.values()}) > 1:
            reasons.append("shape_mismatch")
        if len({tuple(round(float(v), 6) for v in meta["spacing"]) for meta in metas.values()}) > 1:
            reasons.append("spacing_mismatch")
        row["basic_qc_pass"] = len(reasons) == 0
        row["basic_qc_reason"] = "pass" if not reasons else ";".join(reasons)
        row["allowed_as_synthetic_source"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def apply_corrected_labels(raw_df: pd.DataFrame, corrected_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    corrected_files = sorted(corrected_root.glob("*-seg.nii.gz")) if corrected_root.exists() else []
    corrected_by_case = {path.name.replace("-seg.nii.gz", ""): path for path in corrected_files}
    overlay_rows = []
    final_df = raw_df.copy()
    final_df["effective_seg_path"] = final_df["raw_seg_path"]
    final_df["label_source"] = "raw"
    final_df["has_corrected_label"] = False
    final_df["has_illegal_label_after_overlay"] = False
    final_df["illegal_label_values_after_overlay"] = ""
    final_df["final_qc_pass"] = False
    final_df["final_qc_reason"] = ""

    raw_by_case = {row.case_id: row for row in raw_df.itertuples(index=False)}
    for case_id, corrected_path in corrected_by_case.items():
        raw_row = raw_by_case.get(case_id)
        raw_seg_path = Path(raw_row.raw_seg_path) if raw_row is not None and raw_row.raw_seg_path else None
        raw_values, _, _ = unique_label_values(raw_seg_path) if raw_seg_path else ([], False, "missing")
        corrected_values, _, corrected_error = unique_label_values(corrected_path)
        raw_meta = nifti_meta(raw_seg_path) if raw_seg_path else None
        corrected_meta = nifti_meta(corrected_path)
        applied = raw_row is not None and not corrected_error and raw_meta is not None and raw_meta["shape"] == corrected_meta["shape"]
        overlay_rows.append({
            "case_id": case_id,
            "raw_seg_path": as_posix(raw_seg_path),
            "corrected_seg_path": as_posix(corrected_path),
            "raw_unique_labels": ";".join(map(str, raw_values)),
            "corrected_unique_labels": ";".join(map(str, corrected_values)),
            "raw_shape": "x".join(map(str, raw_meta["shape"])) if raw_meta else "",
            "corrected_shape": "x".join(map(str, corrected_meta["shape"])),
            "raw_spacing": ",".join(f"{v:.6g}" for v in raw_meta["spacing"]) if raw_meta else "",
            "corrected_spacing": ",".join(f"{v:.6g}" for v in corrected_meta["spacing"]),
            "raw_affine_hash": raw_meta["affine_hash"] if raw_meta else "",
            "corrected_affine_hash": corrected_meta["affine_hash"],
            "applied": applied,
            "apply_reason": "shape_match" if applied else ("source_case_not_found" if raw_row is None else "corrected_label_error_or_shape_mismatch"),
            "notes": "",
        })
        if applied:
            idx = final_df.index[final_df["case_id"] == case_id]
            final_df.loc[idx, "effective_seg_path"] = as_posix(corrected_path)
            final_df.loc[idx, "label_source"] = "corrected"
            final_df.loc[idx, "has_corrected_label"] = True

    for idx, row in final_df.iterrows():
        values, finite, err = unique_label_values(Path(row["effective_seg_path"])) if row["effective_seg_path"] else ([], False, "missing")
        illegal = [v for v in values if v not in LABELS]
        reasons = []
        if not row["basic_qc_pass"]:
            raw_reasons = str(row["basic_qc_reason"])
            filtered = [part for part in raw_reasons.split(";") if not part.startswith("illegal_label_values")]
            filtered = [part for part in filtered if not part.endswith("_warning") and part != "affine_hash_mismatch"]
            reasons.extend([part for part in filtered if part and part != "pass"])
        if err:
            reasons.append(f"effective_label_read_error:{err}")
        if not finite:
            reasons.append("effective_label_nan_or_inf")
        if illegal:
            reasons.append(f"illegal_label_values_after_overlay:{illegal}")
        final_df.loc[idx, "labels_present_after_overlay"] = ";".join(map(str, values))
        final_df.loc[idx, "has_illegal_label_after_overlay"] = bool(illegal)
        final_df.loc[idx, "illegal_label_values_after_overlay"] = ";".join(map(str, illegal))
        final_df.loc[idx, "final_qc_pass"] = len(reasons) == 0
        final_df.loc[idx, "final_qc_reason"] = "pass" if not reasons else ";".join(reasons)
    return pd.DataFrame(overlay_rows), final_df


def label_stats(final_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    label_rows = []
    lesion_rows = []
    summary: dict[str, object] = {}
    pass_df = final_df[final_df["final_qc_pass"] == True].copy()  # noqa: E712
    label_combo_counter: Counter[str] = Counter()
    lesion_bucket_counter: Counter[str] = Counter()
    lesions_per_case: Counter[int] = Counter()
    case_many_lesions: list[tuple[str, int]] = []
    tiny_cases: list[str] = []
    for case_index, row in enumerate(pass_df.itertuples(index=False), start=1):
        if case_index % 100 == 0:
            print(f"[label_stats] processed {case_index}/{len(pass_df)} cases", flush=True)
        seg_path = Path(row.effective_seg_path)
        img = nib.load(str(seg_path))
        seg = np.asanyarray(img.dataobj)
        spacing = tuple(float(v) for v in img.header.get_zooms()[:3])
        voxel_volume = float(np.prod(spacing))
        unique_values, unique_counts = np.unique(seg, return_counts=True)
        count_map = {int(v): int(c) for v, c in zip(unique_values.tolist(), unique_counts.tolist()) if float(v).is_integer()}
        counts = {label: count_map.get(label, 0) for label in [1, 2, 3, 4]}
        volumes = {label: counts[label] * voxel_volume for label in counts}
        present = [LABELS[label] for label in [1, 2, 3, 4] if counts[label] > 0]
        combo = "+".join(present) if present else "none"
        label_combo_counter[combo] += 1
        label_rows.append({
            "case_id": row.case_id,
            "label_1_voxels": counts[1],
            "label_2_voxels": counts[2],
            "label_3_voxels": counts[3],
            "label_4_voxels": counts[4],
            "label_1_volume_mm3": round(volumes[1], 3),
            "label_2_volume_mm3": round(volumes[2], 3),
            "label_3_volume_mm3": round(volumes[3], 3),
            "label_4_volume_mm3": round(volumes[4], 3),
            "has_NETC": counts[1] > 0,
            "has_SNFH": counts[2] > 0,
            "has_ET": counts[3] > 0,
            "has_RC": counts[4] > 0,
            "label_combination": combo,
        })
        lesion_mask = np.isin(seg, [1, 3, 4])
        lesion_coords = np.argwhere(lesion_mask)
        if lesion_coords.size == 0:
            lesions_per_case[0] += 1
            continue
        crop_mins = lesion_coords.min(axis=0)
        crop_maxs = lesion_coords.max(axis=0) + 1
        crop_slices = tuple(slice(int(crop_mins[axis]), int(crop_maxs[axis])) for axis in range(3))
        lesion_crop = lesion_mask[crop_slices]
        seg_crop = seg[crop_slices]
        components, num_components = ndimage.label(lesion_crop, structure=np.ones((3, 3, 3), dtype=np.uint8))
        lesions_per_case[int(num_components)] += 1
        if num_components >= 5:
            case_many_lesions.append((row.case_id, int(num_components)))
        for lesion_id, slc in enumerate(ndimage.find_objects(components), start=1):
            if slc is None:
                continue
            comp_mask_crop = components[slc] == lesion_id
            voxels = int(comp_mask_crop.sum())
            volume = voxels * voxel_volume
            if volume < 27:
                bucket = "tiny_lt_27mm3"
                tiny_cases.append(row.case_id)
            elif volume <= 275:
                bucket = "small_27_to_275mm3"
            else:
                bucket = "large_gt_275mm3"
            lesion_bucket_counter[bucket] += 1
            local_coords = np.argwhere(comp_mask_crop)
            local_mins = local_coords.min(axis=0)
            local_maxs = local_coords.max(axis=0) + 1
            global_mins = local_mins + crop_mins + np.array([slc[0].start, slc[1].start, slc[2].start])
            global_maxs = local_maxs + crop_mins + np.array([slc[0].start, slc[1].start, slc[2].start])
            global_center = np.rint(local_coords.mean(axis=0) + crop_mins + np.array([slc[0].start, slc[1].start, slc[2].start])).astype(int)
            bbox = [int(global_mins[0]), int(global_maxs[0]), int(global_mins[1]), int(global_maxs[1]), int(global_mins[2]), int(global_maxs[2])]
            center = [int(global_center[0]), int(global_center[1]), int(global_center[2])]
            comp_seg_crop = seg_crop[slc]
            comp_values = sorted(int(v) for v in np.unique(comp_seg_crop[comp_mask_crop]).tolist() if int(v) != 0)
            lesion_rows.append({
                "case_id": row.case_id,
                "lesion_id": lesion_id,
                "component_labels": ";".join(map(str, comp_values)),
                "component_voxels": voxels,
                "component_volume_mm3": round(volume, 3),
                "volume_bucket": bucket,
                "bbox_i_min": bbox[0],
                "bbox_i_max": bbox[1],
                "bbox_j_min": bbox[2],
                "bbox_j_max": bbox[3],
                "bbox_k_min": bbox[4],
                "bbox_k_max": bbox[5],
                "center_i": center[0],
                "center_j": center[1],
                "center_k": center[2],
                "has_ET": 3 in comp_values,
                "has_NETC": 1 in comp_values,
                "has_SNFH": False,
                "has_RC": 4 in comp_values,
            })
    summary["final_qc_pass_cases"] = int(len(pass_df))
    summary["cases_with_any_label"] = int(sum(1 for item in label_rows if item["label_combination"] != "none"))
    summary["label_combo_counter"] = dict(label_combo_counter)
    summary["lesion_bucket_counter"] = dict(lesion_bucket_counter)
    summary["lesions_per_case"] = dict(lesions_per_case)
    summary["many_lesion_cases_top20"] = sorted(case_many_lesions, key=lambda x: x[1], reverse=True)[:20]
    summary["tiny_case_count_unique"] = len(set(tiny_cases))
    return pd.DataFrame(label_rows), pd.DataFrame(lesion_rows), summary


def gligan_source_csv(final_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in final_df.itertuples(index=False):
        if not bool(row.final_qc_pass):
            continue
        seg_path = Path(row.effective_seg_path)
        img = nib.load(str(seg_path))
        seg = np.asanyarray(img.dataobj)
        mask = seg > 0
        bbox, center, size = bbox_and_center(mask)
        usable = all(v <= 96 for v in size) and mask.any()
        rows.append({
            "id": row.case_id,
            "scan_t1ce": row.t1c_path,
            "scan_t2": row.t2w_path,
            "scan_flair": row.t2f_path,
            "scan_t1": row.t1n_path,
            "label": row.effective_seg_path,
            "center_x": center[0],
            "center_y": center[1],
            "center_z": center[2],
            "x_extreme_min": bbox[0],
            "x_extreme_max": bbox[1],
            "y_extreme_min": bbox[2],
            "y_extreme_max": bbox[3],
            "z_extreme_min": bbox[4],
            "z_extreme_max": bbox[5],
            "x_size": size[0],
            "y_size": size[1],
            "z_size": size[2],
            "case_id": row.case_id,
            "source_split": "train",
            "has_corrected_label": row.has_corrected_label,
            "corrected_label_path": row.effective_seg_path if row.has_corrected_label else "",
            "label_values": row.labels_present_after_overlay,
            "lesion_component_id": "whole_positive_mask",
            "lesion_volume_mm3": "",
            "lesion_class_set": row.labels_present_after_overlay,
            "usable_for_gligan96": usable,
            "allowed_as_synthetic_source": usable,
            "exclude_reason": "" if usable else "whole_positive_bbox_exceeds_96_or_empty",
            "shape_x": str(row.shape_seg).split("x")[0] if row.shape_seg else "",
            "shape_y": str(row.shape_seg).split("x")[1] if row.shape_seg and "x" in str(row.shape_seg) else "",
            "shape_z": str(row.shape_seg).split("x")[2] if row.shape_seg and str(row.shape_seg).count("x") >= 2 else "",
            "spacing_x": str(row.spacing_seg).split(",")[0] if row.spacing_seg else "",
            "spacing_y": str(row.spacing_seg).split(",")[1] if row.spacing_seg and "," in str(row.spacing_seg) else "",
            "spacing_z": str(row.spacing_seg).split(",")[2] if row.spacing_seg and str(row.spacing_seg).count(",") >= 2 else "",
            "affine_hash": row.affine_hash_seg,
        })
    return pd.DataFrame(rows)


def nnunet_mapping(final_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    pass_df = final_df[final_df["final_qc_pass"] == True].sort_values("case_id").copy()  # noqa: E712
    rows = []
    source_to_nn = {}
    for idx, row in enumerate(pass_df.itertuples(index=False), start=1):
        nn_id = f"BraTSMET_{idx:06d}"
        source_to_nn[row.case_id] = nn_id
        rows.append({
            "nnunet_case_id": nn_id,
            "source_case_id": row.case_id,
            "t1n_source_path": row.t1n_path,
            "t1c_source_path": row.t1c_path,
            "t2w_source_path": row.t2w_path,
            "t2f_source_path": row.t2f_path,
            "seg_source_path": row.effective_seg_path,
            "label_source": row.label_source,
            "materialization_status": "deferred_no_nifti_copy_on_mac",
        })
    return pd.DataFrame(rows), source_to_nn


def stable_split(mapping_df: pd.DataFrame, val_fraction: float = 0.2) -> list[dict[str, list[str]]]:
    scored = []
    for row in mapping_df.itertuples(index=False):
        digest = hashlib.sha256(f"20260530::{row.source_case_id}".encode("utf-8")).hexdigest()
        score = int(digest[:16], 16) / float(16**16)
        scored.append((score, row.nnunet_case_id))
    scored.sort()
    val_count = int(round(len(scored) * val_fraction))
    val = sorted(case_id for _, case_id in scored[:val_count])
    train = sorted(case_id for _, case_id in scored[val_count:])
    return [{"train": train, "val": val}]


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "无。"
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            text = "" if pd.isna(value) else str(value)
            text = text.replace("|", "\\|").replace("\n", "<br>")
            values.append(text)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_dataset_json(path: Path) -> None:
    dataset = {
        "channel_names": {"0": "t1n", "1": "t1c", "2": "t2w", "3": "t2f"},
        "labels": {"background": 0, "NETC": 1, "SNFH": 2, "ET": 3, "RC": 4},
        "numTraining": 0,
        "file_ending": ".nii.gz",
        "note": "NIfTI files are not materialized on this Mac. Use nnunet_case_mapping_realonly.csv to create symlinks/copies on the training machine.",
    }
    write_json(path, dataset)


def write_templates(dirs: dict[str, Path]) -> None:
    qc_header = [
        "case_id", "source_case_id", "generation_run_id", "generator_checkpoint", "seed",
        "generation_mode", "has_all_modalities", "has_seg", "shape_consistent",
        "spacing_consistent", "affine_valid", "has_nan_or_inf", "label_values_valid",
        "label_is_integer", "empty_mask", "lesion_count", "min_lesion_volume_mm3",
        "max_lesion_volume_mm3", "tiny_lesion_count", "small_lesion_count",
        "large_lesion_count", "roi_boundary_score", "z_continuity_score",
        "teacher_dice", "teacher_lesion_count_diff", "qc_pass", "qc_reject_reason",
        "manual_review_required",
    ]
    with (dirs["qc"] / "qc_metrics_template.csv").open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(qc_header)

    (dirs["qc"] / "qc_rules_v1.md").write_text(
        """# G2 Synthetic QC Rules v1

## 强制拒绝

1. 缺少任一模态或 `seg.nii.gz`。
2. 四模态与 `seg` 的 shape、spacing 或 affine 明显不一致。
3. 图像含 NaN/Inf，或 label 非整数。
4. label 值域不在 `{0,1,2,3,4}`。
5. label 全空且 manifest 未显式允许空 mask。
6. synthetic lesion 出现在脑外全零背景区。
7. 插入 ROI 边界出现明显方块断层。
8. 2D/slice-stitching 结果在 z 轴严重断裂。
9. 缺少 `synthetic_generation_manifest.csv` 或 `generation_log.jsonl`，且无法补建。

## 需要人工复查

1. tiny lesion 数量异常高。
2. ET/NETC 与 SNFH 空间关系不合理。
3. RC 在非术后语境下大量出现。
4. teacher model 与 synthetic label 差异极大。

## 只记录不拒绝

1. FID、MS-SSIM 等生成质量指标。
2. teacher model Dice。
3. lesion-wise count difference。

## lesion 分档

1. `tiny_lt_27mm3`
2. `small_27_to_275mm3`
3. `large_gt_275mm3`

最终是否采用 synthetic data，以真实验证 fold 上的分割和检测消融结果为准。
""",
        encoding="utf-8",
    )

    (dirs["reports"] / "ablation_plan_template.md").write_text(
        """# G2 Synthetic Data Ablation Plan Template

## 实验组

| 实验 | 训练数据 | 验证数据 | 目的 |
|---|---|---|---|
| A | Real only | fixed real fold0 | baseline |
| B | Real + 0.25x accepted synthetic | fixed real fold0 | 小比例合成数据 |
| C | Real + 0.5x accepted synthetic | fixed real fold0 | 中等比例合成数据 |
| D | Real + G1 Regular online-style synth | fixed real fold0 | 对齐 G1 Regular 方案 |
| E | Real + G1 Custom online-style synth | fixed real fold0 | 对齐 G1 Custom 方案 |

## 固定变量

1. 同一 nnU-Net 配置。
2. 同一 fold。
3. 同一 preprocessing。
4. 同一训练 epoch/iteration。
5. 同一后处理。
6. 同一 evaluation 脚本。

## 记录指标

Dice、NSD、lesion-wise F1/AUC、tiny/small/large 分档表现、false positive components、NETC/SNFH/ET/RC 分项表现。
""",
        encoding="utf-8",
    )

    (dirs["reports"] / "G2_synthetic_data_quality_report_template.md").write_text(
        """# G2 Synthetic Data Quality Report Template

## 1. 本轮生成概况

## 2. G1 checkpoint 与生成配置

## 3. 候选病例、通过病例与拒绝病例

## 4. 拒绝原因分布

## 5. label 与 lesion 分布

## 6. tiny/small/large lesion 分布

## 7. 多模态图像质量检查

## 8. teacher model 一致性

## 9. nnU-Net 转换结果

## 10. 下游消融结果

## 11. 已知问题

## 12. 下一轮给 G1/G2/S1/S2 的建议
""",
        encoding="utf-8",
    )


def write_path_check(data_root: Path, train_root: Path, val_root: Path, corrected_root: Path, report_path: Path) -> None:
    items = [
        ("Task data root", data_root, "源数据根目录"),
        ("Training root", train_root, "带标签训练集"),
        ("Validation root", val_root, "无公开标签验证集"),
        ("Corrected labels", corrected_root, "官方修正标签"),
        ("Training zip", data_root / "MICCAI-LH-BraTS2025-MET-Challenge-TrainingData_batch1.zip", "训练集压缩包"),
        ("Validation zip", data_root / "MICCAI-LH-BraTS2025-MET-Challenge-ValidationData_batch1.zip", "验证集压缩包"),
        ("Corrected labels zip", data_root / "MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels_batch1.zip", "修正标签压缩包"),
    ]
    lines = ["# G2 Local Data Paths Check", "", f"生成日期：{RUN_DATE}", "", "| 项目 | 路径 | 是否存在 | 大小 | 备注 |", "|---|---|---|---:|---|"]
    for name, path, note in items:
        lines.append(f"| {name} | `{path}` | {'yes' if path.exists() else 'no'} | {human_size(dir_size(path)) if path.exists() else '0 B'} | {note} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_data_qc_summary(dirs: dict[str, Path], raw_df: pd.DataFrame, val_df: pd.DataFrame, overlay_df: pd.DataFrame, final_df: pd.DataFrame) -> None:
    final_pass = int((final_df["final_qc_pass"] == True).sum())  # noqa: E712
    final_fail = int((final_df["final_qc_pass"] != True).sum())  # noqa: E712
    illegal = final_df[final_df["has_illegal_label_after_overlay"] == True][["case_id", "illegal_label_values_after_overlay", "final_qc_reason"]]  # noqa: E712
    affine_warnings = int(final_df["basic_qc_reason"].fillna("").str.contains("affine_hash_mismatch").sum())
    lines = [
        "# G2 Real Data QC Summary",
        "",
        f"生成日期：{RUN_DATE}",
        "",
        "## 总览",
        "",
        f"1. 训练病例 manifest 行数：{len(raw_df)}。",
        f"2. validation 病例 manifest 行数：{len(val_df)}。",
        f"3. corrected labels 文件数：{len(overlay_df)}。",
        f"4. corrected overlay 后 final QC pass：{final_pass}。",
        f"5. corrected overlay 后 final QC fail：{final_fail}。",
        f"6. affine hash warning 病例数：{affine_warnings}。这类病例 shape/spacing 一致，但模态或 label header affine hash 不完全一致，第一轮记录为 warning，不直接排除。",
        "",
        "## corrected labels",
        "",
        df_to_markdown(overlay_df),
        "",
        "## overlay 后非法标签病例",
        "",
    ]
    if illegal.empty:
        lines.append("无。")
    else:
        lines.append(df_to_markdown(illegal))
    lines.extend([
        "",
        "## 说明",
        "",
        "1. 本轮未复制 NIfTI 数据，仅记录原始路径和有效 label 路径。",
        "2. 图像全体素 NaN/Inf 检查因本地训练与验证数据约 36GB，暂不在 Mac 上全量读取；当前已完成 NIfTI header、shape、spacing、affine hash 与 label 值域检查。",
        "3. affine hash 不一致当前作为 warning；正式训练前由 nnU-Net integrity check 和必要的 header/方向一致性复核兜底。",
        "4. `BraTS-MET-01184-002` 使用 corrected label 后不再保留非法值 8。",
        "5. `BraTS-MET-01094-002` 当前仍含非法值 6，第一轮训练与生成 source 中排除。",
    ])
    (dirs["reports"] / "real_data_qc_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_lesion_summary(dirs: dict[str, Path], summary: dict[str, object]) -> None:
    lesion_bucket = summary.get("lesion_bucket_counter", {})
    lesion_case = summary.get("lesions_per_case", {})
    label_combo = summary.get("label_combo_counter", {})
    lines = [
        "# G2 Real Lesion Distribution Summary",
        "",
        f"生成日期：{RUN_DATE}",
        "",
        "## 关键结果",
        "",
        f"1. final QC pass 病例数：{summary.get('final_qc_pass_cases', 0)}。",
        f"2. 含任意非背景 label 病例数：{summary.get('cases_with_any_label', 0)}。",
        f"3. 含 tiny lesion 的去重病例数：{summary.get('tiny_case_count_unique', 0)}。",
        "",
        "## lesion 体积分档",
        "",
        "| 分档 | lesion 数 |",
        "|---|---:|",
    ]
    for key in ["tiny_lt_27mm3", "small_27_to_275mm3", "large_gt_275mm3"]:
        lines.append(f"| {key} | {dict(lesion_bucket).get(key, 0)} |")
    lines.extend(["", "## 每例 lesion 数分布", "", "| lesion 数 | 病例数 |", "|---:|---:|"])
    for key, value in sorted(dict(lesion_case).items(), key=lambda kv: int(kv[0])):
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## label 组合 Top 20", "", "| label combination | 病例数 |", "|---|---:|"])
    for key, value in Counter(label_combo).most_common(20):
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## 多病灶病例 Top 20", "", "| case_id | lesion_count |", "|---|---:|"])
    for case_id, count in summary.get("many_lesion_cases_top20", []):
        lines.append(f"| {case_id} | {count} |")
    lines.extend([
        "",
        "## 给 synthetic target distribution 的直接含义",
        "",
        "1. 小病灶需要单独提高召回，但不能把所有 synthetic 都做成 tiny，否则会推高假阳性。",
        "2. 多发病例应被纳入第一轮 smoke 和 100-300 候选生成目标。",
        "3. RC 只建议从真实 RC 病例做保守变体，不建议第一轮无条件随机生成。",
        "4. G1 当前提出的 Regular/Custom 在线改造方案需要由 G2 manifest 记录每次标签修改、缩放比例和插入次数。",
    ])
    (dirs["stats"] / "real_lesion_distribution_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_target_distribution(dirs: dict[str, Path], label_df: pd.DataFrame, lesion_df: pd.DataFrame) -> None:
    total_cases = len(label_df)
    rc_cases = int(label_df["has_RC"].sum()) if not label_df.empty else 0
    tiny = int((lesion_df["volume_bucket"] == "tiny_lt_27mm3").sum()) if not lesion_df.empty else 0
    small = int((lesion_df["volume_bucket"] == "small_27_to_275mm3").sum()) if not lesion_df.empty else 0
    large = int((lesion_df["volume_bucket"] == "large_gt_275mm3").sum()) if not lesion_df.empty else 0
    lines = [
        "# G2 Target Synthetic Distribution v1",
        "",
        f"生成日期：{RUN_DATE}",
        "",
        "## 真实分布参考",
        "",
        f"1. 可用真实病例数：{total_cases}。",
        f"2. 含 RC 病例数：{rc_cases}。",
        f"3. tiny/small/large lesion 数：{tiny}/{small}/{large}。",
        "",
        "## 第一轮生成目标",
        "",
        "1. G1 先交付 10-20 个 smoke cases，G2 完成 QC 和 nnU-Net 转换验证。",
        "2. smoke 通过后，再生成 100-300 个候选 synthetic cases。",
        "3. 第一轮 accepted synthetic cases 不超过真实训练病例数的 25%。",
        "4. 每个 source case 默认最多生成 1 个 synthetic case；多发病例专项实验可单独申请例外。",
        "5. source case 只来自 final_qc_pass=true 的训练病例，绝不来自 validation。",
        "6. 优先补 small/tiny lesion 和多发病例，但 tiny lesion 比例不应超过 accepted synthetic 的 35%。",
        "7. RC 只基于真实 RC case 做保守变体，第一轮不做凭空生成 RC。",
        "8. 第一轮不做整例 MRI 从零生成，不做无 manifest/log 的 raw output。",
        "",
        "## 对 G1 当前方案的约束",
        "",
        "1. 60% 概率修改标签、70% 概率将 SNFH/ET 转换等操作必须逐例写入 manifest。",
        "2. 缩放比例、插入肿瘤数量、label_kind、seed 都必须可复现。",
        "3. Regular 与 Custom 应作为两种 generation policy，不能混在一个未标记的数据池里。",
        "4. 在线训练方案需要 S1/S2 训练框架配合；本机 Mac 只准备规则、manifest 和 QC，不运行在线生成训练。",
    ]
    (dirs["stats"] / "target_synthetic_distribution_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_execution_summary(dirs: dict[str, Path], outputs: list[Path]) -> None:
    lines = [
        "# G2 Pretraining Checklist Execution Summary",
        "",
        f"生成日期：{RUN_DATE}",
        "",
        "## 已在本机完成",
        "",
        "1. 外部训练集、验证集、corrected labels 路径检查。",
        "2. 训练集 raw manifest。",
        "3. validation manifest，并标记不可作为 synthetic source。",
        "4. corrected label overlay。",
        "5. overlay 后 final train manifest。",
        "6. label 值域统计与非法标签排查。",
        "7. lesion connected component 统计与 tiny/small/large 分档。",
        "8. G1 GliGAN-compatible source CSV。",
        "9. nnU-Net real-only 映射表与 dataset.json 草案。",
        "10. 固定 fold0 split。",
        "11. synthetic QC 模板、消融模板、报告模板。",
        "",
        "## 暂缓项",
        "",
        "1. 不在本机复制或软链接 31GB 训练 NIfTI 到 nnU-Net raw 目录。",
        "2. 不在本机运行 `nnUNetv2_plan_and_preprocess`。",
        "3. 不在本机训练 GliGAN/diffusion 或执行在线 batch 生成。",
        "4. 不在本机生成大量 synthetic NIfTI。",
        "",
        "## 产物列表",
        "",
    ]
    for path in outputs:
        lines.append(f"- `{path}`")
    (dirs["reports"] / "g2_pretraining_execution_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/Users/hwaigc/比赛+课题/ECNU-NYU2026/2026的task1以及数据")
    parser.add_argument("--results-root", default="/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results")
    parser.add_argument("--force", action="store_true", help="Re-scan NIfTI data even if cached CSV files exist.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_root = data_root / "MICCAI-LH-BraTS2025-MET-Challenge-Training"
    validation_root = data_root / "Validation"
    corrected_root = data_root / "MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels"
    results_root = Path(args.results_root)
    dirs = ensure_dirs(results_root)
    write_readme_files(results_root, dirs)
    outputs: list[Path] = []

    path_report = dirs["reports"] / "local_data_paths_check.md"
    write_path_check(data_root, train_root, validation_root, corrected_root, path_report)
    outputs.append(path_report)

    raw_path = dirs["manifests"] / "real_train_manifest_raw.csv"
    if raw_path.exists() and not args.force:
        raw_df = pd.read_csv(raw_path)
    else:
        raw_df = scan_training(train_root)
        raw_df.to_csv(raw_path, index=False)
    outputs.append(raw_path)

    val_path = dirs["manifests"] / "real_validation_manifest.csv"
    if val_path.exists() and not args.force:
        val_df = pd.read_csv(val_path)
    else:
        val_df = scan_validation(validation_root)
        val_df.to_csv(val_path, index=False)
    outputs.append(val_path)

    overlay_path = dirs["manifests"] / "corrected_label_overlay.csv"
    final_path = dirs["manifests"] / "real_train_manifest.csv"
    if overlay_path.exists() and final_path.exists() and not args.force:
        overlay_df = pd.read_csv(overlay_path)
        final_df = pd.read_csv(final_path)
    else:
        overlay_df, final_df = apply_corrected_labels(raw_df, corrected_root)
        overlay_df.to_csv(overlay_path, index=False)
        final_df.to_csv(final_path, index=False)
    outputs.extend([overlay_path, final_path])

    write_data_qc_summary(dirs, raw_df, val_df, overlay_df, final_df)
    outputs.append(dirs["reports"] / "real_data_qc_summary.md")

    label_df, lesion_df, lesion_summary = label_stats(final_df)
    label_path = dirs["stats"] / "real_label_distribution.csv"
    lesion_path = dirs["stats"] / "real_lesion_distribution.csv"
    label_df.to_csv(label_path, index=False)
    lesion_df.to_csv(lesion_path, index=False)
    outputs.extend([label_path, lesion_path])
    write_lesion_summary(dirs, lesion_summary)
    outputs.append(dirs["stats"] / "real_lesion_distribution_summary.md")
    write_json(dirs["stats"] / "real_lesion_distribution_summary.json", lesion_summary)
    outputs.append(dirs["stats"] / "real_lesion_distribution_summary.json")
    write_target_distribution(dirs, label_df, lesion_df)
    outputs.append(dirs["stats"] / "target_synthetic_distribution_v1.md")

    gligan_df = gligan_source_csv(final_df)
    gligan_path = dirs["manifests"] / "g1_gligan_source_cases_v1.csv"
    gligan_df.to_csv(gligan_path, index=False)
    outputs.append(gligan_path)

    mapping_df, _ = nnunet_mapping(final_df)
    mapping_path = dirs["manifests"] / "nnunet_case_mapping_realonly.csv"
    mapping_df.to_csv(mapping_path, index=False)
    outputs.append(mapping_path)
    write_dataset_json(dirs["nnunet_raw"] / "dataset.json")
    outputs.append(dirs["nnunet_raw"] / "dataset.json")
    split = stable_split(mapping_df)
    split_path = dirs["splits"] / "splits_final_fold0_realval.json"
    write_json(split_path, split)
    outputs.append(split_path)

    write_templates(dirs)
    outputs.extend([
        dirs["qc"] / "qc_metrics_template.csv",
        dirs["qc"] / "qc_rules_v1.md",
        dirs["reports"] / "ablation_plan_template.md",
        dirs["reports"] / "G2_synthetic_data_quality_report_template.md",
    ])
    write_execution_summary(dirs, outputs)
    outputs.append(dirs["reports"] / "g2_pretraining_execution_summary.md")

    print(json.dumps({
        "train_cases": len(raw_df),
        "validation_cases": len(val_df),
        "final_qc_pass": int((final_df["final_qc_pass"] == True).sum()),  # noqa: E712
        "final_qc_fail": int((final_df["final_qc_pass"] != True).sum()),  # noqa: E712
        "lesions": len(lesion_df),
        "gligan_usable_cases": int((gligan_df["usable_for_gligan96"] == True).sum()) if not gligan_df.empty else 0,  # noqa: E712
        "outputs": [str(p) for p in outputs],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
