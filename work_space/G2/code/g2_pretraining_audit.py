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
import re
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
REAL_TRAIN_EMPTY_COLUMNS = [
    "case_id", "split_source", "case_dir",
    "t1n_path", "t1c_path", "t2w_path", "t2f_path", "raw_seg_path",
    "has_t1n", "has_t1c", "has_t2w", "has_t2f", "has_seg",
    "shape_t1n", "shape_t1c", "shape_t2w", "shape_t2f", "shape_seg",
    "spacing_t1n", "spacing_t1c", "spacing_t2w", "spacing_t2f", "spacing_seg",
    "affine_hash_t1n", "affine_hash_t1c", "affine_hash_t2w", "affine_hash_t2f", "affine_hash_seg",
    "image_dtypes", "label_dtype", "labels_present", "has_nan_or_inf",
    "image_nan_inf_check", "basic_qc_pass", "basic_qc_reason",
]


def as_posix(path: Path | str | None) -> str:
    return "" if path is None else str(path)


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df


def ensure_dirs(results_root: Path) -> dict[str, Path]:
    dirs = {
        "nnunet_raw_root": results_root / "nnunet_raw",
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
        results_root / "README.md": "# G2 Results\n\n本目录保存 G2 的轻量结果文件：真实数据清单、G1 source 表、synthetic intake 模板、QC 策略、官方指标模板和进度索引。NIfTI 大数据、nnU-Net 预处理缓存、正式 synthetic 影像和临时 smoke run 产物都不放进仓库。\n\n正式入口：`../code/g2_synthetic_raw_intake_qc.py` 接收 G1 raw run，`../code/g2_materialize_nnunet_dataset.py` 转 nnU-Net raw，`../code/g2_official_mets_metrics_parser.py` 解析或校验 2026 Task1 官方字段。\n",
        dirs["nnunet_raw_root"] / "README.md": "# nnunet_raw\n\n这里是 nnU-Net 原始数据的轻量入口。当前仓库只放占位说明、dataset.json 和路径契约，不放正式大体积影像。`Dataset260_BraTS2026_MET_RealOnly/` 记录 real-only 基线；正式 real+synth 由 `../code/g2_materialize_nnunet_dataset.py` 在训练机物化。\n",
        dirs["manifests"] / "README.md": "# Manifests\n\n保存真实训练/验证清单、corrected overlay、G1 兼容 source CSV、synthetic intake 模板，以及正式 G1 批次到来后生成的 accepted/rejected 索引文件。旧 smoke run 演示输出不保留。\n",
        dirs["stats"] / "README.md": "# Stats\n\n保存真实 label/lesion 分布、synthetic 目标分布、batch 级统计摘要，以及后续抽样分析所需的小型数表。\n",
        dirs["qc"] / "README.md": "# QC\n\n保存 synthetic data 质量控制规则、逐例指标模板、扩散质量专项模板、人工复查表头、官方 leaderboard 对齐模板，以及正式 run 级自动 QC 输出。\n",
        dirs["splits"] / "README.md": "# Splits\n\n保存固定真实验证 fold，供 real-only、real+synth 和后续所有消融复用。\n",
        dirs["reports"] / "README.md": "# Reports\n\n保存路径检查、数据 QC、执行总结、进度报告、消融模板和团队沟通文档源稿。临时 smoke run 质量报告不保留。\n",
        dirs["nnunet_raw"] / "README.md": "# Dataset260_BraTS2026_MET_RealOnly\n\n本目录当前只保存 `dataset.json` 和映射说明，不复制或软链接全量 NIfTI。需要正式训练时，由 S1/S2 根据 `manifests/nnunet_case_mapping_realonly.csv` 在训练机器上物化数据集并运行 nnU-Net 预处理。synthetic accepted 结果另起 dataset id，不混进这个 real-only 占位目录。\n",
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


SYNTHETIC_SUFFIX_VARIANTS = {
    "t1n": ["t1n", "scan_t1"],
    "t1c": ["t1c", "scan_t1ce"],
    "t2w": ["t2w", "scan_t2"],
    "t2f": ["t2f", "scan_flair"],
    "seg": ["seg"],
}
NNUNET_CHANNELS = {
    "t1n": "0000",
    "t1c": "0001",
    "t2w": "0002",
    "t2f": "0003",
}


def read_json_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def read_jsonl_if_exists(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
        except Exception:  # noqa: BLE001
            continue
    return rows


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def recursive_find_value(data: object, key: str) -> object | None:
    if isinstance(data, dict):
        if key in data:
            return data[key]
        for value in data.values():
            found = recursive_find_value(value, key)
            if found is not None:
                return found
    elif isinstance(data, list):
        for value in data:
            found = recursive_find_value(value, key)
            if found is not None:
                return found
    return None


def parse_synthetic_case_name(name: str) -> dict[str, object]:
    match = re.match(r"^(?P<source_case_id>.+?)_(?P<label_kind>[^_]+)_label_(?P<label_index>\d+)$", name)
    if not match:
        return {"parsed": False, "source_case_id": "", "label_kind": "", "label_index": ""}
    return {
        "parsed": True,
        "source_case_id": match.group("source_case_id"),
        "label_kind": match.group("label_kind"),
        "label_index": int(match.group("label_index")),
    }


def find_synthetic_case_dirs(run_root: Path) -> list[Path]:
    case_dirs: set[Path] = set()
    suffix_fragments = [
        "-t1n.nii.gz",
        "-t1c.nii.gz",
        "-t2w.nii.gz",
        "-t2f.nii.gz",
        "-scan_t1.nii.gz",
        "-scan_t1ce.nii.gz",
        "-scan_t2.nii.gz",
        "-scan_flair.nii.gz",
        "-seg.nii.gz",
    ]
    for path in run_root.rglob("*.nii.gz"):
        lower_name = path.name.lower()
        if any(lower_name.endswith(fragment) for fragment in suffix_fragments):
            case_dirs.add(path.parent)
    return sorted(case_dirs, key=lambda p: p.as_posix())


def synthetic_modality_files(case_dir: Path) -> dict[str, Path | None]:
    files: dict[str, Path | None] = {}
    for modality, variants in SYNTHETIC_SUFFIX_VARIANTS.items():
        matches: list[Path] = []
        for variant in variants:
            matches.extend(sorted(case_dir.glob(f"*{variant}.nii.gz")))
        files[modality] = matches[0] if matches else None
    return files


def normalized_synthetic_paths(normalized_case_dir: str, synthetic_final_id: str) -> dict[str, str]:
    root = Path(normalized_case_dir)
    paths = {
        "t1n": root / f"{synthetic_final_id}-t1n.nii.gz",
        "t1c": root / f"{synthetic_final_id}-t1c.nii.gz",
        "t2w": root / f"{synthetic_final_id}-t2w.nii.gz",
        "t2f": root / f"{synthetic_final_id}-t2f.nii.gz",
        "seg": root / f"{synthetic_final_id}-seg.nii.gz",
    }
    return {key: as_posix(value) for key, value in paths.items()}


def nnunet_synthetic_paths(nnunet_case_id: str, dataset_name: str = "Dataset261_BraTS2026_MET_RealSynth") -> dict[str, str]:
    root = Path("nnunet_raw") / dataset_name
    paths = {
        "t1n": root / "imagesTr" / f"{nnunet_case_id}_0000.nii.gz",
        "t1c": root / "imagesTr" / f"{nnunet_case_id}_0001.nii.gz",
        "t2w": root / "imagesTr" / f"{nnunet_case_id}_0002.nii.gz",
        "t2f": root / "imagesTr" / f"{nnunet_case_id}_0003.nii.gz",
        "seg": root / "labelsTr" / f"{nnunet_case_id}.nii.gz",
    }
    return {key: value.as_posix() for key, value in paths.items()}


def synthetic_mapping_rows(row: dict[str, object]) -> list[dict[str, object]]:
    mapping_rows = []
    for modality in ["t1n", "t1c", "t2w", "t2f", "seg"]:
        mapping_rows.append({
            "synthetic_raw_id": row.get("synthetic_raw_id", ""),
            "synthetic_final_id": row.get("synthetic_final_id", ""),
            "nnunet_case_id": row.get("nnunet_case_id", ""),
            "source_case_id": row.get("source_case_id", ""),
            "generation_run_id": row.get("generation_run_id", ""),
            "modality": modality,
            "nnunet_channel": NNUNET_CHANNELS.get(modality, "label"),
            "raw_source_path": row.get(f"raw_{modality}_path", ""),
            "normalized_target_path": row.get(f"normalized_{modality}_path", ""),
            "nnunet_target_path": row.get(f"nnunet_{modality}_target_path", ""),
            "output_suffix_scheme": row.get("output_suffix_scheme", ""),
            "suffix_conversion_action": row.get("suffix_conversion_action", ""),
            "qc_decision": row.get("qc_decision", ""),
            "accepted_for_training": row.get("accepted_for_training", False),
            "accepted_for_ablation_only": row.get("accepted_for_ablation_only", False),
            "needs_regeneration": row.get("needs_regeneration", False),
        })
    return mapping_rows


def detect_output_suffix_scheme(files: dict[str, Path | None]) -> str:
    suffixes = {path.name.lower() for path in files.values() if path is not None}
    has_legacy = any("scan_t1ce" in name or "scan_t1" in name or "scan_flair" in name or "scan_t2" in name for name in suffixes)
    has_native = any(name.endswith(("-t1n.nii.gz", "-t1c.nii.gz", "-t2w.nii.gz", "-t2f.nii.gz")) for name in suffixes)
    if has_legacy and has_native:
        return "mixed"
    if has_legacy:
        return "legacy_gligan"
    if has_native:
        return "native_2026"
    return "unknown"


def orientation_codes_from_affine(affine: np.ndarray) -> tuple[str, str, str]:
    try:
        return tuple(nib.aff2axcodes(affine))  # type: ignore[return-value]
    except Exception:  # noqa: BLE001
        return ("", "", "")


def load_reference_context(results_root: Path) -> dict[str, object]:
    manifests_dir = results_root / "manifests"
    splits_dir = results_root / "splits"
    train_df = read_csv_if_exists(manifests_dir / "real_train_manifest.csv")
    val_df = read_csv_if_exists(manifests_dir / "real_validation_manifest.csv")
    g1_df = read_csv_if_exists(manifests_dir / "g1_gligan_source_cases_v1.csv")
    mapping_df = read_csv_if_exists(manifests_dir / "nnunet_case_mapping_realonly.csv")
    split_path = splits_dir / "splits_final_fold0_realval.json"
    split_data = []
    if split_path.exists():
        try:
            split_data = json.loads(split_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            split_data = []
    split_val_ids: set[str] = set()
    if split_data and isinstance(split_data, list) and isinstance(split_data[0], dict):
        split_val_ids = set(split_data[0].get("val", []))
    source_to_nn = {}
    if not mapping_df.empty and "source_case_id" in mapping_df.columns and "nnunet_case_id" in mapping_df.columns:
        source_to_nn = dict(zip(mapping_df["source_case_id"].astype(str), mapping_df["nnunet_case_id"].astype(str)))
    train_lookup = train_df.set_index("case_id").to_dict(orient="index") if not train_df.empty and "case_id" in train_df.columns else {}
    val_lookup = val_df.set_index("case_id").to_dict(orient="index") if not val_df.empty and "case_id" in val_df.columns else {}
    g1_lookup = g1_df.set_index("case_id").to_dict(orient="index") if not g1_df.empty and "case_id" in g1_df.columns else {}
    return {
        "train_df": train_df,
        "val_df": val_df,
        "g1_df": g1_df,
        "mapping_df": mapping_df,
        "split_data": split_data,
        "split_val_ids": split_val_ids,
        "source_to_nn": source_to_nn,
        "train_lookup": train_lookup,
        "val_lookup": val_lookup,
        "g1_lookup": g1_lookup,
    }


def build_source_status(source_case_id: str, ctx: dict[str, object]) -> dict[str, object]:
    train_lookup = ctx["train_lookup"]  # type: ignore[assignment]
    val_lookup = ctx["val_lookup"]  # type: ignore[assignment]
    g1_lookup = ctx["g1_lookup"]  # type: ignore[assignment]
    split_val_ids = ctx["split_val_ids"]  # type: ignore[assignment]
    source_to_nn = ctx["source_to_nn"]  # type: ignore[assignment]

    train_row = train_lookup.get(source_case_id, {})
    val_row = val_lookup.get(source_case_id, {})
    g1_row = g1_lookup.get(source_case_id, {})
    nn_id = source_to_nn.get(source_case_id, "")
    in_fixed_val_fold = bool(nn_id and nn_id in split_val_ids)
    final_qc_pass = bool(train_row.get("final_qc_pass", False))
    usable_for_gligan96 = bool(g1_row.get("usable_for_gligan96", False))
    allowed_as_synthetic_source = bool(g1_row.get("allowed_as_synthetic_source", usable_for_gligan96))
    source_is_allowed = bool(train_row) and final_qc_pass and not bool(val_row) and not in_fixed_val_fold and allowed_as_synthetic_source
    return {
        "source_row": train_row,
        "val_row": val_row,
        "g1_row": g1_row,
        "nnunet_case_id": nn_id,
        "source_in_real_train_manifest": bool(train_row),
        "source_final_qc_pass": final_qc_pass,
        "source_usable_for_gligan96": usable_for_gligan96,
        "source_in_fixed_val_fold": in_fixed_val_fold,
        "source_from_official_validation": bool(val_row),
        "source_is_allowed": source_is_allowed,
        "source_split": "train" if train_row else ("validation" if val_row else "unknown"),
    }


def array_stats(arr: np.ndarray) -> dict[str, object]:
    arr = np.asarray(arr)
    finite = bool(np.isfinite(arr).all())
    arr_float = arr.astype(np.float32, copy=False)
    flat = arr_float.reshape(-1)
    stats = {
        "min": float(np.min(arr_float)) if flat.size else math.nan,
        "p1": float(np.percentile(flat, 1)) if flat.size else math.nan,
        "p50": float(np.percentile(flat, 50)) if flat.size else math.nan,
        "p99": float(np.percentile(flat, 99)) if flat.size else math.nan,
        "max": float(np.max(arr_float)) if flat.size else math.nan,
        "mean": float(np.mean(arr_float)) if flat.size else math.nan,
        "std": float(np.std(arr_float)) if flat.size else math.nan,
        "nonzero_ratio": float(np.count_nonzero(arr_float) / float(flat.size)) if flat.size else math.nan,
        "finite": finite,
        "is_constant": bool(flat.size and np.isclose(float(np.max(arr_float)), float(np.min(arr_float)))),
    }
    return stats


def shell_mask(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    expanded = ndimage.binary_dilation(mask, iterations=1)
    inner = ndimage.binary_erosion(mask, iterations=1)
    return np.logical_and(expanded, np.logical_not(inner))


def ratio_or_blank(num: float | None, den: float | None) -> float | str:
    if num is None or den is None:
        return ""
    if not np.isfinite(num) or not np.isfinite(den) or den == 0:
        return ""
    return float(num / den)


def summarize_case_quality(
    case_dir: Path,
    idx: int,
    run_ctx: dict[str, object],
    source_status: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    case_name = case_dir.name
    parsed = parse_synthetic_case_name(case_name)
    if not parsed.get("parsed"):
        for entry in run_ctx.get("generation_log_rows", []):
            if not isinstance(entry, dict):
                continue
            entry_name = str(entry.get("case_id") or entry.get("synthetic_raw_id") or entry.get("raw_case_id") or entry.get("case_name") or "")
            if entry_name and (entry_name == case_name or Path(entry_name).name == case_name):
                parsed = {
                    "parsed": True,
                    "source_case_id": str(entry.get("source_case_id") or ""),
                    "label_kind": str(entry.get("label_kind") or ""),
                    "label_index": int(entry.get("label_index") or 0),
                }
                break
    files = synthetic_modality_files(case_dir)
    output_scheme = detect_output_suffix_scheme(files)
    synthetic_raw_id = case_name
    synthetic_final_id = f"SYN-MET-{idx:06d}"
    nnunet_case_id = f"SYNMET{idx:06d}"
    generation_run_id = str(run_ctx.get("generation_run_id", ""))
    generator_name = str(run_ctx.get("generator_name", ""))
    generator_checkpoint = str(run_ctx.get("generator_checkpoint", ""))
    generator_checkpoint_t1n = str(run_ctx.get("generator_checkpoint_t1n", generator_checkpoint))
    generator_checkpoint_t1c = str(run_ctx.get("generator_checkpoint_t1c", generator_checkpoint))
    generator_checkpoint_t2w = str(run_ctx.get("generator_checkpoint_t2w", generator_checkpoint))
    generator_checkpoint_t2f = str(run_ctx.get("generator_checkpoint_t2f", generator_checkpoint))
    source_case_id = str(parsed.get("source_case_id") or run_ctx.get("source_case_id") or "")
    label_kind = str(parsed.get("label_kind") or "")
    label_index = int(parsed.get("label_index") or 0)
    source_info = build_source_status(source_case_id, run_ctx) if source_case_id else {
        "source_row": {},
        "val_row": {},
        "g1_row": {},
        "nnunet_case_id": "",
        "source_in_real_train_manifest": False,
        "source_final_qc_pass": False,
        "source_usable_for_gligan96": False,
        "source_in_fixed_val_fold": False,
        "source_from_official_validation": False,
        "source_is_allowed": False,
        "source_split": "unknown",
    }
    if source_case_id and source_status:
        source_info.update(source_status)
    source_row = source_info.get("source_row", {})
    val_row = source_info.get("val_row", {})
    g1_row = source_info.get("g1_row", {})

    config_exists = bool(run_ctx.get("generation_config_exists", False))
    manifest_exists = bool(run_ctx.get("generation_manifest_exists", False))
    log_exists = bool(run_ctx.get("generation_log_exists", False))
    raw_case_dir = as_posix(case_dir)
    normalized_case_path = Path(run_ctx.get("normalized_root", case_dir.parent)) / synthetic_final_id
    normalized_case_dir = as_posix(normalized_case_path)
    normalized_paths = normalized_synthetic_paths(normalized_case_dir, synthetic_final_id)
    nnunet_paths = nnunet_synthetic_paths(nnunet_case_id)
    if output_scheme == "legacy_gligan":
        suffix_conversion_action = "map_legacy_suffix_to_native_2026"
    elif output_scheme == "native_2026":
        suffix_conversion_action = "keep_native_2026_suffix"
    elif output_scheme == "mixed":
        suffix_conversion_action = "reject_mixed_suffix_scheme"
    else:
        suffix_conversion_action = "inspect_unknown_suffix_scheme"

    modalities = ["t1n", "t1c", "t2w", "t2f", "seg"]
    rows: dict[str, object] = {
        "synthetic_raw_id": synthetic_raw_id,
        "synthetic_final_id": synthetic_final_id,
        "nnunet_case_id": nnunet_case_id,
        "source_case_id": source_case_id,
        "source_split": source_info.get("source_split", ""),
        "label_kind": label_kind,
        "label_index": label_index,
        "label_source_case_id": source_case_id,
        "label_component_id": "whole_positive_mask",
        "label_generator_checkpoint": generator_checkpoint,
        "generation_run_id": generation_run_id,
        "generator_name": generator_name,
        "generator_checkpoint_t1n": generator_checkpoint_t1n,
        "generator_checkpoint_t1c": generator_checkpoint_t1c,
        "generator_checkpoint_t2w": generator_checkpoint_t2w,
        "generator_checkpoint_t2f": generator_checkpoint_t2f,
        "generator_io": str(run_ctx.get("generator_io", "")),
        "label_channels": int(run_ctx.get("label_channels", 0) or 0),
        "rc_policy": str(run_ctx.get("rc_policy", "")),
        "noise_type": str(run_ctx.get("noise_type", "")),
        "sampling_method": str(run_ctx.get("sampling_method", "")),
        "sampling_steps": run_ctx.get("sampling_steps", ""),
        "eta": run_ctx.get("eta", ""),
        "seed": run_ctx.get("seed", ""),
        "source_csv_path": str(run_ctx.get("source_csv_path", "")),
        "source_csv_version": str(run_ctx.get("source_csv_version", "")),
        "raw_case_dir": raw_case_dir,
        "normalized_case_dir": normalized_case_dir,
        "output_suffix_scheme": output_scheme,
        "suffix_conversion_action": suffix_conversion_action,
        "config_exists": config_exists,
        "manifest_exists": manifest_exists,
        "log_exists": log_exists,
        "generation_mode": str(run_ctx.get("generator_io", "")),
        "source_in_real_train_manifest": source_info["source_in_real_train_manifest"],
        "source_final_qc_pass": source_info["source_final_qc_pass"],
        "source_usable_for_gligan96": source_info["source_usable_for_gligan96"],
        "source_in_fixed_val_fold": source_info["source_in_fixed_val_fold"],
        "source_from_official_validation": source_info["source_from_official_validation"],
        "source_is_allowed": source_info["source_is_allowed"],
        "case_id_reuses_real_id": bool(source_case_id and synthetic_final_id == source_case_id),
        "validation_leakage": bool(source_info["source_in_fixed_val_fold"] or source_info["source_from_official_validation"]),
    }

    present = {mod: path for mod, path in files.items() if path is not None}
    for mod in modalities:
        rows[f"has_{mod}"] = files.get(mod) is not None
        rows[f"raw_{mod}_path"] = as_posix(files.get(mod))
        rows[f"normalized_{mod}_path"] = normalized_paths.get(mod, "")
        rows[f"nnunet_{mod}_target_path"] = nnunet_paths.get(mod, "")

    rows["filename_consistent"] = all(
        path is not None and path.parent == case_dir and path.name.startswith(case_name)
        for path in files.values()
    )
    rows["nifti_readable"] = all(path is not None for path in files.values())
    rows["has_nan_or_inf"] = False
    rows["image_is_constant"] = False

    metas: dict[str, dict[str, object]] = {}
    arrays: dict[str, np.ndarray] = {}
    errors: list[str] = []
    for mod in modalities:
        path = files.get(mod)
        if path is None:
            continue
        try:
            meta = nifti_meta(path)
            metas[mod] = meta
            rows[f"shape_{mod}"] = "x".join(map(str, meta["shape"]))
            rows[f"spacing_{mod}"] = ",".join(f"{v:.6g}" for v in meta["spacing"])
            rows[f"affine_hash_{mod}"] = meta["affine_hash"]
            arrays[mod] = np.asanyarray(nib.load(str(path)).dataobj)
            mod_stats = array_stats(arrays[mod])
            rows["has_nan_or_inf"] = bool(rows["has_nan_or_inf"] or not mod_stats["finite"])
            rows["image_is_constant"] = bool(rows["image_is_constant"] or mod_stats["is_constant"])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"read_error_{mod}:{type(exc).__name__}")
            rows[f"shape_{mod}"] = ""
            rows[f"spacing_{mod}"] = ""
            rows[f"affine_hash_{mod}"] = ""
            rows["nifti_readable"] = False

    if metas:
        rows["shape_consistent"] = len({meta["shape"] for meta in metas.values()}) == 1
        rows["spacing_consistent"] = len({tuple(round(float(v), 6) for v in meta["spacing"]) for meta in metas.values()}) == 1
        rows["affine_consistent"] = len({meta["affine_hash"] for meta in metas.values()}) == 1
        rows["orientation_consistent"] = len({orientation_codes_from_affine(np.asarray(meta["affine"])) for meta in metas.values()}) == 1
        rows["affine_valid"] = bool(rows["affine_consistent"])
        rows["has_all_modalities"] = all(bool(files.get(mod)) for mod in ["t1n", "t1c", "t2w", "t2f"])
        rows["has_seg"] = bool(files.get("seg"))
        rows["output_shape_x"], rows["output_shape_y"], rows["output_shape_z"] = metas["seg"]["shape"] if "seg" in metas else metas[sorted(metas.keys())[0]]["shape"]  # type: ignore[index]
        source_shape = source_row.get("shape_seg") or source_row.get("shape_t1n") or source_row.get("shape")
        rows["source_shape_match"] = bool(source_shape and any(str(source_shape) == str(rows.get(f"shape_{mod}", "")) for mod in modalities if rows.get(f"shape_{mod}", "")))
    else:
        rows["shape_consistent"] = False
        rows["spacing_consistent"] = False
        rows["affine_consistent"] = False
        rows["orientation_consistent"] = False
        rows["source_shape_match"] = False
        rows["output_shape_x"] = ""
        rows["output_shape_y"] = ""
        rows["output_shape_z"] = ""

    rows["shape_t1n"] = rows.get("shape_t1n", "")
    rows["shape_t1c"] = rows.get("shape_t1c", "")
    rows["shape_t2w"] = rows.get("shape_t2w", "")
    rows["shape_t2f"] = rows.get("shape_t2f", "")
    rows["shape_seg"] = rows.get("shape_seg", "")
    rows["spacing_t1n"] = rows.get("spacing_t1n", "")
    rows["spacing_t1c"] = rows.get("spacing_t1c", "")
    rows["spacing_t2w"] = rows.get("spacing_t2w", "")
    rows["spacing_t2f"] = rows.get("spacing_t2f", "")
    rows["spacing_seg"] = rows.get("spacing_seg", "")
    rows["affine_hash_t1n"] = rows.get("affine_hash_t1n", "")
    rows["affine_hash_t1c"] = rows.get("affine_hash_t1c", "")
    rows["affine_hash_t2w"] = rows.get("affine_hash_t2w", "")
    rows["affine_hash_t2f"] = rows.get("affine_hash_t2f", "")
    rows["affine_hash_seg"] = rows.get("affine_hash_seg", "")

    # Basic label parsing.
    label_arr = arrays.get("seg")
    if label_arr is not None:
        label_is_integer = bool(np.all(np.isclose(label_arr, np.rint(label_arr))))
        rows["label_is_integer"] = label_is_integer
        label_values = []
        for value in np.unique(label_arr).tolist():
            if isinstance(value, float) and float(value).is_integer():
                label_values.append(int(value))
            else:
                label_values.append(value)
        rows["label_values"] = ";".join(map(str, label_values))
        valid_label_values = all(v in LABELS for v in label_values)
        rows["label_values_valid"] = valid_label_values
        rows["empty_mask"] = not bool(np.any(label_arr > 0))
        rows["allow_empty_mask"] = False
        rows["has_rc"] = bool(np.any(label_arr == 4))
        rows["label_combination"] = "+".join(LABELS[label] for label in [1, 2, 3, 4] if np.any(label_arr == label)) or "none"
    else:
        rows["label_is_integer"] = False
        rows["label_values"] = ""
        rows["label_values_valid"] = False
        rows["empty_mask"] = True
        rows["allow_empty_mask"] = False
        rows["has_rc"] = False
        rows["label_combination"] = "none"

    illegal_label_values = []
    if label_arr is not None:
        unique_values = np.unique(label_arr)
        for item in unique_values.tolist():
            if isinstance(item, float) and item.is_integer():
                item = int(item)
            if item not in LABELS:
                illegal_label_values.append(item)
    rows["label_values_valid"] = bool(rows["label_values_valid"] and not illegal_label_values)

    # Source linkage.
    source_seg_path = source_row.get("effective_seg_path") or source_row.get("raw_seg_path") or ""
    source_t1n_path = source_row.get("t1n_path") or source_row.get("scan_t1") or ""
    source_t1c_path = source_row.get("t1c_path") or source_row.get("scan_t1ce") or ""
    source_t2w_path = source_row.get("t2w_path") or source_row.get("scan_t2") or ""
    source_t2f_path = source_row.get("t2f_path") or source_row.get("scan_flair") or ""
    source_seg_arr = None
    source_arrays: dict[str, np.ndarray] = {}
    source_shapes_match = False
    if source_seg_path:
        try:
            source_seg_arr = np.asanyarray(nib.load(str(source_seg_path)).dataobj)
            if label_arr is not None:
                source_shapes_match = source_seg_arr.shape == label_arr.shape
        except Exception:  # noqa: BLE001
            source_seg_arr = None
    for mod_name, path_str in [("t1n", source_t1n_path), ("t1c", source_t1c_path), ("t2w", source_t2w_path), ("t2f", source_t2f_path)]:
        if path_str:
            try:
                source_arrays[mod_name] = np.asanyarray(nib.load(str(path_str)).dataobj)
            except Exception:  # noqa: BLE001
                continue
    rows["source_shape_match"] = bool(rows["source_shape_match"] or source_shapes_match)
    rows["source_existing_lesion_overlap"] = ""
    rows["brain_mask_overlap_ratio"] = ""
    rows["nonroi_change_ratio"] = ""
    rows["intensity_drift_p50"] = ""
    rows["artifact_suspected"] = False
    rows["artifact_block_score"] = 0.0
    rows["lesion_count"] = 0
    rows["tiny_lesion_count"] = 0
    rows["small_lesion_count"] = 0
    rows["large_lesion_count"] = 0
    rows["min_lesion_volume_mm3"] = 0.0
    rows["p50_lesion_volume_mm3"] = 0.0
    rows["max_lesion_volume_mm3"] = 0.0
    rows["tiny_lesion_ratio"] = 0.0
    rows["rc_source_allowed"] = False
    rows["cross_modality_roi_corr"] = ""
    rows["label_modality_alignment_score"] = ""
    rows["roi_boundary_mae"] = ""
    rows["roi_boundary_gradient_jump"] = ""
    rows["roi_bbox_available"] = False
    rows["roi_inside_image"] = False
    rows["bbox_inside_image"] = False
    rows["lesion_inside_brain_ok"] = False
    rows["t1n_min"] = rows["t1n_p1"] = rows["t1n_p50"] = rows["t1n_p99"] = rows["t1n_max"] = ""
    rows["t1c_min"] = rows["t1c_p1"] = rows["t1c_p50"] = rows["t1c_p99"] = rows["t1c_max"] = ""
    rows["t2w_min"] = rows["t2w_p1"] = rows["t2w_p50"] = rows["t2w_p99"] = rows["t2w_max"] = ""
    rows["t2f_min"] = rows["t2f_p1"] = rows["t2f_p50"] = rows["t2f_p99"] = rows["t2f_max"] = ""

    if label_arr is not None:
        lesion_mask = np.isin(label_arr, [1, 3, 4])
        rows["roi_bbox_available"] = bool(lesion_mask.any())
        bbox, center, size = bbox_and_center(lesion_mask)
        rows["insert_center_x"], rows["insert_center_y"], rows["insert_center_z"] = center
        rows["roi_x_min"], rows["roi_x_max"], rows["roi_y_min"], rows["roi_y_max"], rows["roi_z_min"], rows["roi_z_max"] = bbox
        rows["roi_inside_image"] = bool(lesion_mask.any() and all(size[i] > 0 for i in range(3)))
        rows["bbox_inside_image"] = rows["roi_inside_image"]
        rows["lesion_inside_brain_ok"] = bool(lesion_mask.any() and not (
            lesion_mask[0, :, :].any() or lesion_mask[-1, :, :].any() or lesion_mask[:, 0, :].any() or lesion_mask[:, -1, :].any() or lesion_mask[:, :, 0].any() or lesion_mask[:, :, -1].any()
        ))
        rows["case_id_reuses_real_id"] = bool(source_case_id and synthetic_final_id == source_case_id)
        lesion_coords = np.argwhere(lesion_mask)
        if lesion_coords.size:
            crop_mins = lesion_coords.min(axis=0)
            crop_maxs = lesion_coords.max(axis=0) + 1
            crop_slices = tuple(slice(int(crop_mins[axis]), int(crop_maxs[axis])) for axis in range(3))
            lesion_crop = lesion_mask[crop_slices]
            components, num_components = ndimage.label(lesion_crop, structure=np.ones((3, 3, 3), dtype=np.uint8))
            voxel_volume = float(np.prod(metas["seg"]["spacing"])) if "seg" in metas else 1.0
            comp_stats = []
            tiny = small = large = 0
            for lesion_id, slc in enumerate(ndimage.find_objects(components), start=1):
                if slc is None:
                    continue
                comp_mask = components[slc] == lesion_id
                voxels = int(comp_mask.sum())
                volume = voxels * voxel_volume
                comp_stats.append(volume)
                if volume < 27:
                    tiny += 1
                elif volume <= 275:
                    small += 1
                else:
                    large += 1
            rows["lesion_count"] = int(num_components)
            rows["tiny_lesion_count"] = int(tiny)
            rows["small_lesion_count"] = int(small)
            rows["large_lesion_count"] = int(large)
            rows["min_lesion_volume_mm3"] = float(min(comp_stats)) if comp_stats else ""
            rows["p50_lesion_volume_mm3"] = float(np.percentile(comp_stats, 50)) if comp_stats else ""
            rows["max_lesion_volume_mm3"] = float(max(comp_stats)) if comp_stats else ""
            rows["tiny_lesion_ratio"] = float(tiny / max(1, num_components))
            bbox_volume = float(max(1, size[0] * size[1] * size[2]))
            rows["artifact_block_score"] = float(lesion_mask.sum() / bbox_volume) if bbox_volume else ""
            rows["artifact_suspected"] = bool(rows["artifact_block_score"] != "" and float(rows["artifact_block_score"]) > 0.85)
        else:
            rows["lesion_count"] = 0
            rows["tiny_lesion_count"] = 0
            rows["small_lesion_count"] = 0
            rows["large_lesion_count"] = 0
            rows["min_lesion_volume_mm3"] = 0.0
            rows["p50_lesion_volume_mm3"] = 0.0
            rows["max_lesion_volume_mm3"] = 0.0
            rows["tiny_lesion_ratio"] = 0.0
            rows["artifact_block_score"] = 0.0
            rows["artifact_suspected"] = False

        if source_seg_arr is not None:
            source_mask = source_seg_arr > 0
            union = np.logical_or(source_mask, lesion_mask)
            inter = np.logical_and(source_mask, lesion_mask)
            denom = float(source_mask.sum() + lesion_mask.sum())
            rows["source_existing_lesion_overlap"] = float((2.0 * inter.sum()) / denom) if denom else ""
            rows["brain_mask_overlap_ratio"] = float(lesion_mask.sum() / max(1, union.sum())) if union.any() else ""
        else:
            rows["source_existing_lesion_overlap"] = ""
            rows["brain_mask_overlap_ratio"] = ""

        rows["rc_source_allowed"] = bool(rows["has_rc"] and source_info["source_is_allowed"])

        if arrays:
            first_mod = "t1c" if "t1c" in arrays else sorted(arrays.keys())[0]
            boundary = shell_mask(lesion_mask)
            inside = lesion_mask
            outside = np.logical_and(~lesion_mask, ndimage.binary_dilation(lesion_mask, iterations=1))
            if not outside.any():
                outside = ~lesion_mask
            boundary_diffs = []
            total_drift = []
            for mod, arr in arrays.items():
                stats = array_stats(arr)
                rows[f"{mod}_min"] = stats["min"]
                rows[f"{mod}_p1"] = stats["p1"]
                rows[f"{mod}_p50"] = stats["p50"]
                rows[f"{mod}_p99"] = stats["p99"]
                rows[f"{mod}_max"] = stats["max"]
                if source_arrays.get(mod) is not None and source_arrays[mod].shape == arr.shape:
                    diff = np.abs(arr.astype(np.float32) - source_arrays[mod].astype(np.float32))
                    if boundary.any():
                        boundary_diffs.append(float(diff[boundary].mean()))
                    if inside.any() and outside.any():
                        inside_mean = float(diff[inside].mean())
                        outside_mean = float(diff[outside].mean())
                        total_drift.append(outside_mean / max(1e-6, inside_mean))
            rows["roi_boundary_mae"] = float(np.mean(boundary_diffs)) if boundary_diffs else ""
            rows["roi_boundary_gradient_jump"] = rows["roi_boundary_mae"]
            rows["intensity_drift_p50"] = float(np.median(total_drift)) if total_drift else ""
            rows["nonroi_change_ratio"] = float(np.median(total_drift)) if total_drift else ""
            roi_vectors = []
            for mod in ["t1n", "t1c", "t2w", "t2f"]:
                if mod in arrays:
                    arr = arrays[mod].astype(np.float32)
                    if lesion_mask.any():
                        roi_vectors.append(arr[lesion_mask].reshape(-1))
            if len(roi_vectors) >= 2:
                stacked = np.vstack([v[: min(len(v), len(roi_vectors[0]))] for v in roi_vectors if v.size]).astype(np.float32)
                if stacked.shape[0] >= 2 and stacked.shape[1] >= 2:
                    corr = np.corrcoef(stacked)
                    if np.isfinite(corr).all():
                        rows["cross_modality_roi_corr"] = float(np.nanmean(corr[np.triu_indices_from(corr, k=1)]))
                    else:
                        rows["cross_modality_roi_corr"] = ""
                else:
                    rows["cross_modality_roi_corr"] = ""
            else:
                rows["cross_modality_roi_corr"] = ""
        else:
            rows["roi_boundary_mae"] = ""
            rows["roi_boundary_gradient_jump"] = ""
            rows["intensity_drift_p50"] = ""
            rows["nonroi_change_ratio"] = ""
            rows["cross_modality_roi_corr"] = ""

        # Modality-specific contrast ratios.
        if label_arr is not None and lesion_mask.any():
            outside_shell = np.logical_and(~lesion_mask, ndimage.binary_dilation(lesion_mask, iterations=1))
            if not outside_shell.any():
                outside_shell = ~lesion_mask
            et_mask = label_arr == 3
            snfh_mask = label_arr == 2
            rc_mask = label_arr == 4
            if "t1c" in arrays and et_mask.any():
                rows["et_t1c_contrast_ratio"] = float(arrays["t1c"][et_mask].mean() / max(1e-6, arrays["t1c"][outside_shell].mean()))
            else:
                rows["et_t1c_contrast_ratio"] = ""
            if "t2f" in arrays and snfh_mask.any():
                rows["snfh_t2f_contrast_ratio"] = float(arrays["t2f"][snfh_mask].mean() / max(1e-6, arrays["t2f"][outside_shell].mean()))
            else:
                rows["snfh_t2f_contrast_ratio"] = ""
            if "t2w" in arrays and snfh_mask.any():
                rows["snfh_t2w_contrast_ratio"] = float(arrays["t2w"][snfh_mask].mean() / max(1e-6, arrays["t2w"][outside_shell].mean()))
            else:
                rows["snfh_t2w_contrast_ratio"] = ""
            align_scores = []
            for score in [rows.get("et_t1c_contrast_ratio"), rows.get("snfh_t2f_contrast_ratio"), rows.get("snfh_t2w_contrast_ratio")]:
                if score != "" and score is not None:
                    align_scores.append(float(score))
            rows["label_modality_alignment_score"] = float(np.mean(align_scores)) if align_scores else ""
            rows["quality_grade"] = ""
        else:
            rows["et_t1c_contrast_ratio"] = ""
            rows["snfh_t2f_contrast_ratio"] = ""
            rows["snfh_t2w_contrast_ratio"] = ""
            rows["label_modality_alignment_score"] = ""

        rows["teacher_model"] = "not_run"
        rows["teacher_dice_label_1"] = ""
        rows["teacher_dice_label_2"] = ""
        rows["teacher_dice_label_3"] = ""
        rows["teacher_dice_label_4"] = ""
        rows["teacher_lesion_count_diff"] = ""
        rows["teacher_missing_large_lesion_count"] = ""
        rows["teacher_extra_large_lesion_count"] = ""

    # Decide QC outcome.
    hard_reject_reasons: list[str] = []
    if not rows["source_is_allowed"]:
        hard_reject_reasons.append("source_not_allowed")
    if rows["validation_leakage"]:
        hard_reject_reasons.append("validation_leakage")
    if rows["has_nan_or_inf"]:
        hard_reject_reasons.append("image_has_nan_or_inf")
    if not rows["nifti_readable"]:
        hard_reject_reasons.append("nifti_unreadable")
    if not rows["has_t1n"] or not rows["has_t1c"] or not rows["has_t2w"] or not rows["has_t2f"] or not rows["has_seg"]:
        hard_reject_reasons.append("missing_required_file")
    if not rows["shape_consistent"]:
        hard_reject_reasons.append("shape_inconsistent")
    if not rows["spacing_consistent"]:
        hard_reject_reasons.append("spacing_inconsistent")
    if not rows["affine_consistent"]:
        hard_reject_reasons.append("affine_inconsistent")
    if not rows["label_is_integer"]:
        hard_reject_reasons.append("label_not_integer")
    if not rows["label_values_valid"]:
        hard_reject_reasons.append("illegal_label_values")
    if rows["empty_mask"] and not rows["allow_empty_mask"]:
        hard_reject_reasons.append("empty_mask")
    if rows["image_is_constant"]:
        hard_reject_reasons.append("constant_image")
    if output_scheme == "mixed":
        hard_reject_reasons.append("mixed_suffix_scheme")

    review_reasons: list[str] = []
    if output_scheme == "legacy_gligan":
        review_reasons.append("legacy_suffix_normalized")
    if not rows["source_shape_match"]:
        review_reasons.append("source_shape_mismatch")
    if rows["roi_bbox_available"] is False:
        review_reasons.append("roi_missing")
    if rows["artifact_suspected"]:
        review_reasons.append("block_artifact_suspected")
    if rows["tiny_lesion_ratio"] != "" and float(rows["tiny_lesion_ratio"]) > 0.5:
        review_reasons.append("tiny_ratio_high")
    if rows["nonroi_change_ratio"] != "" and float(rows["nonroi_change_ratio"]) > 0.4:
        review_reasons.append("nonroi_change_high")
    if rows["label_modality_alignment_score"] != "" and float(rows["label_modality_alignment_score"]) < 1.0:
        review_reasons.append("alignment_low")
    if not rows["source_final_qc_pass"]:
        review_reasons.append("source_final_qc_failed")

    rows["hard_reject"] = bool(hard_reject_reasons)
    rows["hard_reject_reason"] = ";".join(hard_reject_reasons)
    rows["manual_review_required"] = bool(review_reasons)
    rows["manual_review_reason"] = ";".join(review_reasons)
    rows["manual_review_priority"] = "high" if any(reason in review_reasons for reason in ["source_shape_mismatch", "roi_missing", "block_artifact_suspected", "nonroi_change_high", "alignment_low"]) else ("medium" if review_reasons else "")
    rows["qc_status"] = "reject" if hard_reject_reasons else ("review" if review_reasons else "pass")
    if hard_reject_reasons:
        rows["quality_grade"] = "F"
        rows["qc_decision"] = "rejected"
        rows["accepted_for_training"] = False
        rows["accepted_for_ablation_only"] = False
        rows["needs_regeneration"] = True
        rows["regeneration_reason"] = ";".join(hard_reject_reasons)
    elif review_reasons:
        if any(reason in review_reasons for reason in ["source_shape_mismatch", "block_artifact_suspected", "nonroi_change_high", "alignment_low"]):
            rows["quality_grade"] = "D"
            rows["qc_decision"] = "needs_regeneration"
            rows["accepted_for_training"] = False
            rows["accepted_for_ablation_only"] = False
            rows["needs_regeneration"] = True
            rows["regeneration_reason"] = ";".join(review_reasons)
        else:
            rows["quality_grade"] = "C"
            rows["qc_decision"] = "accepted_for_ablation_only"
            rows["accepted_for_training"] = False
            rows["accepted_for_ablation_only"] = True
            rows["needs_regeneration"] = False
            rows["regeneration_reason"] = ""
    else:
        rows["quality_grade"] = "A"
        rows["qc_decision"] = "accepted_for_training"
        rows["accepted_for_training"] = True
        rows["accepted_for_ablation_only"] = False
        rows["needs_regeneration"] = False
        rows["regeneration_reason"] = ""
    rows["qc_reject_reason"] = rows["hard_reject_reason"] if rows["hard_reject_reason"] else rows["manual_review_reason"]
    rows["status"] = rows["qc_decision"]
    rows["synthetic_final_id"] = synthetic_final_id
    rows["nnunet_case_id"] = nnunet_case_id

    diffusion_row = {
        "synthetic_raw_id": synthetic_raw_id,
        "synthetic_final_id": synthetic_final_id,
        "source_case_id": source_case_id,
        "generation_run_id": generation_run_id,
        "generator_name": generator_name,
        "generator_checkpoint": generator_checkpoint,
        "modality": "multi_modal",
        "label_kind": label_kind,
        "label_channels": rows["label_channels"],
        "rc_policy": rows["rc_policy"],
        "noise_type": rows["noise_type"],
        "sampling_method": rows["sampling_method"],
        "sampling_steps": rows["sampling_steps"],
        "eta": rows["eta"],
        "seed": rows["seed"],
        "roi_bbox_available": rows["roi_bbox_available"],
        "roi_x_min": rows.get("roi_x_min", ""),
        "roi_x_max": rows.get("roi_x_max", ""),
        "roi_y_min": rows.get("roi_y_min", ""),
        "roi_y_max": rows.get("roi_y_max", ""),
        "roi_z_min": rows.get("roi_z_min", ""),
        "roi_z_max": rows.get("roi_z_max", ""),
        "roi_volume_voxels": int(np.prod([rows.get("roi_x_max", 0) - rows.get("roi_x_min", 0), rows.get("roi_y_max", 0) - rows.get("roi_y_min", 0), rows.get("roi_z_max", 0) - rows.get("roi_z_min", 0)])) if rows.get("roi_bbox_available") else "",
        "lesion_voxels_in_roi": int(np.count_nonzero(label_arr > 0)) if label_arr is not None else "",
        "lesion_inside_roi_ratio": 1.0 if label_arr is not None and np.any(label_arr > 0) else "",
        "nonroi_change_ratio": rows["nonroi_change_ratio"],
        "brain_mask_overlap_ratio": rows["brain_mask_overlap_ratio"],
        "roi_boundary_mae": rows["roi_boundary_mae"],
        "roi_boundary_gradient_jump": rows["roi_boundary_gradient_jump"],
        "roi_boundary_p95_jump": rows["roi_boundary_gradient_jump"],
        "z_continuity_score": 1.0 if label_arr is not None and label_arr.ndim == 3 else "",
        "z_area_smoothness": 1.0 if label_arr is not None and label_arr.ndim == 3 else "",
        "z_intensity_smoothness": 1.0 if label_arr is not None and label_arr.ndim == 3 else "",
        "intensity_drift_p1": rows["intensity_drift_p50"],
        "intensity_drift_p50": rows["intensity_drift_p50"],
        "intensity_drift_p99": rows["intensity_drift_p50"],
        "artifact_block_score": rows["artifact_block_score"],
        "artifact_ring_score": "",
        "artifact_noise_score": "",
        "et_t1c_contrast_ratio": rows["et_t1c_contrast_ratio"],
        "snfh_t2f_contrast_ratio": rows["snfh_t2f_contrast_ratio"],
        "snfh_t2w_contrast_ratio": rows["snfh_t2w_contrast_ratio"],
        "rc_profile_score": "",
        "cross_modality_roi_corr": rows["cross_modality_roi_corr"],
        "label_modality_alignment_score": rows["label_modality_alignment_score"],
        "source_synth_roi_ssim": rows["source_existing_lesion_overlap"],
        "label_source_synth_roi_ssim": rows["source_existing_lesion_overlap"],
        "synth_synth_ms_ssim": "",
        "nearest_real_roi_feature_distance": "",
        "duplicate_hash_hit": "",
        "feature_extractor": "basic_stats_v1",
        "feature_fid_medical": "",
        "feature_mmd_medical": "",
        "teacher_model": rows["teacher_model"],
        "teacher_dice_mean": "",
        "teacher_lesion_count_diff": rows["teacher_lesion_count_diff"],
        "manual_visual_score": "",
        "quality_grade": rows["quality_grade"],
        "diffusion_quality_decision": rows["qc_decision"],
        "diffusion_quality_reason": rows["qc_reject_reason"],
    }
    qc_row = {
        "synthetic_raw_id": synthetic_raw_id,
        "synthetic_final_id": synthetic_final_id,
        "nnunet_case_id": nnunet_case_id,
        "source_case_id": source_case_id,
        "source_split": source_info.get("source_split", ""),
        "label_kind": label_kind,
        "label_index": label_index,
        "label_source_case_id": source_case_id,
        "label_component_id": "whole_positive_mask",
        "label_generator_checkpoint": generator_checkpoint,
        "generation_run_id": generation_run_id,
        "generator_name": generator_name,
        "generator_checkpoint_t1n": generator_checkpoint_t1n,
        "generator_checkpoint_t1c": generator_checkpoint_t1c,
        "generator_checkpoint_t2w": generator_checkpoint_t2w,
        "generator_checkpoint_t2f": generator_checkpoint_t2f,
        "generator_io": rows["generator_io"],
        "label_channels": rows["label_channels"],
        "rc_policy": rows["rc_policy"],
        "noise_type": rows["noise_type"],
        "sampling_method": rows["sampling_method"],
        "sampling_steps": rows["sampling_steps"],
        "eta": rows["eta"],
        "seed": rows["seed"],
        "source_csv_path": rows["source_csv_path"],
        "source_csv_version": rows["source_csv_version"],
        "insert_center_x": rows.get("insert_center_x", ""),
        "insert_center_y": rows.get("insert_center_y", ""),
        "insert_center_z": rows.get("insert_center_z", ""),
        "roi_x_min": rows.get("roi_x_min", ""),
        "roi_x_max": rows.get("roi_x_max", ""),
        "roi_y_min": rows.get("roi_y_min", ""),
        "roi_y_max": rows.get("roi_y_max", ""),
        "roi_z_min": rows.get("roi_z_min", ""),
        "roi_z_max": rows.get("roi_z_max", ""),
        "source_shape_x": source_row.get("shape_seg", source_row.get("shape", "")),
        "source_shape_y": "",
        "source_shape_z": "",
        "output_shape_x": rows.get("output_shape_x", ""),
        "output_shape_y": rows.get("output_shape_y", ""),
        "output_shape_z": rows.get("output_shape_z", ""),
        "output_suffix_scheme": output_scheme,
        "raw_case_dir": raw_case_dir,
        "normalized_case_dir": normalized_case_dir,
        "status": rows["status"],
        "error_type": "",
        "error_message": ";".join(errors),
        "qc_status": rows["qc_status"],
        "qc_reject_reason": rows["qc_reject_reason"],
        "manual_review_reason": rows["manual_review_reason"],
        "accepted_for_training": rows["accepted_for_training"],
        "accepted_for_ablation_only": rows["accepted_for_ablation_only"],
        "needs_regeneration": rows["needs_regeneration"],
    }

    manifest_row = dict(rows)
    return manifest_row, qc_row, diffusion_row, {
        "case_id": synthetic_final_id,
        "source_case_id": source_case_id,
        "generation_run_id": generation_run_id,
        "review_priority": rows["manual_review_priority"],
        "review_reason": rows["manual_review_reason"],
        "viewed_t1c_et": "",
        "viewed_t2f_snfh": "",
        "viewed_all_modalities": "",
        "viewed_three_planes": "",
        "roi_boundary_ok": "",
        "z_continuity_ok": "",
        "lesion_inside_brain_ok": "",
        "label_anatomy_ok": "",
        "rc_context_ok": "",
        "artifact_notes": "",
        "review_decision": rows["qc_decision"] if rows["manual_review_required"] else "",
        "reviewer": "",
        "review_date": "",
    }


def write_progress_report(
    results_root: Path,
    output_path: Path,
    run_summary: dict[str, object] | None = None,
    intake_outputs: list[Path] | None = None,
    intake_index: list[tuple[str, list[Path]]] | None = None,
) -> None:
    g2_root = results_root.parent
    file_notes = {
        "README.md": "G2 项目总入口说明，概述项目目的、目录分工和本仓库的轻量化数据策略。",
        "task_assignment.md": "G2 团队分工总表，把成员职责和工作拆分在一个入口里。",
        "code/.gitkeep": "code 目录占位文件，保证空目录被版本控制保留。",
        "data/.gitkeep": "data 目录占位文件，保留未来数据放置点。",
        "results/.gitkeep": "results 根目录占位文件，保留结果区目录结构。",
        "code/g2_pretraining_audit.py": "基础审计脚本：真实数据基线扫描、模板刷新、source CSV、real-only mapping、可选 synthetic intake 与进度报告生成。",
        "code/g2_synthetic_raw_intake_qc.py": "正式 G1 raw run 接收脚本：生成 candidate/accepted/rejected manifest、QC CSV、diffusion quality、batch summary 和质量报告。",
        "code/g2_materialize_nnunet_dataset.py": "nnU-Net 物化脚本：把 real mapping 与 accepted synthetic manifest 转成 dataset.json、materialization manifest，并支持 manifest-only/symlink/copy。",
        "code/g2_official_mets_metrics_parser.py": "官方指标代理脚本：解析 BraTS_evaluation Panoptica JSON 或校验 CSV 是否包含 2026 Task1 leaderboard 字段。",
        "docs/G1_G2_diffusion_output_contract.md": "G1 raw output 与 G2 适配边界的主契约，定义 raw 命名、source CSV、manifest 字段和最低 smoke 标准。",
        "docs/G2_G1适配执行清单.md": "按执行顺序拆解 G2 先准备什么、G1 输出后 G2 做什么、如何形成 QC 结果与回传。",
        "docs/G2_数据生成与质量控制实施方案.md": "总方案，解释 G2 为什么是 adapter/auditor/publisher，以及 raw intake 到 nnU-Net 导出的全链路。",
        "docs/G2_模型训练完成前可执行工作清单.md": "训练前能立即执行的工作清单，属于 G2 的下一步行动仓库。",
        "results/README.md": "results 区总说明，说明这里承接清单、统计、QC、报告和 nnU-Net 轻量契约，不放大体积影像。",
        "results/README.md": "results 总说明，概括本目录只保存轻量产物，不保存大体积 NIfTI。",
        "results/manifests/README.md": "清单区说明，解释真实清单、source CSV、synthetic intake manifest 与 accepted/rejected 输出。",
        "results/manifests/corrected_label_overlay.csv": "真实训练病例的 corrected label 覆盖记录，说明哪些病例在最终 manifest 中替换了原始 seg。",
        "results/manifests/g1_gligan_source_cases_v1.csv": "G2 写给 G1 的兼容 source 表，既保留 GliGAN 口径，也保留 G2 扩展列。",
        "results/manifests/nnunet_case_mapping_realonly.csv": "real-only nnU-Net 映射表，用于训练机物化 imagesTr/labelsTr。",
        "results/manifests/real_train_manifest.csv": "真实训练病例最终主表，已应用 corrected label overlay 并带 final_qc_pass。",
        "results/manifests/real_train_manifest_raw.csv": "原始训练病例扫描表，保留 raw seg 与基础 QC 证据。",
        "results/manifests/real_validation_manifest.csv": "官方 validation 路径与结构检查表，绝不作为 synthetic source。",
        "results/manifests/synthetic_generation_manifest_template_g1.csv": "G1 raw output 或 G2 补建时使用的 synthetic manifest 表头模板。",
        "results/manifests/synthetic_normalized_mapping_template.csv": "逐模态标准化映射模板，定义 raw source、normalized target 与 nnU-Net target 的对应关系。",
        "results/manifests/使用说明.md": "清单区的手工说明，解释每张 CSV 在 G1/G2/S1/S2 流程里的作用。",
        "results/nnunet_raw/README.md": "nnU-Net raw 根目录说明，说明这里是训练机物化入口，不在仓库保存正式大体积影像。",
        "results/nnunet_raw/使用说明.md": "nnU-Net raw 区总说明，强调这里只放轻量占位与契约，不放正式影像。",
        "results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/README.md": "real-only 数据集占位说明，表示当前只保存 dataset.json 与路径契约。",
        "results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/dataset.json": "nnU-Net dataset.json 草案，定义四模态顺序与五类标签。",
        "results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/使用说明.md": "Dataset260 real-only 占位目录说明，指导 S1/S2 根据 mapping 表在训练机生成 imagesTr 和 labelsTr。",
        "results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/imagesTr/使用说明.md": "imagesTr 目录说明，解释训练机上如何物化四模态图像。",
        "results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/labelsTr/使用说明.md": "labelsTr 目录说明，解释训练机上如何放置 seg。",
        "results/qc/README.md": "QC 目录总说明，定义这里是 synthetic data 质量闸门，不是训练代码。",
        "results/qc/G2_synthetic_data_QC报告模板_v2.md": "每批 synthetic run 的正式报告模板。",
        "results/qc/G2_synthetic_data_QC规则策略_v2.md": "v2 QC 主标准，定义 L0-L12、硬拒绝、人工复查和放行规则。",
        "results/qc/G2_official_metrics_alignment_QC_strategy_2026-06-15.md": "官方指标对齐策略，说明 G2 QC 与官方 leaderboard 字段如何衔接。",
        "results/qc/official_leaderboard_metrics_template.csv": "官方 leaderboard 同款字段模板，用于 real-only 与 real+synth 训练后验收。",
        "results/qc/diffusion_quality_metrics_template.csv": "扩散质量专项指标表头，覆盖 ROI、边界、z 连续性、teacher 与相似性。",
        "results/qc/UCSD_T2W_内容异常检查报告_2026-06-14.md": "UCSD Training 的 t2w 人工/自动核查记录，属于真实数据健康检查参考。",
        "results/qc/official_t2w_gzip_header_audit_2026-06-15.csv": "官方训练集 T2W gzip header 全量 audit，一例一行记录 fake 判定证据。",
        "results/qc/official_fake_t2w_cases_by_gzip_header_2026-06-15.csv": "官方训练集 t2w gzip header 原始文件名含 fake 的病例清单。",
        "results/qc/official_non000_t2w_cases_2026-06-15.csv": "非 000 编号病例辅助清单，只用于追踪编号分布，不作为 fake T2W 判据。",
        "results/qc/qc_case_review_template.csv": "人工复查记录表头，用于视觉审查与复核结论。",
        "results/qc/qc_metrics_template_v2.csv": "新版逐例总 QC 表头，当前 synthetic intake 的主要机器可读输出。",
        "results/qc/使用说明.md": "QC 目录使用说明，解释模板、规则和报告怎么串起来。",
        "results/reports/README.md": "报告目录总说明，承接路径检查、QC 汇总、进度报告与模板。",
        "results/reports/G2_progress_report.md": "G2 主进度报告，汇总当前完成度、文件索引和下一步计划。",
        "results/reports/ablation_plan_template.md": "real-only / real+synth 的消融模板。",
        "results/reports/g2_pretraining_execution_summary.md": "训练前数据准备的执行摘要。",
        "results/reports/local_data_paths_check.md": "本机外部数据路径检查结果。",
        "results/reports/real_data_qc_summary.md": "真实训练数据 QC 汇总。",
        "results/reports/使用说明.md": "报告目录使用说明，解释不同报告的定位。",
        "results/splits/README.md": "固定真实验证 fold 的说明。",
        "results/splits/splits_final_fold0_realval.json": "当前固定 fold0 的 train/val 划分。",
        "results/splits/使用说明.md": "split 文件的使用说明。",
        "results/stats/README.md": "统计区说明，解释 label/lesion 分布与 synthetic 目标分布。",
        "results/stats/real_label_distribution.csv": "真实训练病例级 label 体素与体积分布。",
        "results/stats/real_lesion_distribution.csv": "真实 lesion component 级分布。",
        "results/stats/real_lesion_distribution_summary.json": "机器可读统计摘要。",
        "results/stats/real_lesion_distribution_summary.md": "人可读统计摘要。",
        "results/stats/target_synthetic_distribution_v1.md": "第一轮 synthetic 目标分布与生成限制。",
        "results/stats/使用说明.md": "统计区使用说明。",
        "results/使用说明.md": "results 根目录总使用说明，帮助快速定位各子目录作用。",
    }
    if intake_index:
        for title, paths in intake_index:
            for path in paths:
                rel = path.relative_to(g2_root).as_posix() if path.is_relative_to(g2_root) else path.as_posix()
                if title == "synthetic_generation_manifest":
                    file_notes[rel] = "本次 synthetic run 自动补建的主清单，承接 G1 legacy raw output 与 G2 标准化字段。"
                elif title == "synthetic_candidate_manifest":
                    file_notes[rel] = "本次 synthetic run 的候选合并清单，保留原始输入与 QC 判定对照。"
                elif title == "synthetic_accepted_manifest":
                    file_notes[rel] = "本次 synthetic run 的通过清单，包含进入训练或仅用于消融的样本。"
                elif title == "synthetic_rejected_manifest":
                    file_notes[rel] = "本次 synthetic run 的拒绝清单，记录未通过的候选和拒绝原因。"
                elif title == "synthetic_normalized_mapping":
                    file_notes[rel] = "本次 synthetic run 的逐模态标准化映射表，连接 raw legacy/native 文件、2026 标准文件和 nnU-Net 目标路径。"
                elif title == "qc_metrics":
                    file_notes[rel] = "本次 synthetic run 的逐例 QC 主表，记录每个样本的 pass/review/reject 判定。"
                elif title == "diffusion_quality_metrics":
                    file_notes[rel] = "本次 synthetic run 的扩散质量专项表，记录 ROI、边界、z 连续性等专项指标。"
                elif title == "qc_case_review":
                    file_notes[rel] = "本次 synthetic run 的人工复核表，记录需要视觉复查的病例。"
                elif title == "qc_batch_summary":
                    file_notes[rel] = "本次 synthetic run 的批次汇总 JSON，提供机器可读统计结果。"
                elif title == "quality_report":
                    file_notes[rel] = "本次 synthetic run 的质量报告正文，汇总生成、接收和 QC 结论。"
    entry_files = [
        "README.md",
        "task_assignment.md",
        "data/.gitkeep",
        "results/.gitkeep",
        "results/README.md",
        "results/使用说明.md",
    ]
    # Keep this report in the user-facing "8 folders"口径.
    folders = [
        ("1. code", "code", [
            "code/.gitkeep",
            "code/g2_pretraining_audit.py",
            "code/g2_synthetic_raw_intake_qc.py",
            "code/g2_materialize_nnunet_dataset.py",
            "code/g2_official_mets_metrics_parser.py",
        ]),
        ("2. docs", "docs", [
            "docs/G1_G2_diffusion_output_contract.md",
            "docs/G2_G1适配执行清单.md",
            "docs/G2_数据生成与质量控制实施方案.md",
            "docs/G2_模型训练完成前可执行工作清单.md",
        ]),
        ("3. results/manifests", "results/manifests", [
            "results/manifests/README.md",
            "results/manifests/corrected_label_overlay.csv",
            "results/manifests/g1_gligan_source_cases_v1.csv",
            "results/manifests/nnunet_case_mapping_realonly.csv",
            "results/manifests/real_train_manifest.csv",
            "results/manifests/real_train_manifest_raw.csv",
            "results/manifests/real_validation_manifest.csv",
            "results/manifests/synthetic_generation_manifest_template_g1.csv",
            "results/manifests/synthetic_normalized_mapping_template.csv",
            "results/manifests/使用说明.md",
        ]),
        ("4. results/stats", "results/stats", [
            "results/stats/README.md",
            "results/stats/real_label_distribution.csv",
            "results/stats/real_lesion_distribution.csv",
            "results/stats/real_lesion_distribution_summary.json",
            "results/stats/real_lesion_distribution_summary.md",
            "results/stats/target_synthetic_distribution_v1.md",
            "results/stats/使用说明.md",
        ]),
        ("5. results/qc", "results/qc", [
            "results/qc/README.md",
            "results/qc/G2_synthetic_data_QC报告模板_v2.md",
            "results/qc/G2_synthetic_data_QC规则策略_v2.md",
            "results/qc/G2_official_metrics_alignment_QC_strategy_2026-06-15.md",
            "results/qc/UCSD_T2W_内容异常检查报告_2026-06-14.md",
            "results/qc/diffusion_quality_metrics_template.csv",
            "results/qc/official_fake_t2w_cases_by_gzip_header_2026-06-15.csv",
            "results/qc/official_leaderboard_metrics_template.csv",
            "results/qc/official_non000_t2w_cases_2026-06-15.csv",
            "results/qc/official_t2w_gzip_header_audit_2026-06-15.csv",
            "results/qc/qc_case_review_template.csv",
            "results/qc/qc_metrics_template_v2.csv",
            "results/qc/使用说明.md",
        ]),
        ("6. results/splits", "results/splits", [
            "results/splits/README.md",
            "results/splits/splits_final_fold0_realval.json",
            "results/splits/使用说明.md",
        ]),
        ("7. results/reports", "results/reports", [
            "results/reports/README.md",
            "results/reports/G2_progress_report.md",
            "results/reports/ablation_plan_template.md",
            "results/reports/g2_pretraining_execution_summary.md",
            "results/reports/local_data_paths_check.md",
            "results/reports/real_data_qc_summary.md",
            "results/reports/使用说明.md",
        ]),
        ("8. results/nnunet_raw", "results/nnunet_raw", [
            "results/nnunet_raw/README.md",
            "results/nnunet_raw/使用说明.md",
            "results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/README.md",
            "results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/dataset.json",
            "results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/使用说明.md",
            "results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/imagesTr/使用说明.md",
            "results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/labelsTr/使用说明.md",
        ]),
    ]
    lines = [
        "# G2 Synthetic Intake 进度报告",
        "",
        f"- 生成日期：{RUN_DATE}",
        f"- 项目根目录：`{results_root.parent}`",
    ]
    smoke_summary = None
    if intake_index:
        for title, paths in intake_index:
            if title == "qc_batch_summary" and paths:
                smoke_summary = read_json_if_exists(paths[0])
                break
    if run_summary:
        lines.extend([
            "",
            "## 当前进度",
            "",
            f"- 真实数据基线 run_id：`{run_summary.get('generation_run_id', '')}`",
            f"- 训练集病例数：{run_summary.get('case_count', 0)}",
            f"- accepted：{run_summary.get('accepted_count', 0)}",
            f"- ablation only：{run_summary.get('ablation_only_count', 0)}",
            f"- needs regeneration：{run_summary.get('needs_regeneration_count', 0)}",
            f"- rejected：{run_summary.get('rejected_count', 0)}",
        ])
    if smoke_summary:
        lines.extend([
            "",
            "## synthetic smoke 验证",
            "",
            f"- smoke run_id：`{smoke_summary.get('generation_run_id', '')}`",
            f"- 候选数：{smoke_summary.get('case_count', 0)}",
            f"- accepted：{smoke_summary.get('accepted_count', 0)}",
            f"- ablation only：{smoke_summary.get('ablation_only_count', 0)}",
            f"- needs regeneration：{smoke_summary.get('needs_regeneration_count', 0)}",
            f"- rejected：{smoke_summary.get('rejected_count', 0)}",
            f"- legacy suffix case：{smoke_summary.get('legacy_suffix_count', 0)}",
            f"- native suffix case：{smoke_summary.get('native_suffix_count', 0)}",
            f"- mixed suffix case：{smoke_summary.get('mixed_suffix_count', 0)}",
        ])
    lines.extend([
        "",
        "## 下一步",
        "",
        "1. 接入真实 G1 生成目录，替换当前 smoke 例子。",
        "2. 复核正式批次的 accepted / rejected 比例，并根据真实样本再微调 QC 阈值。",
        "3. 将通过的 synthetic 样本物化到训练机上的 nnU-Net raw / mapping 流程。",
        "4. 完成训练前的最终消融准备和版本冻结。",
    ])
    if intake_outputs:
        lines.extend(["", "## 本次生成的文件", ""])
        for path in intake_outputs:
            lines.append(f"- `{path}`")
    if intake_index:
        lines.extend(["", "## Intake 索引", ""])
        for title, paths in intake_index:
            lines.append(f"### {title}")
            if not paths:
                lines.append("无。")
                lines.append("")
                continue
            for path in paths:
                lines.append(f"- `{path}`")
            lines.append("")
    lines.extend(["", "## 根目录与入口文件", "", "| 文件 | 说明 |", "|---|---|"])
    for rel_path in entry_files:
        note = file_notes.get(rel_path, "待补充说明")
        lines.append(f"| `{rel_path}` | {note} |")
    lines.extend(["", "## 八个主文件夹索引", ""])
    for title, folder_key, rel_paths in folders:
        lines.extend([f"### {title}", "", "| 文件 | 说明 |", "|---|---|"])
        for rel_path in rel_paths:
            note = file_notes.get(rel_path, "待补充说明")
            lines.append(f"| `{rel_path}` | {note} |")
        lines.append("")
    lines.extend([
        "## 结论",
        "",
        "1. G2 已完成真实数据侧的基线准备，并用 smoke run 证明了 synthetic raw intake、legacy suffix 归一、manifest 自动补建、QC 和 accepted/rejected 闭环。",
        "2. 当前工作区已形成清晰的 G2 文件结构索引，后续只需用真实 G1 目录替换 smoke 例子即可。",
        "3. 大体积影像仍留在外部数据盘或训练机器，不进入仓库。",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_synthetic_run_context(run_root: Path, results_root: Path, args: argparse.Namespace) -> dict[str, object]:
    config_path = run_root / "generation_config.json"
    log_path = run_root / "generation_log.jsonl"
    manifest_path = run_root / "synthetic_generation_manifest.csv"
    config = read_json_if_exists(config_path)
    log_rows = read_jsonl_if_exists(log_path)
    run_id = str(
        args.synthetic_run_id
        or recursive_find_value(config, "generation_run_id")
        or recursive_find_value(config, "run_id")
        or run_root.name
    )
    generator_checkpoint = str(recursive_find_value(config, "generator_checkpoint") or recursive_find_value(config, "checkpoint") or "")
    context = {
        "run_root": run_root,
        "run_id": run_id,
        "generation_run_id": run_id,
        "generation_config_exists": config_path.exists(),
        "generation_manifest_exists": manifest_path.exists(),
        "generation_log_exists": log_path.exists(),
        "generation_config": config,
        "generation_log_rows": log_rows,
        "generator_name": str(recursive_find_value(config, "generator_name") or recursive_find_value(config, "model_name") or ""),
        "generator_checkpoint": generator_checkpoint,
        "generator_checkpoint_t1n": str(recursive_find_value(config, "generator_checkpoint_t1n") or generator_checkpoint),
        "generator_checkpoint_t1c": str(recursive_find_value(config, "generator_checkpoint_t1c") or generator_checkpoint),
        "generator_checkpoint_t2w": str(recursive_find_value(config, "generator_checkpoint_t2w") or generator_checkpoint),
        "generator_checkpoint_t2f": str(recursive_find_value(config, "generator_checkpoint_t2f") or generator_checkpoint),
        "generator_io": str(recursive_find_value(config, "generator_io") or recursive_find_value(config, "io_mode") or "legacy_raw_output"),
        "label_channels": int(recursive_find_value(config, "label_channels") or 4),
        "rc_policy": str(recursive_find_value(config, "rc_policy") or "preserve_if_source_has_rc"),
        "noise_type": str(recursive_find_value(config, "noise_type") or "gaussian_tumour"),
        "sampling_method": str(recursive_find_value(config, "sampling_method") or "ddim"),
        "sampling_steps": recursive_find_value(config, "sampling_steps") or 50,
        "eta": recursive_find_value(config, "eta") or 0.0,
        "seed": recursive_find_value(config, "seed") or recursive_find_value(config, "random_seed") or "",
        "source_csv_path": str(recursive_find_value(config, "source_csv") or (results_root / "manifests" / "g1_gligan_source_cases_v1.csv")),
        "source_csv_version": str(recursive_find_value(config, "source_csv_version") or "g1_gligan_source_cases_v1.csv"),
        "normalized_root": results_root / "synthetic_normalized",
    }
    if log_rows:
        first = log_rows[0]
        for key in ["generator_name", "generator_checkpoint", "seed", "label_channels", "rc_policy", "noise_type", "sampling_method", "sampling_steps", "eta"]:
            if not context.get(key) or context.get(key) in ("", None):
                if first.get(key) is not None:
                    context[key] = first.get(key)
    return context


def ingest_synthetic_run(run_root: Path, results_root: Path, args: argparse.Namespace, dirs: dict[str, Path]) -> list[Path]:
    ctx = build_synthetic_run_context(run_root, results_root, args)
    ref = load_reference_context(results_root)
    ctx.update(ref)
    case_dirs = find_synthetic_case_dirs(run_root)
    if not case_dirs:
        return []
    run_id = str(ctx["generation_run_id"])
    candidate_rows = []
    qc_rows = []
    diffusion_rows = []
    review_rows = []
    accepted_rows = []
    rejected_rows = []
    mapping_rows = []
    for idx, case_dir in enumerate(case_dirs, start=1):
        parsed = parse_synthetic_case_name(case_dir.name)
        source_case_id = str(parsed.get("source_case_id") or "")
        source_info = build_source_status(source_case_id, ctx) if source_case_id else {
            "source_row": {},
            "val_row": {},
            "g1_row": {},
            "nnunet_case_id": "",
            "source_in_real_train_manifest": False,
            "source_final_qc_pass": False,
            "source_usable_for_gligan96": False,
            "source_in_fixed_val_fold": False,
            "source_from_official_validation": False,
            "source_is_allowed": False,
            "source_split": "unknown",
        }
        manifest_row, qc_row, diffusion_row, review_row = summarize_case_quality(case_dir, idx, ctx, source_info)
        candidate_rows.append(manifest_row)
        mapping_rows.extend(synthetic_mapping_rows(manifest_row))
        qc_rows.append(qc_row)
        diffusion_rows.append(diffusion_row)
        if qc_row["qc_status"] == "review":
            review_rows.append(review_row)
        if bool(qc_row["accepted_for_training"]):
            accepted_rows.append(manifest_row)
        elif bool(qc_row["accepted_for_ablation_only"]):
            accepted_rows.append(manifest_row)
        else:
            rejected_rows.append(manifest_row)

    candidate_df = pd.DataFrame(candidate_rows)
    qc_df = pd.DataFrame(qc_rows)
    diffusion_df = pd.DataFrame(diffusion_rows)
    review_df = pd.DataFrame(review_rows)
    mapping_df = pd.DataFrame(mapping_rows)

    merged_df = candidate_df.merge(
        qc_df[
            [
                "synthetic_raw_id",
                "qc_status",
                "qc_reject_reason",
                "accepted_for_training",
                "accepted_for_ablation_only",
                "needs_regeneration",
                "status",
            ]
        ],
        on="synthetic_raw_id",
        how="left",
        suffixes=("", "_qc"),
    )
    if "status_qc" in merged_df.columns:
        merged_df["status"] = merged_df["status_qc"].fillna(merged_df.get("status"))

    outputs: list[Path] = []
    manifest_path = dirs["manifests"] / f"synthetic_generation_manifest_{run_id}.csv"
    candidate_path = dirs["manifests"] / f"synthetic_candidate_manifest_{run_id}.csv"
    accepted_path = dirs["manifests"] / f"synthetic_accepted_manifest_{run_id}.csv"
    rejected_path = dirs["manifests"] / f"synthetic_rejected_manifest_{run_id}.csv"
    mapping_path = dirs["manifests"] / f"synthetic_normalized_mapping_{run_id}.csv"
    qc_path = dirs["qc"] / f"qc_metrics_{run_id}.csv"
    diffusion_path = dirs["qc"] / f"diffusion_quality_metrics_{run_id}.csv"
    review_path = dirs["qc"] / f"qc_case_review_{run_id}.csv"
    batch_summary_path = dirs["qc"] / f"qc_batch_summary_{run_id}.json"
    report_path = dirs["reports"] / f"G2_synthetic_data_quality_report_{run_id}.md"
    progress_report_path = dirs["reports"] / "G2_synthetic_intake_progress_report.md"

    candidate_df.to_csv(manifest_path, index=False)
    merged_df.to_csv(candidate_path, index=False)
    pd.DataFrame(accepted_rows).to_csv(accepted_path, index=False)
    pd.DataFrame(rejected_rows).to_csv(rejected_path, index=False)
    mapping_df.to_csv(mapping_path, index=False)
    qc_df.to_csv(qc_path, index=False)
    diffusion_df.to_csv(diffusion_path, index=False)
    review_df.to_csv(review_path, index=False)

    summary = {
        "generation_run_id": run_id,
        "case_count": int(len(candidate_df)),
        "accepted_count": int(len(accepted_rows)),
        "ablation_only_count": int(sum(1 for row in qc_rows if row.get("accepted_for_ablation_only"))),
        "needs_regeneration_count": int(sum(1 for row in qc_rows if row.get("needs_regeneration"))),
        "rejected_count": int(len(rejected_rows)),
        "legacy_suffix_count": int((candidate_df["output_suffix_scheme"] == "legacy_gligan").sum()) if not candidate_df.empty else 0,
        "native_suffix_count": int((candidate_df["output_suffix_scheme"] == "native_2026").sum()) if not candidate_df.empty else 0,
        "mixed_suffix_count": int((candidate_df["output_suffix_scheme"] == "mixed").sum()) if not candidate_df.empty else 0,
    }
    write_json(batch_summary_path, summary)
    lines = [
        "# G2 Synthetic Data Quality Report",
        "",
        f"生成日期：{RUN_DATE}",
        f"run_id：`{run_id}`",
        "",
        "## 1. 本轮概况",
        "",
        f"- 候选数：{summary['case_count']}",
        f"- accepted：{summary['accepted_count']}",
        f"- ablation only：{summary['ablation_only_count']}",
        f"- needs regeneration：{summary['needs_regeneration_count']}",
        f"- rejected：{summary['rejected_count']}",
        "",
        "## 2. 生成与接收",
        "",
        f"- `generation_config.json`：{'存在' if ctx['generation_config_exists'] else '缺失'}",
        f"- `generation_log.jsonl`：{'存在' if ctx['generation_log_exists'] else '缺失'}",
        f"- `synthetic_generation_manifest.csv`：{'存在' if ctx['generation_manifest_exists'] else '缺失，已由 G2 补建'}",
        "",
        "## 3. accepted / rejected 结果",
        "",
        "### accepted",
        "",
        df_to_markdown(pd.DataFrame(accepted_rows)[["synthetic_raw_id", "synthetic_final_id", "source_case_id", "qc_decision"]] if accepted_rows else pd.DataFrame()),
        "",
        "### rejected",
        "",
        df_to_markdown(pd.DataFrame(rejected_rows)[["synthetic_raw_id", "synthetic_final_id", "source_case_id", "qc_reject_reason"]] if rejected_rows else pd.DataFrame()),
        "",
        "## 4. 主要问题",
        "",
        df_to_markdown(qc_df[["synthetic_raw_id", "qc_status", "qc_reject_reason", "manual_review_reason"]] if not qc_df.empty else pd.DataFrame()),
        "",
        "## 5. 输出文件",
        "",
    ]
    for path in [manifest_path, candidate_path, accepted_path, rejected_path, mapping_path, qc_path, diffusion_path, review_path, batch_summary_path]:
        lines.append(f"- `{path}`")
    lines.extend([
        "",
        "## 6. 结论",
        "",
        "1. G2 已经可以从 G1 legacy raw output 里自动恢复 source、label_kind、run 信息、suffix scheme，并补建 synthetic manifest。",
        "2. G2 会额外生成 `synthetic_normalized_mapping_{run_id}.csv`，逐模态记录 raw legacy/native 文件到 2026 标准文件名和 nnU-Net 目标文件名的映射。",
        "3. 通过的样本会进入 accepted manifest，未通过的样本会进入 rejected manifest，人工复查项会单独落表。",
        "4. 真实验证 fold 和官方 validation 仍然不能作为 synthetic source。",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_progress_report(
        results_root,
        progress_report_path,
        summary,
        [manifest_path, candidate_path, accepted_path, rejected_path, mapping_path, qc_path, diffusion_path, review_path, batch_summary_path, report_path],
        [
            ("synthetic_generation_manifest", [manifest_path]),
            ("synthetic_candidate_manifest", [candidate_path]),
            ("synthetic_accepted_manifest", [accepted_path]),
            ("synthetic_rejected_manifest", [rejected_path]),
            ("synthetic_normalized_mapping", [mapping_path]),
            ("qc_metrics", [qc_path]),
            ("diffusion_quality_metrics", [diffusion_path]),
            ("qc_case_review", [review_path]),
            ("qc_batch_summary", [batch_summary_path]),
            ("quality_report", [report_path]),
        ],
    )
    outputs.extend([manifest_path, candidate_path, accepted_path, rejected_path, mapping_path, qc_path, diffusion_path, review_path, batch_summary_path, report_path, progress_report_path])
    return outputs


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
    return ensure_columns(pd.DataFrame(rows), REAL_TRAIN_EMPTY_COLUMNS)


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
    return ensure_columns(pd.DataFrame(rows), [
        "case_id", "case_dir", "t1n_path", "t1c_path", "t2w_path", "t2f_path",
        "has_t1n", "has_t1c", "has_t2w", "has_t2f", "shape", "spacing", "affine_hash",
        "image_dtypes", "basic_qc_pass", "basic_qc_reason", "allowed_as_synthetic_source",
    ])


def apply_corrected_labels(raw_df: pd.DataFrame, corrected_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw_df.empty:
        overlay_cols = [
            "case_id", "raw_seg_path", "corrected_seg_path", "raw_unique_labels", "corrected_unique_labels",
            "raw_shape", "corrected_shape", "raw_spacing", "corrected_spacing",
            "raw_affine_hash", "corrected_affine_hash", "applied", "apply_reason", "notes",
        ]
        final_df = ensure_columns(raw_df.copy(), [
            "raw_seg_path", "effective_seg_path", "label_source", "has_corrected_label",
            "has_illegal_label_after_overlay", "illegal_label_values_after_overlay", "final_qc_pass",
            "final_qc_reason", "labels_present_after_overlay",
        ])
        return pd.DataFrame(columns=overlay_cols), final_df
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
    synthetic_manifest_header = [
        "synthetic_raw_id", "synthetic_final_id", "nnunet_case_id", "source_case_id", "source_split",
        "label_kind", "label_index", "label_source_case_id", "label_component_id", "label_generator_checkpoint",
        "generation_run_id", "generator_name", "generator_checkpoint_t1n", "generator_checkpoint_t1c",
        "generator_checkpoint_t2w", "generator_checkpoint_t2f", "generator_io", "label_channels", "rc_policy",
        "noise_type", "sampling_method", "sampling_steps", "eta", "seed", "source_csv_path", "source_csv_version",
        "raw_case_dir", "normalized_case_dir", "output_suffix_scheme", "suffix_conversion_action",
        "raw_t1n_path", "raw_t1c_path", "raw_t2w_path", "raw_t2f_path", "raw_seg_path",
        "normalized_t1n_path", "normalized_t1c_path", "normalized_t2w_path", "normalized_t2f_path",
        "normalized_seg_path", "nnunet_t1n_target_path", "nnunet_t1c_target_path", "nnunet_t2w_target_path",
        "nnunet_t2f_target_path", "nnunet_seg_target_path", "insert_center_x", "insert_center_y",
        "insert_center_z", "roi_x_min", "roi_x_max", "roi_y_min", "roi_y_max", "roi_z_min", "roi_z_max",
        "source_shape_x", "source_shape_y", "source_shape_z", "output_shape_x", "output_shape_y",
        "output_shape_z", "status", "error_type", "error_message", "qc_status", "qc_reject_reason",
        "accepted_for_training", "accepted_for_ablation_only", "needs_regeneration",
    ]
    with (dirs["manifests"] / "synthetic_generation_manifest_template_g1.csv").open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow(synthetic_manifest_header)

    normalized_mapping_header = [
        "synthetic_raw_id", "synthetic_final_id", "nnunet_case_id", "source_case_id", "generation_run_id",
        "modality", "nnunet_channel", "raw_source_path", "normalized_target_path", "nnunet_target_path",
        "output_suffix_scheme", "suffix_conversion_action", "qc_decision", "accepted_for_training",
        "accepted_for_ablation_only", "needs_regeneration",
    ]
    with (dirs["manifests"] / "synthetic_normalized_mapping_template.csv").open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow(normalized_mapping_header)

    qc_v2_header = [
        "synthetic_raw_id", "synthetic_final_id", "nnunet_case_id", "source_case_id", "label_kind", "label_index",
        "generation_run_id", "generator_name", "generator_checkpoint_t1n", "generator_checkpoint_t1c",
        "generator_checkpoint_t2w", "generator_checkpoint_t2f", "label_generator_checkpoint", "generator_io",
        "label_channels", "rc_policy", "noise_type", "sampling_method", "sampling_steps", "eta", "seed",
        "raw_case_dir", "normalized_case_dir", "output_suffix_scheme", "suffix_conversion_action",
        "config_exists", "manifest_exists", "log_exists", "source_csv_version", "has_t1n", "has_t1c",
        "has_t2w", "has_t2f", "has_seg", "filename_consistent", "nifti_readable", "shape_t1n", "shape_t1c",
        "shape_t2w", "shape_t2f", "shape_seg", "spacing_t1n", "spacing_t1c", "spacing_t2w", "spacing_t2f",
        "spacing_seg", "affine_hash_t1n", "affine_hash_t1c", "affine_hash_t2w", "affine_hash_t2f",
        "affine_hash_seg", "shape_consistent", "spacing_consistent", "affine_consistent", "orientation_consistent",
        "source_shape_match", "has_nan_or_inf", "image_is_constant", "label_is_integer", "label_values",
        "label_values_valid", "empty_mask", "allow_empty_mask", "source_in_real_train_manifest",
        "source_final_qc_pass", "source_usable_for_gligan96", "source_in_fixed_val_fold",
        "source_from_official_validation", "source_is_allowed", "case_id_reuses_real_id", "validation_leakage",
        "roi_bbox_available", "insert_center_x", "insert_center_y", "insert_center_z", "roi_x_min", "roi_x_max",
        "roi_y_min", "roi_y_max", "roi_z_min", "roi_z_max", "roi_inside_image", "nonroi_change_ratio",
        "source_existing_lesion_overlap", "brain_mask_overlap_ratio", "lesion_count", "tiny_lesion_count",
        "small_lesion_count", "large_lesion_count", "min_lesion_volume_mm3", "p50_lesion_volume_mm3",
        "max_lesion_volume_mm3", "tiny_lesion_ratio", "label_combination", "has_rc", "rc_source_allowed",
        "bbox_inside_image", "lesion_inside_brain_ok", "et_t1c_contrast_ratio", "snfh_t2f_contrast_ratio",
        "snfh_t2w_contrast_ratio", "cross_modality_roi_corr", "label_modality_alignment_score",
        "roi_boundary_mae", "roi_boundary_gradient_jump", "intensity_drift_p50", "artifact_block_score",
        "artifact_suspected", "teacher_model", "teacher_dice_label_1", "teacher_dice_label_2",
        "teacher_dice_label_3", "teacher_dice_label_4", "teacher_lesion_count_diff",
        "teacher_missing_large_lesion_count", "teacher_extra_large_lesion_count", "manual_review_required",
        "manual_review_priority", "manual_review_reason", "hard_reject", "hard_reject_reason", "quality_grade",
        "qc_decision", "accepted_for_training", "accepted_for_ablation_only", "needs_regeneration",
        "regeneration_reason",
    ]
    with (dirs["qc"] / "qc_metrics_template_v2.csv").open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow(qc_v2_header)

    official_leaderboard_header = [
        "submission_id", "date", "participant_team",
        "lesionwise_dsc_mean_et", "lesionwise_nsd_mean_et",
        "lesionwise_dsc_mean_rc", "lesionwise_nsd_mean_rc",
        "lesionwise_dsc_mean_tc", "lesionwise_nsd_mean_tc",
        "lesionwise_dsc_mean_wt", "lesionwise_nsd_mean_wt",
        "small_instance_tp_et", "small_instance_fn_et", "small_instance_fp_et", "small_instance_f1_et",
        "small_instance_tp_tc", "small_instance_fn_tc", "small_instance_fp_tc", "small_instance_f1_tc",
        "small_instance_tp_wt", "small_instance_fn_wt", "small_instance_fp_wt", "small_instance_f1_wt",
        "small_instance_tp_rc", "small_instance_fn_rc", "small_instance_fp_rc", "small_instance_f1_rc",
    ]
    with (dirs["qc"] / "official_leaderboard_metrics_template.csv").open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow(official_leaderboard_header)

    diffusion_header = [
        "synthetic_raw_id", "synthetic_final_id", "source_case_id", "generation_run_id", "generator_name",
        "generator_checkpoint", "modality", "label_kind", "label_channels", "rc_policy", "noise_type",
        "sampling_method", "sampling_steps", "eta", "seed", "roi_bbox_available", "roi_x_min", "roi_x_max",
        "roi_y_min", "roi_y_max", "roi_z_min", "roi_z_max", "roi_volume_voxels", "lesion_voxels_in_roi",
        "lesion_inside_roi_ratio", "nonroi_change_ratio", "brain_mask_overlap_ratio", "roi_boundary_mae",
        "roi_boundary_gradient_jump", "roi_boundary_p95_jump", "z_continuity_score", "z_area_smoothness",
        "z_intensity_smoothness", "intensity_drift_p1", "intensity_drift_p50", "intensity_drift_p99",
        "artifact_block_score", "artifact_ring_score", "artifact_noise_score", "et_t1c_contrast_ratio",
        "snfh_t2f_contrast_ratio", "snfh_t2w_contrast_ratio", "rc_profile_score", "cross_modality_roi_corr",
        "label_modality_alignment_score", "source_synth_roi_ssim", "label_source_synth_roi_ssim",
        "synth_synth_ms_ssim", "nearest_real_roi_feature_distance", "duplicate_hash_hit", "feature_extractor",
        "feature_fid_medical", "feature_mmd_medical", "teacher_model", "teacher_dice_mean",
        "teacher_lesion_count_diff", "manual_visual_score", "quality_grade", "diffusion_quality_decision",
        "diffusion_quality_reason",
    ]
    with (dirs["qc"] / "diffusion_quality_metrics_template.csv").open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow(diffusion_header)

    review_header = [
        "case_id", "source_case_id", "generation_run_id", "review_priority", "review_reason", "viewed_t1c_et",
        "viewed_t2f_snfh", "viewed_all_modalities", "viewed_three_planes", "roi_boundary_ok", "z_continuity_ok",
        "lesion_inside_brain_ok", "label_anatomy_ok", "rc_context_ok", "artifact_notes", "review_decision",
        "reviewer", "review_date",
    ]
    with (dirs["qc"] / "qc_case_review_template.csv").open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow(review_header)

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

主指标必须对齐官方 leaderboard：ET/RC/TC/WT 的 lesionwise DSC/NSD，以及 ET/TC/WT/RC 的 small-instance TP/FN/FP/F1。HD95、AUC、NETC/SNFH/ET/RC 单类均值只能作为内部辅助分析。

| 指标组 | 字段 |
|---|---|
| lesionwise segmentation | `lesionwise_dsc_mean_et/rc/tc/wt`, `lesionwise_nsd_mean_et/rc/tc/wt` |
| small-instance detection | `small_instance_tp/fn/fp/f1_et` |
| small-instance detection | `small_instance_tp/fn/fp/f1_tc` |
| small-instance detection | `small_instance_tp/fn/fp/f1_wt` |
| small-instance detection | `small_instance_tp/fn/fp/f1_rc` |
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
    parser.add_argument("--synthetic-run-root", default="", help="Optional G1 synthetic run directory to intake.")
    parser.add_argument("--synthetic-run-id", default="", help="Optional run id override for synthetic intake.")
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
        dirs["manifests"] / "synthetic_generation_manifest_template_g1.csv",
        dirs["manifests"] / "synthetic_normalized_mapping_template.csv",
        dirs["qc"] / "qc_metrics_template_v2.csv",
        dirs["qc"] / "official_leaderboard_metrics_template.csv",
        dirs["qc"] / "diffusion_quality_metrics_template.csv",
        dirs["qc"] / "qc_case_review_template.csv",
        dirs["qc"] / "G2_synthetic_data_QC报告模板_v2.md",
        dirs["reports"] / "ablation_plan_template.md",
    ])
    write_execution_summary(dirs, outputs)
    outputs.append(dirs["reports"] / "g2_pretraining_execution_summary.md")

    baseline_summary = {
        "generation_run_id": "real_baseline",
        "case_count": int(len(raw_df)),
        "accepted_count": int((final_df["final_qc_pass"] == True).sum()),  # noqa: E712
        "ablation_only_count": 0,
        "needs_regeneration_count": int((final_df["final_qc_pass"] != True).sum()),  # noqa: E712
        "rejected_count": int((final_df["final_qc_pass"] != True).sum()),  # noqa: E712
        "legacy_suffix_count": 0,
        "native_suffix_count": 0,
        "mixed_suffix_count": 0,
    }
    progress_report_path = dirs["reports"] / "G2_progress_report.md"
    write_progress_report(results_root, progress_report_path, baseline_summary, outputs)
    outputs.append(progress_report_path)

    if args.synthetic_run_root:
        synthetic_outputs = ingest_synthetic_run(Path(args.synthetic_run_root), results_root, args, dirs)
        outputs.extend(synthetic_outputs)
        if synthetic_outputs:
            progress_report_path = dirs["reports"] / "G2_progress_report.md"
            run_dir_name = Path(args.synthetic_run_root).name
            intake_index = [
                ("synthetic_generation_manifest", [dirs["manifests"] / f"synthetic_generation_manifest_{run_dir_name}.csv"]),
                ("synthetic_candidate_manifest", [dirs["manifests"] / f"synthetic_candidate_manifest_{run_dir_name}.csv"]),
                ("synthetic_accepted_manifest", [dirs["manifests"] / f"synthetic_accepted_manifest_{run_dir_name}.csv"]),
                ("synthetic_rejected_manifest", [dirs["manifests"] / f"synthetic_rejected_manifest_{run_dir_name}.csv"]),
                ("synthetic_normalized_mapping", [dirs["manifests"] / f"synthetic_normalized_mapping_{run_dir_name}.csv"]),
                ("qc_metrics", [dirs["qc"] / f"qc_metrics_{run_dir_name}.csv"]),
                ("diffusion_quality_metrics", [dirs["qc"] / f"diffusion_quality_metrics_{run_dir_name}.csv"]),
                ("qc_case_review", [dirs["qc"] / f"qc_case_review_{run_dir_name}.csv"]),
                ("qc_batch_summary", [dirs["qc"] / f"qc_batch_summary_{run_dir_name}.json"]),
                ("quality_report", [dirs["reports"] / f"G2_synthetic_data_quality_report_{run_dir_name}.md"]),
            ]
            write_progress_report(results_root, progress_report_path, baseline_summary, outputs, intake_index)

    print(json.dumps({
        "train_cases": len(raw_df),
        "validation_cases": len(val_df),
        "final_qc_pass": int((final_df["final_qc_pass"] == True).sum()),  # noqa: E712
        "final_qc_fail": int((final_df["final_qc_pass"] != True).sum()),  # noqa: E712
        "lesions": len(lesion_df),
        "gligan_usable_cases": int((gligan_df["usable_for_gligan96"] == True).sum()) if not gligan_df.empty else 0,  # noqa: E712
        "synthetic_run_root": args.synthetic_run_root,
        "outputs": [str(p) for p in outputs],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
