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
from skimage.metrics import structural_similarity

from g2_create_train_val_test_split import (
    create_train_val_test_split,
    filter_split,
    patient_group,
    write_split_outputs,
)


LABELS = {0: "background", 1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
MODALITIES = {
    "t1n": "t1n",
    "t1c": "t1c",
    "t2w": "t2w",
    "t2f": "t2f",
}
MET_KEYS = {
    "scan_t1ce": "t1c",
    "scan_t2": "t2w",
    "scan_flair": "t2f",
    "scan_t1": "t1n",
}
RUN_DATE = datetime.now().strftime("%Y-%m-%d")
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
PROJECT_ROOT_NAME = "ECNU_EYU_data"


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "work_space" / "G1").exists() and (parent / "work_space" / "G2").exists():
            return parent
    raise RuntimeError(f"Could not locate ECNU_EYU_data project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
DEFAULT_DATA_ROOT = os.environ.get("G2_DATA_ROOT") or str(PROJECT_ROOT / "work_space" / "G1" / "data" / "raw")
G1_DATA_ROOT = PROJECT_ROOT / "work_space" / "G1" / "data"
G1_RAW_ROOT = G1_DATA_ROOT / "raw"
G1_TRAIN_ROOT = G1_RAW_ROOT / "MICCAI-LH-BraTS2025-MET-Challenge-Training"
G1_VALIDATION_ROOT = G1_RAW_ROOT / "Validation"
G1_CORRECTED_ROOT = G1_RAW_ROOT / "MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels"
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


def boolish(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_workspace_path(path_str: str | Path | None, anchor: Path | None = None) -> Path:
    if path_str is None:
        raise FileNotFoundError("empty path")
    path = Path(path_str)
    if path.is_absolute():
        return path
    bases = []
    for base in [anchor, PROJECT_ROOT, G1_RAW_ROOT, G1_TRAIN_ROOT, G1_VALIDATION_ROOT, G1_CORRECTED_ROOT, G1_DATA_ROOT, G1_DATA_ROOT / "input", G1_DATA_ROOT / "input_inference"]:
        if base is not None and base not in bases:
            bases.append(base)
    candidate = (PROJECT_ROOT / path).resolve()
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return candidate


def display_path(path: Path | str | None, anchor: Path | None = None) -> str:
    if path is None:
        return ""
    p = Path(path)
    if anchor is not None:
        try:
            return p.relative_to(anchor).as_posix()
        except Exception:  # noqa: BLE001
            pass
    parts = p.parts
    if PROJECT_ROOT_NAME in parts:
        idx = parts.index(PROJECT_ROOT_NAME)
        return Path(*parts[idx:]).as_posix()
    return p.as_posix()


def display_results_path(path: Path | str | None, results_root: Path | None = None) -> str:
    if path is None:
        return ""
    p = Path(path)
    if results_root is not None:
        try:
            return p.relative_to(results_root).as_posix()
        except Exception:  # noqa: BLE001
            pass
    return display_path(p, PROJECT_ROOT)


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
    (dirs["nnunet_raw"] / "imagesTs").mkdir(parents=True, exist_ok=True)
    (dirs["nnunet_raw"] / "labelsTs").mkdir(parents=True, exist_ok=True)
    return dirs


def write_readme_files(results_root: Path, dirs: dict[str, Path]) -> None:
    readmes = {
        results_root / "README.md": "# G2 Results\n\n本目录只保存轻量 mapping、patient-group split、QC 规则、模板和报告，不保存正式 NIfTI。V2 必须先 composition，V3 使用 completion 专用入口；技术通过但未审批的病例保持 pending。\n",
        dirs["nnunet_raw_root"] / "README.md": "# nnunet_raw\n\n这里是 nnU-Net 原始数据的轻量入口。当前仓库只放占位说明、dataset.json 和路径契约，不放正式大体积影像。`Dataset260_BraTS2026_MET_RealOnly/` 记录 real-only 基线；正式 real+synth 由 `../code/g2_materialize_nnunet_dataset.py` 在训练机物化。\n",
        dirs["manifests"] / "README.md": "# Manifests\n\n`nnunet_case_mapping_master.csv` 保存全部病例身份，`nnunet_case_mapping_realonly.csv` 只保存真实 T2W，`g1_v2_source_manifest.csv` 只放行 master train source。正式 run 输出 rejected、pending、accepted-training 和 accepted-evaluation。\n",
        dirs["stats"] / "README.md": "# Stats\n\n保存真实 label/lesion 分布、synthetic 目标分布、batch 级统计摘要，以及后续抽样分析所需的小型数表。\n",
        dirs["qc"] / "README.md": "# QC\n\n保存技术硬门、质量复核、release 审批和官方训练价值验收产物。未计算指标不写常量，未审批病例不得进入训练。\n",
        dirs["splits"] / "README.md": "# Splits\n\n`splits_master_train_val_test.json` 是全部 1295 例的 patient-group master split；`splits_final_train_val_test.json` 是真实 T2W real-only 派生 split。\n",
        dirs["reports"] / "README.md": "# Reports\n\n保存路径检查、数据 QC、执行总结、进度报告、消融模板和团队沟通文档源稿。临时 smoke run 质量报告不保留。\n",
        dirs["nnunet_raw"] / "README.md": "# Dataset260_BraTS2026_MET_RealOnly\n\n本目录当前只保存 `dataset.json` 和映射说明，不复制或软链接全量 NIfTI。需要正式训练时，由 S1/S2 根据 `manifests/nnunet_case_mapping_realonly.csv` 在训练机器上物化数据集并运行 nnU-Net 预处理。synthetic accepted 结果另起 dataset id，不混进这个 real-only 占位目录。\n",
    }
    for path, text in readmes.items():
        if not path.exists():
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


def index_case_records(
    records: list[dict[str, object]],
) -> tuple[dict[str, dict[str, object]], set[str]]:
    lookup: dict[str, dict[str, object]] = {}
    duplicates: set[str] = set()
    for record in records:
        raw_id = str(
            record.get("synthetic_raw_id")
            or record.get("case_id")
            or record.get("raw_case_id")
            or record.get("case_name")
            or ""
        ).strip()
        if not raw_id:
            continue
        if raw_id in lookup:
            duplicates.add(raw_id)
            continue
        lookup[raw_id] = record
    return lookup, duplicates


def load_fake_t2w_case_ids(results_root: Path) -> set[str]:
    fake_path = results_root / "qc" / "official_fake_t2w_cases_by_gzip_header_2026-06-15.csv"
    fake_df = read_csv_if_exists(fake_path)
    if fake_df.empty or "case_id" not in fake_df.columns:
        return set()
    return set(fake_df["case_id"].astype(str))


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
        if re.match(r"^BraTS-MET-\d{5}-\d{3}$", name):
            return {
                "parsed": True,
                "source_case_id": name,
                "label_kind": "full_generation",
                "label_index": 0,
            }
        return {"parsed": False, "source_case_id": "", "label_kind": "", "label_index": ""}
    return {
        "parsed": True,
        "source_case_id": match.group("source_case_id"),
        "label_kind": match.group("label_kind"),
        "label_index": int(match.group("label_index")),
    }


def apply_generation_mode_override(parsed: dict[str, object], generation_mode: str) -> dict[str, object]:
    if generation_mode not in {"completion", "full_generation"}:
        return parsed
    if not parsed.get("source_case_id"):
        return parsed
    parsed = dict(parsed)
    parsed["label_kind"] = generation_mode
    parsed["label_index"] = int(parsed.get("label_index") or 0)
    parsed["parsed"] = True
    return parsed


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
    case_prefixes: set[str] = set()
    for path in case_dir.glob("*.nii.gz"):
        match = re.match(r"^(BraTS-MET-\d{5}-\d{3})", path.name)
        if match:
            case_prefixes.add(match.group(1))
    if len(case_prefixes) > 1:
        raise ValueError(
            f"flat V2 output contains multiple cases in {case_dir}; "
            "run g2_v2_compose_augmentation.py before synthetic intake"
        )
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


def nnunet_synthetic_paths(
    nnunet_case_id: str,
    split: str = "train",
    dataset_name: str = "Dataset261_BraTS2026_MET_RealSynth",
) -> dict[str, str]:
    if not nnunet_case_id or split not in {"train", "val", "test"}:
        return {key: "" for key in ["t1n", "t1c", "t2w", "t2f", "seg"]}
    root = Path("nnunet_raw") / dataset_name
    image_dir = "imagesTs" if split == "test" else "imagesTr"
    label_dir = "labelsTs" if split == "test" else "labelsTr"
    paths = {
        "t1n": root / image_dir / f"{nnunet_case_id}_0000.nii.gz",
        "t1c": root / image_dir / f"{nnunet_case_id}_0001.nii.gz",
        "t2w": root / image_dir / f"{nnunet_case_id}_0002.nii.gz",
        "t2f": root / image_dir / f"{nnunet_case_id}_0003.nii.gz",
        "seg": root / label_dir / f"{nnunet_case_id}.nii.gz",
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
            "accepted_for_evaluation": row.get("accepted_for_evaluation", False),
            "pending_review": row.get("pending_review", False),
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
        return "legacy_met"
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
    g1_df = read_csv_if_exists(manifests_dir / "g1_v2_source_manifest.csv")
    mapping_path = manifests_dir / "nnunet_case_mapping_master.csv"
    if not mapping_path.exists():
        mapping_path = manifests_dir / "nnunet_case_mapping_realonly.csv"
    mapping_df = read_csv_if_exists(mapping_path)
    fake_t2w_case_ids = load_fake_t2w_case_ids(results_root)
    split_path = splits_dir / "splits_master_train_val_test.json"
    if not split_path.exists():
        split_path = splits_dir / "splits_final_train_val_test.json"
    split_data = []
    if split_path.exists():
        try:
            split_data = json.loads(split_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            split_data = []
    split_val_ids: set[str] = set()
    split_by_nnunet: dict[str, str] = {}
    if split_data and isinstance(split_data, list) and isinstance(split_data[0], dict):
        split_val_ids = set(split_data[0].get("val", []))
        split_val_ids |= set(split_data[0].get("test", []))
        for split_name in ("train", "val", "test"):
            for nnunet_id in split_data[0].get(split_name, []):
                split_by_nnunet[str(nnunet_id)] = split_name
    source_to_nn = {}
    if not mapping_df.empty and "source_case_id" in mapping_df.columns and "nnunet_case_id" in mapping_df.columns:
        source_to_nn = dict(zip(mapping_df["source_case_id"].astype(str), mapping_df["nnunet_case_id"].astype(str)))
    train_lookup = train_df.set_index("case_id").to_dict(orient="index") if not train_df.empty and "case_id" in train_df.columns else {}
    val_lookup = val_df.set_index("case_id").to_dict(orient="index") if not val_df.empty and "case_id" in val_df.columns else {}
    if not g1_df.empty and "source_case_id" in g1_df.columns:
        g1_lookup = g1_df.set_index("source_case_id").to_dict(orient="index")
    elif not g1_df.empty and "case_id" in g1_df.columns:
        g1_lookup = g1_df.set_index("case_id").to_dict(orient="index")
    else:
        g1_lookup = {}
    return {
        "train_df": train_df,
        "val_df": val_df,
        "g1_df": g1_df,
        "mapping_df": mapping_df,
        "split_data": split_data,
        "split_val_ids": split_val_ids,
        "split_by_nnunet": split_by_nnunet,
        "source_to_nn": source_to_nn,
        "train_lookup": train_lookup,
        "val_lookup": val_lookup,
        "g1_lookup": g1_lookup,
        "fake_t2w_case_ids": fake_t2w_case_ids,
    }


def build_source_status(source_case_id: str, ctx: dict[str, object], label_kind: str = "") -> dict[str, object]:
    train_lookup = ctx["train_lookup"]  # type: ignore[assignment]
    val_lookup = ctx["val_lookup"]  # type: ignore[assignment]
    g1_lookup = ctx["g1_lookup"]  # type: ignore[assignment]
    split_val_ids = ctx["split_val_ids"]  # type: ignore[assignment]
    source_to_nn = ctx["source_to_nn"]  # type: ignore[assignment]
    split_by_nnunet = ctx.get("split_by_nnunet", {})  # type: ignore[assignment]
    fake_t2w_case_ids = ctx.get("fake_t2w_case_ids", set())  # type: ignore[assignment]

    train_row = train_lookup.get(source_case_id, {})
    val_row = val_lookup.get(source_case_id, {})
    g1_row = g1_lookup.get(source_case_id, {})
    nn_id = source_to_nn.get(source_case_id, "")
    source_split = split_by_nnunet.get(nn_id, "unknown") if nn_id else "unknown"
    in_fixed_val_fold = source_split in {"val", "test"} or bool(nn_id and nn_id in split_val_ids)
    final_qc_pass = bool(train_row.get("final_qc_pass", val_row.get("final_qc_pass", False)))
    allowed_as_synthetic_source = boolish(
        g1_row.get("allowed_as_v2_source", g1_row.get("allowed_as_synthetic_source", False))
    )
    completion_mode = label_kind == "completion"
    source_is_fake_t2w_case = source_case_id in fake_t2w_case_ids
    if completion_mode:
        source_is_allowed = bool(train_row or val_row) and final_qc_pass
        source_allowed_for_training = bool(train_row) and final_qc_pass and source_split == "train"
    else:
        source_is_allowed = bool(train_row) and final_qc_pass and source_split == "train" and allowed_as_synthetic_source
        source_allowed_for_training = source_is_allowed
    return {
        "source_row": train_row,
        "val_row": val_row,
        "g1_row": g1_row,
        "nnunet_case_id": nn_id,
        "source_in_real_train_manifest": bool(train_row),
        "source_final_qc_pass": final_qc_pass,
        "source_allowed_for_v2": allowed_as_synthetic_source,
        "source_allowed_for_training": source_allowed_for_training,
        "source_is_fake_t2w_case": source_is_fake_t2w_case,
        "source_completion_mode": completion_mode,
        "source_in_fixed_val_fold": in_fixed_val_fold,
        "source_from_official_validation": bool(val_row),
        "source_is_allowed": source_is_allowed,
        "source_split": source_split if source_split != "unknown" else ("official_validation" if val_row else "unknown"),
    }


def stable_synthetic_ids(generation_run_id: str, synthetic_raw_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{generation_run_id}::{synthetic_raw_id}".encode("utf-8")).hexdigest()[:12].upper()
    return f"SYN-MET-{digest}", f"SYNMET_{digest}"


def normalized_ssim(a: np.ndarray, b: np.ndarray) -> float | str:
    if a.shape != b.shape or a.size < 8:
        return ""
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    data_min = float(min(a.min(), b.min()))
    data_max = float(max(a.max(), b.max()))
    data_range = data_max - data_min
    if not np.isfinite(data_range) or data_range <= 0:
        return ""
    min_dim = min(a.shape)
    if a.ndim == 3 and min_dim >= 7:
        return float(structural_similarity(a, b, data_range=data_range))
    return float((2 * np.mean(a) * np.mean(b) + 1e-6) / (np.mean(a) ** 2 + np.mean(b) ** 2 + 1e-6))


def z_smoothness_scores(image: np.ndarray, lesion_mask: np.ndarray) -> tuple[float | str, float | str, float | str]:
    active = np.where(np.any(lesion_mask, axis=(0, 1)))[0]
    if active.size < 2:
        return "", "", ""
    areas = np.asarray([lesion_mask[:, :, index].sum() for index in active], dtype=np.float32)
    area_denom = max(float(np.mean(areas)), 1.0)
    area_score = float(np.clip(1.0 - np.mean(np.abs(np.diff(areas))) / area_denom, 0.0, 1.0))
    means = []
    for index in active:
        mask = lesion_mask[:, :, index]
        means.append(float(np.mean(image[:, :, index][mask])))
    means_array = np.asarray(means, dtype=np.float32)
    intensity_scale = max(float(np.percentile(image[lesion_mask], 99) - np.percentile(image[lesion_mask], 1)), 1e-6)
    intensity_score = float(np.clip(1.0 - np.mean(np.abs(np.diff(means_array))) / intensity_scale, 0.0, 1.0))
    return float((area_score + intensity_score) / 2.0), area_score, intensity_score


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


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gradients = np.gradient(image.astype(np.float32, copy=False))
    squared = np.zeros(image.shape, dtype=np.float32)
    for gradient in gradients:
        squared += np.square(gradient, dtype=np.float32)
    return np.sqrt(squared, dtype=np.float32)


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
    parsed = apply_generation_mode_override(
        parsed,
        str(run_ctx.get("generation_mode_override", "auto")),
    )
    synthetic_raw_id = case_name
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
    source_info = build_source_status(source_case_id, run_ctx, label_kind=label_kind) if source_case_id else {
        "source_row": {},
        "val_row": {},
        "g1_row": {},
        "nnunet_case_id": "",
        "source_in_real_train_manifest": False,
        "source_final_qc_pass": False,
        "source_allowed_for_v2": False,
        "source_allowed_for_training": False,
        "source_is_fake_t2w_case": False,
        "source_completion_mode": label_kind == "completion",
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

    files = synthetic_modality_files(case_dir)
    if files.get("seg") is None:
        source_seg_candidates = [
            str(source_row.get("effective_seg_path", "")),
            str(source_row.get("seg_source_path", "")),
            str(source_row.get("raw_seg_path", "")),
        ]
        for candidate in source_seg_candidates:
            if not candidate:
                continue
            candidate_path = parse_workspace_path(candidate, PROJECT_ROOT)
            if candidate_path.exists():
                files["seg"] = candidate_path
                break
    output_scheme = detect_output_suffix_scheme(files)
    completion_mode = bool(source_info.get("source_completion_mode")) or label_kind == "completion"
    if completion_mode and source_case_id:
        synthetic_final_id = source_case_id
        nnunet_case_id = str(source_info.get("nnunet_case_id", ""))
    else:
        synthetic_final_id, nnunet_case_id = stable_synthetic_ids(generation_run_id, synthetic_raw_id)

    config_exists = bool(run_ctx.get("generation_config_exists", False))
    manifest_exists = bool(run_ctx.get("generation_manifest_exists", False))
    log_exists = bool(run_ctx.get("generation_log_exists", False))
    manifest_lookup = run_ctx.get("generation_manifest_lookup", {})
    log_lookup = run_ctx.get("generation_log_lookup", {})
    manifest_record = manifest_lookup.get(synthetic_raw_id) if isinstance(manifest_lookup, dict) else None
    log_record = log_lookup.get(synthetic_raw_id) if isinstance(log_lookup, dict) else None
    case_metadata_missing: list[str] = []
    if not isinstance(manifest_record, dict):
        case_metadata_missing.append("case_manifest_record")
    else:
        if str(manifest_record.get("status", "")).strip().lower() != "success":
            case_metadata_missing.append("case_manifest_status_success")
        if str(manifest_record.get("source_case_id", "")).strip() != source_case_id:
            case_metadata_missing.append("case_manifest_source_case_id")
    if not isinstance(log_record, dict):
        case_metadata_missing.append("case_log_record")
    else:
        if str(log_record.get("status", "")).strip().lower() != "success":
            case_metadata_missing.append("case_log_status_success")
        if str(log_record.get("source_case_id", "")).strip() != source_case_id:
            case_metadata_missing.append("case_log_source_case_id")
        if str(log_record.get("generation_run_id", "")).strip() != generation_run_id:
            case_metadata_missing.append("case_log_generation_run_id")
        if str(log_record.get("seed", "")).strip() != str(run_ctx.get("seed", "")).strip():
            case_metadata_missing.append("case_log_seed")
    metadata_missing_fields = list(run_ctx.get("metadata_missing_fields", []))
    metadata_missing_fields.extend(case_metadata_missing)
    metadata_missing_fields = list(dict.fromkeys(metadata_missing_fields))
    raw_case_dir = display_path(case_dir, PROJECT_ROOT)
    normalized_case_path = Path(run_ctx.get("normalized_root", case_dir.parent)) / synthetic_final_id
    normalized_case_dir = display_path(normalized_case_path, PROJECT_ROOT)
    normalized_paths = normalized_synthetic_paths(normalized_case_dir, synthetic_final_id)
    target_split = str(source_info.get("source_split", "")) if completion_mode else "train"
    nnunet_paths = nnunet_synthetic_paths(nnunet_case_id, split=target_split)
    if output_scheme == "legacy_met":
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
        "vae_checkpoint": str(run_ctx.get("vae_checkpoint", "")),
        "encdec_checkpoint": str(run_ctx.get("encdec_checkpoint", "")),
        "bbdm_checkpoint": str(run_ctx.get("bbdm_checkpoint", "")),
        "bbdm_s": run_ctx.get("bbdm_s", ""),
        "validation_run": str(run_ctx.get("validation_run", "")),
        "generator_io": str(run_ctx.get("generator_io", "")),
        "label_channels": int(run_ctx.get("label_channels", 0) or 0),
        "rc_policy": str(run_ctx.get("rc_policy", "")),
        "noise_type": str(run_ctx.get("noise_type", "")),
        "sampling_method": str(run_ctx.get("sampling_method", "")),
        "sampling_steps": run_ctx.get("sampling_steps", ""),
        "eta": run_ctx.get("eta", ""),
        "crop_size": run_ctx.get("crop_size", ""),
        "seed": run_ctx.get("seed", ""),
        "source_csv_path": display_results_path(
            run_ctx.get("source_csv_path", ""),
            Path(run_ctx.get("results_root", DEFAULT_RESULTS_ROOT)),
        ),
        "source_csv_version": str(run_ctx.get("source_csv_version", "")),
        "raw_case_dir": raw_case_dir,
        "normalized_case_dir": normalized_case_dir,
        "output_suffix_scheme": output_scheme,
        "suffix_conversion_action": suffix_conversion_action,
        "config_exists": config_exists,
        "manifest_exists": manifest_exists,
        "log_exists": log_exists,
        "manifest_case_record_exists": isinstance(manifest_record, dict),
        "log_case_record_exists": isinstance(log_record, dict),
        "metadata_complete": bool(run_ctx.get("metadata_complete", False) and not case_metadata_missing),
        "metadata_missing_fields": ";".join(metadata_missing_fields),
        "generation_mode": str(run_ctx.get("generator_io", "")),
        "source_in_real_train_manifest": source_info["source_in_real_train_manifest"],
        "source_final_qc_pass": source_info["source_final_qc_pass"],
        "source_allowed_for_v2": source_info.get("source_allowed_for_v2", False),
        "source_allowed_for_training": source_info["source_allowed_for_training"],
        "source_is_fake_t2w_case": source_info["source_is_fake_t2w_case"],
        "source_completion_mode": source_info["source_completion_mode"],
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
    source_seg_path_obj = parse_workspace_path(source_seg_path, PROJECT_ROOT) if source_seg_path else None
    if source_seg_path_obj:
        try:
            source_seg_arr = np.asanyarray(nib.load(str(source_seg_path_obj)).dataobj)
            if label_arr is not None:
                source_shapes_match = source_seg_arr.shape == label_arr.shape
        except Exception:  # noqa: BLE001
            source_seg_arr = None
    if source_seg_arr is not None and source_seg_arr.ndim >= 3:
        rows["source_shape_x"], rows["source_shape_y"], rows["source_shape_z"] = source_seg_arr.shape[:3]
    else:
        rows["source_shape_x"] = ""
        rows["source_shape_y"] = ""
        rows["source_shape_z"] = ""
    for mod_name, path_str in [("t1n", source_t1n_path), ("t1c", source_t1c_path), ("t2w", source_t2w_path), ("t2f", source_t2f_path)]:
        if path_str:
            try:
                source_path_obj = parse_workspace_path(path_str, PROJECT_ROOT)
                source_arrays[mod_name] = np.asanyarray(nib.load(str(source_path_obj)).dataobj)
            except Exception:  # noqa: BLE001
                continue
    rows["source_shape_match"] = bool(rows["source_shape_match"] or source_shapes_match)
    rows["source_modalities_compared"] = ""
    rows["source_modality_comparison_complete"] = False
    rows["source_existing_lesion_overlap"] = ""
    rows["source_seg_change_ratio"] = ""
    rows["brain_mask_overlap_ratio"] = ""
    rows["nonroi_change_ratio"] = ""
    rows["protected_source_change_ratio"] = ""
    rows["intensity_drift_p1"] = ""
    rows["intensity_drift_p50"] = ""
    rows["intensity_drift_p99"] = ""
    rows["source_synth_roi_ssim"] = ""
    rows["z_continuity_score"] = ""
    rows["z_area_smoothness"] = ""
    rows["z_intensity_smoothness"] = ""
    rows["artifact_suspected"] = False
    rows["artifact_block_score"] = ""
    rows["lesion_bbox_fill_ratio"] = ""
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
    rows["roi_boundary_p95_jump"] = ""
    rows["roi_bbox_available"] = False
    rows["roi_inside_image"] = False
    rows["bbox_inside_image"] = False
    rows["lesion_inside_brain_ok"] = False
    rows["t1n_min"] = rows["t1n_p1"] = rows["t1n_p50"] = rows["t1n_p99"] = rows["t1n_max"] = ""
    rows["t1c_min"] = rows["t1c_p1"] = rows["t1c_p50"] = rows["t1c_p99"] = rows["t1c_max"] = ""
    rows["t2w_min"] = rows["t2w_p1"] = rows["t2w_p50"] = rows["t2w_p99"] = rows["t2w_max"] = ""
    rows["t2f_min"] = rows["t2f_p1"] = rows["t2f_p50"] = rows["t2f_p99"] = rows["t2f_max"] = ""
    rows["teacher_model"] = "not_run"
    rows["teacher_dice_label_1"] = ""
    rows["teacher_dice_label_2"] = ""
    rows["teacher_dice_label_3"] = ""
    rows["teacher_dice_label_4"] = ""
    rows["teacher_lesion_count_diff"] = ""
    rows["teacher_missing_large_lesion_count"] = ""
    rows["teacher_extra_large_lesion_count"] = ""

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
            rows["lesion_bbox_fill_ratio"] = float(lesion_mask.sum() / bbox_volume) if bbox_volume else ""
        else:
            rows["lesion_count"] = 0
            rows["tiny_lesion_count"] = 0
            rows["small_lesion_count"] = 0
            rows["large_lesion_count"] = 0
            rows["min_lesion_volume_mm3"] = 0.0
            rows["p50_lesion_volume_mm3"] = 0.0
            rows["max_lesion_volume_mm3"] = 0.0
            rows["tiny_lesion_ratio"] = 0.0
            rows["lesion_bbox_fill_ratio"] = 0.0
            rows["artifact_suspected"] = False

        if source_seg_arr is not None:
            source_mask = source_seg_arr > 0
            inter = np.logical_and(source_mask, lesion_mask)
            denom = float(source_mask.sum() + lesion_mask.sum())
            rows["source_existing_lesion_overlap"] = float((2.0 * inter.sum()) / denom) if denom else ""
            if label_arr is not None and source_seg_arr.shape == label_arr.shape:
                rows["source_seg_change_ratio"] = float(np.mean(source_seg_arr != label_arr))
        else:
            rows["source_existing_lesion_overlap"] = ""

        brain_inputs = [
            source_arrays[mod]
            for mod in ["t1n", "t1c", "t2w", "t2f"]
            if mod in source_arrays and source_arrays[mod].shape == lesion_mask.shape
        ]
        if not brain_inputs:
            brain_inputs = [
                arrays[mod]
                for mod in ["t1n", "t1c", "t2w", "t2f"]
                if mod in arrays and arrays[mod].shape == lesion_mask.shape
            ]
        if brain_inputs and lesion_mask.any():
            brain_mask = np.logical_or.reduce(
                [np.abs(image.astype(np.float32, copy=False)) > 1e-6 for image in brain_inputs]
            )
            rows["brain_mask_overlap_ratio"] = float(np.mean(brain_mask[lesion_mask]))
            rows["lesion_inside_brain_ok"] = bool(
                rows["lesion_inside_brain_ok"]
                and float(rows["brain_mask_overlap_ratio"]) >= 0.95
            )
        else:
            rows["brain_mask_overlap_ratio"] = ""
            rows["lesion_inside_brain_ok"] = False

        rows["rc_source_allowed"] = bool(rows["has_rc"] and source_info["source_is_allowed"])

        if arrays:
            first_mod = "t1c" if "t1c" in arrays else sorted(arrays.keys())[0]
            support_matches = sorted(case_dir.glob("*-generation_support.nii.gz"))
            quality_mask = lesion_mask
            if support_matches:
                try:
                    support_arr = np.asanyarray(nib.load(str(support_matches[0])).dataobj)
                    if support_arr.shape == lesion_mask.shape and np.any(support_arr > 0):
                        quality_mask = support_arr > 0
                except Exception:  # noqa: BLE001
                    pass
            boundary = shell_mask(quality_mask)
            inside = quality_mask
            outside = ~quality_mask
            boundary_diffs = []
            boundary_gradient_diffs = []
            boundary_gradient_values: list[np.ndarray] = []
            normalized_drift_values: list[np.ndarray] = []
            nonroi_change_values = []
            protected_change_values = []
            roi_ssim_values = []
            source_compared_modalities: set[str] = set()
            for mod, arr in arrays.items():
                if mod == "seg":
                    continue
                stats = array_stats(arr)
                rows[f"{mod}_min"] = stats["min"]
                rows[f"{mod}_p1"] = stats["p1"]
                rows[f"{mod}_p50"] = stats["p50"]
                rows[f"{mod}_p99"] = stats["p99"]
                rows[f"{mod}_max"] = stats["max"]
                if source_arrays.get(mod) is not None and source_arrays[mod].shape == arr.shape:
                    source_compared_modalities.add(mod)
                    source_arr = source_arrays[mod].astype(np.float32)
                    arr_float = arr.astype(np.float32)
                    diff = np.abs(arr_float - source_arr)
                    source_scale = max(
                        float(np.percentile(source_arr, 99) - np.percentile(source_arr, 1)),
                        1e-6,
                    )
                    if completion_mode:
                        if mod != "t2w":
                            tolerance = max(source_scale * 1e-6, 1e-6)
                            protected_change_values.append(float(np.mean(diff > tolerance)))
                        continue
                    if boundary.any():
                        boundary_diffs.append(float(diff[boundary].mean() / source_scale))
                        generated_gradient = gradient_magnitude(arr_float)
                        source_gradient = gradient_magnitude(source_arr)
                        normalized_gradient_diff = (
                            np.abs(generated_gradient[boundary] - source_gradient[boundary]) / source_scale
                        )
                        boundary_gradient_diffs.append(float(np.mean(normalized_gradient_diff)))
                        boundary_gradient_values.append(normalized_gradient_diff)
                    if inside.any():
                        normalized_drift_values.append(diff[inside] / source_scale)
                        coords = np.argwhere(inside)
                        mins = coords.min(axis=0)
                        maxs = coords.max(axis=0) + 1
                        slices = tuple(slice(int(mins[axis]), int(maxs[axis])) for axis in range(3))
                        score = normalized_ssim(arr_float[slices], source_arr[slices])
                        if score != "":
                            roi_ssim_values.append(float(score))
                    if outside.any() and not completion_mode:
                        tolerance = max(source_scale * 1e-5, 1e-6)
                        nonroi_change_values.append(float(np.mean(diff[outside] > tolerance)))
            rows["roi_boundary_mae"] = float(np.mean(boundary_diffs)) if boundary_diffs else ""
            rows["roi_boundary_gradient_jump"] = (
                float(np.mean(boundary_gradient_diffs)) if boundary_gradient_diffs else ""
            )
            rows["roi_boundary_p95_jump"] = (
                float(np.percentile(np.concatenate(boundary_gradient_values), 95))
                if boundary_gradient_values
                else ""
            )
            artifact_components = [
                float(value)
                for value in (
                    rows["roi_boundary_mae"],
                    rows["roi_boundary_gradient_jump"],
                    rows["roi_boundary_p95_jump"],
                )
                if value != ""
            ]
            rows["artifact_block_score"] = max(artifact_components) if artifact_components else ""
            rows["artifact_suspected"] = bool(
                rows["artifact_block_score"] != "" and float(rows["artifact_block_score"]) > 0.25
            )
            if normalized_drift_values:
                drift = np.concatenate(normalized_drift_values)
                rows["intensity_drift_p1"] = float(np.percentile(drift, 1))
                rows["intensity_drift_p50"] = float(np.percentile(drift, 50))
                rows["intensity_drift_p99"] = float(np.percentile(drift, 99))
            rows["nonroi_change_ratio"] = float(np.max(nonroi_change_values)) if nonroi_change_values else ""
            rows["protected_source_change_ratio"] = (
                float(np.max(protected_change_values)) if protected_change_values else ""
            )
            rows["source_synth_roi_ssim"] = float(np.mean(roi_ssim_values)) if roi_ssim_values else ""
            required_source_modalities = {"t1n", "t1c", "t2f"} if completion_mode else {"t1n", "t1c", "t2w", "t2f"}
            rows["source_modalities_compared"] = ";".join(sorted(source_compared_modalities))
            rows["source_modality_comparison_complete"] = required_source_modalities.issubset(
                source_compared_modalities
            )
            z_mod = "t2w" if completion_mode and "t2w" in arrays else first_mod
            z_scores = z_smoothness_scores(arrays[z_mod].astype(np.float32), lesion_mask)
            rows["z_continuity_score"], rows["z_area_smoothness"], rows["z_intensity_smoothness"] = z_scores
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
            rows["roi_boundary_p95_jump"] = ""
            rows["intensity_drift_p50"] = ""
            rows["intensity_drift_p1"] = ""
            rows["intensity_drift_p99"] = ""
            rows["nonroi_change_ratio"] = ""
            rows["protected_source_change_ratio"] = ""
            rows["source_synth_roi_ssim"] = ""
            rows["cross_modality_roi_corr"] = ""
            rows["source_modalities_compared"] = ""
            rows["source_modality_comparison_complete"] = False

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
    completion_mode = bool(rows.get("source_completion_mode")) or str(rows.get("label_kind", "")) == "completion"
    hard_reject_reasons: list[str] = []
    if not rows["metadata_complete"]:
        hard_reject_reasons.append(f"metadata_incomplete:{rows['metadata_missing_fields']}")
    if not rows["source_is_allowed"]:
        hard_reject_reasons.append("source_not_allowed")
    if rows["validation_leakage"] and not completion_mode:
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
    if not rows["lesion_inside_brain_ok"]:
        hard_reject_reasons.append("lesion_outside_brain")
    if rows["image_is_constant"]:
        hard_reject_reasons.append("constant_image")
    if output_scheme == "mixed":
        hard_reject_reasons.append("mixed_suffix_scheme")
    if not rows["source_shape_match"]:
        hard_reject_reasons.append("source_shape_mismatch")
    if not rows["source_modality_comparison_complete"]:
        hard_reject_reasons.append("source_modality_comparison_incomplete")
    if rows["source_seg_change_ratio"] == "":
        hard_reject_reasons.append("source_seg_comparison_unavailable")
    elif float(rows["source_seg_change_ratio"]) > 0:
        hard_reject_reasons.append("source_seg_changed")
    if not rows["source_final_qc_pass"]:
        hard_reject_reasons.append("source_final_qc_failed")
    if completion_mode and not rows["source_is_fake_t2w_case"]:
        hard_reject_reasons.append("completion_source_not_marked_fake_t2w")
    if completion_mode and not nnunet_case_id:
        hard_reject_reasons.append("completion_source_missing_master_id")
    if completion_mode and rows["protected_source_change_ratio"] == "":
        hard_reject_reasons.append("completion_protected_modalities_unverifiable")
    elif completion_mode and float(rows["protected_source_change_ratio"]) > 0:
        hard_reject_reasons.append("completion_protected_modalities_changed")
    if not completion_mode and rows["nonroi_change_ratio"] == "":
        hard_reject_reasons.append("v2_nonroi_comparison_unavailable")
    elif not completion_mode and float(rows["nonroi_change_ratio"]) > 1e-4:
        hard_reject_reasons.append("v2_nonroi_changed")

    review_reasons: list[str] = []
    if output_scheme == "legacy_met":
        review_reasons.append("legacy_suffix_normalized")
    if rows["roi_bbox_available"] is False:
        review_reasons.append("roi_missing")
    if rows["artifact_suspected"]:
        review_reasons.append("block_artifact_suspected")
    if rows["tiny_lesion_ratio"] != "" and float(rows["tiny_lesion_ratio"]) > 0.5:
        review_reasons.append("tiny_ratio_high")
    if rows["label_modality_alignment_score"] != "" and float(rows["label_modality_alignment_score"]) < 1.0:
        review_reasons.append("alignment_low")
    if rows["z_continuity_score"] != "" and float(rows["z_continuity_score"]) < 0.5:
        review_reasons.append("z_discontinuity")
    if not completion_mode and rows["intensity_drift_p99"] != "" and float(rows["intensity_drift_p99"]) > 2.0:
        review_reasons.append("extreme_intensity_drift")
    approval_lookup = run_ctx.get("approval_lookup", {})
    approval_row = approval_lookup.get(synthetic_raw_id, {}) if isinstance(approval_lookup, dict) else {}
    approved_for_training = boolish(approval_row.get("approved_for_training", False))
    approved_for_evaluation = boolish(approval_row.get("approved_for_evaluation", False))
    approval_ambiguous_ids = run_ctx.get("approval_ambiguous_ids", set())
    if synthetic_raw_id in approval_ambiguous_ids:
        review_reasons.append("release_approval_ambiguous")
    elif not approved_for_training and not approved_for_evaluation:
        review_reasons.append("release_approval_missing")

    rows["hard_reject"] = bool(hard_reject_reasons)
    rows["hard_reject_reason"] = ";".join(hard_reject_reasons)
    rows["manual_review_required"] = bool(review_reasons)
    rows["manual_review_reason"] = ";".join(review_reasons)
    rows["manual_review_priority"] = "high" if any(reason in review_reasons for reason in ["roi_missing", "block_artifact_suspected", "alignment_low"]) else ("medium" if review_reasons else "")
    rows["accepted_for_training"] = False
    rows["accepted_for_evaluation"] = False
    rows["pending_review"] = False
    if hard_reject_reasons:
        rows["quality_grade"] = "F"
        rows["qc_decision"] = "rejected"
        rows["needs_regeneration"] = True
        rows["regeneration_reason"] = ";".join(hard_reject_reasons)
    elif not review_reasons:
        source_split = str(rows.get("source_split", ""))
        if completion_mode and source_split in {"val", "test"}:
            if approved_for_evaluation:
                rows["quality_grade"] = "A"
                rows["qc_decision"] = "accepted_for_evaluation"
                rows["accepted_for_evaluation"] = True
            else:
                rows["quality_grade"] = "C"
                rows["qc_decision"] = "pending_review"
                rows["pending_review"] = True
        else:
            if approved_for_training:
                rows["quality_grade"] = "A"
                rows["qc_decision"] = "accepted_for_training"
                rows["accepted_for_training"] = True
            else:
                rows["quality_grade"] = "C"
                rows["qc_decision"] = "pending_review"
                rows["pending_review"] = True
        rows["needs_regeneration"] = False
        rows["regeneration_reason"] = ""
    else:
        rows["quality_grade"] = "C"
        rows["qc_decision"] = "pending_review"
        rows["pending_review"] = True
        rows["needs_regeneration"] = False
        rows["regeneration_reason"] = ""
    rows["qc_status"] = rows["qc_decision"]
    rows["release_status"] = rows["qc_decision"]
    rows["qc_reject_reason"] = rows["hard_reject_reason"] if rows["hard_reject_reason"] else rows["manual_review_reason"]
    rows["status"] = rows["qc_decision"]
    rows["synthetic_final_id"] = synthetic_final_id
    rows["nnunet_case_id"] = nnunet_case_id
    rows["error_type"] = "nifti_read_error" if errors else ""
    rows["error_message"] = ";".join(errors)

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
        "crop_size": rows["crop_size"],
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
        "protected_source_change_ratio": rows["protected_source_change_ratio"],
        "source_seg_change_ratio": rows["source_seg_change_ratio"],
        "brain_mask_overlap_ratio": rows["brain_mask_overlap_ratio"],
        "roi_boundary_mae": rows["roi_boundary_mae"],
        "roi_boundary_gradient_jump": rows["roi_boundary_gradient_jump"],
        "roi_boundary_p95_jump": rows["roi_boundary_p95_jump"],
        "z_continuity_score": rows["z_continuity_score"],
        "z_area_smoothness": rows["z_area_smoothness"],
        "z_intensity_smoothness": rows["z_intensity_smoothness"],
        "intensity_drift_p1": rows["intensity_drift_p1"],
        "intensity_drift_p50": rows["intensity_drift_p50"],
        "intensity_drift_p99": rows["intensity_drift_p99"],
        "artifact_block_score": rows["artifact_block_score"],
        "et_t1c_contrast_ratio": rows["et_t1c_contrast_ratio"],
        "snfh_t2f_contrast_ratio": rows["snfh_t2f_contrast_ratio"],
        "snfh_t2w_contrast_ratio": rows["snfh_t2w_contrast_ratio"],
        "cross_modality_roi_corr": rows["cross_modality_roi_corr"],
        "label_modality_alignment_score": rows["label_modality_alignment_score"],
        "source_synth_roi_ssim": rows["source_synth_roi_ssim"],
        "label_source_seg_dice": rows["source_existing_lesion_overlap"],
        "teacher_model": rows["teacher_model"],
        "teacher_lesion_count_diff": rows["teacher_lesion_count_diff"],
        "manual_visual_score": "",
        "quality_grade": rows["quality_grade"],
        "diffusion_quality_decision": rows["qc_decision"],
        "diffusion_quality_reason": rows["qc_reject_reason"],
    }
    qc_row = dict(rows)

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
        "code/g2_create_train_val_test_split.py": "患者分组 master split 脚本，真实 T2W 子集复现 G1 V3 seed=42 划分。",
        "code/g2_synthetic_raw_intake_qc.py": "通用 G1 run 接收脚本，输出 rejected、pending、accepted-training 和 accepted-evaluation。",
        "code/g2_v2_compose_augmentation.py": "V2 composition 脚本：将平铺 ROI 输出回填 source 并恢复完整几何。",
        "code/g2_v3_completion_intake.py": "V3 completion 专用入口：校验 checkpoint、seed、bbdm_s 和 source manifest。",
        "code/g2_materialize_nnunet_dataset.py": "双视图物化脚本：同时生成 nnU-Net 和病例目录、fixed split 与完整性报告。",
        "code/g2_official_mets_metrics_parser.py": "官方指标代理脚本：解析 BraTS_evaluation Panoptica JSON 或校验 CSV 是否包含 2026 Task1 leaderboard 字段。",
        "docs/G1_G2_diffusion_output_contract.md": "G1 raw output 与 G2 适配边界的主契约，定义 raw 命名、source CSV、manifest 字段和最低 smoke 标准。",
        "docs/G2_G1适配执行清单.md": "按执行顺序拆解 G2 先准备什么、G1 输出后 G2 做什么、如何形成 QC 结果与回传。",
        "docs/G2_数据生成与质量控制实施方案.md": "总方案，解释 G2 为什么是 adapter/auditor/publisher，以及 raw intake 到 nnU-Net 导出的全链路。",
        "docs/G2_模型训练完成前可执行工作清单.md": "训练前能立即执行的工作清单，属于 G2 的下一步行动仓库。",
        "results/README.md": "results 总说明，概括本目录只保存轻量产物，不保存大体积 NIfTI。",
        "results/manifests/README.md": "清单区说明，解释真实清单、source CSV、synthetic intake manifest 与 accepted/rejected 输出。",
        "results/manifests/corrected_label_overlay.csv": "真实训练病例的 corrected label 覆盖记录，说明哪些病例在最终 manifest 中替换了原始 seg。",
        "results/manifests/g1_v2_source_manifest.csv": "V2 source 主表；只有 master train 且真实 T2W 的病例允许生成。",
        "results/manifests/nnunet_case_mapping_master.csv": "全部 1295 个可追溯病例身份，包含 265 个 completion 目标。",
        "results/manifests/nnunet_case_mapping_realonly.csv": "real-only nnU-Net 映射表，用于训练机物化 imagesTr/labelsTr。",
        "results/manifests/real_train_manifest.csv": "真实训练病例最终主表，已应用 corrected label overlay 并带 final_qc_pass。",
        "results/manifests/real_train_manifest_raw.csv": "原始训练病例扫描表，保留 raw seg 与基础 QC 证据。",
        "results/manifests/real_validation_manifest.csv": "官方 validation 路径与结构检查表，绝不作为 synthetic source。",
        "results/manifests/synthetic_generation_manifest_template_g1.csv": "G1 raw output 或 G2 补建时使用的 synthetic manifest 表头模板。",
        "results/manifests/synthetic_normalized_mapping_template.csv": "逐模态标准化映射模板，定义 raw source、normalized target 与 nnU-Net target 的对应关系。",
        "results/nnunet_raw/README.md": "nnU-Net raw 根目录说明，说明这里是训练机物化入口，不在仓库保存正式大体积影像。",
        "results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/README.md": "real-only 数据集占位说明，表示当前只保存 dataset.json 与路径契约。",
        "results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/dataset.json": "nnU-Net dataset.json 草案，定义四模态顺序与五类标签。",
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
        "results/reports/README.md": "报告目录总说明，承接路径检查、QC 汇总、进度报告与模板。",
        "results/reports/G2_progress_report.md": "G2 主进度报告，汇总当前完成度、文件索引和下一步计划。",
        "results/reports/ablation_plan_template.md": "real-only / real+synth 的消融模板。",
        "results/reports/local_data_paths_check.md": "本机外部数据路径检查结果。",
        "results/reports/real_data_qc_summary.md": "真实训练数据 QC 汇总。",
        "results/splits/README.md": "固定真实 train/val/test 划分说明。",
        "results/splits/splits_master_train_val_test.json": "全部病例 patient-group master split。",
        "results/splits/splits_final_train_val_test.json": "从 master split 派生的真实 T2W real-only split。",
        "results/splits/splits_final_train_val_test_membership.csv": "逐病例 split membership 表，便于人工核查和脚本读取。",
        "results/stats/README.md": "统计区说明，解释 label/lesion 分布与 synthetic 目标分布。",
        "results/stats/real_label_distribution.csv": "真实训练病例级 label 体素与体积分布。",
        "results/stats/real_lesion_distribution.csv": "真实 lesion component 级分布。",
        "results/stats/real_lesion_distribution_summary.json": "机器可读统计摘要。",
        "results/stats/real_lesion_distribution_summary.md": "人可读统计摘要。",
        "results/stats/target_synthetic_distribution_v1.md": "第一轮 synthetic 目标分布与生成限制。",
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
                    file_notes[rel] = "本次 synthetic run 正式批准进入训练的病例清单。"
                elif title == "synthetic_accepted_evaluation_manifest":
                    file_notes[rel] = "V3 val/test completion 正式批准用于固定评估的病例清单。"
                elif title == "synthetic_pending_review_manifest":
                    file_notes[rel] = "技术 QC 通过但尚未完成 teacher/人工审批的病例清单。"
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
    ]
    # Keep this report in the user-facing "8 folders"口径.
    folders = [
        ("1. code", "code", [
            "code/.gitkeep",
            "code/g2_pretraining_audit.py",
            "code/g2_create_train_val_test_split.py",
            "code/g2_synthetic_raw_intake_qc.py",
            "code/g2_v2_compose_augmentation.py",
            "code/g2_v3_completion_intake.py",
            "code/g2_materialize_nnunet_dataset.py",
            "code/g2_official_mets_metrics_parser.py",
        ]),
        ("2. docs", "docs", [
            "docs/G1_G2_diffusion_output_contract.md",
            "docs/G2_G1适配执行清单.md",
            "docs/G1_G2_服务器训练推理QC运行手册.md",
            "docs/G2_数据生成与质量控制实施方案.md",
            "docs/G2_模型训练完成前可执行工作清单.md",
        ]),
        ("3. results/manifests", "results/manifests", [
            "results/manifests/README.md",
            "results/manifests/corrected_label_overlay.csv",
            "results/manifests/g1_v2_source_manifest.csv",
            "results/manifests/nnunet_case_mapping_master.csv",
            "results/manifests/nnunet_case_mapping_realonly.csv",
            "results/manifests/real_train_manifest.csv",
            "results/manifests/real_train_manifest_raw.csv",
            "results/manifests/real_validation_manifest.csv",
            "results/manifests/synthetic_generation_manifest_template_g1.csv",
            "results/manifests/synthetic_normalized_mapping_template.csv",
        ]),
        ("4. results/stats", "results/stats", [
            "results/stats/README.md",
            "results/stats/real_label_distribution.csv",
            "results/stats/real_lesion_distribution.csv",
            "results/stats/real_lesion_distribution_summary.json",
            "results/stats/real_lesion_distribution_summary.md",
            "results/stats/target_synthetic_distribution_v1.md",
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
        ]),
        ("6. results/splits", "results/splits", [
            "results/splits/README.md",
            "results/splits/splits_master_train_val_test.json",
            "results/splits/splits_master_train_val_test_membership.csv",
            "results/splits/splits_final_train_val_test.json",
            "results/splits/splits_final_train_val_test_membership.csv",
        ]),
        ("7. results/reports", "results/reports", [
            "results/reports/README.md",
            "results/reports/G2_progress_report.md",
            "results/reports/ablation_plan_template.md",
            "results/reports/local_data_paths_check.md",
            "results/reports/real_data_qc_summary.md",
        ]),
        ("8. results/nnunet_raw", "results/nnunet_raw", [
            "results/nnunet_raw/README.md",
            "results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/README.md",
            "results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/dataset.json",
        ]),
    ]
    lines = [
        "# G2 Synthetic Intake 进度报告",
        "",
        f"- 生成日期：{RUN_DATE}",
        f"- 项目根目录：`{PROJECT_ROOT_NAME}`",
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
            f"- accepted for training：{run_summary.get('accepted_training_count', run_summary.get('accepted_count', 0))}",
            f"- accepted for evaluation：{run_summary.get('accepted_evaluation_count', 0)}",
            f"- pending review：{run_summary.get('pending_review_count', 0)}",
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
            f"- accepted for training：{smoke_summary.get('accepted_training_count', 0)}",
            f"- accepted for evaluation：{smoke_summary.get('accepted_evaluation_count', 0)}",
            f"- pending review：{smoke_summary.get('pending_review_count', 0)}",
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
        "1. V3 阶段 6 完成后先运行 completion 专用 intake。",
        "2. V2 正式批量输出必须先运行 composer，再进入通用 QC。",
        "3. pending 病例完成 teacher/人工审批后才可物化。",
        "4. 使用固定真实验证集完成 real-only 与 real+synth 消融。",
    ])
    if intake_outputs:
        lines.extend(["", "## 本次生成的文件", ""])
        for path in intake_outputs:
            lines.append(f"- `{display_results_path(path, results_root)}`")
    if intake_index:
        lines.extend(["", "## Intake 索引", ""])
        for title, paths in intake_index:
            lines.append(f"### {title}")
            if not paths:
                lines.append("无。")
                lines.append("")
                continue
            for path in paths:
                lines.append(f"- `{display_results_path(path, results_root)}`")
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
        "1. G2 已完成 patient-group master split、V2/V3 分流、严格 metadata gate 和三态 release 接口。",
        "2. 当前测试证明接口与小型 NIfTI fixture 可运行；真实生成质量仍须等待服务器 NIfTI 批次验收。",
        "3. 大体积影像仍留在外部数据盘或训练机器，不进入仓库。",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_synthetic_run_context(run_root: Path, results_root: Path, args: argparse.Namespace) -> dict[str, object]:
    config_path = run_root / "generation_config.json"
    inference_path = run_root / "inference_run.json"
    log_path = run_root / "generation_log.jsonl"
    manifest_path = run_root / "synthetic_generation_manifest.csv"
    approval_path = run_root / "g2_approval_manifest.csv"
    config = read_json_if_exists(config_path)
    inference_config = read_json_if_exists(inference_path)
    if not config and inference_config:
        config = dict(inference_config)
    log_rows = read_jsonl_if_exists(log_path)
    manifest_df = read_csv_if_exists(manifest_path)
    manifest_rows = (
        manifest_df.fillna("").to_dict(orient="records")
        if not manifest_df.empty
        else []
    )
    manifest_lookup, manifest_duplicate_ids = index_case_records(manifest_rows)
    log_lookup, log_duplicate_ids = index_case_records(log_rows)
    requested_mode = str(getattr(args, "generation_mode", "auto") or "auto")
    configured_mode = str(
        recursive_find_value(config, "generation_mode")
        or recursive_find_value(config, "generator_io")
        or ""
    )
    if requested_mode == "auto":
        if inference_config:
            requested_mode = "completion"
        elif "completion" in configured_mode:
            requested_mode = "completion"
        elif configured_mode:
            requested_mode = "full_generation"
    configured_run_id = (
        recursive_find_value(config, "generation_run_id")
        or recursive_find_value(config, "run_id")
    )
    run_id = str(args.synthetic_run_id or configured_run_id or run_root.name)
    encdec_checkpoint = str(recursive_find_value(config, "encdec_checkpoint") or "")
    bbdm_checkpoint = str(recursive_find_value(config, "bbdm_checkpoint") or "")
    vae_checkpoint = str(recursive_find_value(config, "vae_weights") or recursive_find_value(config, "vae_checkpoint") or "")
    validation_run = str(recursive_find_value(config, "validation_run") or "")
    generator_checkpoint = str(
        recursive_find_value(config, "generator_checkpoint")
        or recursive_find_value(config, "checkpoint")
        or bbdm_checkpoint
        or encdec_checkpoint
        or ""
    )
    source_csv_value = (
        recursive_find_value(config, "source_csv")
        or recursive_find_value(config, "source_manifest")
        or recursive_find_value(config, "input_manifest")
        or ""
    )
    configured_generator_name = (
        recursive_find_value(config, "generator_name")
        or recursive_find_value(config, "model_name")
    )
    generator_name = str(configured_generator_name or ("g1_missing_t2w_v3" if inference_config else ""))
    seed = recursive_find_value(config, "seed")
    if seed is None:
        seed = recursive_find_value(config, "random_seed")
    checkpoint_dir = str(recursive_find_value(config, "diffusion_checkpoint_dir") or "")
    metadata_missing_fields: list[str] = []
    if not (config_path.exists() or inference_path.exists()):
        metadata_missing_fields.append("generation_config_or_inference_run")
    if not manifest_path.exists():
        metadata_missing_fields.append("synthetic_generation_manifest")
    elif not manifest_rows:
        metadata_missing_fields.append("synthetic_generation_manifest_empty")
    if manifest_duplicate_ids:
        metadata_missing_fields.append("synthetic_generation_manifest_duplicate_ids")
    if not log_path.exists():
        metadata_missing_fields.append("generation_log")
    elif not log_rows:
        metadata_missing_fields.append("generation_log_empty")
    if log_duplicate_ids:
        metadata_missing_fields.append("generation_log_duplicate_ids")
    if not configured_run_id and not args.synthetic_run_id:
        metadata_missing_fields.append("generation_run_id")
    if not configured_generator_name:
        metadata_missing_fields.append("generator_name")
    if seed in (None, ""):
        metadata_missing_fields.append("seed")
    if not source_csv_value:
        metadata_missing_fields.append("source_csv")
    if requested_mode == "completion":
        if not vae_checkpoint:
            metadata_missing_fields.append("vae_weights")
        if not encdec_checkpoint:
            metadata_missing_fields.append("encdec_checkpoint")
        if not bbdm_checkpoint:
            metadata_missing_fields.append("bbdm_checkpoint")
        if recursive_find_value(config, "bbdm_s") in (None, ""):
            metadata_missing_fields.append("bbdm_s")
        if not validation_run:
            metadata_missing_fields.append("validation_run")
    else:
        modal_checkpoints = [
            recursive_find_value(config, f"generator_checkpoint_{mod}")
            for mod in ("t1n", "t1c", "t2w", "t2f")
        ]
        if not checkpoint_dir and not generator_checkpoint and not all(modal_checkpoints):
            metadata_missing_fields.append("diffusion_checkpoints")
        for field in ("sampling_method", "sampling_steps", "eta", "crop_size"):
            if recursive_find_value(config, field) in (None, ""):
                metadata_missing_fields.append(field)

    approval_df = read_csv_if_exists(approval_path)
    approval_lookup: dict[str, dict[str, object]] = {}
    approval_ambiguous_ids: set[str] = set()
    if not approval_df.empty and "synthetic_raw_id" in approval_df.columns:
        approval_rows = approval_df.fillna("").to_dict(orient="records")
        approval_lookup, approval_ambiguous_ids = index_case_records(approval_rows)
        for raw_id in approval_ambiguous_ids:
            approval_lookup.pop(raw_id, None)
    if requested_mode == "completion":
        generator_checkpoints = {
            "t1n": "",
            "t1c": "",
            "t2w": bbdm_checkpoint,
            "t2f": "",
        }
    else:
        generator_checkpoints = {
            mod: str(recursive_find_value(config, f"generator_checkpoint_{mod}") or generator_checkpoint)
            for mod in ("t1n", "t1c", "t2w", "t2f")
        }
    label_channels_value = recursive_find_value(config, "label_channels")
    context = {
        "results_root": results_root,
        "run_root": run_root,
        "run_id": run_id,
        "generation_run_id": run_id,
        "generation_config_exists": config_path.exists() or inference_path.exists(),
        "inference_run_exists": inference_path.exists(),
        "generation_manifest_exists": manifest_path.exists(),
        "generation_log_exists": log_path.exists(),
        "approval_manifest_exists": approval_path.exists(),
        "approval_lookup": approval_lookup,
        "generation_config": config,
        "generation_log_rows": log_rows,
        "generation_manifest_lookup": manifest_lookup,
        "generation_log_lookup": log_lookup,
        "approval_ambiguous_ids": approval_ambiguous_ids,
        "generator_name": generator_name,
        "generator_checkpoint": generator_checkpoint,
        "generator_checkpoint_t1n": generator_checkpoints["t1n"],
        "generator_checkpoint_t1c": generator_checkpoints["t1c"],
        "generator_checkpoint_t2w": generator_checkpoints["t2w"],
        "generator_checkpoint_t2f": generator_checkpoints["t2f"],
        "vae_checkpoint": vae_checkpoint,
        "encdec_checkpoint": encdec_checkpoint,
        "bbdm_checkpoint": bbdm_checkpoint,
        "bbdm_s": recursive_find_value(config, "bbdm_s") or "",
        "validation_run": validation_run,
        "generator_io": str(recursive_find_value(config, "generator_io") or recursive_find_value(config, "io_mode") or configured_mode or "unknown"),
        "generation_mode_override": requested_mode,
        "label_channels": int(label_channels_value) if label_channels_value not in (None, "") else 0,
        "rc_policy": str(recursive_find_value(config, "rc_policy") or ""),
        "noise_type": str(recursive_find_value(config, "noise_type") or ""),
        "sampling_method": str(recursive_find_value(config, "sampling_method") or ""),
        "sampling_steps": recursive_find_value(config, "sampling_steps") if recursive_find_value(config, "sampling_steps") is not None else "",
        "eta": recursive_find_value(config, "eta") if recursive_find_value(config, "eta") is not None else "",
        "crop_size": recursive_find_value(config, "crop_size") if recursive_find_value(config, "crop_size") is not None else "",
        "seed": seed if seed is not None else "",
        "source_csv_path": str(source_csv_value),
        "source_csv_version": str(recursive_find_value(config, "source_csv_version") or Path(str(source_csv_value)).name),
        "metadata_complete": not metadata_missing_fields,
        "metadata_missing_fields": metadata_missing_fields,
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
    accepted_training_rows = []
    accepted_evaluation_rows = []
    pending_rows = []
    rejected_rows = []
    mapping_rows = []
    for idx, case_dir in enumerate(case_dirs, start=1):
        parsed = parse_synthetic_case_name(case_dir.name)
        parsed = apply_generation_mode_override(parsed, str(ctx.get("generation_mode_override", "auto")))
        source_case_id = str(parsed.get("source_case_id") or "")
        label_kind = str(parsed.get("label_kind") or "")
        source_info = build_source_status(source_case_id, ctx, label_kind=label_kind) if source_case_id else {
            "source_row": {},
            "val_row": {},
            "g1_row": {},
            "nnunet_case_id": "",
            "source_in_real_train_manifest": False,
            "source_final_qc_pass": False,
            "source_allowed_for_v2": False,
            "source_allowed_for_training": False,
            "source_is_fake_t2w_case": False,
            "source_completion_mode": label_kind == "completion",
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
        if qc_row["pending_review"]:
            review_rows.append(review_row)
        if bool(qc_row["accepted_for_training"]):
            accepted_training_rows.append(manifest_row)
        elif bool(qc_row["accepted_for_evaluation"]):
            accepted_evaluation_rows.append(manifest_row)
        elif bool(qc_row["pending_review"]):
            pending_rows.append(manifest_row)
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
                "accepted_for_evaluation",
                "pending_review",
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
    evaluation_path = dirs["manifests"] / f"synthetic_accepted_evaluation_manifest_{run_id}.csv"
    pending_path = dirs["manifests"] / f"synthetic_pending_review_manifest_{run_id}.csv"
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
    pd.DataFrame(accepted_training_rows, columns=candidate_df.columns).to_csv(accepted_path, index=False)
    pd.DataFrame(accepted_evaluation_rows, columns=candidate_df.columns).to_csv(evaluation_path, index=False)
    pd.DataFrame(pending_rows, columns=candidate_df.columns).to_csv(pending_path, index=False)
    pd.DataFrame(rejected_rows, columns=candidate_df.columns).to_csv(rejected_path, index=False)
    mapping_df.to_csv(mapping_path, index=False)
    qc_df.to_csv(qc_path, index=False)
    diffusion_df.to_csv(diffusion_path, index=False)
    review_df.to_csv(review_path, index=False)

    summary = {
        "generation_run_id": run_id,
        "case_count": int(len(candidate_df)),
        "accepted_training_count": int(len(accepted_training_rows)),
        "accepted_evaluation_count": int(len(accepted_evaluation_rows)),
        "pending_review_count": int(len(pending_rows)),
        "needs_regeneration_count": int(sum(1 for row in qc_rows if row.get("needs_regeneration"))),
        "rejected_count": int(len(rejected_rows)),
        "legacy_suffix_count": int((candidate_df["output_suffix_scheme"] == "legacy_met").sum()) if not candidate_df.empty else 0,
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
        f"- accepted for training：{summary['accepted_training_count']}",
        f"- accepted for evaluation：{summary['accepted_evaluation_count']}",
        f"- pending review：{summary['pending_review_count']}",
        f"- needs regeneration：{summary['needs_regeneration_count']}",
        f"- rejected：{summary['rejected_count']}",
        "",
        "## 2. 生成与接收",
        "",
        f"- `generation_config.json`：{'存在' if ctx['generation_config_exists'] else '缺失'}",
        f"- `generation_log.jsonl`：{'存在' if ctx['generation_log_exists'] else '缺失'}",
        f"- `synthetic_generation_manifest.csv`：{'存在' if ctx['generation_manifest_exists'] else '缺失，已由 G2 补建'}",
        "",
        "## 3. release 结果",
        "",
        "### accepted for training",
        "",
        df_to_markdown(pd.DataFrame(accepted_training_rows)[["synthetic_raw_id", "synthetic_final_id", "source_case_id", "qc_decision"]] if accepted_training_rows else pd.DataFrame()),
        "",
        "### accepted for evaluation",
        "",
        df_to_markdown(pd.DataFrame(accepted_evaluation_rows)[["synthetic_raw_id", "synthetic_final_id", "source_case_id", "qc_decision"]] if accepted_evaluation_rows else pd.DataFrame()),
        "",
        "### pending review",
        "",
        df_to_markdown(pd.DataFrame(pending_rows)[["synthetic_raw_id", "synthetic_final_id", "source_case_id", "qc_reject_reason"]] if pending_rows else pd.DataFrame()),
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
    for path in [manifest_path, candidate_path, accepted_path, evaluation_path, pending_path, rejected_path, mapping_path, qc_path, diffusion_path, review_path, batch_summary_path]:
        lines.append(f"- `{path}`")
    lines.extend([
        "",
        "## 6. 结论",
        "",
        "1. G2 分别接收 V2 composed augmentation 与 V3 completion，不能直接接收多病例平铺 V2 raw output。",
        "2. G2 会额外生成 `synthetic_normalized_mapping_{run_id}.csv`，逐模态记录 raw legacy/native 文件到 2026 标准文件名和 nnU-Net 目标文件名的映射。",
        "3. 技术通过但没有审批的样本进入 pending manifest，不会自动伪装为 accepted。",
        "4. 真实验证 fold 和官方 validation 仍然不能作为 synthetic source。",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_progress_report(
        results_root,
        progress_report_path,
        summary,
        [manifest_path, candidate_path, accepted_path, evaluation_path, pending_path, rejected_path, mapping_path, qc_path, diffusion_path, review_path, batch_summary_path, report_path],
        [
            ("synthetic_generation_manifest", [manifest_path]),
            ("synthetic_candidate_manifest", [candidate_path]),
            ("synthetic_accepted_manifest", [accepted_path]),
            ("synthetic_accepted_evaluation_manifest", [evaluation_path]),
            ("synthetic_pending_review_manifest", [pending_path]),
            ("synthetic_rejected_manifest", [rejected_path]),
            ("synthetic_normalized_mapping", [mapping_path]),
            ("qc_metrics", [qc_path]),
            ("diffusion_quality_metrics", [diffusion_path]),
            ("qc_case_review", [review_path]),
            ("qc_batch_summary", [batch_summary_path]),
            ("quality_report", [report_path]),
        ],
    )
    outputs.extend([manifest_path, candidate_path, accepted_path, evaluation_path, pending_path, rejected_path, mapping_path, qc_path, diffusion_path, review_path, batch_summary_path, report_path, progress_report_path])
    return outputs


def scan_training(train_root: Path) -> pd.DataFrame:
    rows = []
    for case_dir in find_case_dirs(train_root):
        case_id = case_dir.name
        files = find_modality_files(case_dir, include_seg=True)
        row: dict[str, object] = {
            "case_id": case_id,
            "split_source": "train_ucsd" if "UCSD - Training" in case_dir.parts else "train_top_level",
            "case_dir": display_path(case_dir, train_root),
        }
        metas: dict[str, dict[str, object]] = {}
        reasons = []
        for mod in ["t1n", "t1c", "t2w", "t2f", "seg"]:
            path = files.get(mod)
            row[f"{'raw_seg' if mod == 'seg' else mod}_path"] = display_path(path, train_root)
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
        row: dict[str, object] = {"case_id": case_id, "case_dir": display_path(case_dir, validation_root)}
        metas: dict[str, dict[str, object]] = {}
        reasons = []
        for mod in ["t1n", "t1c", "t2w", "t2f"]:
            path = files.get(mod)
            row[f"{mod}_path"] = display_path(path, validation_root)
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


def apply_corrected_labels(raw_df: pd.DataFrame, corrected_root: Path, data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        raw_seg_path = parse_workspace_path(raw_row.raw_seg_path, data_root / "MICCAI-LH-BraTS2025-MET-Challenge-Training") if raw_row is not None and raw_row.raw_seg_path else None
        raw_values, _, _ = unique_label_values(raw_seg_path) if raw_seg_path else ([], False, "missing")
        corrected_values, _, corrected_error = unique_label_values(corrected_path)
        raw_meta = nifti_meta(raw_seg_path) if raw_seg_path else None
        corrected_meta = nifti_meta(corrected_path)
        applied = raw_row is not None and not corrected_error and raw_meta is not None and raw_meta["shape"] == corrected_meta["shape"]
        overlay_rows.append({
            "case_id": case_id,
            "raw_seg_path": display_path(raw_seg_path, data_root),
            "corrected_seg_path": display_path(corrected_path, data_root),
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
            final_df.loc[idx, "effective_seg_path"] = display_path(corrected_path, corrected_root.parent)
            final_df.loc[idx, "label_source"] = "corrected"
            final_df.loc[idx, "has_corrected_label"] = True

    for idx, row in final_df.iterrows():
        effective_path = parse_workspace_path(row["effective_seg_path"], data_root) if row["effective_seg_path"] else None
        values, finite, err = unique_label_values(effective_path) if effective_path else ([], False, "missing")
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


def label_stats(final_df: pd.DataFrame, data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
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
        seg_path = parse_workspace_path(row.effective_seg_path, data_root)
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


def nnunet_mapping(final_df: pd.DataFrame, data_root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    pass_df = final_df[final_df["final_qc_pass"] == True].sort_values("case_id").copy()  # noqa: E712
    rows = []
    source_to_nn = {}
    for idx, row in enumerate(pass_df.itertuples(index=False), start=1):
        nn_id = f"BraTSMET_{idx:06d}"
        source_to_nn[row.case_id] = nn_id
        rows.append({
            "nnunet_case_id": nn_id,
            "source_case_id": row.case_id,
            "t1n_source_path": display_path(row.t1n_path, data_root),
            "t1c_source_path": display_path(row.t1c_path, data_root),
            "t2w_source_path": display_path(row.t2w_path, data_root),
            "t2f_source_path": display_path(row.t2f_path, data_root),
            "seg_source_path": display_path(row.effective_seg_path, data_root),
            "label_source": row.label_source,
            "materialization_status": "deferred_no_nifti_copy_on_mac",
        })
    return pd.DataFrame(rows), source_to_nn


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
        "generator_checkpoint_t2w", "generator_checkpoint_t2f", "vae_checkpoint", "encdec_checkpoint",
        "bbdm_checkpoint", "bbdm_s", "validation_run", "generator_io", "label_channels", "rc_policy",
        "noise_type", "sampling_method", "sampling_steps", "eta", "crop_size", "seed", "source_csv_path", "source_csv_version",
        "raw_case_dir", "normalized_case_dir", "output_suffix_scheme", "suffix_conversion_action",
        "raw_t1n_path", "raw_t1c_path", "raw_t2w_path", "raw_t2f_path", "raw_seg_path",
        "normalized_t1n_path", "normalized_t1c_path", "normalized_t2w_path", "normalized_t2f_path",
        "normalized_seg_path", "nnunet_t1n_target_path", "nnunet_t1c_target_path", "nnunet_t2w_target_path",
        "nnunet_t2f_target_path", "nnunet_seg_target_path", "insert_center_x", "insert_center_y",
        "insert_center_z", "roi_x_min", "roi_x_max", "roi_y_min", "roi_y_max", "roi_z_min", "roi_z_max",
        "source_shape_x", "source_shape_y", "source_shape_z", "output_shape_x", "output_shape_y",
        "output_shape_z", "status", "error_type", "error_message", "qc_status", "qc_reject_reason",
        "source_allowed_for_training", "source_is_fake_t2w_case", "source_completion_mode",
        "metadata_complete", "metadata_missing_fields", "accepted_for_training",
        "accepted_for_evaluation", "pending_review", "needs_regeneration",
    ]
    with (dirs["manifests"] / "synthetic_generation_manifest_template_g1.csv").open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow(synthetic_manifest_header)

    normalized_mapping_header = [
        "synthetic_raw_id", "synthetic_final_id", "nnunet_case_id", "source_case_id", "generation_run_id",
        "modality", "nnunet_channel", "raw_source_path", "normalized_target_path", "nnunet_target_path",
        "output_suffix_scheme", "suffix_conversion_action", "qc_decision", "accepted_for_training",
        "accepted_for_evaluation", "pending_review", "needs_regeneration",
    ]
    with (dirs["manifests"] / "synthetic_normalized_mapping_template.csv").open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow(normalized_mapping_header)

    qc_v2_header = [
        "synthetic_raw_id", "synthetic_final_id", "nnunet_case_id", "source_case_id", "source_split",
        "label_kind", "label_index", "label_source_case_id", "label_component_id",
        "generation_run_id", "generator_name", "generator_checkpoint_t1n", "generator_checkpoint_t1c",
        "generator_checkpoint_t2w", "generator_checkpoint_t2f", "label_generator_checkpoint", "vae_checkpoint",
        "encdec_checkpoint", "bbdm_checkpoint", "bbdm_s", "validation_run", "generator_io",
        "generation_mode",
        "label_channels", "rc_policy", "noise_type", "sampling_method", "sampling_steps", "eta", "crop_size", "seed",
        "raw_case_dir", "normalized_case_dir", "output_suffix_scheme", "suffix_conversion_action",
        "config_exists", "manifest_exists", "log_exists", "manifest_case_record_exists",
        "log_case_record_exists", "metadata_complete", "metadata_missing_fields",
        "source_csv_path", "source_csv_version", "has_t1n", "has_t1c",
        "has_t2w", "has_t2f", "has_seg", "has_all_modalities", "filename_consistent", "nifti_readable",
        "raw_t1n_path", "raw_t1c_path", "raw_t2w_path", "raw_t2f_path", "raw_seg_path",
        "normalized_t1n_path", "normalized_t1c_path", "normalized_t2w_path", "normalized_t2f_path",
        "normalized_seg_path", "nnunet_t1n_target_path", "nnunet_t1c_target_path", "nnunet_t2w_target_path",
        "nnunet_t2f_target_path", "nnunet_seg_target_path", "shape_t1n", "shape_t1c",
        "shape_t2w", "shape_t2f", "shape_seg", "source_shape_x", "source_shape_y", "source_shape_z",
        "output_shape_x", "output_shape_y", "output_shape_z",
        "spacing_t1n", "spacing_t1c", "spacing_t2w", "spacing_t2f",
        "spacing_seg", "affine_hash_t1n", "affine_hash_t1c", "affine_hash_t2w", "affine_hash_t2f",
        "affine_hash_seg", "shape_consistent", "spacing_consistent", "affine_consistent", "affine_valid",
        "orientation_consistent",
        "source_shape_match", "source_modalities_compared", "source_modality_comparison_complete",
        "has_nan_or_inf", "image_is_constant",
        "t1n_min", "t1n_p1", "t1n_p50", "t1n_p99", "t1n_max",
        "t1c_min", "t1c_p1", "t1c_p50", "t1c_p99", "t1c_max",
        "t2w_min", "t2w_p1", "t2w_p50", "t2w_p99", "t2w_max",
        "t2f_min", "t2f_p1", "t2f_p50", "t2f_p99", "t2f_max",
        "label_is_integer", "label_values",
        "label_values_valid", "empty_mask", "allow_empty_mask", "source_in_real_train_manifest",
        "source_final_qc_pass", "source_allowed_for_v2", "source_allowed_for_training",
        "source_is_fake_t2w_case", "source_completion_mode", "source_in_fixed_val_fold",
        "source_from_official_validation", "source_is_allowed", "case_id_reuses_real_id", "validation_leakage",
        "roi_bbox_available", "insert_center_x", "insert_center_y", "insert_center_z", "roi_x_min", "roi_x_max",
        "roi_y_min", "roi_y_max", "roi_z_min", "roi_z_max", "roi_inside_image", "nonroi_change_ratio",
        "protected_source_change_ratio", "source_seg_change_ratio", "source_existing_lesion_overlap",
        "brain_mask_overlap_ratio", "lesion_count", "tiny_lesion_count",
        "small_lesion_count", "large_lesion_count", "min_lesion_volume_mm3", "p50_lesion_volume_mm3",
        "max_lesion_volume_mm3", "tiny_lesion_ratio", "label_combination", "has_rc", "rc_source_allowed",
        "bbox_inside_image", "lesion_inside_brain_ok", "et_t1c_contrast_ratio", "snfh_t2f_contrast_ratio",
        "snfh_t2w_contrast_ratio", "cross_modality_roi_corr", "label_modality_alignment_score",
        "roi_boundary_mae", "roi_boundary_gradient_jump", "roi_boundary_p95_jump",
        "z_continuity_score", "z_area_smoothness",
        "z_intensity_smoothness", "intensity_drift_p1", "intensity_drift_p50", "intensity_drift_p99",
        "source_synth_roi_ssim", "lesion_bbox_fill_ratio", "artifact_block_score",
        "artifact_suspected", "teacher_model", "teacher_dice_label_1", "teacher_dice_label_2",
        "teacher_dice_label_3", "teacher_dice_label_4", "teacher_lesion_count_diff",
        "teacher_missing_large_lesion_count", "teacher_extra_large_lesion_count", "manual_review_required",
        "manual_review_priority", "manual_review_reason", "hard_reject", "hard_reject_reason", "quality_grade",
        "qc_decision", "qc_status", "qc_reject_reason", "release_status", "status",
        "accepted_for_training", "accepted_for_evaluation", "pending_review", "needs_regeneration",
        "regeneration_reason", "error_type", "error_message",
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
        "sampling_method", "sampling_steps", "eta", "crop_size", "seed", "roi_bbox_available", "roi_x_min", "roi_x_max",
        "roi_y_min", "roi_y_max", "roi_z_min", "roi_z_max", "roi_volume_voxels", "lesion_voxels_in_roi",
        "lesion_inside_roi_ratio", "nonroi_change_ratio", "protected_source_change_ratio",
        "source_seg_change_ratio", "brain_mask_overlap_ratio", "roi_boundary_mae",
        "roi_boundary_gradient_jump", "roi_boundary_p95_jump", "z_continuity_score", "z_area_smoothness",
        "z_intensity_smoothness", "intensity_drift_p1", "intensity_drift_p50", "intensity_drift_p99",
        "artifact_block_score", "et_t1c_contrast_ratio",
        "snfh_t2f_contrast_ratio", "snfh_t2w_contrast_ratio", "cross_modality_roi_corr",
        "label_modality_alignment_score", "source_synth_roi_ssim", "label_source_seg_dice",
        "teacher_model", "teacher_lesion_count_diff", "manual_visual_score", "quality_grade", "diffusion_quality_decision",
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
        lines.append(f"| {name} | `{display_path(path, PROJECT_ROOT)}` | {'yes' if path.exists() else 'no'} | {human_size(dir_size(path)) if path.exists() else '0 B'} | {note} |")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Raw BraTS Task1 data root. Alternatively set G2_DATA_ROOT.")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--force", action="store_true", help="Re-scan NIfTI data even if cached CSV files exist.")
    parser.add_argument("--synthetic-run-root", default="", help="Optional G1 synthetic run directory to intake.")
    parser.add_argument("--synthetic-run-id", default="", help="Optional run id override for synthetic intake.")
    parser.add_argument(
        "--generation-mode",
        choices=["auto", "completion", "full_generation"],
        default="auto",
        help="Interpret plain BraTS-MET case folders as completion or full_generation during synthetic intake.",
    )
    args = parser.parse_args()

    if not args.data_root:
        raise SystemExit("missing --data-root. Pass the raw Task1 data root or set G2_DATA_ROOT.")

    data_root = Path(args.data_root).expanduser().resolve()
    train_root = data_root / "MICCAI-LH-BraTS2025-MET-Challenge-Training"
    validation_root = data_root / "Validation"
    corrected_root = data_root / "MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels"
    results_root = Path(args.results_root).expanduser().resolve()
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
        overlay_df, final_df = apply_corrected_labels(raw_df, corrected_root, data_root)
        overlay_df.to_csv(overlay_path, index=False)
        final_df.to_csv(final_path, index=False)
    outputs.extend([overlay_path, final_path])

    write_data_qc_summary(dirs, raw_df, val_df, overlay_df, final_df)
    outputs.append(dirs["reports"] / "real_data_qc_summary.md")

    label_df, lesion_df, lesion_summary = label_stats(final_df, data_root)
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

    mapping_df, _ = nnunet_mapping(final_df, data_root)
    fake_t2w_ids = load_fake_t2w_case_ids(results_root)
    mapping_df.insert(2, "patient_group", mapping_df["source_case_id"].map(patient_group))
    mapping_df.insert(3, "t2w_status", mapping_df["source_case_id"].map(lambda value: "fake_or_broken" if value in fake_t2w_ids else "authentic"))
    mapping_df.insert(4, "eligible_for_realonly", mapping_df["source_case_id"].map(lambda value: value not in fake_t2w_ids))
    mapping_df.insert(5, "completion_required", mapping_df["source_case_id"].map(lambda value: value in fake_t2w_ids))
    master_mapping_path = dirs["manifests"] / "nnunet_case_mapping_master.csv"
    mapping_df.to_csv(master_mapping_path, index=False)
    outputs.append(master_mapping_path)

    realonly_df = mapping_df[mapping_df["eligible_for_realonly"] == True].copy()  # noqa: E712
    mapping_path = dirs["manifests"] / "nnunet_case_mapping_realonly.csv"
    realonly_df.to_csv(mapping_path, index=False)
    outputs.append(mapping_path)
    write_dataset_json(dirs["nnunet_raw"] / "dataset.json")
    outputs.append(dirs["nnunet_raw"] / "dataset.json")

    master_split = create_train_val_test_split(
        mapping_df.to_dict(orient="records"),
        base_split=None,
        val_fraction_of_train_pool=0.10,
        test_fraction=0.10,
        seed="42",
        anchor_case_ids=set(realonly_df["source_case_id"].astype(str)),
    )
    master_split["mapping_csv"] = "manifests/nnunet_case_mapping_master.csv"
    master_split_path = dirs["splits"] / "splits_master_train_val_test.json"
    master_membership_path = dirs["splits"] / "splits_master_train_val_test_membership.csv"
    write_split_outputs(master_split, mapping_df.to_dict(orient="records"), master_split_path, master_membership_path)
    outputs.extend([master_split_path, master_membership_path])

    train_val_test_split = filter_split(
        master_split,
        set(realonly_df["nnunet_case_id"].astype(str)),
        "realonly_patient_group_train_val_test",
    )
    train_val_test_split["mapping_csv"] = "manifests/nnunet_case_mapping_realonly.csv"
    train_val_test_path = dirs["splits"] / "splits_final_train_val_test.json"
    train_val_test_membership_path = dirs["splits"] / "splits_final_train_val_test_membership.csv"
    write_split_outputs(
        train_val_test_split,
        realonly_df.to_dict(orient="records"),
        train_val_test_path,
        train_val_test_membership_path,
    )
    outputs.extend([train_val_test_path, train_val_test_membership_path])

    split_by_nnunet = {
        str(nnunet_id): split_name
        for split_name in ("train", "val", "test")
        for nnunet_id in master_split[split_name]
    }
    v2_source_df = pd.DataFrame({
        "source_case_id": mapping_df["source_case_id"],
        "patient_group": mapping_df["patient_group"],
        "nnunet_case_id": mapping_df["nnunet_case_id"],
        "split": mapping_df["nnunet_case_id"].map(split_by_nnunet),
        "t2w_status": mapping_df["t2w_status"],
        "allowed_as_v2_source": mapping_df.apply(
            lambda row: split_by_nnunet[str(row["nnunet_case_id"])] == "train" and bool(row["eligible_for_realonly"]),
            axis=1,
        ),
        "t1n_path": mapping_df["t1n_source_path"],
        "t1c_path": mapping_df["t1c_source_path"],
        "t2w_path": mapping_df["t2w_source_path"],
        "t2f_path": mapping_df["t2f_source_path"],
        "seg_path": mapping_df["seg_source_path"],
        "label_source": mapping_df["label_source"],
    })
    v2_source_path = dirs["manifests"] / "g1_v2_source_manifest.csv"
    v2_source_df.to_csv(v2_source_path, index=False)
    outputs.append(v2_source_path)

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
                ("synthetic_accepted_evaluation_manifest", [dirs["manifests"] / f"synthetic_accepted_evaluation_manifest_{run_dir_name}.csv"]),
                ("synthetic_pending_review_manifest", [dirs["manifests"] / f"synthetic_pending_review_manifest_{run_dir_name}.csv"]),
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
        "v2_allowed_source_cases": int(v2_source_df["allowed_as_v2_source"].sum()),
        "synthetic_run_root": args.synthetic_run_root,
        "outputs": [str(p) for p in outputs],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
