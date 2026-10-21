#!/usr/bin/env python3
"""Build immutable RC27 and non-RC X3-small8 pools plus train-only statistics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import (  # noqa: E402
    ALLOWED_LABELS,
    ComponentRecord,
    S2_MODALITIES,
    canonical_json_sha256,
    patient_group,
    sha256_file,
)
from custom_nnunet.met_aug_k1245 import (  # noqa: E402
    ANCHOR_LABELS,
    K1245_MANIFEST_SCHEMA,
    LEGACY_ANCHOR_LABELS,
    extract_k1245_components,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--preprocessed-dir", required=True, type=Path)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--mapping-csv", required=True, type=Path)
    parser.add_argument("--preprocessed-coordinate-audit", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    return parser.parse_args()


def _exclusive_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != 1035 or len(values) != len(set(values)):
        raise RuntimeError(f"K1245 requires the frozen unique 1035 train IDs: {path}")
    return values


def _read_mapping(path: Path, train_ids: set[str]) -> tuple[dict[str, str], dict[str, str]]:
    mapping: dict[str, str] = {}
    groups: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"nnunet_case_id", "source_case_id", "patient_group"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise RuntimeError("K1245 mapping fields are incomplete")
        for row in reader:
            case = row["nnunet_case_id"].strip()
            if case not in train_ids:
                continue
            source = row["source_case_id"].strip()
            group = row["patient_group"].strip()
            if group != patient_group(source):
                raise RuntimeError(f"patient-group mismatch for {case}")
            mapping[case], groups[case] = source, group
    missing = sorted(train_ids - set(mapping))
    if missing:
        raise RuntimeError(f"K1245 mapping misses train cases: {missing[:10]}")
    return mapping, groups


def _raw_geometry_and_hashes(dataset_dir: Path, case_id: str) -> tuple[str, dict[str, str], str]:
    label_path = dataset_dir / "labelsTr" / f"{case_id}.nii.gz"
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    label_image = nib.load(str(label_path))
    shape = tuple(int(value) for value in label_image.shape[:3])
    spacing = tuple(float(value) for value in label_image.header.get_zooms()[:3])
    affine_sha = hashlib.sha256(np.asarray(label_image.affine, dtype=np.float64).tobytes()).hexdigest()
    modalities: dict[str, str] = {}
    for index, modality in enumerate(S2_MODALITIES):
        path = dataset_dir / "imagesTr" / f"{case_id}_{index:04d}.nii.gz"
        image = nib.load(str(path))
        if tuple(image.shape[:3]) != shape or not np.allclose(image.header.get_zooms()[:3], spacing):
            raise RuntimeError(f"raw geometry mismatch: {case_id}/{modality}")
        modalities[modality] = sha256_file(path)
    return sha256_file(label_path), modalities, affine_sha


def _load_preprocessed(dataset, case_id: str) -> np.ndarray:
    data, segmentation, previous, _properties = dataset.load_case(case_id)
    if previous is not None or data.ndim != 4 or data.shape[0] != 4:
        raise RuntimeError(f"unsupported preprocessed case: {case_id}")
    if segmentation.ndim != 4 or segmentation.shape[0] != 1 or segmentation.shape[1:] != data.shape[1:]:
        raise RuntimeError(f"preprocessed alignment mismatch: {case_id}")
    label = np.asarray(segmentation[0], dtype=np.int16).copy()
    values = set(int(value) for value in np.unique(label))
    if not values.issubset(ALLOWED_LABELS | {-1}):
        raise RuntimeError(f"invalid preprocessed labels: {case_id}/{sorted(values)}")
    label[label == -1] = 0
    return label


def _lesion_rows(label: np.ndarray) -> list[dict[str, float]]:
    mapping, count = ndimage.label(
        np.isin(label, tuple(ALLOWED_LABELS - {0})), structure=np.ones((3, 3, 3), bool)
    )
    rows = []
    for index in range(1, count + 1):
        support = mapping == index
        points = np.argwhere(support)
        rows.append({
            "volume_mm3": float(np.count_nonzero(support)),
            "centroid_z": float(points[:, 0].mean()),
            "centroid_y": float(points[:, 1].mean()),
            "centroid_x": float(points[:, 2].mean()),
        })
    return sorted(rows, key=lambda row: -row["volume_mm3"])


def _burden_bucket(count: int) -> str:
    return "1" if count <= 1 else "2_3" if count <= 3 else "4_8" if count <= 8 else "9_16" if count <= 16 else "17_24" if count <= 24 else "gt24"


def _bootstrap_mean_ci(values: list[float], rng: np.random.Generator) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    draws = rng.choice(array, size=(20000, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _empirical_contract(case_lesions: dict[str, list[dict[str, float]]]) -> dict:
    buckets: dict[str, list[float]] = defaultdict(list)
    rank_samples: dict[str, list[float]] = {str(rank): [] for rank in range(2, 9)}
    distances: list[float] = []
    target_counts = {}
    for case_id, lesions in case_lesions.items():
        count = len(lesions)
        target_counts[case_id] = count
        buckets[_burden_bucket(count)].append(
            float(sum(row["volume_mm3"] < 27.0 for row in lesions) / max(1, count))
        )
        for rank in range(2, min(8, count) + 1):
            rank_samples[str(rank)].append(lesions[rank - 1]["volume_mm3"])
        if count > 1:
            centroids = np.asarray(
                [[row["centroid_z"], row["centroid_y"], row["centroid_x"]] for row in lesions]
            )
            matrix = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=2)
            matrix[matrix == 0] = np.inf
            distances.extend(float(value) for value in matrix.min(axis=1))
    keys = ("1", "2_3", "4_8", "9_16", "17_24", "gt24")
    if any(not buckets[key] for key in keys):
        raise RuntimeError("F burden contract has an empty train-only bucket")
    raw = np.asarray([np.mean(buckets[key]) for key in keys], dtype=np.float64)
    monotonic = bool(np.all(np.diff(raw) >= -1e-12))
    normalized = raw / float(np.mean([raw[keys.index(_burden_bucket(v))] for v in target_counts.values()]))
    low, high = 0.0, 10.0
    for _ in range(200):
        middle = (low + high) / 2.0
        mean = np.mean([
            min(0.60, 0.20 * middle * normalized[keys.index(_burden_bucket(v))])
            for v in target_counts.values()
        ])
        if mean < 0.20:
            low = middle
        else:
            high = middle
    factor = (low + high) / 2.0
    probabilities = {
        key: float(min(0.60, 0.20 * factor * normalized[index]))
        for index, key in enumerate(keys)
    }
    rng = np.random.default_rng(20260921)
    ci = {key: _bootstrap_mean_ci(buckets[key], rng) for key in keys}
    ci_separated = all(ci[left][1] <= ci[right][0] for left, right in zip(keys, keys[1:]))
    return {
        "target_counts": dict(sorted(target_counts.items())),
        "burden_small_fraction_mean": {key: float(raw[index]) for index, key in enumerate(keys)},
        "burden_small_fraction_ci95": {key: list(ci[key]) for key in keys},
        "burden_monotonic_increasing": monotonic,
        "adjacent_ci_nonoverlap": ci_separated,
        "f_gate_pass": monotonic and ci_separated,
        "burden_probabilities": probabilities,
        "mean_burden_probability": float(np.mean([probabilities[_burden_bucket(v)] for v in target_counts.values()])),
        "rank_volume_samples": {key: values for key, values in rank_samples.items() if values},
        "nn_distance_samples_mm": distances,
    }


def _write_pool(
    *,
    output: Path,
    records: list[ComponentRecord],
    groups: dict[str, str],
    constraints: dict,
    common: dict,
) -> Path:
    if output.exists():
        raise FileExistsError(f"immutable K1245 pool already exists: {output}")
    (output / "components").mkdir(parents=True)
    for record in records:
        source = Path(record.component_path)
        if not source.is_absolute():
            raise RuntimeError("internal K1245 component staging path is not absolute")
        destination = output / "components" / source.name
        source.replace(destination)
        object.__setattr__(record, "component_path", str(Path("components") / source.name))
        object.__setattr__(record, "label_sha256", sha256_file(destination))
    records_path = output / "components.jsonl"
    records_path.write_text(
        "".join(json.dumps(record.as_mapping(), ensure_ascii=True, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    groups_path = output / "target_case_groups.json"
    groups_path.write_text(
        json.dumps({"schema_version": 1, "case_to_patient_group": dict(sorted(groups.items()))}, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload = {
        **common,
        "schema_version": K1245_MANIFEST_SCHEMA,
        "component_count": len(records),
        "records_file": records_path.name,
        "records_sha256": sha256_file(records_path),
        "target_groups_file": groups_path.name,
        "target_groups_sha256": sha256_file(groups_path),
        "constraints": constraints,
    }
    payload["manifest_sha256"] = canonical_json_sha256(payload, exclude=("manifest_sha256",))
    path = output / "component_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    output = args.output_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"K1245 pool root already exists: {output}")
    plan_sha = sha256_file(args.plan)
    train_ids = _read_ids(args.train_file)
    mapping, groups = _read_mapping(args.mapping_csv, set(train_ids))
    coordinate = json.loads(args.preprocessed_coordinate_audit.read_text(encoding="utf-8"))
    if coordinate.get("status") != "pass" or coordinate.get("configuration_spacing_mm") != [1.0, 1.0, 1.0]:
        raise RuntimeError("K1245 preprocessed coordinate audit is not passing 1 mm")
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    dataset_class = infer_dataset_class(str(args.preprocessed_dir))
    dataset = dataset_class(str(args.preprocessed_dir), train_ids)
    staging = output.parent / f".{output.name}.staging"
    if staging.exists():
        raise FileExistsError(f"K1245 staging root already exists: {staging}")
    staging.mkdir(parents=True)
    pool_records: dict[str, list[ComponentRecord]] = {"rc27": [], "small8": []}
    exclusions = {"rc27": Counter(), "small8": Counter()}
    case_lesions: dict[str, list[dict[str, float]]] = {}
    try:
        for case_id in train_ids:
            label = _load_preprocessed(dataset, case_id)
            case_lesions[case_id] = _lesion_rows(label)
            source_label_sha, modality_sha, affine_sha = _raw_geometry_and_hashes(args.dataset_dir, case_id)
            for token, include_rc, floor in (("rc27", True, 27.0), ("small8", False, 8.0)):
                payloads, dropped = extract_k1245_components(
                    label, (1.0, 1.0, 1.0), min_anchor_volume_mm3=floor,
                    max_bbox_mm=56.0, include_rc=include_rc,
                )
                exclusions[token].update(dropped)
                token_dir = staging / token
                token_dir.mkdir(exist_ok=True)
                for index, component in enumerate(payloads, start=1):
                    component_id = hashlib.sha256(
                        f"k1245_{token}_20260921|{case_id}|{index}".encode()
                    ).hexdigest()[:24]
                    path = token_dir / f"{component_id}.npz"
                    np.savez_compressed(path, label=component["label"])
                    stats = component["stats"]
                    pool_records[token].append(ComponentRecord(
                        component_id=component_id,
                        manifest_version=f"k1245_{token}_20260921_r1",
                        source_case_id=mapping[case_id],
                        patient_group=groups[case_id],
                        split="train",
                        component_path=str(path),
                        label_sha256=sha256_file(path),
                        source_label_sha256=source_label_sha,
                        source_modalities_sha256=modality_sha,
                        source_affine_sha256=affine_sha,
                        spacing_mm=(1.0, 1.0, 1.0),
                        core_volume_mm3=float(stats["core_volume_mm3"]),
                        total_volume_mm3=float(stats["total_volume_mm3"]),
                        bbox_mm=tuple(stats["bbox_mm"]),
                        bbox_voxels=tuple(stats["bbox_voxels"]),
                        class_counts=stats["class_counts"],
                        classes_present=tuple(stats["classes_present"]),
                        core_centroid_norm=tuple(component["core_centroid_norm"]),
                    ))
        output.mkdir(parents=True)
        common = {
            "artifact_status": "experimental_unvalidated",
            "operator_approved": False,
            "formal_gate_status": "not_run_not_passed",
            "coordinate_space": "nnUNetPlans_3d_fullres_preprocessed",
            "plan_sha256": plan_sha,
            "builder_sha256": sha256_file(Path(__file__)),
            "train_file_sha256": sha256_file(args.train_file),
            "mapping_csv_sha256": sha256_file(args.mapping_csv),
            "coordinate_audit_sha256": sha256_file(args.preprocessed_coordinate_audit),
            "train_count": len(train_ids),
        }
        manifests = {}
        for token, include_rc, floor in (("rc27", True, 27.0), ("small8", False, 8.0)):
            manifests[token] = _write_pool(
                output=output / token,
                records=pool_records[token],
                groups=groups,
                constraints={
                    "anchor_labels": sorted(ANCHOR_LABELS if include_rc else LEGACY_ANCHOR_LABELS),
                    "include_rc": include_rc,
                    "min_anchor_volume_mm3": floor,
                    "max_bbox_mm": 56.0,
                    "crop_size": 64,
                    "snfh_policy": "attach_only_if_adjacent_to_one_anchor",
                },
                common={**common, "exclusions": dict(sorted(exclusions[token].items()))},
            )
        empirical = _empirical_contract(case_lesions)
        pure_et = [r.core_volume_mm3 for r in pool_records["rc27"] if r.classes_present == (3,)]
        snfh_et = [r.total_volume_mm3 for r in pool_records["rc27"] if r.classes_present == (2, 3)]
        if not pure_et or not snfh_et:
            raise RuntimeError("K1245 cannot derive E shell contract from train-only pool")
        empirical.update({
            "e_shell_factor": float(max(1e-6, (np.median(snfh_et) / np.median(pure_et)) ** (1.0 / 3.0) - 1.0)),
            "e_volume_p10_mm3": float(np.quantile(snfh_et, 0.10)),
            "e_volume_p90_mm3": float(np.quantile(snfh_et, 0.90)),
        })
        rc_count = sum(4 in record.classes_present for record in pool_records["rc27"])
        if rc_count < 24:
            raise RuntimeError(f"K1245 requires at least 24 eligible RC donors, observed {rc_count}")
        evidence = {
            "artifact_status": "experimental_unvalidated",
            "operator_approved": False,
            "formal_gate_status": "not_run_not_passed",
            "status": "pass" if empirical["f_gate_pass"] else "blocked_f_gate",
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "plan_sha256": plan_sha,
            "manifests": {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in manifests.items()},
            "component_counts": {key: len(value) for key, value in pool_records.items()},
            "rc_component_count": rc_count,
            "empirical": empirical,
        }
        _exclusive_json(output / "K1245_POOL_AND_EMPIRICAL_VALIDATION.json", evidence)
    finally:
        if staging.exists():
            for directory in staging.iterdir():
                if directory.is_dir() and not any(directory.iterdir()):
                    directory.rmdir()
            if not any(staging.iterdir()):
                staging.rmdir()


if __name__ == "__main__":
    main()
