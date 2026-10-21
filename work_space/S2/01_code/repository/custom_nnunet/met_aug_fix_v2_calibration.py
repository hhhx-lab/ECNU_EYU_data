"""Train-only empirical measurements used to freeze the MET-AUG Fix-v2 QC."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from scipy import ndimage

from .met_aug_core import (
    ComponentManifest,
    S2_MODALITIES,
    canonical_json_sha256,
    extract_met_components,
    make_fix_v2_route_a_config,
    sha256_file,
)
from .met_aug_fix_v2 import (
    LABEL_SEMANTICS,
    _extract_boundary_faces,
    _mad,
    _mask_alignment,
    _weighted_quantile,
)


REFERENCE_EVIDENCE_SCHEMA = 1
REFERENCE_RING_INNER_MM = 1.0
REFERENCE_RING_OUTER_MM = 6.0
REFERENCE_MINIMUM_VOXELS = 64
REFERENCE_CDF_POINTS = 129
REFERENCE_EFFECT_POINTS = 33
REFERENCE_CONNECTIVITY = "26_connected_core_and_snfh_attachment_v1"
REFERENCE_SPACING_RULE = "nnUNetPlans_3d_fullres_1mm_physical_faces_v1"
PAIR_KEYS = tuple(
    f"{left}:{right}"
    for index, left in enumerate(S2_MODALITIES)
    for right in S2_MODALITIES[index + 1 :]
)


@dataclass(frozen=True)
class ReferenceCase:
    case_id: str
    patient_group: str
    image: np.ndarray
    segmentation: np.ndarray
    valid_mask: np.ndarray
    spacing_mm: tuple[float, float, float]


def _strict_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("reference evidence contains a non-finite float")
    return value


def normalize_preprocessed_segmentation(segmentation: np.ndarray) -> np.ndarray:
    result = np.asarray(segmentation, dtype=np.int16).copy()
    if result.ndim == 4 and result.shape[0] == 1:
        result = result[0]
    if result.ndim != 3:
        raise ValueError(f"preprocessed segmentation must be 3D, got {result.shape}")
    values = set(int(value) for value in np.unique(result))
    if not values.issubset({-1, 0, 1, 2, 3, 4}):
        raise ValueError(f"preprocessed segmentation has invalid labels: {sorted(values)}")
    result[result == -1] = 0
    return result


def expected_component_id(manifest_version: str, case_id: str, index: int) -> str:
    return hashlib.sha256(
        f"{manifest_version}|{case_id}|{index}".encode("utf-8")
    ).hexdigest()[:24]


def component_instances(
    segmentation: np.ndarray,
    *,
    case_id: str,
    manifest_version: str,
    spacing_mm: tuple[float, float, float],
) -> list[tuple[str, np.ndarray, Mapping[str, Any]]]:
    payloads, _dropped = extract_met_components(
        segmentation,
        spacing_mm,
        min_core_volume_mm3=27.0,
        max_bbox_mm=56.0,
    )
    result: list[tuple[str, np.ndarray, Mapping[str, Any]]] = []
    for index, payload in enumerate(payloads, start=1):
        label = np.zeros(segmentation.shape, dtype=np.int16)
        bbox = tuple(slice(start, stop) for start, stop in payload["source_bbox_voxels"])
        label[bbox] = np.asarray(payload["label"], dtype=np.int16)
        result.append(
            (
                expected_component_id(manifest_version, case_id, index),
                label,
                payload["stats"],
            )
        )
    return result


def reference_ring(
    *,
    component_support: np.ndarray,
    segmentation: np.ndarray,
    valid_mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    inner_mm: float = REFERENCE_RING_INNER_MM,
    outer_mm: float = REFERENCE_RING_OUTER_MM,
) -> np.ndarray:
    distance = ndimage.distance_transform_edt(~component_support, sampling=spacing_mm)
    return (
        ~component_support
        & np.asarray(valid_mask, dtype=bool)
        & (segmentation == 0)
        & (distance >= float(inner_mm))
        & (distance <= float(outer_mm))
    )


def compressed_weighted_sample(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    points: int = REFERENCE_CDF_POINTS,
) -> tuple[list[float], list[float]]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape or values.size == 0:
        raise ValueError("weighted sample must be matching nonempty vectors")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError("weighted sample must be finite")
    if np.any(weights <= 0) or points < 2:
        raise ValueError("weighted sample requires positive weights and at least two points")
    probabilities = (np.arange(points, dtype=np.float64) + 0.5) / points
    compressed = _weighted_quantile(values, weights, probabilities)
    per_point = float(np.sum(weights) / points)
    return compressed.tolist(), [per_point] * points


def _effect_summary(values: np.ndarray) -> list[float]:
    probabilities = (np.arange(REFERENCE_EFFECT_POINTS, dtype=np.float64) + 0.5) / REFERENCE_EFFECT_POINTS
    return np.quantile(values, probabilities, method="linear").astype(np.float64).tolist()


def measure_reference_component(
    case: ReferenceCase,
    *,
    component_id: str,
    label: np.ndarray,
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    support = label != 0
    ring = reference_ring(
        component_support=support,
        segmentation=case.segmentation,
        valid_mask=case.valid_mask,
        spacing_mm=case.spacing_mm,
    )
    ring_voxels = int(np.count_nonzero(ring))
    if ring_voxels < REFERENCE_MINIMUM_VOXELS:
        raise ValueError(f"reference ring has only {ring_voxels} voxels")
    scales: dict[str, float] = {}
    medians: dict[str, float] = {}
    for channel, modality in enumerate(S2_MODALITIES):
        ring_values = case.image[channel][ring].astype(np.float64)
        scale = _mad(ring_values)
        if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
            raise ValueError(f"{modality} reference MAD is zero or non-finite")
        scales[modality] = float(scale)
        medians[modality] = float(np.median(ring_values))

    boundary: dict[str, Any] = {}
    effects: dict[str, Any] = {}
    for label_value in (1, 2, 3):
        label_support = label == label_value
        if not np.any(label_support):
            effects[str(label_value)] = {"status": "not_present"}
            for modality in S2_MODALITIES:
                boundary[f"{label_value}:{modality}"] = {"status": "not_present"}
            continue
        modality_effects: dict[str, Any] = {}
        for channel, modality in enumerate(S2_MODALITIES):
            normalized = (
                case.image[channel].astype(np.float64) - medians[modality]
            ) / scales[modality]
            effect_values = normalized[label_support]
            modality_effects[modality] = {
                "median_contrast": float(np.median(effect_values)),
                "abs_effect_sample": _effect_summary(np.abs(effect_values)),
            }
            faces = _extract_boundary_faces(
                label_cube=label,
                image=case.image[channel],
                label_value=label_value,
                scale=scales[modality],
                spacing_mm=case.spacing_mm,
            )
            if faces.signed.size == 0:
                boundary[f"{label_value}:{modality}"] = {"status": "not_present"}
                continue
            signed, signed_weights = compressed_weighted_sample(
                faces.signed, faces.weights
            )
            absolute, absolute_weights = compressed_weighted_sample(
                faces.absolute, faces.weights
            )
            boundary[f"{label_value}:{modality}"] = {
                "status": "measured",
                "face_count": int(faces.signed.size),
                "area_mm2": float(np.sum(faces.weights)),
                "signed_values": signed,
                "signed_weights": signed_weights,
                "abs_values": absolute,
                "abs_weights": absolute_weights,
            }
        effects[str(label_value)] = {
            "status": "measured",
            "voxel_count": int(np.count_nonzero(label_support)),
            "modalities": modality_effects,
            "pairwise": {},
        }
    return {
        "component_id": component_id,
        "case_id": case.case_id,
        "patient_group": case.patient_group,
        "core_volume_mm3": float(stats["core_volume_mm3"]),
        "total_volume_mm3": float(stats["total_volume_mm3"]),
        "classes_present": list(stats["classes_present"]),
        "class_counts": dict(stats["class_counts"]),
        "reference_ring_voxels": ring_voxels,
        "reference_median": medians,
        "reference_mad": scales,
        "boundary": boundary,
        "effects": effects,
    }


def derive_affected_thresholds(component_rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in component_rows:
        for label_value, effect in row["effects"].items():
            if effect.get("status") != "measured":
                continue
            for modality, values in effect["modalities"].items():
                samples[(str(label_value), str(modality))].extend(
                    float(value) for value in values["abs_effect_sample"]
                )
    result: dict[str, dict[str, float]] = {str(value): {} for value in (1, 2, 3)}
    for label_value in result:
        for modality in S2_MODALITIES:
            values = np.asarray(samples[(label_value, modality)], dtype=np.float64)
            if values.size < 2:
                raise ValueError(f"Reference lacks affected-effect support for {label_value}:{modality}")
            result[label_value][modality] = float(
                max(np.finfo(np.float64).eps, np.quantile(values, 0.25, method="linear"))
            )
    return result


def attach_pairwise_effect_metrics(
    row: dict[str, Any],
    *,
    case: ReferenceCase,
    label: np.ndarray,
    affected_thresholds: Mapping[str, Mapping[str, float]],
) -> None:
    for label_value in (1, 2, 3):
        effect = row["effects"][str(label_value)]
        if effect.get("status") != "measured":
            continue
        support = label == label_value
        masks: dict[str, np.ndarray] = {}
        for channel, modality in enumerate(S2_MODALITIES):
            normalized = (
                case.image[channel].astype(np.float64)
                - float(row["reference_median"][modality])
            ) / float(row["reference_mad"][modality])
            masks[modality] = support & (
                np.abs(normalized) >= float(affected_thresholds[str(label_value)][modality])
            )
            effect["modalities"][modality]["affected_fraction"] = float(
                np.count_nonzero(masks[modality]) / np.count_nonzero(support)
            )
        for pair in PAIR_KEYS:
            left, right = pair.split(":", 1)
            iou, centroid = _mask_alignment(masks[left], masks[right], case.spacing_mm)
            effect["pairwise"][pair] = {
                "iou": iou,
                "centroid_distance_mm": centroid,
            }


def load_partition(path: str | Path) -> dict[str, Any]:
    partition_path = Path(path).expanduser().resolve()
    payload = json.loads(partition_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("status") != "pass":
        raise ValueError("Fix-v2 partition is not a passing schema-1 artifact")
    if payload.get("partition_audit_sha256") != canonical_json_sha256(
        payload, exclude=("partition_audit_sha256",)
    ):
        raise ValueError("Fix-v2 partition audit SHA256 mismatch")
    return payload


def validate_reference_evidence(
    reference: Mapping[str, Any],
    *,
    reference_path: str | Path,
    partition_path: str | Path,
    manifest: ComponentManifest,
    expected_valid_mask_manifest_sha256: str | None = None,
    expected_preprocessed_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Strictly validate a completed Reference artifact before calibration use."""
    reference_file = Path(reference_path).expanduser().resolve()
    partition_file = Path(partition_path).expanduser().resolve()
    partition = load_partition(partition_file)
    if reference.get("schema_version") != REFERENCE_EVIDENCE_SCHEMA:
        raise ValueError("Reference evidence schema drifted")
    if reference.get("status") != "pass" or reference.get("source_partition") != "reference":
        raise ValueError("Reference evidence is not a passing Reference artifact")
    if reference.get("reference_cdf_audit_sha256") != canonical_json_sha256(
        reference, exclude=("reference_cdf_audit_sha256",)
    ):
        raise ValueError("Reference evidence audit SHA256 drifted")
    expected_bindings = {
        "partition_sha256": sha256_file(partition_file),
        "partition_audit_sha256": partition["partition_audit_sha256"],
        "component_manifest_sha256": manifest.identity_sha256,
        "target_groups_sha256": manifest.target_groups_sha256,
    }
    if expected_valid_mask_manifest_sha256 is not None:
        expected_bindings["valid_mask_manifest_sha256"] = (
            expected_valid_mask_manifest_sha256
        )
    if expected_preprocessed_contract_sha256 is not None:
        expected_bindings["preprocessed_contract_sha256"] = (
            expected_preprocessed_contract_sha256
        )
    for key, expected in expected_bindings.items():
        if reference.get(key) != expected:
            raise ValueError(f"Reference evidence {key} drifted")
    for key in ("valid_mask_manifest_sha256", "preprocessed_contract_sha256"):
        value = str(reference.get(key, ""))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"Reference evidence {key} is not a SHA256")

    expected_members = {
        "patient_groups": partition["partitions"]["reference"],
        "target_case_ids": partition["target_case_ids"]["reference"],
        "component_ids": partition["component_ids"]["reference"],
    }
    for key, expected in expected_members.items():
        if reference.get(key) != expected:
            raise ValueError(f"Reference evidence {key} drifted")
    for count_key, member_key in (
        ("patient_group_count", "patient_groups"),
        ("target_case_count", "target_case_ids"),
        ("component_count", "component_ids"),
    ):
        if int(reference.get(count_key, -1)) != len(expected_members[member_key]):
            raise ValueError(f"Reference evidence {count_key} drifted")

    definitions = reference.get("definitions")
    expected_definitions = {
        "modality_order": list(S2_MODALITIES),
        "label_semantics": LABEL_SEMANTICS,
        "connectivity": REFERENCE_CONNECTIVITY,
        "spacing_rule": REFERENCE_SPACING_RULE,
        "reference_ring_inner_mm": REFERENCE_RING_INNER_MM,
        "reference_ring_outer_mm": REFERENCE_RING_OUTER_MM,
        "minimum_reference_voxels": REFERENCE_MINIMUM_VOXELS,
        "cdf_compression": f"weighted_midpoint_quantiles_{REFERENCE_CDF_POINTS}",
        "patient_cluster_weighting": "equal_patient_then_equal_component_v1",
    }
    if definitions != expected_definitions:
        raise ValueError("Reference evidence measurement definitions drifted")

    affected = reference.get("affected_abs_threshold")
    if not isinstance(affected, Mapping) or set(affected) != {"1", "2", "3"}:
        raise ValueError("Reference affected-threshold classes drifted")
    for label_value, modalities in affected.items():
        if not isinstance(modalities, Mapping) or set(modalities) != set(S2_MODALITIES):
            raise ValueError(f"Reference affected thresholds drifted for label {label_value}")
        if any(not np.isfinite(float(value)) or float(value) <= 0 for value in modalities.values()):
            raise ValueError(f"Reference affected thresholds are invalid for label {label_value}")

    expected_component_ids = set(expected_members["component_ids"])
    expected_records = {
        record.component_id: record
        for record in manifest.records
        if record.component_id in expected_component_ids
    }
    if set(expected_records) != expected_component_ids:
        raise ValueError("Reference partition contains unknown manifest components")
    components = reference.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("Reference evidence has no usable components")
    measured_ids: set[str] = set()
    measured_groups: set[str] = set()
    boundary_keys = {
        f"{label_value}:{modality}"
        for label_value in (1, 2, 3)
        for modality in S2_MODALITIES
    }
    for row in components:
        if not isinstance(row, Mapping):
            raise ValueError("Reference component row is malformed")
        component_id = str(row.get("component_id", ""))
        if component_id in measured_ids or component_id not in expected_records:
            raise ValueError(f"Reference component identity is invalid: {component_id}")
        record = expected_records[component_id]
        target_case_id = str(row.get("case_id", ""))
        if target_case_id not in expected_members["target_case_ids"]:
            raise ValueError(f"{component_id}: target case escaped Reference")
        if manifest.target_groups.get(target_case_id) != record.patient_group:
            raise ValueError(f"{component_id}: target/source patient mapping drifted")
        if row.get("patient_group") != record.patient_group:
            raise ValueError(f"{component_id}: patient group drifted")
        if record.patient_group not in set(expected_members["patient_groups"]):
            raise ValueError(f"{component_id}: patient group escaped Reference")
        if list(row.get("classes_present", ())) != list(record.classes_present):
            raise ValueError(f"{component_id}: classes drifted")
        if dict(row.get("class_counts", {})) != dict(record.class_counts):
            raise ValueError(f"{component_id}: class counts drifted")
        for key, expected in (
            ("core_volume_mm3", record.core_volume_mm3),
            ("total_volume_mm3", record.total_volume_mm3),
        ):
            observed = float(row.get(key, np.nan))
            if not np.isfinite(observed) or not np.isclose(observed, expected):
                raise ValueError(f"{component_id}: {key} drifted")
        if int(row.get("reference_ring_voxels", 0)) < REFERENCE_MINIMUM_VOXELS:
            raise ValueError(f"{component_id}: reference ring is undersized")
        medians = row.get("reference_median")
        mads = row.get("reference_mad")
        if not isinstance(medians, Mapping) or set(medians) != set(S2_MODALITIES):
            raise ValueError(f"{component_id}: reference medians drifted")
        if not isinstance(mads, Mapping) or set(mads) != set(S2_MODALITIES):
            raise ValueError(f"{component_id}: reference MADs drifted")
        if any(not np.isfinite(float(value)) for value in medians.values()):
            raise ValueError(f"{component_id}: reference median is non-finite")
        if any(not np.isfinite(float(value)) or float(value) <= 0 for value in mads.values()):
            raise ValueError(f"{component_id}: reference MAD is invalid")

        boundary = row.get("boundary")
        if not isinstance(boundary, Mapping) or set(boundary) != boundary_keys:
            raise ValueError(f"{component_id}: boundary strata drifted")
        for key, measurement in boundary.items():
            if not isinstance(measurement, Mapping):
                raise ValueError(f"{component_id}: boundary {key} is malformed")
            status = measurement.get("status")
            if status == "not_present":
                continue
            if status != "measured":
                raise ValueError(f"{component_id}: boundary {key} status drifted")
            area = float(measurement.get("area_mm2", np.nan))
            face_count = int(measurement.get("face_count", 0))
            if not np.isfinite(area) or area <= 0 or face_count <= 0:
                raise ValueError(f"{component_id}: boundary {key} geometry is invalid")
            for prefix in ("signed", "abs"):
                values = np.asarray(measurement.get(f"{prefix}_values", ()), dtype=np.float64)
                weights = np.asarray(measurement.get(f"{prefix}_weights", ()), dtype=np.float64)
                if (
                    values.shape != (REFERENCE_CDF_POINTS,)
                    or weights.shape != values.shape
                    or not np.all(np.isfinite(values))
                    or not np.all(np.isfinite(weights))
                    or np.any(weights <= 0)
                    or not np.isclose(float(np.sum(weights)), area)
                ):
                    raise ValueError(f"{component_id}: boundary {key} {prefix} CDF is invalid")
                if prefix == "abs" and np.any(values < 0):
                    raise ValueError(f"{component_id}: boundary {key} absolute CDF is negative")

        effects = row.get("effects")
        if not isinstance(effects, Mapping) or set(effects) != {"1", "2", "3"}:
            raise ValueError(f"{component_id}: cross-modal classes drifted")
        for label_value, effect in effects.items():
            if not isinstance(effect, Mapping):
                raise ValueError(f"{component_id}: effect {label_value} is malformed")
            status = effect.get("status")
            if status == "not_present":
                continue
            if status != "measured" or int(effect.get("voxel_count", 0)) <= 0:
                raise ValueError(f"{component_id}: effect {label_value} status drifted")
            modalities = effect.get("modalities")
            if not isinstance(modalities, Mapping) or set(modalities) != set(S2_MODALITIES):
                raise ValueError(f"{component_id}: effect {label_value} modalities drifted")
            for modality, measurement in modalities.items():
                sample = np.asarray(measurement.get("abs_effect_sample", ()), dtype=np.float64)
                values = (
                    float(measurement.get("median_contrast", np.nan)),
                    float(measurement.get("affected_fraction", np.nan)),
                )
                if (
                    sample.shape != (REFERENCE_EFFECT_POINTS,)
                    or not np.all(np.isfinite(sample))
                    or np.any(sample < 0)
                    or not all(np.isfinite(value) for value in values)
                    or not 0.0 <= values[1] <= 1.0
                ):
                    raise ValueError(
                        f"{component_id}: effect {label_value}:{modality} is invalid"
                    )
            pairwise = effect.get("pairwise")
            if not isinstance(pairwise, Mapping) or set(pairwise) != set(PAIR_KEYS):
                raise ValueError(f"{component_id}: effect {label_value} pairs drifted")
            for pair, measurement in pairwise.items():
                iou = float(measurement.get("iou", np.nan))
                centroid = float(measurement.get("centroid_distance_mm", np.nan))
                if not np.isfinite(iou) or not 0.0 <= iou <= 1.0:
                    raise ValueError(f"{component_id}: pair {pair} IoU is invalid")
                if not np.isfinite(centroid) or centroid < 0:
                    raise ValueError(f"{component_id}: pair {pair} centroid is invalid")
        measured_ids.add(component_id)
        measured_groups.add(record.patient_group)

    usable_count = len(measured_ids)
    excluded_count = len(expected_component_ids - measured_ids)
    exclusions = reference.get("exclusions")
    if not isinstance(exclusions, Mapping) or any(int(value) <= 0 for value in exclusions.values()):
        raise ValueError("Reference exclusions are malformed")
    if sum(int(value) for value in exclusions.values()) != excluded_count:
        raise ValueError("Reference exclusion reasons do not balance")
    expected_counts = {
        "usable_component_count": usable_count,
        "excluded_component_count": excluded_count,
        "usable_patient_group_count": len(measured_groups),
    }
    for key, expected in expected_counts.items():
        if int(reference.get(key, -1)) != expected:
            raise ValueError(f"Reference evidence {key} does not balance")
    return {
        "status": "pass",
        "reference_file_sha256": sha256_file(reference_file),
        "reference_cdf_audit_sha256": reference["reference_cdf_audit_sha256"],
        **expected_counts,
    }


def build_reference_evidence(
    *,
    manifest: ComponentManifest,
    partition_path: str | Path,
    valid_mask_manifest_sha256: str,
    preprocessed_contract_sha256: str,
    case_loader: Callable[[str, str], ReferenceCase],
    workers: int = 1,
) -> dict[str, Any]:
    if int(workers) <= 0:
        raise ValueError("Reference worker count must be positive")
    partition_file = Path(partition_path).expanduser().resolve()
    partition = load_partition(partition_file)
    if partition["component_manifest_sha256"] != manifest.identity_sha256:
        raise ValueError("partition and component manifest identities differ")
    if partition["target_groups_sha256"] != manifest.target_groups_sha256:
        raise ValueError("partition and target group identities differ")
    expected_ids = set(str(value) for value in partition["component_ids"]["reference"])
    expected_records = {
        record.component_id: record
        for record in manifest.records
        if record.component_id in expected_ids
    }
    if set(expected_records) != expected_ids:
        raise ValueError("Reference partition contains unknown component IDs")

    manifest_version = manifest.records[0].manifest_version

    def measure_case(case_id_value: str) -> tuple[list[dict[str, Any]], Counter[str], set[str]]:
        case_id = str(case_id_value)
        patient_group = manifest.target_groups[str(case_id)]
        case = case_loader(str(case_id), patient_group)
        if case.image.ndim != 4 or case.image.shape[0] != 4:
            raise ValueError(f"{case_id}: expected a four-channel image")
        if case.segmentation.shape != case.image.shape[1:]:
            raise ValueError(f"{case_id}: image/segmentation shapes differ")
        if case.valid_mask.shape != case.segmentation.shape:
            raise ValueError(f"{case_id}: valid-mask shape differs")
        if not np.all(np.isfinite(case.image)):
            raise ValueError(f"{case_id}: image contains non-finite values")
        case_rows: list[dict[str, Any]] = []
        case_excluded: Counter[str] = Counter()
        case_observed_ids: set[str] = set()
        for component_id, label, stats in component_instances(
            case.segmentation,
            case_id=str(case_id),
            manifest_version=manifest_version,
            spacing_mm=case.spacing_mm,
        ):
            if component_id not in expected_ids:
                raise ValueError(f"{case_id}: rebuilt an unexpected component {component_id}")
            record = expected_records[component_id]
            if not np.allclose(record.spacing_mm, case.spacing_mm, atol=1e-6):
                raise ValueError(f"{component_id}: component spacing drifted")
            if tuple(stats["classes_present"]) != record.classes_present:
                raise ValueError(f"{component_id}: class composition drifted")
            if not np.isclose(float(stats["core_volume_mm3"]), record.core_volume_mm3):
                raise ValueError(f"{component_id}: core volume drifted")
            if component_id in case_observed_ids:
                raise ValueError(f"{case_id}: rebuilt duplicate component {component_id}")
            case_observed_ids.add(component_id)
            try:
                row = measure_reference_component(
                    case,
                    component_id=component_id,
                    label=label,
                    stats=stats,
                )
            except ValueError as exc:
                case_excluded[str(exc)] += 1
                continue
            case_rows.append(row)
        return case_rows, case_excluded, case_observed_ids

    case_ids = [str(value) for value in partition["target_case_ids"]["reference"]]
    if int(workers) == 1:
        first_pass = map(measure_case, case_ids)
        first_results = list(first_pass)
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            first_results = list(executor.map(measure_case, case_ids))
    rows: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    observed_ids: set[str] = set()
    for case_rows, case_excluded, case_observed_ids in first_results:
        overlap = observed_ids & case_observed_ids
        if overlap:
            raise RuntimeError(f"Reference rebuilt duplicate components: {sorted(overlap)[:10]}")
        rows.extend(case_rows)
        excluded.update(case_excluded)
        observed_ids.update(case_observed_ids)
    if observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids)
        extra = sorted(observed_ids - expected_ids)
        raise RuntimeError(
            f"Reference component accounting drifted: missing={missing[:10]}, extra={extra[:10]}"
        )
    if not rows:
        raise RuntimeError("Reference extraction produced no usable components")
    measured_ids = {str(row["component_id"]) for row in rows}

    affected_thresholds = derive_affected_thresholds(rows)
    # A second streaming pass computes affected-mask geometry after the global
    # train-only thresholds are known. Keeping full 3D cases/components from the
    # first pass would otherwise consume tens of GiB.
    row_by_component = {str(row["component_id"]): row for row in rows}
    def attach_case_pairwise(case_id_value: str) -> set[str]:
        case_id = str(case_id_value)
        patient_group = manifest.target_groups[str(case_id)]
        case = case_loader(str(case_id), patient_group)
        completed: set[str] = set()
        for component_id, label, _stats in component_instances(
            case.segmentation,
            case_id=str(case_id),
            manifest_version=manifest_version,
            spacing_mm=case.spacing_mm,
        ):
            row = row_by_component.get(component_id)
            if row is None:
                continue
            attach_pairwise_effect_metrics(
                row,
                case=case,
                label=label,
                affected_thresholds=affected_thresholds,
            )
            completed.add(component_id)
        return completed

    if int(workers) == 1:
        second_results = list(map(attach_case_pairwise, case_ids))
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            second_results = list(executor.map(attach_case_pairwise, case_ids))
    pairwise_completed: set[str] = set()
    for completed in second_results:
        overlap = pairwise_completed & completed
        if overlap:
            raise RuntimeError(f"Reference pairwise duplicated components: {sorted(overlap)[:10]}")
        pairwise_completed.update(completed)
    if pairwise_completed != measured_ids:
        missing = sorted(measured_ids - pairwise_completed)
        raise RuntimeError(f"Reference pairwise pass missed components: {missing[:10]}")
    rows.sort(key=lambda value: value["component_id"])
    measured_groups = sorted({str(row["patient_group"]) for row in rows})
    payload: dict[str, Any] = {
        "schema_version": REFERENCE_EVIDENCE_SCHEMA,
        "status": "pass",
        "source_partition": "reference",
        "partition_sha256": sha256_file(partition_file),
        "partition_audit_sha256": partition["partition_audit_sha256"],
        "component_manifest_sha256": manifest.identity_sha256,
        "target_groups_sha256": manifest.target_groups_sha256,
        "valid_mask_manifest_sha256": valid_mask_manifest_sha256,
        "preprocessed_contract_sha256": preprocessed_contract_sha256,
        "patient_groups": partition["partitions"]["reference"],
        "target_case_ids": partition["target_case_ids"]["reference"],
        "component_ids": partition["component_ids"]["reference"],
        "patient_group_count": len(partition["partitions"]["reference"]),
        "target_case_count": len(partition["target_case_ids"]["reference"]),
        "component_count": len(partition["component_ids"]["reference"]),
        "usable_component_count": len(rows),
        "usable_patient_group_count": len(measured_groups),
        "excluded_component_count": len(expected_ids) - len(rows),
        "exclusions": dict(sorted(excluded.items())),
        "definitions": {
            "modality_order": list(S2_MODALITIES),
            "label_semantics": LABEL_SEMANTICS,
            "connectivity": REFERENCE_CONNECTIVITY,
            "spacing_rule": REFERENCE_SPACING_RULE,
            "reference_ring_inner_mm": REFERENCE_RING_INNER_MM,
            "reference_ring_outer_mm": REFERENCE_RING_OUTER_MM,
            "minimum_reference_voxels": REFERENCE_MINIMUM_VOXELS,
            "cdf_compression": f"weighted_midpoint_quantiles_{REFERENCE_CDF_POINTS}",
            "patient_cluster_weighting": "equal_patient_then_equal_component_v1",
        },
        "affected_abs_threshold": affected_thresholds,
        "components": rows,
    }
    payload = _strict_json(payload)
    payload["reference_cdf_audit_sha256"] = canonical_json_sha256(
        payload, exclude=("reference_cdf_audit_sha256",)
    )
    return payload


def _reference_boundary_pool(
    reference: Mapping[str, Any], label_value: int, modality: str
) -> tuple[list[float], list[float]]:
    contributing: list[tuple[str, list[float]]] = []
    components_per_patient: Counter[str] = Counter()
    key = f"{label_value}:{modality}"
    for row in reference["components"]:
        measured = row["boundary"][key]
        if measured.get("status") != "measured":
            continue
        patient = str(row["patient_group"])
        values = [float(value) for value in measured["signed_values"]]
        if len(values) < 2:
            continue
        contributing.append((patient, values))
        components_per_patient[patient] += 1
    if not contributing:
        raise ValueError(f"Reference has no boundary pool for {key}")
    values: list[float] = []
    weights: list[float] = []
    for patient, sample in contributing:
        per_value = 1.0 / (components_per_patient[patient] * len(sample))
        values.extend(sample)
        weights.extend([per_value] * len(sample))
    return values, weights


def build_measurement_calibration(
    *,
    reference: Mapping[str, Any],
    reference_path: str | Path,
    partition_path: str | Path,
    manifest: ComponentManifest,
    boundary_policy: str,
    halo_radius_mm: float,
) -> dict[str, Any]:
    """Build an intentionally non-selective calibration for metric collection.

    This artifact is never eligible for Gate-0 or training. It carries real
    source hashes and real Reference CDF values, while all decision limits are
    deliberately wide so Development can be measured before thresholds exist.
    """
    if boundary_policy == "label_only_qc_v1" and halo_radius_mm != 0.0:
        raise ValueError("label-only measurement requires zero halo radius")
    if boundary_policy != "label_only_qc_v1" and halo_radius_mm <= 0.0:
        raise ValueError("halo measurement requires a positive radius")
    partition_file = Path(partition_path).expanduser().resolve()
    reference_file = Path(reference_path).expanduser().resolve()
    partition = load_partition(partition_file)
    if reference.get("reference_cdf_audit_sha256") != canonical_json_sha256(
        reference, exclude=("reference_cdf_audit_sha256",)
    ):
        raise ValueError("Reference CDF audit has drifted")
    huge = 1.0e9
    quantile_names = ("q01", "q05", "q50", "q90", "q95", "q99")
    minimum_mad: dict[str, float] = {}
    for modality in S2_MODALITIES:
        values = np.asarray(
            [float(row["reference_mad"][modality]) for row in reference["components"]],
            dtype=np.float64,
        )
        minimum_mad[modality] = float(
            max(np.finfo(np.float64).eps, np.min(values) * 0.1)
        )

    boundary_thresholds: list[dict[str, Any]] = []
    for label_value in (1, 2, 3):
        for modality in S2_MODALITIES:
            signed, weights = _reference_boundary_pool(reference, label_value, modality)
            boundary_thresholds.append(
                {
                    "label": label_value,
                    "modality": modality,
                    "core_volume_mm3": [0.0, huge],
                    "boundary_area_mm2": [0.0, huge],
                    "min_standard_area_mm2": 1.0,
                    "reference_signed_values": signed,
                    "reference_signed_weights": weights,
                    "reference_abs_values": [abs(value) for value in signed],
                    "reference_abs_weights": weights,
                    "ks_signed_max": 1.0,
                    "ks_abs_max": 1.0,
                    "quantile_intervals": {
                        **{f"signed_{name}": [-huge, huge] for name in quantile_names},
                        **{f"abs_{name}": [0.0, huge] for name in quantile_names},
                    },
                    "signed_envelope": [-huge, huge],
                    "abs_upper": huge,
                    "max_abnormal_fraction": 1.0,
                    "max_patch_area_mm2": huge,
                    "max_patch_fraction": 1.0,
                    "small_q95_abs_max": huge,
                    "small_max_abs": huge,
                }
            )

    raw_qc = {
        modality: {
            "residual_quantile_intervals": {
                name: [-huge, huge] for name in quantile_names
            },
            "extreme_abs_z": huge,
            "max_extreme_fraction": 1.0,
            "max_component_voxels": 64**3,
            "max_bbox_fill_ratio": 1.0,
            "max_axis_ratio": 64.0,
            "max_plane_fraction": 1.0,
        }
        for modality in S2_MODALITIES
    }
    cross_modal: dict[str, Any] = {}
    for label_value in (1, 2, 3):
        vectors = []
        for row in reference["components"]:
            effect = row["effects"][str(label_value)]
            if effect.get("status") != "measured":
                continue
            vectors.append(
                [
                    float(effect["modalities"][modality]["median_contrast"])
                    for modality in S2_MODALITIES
                ]
            )
        if not vectors:
            raise ValueError(f"Reference lacks cross-modal vectors for label {label_value}")
        mean = np.mean(np.asarray(vectors, dtype=np.float64), axis=0)
        cross_modal[str(label_value)] = {
            "minimum_voxels": 1,
            "contrast_intervals": {
                modality: [-huge, huge] for modality in S2_MODALITIES
            },
            "mean": mean.tolist(),
            "inverse_covariance": (np.eye(4, dtype=np.float64) * 1.0e-12).tolist(),
            "max_mahalanobis": huge,
            "affected_abs_threshold": {
                modality: float(reference["affected_abs_threshold"][str(label_value)][modality])
                for modality in S2_MODALITIES
            },
            "pairwise": {
                pair: {"iou": [0.0, 1.0], "centroid_distance_mm": huge}
                for pair in PAIR_KEYS
            },
        }

    calibration: dict[str, Any] = {
        "schema_version": 1,
        "status": "frozen",
        "calibration_role": "measurement_only_not_gate_eligible",
        "boundary_policy": boundary_policy,
        "modality_order": list(S2_MODALITIES),
        "label_semantics": LABEL_SEMANTICS,
        "epsilon": 1.0e-6,
        "geometry": {
            "halo_radius_mm": float(halo_radius_mm),
            "reference_ring_inner_mm": REFERENCE_RING_INNER_MM,
            "reference_ring_outer_mm": REFERENCE_RING_OUTER_MM,
            "minimum_reference_voxels": REFERENCE_MINIMUM_VOXELS,
            "harmonization_ring_inner_fraction": 0.25,
            "harmonization_ring_outer_fraction": 0.75,
        },
        "raw_qc": {"modalities": raw_qc},
        "boundary_qc": {
            "minimum_mad": minimum_mad,
            "thresholds": boundary_thresholds,
            "event_max_ratio": huge,
        },
        "cross_modal_qc": {"classes": cross_modal},
        "candidate_qc": {
            "modalities": {
                modality: {
                    "residual_retention": [0.0, huge],
                    "candidate_abs_z_q99": huge,
                }
                for modality in S2_MODALITIES
            }
        },
        "harmonization": {
            "minimum_ring_voxels": 1,
            "shell_edges": [0.0, 0.25, 0.5, 0.75, 1.0],
            "modalities": {
                modality: {
                    "gain": [1.0e-6, 1.0e6],
                    "offset": [-huge, huge],
                    "max_amplification_ratio": huge,
                    "max_halo_to_lesion_ratio": huge,
                    "radial_shell_upper": [huge, huge, huge, huge],
                }
                for modality in S2_MODALITIES
            },
        },
        "halo_qc": {
            "modalities": {
                modality: {
                    "residual_abs_z_q95": huge,
                    "gradient_difference_q99": huge,
                    "ncc_min": -1.0,
                    "gradient_cosine_q05_min": -1.0,
                    "outer_residual_abs_z_q99": huge,
                    "outer_gradient_delta_abs_z_q99": huge,
                    "outer_max_abs_z": huge,
                    "outer_abnormal_abs_z": huge,
                    "outer_max_abnormal_fraction": 1.0,
                    "outer_max_patch_area_mm2": huge,
                    "structure_tensor_sigma_mm": 1.0,
                    "structure_anisotropy_min": 0.0,
                    "structure_direction_cosine_q05_min": 0.0,
                    "minimum_structure_voxels": 1,
                }
                for modality in S2_MODALITIES
            }
        },
        "source_audit": {
            "partition_sha256": sha256_file(partition_file),
            "partition_audit_sha256": partition["partition_audit_sha256"],
            "reference_cdf_sha256": sha256_file(reference_file),
            "reference_cdf_audit_sha256": reference["reference_cdf_audit_sha256"],
            "component_manifest_sha256": manifest.identity_sha256,
            "target_groups_sha256": manifest.target_groups_sha256,
            "patient_group_count": int(partition["patient_group_count"]),
            "component_count": len(manifest.records),
        },
    }
    return _strict_json(calibration)


def build_measurement_route_config(
    manifest: ComponentManifest,
    *,
    calibration_sha256: str,
    boundary_policy: str,
    seed: int = 20260725,
) -> dict[str, Any]:
    return make_fix_v2_route_a_config(
        manifest,
        boundary_policy=boundary_policy,
        calibration_sha256=calibration_sha256,
        seed=seed,
    )
