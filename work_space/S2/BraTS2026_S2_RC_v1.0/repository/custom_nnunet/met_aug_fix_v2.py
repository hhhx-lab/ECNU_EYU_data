"""Fail-closed Fix-v2 generation geometry, harmonization, and quality gates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy import ndimage

try:
    from .met_aug_core import (
        CandidateProcessingResult,
        FIX_V2_BOUNDARY_POLICIES,
        MetAugContractError,
        S2_MODALITIES,
        sha256_file,
    )
except ImportError:
    from met_aug_core import (  # type: ignore
        CandidateProcessingResult,
        FIX_V2_BOUNDARY_POLICIES,
        MetAugContractError,
        S2_MODALITIES,
        sha256_file,
    )


FIX_V2_CALIBRATION_SCHEMA = 1
LABEL_SEMANTICS = {
    "0": "background",
    "1": "NETC",
    "2": "SNFH",
    "3": "ET",
    "4": "RC",
}
QUANTILE_NAMES = {
    "q01": 0.01,
    "q05": 0.05,
    "q50": 0.50,
    "q90": 0.90,
    "q95": 0.95,
    "q99": 0.99,
}


class _Reject(RuntimeError):
    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _finite_float(value: Any, *, label: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise MetAugContractError(f"{label} must be finite")
    return result


def _positive_float(value: Any, *, label: str, allow_zero: bool = False) -> float:
    result = _finite_float(value, label=label)
    invalid = result < 0 if allow_zero else result <= 0
    if invalid:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise MetAugContractError(f"{label} must be {qualifier}")
    return result


def _interval(value: Any, *, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise MetAugContractError(f"{label} must be a two-value list")
    lower = _finite_float(value[0], label=f"{label}[0]")
    upper = _finite_float(value[1], label=f"{label}[1]")
    if lower > upper:
        raise MetAugContractError(f"{label} lower bound exceeds upper bound")
    return lower, upper


def _bounded_float(
    value: Any, *, label: str, lower: float, upper: float
) -> float:
    result = _finite_float(value, label=label)
    if not lower <= result <= upper:
        raise MetAugContractError(f"{label} must be in [{lower}, {upper}]")
    return result


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MetAugContractError(f"{label} must be an object")
    return value


def _require_keys(value: Mapping[str, Any], required: Iterable[str], *, label: str) -> None:
    missing = sorted(set(required) - set(value))
    if missing:
        raise MetAugContractError(f"{label} misses fields: {missing}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _mad(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        name: float(np.quantile(values, probability, method="linear"))
        for name, probability in QUANTILE_NAMES.items()
    }


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probabilities: Iterable[float]
) -> np.ndarray:
    if values.size == 0 or values.size != weights.size:
        raise ValueError("weighted quantile requires matching nonempty arrays")
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_weights = weights[order]
    total = float(np.sum(ordered_weights))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("weighted quantile requires positive finite weights")
    cumulative = (np.cumsum(ordered_weights) - 0.5 * ordered_weights) / total
    return np.interp(np.asarray(tuple(probabilities), dtype=np.float64), cumulative, ordered_values)


def _weighted_ks_distance(
    sample_values: np.ndarray,
    sample_weights: np.ndarray,
    reference_values: np.ndarray,
    reference_weights: np.ndarray,
) -> float:
    if sample_values.size == 0 or reference_values.size == 0:
        raise ValueError("KS distance requires nonempty samples")
    points = np.unique(np.concatenate((sample_values, reference_values)))

    def cdf(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ordered_values = values[order]
        ordered_weights = weights[order]
        cumulative = np.cumsum(ordered_weights)
        total = float(cumulative[-1])
        indices = np.searchsorted(ordered_values, points, side="right") - 1
        result = np.zeros(points.shape, dtype=np.float64)
        present = indices >= 0
        result[present] = cumulative[indices[present]] / total
        return result

    return float(np.max(np.abs(cdf(sample_values, sample_weights) - cdf(reference_values, reference_weights))))


def _in_interval(value: float, interval: tuple[float, float]) -> bool:
    return interval[0] <= value <= interval[1]


def _component_shape_metrics(mask: np.ndarray) -> dict[str, float]:
    if not np.any(mask):
        return {
            "max_component_voxels": 0,
            "max_bbox_fill_ratio": 0.0,
            "max_axis_ratio": 0.0,
            "max_plane_fraction": 0.0,
        }
    labelled, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    max_voxels = 0
    max_fill = 0.0
    max_axis_ratio = 0.0
    max_plane_fraction = 0.0
    for component_id in range(1, count + 1):
        component = labelled == component_id
        points = np.argwhere(component)
        voxels = int(points.shape[0])
        extents = points.max(axis=0) - points.min(axis=0) + 1
        bbox_voxels = int(np.prod(extents))
        max_voxels = max(max_voxels, voxels)
        max_fill = max(max_fill, float(voxels / bbox_voxels))
        max_axis_ratio = max(max_axis_ratio, float(extents.max() / max(1, extents.min())))
        for axis in range(3):
            plane_counts = np.count_nonzero(component, axis=tuple(
                index for index in range(3) if index != axis
            ))
            max_plane_fraction = max(
                max_plane_fraction,
                float(np.max(plane_counts) / voxels),
            )
    return {
        "max_component_voxels": max_voxels,
        "max_bbox_fill_ratio": max_fill,
        "max_axis_ratio": max_axis_ratio,
        "max_plane_fraction": max_plane_fraction,
    }


@dataclass(frozen=True)
class FixV2Calibration:
    path: Path
    sha256: str
    payload: Mapping[str, Any]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        expected_policy: str | None = None,
    ) -> "FixV2Calibration":
        calibration_path = Path(path).expanduser().resolve()
        if not calibration_path.is_file():
            raise FileNotFoundError(f"Fix-v2 calibration is missing: {calibration_path}")
        observed_sha256 = sha256_file(calibration_path)
        if expected_sha256 is not None and observed_sha256 != expected_sha256:
            raise MetAugContractError("Fix-v2 calibration SHA256 mismatch")
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise MetAugContractError("Fix-v2 calibration root must be an object")
        cls._validate_payload(payload, expected_policy=expected_policy)
        return cls(calibration_path, observed_sha256, payload)

    @staticmethod
    def validate_payload(
        payload: Mapping[str, Any], *, expected_policy: str | None = None
    ) -> None:
        FixV2Calibration._validate_payload(
            payload,
            expected_policy=expected_policy,
        )

    @staticmethod
    def _validate_payload(
        payload: Mapping[str, Any], *, expected_policy: str | None
    ) -> None:
        _require_keys(
            payload,
            {
                "schema_version",
                "status",
                "boundary_policy",
                "modality_order",
                "label_semantics",
                "epsilon",
                "geometry",
                "raw_qc",
                "boundary_qc",
                "cross_modal_qc",
                "candidate_qc",
                "harmonization",
                "halo_qc",
                "source_audit",
            },
            label="Fix-v2 calibration",
        )
        if int(payload["schema_version"]) != FIX_V2_CALIBRATION_SCHEMA:
            raise MetAugContractError("unsupported Fix-v2 calibration schema")
        if payload["status"] != "frozen":
            raise MetAugContractError("Fix-v2 calibration is not frozen")
        policy = str(payload["boundary_policy"])
        if policy not in FIX_V2_BOUNDARY_POLICIES:
            raise MetAugContractError(f"unsupported calibration policy: {policy}")
        if expected_policy is not None and policy != expected_policy:
            raise MetAugContractError("calibration policy does not match route config")
        if tuple(payload["modality_order"]) != S2_MODALITIES:
            raise MetAugContractError("Fix-v2 modality order has drifted")
        if dict(payload["label_semantics"]) != LABEL_SEMANTICS:
            raise MetAugContractError("Fix-v2 label semantics have drifted")
        _positive_float(payload["epsilon"], label="epsilon")
        source_audit = _require_mapping(payload["source_audit"], label="source_audit")
        _require_keys(
            source_audit,
            {
                "partition_sha256",
                "partition_audit_sha256",
                "reference_cdf_sha256",
                "reference_cdf_audit_sha256",
                "component_manifest_sha256",
                "target_groups_sha256",
                "patient_group_count",
                "component_count",
            },
            label="source_audit",
        )
        for key in (
            "partition_sha256",
            "partition_audit_sha256",
            "reference_cdf_sha256",
            "reference_cdf_audit_sha256",
            "component_manifest_sha256",
            "target_groups_sha256",
        ):
            value = str(source_audit[key])
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise MetAugContractError(f"source_audit {key} is malformed")
        for key in ("patient_group_count", "component_count"):
            if int(source_audit[key]) <= 0:
                raise MetAugContractError(f"source_audit {key} must be positive")
        geometry = _require_mapping(payload["geometry"], label="geometry")
        _require_keys(
            geometry,
            {
                "halo_radius_mm",
                "reference_ring_inner_mm",
                "reference_ring_outer_mm",
                "minimum_reference_voxels",
                "harmonization_ring_inner_fraction",
                "harmonization_ring_outer_fraction",
            },
            label="geometry",
        )
        halo_radius = _positive_float(
            geometry["halo_radius_mm"], label="halo_radius_mm", allow_zero=True
        )
        if policy == "label_only_qc_v1" and halo_radius != 0.0:
            raise MetAugContractError("label-only policy requires zero halo radius")
        if policy != "label_only_qc_v1" and halo_radius <= 0.0:
            raise MetAugContractError("halo policy requires positive halo radius")
        inner = _positive_float(
            geometry["reference_ring_inner_mm"], label="reference ring inner", allow_zero=True
        )
        outer = _positive_float(
            geometry["reference_ring_outer_mm"], label="reference ring outer"
        )
        if outer <= inner:
            raise MetAugContractError("reference ring radii are invalid")
        ring_inner = _finite_float(
            geometry["harmonization_ring_inner_fraction"], label="harmonization ring inner"
        )
        ring_outer = _finite_float(
            geometry["harmonization_ring_outer_fraction"], label="harmonization ring outer"
        )
        if not 0 <= ring_inner < ring_outer <= 1:
            raise MetAugContractError("harmonization ring fractions are invalid")
        if int(geometry["minimum_reference_voxels"]) <= 0:
            raise MetAugContractError("minimum_reference_voxels must be positive")

        raw_qc = _require_mapping(payload["raw_qc"], label="raw_qc")
        _validate_per_modality(raw_qc, label="raw_qc")
        for modality, value in raw_qc["modalities"].items():
            _validate_raw_modality(
                _require_mapping(value, label=f"raw_qc {modality}"),
                label=f"raw_qc {modality}",
            )
        candidate_qc = _require_mapping(payload["candidate_qc"], label="candidate_qc")
        _validate_per_modality(candidate_qc, label="candidate_qc")
        for modality, value in candidate_qc["modalities"].items():
            _validate_candidate_modality(
                _require_mapping(value, label=f"candidate_qc {modality}"),
                label=f"candidate_qc {modality}",
            )
        boundary_qc = _require_mapping(payload["boundary_qc"], label="boundary_qc")
        _require_keys(
            boundary_qc,
            {"minimum_mad", "thresholds", "event_max_ratio"},
            label="boundary_qc",
        )
        minimum_mad = _require_mapping(
            boundary_qc["minimum_mad"], label="boundary minimum_mad"
        )
        if set(minimum_mad) != set(S2_MODALITIES):
            raise MetAugContractError("boundary minimum_mad lacks modalities")
        for modality in S2_MODALITIES:
            _positive_float(minimum_mad[modality], label=f"minimum_mad {modality}")
        if not isinstance(boundary_qc["thresholds"], list) or not boundary_qc["thresholds"]:
            raise MetAugContractError("boundary_qc thresholds are empty")
        for index, threshold in enumerate(boundary_qc["thresholds"]):
            _validate_boundary_threshold(
                _require_mapping(threshold, label=f"boundary threshold {index}"),
                label=f"boundary threshold {index}",
            )
        _validate_boundary_threshold_partition(boundary_qc["thresholds"])
        _positive_float(
            boundary_qc["event_max_ratio"], label="boundary event_max_ratio"
        )

        cross_modal = _require_mapping(payload["cross_modal_qc"], label="cross_modal_qc")
        classes = _require_mapping(cross_modal.get("classes"), label="cross_modal classes")
        if set(classes) != {"1", "2", "3"}:
            raise MetAugContractError("cross-modal calibration must define labels 1/2/3")
        for label_value, value in classes.items():
            _validate_cross_modal_class(
                _require_mapping(value, label=f"cross-modal class {label_value}"),
                label=f"cross-modal class {label_value}",
            )

        harmonization = _require_mapping(payload["harmonization"], label="harmonization")
        _require_keys(
            harmonization,
            {"minimum_ring_voxels", "shell_edges", "modalities"},
            label="harmonization",
        )
        shell_edges = tuple(float(item) for item in harmonization["shell_edges"])
        if len(shell_edges) < 2 or shell_edges[0] != 0.0 or shell_edges[-1] != 1.0:
            raise MetAugContractError("harmonization shell_edges must span 0..1")
        if any(left >= right for left, right in zip(shell_edges, shell_edges[1:])):
            raise MetAugContractError("harmonization shell_edges are not increasing")
        _validate_per_modality(harmonization, label="harmonization")
        if int(harmonization["minimum_ring_voxels"]) <= 0:
            raise MetAugContractError("harmonization minimum_ring_voxels must be positive")
        for modality, value in harmonization["modalities"].items():
            _validate_harmonization_modality(
                _require_mapping(value, label=f"harmonization {modality}"),
                label=f"harmonization {modality}",
                shell_count=len(shell_edges) - 1,
            )
        halo_qc = _require_mapping(payload["halo_qc"], label="halo_qc")
        _validate_per_modality(halo_qc, label="halo_qc")
        for modality, value in halo_qc["modalities"].items():
            _validate_halo_modality(
                _require_mapping(value, label=f"halo_qc {modality}"),
                label=f"halo_qc {modality}",
            )

    @property
    def boundary_policy(self) -> str:
        return str(self.payload["boundary_policy"])

    @property
    def epsilon(self) -> float:
        return float(self.payload["epsilon"])


def _validate_per_modality(value: Mapping[str, Any], *, label: str) -> None:
    modalities = _require_mapping(value.get("modalities"), label=f"{label} modalities")
    if set(modalities) != set(S2_MODALITIES):
        raise MetAugContractError(f"{label} modalities have drifted")
    for modality in S2_MODALITIES:
        if not isinstance(modalities[modality], Mapping):
            raise MetAugContractError(f"{label} {modality} must be an object")


def _validate_raw_modality(value: Mapping[str, Any], *, label: str) -> None:
    required = {
        "residual_quantile_intervals",
        "extreme_abs_z",
        "max_extreme_fraction",
        "max_component_voxels",
        "max_bbox_fill_ratio",
        "max_axis_ratio",
        "max_plane_fraction",
    }
    _require_keys(value, required, label=label)
    intervals = _require_mapping(
        value["residual_quantile_intervals"], label=f"{label} intervals"
    )
    if set(intervals) != set(QUANTILE_NAMES):
        raise MetAugContractError(f"{label} residual quantiles have drifted")
    for name, interval in intervals.items():
        _interval(interval, label=f"{label} {name}")
    _positive_float(value["extreme_abs_z"], label=f"{label} extreme_abs_z")
    _bounded_float(
        value["max_extreme_fraction"],
        label=f"{label} max_extreme_fraction",
        lower=0.0,
        upper=1.0,
    )
    if int(value["max_component_voxels"]) < 0:
        raise MetAugContractError(f"{label} max_component_voxels must be nonnegative")
    _bounded_float(
        value["max_bbox_fill_ratio"],
        label=f"{label} max_bbox_fill_ratio",
        lower=0.0,
        upper=1.0,
    )
    if _finite_float(value["max_axis_ratio"], label=f"{label} max_axis_ratio") < 1.0:
        raise MetAugContractError(f"{label} max_axis_ratio must be at least one")
    _bounded_float(
        value["max_plane_fraction"],
        label=f"{label} max_plane_fraction",
        lower=0.0,
        upper=1.0,
    )


def _validate_candidate_modality(value: Mapping[str, Any], *, label: str) -> None:
    _require_keys(
        value,
        {"residual_retention", "candidate_abs_z_q99"},
        label=label,
    )
    retention = _interval(value["residual_retention"], label=f"{label} retention")
    if retention[0] < 0:
        raise MetAugContractError(f"{label} retention must be nonnegative")
    _positive_float(
        value["candidate_abs_z_q99"], label=f"{label} candidate_abs_z_q99"
    )


def _validate_boundary_threshold(value: Mapping[str, Any], *, label: str) -> None:
    required = {
        "label",
        "modality",
        "core_volume_mm3",
        "boundary_area_mm2",
        "min_standard_area_mm2",
        "reference_signed_values",
        "reference_signed_weights",
        "reference_abs_values",
        "reference_abs_weights",
        "ks_signed_max",
        "ks_abs_max",
        "quantile_intervals",
        "signed_envelope",
        "abs_upper",
        "max_abnormal_fraction",
        "max_patch_area_mm2",
        "max_patch_fraction",
        "small_q95_abs_max",
        "small_max_abs",
    }
    _require_keys(value, required, label=label)
    if int(value["label"]) not in {1, 2, 3}:
        raise MetAugContractError(f"{label} has invalid class")
    if str(value["modality"]) not in S2_MODALITIES:
        raise MetAugContractError(f"{label} has invalid modality")
    core_interval = _interval(value["core_volume_mm3"], label=f"{label} core volume")
    area_interval = _interval(value["boundary_area_mm2"], label=f"{label} boundary area")
    if core_interval[0] < 0 or area_interval[0] < 0:
        raise MetAugContractError(f"{label} ranges must be nonnegative")
    _positive_float(value["min_standard_area_mm2"], label=f"{label} minimum area")
    for prefix in ("reference_signed", "reference_abs"):
        values = np.asarray(value[f"{prefix}_values"], dtype=np.float64)
        weights = np.asarray(value[f"{prefix}_weights"], dtype=np.float64)
        if values.ndim != 1 or weights.shape != values.shape or values.size < 2:
            raise MetAugContractError(f"{label} {prefix} reference is malformed")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise MetAugContractError(f"{label} {prefix} reference is invalid")
    for key in (
        "ks_signed_max",
        "ks_abs_max",
        "abs_upper",
        "max_abnormal_fraction",
        "max_patch_area_mm2",
        "max_patch_fraction",
        "small_q95_abs_max",
        "small_max_abs",
    ):
        _positive_float(value[key], label=f"{label} {key}", allow_zero=True)
    for key in ("ks_signed_max", "ks_abs_max", "max_abnormal_fraction", "max_patch_fraction"):
        _bounded_float(
            value[key], label=f"{label} {key}", lower=0.0, upper=1.0
        )
    _interval(value["signed_envelope"], label=f"{label} signed envelope")
    quantile_intervals = _require_mapping(
        value["quantile_intervals"], label=f"{label} quantile intervals"
    )
    expected_quantiles = {
        *(f"signed_{name}" for name in QUANTILE_NAMES),
        *(f"abs_{name}" for name in QUANTILE_NAMES),
    }
    if set(quantile_intervals) != expected_quantiles:
        raise MetAugContractError(f"{label} quantile intervals have drifted")
    for key, interval in quantile_intervals.items():
        if key not in {
            *(f"signed_{name}" for name in QUANTILE_NAMES),
            *(f"abs_{name}" for name in QUANTILE_NAMES),
        }:
            raise MetAugContractError(f"{label} has unsupported quantile {key}")
        _interval(interval, label=f"{label} {key}")


def _validate_boundary_threshold_partition(values: Iterable[Mapping[str, Any]]) -> None:
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for value in values:
        key = (int(value["label"]), str(value["modality"]))
        grouped.setdefault(key, []).append(value)
    expected = {
        (label_value, modality)
        for label_value in (1, 2, 3)
        for modality in S2_MODALITIES
    }
    if set(grouped) != expected:
        raise MetAugContractError("boundary thresholds do not cover every class/modality")
    for key, group in grouped.items():
        for index, left in enumerate(group):
            left_core = _interval(left["core_volume_mm3"], label=f"{key} core")
            left_area = _interval(left["boundary_area_mm2"], label=f"{key} area")
            for right in group[index + 1 :]:
                right_core = _interval(right["core_volume_mm3"], label=f"{key} core")
                right_area = _interval(right["boundary_area_mm2"], label=f"{key} area")
                core_overlap = max(left_core[0], right_core[0]) <= min(
                    left_core[1], right_core[1]
                )
                area_overlap = max(left_area[0], right_area[0]) <= min(
                    left_area[1], right_area[1]
                )
                if core_overlap and area_overlap:
                    raise MetAugContractError(
                        f"boundary threshold ranges overlap for {key[0]}:{key[1]}"
                    )


def _validate_cross_modal_class(value: Mapping[str, Any], *, label: str) -> None:
    _require_keys(
        value,
        {
            "minimum_voxels",
            "contrast_intervals",
            "mean",
            "inverse_covariance",
            "max_mahalanobis",
            "affected_abs_threshold",
            "pairwise",
        },
        label=label,
    )
    if int(value["minimum_voxels"]) <= 0:
        raise MetAugContractError(f"{label} minimum_voxels must be positive")
    intervals = _require_mapping(value["contrast_intervals"], label=f"{label} intervals")
    affected = _require_mapping(value["affected_abs_threshold"], label=f"{label} affected")
    if set(intervals) != set(S2_MODALITIES) or set(affected) != set(S2_MODALITIES):
        raise MetAugContractError(f"{label} modality keys have drifted")
    for modality in S2_MODALITIES:
        _interval(intervals[modality], label=f"{label} {modality} interval")
        _positive_float(affected[modality], label=f"{label} {modality} affected")
    mean = np.asarray(value["mean"], dtype=np.float64)
    inverse = np.asarray(value["inverse_covariance"], dtype=np.float64)
    if mean.shape != (4,) or inverse.shape != (4, 4):
        raise MetAugContractError(f"{label} multivariate shape is invalid")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(inverse)):
        raise MetAugContractError(f"{label} multivariate values are non-finite")
    if not np.allclose(inverse, inverse.T, rtol=0.0, atol=1e-8):
        raise MetAugContractError(f"{label} inverse covariance is not symmetric")
    if float(np.min(np.linalg.eigvalsh(inverse))) <= 0.0:
        raise MetAugContractError(f"{label} inverse covariance is not positive definite")
    _positive_float(value["max_mahalanobis"], label=f"{label} max_mahalanobis")
    pairwise = _require_mapping(value["pairwise"], label=f"{label} pairwise")
    modality_index = {modality: index for index, modality in enumerate(S2_MODALITIES)}
    for pair, limits in pairwise.items():
        left, separator, right = str(pair).partition(":")
        if (
            separator != ":"
            or left not in modality_index
            or right not in modality_index
            or modality_index[left] >= modality_index[right]
        ):
            raise MetAugContractError(f"{label} pair key is invalid: {pair}")
        mapping = _require_mapping(limits, label=f"{label} pair {pair}")
        _require_keys(mapping, {"iou", "centroid_distance_mm"}, label=f"{label} pair {pair}")
        iou = _interval(mapping["iou"], label=f"{label} pair {pair} iou")
        if iou[0] < 0 or iou[1] > 1:
            raise MetAugContractError(f"{label} pair {pair} IoU is invalid")
        _positive_float(
            mapping["centroid_distance_mm"], label=f"{label} pair {pair} centroid", allow_zero=True
        )


def _validate_harmonization_modality(
    value: Mapping[str, Any], *, label: str, shell_count: int
) -> None:
    required = {
        "gain",
        "offset",
        "max_amplification_ratio",
        "max_halo_to_lesion_ratio",
        "radial_shell_upper",
    }
    _require_keys(value, required, label=label)
    gain = _interval(value["gain"], label=f"{label} gain")
    if gain[0] <= 0:
        raise MetAugContractError(f"{label} gain must remain positive")
    _interval(value["offset"], label=f"{label} offset")
    _positive_float(
        value["max_amplification_ratio"], label=f"{label} amplification"
    )
    _positive_float(
        value["max_halo_to_lesion_ratio"], label=f"{label} halo/lesion"
    )
    radial = value["radial_shell_upper"]
    if not isinstance(radial, list) or len(radial) != shell_count:
        raise MetAugContractError(f"{label} radial shell limits have drifted")
    for index, limit in enumerate(radial):
        _positive_float(limit, label=f"{label} radial shell {index}")


def _validate_halo_modality(value: Mapping[str, Any], *, label: str) -> None:
    required = {
        "residual_abs_z_q95",
        "gradient_difference_q99",
        "ncc_min",
        "gradient_cosine_q05_min",
        "outer_residual_abs_z_q99",
        "outer_gradient_delta_abs_z_q99",
        "outer_max_abs_z",
        "outer_abnormal_abs_z",
        "outer_max_abnormal_fraction",
        "outer_max_patch_area_mm2",
        "structure_tensor_sigma_mm",
        "structure_anisotropy_min",
        "structure_direction_cosine_q05_min",
        "minimum_structure_voxels",
    }
    _require_keys(value, required, label=label)
    _positive_float(value["residual_abs_z_q95"], label=f"{label} residual")
    _positive_float(
        value["gradient_difference_q99"], label=f"{label} gradient difference"
    )
    _bounded_float(value["ncc_min"], label=f"{label} ncc_min", lower=-1.0, upper=1.0)
    _bounded_float(
        value["gradient_cosine_q05_min"],
        label=f"{label} gradient cosine",
        lower=-1.0,
        upper=1.0,
    )
    for key in (
        "outer_residual_abs_z_q99",
        "outer_gradient_delta_abs_z_q99",
        "outer_max_abs_z",
        "outer_abnormal_abs_z",
        "outer_max_patch_area_mm2",
        "structure_tensor_sigma_mm",
    ):
        _positive_float(value[key], label=f"{label} {key}")
    _bounded_float(
        value["outer_max_abnormal_fraction"],
        label=f"{label} outer abnormal fraction",
        lower=0.0,
        upper=1.0,
    )
    _bounded_float(
        value["structure_anisotropy_min"],
        label=f"{label} structure anisotropy",
        lower=0.0,
        upper=1.0,
    )
    _bounded_float(
        value["structure_direction_cosine_q05_min"],
        label=f"{label} structure direction cosine",
        lower=0.0,
        upper=1.0,
    )
    if int(value["minimum_structure_voxels"]) <= 0:
        raise MetAugContractError(f"{label} minimum_structure_voxels must be positive")


@dataclass(frozen=True)
class FixV2Geometry:
    label_support: np.ndarray
    image_support: np.ndarray
    harmonization_ring: np.ndarray
    reference_ring: np.ndarray
    alpha: np.ndarray
    distance_from_label_mm: np.ndarray


@dataclass(frozen=True)
class _BoundaryFaces:
    signed: np.ndarray
    absolute: np.ndarray
    weights: np.ndarray
    coordinates: np.ndarray
    coordinate_weights: np.ndarray


class FixV2CandidateProcessor:
    processor_policy = "fix_v2_qc_v1"

    def __init__(self, calibration: FixV2Calibration):
        self.calibration = calibration
        self.boundary_policy = calibration.boundary_policy
        self.calibration_sha256 = calibration.sha256
        source_audit = calibration.payload["source_audit"]
        self.component_manifest_sha256 = str(
            source_audit["component_manifest_sha256"]
        )
        self.target_groups_sha256 = str(source_audit["target_groups_sha256"])

    @classmethod
    def load(
        cls,
        calibration_path: str | Path,
        *,
        expected_sha256: str,
        expected_policy: str,
    ) -> "FixV2CandidateProcessor":
        return cls(
            FixV2Calibration.load(
                calibration_path,
                expected_sha256=expected_sha256,
                expected_policy=expected_policy,
            )
        )

    def process(
        self,
        *,
        original_image: np.ndarray,
        original_segmentation: np.ndarray,
        label_cube: np.ndarray,
        valid_mask: np.ndarray,
        spacing_mm: tuple[float, float, float],
        core_volume_mm3: float,
        seed: int,
        backend: Any,
    ) -> CandidateProcessingResult:
        label_support = label_cube != 0
        unchanged = CandidateProcessingResult(
            image=original_image.copy(),
            segmentation=original_segmentation.copy(),
            image_support=label_support.copy(),
            label_support=label_support.copy(),
        )
        try:
            self._validate_inputs(
                original_image,
                original_segmentation,
                label_cube,
                valid_mask,
                spacing_mm,
            )
            geometry = self._build_geometry(
                label_cube=label_cube,
                original_segmentation=original_segmentation,
                valid_mask=valid_mask,
                spacing_mm=spacing_mm,
            )
            raw = self._generate(
                backend=backend,
                original_image=original_image,
                label_cube=label_cube,
                geometry=geometry,
                seed=seed,
            )
            scales, reference = self._reference_statistics(original_image, geometry)
            raw_metrics = self._raw_qc(
                original=original_image,
                generated=raw,
                geometry=geometry,
                scales=scales,
            )
            raw_metrics = {
                **dict(raw_metrics),
                "cross_modal": self._cross_modal_qc(
                    candidate=raw,
                    label_cube=label_cube,
                    geometry=geometry,
                    spacing_mm=spacing_mm,
                    scales=scales,
                    reference=reference,
                    failure_reason="RAW_GENERATION_QC_FAIL",
                ),
            }
            pre_harmonization = original_image + geometry.alpha[None] * (
                raw - original_image
            )
            harmonization_metrics: Mapping[str, Any] = {"policy": "disabled"}
            candidate = pre_harmonization
            harmonized_generation = raw
            if self.boundary_policy == "halo_cosine_harmonized_v1":
                harmonized, harmonization_metrics = self._harmonize(
                    original=original_image,
                    generated=raw,
                    geometry=geometry,
                )
                candidate = original_image + geometry.alpha[None] * (
                    harmonized - original_image
                )
                harmonized_generation = harmonized
            candidate_segmentation = original_segmentation.copy()
            candidate_segmentation[label_support] = label_cube[label_support].astype(
                candidate_segmentation.dtype, copy=False
            )
            candidate_metrics = self._candidate_qc(
                original=original_image,
                original_segmentation=original_segmentation,
                raw=raw,
                pre_harmonization=pre_harmonization,
                candidate=candidate,
                candidate_segmentation=candidate_segmentation,
                label_cube=label_cube,
                geometry=geometry,
                spacing_mm=spacing_mm,
                core_volume_mm3=core_volume_mm3,
                scales=scales,
                reference=reference,
                harmonization_metrics=harmonization_metrics,
            )
            metadata = {
                "schema_version": FIX_V2_CALIBRATION_SCHEMA,
                "boundary_policy": self.boundary_policy,
                "calibration_sha256": self.calibration_sha256,
                "geometry": {
                    "label_support_voxels": int(np.count_nonzero(geometry.label_support)),
                    "image_support_voxels": int(np.count_nonzero(geometry.image_support)),
                    "harmonization_ring_voxels": int(
                        np.count_nonzero(geometry.harmonization_ring)
                    ),
                    "reference_ring_voxels": int(np.count_nonzero(geometry.reference_ring)),
                },
                "raw_qc": raw_metrics,
                "harmonization": harmonization_metrics,
                "candidate_qc": candidate_metrics,
            }
            return CandidateProcessingResult(
                image=candidate.astype(np.float32, copy=False),
                segmentation=candidate_segmentation,
                image_support=geometry.image_support,
                label_support=geometry.label_support,
                metadata=_jsonable(metadata),
                evidence={
                    "raw_generation": raw,
                    "pre_harmonization": pre_harmonization,
                    "harmonized_generation": harmonized_generation,
                    "candidate": candidate,
                    "alpha": geometry.alpha,
                    "image_support": geometry.image_support,
                    "label_support": geometry.label_support,
                    "reference_ring": geometry.reference_ring,
                    "harmonization_ring": geometry.harmonization_ring,
                    **{
                        f"boundary_label_{label_value}": _inside_boundary_voxels(
                            label_cube, label_value
                        )
                        for label_value in (1, 2, 3)
                    },
                },
            )
        except _Reject as exc:
            return CandidateProcessingResult(
                image=unchanged.image,
                segmentation=unchanged.segmentation,
                image_support=unchanged.image_support,
                label_support=unchanged.label_support,
                reason=exc.reason,
                metadata={
                    "schema_version": FIX_V2_CALIBRATION_SCHEMA,
                    "boundary_policy": self.boundary_policy,
                    "calibration_sha256": self.calibration_sha256,
                    "detail": exc.detail,
                },
            )

    @staticmethod
    def _validate_inputs(
        image: np.ndarray,
        segmentation: np.ndarray,
        label_cube: np.ndarray,
        valid_mask: np.ndarray,
        spacing_mm: tuple[float, float, float],
    ) -> None:
        if image.ndim != 4 or image.shape[0] != 4:
            raise _Reject("COMMIT_CONTRACT_FAIL", "Fix-v2 expects four image channels")
        if segmentation.shape != image.shape[1:] or label_cube.shape != segmentation.shape:
            raise _Reject("COMMIT_CONTRACT_FAIL", "Fix-v2 crop shapes disagree")
        if valid_mask.shape != segmentation.shape:
            raise _Reject("COMMIT_CONTRACT_FAIL", "Fix-v2 valid mask shape disagrees")
        if not np.all(np.isfinite(image)):
            raise _Reject("RAW_GENERATION_QC_FAIL", "input image is non-finite")
        values = set(int(value) for value in np.unique(label_cube))
        if 4 in values or not values.issubset({0, 1, 2, 3}):
            raise _Reject("LABEL_CONTRACT_FAIL", f"unsupported inserted labels: {sorted(values)}")
        if not np.any(label_cube != 0):
            raise _Reject("LABEL_CONTRACT_FAIL", "inserted label support is empty")
        if len(spacing_mm) != 3 or any(
            not np.isfinite(value) or value <= 0 for value in spacing_mm
        ):
            raise _Reject("COMMIT_CONTRACT_FAIL", "spacing is invalid")

    def _build_geometry(
        self,
        *,
        label_cube: np.ndarray,
        original_segmentation: np.ndarray,
        valid_mask: np.ndarray,
        spacing_mm: tuple[float, float, float],
    ) -> FixV2Geometry:
        geometry_config = self.calibration.payload["geometry"]
        label_support = label_cube != 0
        distance_from_label = ndimage.distance_transform_edt(
            ~label_support, sampling=spacing_mm
        )
        halo_radius = float(geometry_config["halo_radius_mm"])
        if self.boundary_policy == "label_only_qc_v1":
            image_support = label_support.copy()
            alpha = label_support.astype(np.float32)
            harmonization_ring = np.zeros(label_support.shape, dtype=bool)
        else:
            image_support = label_support | (distance_from_label <= halo_radius + 1e-8)
            alpha = np.zeros(label_support.shape, dtype=np.float32)
            alpha[label_support] = 1.0
            halo = image_support & ~label_support
            alpha[halo] = 0.5 * (
                1.0 + np.cos(np.pi * distance_from_label[halo] / halo_radius)
            )
            inner_fraction = float(
                geometry_config["harmonization_ring_inner_fraction"]
            )
            outer_fraction = float(
                geometry_config["harmonization_ring_outer_fraction"]
            )
            normalized = distance_from_label / halo_radius
            harmonization_ring = (
                halo
                & (normalized >= inner_fraction)
                & (normalized <= outer_fraction)
            )

        if np.any(image_support & ~valid_mask.astype(bool)):
            raise _Reject("HALO_PLACEMENT_INVALID", "image support leaves valid brain mask")
        if np.any((original_segmentation != 0) & image_support):
            raise _Reject("HALO_PLACEMENT_INVALID", "image support overlaps existing labels")
        edge = np.zeros(image_support.shape, dtype=bool)
        for axis in range(3):
            low = [slice(None)] * 3
            high = [slice(None)] * 3
            low[axis] = 0
            high[axis] = -1
            edge[tuple(low)] = True
            edge[tuple(high)] = True
        if np.any(image_support & edge):
            raise _Reject("HALO_PLACEMENT_INVALID", "image support touches crop boundary")

        distance_from_image_support = ndimage.distance_transform_edt(
            ~image_support, sampling=spacing_mm
        )
        reference_inner = float(geometry_config["reference_ring_inner_mm"])
        reference_outer = float(geometry_config["reference_ring_outer_mm"])
        reference_ring = (
            ~image_support
            & valid_mask.astype(bool)
            & (original_segmentation == 0)
            & (distance_from_image_support >= reference_inner)
            & (distance_from_image_support <= reference_outer)
        )
        minimum_reference = int(geometry_config["minimum_reference_voxels"])
        if int(np.count_nonzero(reference_ring)) < minimum_reference:
            raise _Reject(
                "BOUNDARY_QC_INSUFFICIENT_SUPPORT",
                "local reference ring has too few valid voxels",
            )
        if self.boundary_policy == "halo_cosine_harmonized_v1":
            minimum_ring = int(self.calibration.payload["harmonization"]["minimum_ring_voxels"])
            if int(np.count_nonzero(harmonization_ring)) < minimum_ring:
                raise _Reject(
                    "HARMONIZATION_FAIL", "generated harmonization ring is too small"
                )
        return FixV2Geometry(
            label_support=label_support,
            image_support=image_support,
            harmonization_ring=harmonization_ring,
            reference_ring=reference_ring,
            alpha=alpha,
            distance_from_label_mm=distance_from_label,
        )

    def _generate(
        self,
        *,
        backend: Any,
        original_image: np.ndarray,
        label_cube: np.ndarray,
        geometry: FixV2Geometry,
        seed: int,
    ) -> np.ndarray:
        try:
            # Every schema-4 candidate uses the same explicit-mask EDM entry
            # point. Candidate A passes H=L; B/C pass their expanded H. Legacy
            # schema-2/3 callers still invoke the backend without this argument.
            generated = backend.generate(
                original_image,
                label_cube,
                seed=seed,
                inpaint_support=geometry.image_support,
            )
        except Exception as exc:
            raise _Reject(
                "BACKEND_FAILURE", f"{type(exc).__name__}: {exc}"
            ) from exc
        generated = np.asarray(generated, dtype=np.float32)
        if generated.shape != original_image.shape or not np.all(np.isfinite(generated)):
            raise _Reject("BACKEND_FAILURE", "backend output is malformed or non-finite")
        if np.any(generated[:, ~geometry.image_support] != original_image[:, ~geometry.image_support]):
            raise _Reject("COMMIT_CONTRACT_FAIL", "backend changed image outside support")
        return generated

    def _reference_statistics(
        self, image: np.ndarray, geometry: FixV2Geometry
    ) -> tuple[dict[str, float], dict[str, float]]:
        minimum_mad = self.calibration.payload["boundary_qc"]["minimum_mad"]
        scales: dict[str, float] = {}
        medians: dict[str, float] = {}
        for index, modality in enumerate(S2_MODALITIES):
            values = image[index][geometry.reference_ring].astype(np.float64)
            medians[modality] = float(np.median(values))
            scale = _mad(values)
            if scale < float(minimum_mad[modality]):
                raise _Reject(
                    "BOUNDARY_QC_INSUFFICIENT_SUPPORT",
                    f"{modality} local reference MAD is too small",
                )
            scales[modality] = max(scale, self.calibration.epsilon)
        return scales, medians

    def _raw_qc(
        self,
        *,
        original: np.ndarray,
        generated: np.ndarray,
        geometry: FixV2Geometry,
        scales: Mapping[str, float],
    ) -> Mapping[str, Any]:
        result: dict[str, Any] = {"status": "pass", "modalities": {}}
        config = self.calibration.payload["raw_qc"]["modalities"]
        for channel, modality in enumerate(S2_MODALITIES):
            threshold = config[modality]
            _require_keys(
                threshold,
                {
                    "residual_quantile_intervals",
                    "extreme_abs_z",
                    "max_extreme_fraction",
                    "max_component_voxels",
                    "max_bbox_fill_ratio",
                    "max_axis_ratio",
                    "max_plane_fraction",
                },
                label=f"raw_qc {modality}",
            )
            residual_z = (
                generated[channel][geometry.label_support]
                - original[channel][geometry.label_support]
            ) / scales[modality]
            quantiles = _quantiles(residual_z)
            intervals = _require_mapping(
                threshold["residual_quantile_intervals"],
                label=f"raw_qc {modality} intervals",
            )
            failures: list[str] = []
            for name, interval_value in intervals.items():
                if name not in quantiles:
                    raise MetAugContractError(f"unsupported raw quantile: {name}")
                if not _in_interval(
                    quantiles[name], _interval(interval_value, label=f"raw {modality} {name}")
                ):
                    failures.append(name)
            extreme = np.zeros(geometry.label_support.shape, dtype=bool)
            extreme[geometry.label_support] = np.abs(residual_z) > float(
                threshold["extreme_abs_z"]
            )
            extreme_fraction = float(
                np.count_nonzero(extreme) / np.count_nonzero(geometry.label_support)
            )
            shape = _component_shape_metrics(extreme)
            checks = {
                "extreme_fraction": extreme_fraction,
                **shape,
            }
            if extreme_fraction > float(threshold["max_extreme_fraction"]):
                failures.append("extreme_fraction")
            for metric, limit_key in (
                ("max_component_voxels", "max_component_voxels"),
                ("max_bbox_fill_ratio", "max_bbox_fill_ratio"),
                ("max_axis_ratio", "max_axis_ratio"),
                ("max_plane_fraction", "max_plane_fraction"),
            ):
                if checks[metric] > float(threshold[limit_key]):
                    failures.append(metric)
            result["modalities"][modality] = {
                "quantiles": quantiles,
                **checks,
                "failures": failures,
            }
            if failures:
                raise _Reject(
                    "RAW_GENERATION_QC_FAIL",
                    f"{modality} raw generation failed: {sorted(failures)}",
                )
        return result

    def _harmonize(
        self,
        *,
        original: np.ndarray,
        generated: np.ndarray,
        geometry: FixV2Geometry,
    ) -> tuple[np.ndarray, Mapping[str, Any]]:
        result = generated.copy()
        metrics: dict[str, Any] = {"policy": "median_mad_v1", "modalities": {}}
        config = self.calibration.payload["harmonization"]["modalities"]
        ring = geometry.harmonization_ring
        for channel, modality in enumerate(S2_MODALITIES):
            thresholds = config[modality]
            _require_keys(
                thresholds,
                {
                    "gain",
                    "offset",
                    "max_amplification_ratio",
                    "max_halo_to_lesion_ratio",
                    "radial_shell_upper",
                },
                label=f"harmonization {modality}",
            )
            original_values = original[channel][ring].astype(np.float64)
            generated_values = generated[channel][ring].astype(np.float64)
            generated_mad = _mad(generated_values)
            if generated_mad <= self.calibration.epsilon:
                raise _Reject("HARMONIZATION_FAIL", f"{modality} generated MAD is too small")
            gain = _mad(original_values) / generated_mad
            offset = float(np.median(original_values) - gain * np.median(generated_values))
            if not _in_interval(gain, _interval(thresholds["gain"], label=f"{modality} gain")):
                raise _Reject("HARMONIZATION_FAIL", f"{modality} gain is outside calibration")
            if not _in_interval(offset, _interval(thresholds["offset"], label=f"{modality} offset")):
                raise _Reject("HARMONIZATION_FAIL", f"{modality} offset is outside calibration")
            result[channel] = gain * generated[channel] + offset
            metrics["modalities"][modality] = {"gain": gain, "offset": offset}
        return result, metrics

    def _candidate_qc(
        self,
        *,
        original: np.ndarray,
        original_segmentation: np.ndarray,
        raw: np.ndarray,
        pre_harmonization: np.ndarray,
        candidate: np.ndarray,
        candidate_segmentation: np.ndarray,
        label_cube: np.ndarray,
        geometry: FixV2Geometry,
        spacing_mm: tuple[float, float, float],
        core_volume_mm3: float,
        scales: Mapping[str, float],
        reference: Mapping[str, float],
        harmonization_metrics: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not np.all(np.isfinite(candidate)):
            raise _Reject("CANDIDATE_CONTENT_QC_FAIL", "candidate image is non-finite")
        if np.any(candidate[:, ~geometry.image_support] != original[:, ~geometry.image_support]):
            raise _Reject("COMMIT_CONTRACT_FAIL", "candidate changed image outside support")
        if not np.array_equal(
            candidate_segmentation[~geometry.label_support],
            original_segmentation[~geometry.label_support],
        ):
            raise _Reject(
                "COMMIT_CONTRACT_FAIL",
                "candidate changed segmentation outside label support",
            )
        if not np.array_equal(
            candidate_segmentation[geometry.label_support],
            label_cube[geometry.label_support],
        ):
            raise _Reject(
                "COMMIT_CONTRACT_FAIL",
                "candidate labels do not match the inserted component",
            )
        boundary = self._boundary_qc(
            candidate=candidate,
            label_cube=label_cube,
            geometry=geometry,
            spacing_mm=spacing_mm,
            core_volume_mm3=core_volume_mm3,
            scales=scales,
        )
        cross_modal = self._cross_modal_qc(
            candidate=candidate,
            label_cube=label_cube,
            geometry=geometry,
            spacing_mm=spacing_mm,
            scales=scales,
            reference=reference,
        )
        content = self._candidate_content_qc(
            original=original,
            raw=raw,
            candidate=candidate,
            geometry=geometry,
            scales=scales,
        )
        halo: Mapping[str, Any] = {"status": "not_applicable"}
        if self.boundary_policy != "label_only_qc_v1":
            halo = self._halo_qc(
                original=original,
                candidate=candidate,
                geometry=geometry,
                spacing_mm=spacing_mm,
                scales=scales,
            )
        amplification: Mapping[str, Any] = {"status": "not_applicable"}
        if self.boundary_policy == "halo_cosine_harmonized_v1":
            amplification = self._harmonization_amplification_qc(
                original=original,
                pre_harmonization=pre_harmonization,
                candidate=candidate,
                geometry=geometry,
                scales=scales,
                harmonization_metrics=harmonization_metrics,
            )
        return {
            "status": "pass",
            "boundary": boundary,
            "cross_modal": cross_modal,
            "content": content,
            "halo": halo,
            "harmonization_amplification": amplification,
        }

    def _boundary_qc(
        self,
        *,
        candidate: np.ndarray,
        label_cube: np.ndarray,
        geometry: FixV2Geometry,
        spacing_mm: tuple[float, float, float],
        core_volume_mm3: float,
        scales: Mapping[str, float],
    ) -> Mapping[str, Any]:
        if np.any(label_cube == 4):
            raise _Reject("LABEL_CONTRACT_FAIL", "Route A produced an RC boundary")
        report: dict[str, Any] = {"status": "pass", "strata": {}}
        max_ratio = 0.0
        for label_value in (1, 2, 3):
            if not np.any(label_cube == label_value):
                for modality in S2_MODALITIES:
                    report["strata"][f"{label_value}:{modality}"] = {
                        "status": "not_present"
                    }
                continue
            for channel, modality in enumerate(S2_MODALITIES):
                faces = _extract_boundary_faces(
                    label_cube=label_cube,
                    image=candidate[channel],
                    label_value=label_value,
                    scale=float(scales[modality]),
                    spacing_mm=spacing_mm,
                )
                if faces.signed.size == 0:
                    report["strata"][f"{label_value}:{modality}"] = {
                        "status": "not_present"
                    }
                    continue
                area = float(np.sum(faces.weights))
                threshold = self._select_boundary_threshold(
                    label_value=label_value,
                    modality=modality,
                    core_volume_mm3=core_volume_mm3,
                    boundary_area_mm2=area,
                )
                metrics, ratio = _evaluate_boundary_faces(
                    faces,
                    threshold,
                    geometry.label_support.shape,
                )
                report["strata"][f"{label_value}:{modality}"] = metrics
                max_ratio = max(max_ratio, ratio)
                if metrics["status"] != "pass":
                    raise _Reject(
                        "CANDIDATE_BOUNDARY_QC_FAIL",
                        f"boundary stratum {label_value}:{modality} failed: {metrics['failures']}",
                    )
        event_limit = float(self.calibration.payload["boundary_qc"]["event_max_ratio"])
        report["event_max_ratio"] = max_ratio
        report["event_max_ratio_limit"] = event_limit
        if max_ratio > event_limit:
            raise _Reject(
                "CANDIDATE_BOUNDARY_QC_FAIL",
                "event-level boundary maximum exceeds calibration",
            )
        return report

    def _select_boundary_threshold(
        self,
        *,
        label_value: int,
        modality: str,
        core_volume_mm3: float,
        boundary_area_mm2: float,
    ) -> Mapping[str, Any]:
        matches: list[Mapping[str, Any]] = []
        for value in self.calibration.payload["boundary_qc"]["thresholds"]:
            if int(value["label"]) != label_value or str(value["modality"]) != modality:
                continue
            core_range = _interval(value["core_volume_mm3"], label="core volume")
            area_range = _interval(value["boundary_area_mm2"], label="boundary area")
            if core_range[0] <= core_volume_mm3 <= core_range[1] and area_range[0] <= boundary_area_mm2 <= area_range[1]:
                matches.append(value)
        if len(matches) != 1:
            raise _Reject(
                "BOUNDARY_QC_INSUFFICIENT_SUPPORT",
                f"boundary calibration match count is {len(matches)} for {label_value}:{modality}",
            )
        return matches[0]

    def _cross_modal_qc(
        self,
        *,
        candidate: np.ndarray,
        label_cube: np.ndarray,
        geometry: FixV2Geometry,
        spacing_mm: tuple[float, float, float],
        scales: Mapping[str, float],
        reference: Mapping[str, float],
        failure_reason: str = "CANDIDATE_CROSS_MODAL_QC_FAIL",
    ) -> Mapping[str, Any]:
        report: dict[str, Any] = {"status": "pass", "classes": {}}
        classes = self.calibration.payload["cross_modal_qc"]["classes"]
        for label_value in (1, 2, 3):
            support = label_cube == label_value
            if not np.any(support):
                report["classes"][str(label_value)] = {"status": "not_present"}
                continue
            config = classes[str(label_value)]
            if int(np.count_nonzero(support)) < int(config["minimum_voxels"]):
                raise _Reject(
                    failure_reason,
                    f"label {label_value} has insufficient cross-modal support",
                )
            contrasts: list[float] = []
            affected_masks: dict[str, np.ndarray] = {}
            modality_metrics: dict[str, Any] = {}
            failures: list[str] = []
            for channel, modality in enumerate(S2_MODALITIES):
                normalized = (
                    candidate[channel] - float(reference[modality])
                ) / float(scales[modality])
                values = normalized[support]
                contrast = float(np.median(values))
                contrasts.append(contrast)
                interval = _interval(
                    config["contrast_intervals"][modality],
                    label=f"cross-modal {label_value} {modality}",
                )
                if not _in_interval(contrast, interval):
                    failures.append(f"{modality}:contrast")
                affected = support & (
                    np.abs(normalized)
                    >= float(config["affected_abs_threshold"][modality])
                )
                affected_masks[modality] = affected
                modality_metrics[modality] = {
                    "median_contrast": contrast,
                    "affected_fraction": float(
                        np.count_nonzero(affected) / np.count_nonzero(support)
                    ),
                }
            vector = np.asarray(contrasts, dtype=np.float64)
            mean = np.asarray(config["mean"], dtype=np.float64)
            inverse = np.asarray(config["inverse_covariance"], dtype=np.float64)
            difference = vector - mean
            squared = float(difference @ inverse @ difference)
            mahalanobis = float(np.sqrt(max(0.0, squared)))
            if mahalanobis > float(config["max_mahalanobis"]):
                failures.append("mahalanobis")
            pair_metrics: dict[str, Any] = {}
            for pair, limits in config["pairwise"].items():
                left, right = pair.split(":", 1)
                iou, centroid_distance = _mask_alignment(
                    affected_masks[left], affected_masks[right], spacing_mm
                )
                iou_interval = _interval(limits["iou"], label=f"{pair} iou")
                if not _in_interval(iou, iou_interval):
                    failures.append(f"{pair}:iou")
                if centroid_distance > float(limits["centroid_distance_mm"]):
                    failures.append(f"{pair}:centroid")
                pair_metrics[pair] = {
                    "iou": iou,
                    "centroid_distance_mm": centroid_distance,
                }
            report["classes"][str(label_value)] = {
                "status": "pass" if not failures else "fail",
                "modalities": modality_metrics,
                "effect_vector": contrasts,
                "mahalanobis": mahalanobis,
                "pairwise": pair_metrics,
                "failures": failures,
            }
            if failures:
                raise _Reject(
                    failure_reason,
                    f"label {label_value} cross-modal failures: {sorted(failures)}",
                )
        return report

    def _candidate_content_qc(
        self,
        *,
        original: np.ndarray,
        raw: np.ndarray,
        candidate: np.ndarray,
        geometry: FixV2Geometry,
        scales: Mapping[str, float],
    ) -> Mapping[str, Any]:
        config = self.calibration.payload["candidate_qc"]["modalities"]
        report: dict[str, Any] = {"status": "pass", "modalities": {}}
        for channel, modality in enumerate(S2_MODALITIES):
            thresholds = config[modality]
            _require_keys(
                thresholds,
                {"residual_retention", "candidate_abs_z_q99"},
                label=f"candidate_qc {modality}",
            )
            raw_residual = np.abs(
                raw[channel][geometry.label_support]
                - original[channel][geometry.label_support]
            )
            candidate_residual = np.abs(
                candidate[channel][geometry.label_support]
                - original[channel][geometry.label_support]
            )
            raw_q95 = float(np.quantile(raw_residual, 0.95))
            candidate_q95 = float(np.quantile(candidate_residual, 0.95))
            retention = candidate_q95 / max(raw_q95, self.calibration.epsilon)
            candidate_abs_z_q99 = float(
                np.quantile(candidate_residual / float(scales[modality]), 0.99)
            )
            failures: list[str] = []
            if not _in_interval(
                retention,
                _interval(
                    thresholds["residual_retention"],
                    label=f"candidate {modality} retention",
                ),
            ):
                failures.append("residual_retention")
            if candidate_abs_z_q99 > float(thresholds["candidate_abs_z_q99"]):
                failures.append("candidate_abs_z_q99")
            report["modalities"][modality] = {
                "residual_retention": retention,
                "candidate_abs_z_q99": candidate_abs_z_q99,
                "failures": failures,
            }
            if failures:
                raise _Reject(
                    "CANDIDATE_CONTENT_QC_FAIL",
                    f"{modality} candidate content failures: {sorted(failures)}",
                )
        return report

    def _halo_qc(
        self,
        *,
        original: np.ndarray,
        candidate: np.ndarray,
        geometry: FixV2Geometry,
        spacing_mm: tuple[float, float, float],
        scales: Mapping[str, float],
    ) -> Mapping[str, Any]:
        halo = geometry.image_support & ~geometry.label_support
        config = self.calibration.payload["halo_qc"]["modalities"]
        report: dict[str, Any] = {"status": "pass", "modalities": {}}
        for channel, modality in enumerate(S2_MODALITIES):
            thresholds = config[modality]
            _require_keys(
                thresholds,
                {
                    "residual_abs_z_q95",
                    "gradient_difference_q99",
                    "ncc_min",
                    "gradient_cosine_q05_min",
                    "outer_residual_abs_z_q99",
                    "outer_gradient_delta_abs_z_q99",
                    "outer_max_abs_z",
                    "outer_abnormal_abs_z",
                    "outer_max_abnormal_fraction",
                    "outer_max_patch_area_mm2",
                    "structure_tensor_sigma_mm",
                    "structure_anisotropy_min",
                    "structure_direction_cosine_q05_min",
                    "minimum_structure_voxels",
                },
                label=f"halo_qc {modality}",
            )
            original_values = original[channel][halo].astype(np.float64)
            candidate_values = candidate[channel][halo].astype(np.float64)
            residual_q95 = float(
                np.quantile(
                    np.abs(candidate_values - original_values) / scales[modality],
                    0.95,
                )
            )
            original_gradients = np.stack(
                np.gradient(original[channel], *spacing_mm), axis=0
            )
            candidate_gradients = np.stack(
                np.gradient(candidate[channel], *spacing_mm), axis=0
            )
            gradient_difference = np.linalg.norm(
                candidate_gradients - original_gradients, axis=0
            )
            gradient_difference_q99 = float(
                np.quantile(gradient_difference[halo] / scales[modality], 0.99)
            )
            dot = np.sum(original_gradients * candidate_gradients, axis=0)
            denominator = np.linalg.norm(original_gradients, axis=0) * np.linalg.norm(
                candidate_gradients, axis=0
            )
            informative = halo & (denominator > self.calibration.epsilon)
            gradient_cosine_q05 = 1.0
            if np.any(informative):
                gradient_cosine_q05 = float(
                    np.quantile(dot[informative] / denominator[informative], 0.05)
                )
            ncc = _normalized_cross_correlation(original_values, candidate_values)
            outer_boundary = _halo_outer_boundary_metrics(
                original=original[channel],
                candidate=candidate[channel],
                image_support=geometry.image_support,
                spacing_mm=spacing_mm,
                scale=float(scales[modality]),
                abnormal_abs_z=float(thresholds["outer_abnormal_abs_z"]),
            )
            structure = _structure_tensor_alignment(
                original=original[channel],
                candidate=candidate[channel],
                mask=halo,
                spacing_mm=spacing_mm,
                sigma_mm=float(thresholds["structure_tensor_sigma_mm"]),
                anisotropy_min=float(thresholds["structure_anisotropy_min"]),
            )
            failures: list[str] = []
            if residual_q95 > float(thresholds["residual_abs_z_q95"]):
                failures.append("residual_abs_z_q95")
            if gradient_difference_q99 > float(thresholds["gradient_difference_q99"]):
                failures.append("gradient_difference_q99")
            if ncc < float(thresholds["ncc_min"]):
                failures.append("ncc")
            if gradient_cosine_q05 < float(thresholds["gradient_cosine_q05_min"]):
                failures.append("gradient_cosine")
            if outer_boundary["residual_abs_z_q99"] > float(
                thresholds["outer_residual_abs_z_q99"]
            ):
                failures.append("outer_residual_abs_z_q99")
            if outer_boundary["gradient_delta_abs_z_q99"] > float(
                thresholds["outer_gradient_delta_abs_z_q99"]
            ):
                failures.append("outer_gradient_delta_abs_z_q99")
            if outer_boundary["max_abs_z"] > float(thresholds["outer_max_abs_z"]):
                failures.append("outer_max_abs_z")
            if outer_boundary["abnormal_fraction"] > float(
                thresholds["outer_max_abnormal_fraction"]
            ):
                failures.append("outer_abnormal_fraction")
            if outer_boundary["max_patch_area_mm2"] > float(
                thresholds["outer_max_patch_area_mm2"]
            ):
                failures.append("outer_max_patch_area_mm2")
            if structure["informative_voxels"] < int(
                thresholds["minimum_structure_voxels"]
            ):
                failures.append("structure_support")
            if structure["direction_cosine_q05"] < float(
                thresholds["structure_direction_cosine_q05_min"]
            ):
                failures.append("structure_direction")
            report["modalities"][modality] = {
                "residual_abs_z_q95": residual_q95,
                "gradient_difference_q99": gradient_difference_q99,
                "ncc": ncc,
                "gradient_cosine_q05": gradient_cosine_q05,
                "outer_boundary": outer_boundary,
                "structure_tensor": structure,
                "failures": failures,
            }
            if failures:
                raise _Reject(
                    "CANDIDATE_BOUNDARY_QC_FAIL",
                    f"{modality} halo failures: {sorted(failures)}",
                )
        return report

    def _harmonization_amplification_qc(
        self,
        *,
        original: np.ndarray,
        pre_harmonization: np.ndarray,
        candidate: np.ndarray,
        geometry: FixV2Geometry,
        scales: Mapping[str, float],
        harmonization_metrics: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        halo = geometry.image_support & ~geometry.label_support
        label = geometry.label_support
        config = self.calibration.payload["harmonization"]
        shell_edges = np.asarray(config["shell_edges"], dtype=np.float64)
        radius = float(self.calibration.payload["geometry"]["halo_radius_mm"])
        normalized_distance = geometry.distance_from_label_mm / radius
        report: dict[str, Any] = {"status": "pass", "modalities": {}}
        for channel, modality in enumerate(S2_MODALITIES):
            thresholds = config["modalities"][modality]
            pre_residual = np.abs(pre_harmonization[channel] - original[channel])
            final_residual = np.abs(candidate[channel] - original[channel])
            pre_q95 = float(np.quantile(pre_residual[halo], 0.95))
            final_halo_q95 = float(np.quantile(final_residual[halo], 0.95))
            final_label_q95 = float(np.quantile(final_residual[label], 0.95))
            amplification = final_halo_q95 / max(pre_q95, self.calibration.epsilon)
            halo_to_lesion = final_halo_q95 / max(
                final_label_q95, self.calibration.epsilon
            )
            shell_q95: list[float] = []
            failures: list[str] = []
            shell_limits = tuple(float(value) for value in thresholds["radial_shell_upper"])
            if len(shell_limits) != len(shell_edges) - 1:
                raise MetAugContractError(
                    f"{modality} radial shell limit count is invalid"
                )
            for index, (left, right) in enumerate(zip(shell_edges, shell_edges[1:])):
                shell = halo & (normalized_distance >= left) & (
                    normalized_distance <= right if index == len(shell_limits) - 1 else normalized_distance < right
                )
                if not np.any(shell):
                    raise _Reject(
                        "HARMONIZATION_FAIL",
                        f"{modality} radial shell {index} is empty",
                    )
                value = float(
                    np.quantile(
                        final_residual[shell] / float(scales[modality]), 0.95
                    )
                )
                shell_q95.append(value)
                if value > shell_limits[index]:
                    failures.append(f"radial_shell_{index}")
            if amplification > float(thresholds["max_amplification_ratio"]):
                failures.append("amplification_ratio")
            if halo_to_lesion > float(thresholds["max_halo_to_lesion_ratio"]):
                failures.append("halo_to_lesion_ratio")
            modality_harmonization = harmonization_metrics["modalities"][modality]
            report["modalities"][modality] = {
                **dict(modality_harmonization),
                "amplification_ratio": amplification,
                "halo_to_lesion_ratio": halo_to_lesion,
                "radial_shell_q95": shell_q95,
                "failures": failures,
            }
            if failures:
                raise _Reject(
                    "CANDIDATE_CONTENT_QC_FAIL",
                    f"{modality} harmonization amplification failed: {sorted(failures)}",
                )
        return report

def _extract_boundary_faces(
    *,
    label_cube: np.ndarray,
    image: np.ndarray,
    label_value: int,
    scale: float,
    spacing_mm: tuple[float, float, float],
) -> _BoundaryFaces:
    signed_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    coordinate_parts: list[np.ndarray] = []
    coordinate_weight_parts: list[np.ndarray] = []
    shape = np.asarray(label_cube.shape, dtype=np.int64)
    for axis in range(3):
        face_area = float(
            np.prod([spacing_mm[index] for index in range(3) if index != axis])
        )
        for direction in (-1, 1):
            inner_slices = [slice(None)] * 3
            outer_slices = [slice(None)] * 3
            if direction > 0:
                inner_slices[axis] = slice(0, -1)
                outer_slices[axis] = slice(1, None)
                offset = np.zeros(3, dtype=np.int64)
            else:
                inner_slices[axis] = slice(1, None)
                outer_slices[axis] = slice(0, -1)
                offset = np.zeros(3, dtype=np.int64)
                offset[axis] = 1
            inner = tuple(inner_slices)
            outer = tuple(outer_slices)
            mask = (label_cube[inner] == label_value) & (label_cube[outer] == 0)
            if not np.any(mask):
                continue
            values = (image[outer][mask] - image[inner][mask]) / (
                float(spacing_mm[axis]) * scale
            )
            coordinates = np.argwhere(mask).astype(np.int64) + offset
            if np.any(coordinates < 0) or np.any(coordinates >= shape):
                raise RuntimeError("boundary coordinate construction escaped crop")
            signed_parts.append(values.astype(np.float64, copy=False))
            weight_parts.append(np.full(values.shape, face_area, dtype=np.float64))
            coordinate_parts.append(coordinates)
            coordinate_weight_parts.append(
                np.full(values.shape, face_area, dtype=np.float64)
            )
    if not signed_parts:
        return _BoundaryFaces(
            signed=np.empty(0, dtype=np.float64),
            absolute=np.empty(0, dtype=np.float64),
            weights=np.empty(0, dtype=np.float64),
            coordinates=np.empty((0, 3), dtype=np.int64),
            coordinate_weights=np.empty(0, dtype=np.float64),
        )
    signed = np.concatenate(signed_parts)
    return _BoundaryFaces(
        signed=signed,
        absolute=np.abs(signed),
        weights=np.concatenate(weight_parts),
        coordinates=np.concatenate(coordinate_parts, axis=0),
        coordinate_weights=np.concatenate(coordinate_weight_parts),
    )


def _inside_boundary_voxels(label_cube: np.ndarray, label_value: int) -> np.ndarray:
    """Return inside voxels owning at least one outer lesion/background face."""
    lesion = label_cube != 0
    owned = label_cube == int(label_value)
    result = np.zeros(label_cube.shape, dtype=bool)
    for axis in range(3):
        for direction in (-1, 1):
            inside_slice = [slice(None)] * 3
            outside_slice = [slice(None)] * 3
            if direction < 0:
                inside_slice[axis] = slice(1, None)
                outside_slice[axis] = slice(None, -1)
            else:
                inside_slice[axis] = slice(None, -1)
                outside_slice[axis] = slice(1, None)
            inside = tuple(inside_slice)
            outside = tuple(outside_slice)
            face = owned[inside] & ~lesion[outside]
            result_view = result[inside]
            result_view[face] = True
    return result


def _evaluate_boundary_faces(
    faces: _BoundaryFaces,
    threshold: Mapping[str, Any],
    shape: tuple[int, int, int],
) -> tuple[Mapping[str, Any], float]:
    if faces.signed.size == 0:
        return {"status": "not_present", "failures": []}, 0.0
    area = float(np.sum(faces.weights))
    failures: list[str] = []
    ratios: list[float] = []
    signed_envelope = _interval(threshold["signed_envelope"], label="signed envelope")
    abs_upper = float(threshold["abs_upper"])
    abnormal_faces = (
        (faces.signed < signed_envelope[0])
        | (faces.signed > signed_envelope[1])
        | (faces.absolute > abs_upper)
    )
    abnormal_area = float(np.sum(faces.weights[abnormal_faces]))
    abnormal_fraction = abnormal_area / area
    area_by_voxel = np.zeros(shape, dtype=np.float64)
    if np.any(abnormal_faces):
        coordinates = faces.coordinates[abnormal_faces]
        np.add.at(
            area_by_voxel,
            tuple(coordinates[:, axis] for axis in range(3)),
            faces.coordinate_weights[abnormal_faces],
        )
    labelled, count = ndimage.label(
        area_by_voxel > 0, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    max_patch_area = 0.0
    for component_id in range(1, count + 1):
        max_patch_area = max(
            max_patch_area, float(np.sum(area_by_voxel[labelled == component_id]))
        )
    max_patch_fraction = max_patch_area / area
    quantile_probabilities = tuple(QUANTILE_NAMES.values())
    signed_quantiles = _weighted_quantile(
        faces.signed, faces.weights, quantile_probabilities
    )
    abs_quantiles = _weighted_quantile(
        faces.absolute, faces.weights, quantile_probabilities
    )
    quantiles = {
        **{
            f"signed_{name}": float(value)
            for name, value in zip(QUANTILE_NAMES, signed_quantiles)
        },
        **{
            f"abs_{name}": float(value)
            for name, value in zip(QUANTILE_NAMES, abs_quantiles)
        },
    }
    branch = "standard" if area >= float(threshold["min_standard_area_mm2"]) else "small_sample"
    ks_signed = None
    ks_abs = None
    if branch == "standard":
        reference_signed = np.asarray(
            threshold["reference_signed_values"], dtype=np.float64
        )
        reference_signed_weights = np.asarray(
            threshold["reference_signed_weights"], dtype=np.float64
        )
        reference_abs = np.asarray(threshold["reference_abs_values"], dtype=np.float64)
        reference_abs_weights = np.asarray(
            threshold["reference_abs_weights"], dtype=np.float64
        )
        ks_signed = _weighted_ks_distance(
            faces.signed, faces.weights, reference_signed, reference_signed_weights
        )
        ks_abs = _weighted_ks_distance(
            faces.absolute, faces.weights, reference_abs, reference_abs_weights
        )
        for name, value, limit in (
            ("ks_signed", ks_signed, float(threshold["ks_signed_max"])),
            ("ks_abs", ks_abs, float(threshold["ks_abs_max"])),
        ):
            ratios.append(value / max(limit, np.finfo(np.float64).eps))
            if value > limit:
                failures.append(name)
        for name, interval_value in threshold["quantile_intervals"].items():
            interval = _interval(interval_value, label=f"boundary {name}")
            value = quantiles[name]
            width = max(interval[1] - interval[0], np.finfo(np.float64).eps)
            deviation = max(interval[0] - value, value - interval[1], 0.0)
            ratios.append(1.0 + deviation / width if deviation > 0 else 0.0)
            if not _in_interval(value, interval):
                failures.append(name)
    else:
        q95 = quantiles["abs_q95"]
        maximum = float(np.max(faces.absolute))
        q95_limit = float(threshold["small_q95_abs_max"])
        maximum_limit = float(threshold["small_max_abs"])
        ratios.extend(
            (
                q95 / max(q95_limit, np.finfo(np.float64).eps),
                maximum / max(maximum_limit, np.finfo(np.float64).eps),
            )
        )
        if q95 > q95_limit:
            failures.append("small_q95_abs")
        if maximum > maximum_limit:
            failures.append("small_max_abs")
    for name, value, limit in (
        ("abnormal_fraction", abnormal_fraction, float(threshold["max_abnormal_fraction"])),
        ("max_patch_area_mm2", max_patch_area, float(threshold["max_patch_area_mm2"])),
        ("max_patch_fraction", max_patch_fraction, float(threshold["max_patch_fraction"])),
    ):
        ratios.append(value / max(limit, np.finfo(np.float64).eps))
        if value > limit:
            failures.append(name)
    return {
        "status": "pass" if not failures else "fail",
        "branch": branch,
        "face_count": int(faces.signed.size),
        "area_mm2": area,
        "ks_signed": ks_signed,
        "ks_abs": ks_abs,
        "quantiles": quantiles,
        "abnormal_fraction": abnormal_fraction,
        "max_patch_area_mm2": max_patch_area,
        "max_patch_fraction": max_patch_fraction,
        "failures": sorted(set(failures)),
    }, max(ratios, default=0.0)


def _mask_alignment(
    left: np.ndarray,
    right: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> tuple[float, float]:
    union = int(np.count_nonzero(left | right))
    intersection = int(np.count_nonzero(left & right))
    iou = 1.0 if union == 0 else float(intersection / union)
    if not np.any(left) and not np.any(right):
        return iou, 0.0
    if not np.any(left) or not np.any(right):
        # A finite sentinel keeps rejected-event metadata strict-JSON serializable.
        shape_diagonal_mm = float(
            np.linalg.norm(np.asarray(left.shape, dtype=np.float64) * np.asarray(spacing_mm))
        )
        return iou, shape_diagonal_mm
    left_centroid = np.argwhere(left).mean(axis=0) * np.asarray(spacing_mm)
    right_centroid = np.argwhere(right).mean(axis=0) * np.asarray(spacing_mm)
    return iou, float(np.linalg.norm(left_centroid - right_centroid))


def _halo_outer_boundary_metrics(
    *,
    original: np.ndarray,
    candidate: np.ndarray,
    image_support: np.ndarray,
    spacing_mm: tuple[float, float, float],
    scale: float,
    abnormal_abs_z: float,
) -> dict[str, float | int]:
    residual_parts: list[np.ndarray] = []
    gradient_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    coordinate_parts: list[np.ndarray] = []
    shape = np.asarray(image_support.shape, dtype=np.int64)
    for axis in range(3):
        face_area = float(
            np.prod([spacing_mm[index] for index in range(3) if index != axis])
        )
        for direction in (-1, 1):
            inner_slices = [slice(None)] * 3
            outer_slices = [slice(None)] * 3
            offset = np.zeros(3, dtype=np.int64)
            if direction > 0:
                inner_slices[axis] = slice(None, -1)
                outer_slices[axis] = slice(1, None)
            else:
                inner_slices[axis] = slice(1, None)
                outer_slices[axis] = slice(None, -1)
                offset[axis] = 1
            inner = tuple(inner_slices)
            outer = tuple(outer_slices)
            faces = image_support[inner] & ~image_support[outer]
            if not np.any(faces):
                continue
            residual = np.abs(candidate[inner][faces] - original[inner][faces]) / scale
            original_normal = (original[outer][faces] - original[inner][faces]) / float(
                spacing_mm[axis]
            )
            candidate_normal = (
                candidate[outer][faces] - candidate[inner][faces]
            ) / float(spacing_mm[axis])
            gradient_delta = np.abs(candidate_normal - original_normal) / scale
            coordinates = np.argwhere(faces).astype(np.int64) + offset
            if np.any(coordinates < 0) or np.any(coordinates >= shape):
                raise RuntimeError("halo boundary coordinate escaped crop")
            residual_parts.append(residual.astype(np.float64, copy=False))
            gradient_parts.append(gradient_delta.astype(np.float64, copy=False))
            weight_parts.append(np.full(residual.shape, face_area, dtype=np.float64))
            coordinate_parts.append(coordinates)
    if not residual_parts:
        raise _Reject("BOUNDARY_QC_INSUFFICIENT_SUPPORT", "halo outer boundary is empty")
    residual = np.concatenate(residual_parts)
    gradient_delta = np.concatenate(gradient_parts)
    weights = np.concatenate(weight_parts)
    coordinates = np.concatenate(coordinate_parts, axis=0)
    combined = np.maximum(residual, gradient_delta)
    abnormal = combined > abnormal_abs_z
    total_area = float(np.sum(weights))
    abnormal_area = float(np.sum(weights[abnormal]))
    area_by_voxel = np.zeros(tuple(int(value) for value in shape), dtype=np.float64)
    if np.any(abnormal):
        np.add.at(
            area_by_voxel,
            tuple(coordinates[abnormal, axis] for axis in range(3)),
            weights[abnormal],
        )
    labelled, count = ndimage.label(
        area_by_voxel > 0,
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    max_patch_area = max(
        (
            float(np.sum(area_by_voxel[labelled == component_id]))
            for component_id in range(1, count + 1)
        ),
        default=0.0,
    )
    return {
        "face_count": int(residual.size),
        "area_mm2": total_area,
        "residual_abs_z_q99": float(np.quantile(residual, 0.99)),
        "gradient_delta_abs_z_q99": float(np.quantile(gradient_delta, 0.99)),
        "max_abs_z": float(np.max(combined)),
        "abnormal_fraction": abnormal_area / total_area,
        "max_patch_area_mm2": max_patch_area,
    }


def _structure_tensor_alignment(
    *,
    original: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    sigma_mm: float,
    anisotropy_min: float,
) -> dict[str, float | int]:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return {"informative_voxels": 0, "direction_cosine_q05": 1.0}
    sigma_voxels = tuple(sigma_mm / float(value) for value in spacing_mm)

    def tensors(image: np.ndarray) -> np.ndarray:
        gradients = np.stack(np.gradient(image, *spacing_mm), axis=0)
        result = np.empty((coordinates.shape[0], 3, 3), dtype=np.float64)
        index = tuple(coordinates[:, axis] for axis in range(3))
        for left in range(3):
            for right in range(left, 3):
                product = ndimage.gaussian_filter(
                    gradients[left] * gradients[right],
                    sigma=sigma_voxels,
                    mode="nearest",
                )
                result[:, left, right] = product[index]
                result[:, right, left] = result[:, left, right]
        return result

    original_values, original_vectors = np.linalg.eigh(tensors(original))
    candidate_values, candidate_vectors = np.linalg.eigh(tensors(candidate))
    epsilon = np.finfo(np.float64).eps
    original_anisotropy = (
        original_values[:, 2] - original_values[:, 1]
    ) / np.maximum(np.abs(original_values[:, 2]), epsilon)
    candidate_anisotropy = (
        candidate_values[:, 2] - candidate_values[:, 1]
    ) / np.maximum(np.abs(candidate_values[:, 2]), epsilon)
    informative = (original_anisotropy >= anisotropy_min) & (
        candidate_anisotropy >= anisotropy_min
    )
    if not np.any(informative):
        return {"informative_voxels": 0, "direction_cosine_q05": 1.0}
    cosine = np.abs(
        np.sum(
            original_vectors[informative, :, 2]
            * candidate_vectors[informative, :, 2],
            axis=1,
        )
    )
    return {
        "informative_voxels": int(np.count_nonzero(informative)),
        "direction_cosine_q05": float(np.quantile(cosine, 0.05)),
    }


def _normalized_cross_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    if denominator <= np.finfo(np.float64).eps:
        return 1.0 if np.allclose(left, right, rtol=0.0, atol=0.0) else 0.0
    return float(np.dot(left_centered, right_centered) / denominator)
