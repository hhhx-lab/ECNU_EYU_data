"""Deterministic candidate selection and train-only Fix-v2 threshold derivation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping

import numpy as np

from .met_aug_core import S2_MODALITIES, canonical_json_sha256
from .met_aug_fix_v2 import FixV2Calibration, QUANTILE_NAMES


PAIR_KEYS = tuple(
    f"{left}:{right}"
    for left_index, left in enumerate(S2_MODALITIES)
    for right in S2_MODALITIES[left_index + 1 :]
)


def _finite(values: Iterable[float], *, label: str) -> np.ndarray:
    array = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} requires finite nonempty values")
    return array


def expanded_interval(
    values: Iterable[float],
    *,
    label: str,
    expansion_iqr: float = 0.75,
    lower_limit: float | None = None,
    upper_limit: float | None = None,
) -> list[float]:
    """Return a deterministic outer tolerance interval around observed values."""

    array = _finite(values, label=label)
    q25, q75 = np.quantile(array, (0.25, 0.75), method="linear")
    span = max(
        float(q75 - q25),
        float(np.ptp(array)) * 0.1,
        max(1.0, abs(float(np.median(array)))) * 1.0e-6,
    )
    lower = float(np.min(array) - expansion_iqr * span)
    upper = float(np.max(array) + expansion_iqr * span)
    if lower_limit is not None:
        lower = max(float(lower_limit), lower)
    if upper_limit is not None:
        upper = min(float(upper_limit), upper)
    if lower > upper:
        raise ValueError(f"{label} interval collapsed")
    return [lower, upper]


def reference_development_interval(
    reference_values: Iterable[float],
    development_values: Iterable[float],
    *,
    label: str,
    lower_limit: float | None = None,
    upper_limit: float | None = None,
) -> list[float]:
    """Use Reference extreme quantiles plus the accepted Development envelope."""

    reference = _finite(reference_values, label=f"{label} Reference")
    development = _finite(development_values, label=f"{label} Development")
    low, high = np.quantile(reference, (0.001, 0.999), method="linear")
    q25, q75 = np.quantile(reference, (0.25, 0.75), method="linear")
    margin = max(float(q75 - q25) * 0.1, 1.0e-6)
    lower = min(float(low), float(np.min(development))) - margin
    upper = max(float(high), float(np.max(development))) + margin
    if lower_limit is not None:
        lower = max(float(lower_limit), lower)
    if upper_limit is not None:
        upper = min(float(upper_limit), upper)
    if lower > upper:
        raise ValueError(f"{label} interval collapsed")
    return [lower, upper]


def _class_metrics(
    metadata_rows: Iterable[Mapping[str, Any]], label_value: int
) -> list[Mapping[str, Any]]:
    measured: list[Mapping[str, Any]] = []
    key = str(label_value)
    for metadata in metadata_rows:
        value = metadata["candidate_qc"]["cross_modal"]["classes"][key]
        if value.get("status") != "not_present":
            measured.append(value)
    if not measured:
        raise ValueError(f"accepted Development lacks label {label_value}")
    return measured


def _reference_effects(
    reference: Mapping[str, Any], label_value: int
) -> list[Mapping[str, Any]]:
    values = [
        component["effects"][str(label_value)]
        for component in reference["components"]
        if component["effects"][str(label_value)].get("status") == "measured"
    ]
    if not values:
        raise ValueError(f"Reference lacks label {label_value} effects")
    return values


def _joint_model(
    reference_effects: Iterable[Mapping[str, Any]],
    development_effects: Iterable[Mapping[str, Any]],
) -> tuple[list[float], list[list[float]], float]:
    reference_vectors = np.asarray(
        [
            [float(value["modalities"][modality]["median_contrast"]) for modality in S2_MODALITIES]
            for value in reference_effects
        ],
        dtype=np.float64,
    )
    development_vectors = np.asarray(
        [value["effect_vector"] for value in development_effects], dtype=np.float64
    )
    lower, upper = np.quantile(
        reference_vectors, (0.005, 0.995), axis=0, method="linear"
    )
    winsorized = np.clip(reference_vectors, lower, upper)
    mean = np.mean(winsorized, axis=0)
    covariance = np.cov(winsorized, rowvar=False, ddof=1)
    diagonal = np.diag(np.diag(covariance))
    scale = max(float(np.trace(covariance) / 4.0), 1.0e-6)
    shrunk = 0.8 * covariance + 0.2 * diagonal + np.eye(4) * scale * 1.0e-6
    inverse = np.linalg.inv(shrunk)

    def distances(vectors: np.ndarray) -> np.ndarray:
        differences = vectors - mean
        return np.sqrt(np.maximum(0.0, np.einsum("ni,ij,nj->n", differences, inverse, differences)))

    reference_limit = float(
        np.quantile(distances(reference_vectors), 0.999, method="linear")
    )
    development_limit = float(np.max(distances(development_vectors)))
    maximum = max(reference_limit, development_limit) * 1.10 + 1.0e-6
    return mean.tolist(), inverse.tolist(), maximum


def derive_calibration(
    *,
    measurement_calibration: Mapping[str, Any],
    reference: Mapping[str, Any],
    accepted_metadata: list[Mapping[str, Any]],
    accepted_raw_max_abs_z: Mapping[str, float],
    source_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the immutable A-label-only QC candidate from train-only evidence."""

    if not accepted_metadata:
        raise ValueError("threshold derivation requires accepted Development rows")
    if measurement_calibration.get("boundary_policy") != "label_only_qc_v1":
        raise ValueError("threshold derivation only supports the selected A policy")
    payload = deepcopy(dict(measurement_calibration))
    payload["status"] = "frozen"
    payload["calibration_role"] = "qc_holdout_candidate_not_gate_eligible"

    for modality in S2_MODALITIES:
        raw_rows = [metadata["raw_qc"]["modalities"][modality] for metadata in accepted_metadata]
        raw_threshold = payload["raw_qc"]["modalities"][modality]
        raw_threshold["residual_quantile_intervals"] = {
            name: expanded_interval(
                (row["quantiles"][name] for row in raw_rows),
                label=f"raw {modality} {name}",
                expansion_iqr=1.0,
            )
            for name in QUANTILE_NAMES
        }
        raw_threshold["extreme_abs_z"] = (
            float(accepted_raw_max_abs_z[modality]) * 1.05 + 1.0e-6
        )
        raw_threshold.update(
            {
                "max_extreme_fraction": 0.02,
                "max_component_voxels": 512,
                "max_bbox_fill_ratio": 0.95,
                "max_axis_ratio": 16.0,
                "max_plane_fraction": 0.50,
            }
        )

        content_rows = [
            metadata["candidate_qc"]["content"]["modalities"][modality]
            for metadata in accepted_metadata
        ]
        content_threshold = payload["candidate_qc"]["modalities"][modality]
        content_threshold["residual_retention"] = expanded_interval(
            (row["residual_retention"] for row in content_rows),
            label=f"candidate {modality} retention",
            expansion_iqr=0.1,
            lower_limit=0.0,
        )
        content_threshold["candidate_abs_z_q99"] = max(
            float(row["candidate_abs_z_q99"]) for row in content_rows
        ) * 1.10 + 1.0e-6

    for threshold in payload["boundary_qc"]["thresholds"]:
        label_value = int(threshold["label"])
        modality = str(threshold["modality"])
        key = f"{label_value}:{modality}"
        rows = []
        for metadata in accepted_metadata:
            value = metadata["candidate_qc"]["boundary"]["strata"][key]
            if value.get("status") != "not_present":
                rows.append(value)
        if not rows:
            raise ValueError(f"accepted Development lacks boundary stratum {key}")
        threshold["ks_signed_max"] = min(
            1.0,
            max(float(row["ks_signed"]) for row in rows) + 0.05,
        )
        threshold["ks_abs_max"] = min(
            1.0,
            max(float(row["ks_abs"]) for row in rows) + 0.05,
        )
        threshold["quantile_intervals"] = {
            name: expanded_interval(
                (row["quantiles"][name] for row in rows),
                label=f"boundary {key} {name}",
                expansion_iqr=1.0,
                lower_limit=0.0 if name.startswith("abs_") else None,
            )
            for name in threshold["quantile_intervals"]
        }
        signed_reference = _finite(
            threshold["reference_signed_values"], label=f"boundary {key} signed Reference"
        )
        absolute_reference = _finite(
            threshold["reference_abs_values"], label=f"boundary {key} abs Reference"
        )
        signed_span = max(float(np.ptp(signed_reference)), 1.0e-6)
        threshold["signed_envelope"] = [
            float(np.min(signed_reference) - 0.05 * signed_span),
            float(np.max(signed_reference) + 0.05 * signed_span),
        ]
        threshold["abs_upper"] = float(np.max(absolute_reference) * 1.05 + 1.0e-6)
        threshold["max_abnormal_fraction"] = 0.10
        threshold["max_patch_area_mm2"] = max(
            4.0, max(float(row["area_mm2"]) for row in rows) * 0.10
        )
        threshold["max_patch_fraction"] = 0.10
        threshold["small_q95_abs_max"] = max(
            float(row["quantiles"]["abs_q95"]) for row in rows
        ) * 1.10 + 1.0e-6
        threshold["small_max_abs"] = max(
            float(row["quantiles"]["abs_q99"]) for row in rows
        ) * 1.25 + 1.0e-6
    payload["boundary_qc"]["event_max_ratio"] = 1.0

    for label_value in (1, 2, 3):
        reference_effects = _reference_effects(reference, label_value)
        development_effects = _class_metrics(accepted_metadata, label_value)
        config = payload["cross_modal_qc"]["classes"][str(label_value)]
        config["contrast_intervals"] = {}
        for modality in S2_MODALITIES:
            config["contrast_intervals"][modality] = reference_development_interval(
                (
                    value["modalities"][modality]["median_contrast"]
                    for value in reference_effects
                ),
                (
                    value["modalities"][modality]["median_contrast"]
                    for value in development_effects
                ),
                label=f"cross-modal {label_value} {modality}",
            )
        mean, inverse, maximum = _joint_model(reference_effects, development_effects)
        config["mean"] = mean
        config["inverse_covariance"] = inverse
        config["max_mahalanobis"] = maximum
        for pair in PAIR_KEYS:
            reference_pair = [value["pairwise"][pair] for value in reference_effects]
            development_pair = [value["pairwise"][pair] for value in development_effects]
            config["pairwise"][pair] = {
                "iou": reference_development_interval(
                    (value["iou"] for value in reference_pair),
                    (value["iou"] for value in development_pair),
                    label=f"cross-modal {label_value} {pair} IoU",
                    lower_limit=0.0,
                    upper_limit=1.0,
                ),
                "centroid_distance_mm": max(
                    float(
                        np.quantile(
                            _finite(
                                (value["centroid_distance_mm"] for value in reference_pair),
                                label=f"cross-modal {label_value} {pair} centroid Reference",
                            ),
                            0.999,
                            method="linear",
                        )
                    ),
                    max(float(value["centroid_distance_mm"]) for value in development_pair),
                )
                * 1.10
                + 1.0e-6,
            }

    payload["effective_rate_contract"] = {
        "p_select": 0.20,
        "minimum_generation_pass_rate": 0.80,
        "minimum_effective_aug_rate": 0.16,
        "no_resampling_after_qc_reject": True,
    }
    payload["inactive_policy_fields"] = ["harmonization", "halo_qc"]
    payload["threshold_derivation"] = {
        "schema_version": 1,
        "method": "reference_extreme_quantiles_plus_accepted_development_outer_envelopes_v1",
        "reference_tail_probabilities": [0.001, 0.999],
        "reference_covariance_winsorization": [0.005, 0.995],
        "covariance_diagonal_shrinkage": 0.20,
        "accepted_development_count": len(accepted_metadata),
        "source_bindings": dict(source_bindings),
    }
    payload["threshold_derivation"]["threshold_payload_sha256"] = canonical_json_sha256(
        {
            key: payload[key]
            for key in (
                "geometry",
                "raw_qc",
                "boundary_qc",
                "cross_modal_qc",
                "candidate_qc",
                "harmonization",
                "halo_qc",
                "effective_rate_contract",
            )
        }
    )
    payload["source_audit"].update(dict(source_bindings))
    FixV2Calibration.validate_payload(payload, expected_policy="label_only_qc_v1")
    return payload
