"""Minimal Fix-v3 QC aggregation layered on the frozen Fix-v2 measurements."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

try:
    from .met_aug_core import FIX_V3_PROCESSOR_POLICY, S2_MODALITIES
    from .met_aug_fix_v2 import (
        FixV2CandidateProcessor,
        FixV2Geometry,
        _Reject,
        _component_shape_metrics,
        _evaluate_boundary_faces,
        _extract_boundary_faces,
        _in_interval,
        _interval,
        _mask_alignment,
        _quantiles,
        _require_keys,
        _require_mapping,
    )
except ImportError:
    from met_aug_core import FIX_V3_PROCESSOR_POLICY, S2_MODALITIES  # type: ignore
    from met_aug_fix_v2 import (  # type: ignore
        FixV2CandidateProcessor,
        FixV2Geometry,
        _Reject,
        _component_shape_metrics,
        _evaluate_boundary_faces,
        _extract_boundary_faces,
        _in_interval,
        _interval,
        _mask_alignment,
        _quantiles,
        _require_keys,
        _require_mapping,
    )

FIX_V3_SMALL_BOUNDARY_AREA_MM2 = 128.0
FIX_V3_SEVERE_EXCURSION_WIDTHS = 1.0
FIX_V3_SEVERE_BOUNDARY_RATIO_MULTIPLIER = 2.0
FIX_V3_LARGE_ET_MIN_VOXELS = 2048
FIX_V3_LARGE_ET_T1C_CONTRAST_MIN = 0.25
FIX_V3_LARGE_ET_T1C_AFFECTED_FRACTION_MIN = 0.35


def _fix_v3_boundary_threshold(threshold: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(threshold)
    result["min_standard_area_mm2"] = max(
        float(result["min_standard_area_mm2"]),
        FIX_V3_SMALL_BOUNDARY_AREA_MM2,
    )
    return result


def _interval_excursion_widths(value: float, interval: Iterable[float]) -> float:
    lower, upper = (float(item) for item in interval)
    width = upper - lower
    if width <= 0:
        width = max(abs(lower), abs(upper), 1.0)
    deviation = max(lower - float(value), float(value) - upper, 0.0)
    return float(deviation / width)


def _boundary_failure_decision(
    failures: Iterable[str],
    *,
    ratio: float,
    event_limit: float,
) -> dict[str, Any]:
    family_by_failure = {
        "ks_signed": "ks",
        "ks_abs": "ks",
        "small_q95_abs": "absolute_quantile",
        "small_max_abs": "localized_patch",
        "abnormal_fraction": "localized_patch",
        "max_patch_area_mm2": "localized_patch",
        "max_patch_fraction": "localized_patch",
    }
    soft_families: set[str] = set()
    hard_failures: set[str] = set()
    observed = sorted(set(str(value) for value in failures))
    for failure in observed:
        if failure.startswith("signed_q"):
            soft_families.add("signed_quantile")
        elif failure.startswith("abs_q") or failure == "small_q95_abs":
            soft_families.add("absolute_quantile")
        else:
            family = family_by_failure.get(failure)
            if family == "localized_patch" or family is None:
                hard_failures.add(failure)
            else:
                soft_families.add(family)
    limit = float(event_limit)
    severe_ratio = bool(
        limit > 0
        and float(ratio)
        > limit * FIX_V3_SEVERE_BOUNDARY_RATIO_MULTIPLIER
    )
    reject = bool(hard_failures) or len(soft_families) >= 2 or severe_ratio
    return {
        "reject": reject,
        "observed_failures": observed,
        "soft_families": sorted(soft_families),
        "hard_failures": sorted(hard_failures),
        "severe_ratio": severe_ratio,
    }


def _raw_failure_decision(
    *,
    failures: Iterable[str],
    quantiles: Mapping[str, float],
    intervals: Mapping[str, Iterable[float]],
) -> dict[str, Any]:
    family_by_quantile = {
        "q01": "lower_tail",
        "q05": "lower_tail",
        "q50": "center",
        "q90": "upper_tail",
        "q95": "upper_tail",
        "q99": "upper_tail",
    }
    hard_names = {
        "extreme_fraction",
        "max_component_voxels",
        "max_bbox_fill_ratio",
        "max_axis_ratio",
        "max_plane_fraction",
    }
    observed = sorted(set(str(value) for value in failures))
    soft_families: set[str] = set()
    hard_failures: set[str] = set()
    severe_quantiles: set[str] = set()
    for failure in observed:
        family = family_by_quantile.get(failure)
        if family is None:
            if failure in hard_names or failure not in intervals:
                hard_failures.add(failure)
            continue
        soft_families.add(family)
        if failure not in quantiles or failure not in intervals:
            hard_failures.add(failure)
            continue
        if _interval_excursion_widths(
            float(quantiles[failure]), intervals[failure]
        ) >= FIX_V3_SEVERE_EXCURSION_WIDTHS:
            severe_quantiles.add(failure)
    reject = (
        bool(hard_failures)
        or bool(severe_quantiles)
        or len(soft_families) >= 2
    )
    return {
        "reject": reject,
        "observed_failures": observed,
        "soft_families": sorted(soft_families),
        "hard_failures": sorted(hard_failures),
        "severe_quantiles": sorted(severe_quantiles),
    }


def _content_failure_decision(
    modality_metrics: Mapping[str, Mapping[str, Any]],
    thresholds: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    failed_modalities: set[str] = set()
    severe_failures: set[str] = set()
    multi_signal_modalities: set[str] = set()
    for modality, metrics in modality_metrics.items():
        failures = sorted(set(str(value) for value in metrics.get("failures", ())))
        if not failures:
            continue
        failed_modalities.add(modality)
        if len(failures) >= 2:
            multi_signal_modalities.add(modality)
        config = thresholds[modality]
        if "candidate_abs_z_q99" in failures:
            limit = float(config["candidate_abs_z_q99"])
            value = float(metrics["candidate_abs_z_q99"])
            if value > max(limit * 2.0, limit + np.finfo(np.float64).eps):
                severe_failures.add(f"{modality}:candidate_abs_z_q99")
        if "residual_retention" in failures:
            excursion = _interval_excursion_widths(
                float(metrics["residual_retention"]),
                config["residual_retention"],
            )
            if excursion >= FIX_V3_SEVERE_EXCURSION_WIDTHS:
                severe_failures.add(f"{modality}:residual_retention")
    reject = (
        len(failed_modalities) >= 2
        or bool(severe_failures)
        or bool(multi_signal_modalities)
    )
    return {
        "reject": reject,
        "failed_modalities": sorted(failed_modalities),
        "severe_failures": sorted(severe_failures),
        "multi_signal_modalities": sorted(multi_signal_modalities),
    }


def _large_region_low_salience_failure(
    *,
    label_value: int,
    support_voxels: int,
    modality_metrics: Mapping[str, Mapping[str, Any]],
) -> bool:
    if int(label_value) != 3 or int(support_voxels) < FIX_V3_LARGE_ET_MIN_VOXELS:
        return False
    t1c = modality_metrics.get("t1c")
    if not isinstance(t1c, Mapping):
        return True
    return bool(
        float(t1c["median_contrast"]) < FIX_V3_LARGE_ET_T1C_CONTRAST_MIN
        and float(t1c["affected_fraction"])
        < FIX_V3_LARGE_ET_T1C_AFFECTED_FRACTION_MIN
    )


class FixV3CandidateProcessor(FixV2CandidateProcessor):
    """Fix-v2 measurements with less correlated vetoes and one narrow hard check."""

    processor_policy = FIX_V3_PROCESSOR_POLICY

    def _raw_qc(
        self,
        *,
        original: np.ndarray,
        generated: np.ndarray,
        geometry: FixV2Geometry,
        scales: Mapping[str, float],
    ) -> Mapping[str, Any]:
        result: dict[str, Any] = {
            "status": "pass",
            "processor_policy": self.processor_policy,
            "modalities": {},
        }
        config = self.calibration.payload["raw_qc"]["modalities"]
        event_soft_families: set[str] = set()
        event_hard_failures: set[str] = set()
        event_severe_quantiles: set[str] = set()
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
                    raise ValueError(f"unsupported raw quantile: {name}")
                if not _in_interval(
                    quantiles[name],
                    _interval(interval_value, label=f"raw {modality} {name}"),
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
            checks = {"extreme_fraction": extreme_fraction, **shape}
            if extreme_fraction > float(threshold["max_extreme_fraction"]):
                failures.append("extreme_fraction")
            for metric in (
                "max_component_voxels",
                "max_bbox_fill_ratio",
                "max_axis_ratio",
                "max_plane_fraction",
            ):
                if checks[metric] > float(threshold[metric]):
                    failures.append(metric)
            normalized_intervals = {
                name: _interval(value, label=f"raw {modality} {name}")
                for name, value in intervals.items()
            }
            decision = _raw_failure_decision(
                failures=failures,
                quantiles=quantiles,
                intervals=normalized_intervals,
            )
            event_soft_families.update(decision["soft_families"])
            event_hard_failures.update(
                f"{modality}:{value}" for value in decision["hard_failures"]
            )
            event_severe_quantiles.update(
                f"{modality}:{value}" for value in decision["severe_quantiles"]
            )
            result["modalities"][modality] = {
                "quantiles": quantiles,
                **checks,
                "failures": sorted(set(failures)),
                "decision": decision,
            }
        reject = (
            bool(event_hard_failures)
            or bool(event_severe_quantiles)
            or len(event_soft_families) >= 2
        )
        result["decision"] = {
            "reject": reject,
            "soft_families": sorted(event_soft_families),
            "hard_failures": sorted(event_hard_failures),
            "severe_quantiles": sorted(event_severe_quantiles),
        }
        if reject:
            raise _Reject(
                "RAW_GENERATION_QC_FAIL",
                f"Fix-v3 raw generation failures: {result['decision']}",
            )
        return result

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
        report: dict[str, Any] = {
            "status": "pass",
            "processor_policy": self.processor_policy,
            "strata": {},
        }
        max_ratio = 0.0
        event_failures: set[str] = set()
        event_limit = float(self.calibration.payload["boundary_qc"]["event_max_ratio"])
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
                threshold = _fix_v3_boundary_threshold(
                    self._select_boundary_threshold(
                        label_value=label_value,
                        modality=modality,
                        core_volume_mm3=core_volume_mm3,
                        boundary_area_mm2=area,
                    )
                )
                measured, ratio = _evaluate_boundary_faces(
                    faces,
                    threshold,
                    geometry.label_support.shape,
                )
                metrics = dict(measured)
                original_status = str(metrics["status"])
                decision = _boundary_failure_decision(
                    metrics.get("failures", ()),
                    ratio=ratio,
                    event_limit=event_limit,
                )
                metrics["fix_v2_status"] = original_status
                metrics["status"] = "fail" if decision["reject"] else "pass"
                metrics["decision"] = decision
                report["strata"][f"{label_value}:{modality}"] = metrics
                max_ratio = max(max_ratio, ratio)
                event_failures.update(str(value) for value in metrics["failures"])
        event_decision = _boundary_failure_decision(
            event_failures,
            ratio=max_ratio,
            event_limit=event_limit,
        )
        report["event_max_ratio"] = max_ratio
        report["event_max_ratio_limit"] = event_limit
        report["decision"] = event_decision
        if event_decision["reject"]:
            raise _Reject(
                "CANDIDATE_BOUNDARY_QC_FAIL",
                f"Fix-v3 boundary failures: {event_decision}",
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
        report: dict[str, Any] = {
            "status": "pass",
            "processor_policy": self.processor_policy,
            "modalities": {},
        }
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
            if raw_q95 <= self.calibration.epsilon:
                retention = (
                    1.0
                    if candidate_q95 <= self.calibration.epsilon
                    else float("inf")
                )
            else:
                retention = candidate_q95 / raw_q95
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
        decision = _content_failure_decision(report["modalities"], config)
        report["decision"] = decision
        if decision["reject"]:
            raise _Reject(
                "CANDIDATE_CONTENT_QC_FAIL",
                f"Fix-v3 candidate content failures: {decision}",
            )
        return report

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
        report: dict[str, Any] = {
            "status": "pass",
            "processor_policy": self.processor_policy,
            "classes": {},
        }
        classes = self.calibration.payload["cross_modal_qc"]["classes"]
        event_families: set[str] = set()
        severe_failures: set[str] = set()
        hard_failures: set[str] = set()
        for label_value in (1, 2, 3):
            support = label_cube == label_value
            support_voxels = int(np.count_nonzero(support))
            if support_voxels == 0:
                report["classes"][str(label_value)] = {"status": "not_present"}
                continue
            config = classes[str(label_value)]
            if support_voxels < int(config["minimum_voxels"]):
                raise _Reject(
                    failure_reason,
                    f"label {label_value} has insufficient cross-modal support",
                )
            contrasts: list[float] = []
            affected_masks: dict[str, np.ndarray] = {}
            modality_metrics: dict[str, Any] = {}
            failures: list[str] = []
            class_families: set[str] = set()
            class_severe: set[str] = set()
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
                    failure = f"{modality}:contrast"
                    failures.append(failure)
                    class_families.add("contrast")
                    if _interval_excursion_widths(contrast, interval) >= (
                        FIX_V3_SEVERE_EXCURSION_WIDTHS
                    ):
                        class_severe.add(failure)
                affected = support & (
                    np.abs(normalized)
                    >= float(config["affected_abs_threshold"][modality])
                )
                affected_masks[modality] = affected
                modality_metrics[modality] = {
                    "median_contrast": contrast,
                    "affected_fraction": float(
                        np.count_nonzero(affected) / support_voxels
                    ),
                }
            vector = np.asarray(contrasts, dtype=np.float64)
            mean = np.asarray(config["mean"], dtype=np.float64)
            inverse = np.asarray(config["inverse_covariance"], dtype=np.float64)
            difference = vector - mean
            squared = float(difference @ inverse @ difference)
            mahalanobis = float(np.sqrt(max(0.0, squared)))
            mahalanobis_limit = float(config["max_mahalanobis"])
            if mahalanobis > mahalanobis_limit:
                failures.append("mahalanobis")
                class_families.add("effect_vector")
                if mahalanobis > max(
                    mahalanobis_limit * 2.0,
                    mahalanobis_limit + np.finfo(np.float64).eps,
                ):
                    class_severe.add("mahalanobis")
            pair_metrics: dict[str, Any] = {}
            for pair, limits in config["pairwise"].items():
                left, right = pair.split(":", 1)
                iou, centroid_distance = _mask_alignment(
                    affected_masks[left], affected_masks[right], spacing_mm
                )
                iou_interval = _interval(limits["iou"], label=f"{pair} iou")
                if not _in_interval(iou, iou_interval):
                    failures.append(f"{pair}:iou")
                    class_families.add("spatial_overlap")
                    if _interval_excursion_widths(iou, iou_interval) >= (
                        FIX_V3_SEVERE_EXCURSION_WIDTHS
                    ):
                        class_severe.add(f"{pair}:iou")
                centroid_limit = float(limits["centroid_distance_mm"])
                if centroid_distance > centroid_limit:
                    failures.append(f"{pair}:centroid")
                    class_families.add("spatial_offset")
                    if centroid_distance > max(
                        centroid_limit * 2.0,
                        centroid_limit + np.finfo(np.float64).eps,
                    ):
                        class_severe.add(f"{pair}:centroid")
                pair_metrics[pair] = {
                    "iou": iou,
                    "centroid_distance_mm": centroid_distance,
                }
            low_salience = _large_region_low_salience_failure(
                label_value=label_value,
                support_voxels=support_voxels,
                modality_metrics=modality_metrics,
            )
            if low_salience:
                hard_failures.add(f"label_{label_value}:large_et_low_salience")
            event_families.update(class_families)
            severe_failures.update(
                f"label_{label_value}:{value}" for value in class_severe
            )
            class_reject = (
                low_salience
                or bool(class_severe)
                or len(class_families) >= 2
            )
            report["classes"][str(label_value)] = {
                "status": "fail" if class_reject else "pass",
                "modalities": modality_metrics,
                "effect_vector": contrasts,
                "mahalanobis": mahalanobis,
                "pairwise": pair_metrics,
                "failures": sorted(set(failures)),
                "failure_families": sorted(class_families),
                "severe_failures": sorted(class_severe),
                "large_et_low_salience": low_salience,
            }
        reject = (
            bool(hard_failures)
            or bool(severe_failures)
            or len(event_families) >= 2
        )
        report["decision"] = {
            "reject": reject,
            "failure_families": sorted(event_families),
            "severe_failures": sorted(severe_failures),
            "hard_failures": sorted(hard_failures),
        }
        if reject:
            details = sorted(hard_failures | severe_failures | event_families)
            raise _Reject(failure_reason, f"Fix-v3 cross-modal failures: {details}")
        return report
