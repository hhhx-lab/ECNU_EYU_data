"""Exploratory 125-epoch MET-AUG B-F screening routes.

The completed Fix-v3 Route A runtime remains immutable.  This module wraps its
validated engine with one explicitly frozen screening variable per route.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import ndimage

try:
    from .met_aug_core import (
        ALLOWED_LABELS,
        CORE_LABELS,
        ComponentManifest,
        ComponentRecord,
        EventContext,
        EventResult,
        MetAugContractError,
        MetAugEngine,
        RouteConfig,
        _seed_from_event,
        canonical_json_sha256,
        find_placement,
        label_statistics,
        patient_group,
        physical_ball,
        sha256_file,
    )
except ImportError:
    from met_aug_core import (  # type: ignore
        ALLOWED_LABELS,
        CORE_LABELS,
        ComponentManifest,
        ComponentRecord,
        EventContext,
        EventResult,
        MetAugContractError,
        MetAugEngine,
        RouteConfig,
        _seed_from_event,
        canonical_json_sha256,
        find_placement,
        label_statistics,
        patient_group,
        physical_ball,
        sha256_file,
    )


SCREEN_SCHEMA = 1
SCREEN_ROUTES = frozenset({"MET-AUG-B", "MET-AUG-C40", "MET-AUG-D", "MET-AUG-E-SNFH", "MET-AUG-F"})
COMMON_RANDOM_ROUTE = "MET-AUG-BF-SCREEN-COMMON-20260831"
SECOND_TUMOUR_PATCH_OFFSET = 1_000_000_000


def _require_probability(value: Any, *, label: str) -> float:
    probability = float(value)
    if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise MetAugContractError(f"{label} must be a finite probability")
    return probability


@dataclass(frozen=True)
class ScreeningPolicy:
    path: Path
    route_id: str
    plan_sha256: str
    base_route_config_sha256: str
    p_select: float
    scale_policy: str
    second_tumour_probability: float
    topology_policy: str
    topology_probability: float
    target_counts_sha256: str | None
    burden_probabilities: Mapping[str, float]
    target_counts: Mapping[str, int]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        base_config: RouteConfig,
        target_counts_path: str | Path | None = None,
    ) -> "ScreeningPolicy":
        resolved = Path(path).expanduser().resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCREEN_SCHEMA:
            raise MetAugContractError("unsupported MET-AUG screening schema")
        expected_sha = str(payload.get("screen_config_sha256", ""))
        if expected_sha != canonical_json_sha256(payload, exclude=("screen_config_sha256",)):
            raise MetAugContractError("MET-AUG screening config SHA256 is not canonical")
        if payload.get("artifact_status") != "experimental_unvalidated":
            raise MetAugContractError("screening artifact status must remain experimental_unvalidated")
        if payload.get("operator_approved") is not False:
            raise MetAugContractError("screening operator_approved must remain false")
        if payload.get("formal_gate_status") != "not_run_not_passed":
            raise MetAugContractError("screening formal gate must remain not_run_not_passed")
        route_id = str(payload.get("route_id", ""))
        if route_id not in SCREEN_ROUTES:
            raise MetAugContractError(f"unsupported MET-AUG screening route: {route_id}")
        if payload.get("base_route_config_sha256") != sha256_file(base_config.path):
            raise MetAugContractError("screening config does not bind the Fix-v3 Route A config")
        plan_sha = str(payload.get("plan_sha256", ""))
        if len(plan_sha) != 64:
            raise MetAugContractError("screening config does not bind the 20260831 plan")

        target_counts: dict[str, int] = {}
        counts_sha = payload.get("target_counts_sha256")
        if route_id == "MET-AUG-F":
            if target_counts_path is None:
                raise MetAugContractError("MET-AUG-F requires frozen target lesion counts")
            counts_path = Path(target_counts_path).expanduser().resolve()
            if not counts_path.is_file() or sha256_file(counts_path) != counts_sha:
                raise MetAugContractError("MET-AUG-F target lesion counts are missing or drifted")
            counts_payload = json.loads(counts_path.read_text(encoding="utf-8"))
            if counts_payload.get("status") != "pass" or counts_payload.get("split") != "train_only":
                raise MetAugContractError("MET-AUG-F target lesion counts are not train-only pass evidence")
            target_counts = {
                str(key): int(value)
                for key, value in counts_payload.get("case_to_native_lesion_count", {}).items()
            }
            if set(target_counts) != set(base_config_manifest_cases(base_config)):
                raise MetAugContractError("MET-AUG-F lesion counts do not cover the frozen train targets")
        elif counts_sha is not None:
            raise MetAugContractError("only MET-AUG-F may bind target lesion counts")

        policy = cls(
            path=resolved,
            route_id=route_id,
            plan_sha256=plan_sha,
            base_route_config_sha256=str(payload["base_route_config_sha256"]),
            p_select=_require_probability(payload.get("p_select"), label="p_select"),
            scale_policy=str(payload.get("scale_policy", "none")),
            second_tumour_probability=_require_probability(
                payload.get("second_tumour_probability", 0.0),
                label="second_tumour_probability",
            ),
            topology_policy=str(payload.get("topology_policy", "none")),
            topology_probability=_require_probability(
                payload.get("topology_probability", 0.0),
                label="topology_probability",
            ),
            target_counts_sha256=str(counts_sha) if counts_sha is not None else None,
            burden_probabilities={
                str(key): _require_probability(value, label=f"burden probability {key}")
                for key, value in payload.get("burden_probabilities", {}).items()
            },
            target_counts=target_counts,
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        expected = {
            "MET-AUG-B": (0.20, "core_volume_controlled_v1", 0.0, "none", 0.0),
            "MET-AUG-C40": (0.40, "none", 0.0, "none", 0.0),
            "MET-AUG-D": (0.20, "none", 0.20, "none", 0.0),
            "MET-AUG-E-SNFH": (0.20, "none", 0.0, "pure_et_snfh_shell_1mm_v1", 0.10),
            "MET-AUG-F": (0.20, "none", 0.0, "none", 0.0),
        }[self.route_id]
        observed = (
            self.p_select,
            self.scale_policy,
            self.second_tumour_probability,
            self.topology_policy,
            self.topology_probability,
        )
        if observed != expected:
            raise MetAugContractError(
                f"screening route contract drifted for {self.route_id}: {observed!r}"
            )
        if self.route_id == "MET-AUG-F":
            if set(self.burden_probabilities) != {"0_1", "2_4", "5_9", "10_plus"}:
                raise MetAugContractError("MET-AUG-F requires all four burden probabilities")
            if not self.target_counts:
                raise MetAugContractError("MET-AUG-F target counts are empty")
            mean_probability = float(
                np.mean([self.selection_probability(case_id) for case_id in self.target_counts])
            )
            if abs(mean_probability - 0.20) > 1e-12:
                raise MetAugContractError(
                    f"MET-AUG-F mean p_select must equal 0.20, observed {mean_probability:.12f}"
                )
        elif self.burden_probabilities or self.target_counts:
            raise MetAugContractError("non-F routes cannot contain burden allocation")

    def selection_probability(self, case_id: str) -> float:
        if self.route_id != "MET-AUG-F":
            return self.p_select
        if case_id not in self.target_counts:
            raise MetAugContractError(f"MET-AUG-F target count is missing: {case_id}")
        count = int(self.target_counts[case_id])
        key = "0_1" if count <= 1 else "2_4" if count <= 4 else "5_9" if count <= 9 else "10_plus"
        return float(self.burden_probabilities[key])


def base_config_manifest_cases(config: RouteConfig) -> tuple[str, ...]:
    """Return cases attached at runtime by ``ScreenMetAugEngine``.

    RouteConfig intentionally does not own the manifest.  The loader replaces
    this function for its one validation call through a temporary attribute.
    """
    cases = getattr(config, "_screen_manifest_cases", None)
    if cases is None:
        raise MetAugContractError("screening config was loaded without manifest cases")
    return tuple(str(value) for value in cases)


def load_screening_policy(
    path: str | Path,
    *,
    base_config: RouteConfig,
    manifest: ComponentManifest,
    target_counts_path: str | Path | None = None,
) -> ScreeningPolicy:
    # Frozen dataclasses reject normal assignment; the temporary binding is
    # removed immediately after validation and never enters route provenance.
    object.__setattr__(base_config, "_screen_manifest_cases", tuple(manifest.target_groups))
    try:
        return ScreeningPolicy.load(
            path,
            base_config=base_config,
            target_counts_path=target_counts_path,
        )
    finally:
        object.__delattr__(base_config, "_screen_manifest_cases")


def _updated_record(record: ComponentRecord, label: np.ndarray) -> ComponentRecord:
    stats = label_statistics(label, record.spacing_mm)
    return replace(
        record,
        core_volume_mm3=float(stats["core_volume_mm3"]),
        total_volume_mm3=float(stats["total_volume_mm3"]),
        bbox_mm=tuple(float(value) for value in stats["bbox_mm"]),
        bbox_voxels=tuple(int(value) for value in stats["bbox_voxels"]),
        class_counts={str(key): int(value) for key, value in stats["class_counts"].items()},
        classes_present=tuple(int(value) for value in stats["classes_present"]),
    )


def controlled_scale_label(
    label: np.ndarray,
    record: ComponentRecord,
    rng: np.random.Generator,
    *,
    max_attempts: int = 8,
) -> tuple[np.ndarray | None, ComponentRecord | None, Mapping[str, Any]]:
    volume = float(record.core_volume_mm3)
    if volume <= 49.0:
        lower, upper = 1.0, 1.728
    elif volume <= 500.0:
        lower, upper = max(27.0 / volume, 0.50), 1.25
    else:
        lower, upper = max(27.0 / volume, 0.15), 0.60
    original_classes = tuple(sorted(int(value) for value in np.unique(label[label != 0])))
    for attempt in range(1, max_attempts + 1):
        volume_ratio = float(rng.uniform(lower, upper))
        linear_scale = float(volume_ratio ** (1.0 / 3.0))
        scaled = ndimage.zoom(label, zoom=linear_scale, order=0, mode="nearest", prefilter=False)
        scaled = scaled.astype(np.int16, copy=False)
        if not np.any(scaled):
            continue
        classes = tuple(sorted(int(value) for value in np.unique(scaled[scaled != 0])))
        if classes != original_classes or any(int(value) > 56 for value in scaled.shape):
            continue
        updated = _updated_record(record, scaled)
        if updated.core_volume_mm3 < 27.0 or any(value > 56.0 + 1e-8 for value in updated.bbox_mm):
            continue
        return scaled, updated, {
            "screen_transform": "core_volume_controlled_v1",
            "scale_attempt": attempt,
            "target_volume_ratio": volume_ratio,
            "linear_scale": linear_scale,
            "observed_core_volume_mm3": updated.core_volume_mm3,
        }
    return None, None, {
        "screen_transform": "core_volume_controlled_v1",
        "scale_attempts": max_attempts,
    }


def add_snfh_shell(
    label: np.ndarray,
    record: ComponentRecord,
) -> tuple[np.ndarray | None, ComponentRecord | None, Mapping[str, Any]]:
    if tuple(record.classes_present) != (3,):
        return label, record, {"screen_transform": "none", "topology_eligible": False}
    padded = np.pad(label.astype(np.int16, copy=False), 2, mode="constant")
    support = padded != 0
    shell = ndimage.binary_dilation(
        support,
        structure=physical_ball(1.0, record.spacing_mm),
        border_value=0,
    ) & ~support
    if not np.any(shell):
        return None, None, {"screen_transform": "pure_et_snfh_shell_1mm_v1", "detail": "empty shell"}
    candidate = padded.copy()
    candidate[shell] = 2
    nonzero = np.argwhere(candidate != 0)
    starts = nonzero.min(axis=0)
    stops = nonzero.max(axis=0) + 1
    candidate = candidate[tuple(slice(int(a), int(b)) for a, b in zip(starts, stops))]
    if any(int(value) > 56 for value in candidate.shape):
        return None, None, {"screen_transform": "pure_et_snfh_shell_1mm_v1", "detail": "bbox exceeds crop"}
    updated = _updated_record(record, candidate)
    if tuple(updated.classes_present) != (2, 3):
        return None, None, {"screen_transform": "pure_et_snfh_shell_1mm_v1", "detail": "class contract"}
    return candidate, updated, {
        "screen_transform": "pure_et_snfh_shell_1mm_v1",
        "topology_eligible": True,
        "added_snfh_voxels": int(np.count_nonzero(shell)),
    }


class ScreenMetAugEngine(MetAugEngine):
    """Fix-v3 engine with exactly one exploratory B-F variable enabled."""

    def __init__(self, *, screening_policy: ScreeningPolicy, **kwargs) -> None:
        super().__init__(**kwargs)
        self.screening_policy = screening_policy
        self._force_second_tumour = False

    def _event_identity(self, context: EventContext) -> tuple[str, int]:
        seed = _seed_from_event(
            global_seed=self.config.seed,
            epoch=context.epoch,
            rank=context.rank,
            worker=context.worker,
            case_id=context.case_id,
            patch_index=context.patch_index,
            route_id=COMMON_RANDOM_ROUTE,
        )
        event_id = hashlib.sha256(
            f"{self.screening_policy.route_id}|{context.case_id}|{context.epoch}|{context.rank}|{context.worker}|{context.patch_index}".encode("utf-8")
        ).hexdigest()[:24]
        return event_id, seed

    def _prepare_label(
        self,
        label: np.ndarray,
        record: ComponentRecord,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray | None, ComponentRecord | None, Mapping[str, Any], str | None]:
        route = self.screening_policy.route_id
        if route == "MET-AUG-B":
            scaled, updated, metadata = controlled_scale_label(label, record, rng)
            return scaled, updated, metadata, None if scaled is not None else "B_SCALE_NO_OP"
        if route == "MET-AUG-E-SNFH" and float(rng.random()) < self.screening_policy.topology_probability:
            candidate, updated, metadata = add_snfh_shell(label, record)
            return candidate, updated, metadata, None if candidate is not None else "E_TOPOLOGY_FALLBACK"
        return label, record, {"screen_transform": "none"}, None

    def plan(
        self,
        *,
        segmentation: np.ndarray,
        valid_mask: np.ndarray,
        context: EventContext,
        inputs_prevalidated: bool = False,
    ) -> EventResult:
        event_id, seed = self._event_identity(context)
        target_group = self.manifest.target_groups.get(context.case_id)
        if target_group is None:
            raise MetAugContractError(f"target case is absent from frozen train group map: {context.case_id}")
        if target_group != patient_group(target_group) and target_group != patient_group(context.case_id):
            raise MetAugContractError(f"target group has malformed value for {context.case_id}: {target_group}")
        if segmentation.ndim == 4 and segmentation.shape[0] == 1:
            segmentation = segmentation[0]
        if segmentation.ndim != 3 or valid_mask.shape != segmentation.shape:
            raise MetAugContractError("screening route requires matching segmentation and valid-mask patches")
        if not inputs_prevalidated and not set(int(value) for value in np.unique(segmentation)).issubset(ALLOWED_LABELS | {-1}):
            raise MetAugContractError("target segmentation has invalid classes")
        rng = np.random.default_rng(seed)
        probability = 1.0 if self._force_second_tumour else self.screening_policy.selection_probability(context.case_id)
        base_metadata: dict[str, Any] = {
            "route_id": self.screening_policy.route_id,
            "target_case_id": context.case_id,
            "target_patient_group": target_group,
            "epoch": context.epoch,
            "rank": context.rank,
            "worker": context.worker,
            "patch_index": context.patch_index,
            "selection_probability": probability,
            "tumour_index": 2 if self._force_second_tumour else 1,
        }
        if float(rng.random()) >= probability:
            return EventResult("NO_OP", "NOT_SELECTED", event_id, seed, metadata=base_metadata)
        record = self.sampler.choose(rng, target_group)
        if record is None:
            return EventResult("NO_OP", "NO_ELIGIBLE_DONOR", event_id, seed, metadata=base_metadata)
        if record.patient_group == target_group:
            raise MetAugContractError("same patient-group donor escaped sampler")
        try:
            label = self.sampler.load_label(record)
            label, runtime_record, transform_metadata, rejection = self._prepare_label(label, record, rng)
        except (OSError, ValueError, MetAugContractError) as exc:
            return EventResult(
                "NO_OP", "LABEL_INVALID", event_id, seed, record=record,
                metadata={**base_metadata, "detail": str(exc)},
            )
        if rejection is not None or label is None or runtime_record is None:
            return EventResult(
                "NO_OP", rejection or "LABEL_INVALID", event_id, seed, record=record,
                metadata={**base_metadata, **transform_metadata},
            )
        stats = label_statistics(label, runtime_record.spacing_mm)
        if (
            stats["core_volume_mm3"] < self.config.min_core_volume_mm3
            or any(value > self.config.max_bbox_mm + 1e-8 for value in stats["bbox_mm"])
            or tuple(stats["classes_present"]) != runtime_record.classes_present
        ):
            return EventResult("NO_OP", "LABEL_INVALID", event_id, seed, record=runtime_record, metadata=base_metadata)
        core_voxels = sum(int(stats["class_counts"].get(str(value), 0)) for value in CORE_LABELS)
        if not self.config.is_support_eligible(
            total_support_voxels=int(stats["support_voxels"]),
            core_voxels=core_voxels,
        ):
            return EventResult(
                "NO_OP", "SCREEN_SUPPORT_INELIGIBLE", event_id, seed, record=runtime_record,
                metadata={**base_metadata, **transform_metadata},
            )
        placement = find_placement(
            label=label,
            record=runtime_record,
            segmentation=segmentation,
            valid_mask=valid_mask,
            context=context,
            rng=rng,
            crop_size=self.config.crop_size,
            clearance_mm=self.config.clearance_mm,
            boundary_clearance_mm=self.config.boundary_clearance_mm,
        )
        if placement is None:
            return EventResult(
                "NO_OP", "NO_VALID_PLACEMENT", event_id, seed, record=runtime_record,
                metadata={**base_metadata, **transform_metadata},
            )
        return EventResult(
            "PLACEMENT_VALID", None, event_id, seed, record=runtime_record,
            placement=placement, metadata={**base_metadata, **transform_metadata},
        )

    def apply(self, *, image, segmentation, valid_mask, context):
        if self.screening_policy.route_id != "MET-AUG-D":
            return super().apply(
                image=image,
                segmentation=segmentation,
                valid_mask=valid_mask,
                context=context,
            )
        primary_context = replace(context, patch_index=int(context.patch_index) * 2)
        image_after, segmentation_after, primary = super().apply(
            image=image,
            segmentation=segmentation,
            valid_mask=valid_mask,
            context=primary_context,
        )
        if primary.state != "COMMITTED":
            return image_after, segmentation_after, primary
        second_context = replace(
            context,
            patch_index=int(context.patch_index) * 2 + SECOND_TUMOUR_PATCH_OFFSET,
        )
        second_event_id, second_seed = self._event_identity(second_context)
        second_rng = np.random.default_rng(primary.event_seed ^ 0x5EC0D)
        if float(second_rng.random()) >= self.screening_policy.second_tumour_probability:
            skipped = EventResult(
                "NO_OP",
                "SECOND_NOT_SELECTED",
                second_event_id,
                second_seed,
                metadata={
                    "route_id": self.screening_policy.route_id,
                    "target_case_id": context.case_id,
                    "epoch": context.epoch,
                    "patch_index": context.patch_index,
                    "tumour_index": 2,
                },
            )
            self._append(skipped)
            return image_after, segmentation_after, primary
        self._force_second_tumour = True
        try:
            return super().apply(
                image=image_after,
                segmentation=segmentation_after,
                valid_mask=valid_mask,
                context=second_context,
            )
        finally:
            self._force_second_tumour = False
