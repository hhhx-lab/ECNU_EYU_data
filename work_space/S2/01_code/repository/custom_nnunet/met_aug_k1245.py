"""Isolated RC-aware MET-AUG policies for the preregistered K1/K2/K4/K5 study.

The historical Route A, FixV3, and Screen125 contracts remain immutable.  This
module accepts only the new schema-1 K1245 manifests/configs and records every
fallback that is needed when an RC or topology request cannot be fulfilled.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy import ndimage

from .met_aug_core import (
    ALLOWED_LABELS,
    CandidateProcessingResult,
    ComponentManifest,
    ComponentRecord,
    EventContext,
    EventResult,
    MetAugContractError,
    MetAugEngine,
    Placement,
    S2_MODALITIES,
    _centered_label_cube,
    _seed_from_event,
    canonical_json_sha256,
    patient_group,
    physical_ball,
    sha256_file,
)
from .met_aug_fix_v2 import FixV2Geometry, _Reject, _extract_boundary_faces
from .met_aug_fix_v3 import FixV3CandidateProcessor


K1245_MANIFEST_SCHEMA = 1
K1245_CONFIG_SCHEMA = 1
K1245_PLAN_ROUTE = "MET-AUG-K1245-20260921"
ANCHOR_LABELS = frozenset({1, 3, 4})
LEGACY_ANCHOR_LABELS = frozenset({1, 3})
K_ROUTE_IDS = frozenset({"K1", "K2", "K4", "K5"})


def _labels(value: np.ndarray) -> set[int]:
    return set(int(item) for item in np.unique(value))


def anchor_statistics(
    label: np.ndarray,
    spacing_mm: Iterable[float],
    *,
    anchor_labels: frozenset[int] = ANCHOR_LABELS,
) -> dict[str, Any]:
    if label.ndim != 3:
        raise MetAugContractError(f"component label must be 3D, got {label.shape}")
    values = _labels(label)
    if not values.issubset(ALLOWED_LABELS):
        raise MetAugContractError(
            f"label contains unsupported classes: {sorted(values - ALLOWED_LABELS)}"
        )
    spacing = np.asarray(tuple(float(value) for value in spacing_mm), dtype=np.float64)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise MetAugContractError(f"invalid spacing: {tuple(spacing)}")
    support = label != 0
    anchor = np.isin(label, tuple(anchor_labels))
    if not np.any(anchor):
        raise MetAugContractError("component label has no configured anchor class")
    points = np.argwhere(support)
    lower = points.min(axis=0)
    upper = points.max(axis=0) + 1
    bbox_shape = upper - lower
    classes = tuple(sorted(int(value) for value in np.unique(label[support])))
    counts = {str(value): int(np.count_nonzero(label == value)) for value in classes}
    voxel_volume = float(np.prod(spacing))
    return {
        "classes_present": classes,
        "class_counts": counts,
        # Keep the historical field name for ComponentRecord compatibility.
        "core_volume_mm3": float(np.count_nonzero(anchor) * voxel_volume),
        "anchor_labels": sorted(anchor_labels),
        "total_volume_mm3": float(np.count_nonzero(support) * voxel_volume),
        "bbox_mm": tuple(float(value) for value in bbox_shape * spacing),
        "bbox_voxels": tuple(int(value) for value in bbox_shape),
        "support_voxels": int(np.count_nonzero(support)),
        "bbox_slices": tuple(slice(int(a), int(b)) for a, b in zip(lower, upper)),
    }


def extract_k1245_components(
    label: np.ndarray,
    spacing_mm: Iterable[float],
    *,
    min_anchor_volume_mm3: float,
    max_bbox_mm: float = 56.0,
    include_rc: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Extract 26-connected anchors and uniquely attach adjacent SNFH.

    RC-aware pools use NETC/ET/RC as anchors.  The X3 pool preserves the old
    non-RC source exclusion and uses NETC/ET anchors with an 8 mm3 floor.
    """
    if label.ndim != 3:
        raise MetAugContractError(f"source label must be 3D, got {label.shape}")
    values = _labels(label)
    if not values.issubset(ALLOWED_LABELS):
        raise MetAugContractError(
            f"source label contains unsupported classes: {sorted(values - ALLOWED_LABELS)}"
        )
    if not include_rc and 4 in values:
        return [], {"source_contains_rc": 1}
    anchors = ANCHOR_LABELS if include_rc else LEGACY_ANCHOR_LABELS
    spacing = tuple(float(value) for value in spacing_mm)
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise MetAugContractError(f"invalid spacing: {spacing}")
    structure = np.ones((3, 3, 3), dtype=bool)
    anchor_mask = np.isin(label, tuple(anchors))
    anchor_map, anchor_count = ndimage.label(anchor_mask, structure=structure)
    if anchor_count == 0:
        return [], {"no_anchor": 1}
    attached: dict[int, np.ndarray] = {
        index: np.zeros_like(anchor_mask, dtype=bool)
        for index in range(1, anchor_count + 1)
    }
    exclusions: dict[str, int] = defaultdict(int)
    snfh_map, snfh_count = ndimage.label(label == 2, structure=structure)
    for index in range(1, snfh_count + 1):
        snfh = snfh_map == index
        touching = np.unique(anchor_map[ndimage.binary_dilation(snfh, structure=structure)])
        touching = touching[touching != 0]
        if len(touching) == 1:
            attached[int(touching[0])] |= snfh
        elif len(touching) == 0:
            exclusions["unattached_snfh"] += 1
        else:
            exclusions["ambiguous_snfh"] += 1
    result: list[dict[str, Any]] = []
    for index in range(1, anchor_count + 1):
        component_mask = (anchor_map == index) | attached[index]
        component = np.zeros_like(label, dtype=np.int16)
        component[component_mask] = label[component_mask]
        stats = anchor_statistics(component, spacing, anchor_labels=anchors)
        if stats["core_volume_mm3"] < float(min_anchor_volume_mm3):
            exclusions["anchor_below_floor"] += 1
            continue
        if any(value > float(max_bbox_mm) + 1e-8 for value in stats["bbox_mm"]):
            exclusions["bbox_exceeds_limit"] += 1
            continue
        bbox = stats.pop("bbox_slices")
        anchor_points = np.argwhere(np.isin(component, tuple(anchors)))
        result.append(
            {
                "label": component[bbox].copy(),
                "stats": stats,
                "core_centroid_norm": tuple(
                    float(value)
                    for value in anchor_points.mean(axis=0)
                    / np.asarray(label.shape, dtype=np.float64)
                ),
                "source_bbox_voxels": tuple(
                    (int(part.start), int(part.stop)) for part in bbox
                ),
            }
        )
    return result, dict(exclusions)


@dataclass(frozen=True)
class K1245Manifest:
    path: Path
    root: Path
    identity_sha256: str
    records_sha256: str
    records: tuple[ComponentRecord, ...]
    target_groups_path: Path
    target_groups_sha256: str
    target_groups: Mapping[str, str]
    include_rc: bool
    min_anchor_volume_mm3: float
    max_bbox_mm: float

    @classmethod
    def load(cls, path: str | Path) -> "K1245Manifest":
        resolved = Path(path).expanduser().resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", -1)) != K1245_MANIFEST_SCHEMA:
            raise MetAugContractError("unsupported K1245 component manifest schema")
        expected = str(payload.get("manifest_sha256", ""))
        actual = canonical_json_sha256(payload, exclude=("manifest_sha256",))
        if expected != actual:
            raise MetAugContractError("K1245 component manifest SHA256 mismatch")
        if payload.get("coordinate_space") != "nnUNetPlans_3d_fullres_preprocessed":
            raise MetAugContractError("K1245 manifest coordinate space drifted")
        constraints = payload.get("constraints", {})
        include_rc = bool(constraints.get("include_rc"))
        anchors = tuple(int(value) for value in constraints.get("anchor_labels", ()))
        expected_anchors = tuple(sorted(ANCHOR_LABELS if include_rc else LEGACY_ANCHOR_LABELS))
        if anchors != expected_anchors:
            raise MetAugContractError("K1245 anchor-label contract drifted")
        minimum = float(constraints.get("min_anchor_volume_mm3", -1))
        maximum = float(constraints.get("max_bbox_mm", -1))
        if minimum not in {8.0, 27.0} or maximum != 56.0:
            raise MetAugContractError("K1245 component size contract drifted")
        records_path = resolved.parent / str(payload.get("records_file"))
        groups_path = resolved.parent / str(payload.get("target_groups_file"))
        if sha256_file(records_path) != payload.get("records_sha256"):
            raise MetAugContractError("K1245 component records SHA256 mismatch")
        if sha256_file(groups_path) != payload.get("target_groups_sha256"):
            raise MetAugContractError("K1245 target groups SHA256 mismatch")
        records = tuple(
            ComponentRecord.from_mapping(json.loads(line))
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if not records or len(records) != int(payload.get("component_count", -1)):
            raise MetAugContractError("K1245 component count mismatch")
        for record in records:
            _validate_k_record(
                record,
                include_rc=include_rc,
                min_anchor_volume_mm3=minimum,
                max_bbox_mm=maximum,
            )
        groups_payload = json.loads(groups_path.read_text(encoding="utf-8"))
        if groups_payload.get("schema_version") != 1:
            raise MetAugContractError("K1245 target-group schema drifted")
        groups = {
            str(key): str(value)
            for key, value in groups_payload.get("case_to_patient_group", {}).items()
        }
        if not groups:
            raise MetAugContractError("K1245 target-group map is empty")
        if include_rc and not any(4 in record.classes_present for record in records):
            raise MetAugContractError("RC-aware K1245 pool contains no RC donors")
        return cls(
            path=resolved,
            root=resolved.parent,
            identity_sha256=actual,
            records_sha256=sha256_file(records_path),
            records=records,
            target_groups_path=groups_path,
            target_groups_sha256=sha256_file(groups_path),
            target_groups=groups,
            include_rc=include_rc,
            min_anchor_volume_mm3=minimum,
            max_bbox_mm=maximum,
        )


def _validate_k_record(
    record: ComponentRecord,
    *,
    include_rc: bool,
    min_anchor_volume_mm3: float,
    max_bbox_mm: float,
) -> None:
    if record.split != "train" or record.patient_group != patient_group(record.source_case_id):
        raise MetAugContractError(f"invalid K1245 donor identity: {record.component_id}")
    allowed = {1, 2, 3, 4} if include_rc else {1, 2, 3}
    if set(record.classes_present) - allowed:
        raise MetAugContractError(f"invalid K1245 donor classes: {record.component_id}")
    anchors = ANCHOR_LABELS if include_rc else LEGACY_ANCHOR_LABELS
    if not set(record.classes_present) & anchors:
        raise MetAugContractError(f"K1245 donor has no anchor: {record.component_id}")
    if record.core_volume_mm3 < min_anchor_volume_mm3:
        raise MetAugContractError(f"K1245 donor is below floor: {record.component_id}")
    if any(value > max_bbox_mm + 1e-8 for value in record.bbox_mm):
        raise MetAugContractError(f"K1245 donor exceeds 56 mm: {record.component_id}")
    if set(record.source_modalities_sha256) != set(S2_MODALITIES):
        raise MetAugContractError(f"K1245 donor lacks modality hashes: {record.component_id}")


@dataclass(frozen=True)
class K1245Config:
    path: Path
    route_id: str
    plan_sha256: str
    manifest_sha256: str
    p_select: float
    min_anchor_volume_mm3: float
    max_bbox_mm: float
    x1_rc_probability: float
    use_b: bool
    use_d: bool
    use_e: bool
    use_f: bool
    use_x2: bool
    use_x3: bool
    second_tumour_probability: float
    e_probability: float
    e_shell_factor: float
    e_volume_p10_mm3: float
    e_volume_p90_mm3: float
    burden_probabilities: Mapping[str, float]
    target_counts: Mapping[str, int]
    rank_volume_samples: Mapping[str, tuple[float, ...]]
    nn_distance_samples_mm: tuple[float, ...]
    crop_size: int = 64
    clearance_mm: float = 5.0
    boundary_clearance_mm: float = 3.0
    fix_v3: bool = True
    fix_v2: None = None

    @property
    def candidate_policy(self) -> bool:
        return True

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        manifest: K1245Manifest,
        plan_path: str | Path,
    ) -> "K1245Config":
        resolved = Path(path).expanduser().resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if payload.get("schema_version") != K1245_CONFIG_SCHEMA:
            raise MetAugContractError("unsupported K1245 config schema")
        if payload.get("config_sha256") != canonical_json_sha256(payload, exclude=("config_sha256",)):
            raise MetAugContractError("K1245 config SHA256 mismatch")
        route = str(payload.get("route_id"))
        if route not in K_ROUTE_IDS:
            raise MetAugContractError(f"unsupported K route: {route}")
        if payload.get("plan_sha256") != sha256_file(plan_path):
            raise MetAugContractError("K1245 plan binding drifted")
        if payload.get("manifest_sha256") != manifest.identity_sha256:
            raise MetAugContractError("K1245 manifest binding drifted")
        flags = payload.get("features", {})
        empirical = payload.get("empirical", {})
        result = cls(
            path=resolved,
            route_id=route,
            plan_sha256=str(payload["plan_sha256"]),
            manifest_sha256=str(payload["manifest_sha256"]),
            p_select=float(payload["p_select"]),
            min_anchor_volume_mm3=float(payload["min_anchor_volume_mm3"]),
            max_bbox_mm=float(payload["max_bbox_mm"]),
            x1_rc_probability=float(payload.get("x1_rc_probability", 0.0)),
            use_b=bool(flags.get("b")),
            use_d=bool(flags.get("d")),
            use_e=bool(flags.get("e")),
            use_f=bool(flags.get("f")),
            use_x2=bool(flags.get("x2")),
            use_x3=bool(flags.get("x3")),
            second_tumour_probability=float(payload.get("second_tumour_probability", 0.0)),
            e_probability=float(payload.get("e_probability", 0.0)),
            e_shell_factor=float(empirical.get("e_shell_factor", 0.0)),
            e_volume_p10_mm3=float(empirical.get("e_volume_p10_mm3", 0.0)),
            e_volume_p90_mm3=float(empirical.get("e_volume_p90_mm3", 0.0)),
            burden_probabilities={
                str(key): float(value)
                for key, value in empirical.get("burden_probabilities", {}).items()
            },
            target_counts={
                str(key): int(value)
                for key, value in empirical.get("target_counts", {}).items()
            },
            rank_volume_samples={
                str(key): tuple(float(item) for item in value)
                for key, value in empirical.get("rank_volume_samples", {}).items()
            },
            nn_distance_samples_mm=tuple(
                float(value) for value in empirical.get("nn_distance_samples_mm", ())
            ),
        )
        result.validate(manifest)
        return result

    def validate(self, manifest: K1245Manifest) -> None:
        expected = {
            "K1": (True, False, True, True, False, False),
            "K2": (True, True, True, False, True, False),
            "K4": (False, False, True, False, False, True),
            "K5": (True, True, True, True, False, False),
        }[self.route_id]
        observed = (self.use_b, self.use_d, self.use_e, self.use_f, self.use_x2, self.use_x3)
        if observed != expected:
            raise MetAugContractError(f"{self.route_id} feature contract drifted: {observed}")
        if self.p_select != 0.20 or self.max_bbox_mm != 56.0:
            raise MetAugContractError("K1245 selection/crop contract drifted")
        if self.use_x3 != (self.min_anchor_volume_mm3 == 8.0):
            raise MetAugContractError("K4/X3 floor contract drifted")
        if self.use_x3 and manifest.include_rc:
            raise MetAugContractError("K4 must use the isolated non-RC small-lesion pool")
        if not self.use_x3 and (not manifest.include_rc or self.min_anchor_volume_mm3 != 27.0):
            raise MetAugContractError("K1/K2/K5 require the RC-aware 27 mm3 pool")
        if self.x1_rc_probability != (0.0 if self.use_x3 else 0.023):
            raise MetAugContractError("X1 RC probability contract drifted")
        if self.second_tumour_probability != (0.20 if self.use_d else 0.0):
            raise MetAugContractError("D second-tumour contract drifted")
        if self.e_probability != 0.10 or self.e_shell_factor <= 0:
            raise MetAugContractError("E topology contract is incomplete")
        if self.use_f:
            keys = {"1", "2_3", "4_8", "9_16", "17_24", "gt24"}
            if set(self.burden_probabilities) != keys or not self.target_counts:
                raise MetAugContractError("F empirical burden contract is incomplete")
            values = [self.burden_probabilities[key] for key in ("1", "2_3", "4_8", "9_16", "17_24", "gt24")]
            if any(right + 1e-12 < left for left, right in zip(values, values[1:])):
                raise MetAugContractError("F weights are not monotonic increasing")
            mean = float(np.mean([self.selection_probability(case, 0) for case in self.target_counts]))
            if abs(mean - 0.20) > 1e-4:
                raise MetAugContractError(f"F mean probability drifted: {mean}")
        if self.use_d and (not self.rank_volume_samples or not self.nn_distance_samples_mm):
            raise MetAugContractError("D empirical rank/distance contract is incomplete")

    def selection_probability(self, case_id: str, epoch: int) -> float:
        value = self.p_select
        if self.use_f:
            count = self.target_counts.get(case_id)
            if count is None:
                raise MetAugContractError(f"missing F target count: {case_id}")
            key = "1" if count <= 1 else "2_3" if count <= 3 else "4_8" if count <= 8 else "9_16" if count <= 16 else "17_24" if count <= 24 else "gt24"
            value = self.burden_probabilities[key]
        if self.use_x2 and epoch >= 83:
            value *= max(0.0, (125.0 - float(epoch)) / 42.0)
        return float(value)


class K1245Sampler:
    def __init__(self, manifest: K1245Manifest, *, cache_size: int = 128):
        self.manifest = manifest
        self.cache_size = cache_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._pools: dict[bool, dict[str, list[ComponentRecord]]] = {
            False: defaultdict(list),
            True: defaultdict(list),
        }
        for record in manifest.records:
            self._pools[4 in record.classes_present][record.patient_group].append(record)

    def choose(
        self,
        rng: np.random.Generator,
        target_group: str,
        *,
        request_rc: bool,
    ) -> tuple[ComponentRecord | None, bool]:
        def choose_pool(is_rc: bool) -> ComponentRecord | None:
            groups = sorted(group for group in self._pools[is_rc] if group != target_group)
            if not groups:
                return None
            group = groups[int(rng.integers(0, len(groups)))]
            values = self._pools[is_rc][group]
            return values[int(rng.integers(0, len(values)))]

        record = choose_pool(request_rc)
        fallback = request_rc and record is None
        if record is None:
            record = choose_pool(False)
        return record, fallback

    def load_label(self, record: ComponentRecord) -> np.ndarray:
        cached = self._cache.get(record.component_id)
        if cached is not None:
            self._cache.move_to_end(record.component_id)
            return cached.copy()
        path = (self.manifest.root / record.component_path).resolve()
        if self.manifest.root not in path.parents or sha256_file(path) != record.label_sha256:
            raise MetAugContractError(f"K1245 component payload drifted: {record.component_id}")
        with np.load(path, allow_pickle=False) as payload:
            label = payload["label"].astype(np.int16, copy=True)
        stats = anchor_statistics(
            label,
            record.spacing_mm,
            anchor_labels=ANCHOR_LABELS if self.manifest.include_rc else LEGACY_ANCHOR_LABELS,
        )
        if tuple(stats["classes_present"]) != record.classes_present:
            raise MetAugContractError(f"K1245 component classes drifted: {record.component_id}")
        self._cache[record.component_id] = label
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return label.copy()


class RCAwareFixV3CandidateProcessor(FixV3CandidateProcessor):
    """Preserve FixV3 checks and add explicit uncalibrated RC guardrails."""

    processor_policy = "fix_v3_rc_guarded_qc_v1"

    def bind_extended_manifest(
        self,
        *,
        legacy_manifest_sha256: str,
        extended_manifest_sha256: str,
    ) -> None:
        if self.component_manifest_sha256 != legacy_manifest_sha256:
            raise MetAugContractError("FixV3 calibration does not bind the declared legacy manifest")
        self.legacy_component_manifest_sha256 = legacy_manifest_sha256
        self.component_manifest_sha256 = extended_manifest_sha256

    @staticmethod
    def _validate_inputs(
        image: np.ndarray,
        segmentation: np.ndarray,
        label_cube: np.ndarray,
        valid_mask: np.ndarray,
        spacing_mm: tuple[float, float, float],
    ) -> None:
        """Validate the shared FixV3 contract while preserving legal RC labels.

        The historical validator rejects label 4 because legacy Route A never
        inserted RC.  K1245 keeps that validator's shape/finite/spacing checks,
        but uses a validation-only copy with RC mapped to an existing foreground
        class.  All actual geometry, G1 conditioning, and RC guardrails still
        receive the original label cube.
        """
        values = _labels(label_cube)
        if not values.issubset(ALLOWED_LABELS):
            raise _Reject(
                "LABEL_CONTRACT_FAIL",
                f"unsupported inserted labels: {sorted(values)}",
            )
        validation_label = label_cube.copy()
        validation_label[validation_label == 4] = 1
        FixV3CandidateProcessor._validate_inputs(
            image,
            segmentation,
            validation_label,
            valid_mask,
            spacing_mm,
        )

    def _boundary_qc(self, **kwargs) -> Mapping[str, Any]:
        label_cube = kwargs["label_cube"]
        if not np.any(label_cube == 4):
            return super()._boundary_qc(**kwargs)
        without_rc = label_cube.copy()
        without_rc[without_rc == 4] = 0
        base = super()._boundary_qc(**{**kwargs, "label_cube": without_rc})
        rc: dict[str, Any] = {"status": "guardrail_pass_uncalibrated", "modalities": {}}
        for channel, modality in enumerate(S2_MODALITIES):
            faces = _extract_boundary_faces(
                label_cube=label_cube,
                image=kwargs["candidate"][channel],
                label_value=4,
                scale=float(kwargs["scales"][modality]),
                spacing_mm=kwargs["spacing_mm"],
            )
            if not np.all(np.isfinite(faces.absolute)) or faces.absolute.size == 0:
                raise MetAugContractError(f"RC boundary guardrail failed for {modality}")
            rc["modalities"][modality] = {
                "face_count": int(faces.absolute.size),
                "abs_q95": float(np.quantile(faces.absolute, 0.95)),
                "abs_q99": float(np.quantile(faces.absolute, 0.99)),
            }
        return {**dict(base), "rc": rc, "rc_calibration_status": "not_formally_calibrated"}

    def _cross_modal_qc(self, **kwargs) -> Mapping[str, Any]:
        label_cube = kwargs["label_cube"]
        base = super()._cross_modal_qc(**kwargs)
        support = label_cube == 4
        if not np.any(support):
            return base
        count = int(np.count_nonzero(support))
        if count < 8:
            raise MetAugContractError("RC support is below the fixed eight-voxel guardrail")
        metrics = {}
        affected_modalities = 0
        for channel, modality in enumerate(S2_MODALITIES):
            values = (
                kwargs["candidate"][channel][support] - float(kwargs["reference"][modality])
            ) / float(kwargs["scales"][modality])
            if not np.all(np.isfinite(values)):
                raise MetAugContractError(f"RC cross-modal values are non-finite: {modality}")
            median = float(np.median(values))
            spread = float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
            affected = float(np.mean(np.abs(values) >= 0.25))
            affected_modalities += int(affected > 0.05)
            metrics[modality] = {
                "median_contrast": median,
                "q05_q95_width": spread,
                "affected_fraction_abs_z_ge_0_25": affected,
            }
        if affected_modalities < 2:
            raise MetAugContractError("RC generation lacks a detectable effect in two modalities")
        result = dict(base)
        result["rc"] = {
            "status": "guardrail_pass_uncalibrated",
            "support_voxels": count,
            "affected_modalities": affected_modalities,
            "modalities": metrics,
        }
        result["rc_calibration_status"] = "not_formally_calibrated"
        return result


def _updated_record(
    record: ComponentRecord,
    label: np.ndarray,
    *,
    anchors: frozenset[int],
) -> ComponentRecord:
    stats = anchor_statistics(label, record.spacing_mm, anchor_labels=anchors)
    return replace(
        record,
        core_volume_mm3=float(stats["core_volume_mm3"]),
        total_volume_mm3=float(stats["total_volume_mm3"]),
        bbox_mm=tuple(float(value) for value in stats["bbox_mm"]),
        bbox_voxels=tuple(int(value) for value in stats["bbox_voxels"]),
        class_counts={str(key): int(value) for key, value in stats["class_counts"].items()},
        classes_present=tuple(int(value) for value in stats["classes_present"]),
    )


def _phenotype_scaled(
    label: np.ndarray,
    record: ComponentRecord,
    rng: np.random.Generator,
    records: tuple[ComponentRecord, ...],
    *,
    anchors: frozenset[int],
) -> tuple[np.ndarray, ComponentRecord, Mapping[str, Any]]:
    if 4 in record.classes_present:
        return label, record, {"b_result": "B_RC_EXEMPT", "linear_scale": 1.0}
    volumes = np.asarray(
        [item.core_volume_mm3 for item in records if item.classes_present == record.classes_present],
        dtype=np.float64,
    )
    lower = record.core_volume_mm3 * (1.0 / 1.5) ** 3
    upper = record.core_volume_mm3 * 1.5**3
    feasible = volumes[(volumes >= lower) & (volumes <= upper)]
    original_counts = np.asarray(
        [record.class_counts[str(value)] for value in record.classes_present], dtype=np.float64
    )
    original_fraction = original_counts / original_counts.sum()
    for attempt in range(1, 6):
        target = float(feasible[int(rng.integers(0, len(feasible)))]) if feasible.size else float(record.core_volume_mm3)
        scale = float((target / record.core_volume_mm3) ** (1.0 / 3.0))
        scaled = ndimage.zoom(label, zoom=scale, order=0, mode="nearest", prefilter=False).astype(np.int16)
        if not np.any(scaled) or any(value > 56 for value in scaled.shape):
            continue
        updated = _updated_record(record, scaled, anchors=anchors)
        if updated.core_volume_mm3 < 27 or any(value > 56.0 + 1e-8 for value in updated.bbox_mm):
            continue
        if updated.classes_present != record.classes_present:
            continue
        counts = np.asarray([updated.class_counts[str(value)] for value in record.classes_present], dtype=np.float64)
        fractions = counts / counts.sum()
        relative = np.abs(fractions - original_fraction) / np.maximum(original_fraction, 1e-12)
        if np.any(relative > 0.25 + 1e-12):
            continue
        return scaled, updated, {
            "b_result": "B_SCALE_APPLIED",
            "b_attempt": attempt,
            "target_anchor_volume_mm3": target,
            "linear_scale": scale,
            "max_class_fraction_relative_change": float(relative.max(initial=0.0)),
        }
    return label, record, {"b_result": "B_SCALE_NO_OP", "linear_scale": 1.0}


def _adaptive_snfh_shell(
    label: np.ndarray,
    record: ComponentRecord,
    config: K1245Config,
) -> tuple[np.ndarray, ComponentRecord, Mapping[str, Any]]:
    if record.classes_present != (3,):
        return label, record, {"e_result": "E_NOT_ELIGIBLE"}
    equivalent_radius = (3.0 * record.core_volume_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)
    radius = float(np.clip(config.e_shell_factor * equivalent_radius, 1.0, 10.0))
    padding = int(np.ceil(radius / min(record.spacing_mm))) + 1
    candidate = np.pad(label, padding, mode="constant")
    support = candidate != 0
    shell = ndimage.binary_dilation(
        support,
        structure=physical_ball(radius, record.spacing_mm),
        border_value=0,
    ) & ~support
    candidate[shell] = 2
    points = np.argwhere(candidate != 0)
    lower, upper = points.min(axis=0), points.max(axis=0) + 1
    candidate = candidate[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))]
    updated = _updated_record(record, candidate, anchors=LEGACY_ANCHOR_LABELS)
    if (
        any(value > 56.0 + 1e-8 for value in updated.bbox_mm)
        or not (config.e_volume_p10_mm3 <= updated.total_volume_mm3 <= config.e_volume_p90_mm3)
    ):
        return label, record, {"e_result": "E_TOPOLOGY_FALLBACK", "shell_radius_mm": radius}
    return candidate, updated, {
        "e_result": "E_SNFH_APPLIED",
        "shell_radius_mm": radius,
        "added_snfh_voxels": int(np.count_nonzero(shell)),
    }


def _rank_scaled(
    label: np.ndarray,
    record: ComponentRecord,
    config: K1245Config,
    rng: np.random.Generator,
    rank: int,
    *,
    anchors: frozenset[int],
) -> tuple[np.ndarray, ComponentRecord, Mapping[str, Any]]:
    if 4 in record.classes_present:
        return label, record, {"d_scale_result": "B_RC_EXEMPT", "target_rank": int(min(8, max(2, rank)))}
    key = str(min(8, max(2, rank)))
    samples = config.rank_volume_samples.get(key, ())
    if not samples:
        return label, record, {"d_scale_result": "D_RANK_SCALE_NO_OP", "target_rank": int(key)}
    target = float(samples[int(rng.integers(0, len(samples)))])
    scale = float(np.clip((target / record.core_volume_mm3) ** (1.0 / 3.0), 1.0 / 1.5, 1.5))
    scaled = ndimage.zoom(label, scale, order=0, mode="nearest", prefilter=False).astype(np.int16)
    if not np.any(scaled) or any(value > 56 for value in scaled.shape):
        return label, record, {"d_scale_result": "D_RANK_SCALE_NO_OP", "target_rank": int(key)}
    updated = _updated_record(record, scaled, anchors=anchors)
    return scaled, updated, {
        "d_scale_result": "D_RANK_SCALE_APPLIED",
        "target_rank": int(key),
        "target_anchor_volume_mm3": target,
        "linear_scale": scale,
    }


def _native_lesion_count(segmentation: np.ndarray) -> int:
    structure = np.ones((3, 3, 3), dtype=bool)
    _, count = ndimage.label(np.isin(segmentation, tuple(ALLOWED_LABELS - {0})), structure=structure)
    return int(count)


class K1245Engine(MetAugEngine):
    def __init__(
        self,
        *,
        manifest: K1245Manifest,
        config: K1245Config,
        backend: Any,
        audit_sink: Any,
        candidate_processor: RCAwareFixV3CandidateProcessor,
    ) -> None:
        config.validate(manifest)
        if candidate_processor.component_manifest_sha256 != manifest.identity_sha256:
            raise MetAugContractError("K1245 processor does not bind the extended manifest")
        self.manifest = manifest
        self.config = config
        self.backend = backend
        self.audit_sink = audit_sink
        self.candidate_processor = candidate_processor
        self.sampler = K1245Sampler(manifest)
        self._force_second_tumour = False

    def _event_identity(self, context: EventContext) -> tuple[str, int]:
        seed = _seed_from_event(
            global_seed=20260724,
            epoch=context.epoch,
            rank=context.rank,
            worker=context.worker,
            case_id=context.case_id,
            patch_index=context.patch_index,
            route_id=K1245_PLAN_ROUTE,
        )
        event_id = hashlib.sha256(
            f"{self.config.route_id}|{context.case_id}|{context.epoch}|{context.rank}|{context.worker}|{context.patch_index}".encode()
        ).hexdigest()[:24]
        return event_id, seed

    def plan(self, *, segmentation, valid_mask, context, inputs_prevalidated=False) -> EventResult:
        del inputs_prevalidated
        event_id, seed = self._event_identity(context)
        target_group = self.manifest.target_groups.get(context.case_id)
        if target_group is None:
            raise MetAugContractError(f"target case is absent from K1245 group map: {context.case_id}")
        seg = segmentation[0] if segmentation.ndim == 4 else segmentation
        if seg.ndim != 3 or valid_mask.shape != seg.shape:
            raise MetAugContractError("K1245 segmentation/valid-mask shape mismatch")
        if not _labels(seg).issubset(ALLOWED_LABELS | {-1}):
            raise MetAugContractError("K1245 target segmentation has invalid labels")
        rng = np.random.default_rng(seed)
        probability = 1.0 if self._force_second_tumour else self.config.selection_probability(context.case_id, context.epoch)
        metadata: dict[str, Any] = {
            "route_id": self.config.route_id,
            "target_case_id": context.case_id,
            "epoch": context.epoch,
            "patch_index": context.patch_index,
            "selection_probability": probability,
            "tumour_index": 2 if self._force_second_tumour else 1,
        }
        if float(rng.random()) >= probability:
            return EventResult("NO_OP", "NOT_SELECTED", event_id, seed, metadata=metadata)
        request_rc = self.config.x1_rc_probability > 0 and float(rng.random()) < self.config.x1_rc_probability
        record, rc_fallback = self.sampler.choose(rng, target_group, request_rc=request_rc)
        metadata.update({
            "x1_rc_requested": request_rc,
            "x1_rc_fallback": rc_fallback,
            "x1_result": "RC_QUOTA_NO_OP" if rc_fallback else "RC_SELECTED" if request_rc else "NON_RC_SELECTED",
        })
        if record is None:
            return EventResult("NO_OP", "NO_ELIGIBLE_DONOR", event_id, seed, metadata=metadata)
        label = self.sampler.load_label(record)
        anchors = ANCHOR_LABELS if self.manifest.include_rc else LEGACY_ANCHOR_LABELS
        runtime_record = record
        if self.config.use_b:
            label, runtime_record, details = _phenotype_scaled(
                label, runtime_record, rng, self.manifest.records, anchors=anchors
            )
            metadata.update(details)
        if self.config.use_d:
            rank = _native_lesion_count(seg) + (2 if self._force_second_tumour else 1)
            label, runtime_record, details = _rank_scaled(
                label, runtime_record, self.config, rng, rank, anchors=anchors
            )
            metadata.update(details)
        if self.config.use_e and float(rng.random()) < self.config.e_probability:
            label, runtime_record, details = _adaptive_snfh_shell(label, runtime_record, self.config)
            metadata.update(details)
        stats = anchor_statistics(label, runtime_record.spacing_mm, anchor_labels=anchors)
        if (
            stats["core_volume_mm3"] < self.config.min_anchor_volume_mm3
            or any(value > self.config.max_bbox_mm + 1e-8 for value in stats["bbox_mm"])
        ):
            return EventResult("NO_OP", "LABEL_INVALID", event_id, seed, record=record, metadata=metadata)
        if self.config.use_x3:
            _, connected = ndimage.label(np.isin(label, tuple(anchors)), structure=np.ones((3, 3, 3), bool))
            if connected != 1 or int(np.count_nonzero(np.isin(label, tuple(anchors)))) < 8:
                return EventResult("NO_OP", "X3_CONNECTIVITY_FAIL", event_id, seed, record=record, metadata=metadata)
        placement = self._find_placement(label, runtime_record, seg, valid_mask, context, rng)
        if placement is None:
            reason = "PLACEMENT_NO_OP" if self.config.use_d else "NO_VALID_PLACEMENT"
            return EventResult("NO_OP", reason, event_id, seed, record=runtime_record, metadata=metadata)
        return EventResult(
            "PLACEMENT_VALID", None, event_id, seed, record=runtime_record,
            placement=placement, metadata=metadata,
        )

    def _find_placement(self, label, record, segmentation, valid_mask, context, rng) -> Placement | None:
        if not self.config.use_d:
            from .met_aug_core import find_placement
            return find_placement(
                label=label,
                record=record,
                segmentation=segmentation,
                valid_mask=valid_mask,
                context=context,
                rng=rng,
                crop_size=64,
                clearance_mm=5.0,
                boundary_clearance_mm=3.0,
            )
        label_cube, support = _centered_label_cube(label, 64)
        valid_inner = ndimage.binary_erosion(
            valid_mask.astype(bool), physical_ball(3.0, record.spacing_mm), border_value=0
        )
        foreground = np.isin(segmentation, tuple(ALLOWED_LABELS - {0}))
        component_map, count = ndimage.label(foreground, structure=np.ones((3, 3, 3), bool))
        centroids = [np.asarray(ndimage.center_of_mass(foreground, component_map, index)) for index in range(1, count + 1)]
        if not centroids:
            return None
        forbidden = ndimage.binary_dilation(
            foreground, physical_ball(5.0, record.spacing_mm), border_value=0
        )
        shape = np.asarray(segmentation.shape)
        for attempt in range(1, 11):
            anchor = centroids[int(rng.integers(0, len(centroids)))]
            distance = float(self.config.nn_distance_samples_mm[int(rng.integers(0, len(self.config.nn_distance_samples_mm)))])
            vector = rng.normal(size=3)
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            centre = np.rint(anchor + vector * distance / np.asarray(record.spacing_mm)).astype(int)
            start, stop = centre - 32, centre + 32
            if np.any(start < 0) or np.any(stop > shape):
                continue
            slices = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
            if not np.all(valid_inner[slices][support]) or np.any(forbidden[slices][support]):
                continue
            return Placement(tuple(int(value) for value in start), label_cube, support, attempt, "empirical_nn_distance")
        return None

    def apply(self, *, image, segmentation, valid_mask, context):
        if not self.config.use_d:
            return super().apply(image=image, segmentation=segmentation, valid_mask=valid_mask, context=context)
        primary_context = replace(context, patch_index=int(context.patch_index) * 2)
        image_after, seg_after, primary = super().apply(
            image=image, segmentation=segmentation, valid_mask=valid_mask, context=primary_context
        )
        if primary.state != "COMMITTED":
            return image_after, seg_after, primary
        rng = np.random.default_rng(primary.event_seed ^ 0x5EC0D)
        if float(rng.random()) >= self.config.second_tumour_probability:
            skipped = EventResult(
                "NO_OP", "SECOND_NOT_SELECTED", primary.event_id, primary.event_seed,
                metadata={"route_id": self.config.route_id, "tumour_index": 2},
            )
            self._append(skipped)
            return image_after, seg_after, primary
        self._force_second_tumour = True
        try:
            return super().apply(
                image=image_after,
                segmentation=seg_after,
                valid_mask=valid_mask,
                context=replace(context, patch_index=int(context.patch_index) * 2 + 1_000_000_000),
            )
        finally:
            self._force_second_tumour = False
