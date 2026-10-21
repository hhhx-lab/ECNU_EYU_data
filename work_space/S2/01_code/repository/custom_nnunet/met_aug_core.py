"""Train-only MET-AUG component contracts and atomic Route A insertion logic.

This module intentionally does not import nnU-Net, Torch, or the G1 generator.  It
contains the deterministic, testable part of the bridge: component validation,
route-config binding, donor sampling, placement, and an all-or-nothing write
transaction.  The nnU-Net and diffusion adapters are thin callers around it.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

import numpy as np
from scipy import ndimage


ALLOWED_LABELS = frozenset({0, 1, 2, 3, 4})
CORE_LABELS = frozenset({1, 3})
ROUTE_A = "MET-AUG-A"
S2_MODALITIES = ("t1n", "t1c", "t2w", "t2f")
COMPONENT_MANIFEST_SCHEMA = 2
VALID_MASK_MANIFEST_SCHEMA = 2
ROUTE_CONFIG_SCHEMA = 2
COMPACT_SUPPORT_ROUTE_CONFIG_SCHEMA = 3
FIX_V2_ROUTE_CONFIG_SCHEMA = 4
FIX_V3_ROUTE_CONFIG_SCHEMA = 5
COMPACT_SUPPORT_POLICY = "compact_support_v1"
FIX_V2_PROCESSOR_POLICY = "fix_v2_qc_v1"
FIX_V3_PROCESSOR_POLICY = "fix_v3_qc_v1"
FIX_V2_BOUNDARY_POLICIES = frozenset(
    {"label_only_qc_v1", "halo_cosine_v1", "halo_cosine_harmonized_v1"}
)
SUPPORTED_ROUTE_CONFIG_SCHEMAS = frozenset(
    {
        ROUTE_CONFIG_SCHEMA,
        COMPACT_SUPPORT_ROUTE_CONFIG_SCHEMA,
        FIX_V2_ROUTE_CONFIG_SCHEMA,
        FIX_V3_ROUTE_CONFIG_SCHEMA,
    }
)


class MetAugContractError(RuntimeError):
    """Raised when a frozen MET-AUG contract is missing or inconsistent."""


class MetAugAuditError(RuntimeError):
    """Raised when a transaction cannot be recorded safely."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any], *, exclude: Iterable[str] = ()) -> str:
    excluded = set(exclude)
    payload = {key: value[key] for key in value if key not in excluded}
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_or_validate_immutable_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    label: str,
) -> str:
    """Create a JSON contract once, then reject any content drift on reuse."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    expected = json.loads(encoded)
    try:
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return "created"
    except FileExistsError:
        try:
            observed = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MetAugContractError(
                f"existing {label} is unreadable and must not be overwritten: {destination}"
            ) from exc
        if not isinstance(observed, dict):
            raise MetAugContractError(f"existing {label} is not a JSON object: {destination}")
        if observed != expected:
            all_keys = set(observed) | set(expected)
            changed = sorted(key for key in all_keys if observed.get(key) != expected.get(key))
            raise MetAugContractError(
                f"existing {label} does not match this run; refusing to overwrite {destination}; "
                f"changed_keys={changed}"
            )
        return "validated"


def patient_group(case_id: str) -> str:
    """Collapse only the documented three-digit BraTS-MET temporal suffix."""
    prefix, separator, suffix = str(case_id).rpartition("-")
    return prefix if separator and len(suffix) == 3 and suffix.isdigit() else str(case_id)


def parse_classes(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(part) for part in value.split("+") if part)
    return tuple(sorted(int(part) for part in value))


def classes_key(classes: Iterable[int]) -> str:
    return "+".join(str(value) for value in sorted(set(int(item) for item in classes)))


def physical_ball(radius_mm: float, spacing_mm: Iterable[float]) -> np.ndarray:
    spacing = np.asarray(tuple(float(value) for value in spacing_mm), dtype=np.float64)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError(f"spacing_mm must contain three positive values, got {spacing!r}")
    half_width = np.ceil(float(radius_mm) / spacing).astype(int)
    zz, yy, xx = np.mgrid[
        -half_width[0]:half_width[0] + 1,
        -half_width[1]:half_width[1] + 1,
        -half_width[2]:half_width[2] + 1,
    ]
    distance = np.sqrt(
        (zz * spacing[0]) ** 2
        + (yy * spacing[1]) ** 2
        + (xx * spacing[2]) ** 2
    )
    return distance <= float(radius_mm) + 1e-8


def _label_bbox(label: np.ndarray) -> tuple[slice, slice, slice]:
    points = np.argwhere(label != 0)
    if points.size == 0:
        raise MetAugContractError("component label has no foreground")
    lower = points.min(axis=0)
    upper = points.max(axis=0) + 1
    return tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper))  # type: ignore[return-value]


def label_statistics(label: np.ndarray, spacing_mm: Iterable[float]) -> dict[str, Any]:
    values = set(int(value) for value in np.unique(label))
    if not values.issubset(ALLOWED_LABELS):
        raise MetAugContractError(f"label contains unsupported classes: {sorted(values - ALLOWED_LABELS)}")
    if label.ndim != 3:
        raise MetAugContractError(f"component label must be 3D, got {label.shape}")
    spacing = np.asarray(tuple(float(value) for value in spacing_mm), dtype=np.float64)
    voxel_volume = float(np.prod(spacing))
    support = label != 0
    core = np.isin(label, tuple(CORE_LABELS))
    if not np.any(core):
        raise MetAugContractError("component label has no NETC/ET core")
    bbox = _label_bbox(label)
    bbox_shape = np.asarray([part.stop - part.start for part in bbox], dtype=np.float64)
    classes = tuple(sorted(int(value) for value in np.unique(label[support])))
    counts = {str(value): int(np.count_nonzero(label == value)) for value in classes}
    return {
        "classes_present": classes,
        "class_counts": counts,
        "core_volume_mm3": float(np.count_nonzero(core) * voxel_volume),
        "total_volume_mm3": float(np.count_nonzero(support) * voxel_volume),
        "bbox_mm": tuple(float(value) for value in bbox_shape * spacing),
        "bbox_voxels": tuple(int(value) for value in bbox_shape),
        "support_voxels": int(np.count_nonzero(support)),
    }


def extract_met_components(
    label: np.ndarray,
    spacing_mm: Iterable[float],
    *,
    min_core_volume_mm3: float = 27.0,
    max_bbox_mm: float = 56.0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Split one segmentation into conservative train-only donor components.

    Cores are 26-connected NETC/ET components.  A SNFH connected component is
    retained only when it is adjacent to exactly one core.  This deliberately
    drops ambiguous edema instead of assigning it to a nearby lesion by a
    fragile tie-break.
    """
    if label.ndim != 3:
        raise MetAugContractError(f"source label must be 3D, got {label.shape}")
    values = set(int(value) for value in np.unique(label))
    if not values.issubset(ALLOWED_LABELS):
        raise MetAugContractError(f"source label contains unsupported classes: {sorted(values - ALLOWED_LABELS)}")
    if np.any(label == 4):
        return [], {"source_contains_rc": 1}

    spacing = tuple(float(value) for value in spacing_mm)
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise MetAugContractError(f"invalid spacing: {spacing}")
    structure = np.ones((3, 3, 3), dtype=bool)
    core_mask = np.isin(label, tuple(CORE_LABELS))
    core_labels, core_count = ndimage.label(core_mask, structure=structure)
    if core_count == 0:
        return [], {"no_core": 1}

    attached_snfh: dict[int, np.ndarray] = {
        index: np.zeros_like(core_mask, dtype=bool) for index in range(1, core_count + 1)
    }
    snfh_labels, snfh_count = ndimage.label(label == 2, structure=structure)
    exclusions: dict[str, int] = defaultdict(int)
    for index in range(1, snfh_count + 1):
        snfh_component = snfh_labels == index
        touching = np.unique(core_labels[ndimage.binary_dilation(snfh_component, structure=structure)])
        touching = touching[touching != 0]
        if len(touching) == 1:
            attached_snfh[int(touching[0])] |= snfh_component
        elif len(touching) == 0:
            exclusions["unattached_snfh"] += 1
        else:
            exclusions["ambiguous_snfh"] += 1

    components: list[dict[str, Any]] = []
    for index in range(1, core_count + 1):
        component_mask = (core_labels == index) | attached_snfh[index]
        component = np.zeros_like(label, dtype=np.int16)
        component[component_mask] = label[component_mask]
        stats = label_statistics(component, spacing)
        if stats["core_volume_mm3"] < float(min_core_volume_mm3):
            exclusions["core_below_floor"] += 1
            continue
        if any(value > float(max_bbox_mm) + 1e-8 for value in stats["bbox_mm"]):
            exclusions["bbox_exceeds_limit"] += 1
            continue
        bbox = _label_bbox(component)
        crop = component[bbox].copy()
        core_points = np.argwhere(np.isin(component, tuple(CORE_LABELS)))
        centroid = core_points.mean(axis=0) / np.asarray(label.shape, dtype=np.float64)
        components.append(
            {
                "label": crop,
                "stats": stats,
                "core_centroid_norm": tuple(float(value) for value in centroid),
                "source_bbox_voxels": tuple(
                    (int(part.start), int(part.stop)) for part in bbox
                ),
            }
        )
    return components, dict(exclusions)


@dataclass(frozen=True)
class ComponentRecord:
    component_id: str
    manifest_version: str
    source_case_id: str
    patient_group: str
    split: str
    component_path: str
    label_sha256: str
    source_label_sha256: str
    source_modalities_sha256: Mapping[str, str]
    source_affine_sha256: str
    spacing_mm: tuple[float, float, float]
    core_volume_mm3: float
    total_volume_mm3: float
    bbox_mm: tuple[float, float, float]
    bbox_voxels: tuple[int, int, int]
    class_counts: Mapping[str, int]
    classes_present: tuple[int, ...]
    core_centroid_norm: tuple[float, float, float]

    @property
    def stratum(self) -> tuple[str, str]:
        return classes_key(self.classes_present), core_volume_bin(self.core_volume_mm3)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ComponentRecord":
        required = {
            "component_id", "manifest_version", "source_case_id", "patient_group", "split",
            "component_path", "label_sha256", "source_label_sha256", "source_modalities_sha256", "source_affine_sha256",
            "spacing_mm", "core_volume_mm3", "total_volume_mm3", "bbox_mm", "bbox_voxels",
            "class_counts", "classes_present", "core_centroid_norm",
        }
        missing = sorted(required - set(value))
        if missing:
            raise MetAugContractError(f"component manifest row misses fields: {missing}")
        return cls(
            component_id=str(value["component_id"]),
            manifest_version=str(value["manifest_version"]),
            source_case_id=str(value["source_case_id"]),
            patient_group=str(value["patient_group"]),
            split=str(value["split"]),
            component_path=str(value["component_path"]),
            label_sha256=str(value["label_sha256"]),
            source_label_sha256=str(value["source_label_sha256"]),
            source_modalities_sha256={str(key): str(item) for key, item in value["source_modalities_sha256"].items()},
            source_affine_sha256=str(value["source_affine_sha256"]),
            spacing_mm=tuple(float(item) for item in value["spacing_mm"]),
            core_volume_mm3=float(value["core_volume_mm3"]),
            total_volume_mm3=float(value["total_volume_mm3"]),
            bbox_mm=tuple(float(item) for item in value["bbox_mm"]),
            bbox_voxels=tuple(int(item) for item in value["bbox_voxels"]),
            class_counts={str(key): int(item) for key, item in value["class_counts"].items()},
            classes_present=parse_classes(value["classes_present"]),
            core_centroid_norm=tuple(float(item) for item in value["core_centroid_norm"]),
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "manifest_version": self.manifest_version,
            "source_case_id": self.source_case_id,
            "patient_group": self.patient_group,
            "split": self.split,
            "component_path": self.component_path,
            "label_sha256": self.label_sha256,
            "source_label_sha256": self.source_label_sha256,
            "source_modalities_sha256": dict(self.source_modalities_sha256),
            "source_affine_sha256": self.source_affine_sha256,
            "spacing_mm": list(self.spacing_mm),
            "core_volume_mm3": self.core_volume_mm3,
            "total_volume_mm3": self.total_volume_mm3,
            "bbox_mm": list(self.bbox_mm),
            "bbox_voxels": list(self.bbox_voxels),
            "class_counts": dict(self.class_counts),
            "classes_present": list(self.classes_present),
            "core_centroid_norm": list(self.core_centroid_norm),
        }


def core_volume_bin(core_volume_mm3: float) -> str:
    if core_volume_mm3 < 27.0:
        return "below_floor"
    if core_volume_mm3 <= 49.0:
        return "27_49"
    if core_volume_mm3 <= 275.0:
        return "50_275"
    return "gt_275"


@dataclass(frozen=True)
class ComponentManifest:
    path: Path
    root: Path
    identity_sha256: str
    records_sha256: str
    records: tuple[ComponentRecord, ...]
    target_groups_path: Path
    target_groups_sha256: str
    target_groups: Mapping[str, str]

    @classmethod
    def load(cls, path: str | Path) -> "ComponentManifest":
        manifest_path = Path(path).expanduser().resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_identity = str(payload.get("manifest_sha256", ""))
        actual_identity = canonical_json_sha256(payload, exclude=("manifest_sha256",))
        if not expected_identity or expected_identity != actual_identity:
            raise MetAugContractError("component manifest SHA256 does not match its frozen content")
        if payload.get("schema_version") != COMPONENT_MANIFEST_SCHEMA:
            raise MetAugContractError("unsupported component manifest schema")
        if payload.get("coordinate_space") != "nnUNetPlans_3d_fullres_preprocessed":
            raise MetAugContractError("component manifest is not in the S2 preprocessed coordinate space")
        for key in (
            "builder_code_sha256",
            "component_core_sha256",
            "nnunet_plans_sha256",
            "train_file_sha256",
            "mapping_csv_sha256",
        ):
            value = payload.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise MetAugContractError(f"component manifest does not bind {key}")
        root = manifest_path.parent
        records_path = root / str(payload.get("records_file", "components.jsonl"))
        if not records_path.is_file():
            raise FileNotFoundError(f"missing component records: {records_path}")
        records_sha = sha256_file(records_path)
        if records_sha != payload.get("records_sha256"):
            raise MetAugContractError("component record JSONL SHA256 mismatch")
        records = tuple(
            ComponentRecord.from_mapping(json.loads(line))
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if not records or len(records) != int(payload.get("component_count", -1)):
            raise MetAugContractError("component record count mismatch")
        for record in records:
            _validate_record(record)
        groups_path = root / str(payload.get("target_groups_file", "target_case_groups.json"))
        if not groups_path.is_file():
            raise FileNotFoundError(f"missing target case groups: {groups_path}")
        groups_sha = sha256_file(groups_path)
        if groups_sha != payload.get("target_groups_sha256"):
            raise MetAugContractError("target case group SHA256 mismatch")
        groups_payload = json.loads(groups_path.read_text(encoding="utf-8"))
        if groups_payload.get("schema_version") != 1:
            raise MetAugContractError("unsupported target case group schema")
        groups = {str(key): str(value) for key, value in groups_payload.get("case_to_patient_group", {}).items()}
        if not groups:
            raise MetAugContractError("target case group mapping is empty")
        return cls(
            path=manifest_path,
            root=root,
            identity_sha256=actual_identity,
            records_sha256=records_sha,
            records=records,
            target_groups_path=groups_path,
            target_groups_sha256=groups_sha,
            target_groups=groups,
        )


def _validate_record(record: ComponentRecord) -> None:
    if record.split != "train":
        raise MetAugContractError(f"component {record.component_id} is not train-only")
    if record.patient_group != patient_group(record.source_case_id):
        raise MetAugContractError(f"component {record.component_id} has an invalid patient group")
    if set(record.classes_present) - {1, 2, 3}:
        raise MetAugContractError(f"component {record.component_id} contains RC or invalid classes")
    if not set(record.classes_present) & CORE_LABELS:
        raise MetAugContractError(f"component {record.component_id} has no core classes")
    if record.core_volume_mm3 < 27.0:
        raise MetAugContractError(f"component {record.component_id} is below the core-volume floor")
    if any(value > 56.0 + 1e-8 for value in record.bbox_mm):
        raise MetAugContractError(f"component {record.component_id} exceeds the 56 mm crop limit")
    if len(record.spacing_mm) != 3 or any(value <= 0 for value in record.spacing_mm):
        raise MetAugContractError(f"component {record.component_id} has invalid spacing")
    if set(record.source_modalities_sha256) != set(S2_MODALITIES):
        raise MetAugContractError(f"component {record.component_id} lacks four modality hashes")
    if len(record.source_affine_sha256) != 64:
        raise MetAugContractError(f"component {record.component_id} lacks an affine SHA256")


@dataclass(frozen=True)
class RouteStratum:
    classes_present: tuple[int, ...]
    core_volume_bin: str
    weight: float

    @property
    def key(self) -> tuple[str, str]:
        return classes_key(self.classes_present), self.core_volume_bin

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RouteStratum":
        result = cls(
            classes_present=parse_classes(value["classes_present"]),
            core_volume_bin=str(value["core_volume_bin"]),
            weight=float(value["weight"]),
        )
        if result.weight <= 0:
            raise MetAugContractError("route stratum weight must be positive")
        return result

    def as_mapping(self) -> dict[str, Any]:
        return {
            "classes_present": list(self.classes_present),
            "core_volume_bin": self.core_volume_bin,
            "weight": self.weight,
        }


def component_record_support_counts(record: ComponentRecord) -> tuple[int, int]:
    counts = {int(key): int(value) for key, value in record.class_counts.items()}
    if any(value < 0 for value in counts.values()):
        raise MetAugContractError(
            f"component {record.component_id} has negative class counts"
        )
    if set(counts) != set(record.classes_present):
        raise MetAugContractError(
            f"component {record.component_id} class counts do not match classes_present"
        )
    total_support_voxels = sum(counts.values())
    core_voxels = sum(counts.get(label, 0) for label in CORE_LABELS)
    if total_support_voxels <= 0 or core_voxels <= 0:
        raise MetAugContractError(
            f"component {record.component_id} has invalid support/core counts"
        )
    return total_support_voxels, core_voxels


@dataclass(frozen=True)
class CompactSupportEligibility:
    policy: str
    max_total_support_voxels: int
    max_total_to_core_ratio: float
    eligible_component_count: int
    excluded_component_count: int
    eligible_by_core_volume_bin: Mapping[str, int]
    eligible_patient_groups_by_core_volume_bin: Mapping[str, int]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CompactSupportEligibility":
        required = {
            "policy",
            "max_total_support_voxels",
            "max_total_to_core_ratio",
            "eligible_component_count",
            "excluded_component_count",
            "eligible_by_core_volume_bin",
            "eligible_patient_groups_by_core_volume_bin",
        }
        missing = sorted(required - set(value))
        if missing:
            raise MetAugContractError(
                f"compact-support eligibility misses fields: {missing}"
            )
        result = cls(
            policy=str(value["policy"]),
            max_total_support_voxels=int(value["max_total_support_voxels"]),
            max_total_to_core_ratio=float(value["max_total_to_core_ratio"]),
            eligible_component_count=int(value["eligible_component_count"]),
            excluded_component_count=int(value["excluded_component_count"]),
            eligible_by_core_volume_bin={
                str(key): int(count)
                for key, count in value["eligible_by_core_volume_bin"].items()
            },
            eligible_patient_groups_by_core_volume_bin={
                str(key): int(count)
                for key, count in value[
                    "eligible_patient_groups_by_core_volume_bin"
                ].items()
            },
        )
        if result.policy != COMPACT_SUPPORT_POLICY:
            raise MetAugContractError(
                f"unsupported donor eligibility policy: {result.policy}"
            )
        if result.max_total_support_voxels <= 0:
            raise MetAugContractError(
                "compact-support max_total_support_voxels must be positive"
            )
        if not np.isfinite(result.max_total_to_core_ratio) or result.max_total_to_core_ratio < 1.0:
            raise MetAugContractError(
                "compact-support max_total_to_core_ratio must be finite and at least 1"
            )
        return result

    def accepts_counts(self, *, total_support_voxels: int, core_voxels: int) -> bool:
        if total_support_voxels <= 0 or core_voxels <= 0:
            return False
        return (
            int(total_support_voxels) <= self.max_total_support_voxels
            and float(total_support_voxels) / float(core_voxels)
            <= self.max_total_to_core_ratio + 1e-12
        )

    def accepts_record(self, record: ComponentRecord) -> bool:
        total_support_voxels, core_voxels = component_record_support_counts(record)
        return self.accepts_counts(
            total_support_voxels=total_support_voxels,
            core_voxels=core_voxels,
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "max_total_support_voxels": self.max_total_support_voxels,
            "max_total_to_core_ratio": self.max_total_to_core_ratio,
            "eligible_component_count": self.eligible_component_count,
            "excluded_component_count": self.excluded_component_count,
            "eligible_by_core_volume_bin": dict(
                sorted(self.eligible_by_core_volume_bin.items())
            ),
            "eligible_patient_groups_by_core_volume_bin": dict(
                sorted(self.eligible_patient_groups_by_core_volume_bin.items())
            ),
        }


@dataclass(frozen=True)
class FixV2RoutePolicy:
    boundary_policy: str
    calibration_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FixV2RoutePolicy":
        required = {"boundary_policy", "calibration_sha256"}
        missing = sorted(required - set(value))
        if missing:
            raise MetAugContractError(f"Fix-v2 policy misses fields: {missing}")
        unexpected = sorted(set(value) - required)
        if unexpected:
            raise MetAugContractError(
                f"Fix-v2 policy has unexpected fields: {unexpected}"
            )
        result = cls(
            boundary_policy=str(value["boundary_policy"]),
            calibration_sha256=str(value["calibration_sha256"]),
        )
        if result.boundary_policy not in FIX_V2_BOUNDARY_POLICIES:
            raise MetAugContractError(
                f"unsupported Fix-v2 boundary policy: {result.boundary_policy}"
            )
        if len(result.calibration_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in result.calibration_sha256
        ):
            raise MetAugContractError("Fix-v2 calibration SHA256 is malformed")
        return result

    def as_mapping(self) -> dict[str, str]:
        return {
            "boundary_policy": self.boundary_policy,
            "calibration_sha256": self.calibration_sha256,
        }

    @property
    def processor_policy(self) -> str:
        return FIX_V2_PROCESSOR_POLICY


@dataclass(frozen=True)
class FixV3RoutePolicy:
    boundary_policy: str
    calibration_sha256: str
    processor_policy: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FixV3RoutePolicy":
        required = {"boundary_policy", "calibration_sha256", "processor_policy"}
        missing = sorted(required - set(value))
        if missing:
            raise MetAugContractError(f"Fix-v3 policy misses fields: {missing}")
        unexpected = sorted(set(value) - required)
        if unexpected:
            raise MetAugContractError(
                f"Fix-v3 policy has unexpected fields: {unexpected}"
            )
        result = cls(
            boundary_policy=str(value["boundary_policy"]),
            calibration_sha256=str(value["calibration_sha256"]),
            processor_policy=str(value["processor_policy"]),
        )
        if result.boundary_policy not in FIX_V2_BOUNDARY_POLICIES:
            raise MetAugContractError(
                f"unsupported Fix-v3 boundary policy: {result.boundary_policy}"
            )
        if result.processor_policy != FIX_V3_PROCESSOR_POLICY:
            raise MetAugContractError(
                f"unsupported Fix-v3 processor policy: {result.processor_policy}"
            )
        if len(result.calibration_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in result.calibration_sha256
        ):
            raise MetAugContractError("Fix-v3 calibration SHA256 is malformed")
        return result

    def as_mapping(self) -> dict[str, str]:
        return {
            "boundary_policy": self.boundary_policy,
            "calibration_sha256": self.calibration_sha256,
            "processor_policy": self.processor_policy,
        }


def _make_compact_support_eligibility(
    records: Iterable[ComponentRecord],
    *,
    max_total_support_voxels: int,
    max_total_to_core_ratio: float,
) -> CompactSupportEligibility:
    provisional = CompactSupportEligibility(
        policy=COMPACT_SUPPORT_POLICY,
        max_total_support_voxels=int(max_total_support_voxels),
        max_total_to_core_ratio=float(max_total_to_core_ratio),
        eligible_component_count=0,
        excluded_component_count=0,
        eligible_by_core_volume_bin={},
        eligible_patient_groups_by_core_volume_bin={},
    )
    if provisional.max_total_support_voxels <= 0:
        raise ValueError("max_total_support_voxels must be positive")
    if (
        not np.isfinite(provisional.max_total_to_core_ratio)
        or provisional.max_total_to_core_ratio < 1.0
    ):
        raise ValueError("max_total_to_core_ratio must be finite and at least 1")

    frozen_records = tuple(records)
    eligible = tuple(
        record for record in frozen_records if provisional.accepts_record(record)
    )
    counts: dict[str, int] = defaultdict(int)
    groups: dict[str, set[str]] = defaultdict(set)
    for record in eligible:
        volume_bin = record.stratum[1]
        counts[volume_bin] += 1
        groups[volume_bin].add(record.patient_group)
    return CompactSupportEligibility(
        policy=COMPACT_SUPPORT_POLICY,
        max_total_support_voxels=provisional.max_total_support_voxels,
        max_total_to_core_ratio=provisional.max_total_to_core_ratio,
        eligible_component_count=len(eligible),
        excluded_component_count=len(frozen_records) - len(eligible),
        eligible_by_core_volume_bin=dict(sorted(counts.items())),
        eligible_patient_groups_by_core_volume_bin={
            key: len(value) for key, value in sorted(groups.items())
        },
    )


@dataclass(frozen=True)
class RouteConfig:
    path: Path
    schema_version: int
    route_id: str
    seed: int
    component_manifest_sha256: str
    target_groups_sha256: str
    p_select: float
    max_tumours: int
    second_tumour_probability: float
    scale_min: float
    scale_max: float
    preserve_classes: bool
    crop_size: int
    clearance_mm: float
    boundary_clearance_mm: float
    min_core_volume_mm3: float
    max_bbox_mm: float
    strata: tuple[RouteStratum, ...]
    donor_eligibility: CompactSupportEligibility | None = None
    fix_v2: FixV2RoutePolicy | None = None
    fix_v3: FixV3RoutePolicy | None = None

    @classmethod
    def load(cls, path: str | Path, manifest: ComponentManifest) -> "RouteConfig":
        route_path = Path(path).expanduser().resolve()
        payload = json.loads(route_path.read_text(encoding="utf-8"))
        schema_version = int(payload.get("schema_version", -1))
        if schema_version not in SUPPORTED_ROUTE_CONFIG_SCHEMAS:
            raise MetAugContractError("unsupported MET-AUG route config schema")
        if payload.get("route_id") != ROUTE_A:
            raise MetAugContractError("only MET-AUG-A is implemented and may be enabled")
        if payload.get("component_manifest_sha256") != manifest.identity_sha256:
            raise MetAugContractError("route config does not bind the loaded component manifest")
        if payload.get("target_groups_sha256") != manifest.target_groups_sha256:
            raise MetAugContractError("route config does not bind the target group map")
        eligibility_payload = payload.get("donor_eligibility")
        if schema_version == COMPACT_SUPPORT_ROUTE_CONFIG_SCHEMA and not isinstance(
            eligibility_payload, Mapping
        ):
            raise MetAugContractError(
                "compact-support Route A config lacks donor_eligibility"
            )
        if eligibility_payload is not None and not isinstance(
            eligibility_payload, Mapping
        ):
            raise MetAugContractError("donor_eligibility must be an object")
        fix_v2_payload = payload.get("fix_v2")
        if schema_version == FIX_V2_ROUTE_CONFIG_SCHEMA and not isinstance(
            fix_v2_payload, Mapping
        ):
            raise MetAugContractError("Fix-v2 Route A config lacks fix_v2 policy")
        fix_v3_payload = payload.get("fix_v3")
        if schema_version == FIX_V3_ROUTE_CONFIG_SCHEMA and not isinstance(
            fix_v3_payload, Mapping
        ):
            raise MetAugContractError("Fix-v3 Route A config lacks fix_v3 policy")
        config = cls(
            path=route_path,
            schema_version=schema_version,
            route_id=str(payload["route_id"]),
            seed=int(payload["seed"]),
            component_manifest_sha256=str(payload["component_manifest_sha256"]),
            target_groups_sha256=str(payload["target_groups_sha256"]),
            p_select=float(payload["p_select"]),
            max_tumours=int(payload["max_tumours"]),
            second_tumour_probability=float(payload["second_tumour_probability"]),
            scale_min=float(payload["scale_min"]),
            scale_max=float(payload["scale_max"]),
            preserve_classes=bool(payload["preserve_classes"]),
            crop_size=int(payload["crop_size"]),
            clearance_mm=float(payload["clearance_mm"]),
            boundary_clearance_mm=float(payload["boundary_clearance_mm"]),
            min_core_volume_mm3=float(payload["min_core_volume_mm3"]),
            max_bbox_mm=float(payload["max_bbox_mm"]),
            strata=tuple(RouteStratum.from_mapping(item) for item in payload["strata"]),
            donor_eligibility=(
                CompactSupportEligibility.from_mapping(eligibility_payload)
                if isinstance(eligibility_payload, Mapping)
                else None
            ),
            fix_v2=(
                FixV2RoutePolicy.from_mapping(fix_v2_payload)
                if isinstance(fix_v2_payload, Mapping)
                else None
            ),
            fix_v3=(
                FixV3RoutePolicy.from_mapping(fix_v3_payload)
                if isinstance(fix_v3_payload, Mapping)
                else None
            ),
        )
        if schema_version == ROUTE_CONFIG_SCHEMA and eligibility_payload is not None:
            raise MetAugContractError(
                "legacy Route A schema must not contain donor_eligibility"
            )
        if schema_version != FIX_V2_ROUTE_CONFIG_SCHEMA and fix_v2_payload is not None:
            raise MetAugContractError(
                "legacy Route A schemas must not contain fix_v2 policy"
            )
        if schema_version != FIX_V3_ROUTE_CONFIG_SCHEMA and fix_v3_payload is not None:
            raise MetAugContractError(
                "non-Fix-v3 Route A schemas must not contain fix_v3 policy"
            )
        config.validate(manifest)
        return config

    def is_record_eligible(self, record: ComponentRecord) -> bool:
        return (
            self.donor_eligibility is None
            or self.donor_eligibility.accepts_record(record)
        )

    @property
    def candidate_policy(self) -> FixV2RoutePolicy | FixV3RoutePolicy | None:
        return self.fix_v3 if self.fix_v3 is not None else self.fix_v2

    def is_support_eligible(
        self,
        *,
        total_support_voxels: int,
        core_voxels: int,
    ) -> bool:
        return (
            self.donor_eligibility is None
            or self.donor_eligibility.accepts_counts(
                total_support_voxels=total_support_voxels,
                core_voxels=core_voxels,
            )
        )

    def eligible_records(
        self, manifest: ComponentManifest
    ) -> tuple[ComponentRecord, ...]:
        return tuple(
            record for record in manifest.records if self.is_record_eligible(record)
        )

    def validate(self, manifest: ComponentManifest) -> None:
        if self.p_select != 0.20:
            raise MetAugContractError("MET-AUG-A requires frozen p_select=0.20")
        if self.max_tumours != 1 or self.second_tumour_probability != 0.0:
            raise MetAugContractError("MET-AUG-A must keep exactly one candidate tumour")
        if self.scale_min != 1.0 or self.scale_max != 1.0:
            raise MetAugContractError("MET-AUG-A forbids component scaling")
        if not self.preserve_classes:
            raise MetAugContractError("MET-AUG-A forbids class remapping")
        if self.crop_size != 64:
            raise MetAugContractError("MET-AUG-A requires the frozen 64^3 diffusion crop")
        if self.clearance_mm != 5.0 or self.boundary_clearance_mm != 3.0:
            raise MetAugContractError("Route A clearance contract has drifted")
        if self.min_core_volume_mm3 != 27.0 or self.max_bbox_mm != 56.0:
            raise MetAugContractError("Route A component constraints have drifted")
        if (self.schema_version == FIX_V2_ROUTE_CONFIG_SCHEMA) != (
            self.fix_v2 is not None
        ):
            raise MetAugContractError("Fix-v2 schema and policy disagree")
        if (self.schema_version == FIX_V3_ROUTE_CONFIG_SCHEMA) != (
            self.fix_v3 is not None
        ):
            raise MetAugContractError("Fix-v3 schema and policy disagree")
        if self.fix_v2 is not None and self.fix_v3 is not None:
            raise MetAugContractError("Route A cannot bind Fix-v2 and Fix-v3 together")
        eligible_records = self.eligible_records(manifest)
        if not eligible_records:
            raise MetAugContractError("route config excludes every donor component")
        if self.donor_eligibility is not None:
            expected = _make_compact_support_eligibility(
                manifest.records,
                max_total_support_voxels=(
                    self.donor_eligibility.max_total_support_voxels
                ),
                max_total_to_core_ratio=(
                    self.donor_eligibility.max_total_to_core_ratio
                ),
            )
            if self.donor_eligibility.as_mapping() != expected.as_mapping():
                raise MetAugContractError(
                    "compact-support eligibility audit does not match the frozen component manifest"
                )
        available = {record.stratum for record in eligible_records}
        configured = [stratum.key for stratum in self.strata]
        if not configured or len(configured) != len(set(configured)):
            raise MetAugContractError("route strata are empty or duplicated")
        if set(configured) != available:
            raise MetAugContractError(
                "route strata do not exactly match the frozen component manifest: "
                f"missing={sorted(available - set(configured))}, "
                f"unexpected={sorted(set(configured) - available)}"
            )


def make_route_a_config(
    manifest: ComponentManifest,
    *,
    seed: int = 20260725,
    max_total_support_voxels: int | None = None,
    max_total_to_core_ratio: float | None = None,
) -> dict[str, Any]:
    """Create the frozen Route A distribution from a frozen component manifest."""
    if (max_total_support_voxels is None) != (max_total_to_core_ratio is None):
        raise ValueError(
            "compact-support thresholds must be provided together"
        )
    eligibility = None
    if max_total_support_voxels is not None and max_total_to_core_ratio is not None:
        eligibility = _make_compact_support_eligibility(
            manifest.records,
            max_total_support_voxels=max_total_support_voxels,
            max_total_to_core_ratio=max_total_to_core_ratio,
        )
    eligible_records = tuple(
        record
        for record in manifest.records
        if eligibility is None or eligibility.accepts_record(record)
    )
    if not eligible_records:
        raise MetAugContractError("compact-support policy excludes every donor component")
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in eligible_records:
        grouped[record.stratum].add(record.patient_group)
    strata: list[RouteStratum] = []
    for (class_key, volume_bin), groups in sorted(grouped.items()):
        multiplier = 1.5 if volume_bin in {"27_49", "50_275"} else 1.0
        strata.append(
            RouteStratum(
                classes_present=parse_classes(class_key),
                core_volume_bin=volume_bin,
                weight=float(len(groups) * multiplier),
            )
        )
    payload = {
        "schema_version": (
            COMPACT_SUPPORT_ROUTE_CONFIG_SCHEMA
            if eligibility is not None
            else ROUTE_CONFIG_SCHEMA
        ),
        "route_id": ROUTE_A,
        "seed": int(seed),
        "component_manifest_sha256": manifest.identity_sha256,
        "target_groups_sha256": manifest.target_groups_sha256,
        "p_select": 0.20,
        "max_tumours": 1,
        "second_tumour_probability": 0.0,
        "scale_min": 1.0,
        "scale_max": 1.0,
        "preserve_classes": True,
        "crop_size": 64,
        "clearance_mm": 5.0,
        "boundary_clearance_mm": 3.0,
        "min_core_volume_mm3": 27.0,
        "max_bbox_mm": 56.0,
        "strata": [item.as_mapping() for item in strata],
    }
    if eligibility is not None:
        payload["donor_eligibility"] = eligibility.as_mapping()
    return payload


def make_fix_v2_route_a_config(
    manifest: ComponentManifest,
    *,
    boundary_policy: str,
    calibration_sha256: str,
    seed: int = 20260725,
    max_total_support_voxels: int | None = None,
    max_total_to_core_ratio: float | None = None,
) -> dict[str, Any]:
    payload = make_route_a_config(
        manifest,
        seed=seed,
        max_total_support_voxels=max_total_support_voxels,
        max_total_to_core_ratio=max_total_to_core_ratio,
    )
    policy = FixV2RoutePolicy.from_mapping(
        {
            "boundary_policy": boundary_policy,
            "calibration_sha256": calibration_sha256,
        }
    )
    payload["schema_version"] = FIX_V2_ROUTE_CONFIG_SCHEMA
    payload["fix_v2"] = policy.as_mapping()
    return payload


def make_fix_v3_route_a_config(
    manifest: ComponentManifest,
    *,
    boundary_policy: str,
    calibration_sha256: str,
    seed: int = 20260725,
    max_total_support_voxels: int | None = None,
    max_total_to_core_ratio: float | None = None,
) -> dict[str, Any]:
    payload = make_route_a_config(
        manifest,
        seed=seed,
        max_total_support_voxels=max_total_support_voxels,
        max_total_to_core_ratio=max_total_to_core_ratio,
    )
    policy = FixV3RoutePolicy.from_mapping(
        {
            "boundary_policy": boundary_policy,
            "calibration_sha256": calibration_sha256,
            "processor_policy": FIX_V3_PROCESSOR_POLICY,
        }
    )
    payload["schema_version"] = FIX_V3_ROUTE_CONFIG_SCHEMA
    payload["fix_v3"] = policy.as_mapping()
    return payload


class ComponentSampler:
    def __init__(self, manifest: ComponentManifest, config: RouteConfig, *, cache_size: int = 128):
        self.manifest = manifest
        self.config = config
        self.cache_size = int(cache_size)
        self._by_stratum: dict[tuple[str, str], dict[str, list[ComponentRecord]]] = defaultdict(lambda: defaultdict(list))
        for record in manifest.records:
            if not config.is_record_eligible(record):
                continue
            self._by_stratum[record.stratum][record.patient_group].append(record)
        self._strata = list(config.strata)
        self._weights = np.asarray([item.weight for item in self._strata], dtype=np.float64)
        self._weights /= self._weights.sum()
        self._label_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def choose(self, rng: np.random.Generator, target_patient_group: str) -> ComponentRecord | None:
        stratum = self._strata[int(rng.choice(len(self._strata), p=self._weights))]
        options = self._by_stratum[stratum.key]
        donor_groups = sorted(group for group in options if group != target_patient_group)
        if not donor_groups:
            return None
        group = donor_groups[int(rng.integers(0, len(donor_groups)))]
        records = options[group]
        return records[int(rng.integers(0, len(records)))]

    def load_label(self, record: ComponentRecord) -> np.ndarray:
        cached = self._label_cache.get(record.component_id)
        if cached is not None:
            self._label_cache.move_to_end(record.component_id)
            return cached.copy()
        path = (self.manifest.root / record.component_path).resolve()
        if self.manifest.root not in path.parents:
            raise MetAugContractError(f"component path escapes manifest root: {record.component_path}")
        if not path.is_file():
            raise FileNotFoundError(f"component payload is missing: {path}")
        if sha256_file(path) != record.label_sha256:
            raise MetAugContractError(f"component payload SHA256 drifted: {record.component_id}")
        with np.load(path, allow_pickle=False) as payload:
            if "label" not in payload:
                raise MetAugContractError(f"component payload has no label array: {path}")
            label = payload["label"].astype(np.int16, copy=True)
        stats = label_statistics(label, record.spacing_mm)
        if tuple(stats["classes_present"]) != record.classes_present:
            raise MetAugContractError(f"component label classes drifted: {record.component_id}")
        if not np.isclose(stats["core_volume_mm3"], record.core_volume_mm3):
            raise MetAugContractError(f"component core volume drifted: {record.component_id}")
        if not np.isclose(stats["total_volume_mm3"], record.total_volume_mm3):
            raise MetAugContractError(f"component total volume drifted: {record.component_id}")
        observed_class_counts = {
            str(label_value): int(np.count_nonzero(label == label_value))
            for label_value in stats["classes_present"]
        }
        if observed_class_counts != dict(record.class_counts):
            raise MetAugContractError(f"component class counts drifted: {record.component_id}")
        core_voxels = sum(
            observed_class_counts.get(str(label_value), 0)
            for label_value in CORE_LABELS
        )
        if not self.config.is_support_eligible(
            total_support_voxels=int(stats["support_voxels"]),
            core_voxels=core_voxels,
        ):
            raise MetAugContractError(
                f"component payload violates compact-support eligibility: {record.component_id}"
            )
        self._label_cache[record.component_id] = label
        self._label_cache.move_to_end(record.component_id)
        while len(self._label_cache) > self.cache_size:
            self._label_cache.popitem(last=False)
        return label.copy()


def _seed_from_event(
    *,
    global_seed: int,
    epoch: int,
    rank: int,
    worker: int,
    case_id: str,
    patch_index: int,
    route_id: str,
) -> int:
    text = "|".join(
        str(value)
        for value in (global_seed, epoch, rank, worker, case_id, patch_index, route_id)
    )
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


@dataclass(frozen=True)
class EventContext:
    epoch: int
    rank: int
    worker: int
    case_id: str
    patch_index: int
    patch_origin: tuple[int, int, int] = (0, 0, 0)
    full_shape: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class Placement:
    crop_start: tuple[int, int, int]
    label_cube: np.ndarray
    support: np.ndarray
    attempts: int
    placement_strategy: str


@dataclass(frozen=True)
class PlannedInsertion:
    record: ComponentRecord
    placement: Placement
    event_seed: int
    event_id: str


@dataclass(frozen=True)
class EventResult:
    state: str
    reason: str | None
    event_id: str
    event_seed: int
    record: ComponentRecord | None = None
    placement: Placement | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    evidence: Mapping[str, np.ndarray] = field(default_factory=dict, repr=False)

    def audit_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "event_id": self.event_id,
            "event_seed": self.event_seed,
            "state": self.state,
            "reason": self.reason,
            **dict(self.metadata),
        }
        if self.record is not None:
            result.update(
                {
                    "component_id": self.record.component_id,
                    "source_case_id": self.record.source_case_id,
                    "donor_patient_group": self.record.patient_group,
                    "classes_present": list(self.record.classes_present),
                    "core_volume_mm3": self.record.core_volume_mm3,
                }
            )
        if self.placement is not None:
            result.update(
                {
                    "crop_start": list(self.placement.crop_start),
                    "placement_attempts": self.placement.attempts,
                    "placement_strategy": self.placement.placement_strategy,
                    "support_voxels": int(np.count_nonzero(self.placement.support)),
                }
            )
        return result


class AuditSink(Protocol):
    def append(self, event: Mapping[str, Any]) -> None: ...


class JsonlAuditSink:
    """Append-only audit sink.  A write error is a hard transaction failure."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()

    def append(self, event: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(dict(event), ensure_ascii=True, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()


class MemoryAuditSink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def append(self, event: Mapping[str, Any]) -> None:
        self.events.append(dict(event))


class InpaintingBackend(Protocol):
    def generate(
        self,
        image_crop: np.ndarray,
        label_crop: np.ndarray,
        *,
        seed: int,
        inpaint_support: np.ndarray | None = None,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class CandidateProcessingResult:
    image: np.ndarray
    segmentation: np.ndarray
    image_support: np.ndarray
    label_support: np.ndarray
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    evidence: Mapping[str, np.ndarray] = field(default_factory=dict, repr=False)


class CandidateProcessor(Protocol):
    processor_policy: str
    boundary_policy: str
    calibration_sha256: str
    component_manifest_sha256: str
    target_groups_sha256: str

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
        backend: InpaintingBackend,
    ) -> CandidateProcessingResult: ...


def _centered_label_cube(label: np.ndarray, crop_size: int) -> tuple[np.ndarray, np.ndarray]:
    if label.ndim != 3:
        raise MetAugContractError(f"component payload must be 3D, got {label.shape}")
    if any(value > crop_size - 8 for value in label.shape):
        raise MetAugContractError(f"component label does not leave a 4 voxel crop context: {label.shape}")
    result = np.zeros((crop_size, crop_size, crop_size), dtype=np.int16)
    starts = tuple((crop_size - int(size)) // 2 for size in label.shape)
    ends = tuple(start + int(size) for start, size in zip(starts, label.shape))
    result[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]] = label
    return result, result != 0


def _candidate_centres(
    valid_mask: np.ndarray,
    record: ComponentRecord,
    context: EventContext,
    rng: np.random.Generator,
    max_attempts: int,
) -> Iterable[tuple[tuple[int, int, int], str]]:
    shape = np.asarray(valid_mask.shape, dtype=np.int64)
    if context.full_shape is not None:
        desired_global = np.rint(np.asarray(record.core_centroid_norm) * np.asarray(context.full_shape)).astype(int)
        desired = desired_global - np.asarray(context.patch_origin)
        yield tuple(int(value) for value in desired), "donor_centroid_prior"
    else:
        desired = np.rint(np.asarray(record.core_centroid_norm) * shape).astype(int)
        yield tuple(int(value) for value in desired), "patch_relative_prior"
    valid_points = np.argwhere(valid_mask)
    if valid_points.size == 0:
        return
    for _ in range(max(0, max_attempts - 1)):
        point = valid_points[int(rng.integers(0, len(valid_points)))]
        yield tuple(int(value) for value in point), "valid_mask_fallback"


def find_placement(
    *,
    label: np.ndarray,
    record: ComponentRecord,
    segmentation: np.ndarray,
    valid_mask: np.ndarray,
    context: EventContext,
    rng: np.random.Generator,
    crop_size: int,
    clearance_mm: float,
    boundary_clearance_mm: float,
    max_attempts: int = 50,
) -> Placement | None:
    if segmentation.ndim == 4 and segmentation.shape[0] == 1:
        segmentation = segmentation[0]
    if segmentation.ndim != 3 or valid_mask.shape != segmentation.shape:
        raise MetAugContractError("segmentation and valid mask must have matching 3D shapes")
    label_cube, local_support = _centered_label_cube(label, crop_size)
    if not np.any(local_support):
        return None
    spacing = record.spacing_mm
    valid_inner = ndimage.binary_erosion(
        valid_mask.astype(bool),
        structure=physical_ball(boundary_clearance_mm, spacing),
        border_value=0,
    )
    foreground = np.isin(segmentation, tuple(ALLOWED_LABELS - {0}))
    forbidden = ndimage.binary_dilation(
        foreground,
        structure=physical_ball(clearance_mm, spacing),
        border_value=0,
    )
    shape = np.asarray(segmentation.shape, dtype=int)
    half = crop_size // 2
    for attempt, (centre, strategy) in enumerate(
        _candidate_centres(valid_inner, record, context, rng, max_attempts), start=1
    ):
        centre_array = np.asarray(centre, dtype=int)
        crop_start = centre_array - half
        crop_end = crop_start + crop_size
        if np.any(crop_start < 0) or np.any(crop_end > shape):
            continue
        slices = tuple(slice(int(start), int(stop)) for start, stop in zip(crop_start, crop_end))
        valid_crop = valid_inner[slices]
        forbidden_crop = forbidden[slices]
        if not np.all(valid_crop[local_support]):
            continue
        if np.any(forbidden_crop[local_support]):
            continue
        return Placement(
            crop_start=tuple(int(value) for value in crop_start),
            label_cube=label_cube,
            support=local_support,
            attempts=attempt,
            placement_strategy=strategy,
        )
    return None


class MetAugEngine:
    """Frozen Route A planner and all-or-nothing four-modality transaction."""

    def __init__(
        self,
        *,
        manifest: ComponentManifest,
        config: RouteConfig,
        backend: InpaintingBackend | None,
        audit_sink: AuditSink,
        candidate_processor: CandidateProcessor | None = None,
    ) -> None:
        config.validate(manifest)
        self.manifest = manifest
        self.config = config
        self.backend = backend
        self.audit_sink = audit_sink
        self.candidate_processor = candidate_processor
        route_policy = config.candidate_policy
        if route_policy is None and candidate_processor is not None:
            raise MetAugContractError(
                "legacy Route A config cannot use a versioned candidate processor"
            )
        if route_policy is not None and candidate_processor is not None:
            if candidate_processor.processor_policy != route_policy.processor_policy:
                raise MetAugContractError(
                    "candidate processor policy does not match route config"
                )
            if candidate_processor.boundary_policy != route_policy.boundary_policy:
                raise MetAugContractError(
                    "candidate processor boundary policy does not match route config"
                )
            if candidate_processor.calibration_sha256 != route_policy.calibration_sha256:
                raise MetAugContractError(
                    "candidate processor calibration does not match route config"
                )
            if candidate_processor.component_manifest_sha256 != manifest.identity_sha256:
                raise MetAugContractError(
                    "Fix-v2 calibration does not bind the loaded component manifest"
                )
            if candidate_processor.target_groups_sha256 != manifest.target_groups_sha256:
                raise MetAugContractError(
                    "Fix-v2 calibration does not bind the loaded target group map"
                )
        self.sampler = ComponentSampler(manifest, config)

    def _event_identity(self, context: EventContext) -> tuple[str, int]:
        seed = _seed_from_event(
            global_seed=self.config.seed,
            epoch=context.epoch,
            rank=context.rank,
            worker=context.worker,
            case_id=context.case_id,
            patch_index=context.patch_index,
            route_id=self.config.route_id,
        )
        event_id = hashlib.sha256(
            f"{self.config.route_id}|{context.case_id}|{context.epoch}|{context.rank}|{context.worker}|{context.patch_index}".encode("utf-8")
        ).hexdigest()[:24]
        return event_id, seed

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
            raise MetAugContractError("Route A requires matching segmentation and explicit valid-mask patches")
        if not inputs_prevalidated and not set(
            int(value) for value in np.unique(segmentation)
        ).issubset(ALLOWED_LABELS | {-1}):
            raise MetAugContractError("target segmentation has invalid classes")
        rng = np.random.default_rng(seed)
        base_metadata = {
            "route_id": self.config.route_id,
            "target_case_id": context.case_id,
            "target_patient_group": target_group,
            "epoch": context.epoch,
            "rank": context.rank,
            "worker": context.worker,
            "patch_index": context.patch_index,
        }
        if float(rng.random()) >= self.config.p_select:
            return EventResult("NO_OP", "NOT_SELECTED", event_id, seed, metadata=base_metadata)
        record = self.sampler.choose(rng, target_group)
        if record is None:
            return EventResult("NO_OP", "NO_ELIGIBLE_DONOR", event_id, seed, metadata=base_metadata)
        if record.patient_group == target_group:
            raise MetAugContractError("same patient-group donor escaped sampler")
        try:
            label = self.sampler.load_label(record)
            stats = label_statistics(label, record.spacing_mm)
        except (OSError, ValueError, MetAugContractError) as exc:
            return EventResult(
                "NO_OP", "LABEL_INVALID", event_id, seed, record=record,
                metadata={**base_metadata, "detail": str(exc)},
            )
        if (
            stats["core_volume_mm3"] < self.config.min_core_volume_mm3
            or any(value > self.config.max_bbox_mm + 1e-8 for value in stats["bbox_mm"])
            or tuple(stats["classes_present"]) != record.classes_present
        ):
            return EventResult("NO_OP", "LABEL_INVALID", event_id, seed, record=record, metadata=base_metadata)
        core_voxels = sum(
            int(stats["class_counts"].get(str(label_value), 0))
            for label_value in CORE_LABELS
        )
        if not self.config.is_support_eligible(
            total_support_voxels=int(stats["support_voxels"]),
            core_voxels=core_voxels,
        ):
            raise MetAugContractError(
                f"selected donor violates compact-support eligibility: {record.component_id}"
            )
        placement = find_placement(
            label=label,
            record=record,
            segmentation=segmentation,
            valid_mask=valid_mask,
            context=context,
            rng=rng,
            crop_size=self.config.crop_size,
            clearance_mm=self.config.clearance_mm,
            boundary_clearance_mm=self.config.boundary_clearance_mm,
        )
        if placement is None:
            return EventResult("NO_OP", "NO_VALID_PLACEMENT", event_id, seed, record=record, metadata=base_metadata)
        return EventResult("PLACEMENT_VALID", None, event_id, seed, record=record, placement=placement, metadata=base_metadata)

    def simulate(
        self,
        *,
        segmentation: np.ndarray,
        valid_mask: np.ndarray,
        context: EventContext,
        inputs_prevalidated: bool = False,
    ) -> EventResult:
        result = self.plan(
            segmentation=segmentation,
            valid_mask=valid_mask,
            context=context,
            inputs_prevalidated=inputs_prevalidated,
        )
        self._append(result)
        return result

    def apply(
        self,
        *,
        image: np.ndarray,
        segmentation: np.ndarray,
        valid_mask: np.ndarray,
        context: EventContext,
    ) -> tuple[np.ndarray, np.ndarray, EventResult]:
        if image.ndim != 4 or image.shape[0] != 4:
            raise MetAugContractError(f"Route A requires image shape (4,Z,Y,X), got {image.shape}")
        if segmentation.ndim != 4 or segmentation.shape[0] != 1:
            raise MetAugContractError(f"Route A requires segmentation shape (1,Z,Y,X), got {segmentation.shape}")
        if image.shape[1:] != segmentation.shape[1:] or valid_mask.shape != image.shape[1:]:
            raise MetAugContractError("Route A image, segmentation, and valid-mask shapes disagree")
        result = self.plan(segmentation=segmentation, valid_mask=valid_mask, context=context)
        if result.state != "PLACEMENT_VALID":
            self._append(result)
            return image, segmentation, result
        if self.backend is None:
            result = EventResult(
                "NO_OP", "DIFFUSION_BACKEND_DISABLED", result.event_id, result.event_seed,
                record=result.record, placement=result.placement, metadata=result.metadata,
            )
            self._append(result)
            return image, segmentation, result
        assert result.placement is not None
        placement = result.placement
        start = placement.crop_start
        stop = tuple(value + self.config.crop_size for value in start)
        slices = tuple(slice(begin, end) for begin, end in zip(start, stop))
        original_crop = image[(slice(None),) + slices].astype(np.float32, copy=True)
        if self.config.candidate_policy is not None:
            return self._apply_fix_v2(
                image=image,
                segmentation=segmentation,
                valid_mask=valid_mask,
                result=result,
                slices=slices,
                original_crop=original_crop,
            )
        try:
            generated = self.backend.generate(original_crop, placement.label_cube, seed=result.event_seed)
        except Exception as exc:
            result = EventResult(
                "NO_OP", "MODALITY_QC_FAIL", result.event_id, result.event_seed,
                record=result.record, placement=placement,
                metadata={**result.metadata, "detail": f"backend: {type(exc).__name__}: {exc}"},
            )
            self._append(result)
            return image, segmentation, result
        if generated.shape != original_crop.shape or not np.all(np.isfinite(generated)):
            result = EventResult(
                "NO_OP", "MODALITY_QC_FAIL", result.event_id, result.event_seed,
                record=result.record, placement=placement,
                metadata={**result.metadata, "detail": "generated four-modality crop is non-finite or malformed"},
            )
            self._append(result)
            return image, segmentation, result
        draft_image = image.copy()
        draft_segmentation = segmentation.copy()
        support = placement.support
        for channel in range(4):
            updated = draft_image[(channel,) + slices]
            updated[support] = generated[channel][support].astype(updated.dtype, copy=False)
        seg_crop = draft_segmentation[(0,) + slices]
        seg_crop[support] = placement.label_cube[support].astype(seg_crop.dtype, copy=False)
        if not self._validate_commit(image, segmentation, draft_image, draft_segmentation, placement):
            result = EventResult(
                "NO_OP", "COMMIT_QC_FAIL", result.event_id, result.event_seed,
                record=result.record, placement=placement, metadata=result.metadata,
            )
            self._append(result)
            return image, segmentation, result
        result = EventResult(
            "COMMITTED", None, result.event_id, result.event_seed,
            record=result.record, placement=placement, metadata=result.metadata,
        )
        try:
            self._append(result)
        except Exception as exc:
            raise MetAugAuditError("MET-AUG audit write failed; transaction was not committed") from exc
        return draft_image, draft_segmentation, result

    def _apply_fix_v2(
        self,
        *,
        image: np.ndarray,
        segmentation: np.ndarray,
        valid_mask: np.ndarray,
        result: EventResult,
        slices: tuple[slice, slice, slice],
        original_crop: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, EventResult]:
        version = "fix_v3" if self.config.fix_v3 is not None else "fix_v2"
        if self.candidate_processor is None:
            raise MetAugContractError(
                f"{version} Route A requires a frozen candidate processor"
            )
        if self.backend is None:
            raise MetAugContractError(f"{version} Route A requires a diffusion backend")
        assert result.placement is not None
        assert result.record is not None
        placement = result.placement
        original_segmentation = segmentation[(0,) + slices].astype(
            segmentation.dtype, copy=True
        )
        valid_crop = valid_mask[slices].astype(bool, copy=False)
        try:
            processed = self.candidate_processor.process(
                original_image=original_crop,
                original_segmentation=original_segmentation,
                label_cube=placement.label_cube,
                valid_mask=valid_crop,
                spacing_mm=tuple(float(value) for value in result.record.spacing_mm),
                core_volume_mm3=float(result.record.core_volume_mm3),
                seed=result.event_seed,
                backend=self.backend,
            )
        except Exception as exc:
            failed = EventResult(
                "NO_OP",
                "FIX_V3_PROCESSING_FAIL" if version == "fix_v3" else "FIX_V2_PROCESSING_FAIL",
                result.event_id,
                result.event_seed,
                record=result.record,
                placement=placement,
                metadata={
                    **result.metadata,
                    "detail": f"processor: {type(exc).__name__}: {exc}",
                },
            )
            self._append(failed)
            return image, segmentation, failed

        processor_metadata = {version: dict(processed.metadata)}
        if processed.reason is not None:
            rejected = EventResult(
                "NO_OP",
                processed.reason,
                result.event_id,
                result.event_seed,
                record=result.record,
                placement=placement,
                metadata={**result.metadata, **processor_metadata},
                evidence=processed.evidence,
            )
            self._append(rejected)
            return image, segmentation, rejected

        expected_image_shape = original_crop.shape
        expected_segmentation_shape = original_segmentation.shape
        support_shape = original_segmentation.shape
        if (
            processed.image.shape != expected_image_shape
            or processed.segmentation.shape != expected_segmentation_shape
            or processed.image_support.shape != support_shape
            or processed.label_support.shape != support_shape
            or not np.all(np.isfinite(processed.image))
        ):
            rejected = EventResult(
                "NO_OP",
                "COMMIT_CONTRACT_FAIL",
                result.event_id,
                result.event_seed,
                record=result.record,
                placement=placement,
                metadata={
                    **result.metadata,
                    **processor_metadata,
                    "detail": f"{version} processor returned malformed candidate arrays",
                },
            )
            self._append(rejected)
            return image, segmentation, rejected

        local_image_support = processed.image_support.astype(bool, copy=False)
        local_label_support = processed.label_support.astype(bool, copy=False)
        if (
            not np.array_equal(local_label_support, placement.support)
            or np.any(local_label_support & ~local_image_support)
        ):
            rejected = EventResult(
                "NO_OP",
                "COMMIT_CONTRACT_FAIL",
                result.event_id,
                result.event_seed,
                record=result.record,
                placement=placement,
                metadata={
                    **result.metadata,
                    **processor_metadata,
                    "detail": f"{version} support masks violate the placement contract",
                },
            )
            self._append(rejected)
            return image, segmentation, rejected

        draft_image = image.copy()
        draft_segmentation = segmentation.copy()
        draft_image[(slice(None),) + slices] = processed.image.astype(
            draft_image.dtype, copy=False
        )
        draft_segmentation[(0,) + slices] = processed.segmentation.astype(
            draft_segmentation.dtype, copy=False
        )
        if not self._validate_candidate_commit(
            image,
            segmentation,
            draft_image,
            draft_segmentation,
            placement,
            local_image_support=local_image_support,
            local_label_support=local_label_support,
        ):
            rejected = EventResult(
                "NO_OP",
                "COMMIT_CONTRACT_FAIL",
                result.event_id,
                result.event_seed,
                record=result.record,
                placement=placement,
                metadata={**result.metadata, **processor_metadata},
            )
            self._append(rejected)
            return image, segmentation, rejected

        committed = EventResult(
            "COMMITTED",
            None,
            result.event_id,
            result.event_seed,
            record=result.record,
            placement=placement,
            metadata={**result.metadata, **processor_metadata},
            evidence=processed.evidence,
        )
        try:
            self._append(committed)
        except Exception as exc:
            raise MetAugAuditError(
                "MET-AUG audit write failed; transaction was not committed"
            ) from exc
        return draft_image, draft_segmentation, committed

    def _validate_commit(
        self,
        before_image: np.ndarray,
        before_segmentation: np.ndarray,
        after_image: np.ndarray,
        after_segmentation: np.ndarray,
        placement: Placement,
    ) -> bool:
        return self._validate_candidate_commit(
            before_image,
            before_segmentation,
            after_image,
            after_segmentation,
            placement,
            local_image_support=placement.support,
            local_label_support=placement.support,
        )

    def _validate_candidate_commit(
        self,
        before_image: np.ndarray,
        before_segmentation: np.ndarray,
        after_image: np.ndarray,
        after_segmentation: np.ndarray,
        placement: Placement,
        *,
        local_image_support: np.ndarray,
        local_label_support: np.ndarray,
    ) -> bool:
        start = placement.crop_start
        stop = tuple(value + self.config.crop_size for value in start)
        slices = tuple(slice(begin, end) for begin, end in zip(start, stop))
        global_image_support = np.zeros(before_segmentation.shape[1:], dtype=bool)
        global_image_support[slices] = local_image_support
        global_label_support = np.zeros(before_segmentation.shape[1:], dtype=bool)
        global_label_support[slices] = local_label_support
        if not np.all(np.isfinite(after_image)):
            return False
        changed_image = np.any(after_image != before_image, axis=0)
        changed_seg = after_segmentation[0] != before_segmentation[0]
        if np.any(changed_image & ~global_image_support) or np.any(
            changed_seg & ~global_label_support
        ):
            return False
        values = set(int(value) for value in np.unique(after_segmentation))
        if not values.issubset(ALLOWED_LABELS | {-1}):
            return False
        expected = placement.label_cube[local_label_support]
        actual = after_segmentation[(0,) + slices][local_label_support]
        return bool(np.array_equal(actual, expected))

    def _append(self, result: EventResult) -> None:
        try:
            self.audit_sink.append(result.audit_mapping())
        except Exception as exc:
            raise MetAugAuditError("unable to append required MET-AUG event audit") from exc
