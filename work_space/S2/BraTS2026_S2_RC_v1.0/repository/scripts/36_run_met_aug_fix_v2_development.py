#!/usr/bin/env python3
"""Generate blinded A/B/C Development evidence with one shared G1 runtime."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import (  # noqa: E402
    ComponentManifest,
    EventContext,
    MemoryAuditSink,
    MetAugEngine,
    RouteConfig,
    canonical_json_sha256,
    sha256_file,
)
from custom_nnunet.met_aug_diffusion import (  # noqa: E402
    G1FourModalityInpaintingBackend,
)
from custom_nnunet.met_aug_fix_v2 import (  # noqa: E402
    FixV2CandidateProcessor,
)
from custom_nnunet.met_aug_gate2 import (  # noqa: E402
    load_smoke_manifest,
    load_valid_mask_assets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-manifest", required=True)
    parser.add_argument("--measurement-config-index", required=True)
    parser.add_argument("--event-manifest", required=True)
    parser.add_argument("--valid-mask-manifest", required=True)
    parser.add_argument("--preprocessed-dir", required=True)
    parser.add_argument("--g1-code-dir", required=True)
    parser.add_argument("--g1-checkpoint-root", required=True)
    parser.add_argument("--g1-checkpoint-selection", required=True)
    parser.add_argument("--g2-parent-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--blind-seed", type=int, default=20260728)
    return parser.parse_args()


def _load_renderer():
    path = REPOSITORY_ROOT / "scripts" / "18_run_met_aug_gate2_smoke.py"
    spec = importlib.util.spec_from_file_location("fix_v2_gate2_renderer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the frozen Gate-2 renderer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render_montage


def _context(entry: dict[str, Any]) -> EventContext:
    value = entry["event"]
    return EventContext(
        epoch=int(value["epoch"]),
        rank=int(value["rank"]),
        worker=int(value["worker"]),
        case_id=str(value["case_id"]),
        patch_index=int(value["patch_index"]),
        patch_origin=tuple(int(item) for item in value.get("patch_origin", (0, 0, 0))),
        full_shape=tuple(int(item) for item in value["full_shape"]),
    )


def _load_case(dataset, case_id: str) -> tuple[np.ndarray, np.ndarray]:
    image, segmentation, previous_stage, _properties = dataset.load_case(case_id)
    if previous_stage is not None:
        raise ValueError(f"{case_id}: cascaded data is forbidden")
    image = np.asarray(image, dtype=np.float32)
    segmentation = np.asarray(segmentation, dtype=np.int16)
    if image.ndim != 4 or image.shape[0] != 4:
        raise ValueError(f"{case_id}: expected four image channels")
    if segmentation.ndim != 4 or segmentation.shape[0] != 1:
        raise ValueError(f"{case_id}: expected one segmentation channel")
    if image.shape[1:] != segmentation.shape[1:] or not np.all(np.isfinite(image)):
        raise ValueError(f"{case_id}: malformed preprocessed case")
    return image, segmentation


class _ReplayBackend:
    def __init__(self, generated: np.ndarray):
        self.generated = np.asarray(generated, dtype=np.float32)

    def generate(self, image, label, *, seed, inpaint_support=None):
        if self.generated.shape != image.shape:
            raise ValueError("replayed raw generation shape drifted")
        support = label != 0 if inpaint_support is None else np.asarray(inpaint_support, dtype=bool)
        if np.any(self.generated[:, ~support] != image[:, ~support]):
            raise ValueError("replayed raw generation changed the known region")
        return self.generated.copy()


class _CaptureBackend:
    def __init__(self, backend: Any):
        self.backend = backend
        self.generated: np.ndarray | None = None

    def generate(self, image, label, *, seed, inpaint_support=None):
        generated = self.backend.generate(
            image,
            label,
            seed=seed,
            inpaint_support=inpaint_support,
        )
        self.generated = np.asarray(generated, dtype=np.float32).copy()
        return generated


class _UnavailableReplayBackend:
    def generate(self, image, label, *, seed, inpaint_support=None):
        raise RuntimeError(
            "paired C candidate reached generation although its B candidate "
            "was rejected before generation"
        )


def _blind_code(seed: int, event_id: str, candidate_id: str) -> str:
    return canonical_json_sha256(
        {"seed": int(seed), "event_id": event_id, "candidate_id": candidate_id}
    )[:16]


def _candidate_groups(index: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    """Return an explicit A, B(radius), C(radius) topological execution order."""
    by_radius_b: dict[float, dict[str, Any]] = {}
    by_radius_c: dict[float, dict[str, Any]] = {}
    candidates = list(index["candidates"])
    candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("measurement config index has duplicate candidate IDs")
    a_candidates = [
        candidate
        for candidate in candidates
        if str(candidate["candidate_id"]) == "A_label_only"
    ]
    if len(a_candidates) != 1:
        raise ValueError("measurement config index must contain exactly one A candidate")
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        radius = float(candidate["halo_radius_mm"])
        if candidate_id == "A_label_only":
            if radius != 0.0 or candidate["boundary_policy"] != "label_only_qc_v1":
                raise ValueError("A candidate policy or radius drifted")
        elif candidate_id.startswith("B_"):
            if radius in by_radius_b or candidate["boundary_policy"] != "halo_cosine_v1":
                raise ValueError("B candidate radius or policy drifted")
            by_radius_b[radius] = candidate
        elif candidate_id.startswith("C_"):
            if (
                radius in by_radius_c
                or candidate["boundary_policy"] != "halo_cosine_harmonized_v1"
            ):
                raise ValueError("C candidate radius or policy drifted")
            by_radius_c[radius] = candidate
        else:
            raise ValueError(f"unsupported measurement candidate: {candidate_id}")
    if set(by_radius_b) != set(by_radius_c) or not by_radius_b:
        raise ValueError("B/C candidate radii are not exactly paired")
    result: list[tuple[dict[str, Any], dict[str, Any] | None]] = [
        (a_candidates[0], None)
    ]
    for radius in sorted(by_radius_b):
        b_candidate = by_radius_b[radius]
        result.append((b_candidate, None))
        result.append((by_radius_c[radius], b_candidate))
    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Development output already exists: {output_dir}")
    manifest = ComponentManifest.load(args.component_manifest)
    index_path = Path(args.measurement_config_index).expanduser().resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("status") != "measurement_only_not_gate_eligible":
        raise ValueError("measurement config index has an invalid role")
    if index.get("component_manifest_sha256") != manifest.identity_sha256:
        raise ValueError("measurement config index binds another component manifest")
    config_root = index_path.parent
    processors: dict[str, FixV2CandidateProcessor] = {}
    configs: dict[str, RouteConfig] = {}
    for candidate in index["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        calibration_path = config_root / str(candidate["calibration_file"])
        config_path = config_root / str(candidate["route_config_file"])
        if sha256_file(calibration_path) != candidate["calibration_sha256"]:
            raise ValueError(f"{candidate_id}: measurement calibration drifted")
        if sha256_file(config_path) != candidate["route_config_sha256"]:
            raise ValueError(f"{candidate_id}: route config drifted")
        config = RouteConfig.load(config_path, manifest)
        processors[candidate_id] = FixV2CandidateProcessor.load(
            calibration_path,
            expected_sha256=str(candidate["calibration_sha256"]),
            expected_policy=str(candidate["boundary_policy"]),
        )
        configs[candidate_id] = config
    candidates = _candidate_groups(index)
    a_candidate = next(
        candidate for candidate, _paired in candidates if candidate["candidate_id"] == "A_label_only"
    )
    planning_config = configs[str(a_candidate["candidate_id"])]

    event_manifest_path = Path(args.event_manifest).expanduser().resolve()
    event_manifest = load_smoke_manifest(
        event_manifest_path,
        manifest=manifest,
        config=planning_config,
        valid_mask_manifest_path=args.valid_mask_manifest,
    )
    if event_manifest.get("calibration_partition") != "development":
        raise ValueError("Development runner received a non-Development manifest")
    if event_manifest.get("component_manifest_sha256") != manifest.identity_sha256:
        raise ValueError("Development manifest binds another component pool")
    events = list(event_manifest.get("smoke_cases", ()))
    if len(events) != int(event_manifest.get("smoke_count", -1)) or not events:
        raise ValueError("Development event count is malformed")
    target_ids = {str(entry["target_case_id"]) for entry in events}
    assets = load_valid_mask_assets(args.valid_mask_manifest, expected_ids=target_ids)
    preprocessed_dir = Path(args.preprocessed_dir).expanduser().resolve()
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

    dataset_class = infer_dataset_class(str(preprocessed_dir))
    dataset = dataset_class(str(preprocessed_dir), sorted(target_ids))
    planner = MetAugEngine(
        manifest=manifest,
        config=planning_config,
        backend=None,
        audit_sink=MemoryAuditSink(),
    )
    backend = G1FourModalityInpaintingBackend(
        g1_code_dir=args.g1_code_dir,
        checkpoint_root=args.g1_checkpoint_root,
        checkpoint_selection=args.g1_checkpoint_selection,
        qc_gate=args.g2_parent_gate,
        device=args.device,
    )
    render_montage = _load_renderer()
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts_dir = output_dir / "artifacts"
    montages_dir = output_dir / "montages"
    artifacts_dir.mkdir()
    montages_dir.mkdir()
    rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, str]] = []
    state_counts: dict[str, Counter[str]] = defaultdict(Counter)
    started = time.monotonic()
    for event_index, entry in enumerate(events):
        case_id = str(entry["target_case_id"])
        image, segmentation = _load_case(dataset, case_id)
        valid_mask, _foreground = assets[case_id]
        context = _context(entry)
        planned = planner.plan(
            segmentation=segmentation,
            valid_mask=valid_mask,
            context=context,
        )
        if (
            planned.state != "PLACEMENT_VALID"
            or planned.event_id != entry["event_id"]
            or planned.event_seed != entry["event_seed"]
            or planned.record is None
            or planned.record.component_id != entry["donor_component_id"]
            or planned.placement is None
            or list(planned.placement.crop_start) != entry["planned"]["crop_start"]
        ):
            raise RuntimeError(f"{entry['smoke_id']}: frozen Development plan drifted")
        placement = planned.placement
        start = placement.crop_start
        slices = tuple(slice(value, value + planning_config.crop_size) for value in start)
        original = image[(slice(None),) + slices].astype(np.float32, copy=True)
        original_seg = segmentation[(0,) + slices].astype(np.int16, copy=True)
        valid_crop = valid_mask[slices].astype(bool, copy=True)
        raw_by_candidate: dict[str, np.ndarray | None] = {}
        for candidate, paired_b in candidates:
            candidate_id = str(candidate["candidate_id"])
            processor = processors[candidate_id]
            capture_backend: _CaptureBackend | None = None
            if paired_b is not None:
                paired_id = str(paired_b["candidate_id"])
                if paired_id not in raw_by_candidate:
                    raise RuntimeError(f"{candidate_id}: paired B candidate did not execute first")
                paired_raw = raw_by_candidate[paired_id]
                candidate_backend: Any = (
                    _ReplayBackend(paired_raw)
                    if paired_raw is not None
                    else _UnavailableReplayBackend()
                )
            else:
                capture_backend = _CaptureBackend(backend)
                candidate_backend = capture_backend
            event_started = time.monotonic()
            processed = processor.process(
                original_image=original,
                original_segmentation=original_seg,
                label_cube=placement.label_cube,
                valid_mask=valid_crop,
                spacing_mm=planned.record.spacing_mm,
                core_volume_mm3=planned.record.core_volume_mm3,
                seed=planned.event_seed,
                backend=candidate_backend,
            )
            if capture_backend is not None:
                raw_by_candidate[candidate_id] = capture_backend.generated
            elapsed = time.monotonic() - event_started
            status = "measured" if processed.reason is None else "rejected_before_threshold_freeze"
            state_counts[candidate_id][processed.reason or "MEASURED"] += 1
            blind_code = _blind_code(args.blind_seed, planned.event_id, candidate_id)
            artifact_path = artifacts_dir / f"{blind_code}.npz"
            montage_path = montages_dir / f"{blind_code}.png"
            artifact_payload: dict[str, Any] = {
                "original": original,
                "original_segmentation": original_seg,
                "inserted_label": placement.label_cube,
                "valid_mask": valid_crop,
                "candidate": processed.image,
                "candidate_segmentation": processed.segmentation,
                "event_json": np.asarray(json.dumps(entry, ensure_ascii=True, sort_keys=True)),
                "metadata_json": np.asarray(json.dumps(processed.metadata, ensure_ascii=True, sort_keys=True)),
            }
            if processed.reason is None:
                for key, value in processed.evidence.items():
                    artifact_payload[str(key)] = np.asarray(value)
                raw = np.asarray(processed.evidence["raw_generation"], dtype=np.float32)
                if paired_b is None:
                    captured = raw_by_candidate[candidate_id]
                    if captured is None or not np.array_equal(raw, captured):
                        raise RuntimeError(f"{candidate_id}: captured raw output drifted")
                else:
                    paired_raw = raw_by_candidate[str(paired_b["candidate_id"])]
                    if paired_raw is None or not np.array_equal(raw, paired_raw):
                        raise RuntimeError(
                            f"{candidate_id}: B/C raw arrays are not byte-identical"
                        )
            np.savez_compressed(artifact_path, **artifact_payload)
            if processed.reason is None:
                render_montage(
                    image_before=original,
                    image_after=processed.image,
                    segmentation_after=processed.segmentation,
                    support=np.asarray(processed.evidence["image_support"], dtype=bool),
                    output=montage_path,
                    raw_generation=np.asarray(processed.evidence["raw_generation"]),
                    harmonized_generation=np.asarray(processed.evidence["harmonized_generation"]),
                    pre_harmonization=np.asarray(processed.evidence["pre_harmonization"]),
                    label_support=np.asarray(processed.evidence["label_support"], dtype=bool),
                    boundary_masks=tuple(
                        np.asarray(processed.evidence[f"boundary_label_{value}"], dtype=bool)
                        for value in (1, 2, 3)
                    ),
                    boundary_policy=str(candidate["boundary_policy"]),
                    qc_metadata=dict(processed.metadata),
                    display_label=blind_code,
                )
                review_rows.append(
                    {
                        "blind_code": blind_code,
                        "event_id": planned.event_id,
                        "core_volume_bin": str(entry["core_volume_bin"]),
                        "montage_path": str(montage_path.relative_to(output_dir)),
                        "artifact_path": str(artifact_path.relative_to(output_dir)),
                        "evidence_sha256": sha256_file(artifact_path),
                        "review_decision": "pending",
                        "reviewer": "",
                        "reviewed_at_utc": "",
                        "notes": "",
                    }
                )
            row = {
                "event_index": event_index,
                "smoke_id": entry["smoke_id"],
                "event_id": planned.event_id,
                "event_seed": planned.event_seed,
                "target_case_id": case_id,
                "target_patient_group": entry["target_patient_group"],
                "donor_component_id": entry["donor_component_id"],
                "donor_patient_group": entry["donor_patient_group"],
                "core_volume_bin": entry["core_volume_bin"],
                "candidate_id": candidate_id,
                "blind_code": blind_code,
                "boundary_policy": candidate["boundary_policy"],
                "halo_radius_mm": candidate["halo_radius_mm"],
                "status": status,
                "reason": processed.reason,
                "elapsed_seconds": elapsed,
                "artifact_path": str(artifact_path.relative_to(output_dir)),
                "artifact_sha256": sha256_file(artifact_path),
                "montage_path": str(montage_path.relative_to(output_dir)) if montage_path.is_file() else "",
                "montage_sha256": sha256_file(montage_path) if montage_path.is_file() else "",
                "metadata": processed.metadata,
            }
            row["row_sha256"] = canonical_json_sha256(row, exclude=("row_sha256",))
            rows.append(row)
    results_path = output_dir / "development_measurements.jsonl"
    results_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    review_path = output_dir / "blinded_manual_review_template.csv"
    fields = (
        "blind_code", "event_id", "core_volume_bin", "montage_path",
        "artifact_path", "evidence_sha256", "review_decision", "reviewer",
        "reviewed_at_utc", "notes",
    )
    with review_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(review_rows, key=lambda row: row["blind_code"]))
    private_map = output_dir / "PRIVATE_BLINDING_MAP.json"
    private_payload = {
        "schema_version": 1,
        "blind_seed": args.blind_seed,
        "entries": [
            {"blind_code": row["blind_code"], "event_id": row["event_id"], "candidate_id": row["candidate_id"]}
            for row in rows
        ],
    }
    private_map.write_text(
        json.dumps(private_payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    private_map.chmod(0o400)
    report = {
        "schema_version": 1,
        "status": "hold_for_blinded_manual_review",
        "event_count": len(events),
        "candidate_count": len(candidates),
        "attempt_count": len(rows),
        "reviewable_count": len(review_rows),
        "state_counts": {
            candidate: dict(sorted(counts.items()))
            for candidate, counts in sorted(state_counts.items())
        },
        "elapsed_seconds": time.monotonic() - started,
        "component_manifest_sha256": manifest.identity_sha256,
        "measurement_config_index_sha256": sha256_file(index_path),
        "event_manifest_sha256": sha256_file(event_manifest_path),
        "valid_mask_manifest_sha256": sha256_file(args.valid_mask_manifest),
        "g1_checkpoint_selection_sha256": sha256_file(args.g1_checkpoint_selection),
        "g2_parent_gate_sha256": sha256_file(args.g2_parent_gate),
        "g1_runtime_code": backend.runtime_code,
        "measurements_file": results_path.name,
        "measurements_sha256": sha256_file(results_path),
        "manual_review_template": review_path.name,
        "manual_review_template_sha256": sha256_file(review_path),
        "private_blinding_map_sha256": sha256_file(private_map),
        "runner_sha256": sha256_file(Path(__file__)),
    }
    report["report_sha256"] = canonical_json_sha256(report, exclude=("report_sha256",))
    report_path = output_dir / "DEVELOPMENT_REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
