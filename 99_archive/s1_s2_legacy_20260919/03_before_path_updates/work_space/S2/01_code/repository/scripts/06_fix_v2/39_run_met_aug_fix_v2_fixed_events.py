#!/usr/bin/env python3
"""Run one immutable Fix-v2 fixed-event set with auditable rejected previews."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import (  # noqa: E402
    ComponentManifest,
    EventContext,
    JsonlAuditSink,
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


STAGE_CONTRACTS = {
    "qc_holdout": ("qc_holdout", 48, 16),
    "gate1b_replay": ("reference", 96, 32),
    "gate2": ("reference", 120, 40),
}
EXPECTED_QC_REASONS = {
    "BOUNDARY_QC_INSUFFICIENT_SUPPORT",
    "RAW_GENERATION_QC_FAIL",
    "HARMONIZATION_FAIL",
    "CANDIDATE_BOUNDARY_QC_FAIL",
    "CANDIDATE_CROSS_MODAL_QC_FAIL",
    "CANDIDATE_CONTENT_QC_FAIL",
}
REVIEW_FIELDS = (
    "blind_code",
    "event_id",
    "core_volume_bin",
    "montage_path",
    "artifact_path",
    "evidence_sha256",
    "review_decision",
    "reviewer",
    "reviewed_at_utc",
    "notes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGE_CONTRACTS), required=True)
    parser.add_argument("--component-manifest", required=True)
    parser.add_argument("--route-config", required=True)
    parser.add_argument("--strict-calibration", required=True)
    parser.add_argument("--measurement-calibration", required=True)
    parser.add_argument("--valid-mask-manifest", required=True)
    parser.add_argument("--event-manifest", required=True)
    parser.add_argument("--preprocessed-dir", required=True)
    parser.add_argument("--g1-code-dir", required=True)
    parser.add_argument("--g1-checkpoint-root", required=True)
    parser.add_argument("--g1-checkpoint-selection", required=True)
    parser.add_argument("--g2-parent-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--blind-seed", type=int, required=True)
    return parser.parse_args()


def _load_renderer():
    path = REPOSITORY_ROOT / "scripts" / "18_run_met_aug_gate2_smoke.py"
    spec = importlib.util.spec_from_file_location("fix_v2_fixed_event_renderer", path)
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


class _CaptureBackend:
    def __init__(self, backend: Any):
        self.backend = backend
        self.generated: np.ndarray | None = None

    def reset(self) -> None:
        self.generated = None

    def generate(self, image, label, *, seed, inpaint_support=None):
        if self.generated is not None:
            raise RuntimeError("backend capture was not reset between fixed events")
        generated = self.backend.generate(
            image,
            label,
            seed=seed,
            inpaint_support=inpaint_support,
        )
        self.generated = np.asarray(generated, dtype=np.float32).copy()
        return generated


class _ReplayBackend:
    def __init__(self, generated: np.ndarray):
        self.generated = np.asarray(generated, dtype=np.float32)

    def generate(self, image, label, *, seed, inpaint_support=None):
        support = label != 0 if inpaint_support is None else np.asarray(inpaint_support, dtype=bool)
        if self.generated.shape != image.shape:
            raise ValueError("fixed-event replay raw shape drifted")
        if np.any(self.generated[:, ~support] != image[:, ~support]):
            raise ValueError("fixed-event replay changed the known region")
        return self.generated.copy()


def _blind_code(seed: int, stage: str, event_id: str) -> str:
    return canonical_json_sha256(
        {"blind_seed": int(seed), "stage": stage, "event_id": event_id}
    )[:16]


def _exclusive_json(path: Path, payload: dict[str, Any], mode: int = 0o444) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _frozen_identity_violations(entry: dict[str, Any], result) -> list[str]:
    violations: list[str] = []
    if result.event_id != entry["event_id"] or result.event_seed != int(entry["event_seed"]):
        violations.append("event_identity_drift")
    if result.record is None or result.record.component_id != entry["donor_component_id"]:
        violations.append("donor_component_drift")
    if result.placement is None:
        violations.append("placement_missing")
        return violations
    planned = entry["planned"]
    if list(result.placement.crop_start) != planned["crop_start"]:
        violations.append("crop_start_drift")
    if int(np.count_nonzero(result.placement.support)) != int(planned["support_voxels"]):
        violations.append("support_size_drift")
    return violations


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"fixed-event output already exists: {output_dir}")
    expected_partition, expected_count, expected_per_bin = STAGE_CONTRACTS[args.stage]
    manifest = ComponentManifest.load(args.component_manifest)
    config = RouteConfig.load(args.route_config, manifest)
    if config.fix_v2 is None or config.fix_v2.boundary_policy != "label_only_qc_v1":
        raise ValueError("fixed-event runner requires the selected A Fix-v2 route")
    strict_calibration_path = Path(args.strict_calibration).expanduser().resolve()
    strict_processor = FixV2CandidateProcessor.load(
        strict_calibration_path,
        expected_sha256=config.fix_v2.calibration_sha256,
        expected_policy=config.fix_v2.boundary_policy,
    )
    measurement_calibration_path = Path(args.measurement_calibration).expanduser().resolve()
    measurement_processor = FixV2CandidateProcessor.load(
        measurement_calibration_path,
        expected_sha256=sha256_file(measurement_calibration_path),
        expected_policy="label_only_qc_v1",
    )
    if measurement_processor.calibration.payload.get("calibration_role") != (
        "measurement_only_not_gate_eligible"
    ):
        raise ValueError("preview processor is not the frozen measurement-only A calibration")

    event_manifest_path = Path(args.event_manifest).expanduser().resolve()
    event_manifest = load_smoke_manifest(
        event_manifest_path,
        manifest=manifest,
        config=config,
        valid_mask_manifest_path=args.valid_mask_manifest,
    )
    if event_manifest.get("calibration_partition") != expected_partition:
        raise ValueError("fixed-event manifest partition does not match the requested stage")
    events = list(event_manifest["smoke_cases"])
    if len(events) != expected_count or int(event_manifest["smoke_count"]) != expected_count:
        raise ValueError("fixed-event manifest count drifted")
    if any(
        int(event_manifest["per_volume_bin"].get(volume_bin, -1)) != expected_per_bin
        for volume_bin in ("27_49", "50_275", "gt_275")
    ):
        raise ValueError("fixed-event manifest volume-bin denominator drifted")
    target_ids = {str(entry["target_case_id"]) for entry in events}
    assets = load_valid_mask_assets(args.valid_mask_manifest, expected_ids=target_ids)

    preprocessed_dir = Path(args.preprocessed_dir).expanduser().resolve()
    if not preprocessed_dir.is_dir():
        raise FileNotFoundError(f"missing preprocessed directory: {preprocessed_dir}")
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

    dataset_class = infer_dataset_class(str(preprocessed_dir))
    dataset = dataset_class(str(preprocessed_dir), sorted(target_ids))
    backend = G1FourModalityInpaintingBackend(
        g1_code_dir=args.g1_code_dir,
        checkpoint_root=args.g1_checkpoint_root,
        checkpoint_selection=args.g1_checkpoint_selection,
        qc_gate=args.g2_parent_gate,
        device=args.device,
    )
    capture_backend = _CaptureBackend(backend)
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts_dir = output_dir / "artifacts"
    montages_dir = output_dir / "montages"
    artifacts_dir.mkdir()
    montages_dir.mkdir()
    event_audit_path = output_dir / "transaction_events.jsonl"
    engine = MetAugEngine(
        manifest=manifest,
        config=config,
        backend=capture_backend,
        audit_sink=JsonlAuditSink(event_audit_path),
        candidate_processor=strict_processor,
    )
    render_montage = _load_renderer()
    rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, str]] = []
    private_entries: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    global_violations: list[str] = []

    for event_index, entry in enumerate(events):
        case_id = str(entry["target_case_id"])
        image_before, segmentation_before = _load_case(dataset, case_id)
        valid_mask, _foreground = assets[case_id]
        capture_backend.reset()
        image_after, segmentation_after, result = engine.apply(
            image=image_before,
            segmentation=segmentation_before,
            valid_mask=valid_mask,
            context=_context(entry),
        )
        violations = _frozen_identity_violations(entry, result)
        if result.state not in {"COMMITTED", "NO_OP"}:
            violations.append(f"unexpected_transaction_state:{result.state}")
        if result.state == "NO_OP" and result.reason not in EXPECTED_QC_REASONS:
            violations.append(f"unexpected_rejection_reason:{result.reason}")
        if result.state == "NO_OP" and (
            not np.array_equal(image_after, image_before)
            or not np.array_equal(segmentation_after, segmentation_before)
        ):
            violations.append("rejected_transaction_changed_full_case")
        if result.state == "COMMITTED" and result.reason is not None:
            violations.append("committed_transaction_has_reason")
        if not np.all(np.isfinite(image_after)):
            violations.append("nonfinite_full_case_output")
        if capture_backend.generated is None:
            violations.append("raw_generation_not_captured")
            raw = np.zeros((4, 64, 64, 64), dtype=np.float32)
        else:
            raw = capture_backend.generated
        if result.placement is None or result.record is None:
            raise RuntimeError(f"{entry['smoke_id']}: fixed placement evidence is missing")
        start = result.placement.crop_start
        slices = tuple(slice(value, value + 64) for value in start)
        original_crop = image_before[(slice(None),) + slices].astype(np.float32, copy=True)
        original_segmentation = segmentation_before[(0,) + slices].astype(np.int16, copy=True)
        valid_crop = valid_mask[slices].astype(bool, copy=True)
        preview = measurement_processor.process(
            original_image=original_crop,
            original_segmentation=original_segmentation,
            label_cube=result.placement.label_cube,
            valid_mask=valid_crop,
            spacing_mm=result.record.spacing_mm,
            core_volume_mm3=result.record.core_volume_mm3,
            seed=result.event_seed,
            backend=_ReplayBackend(raw),
        )
        if preview.reason is not None:
            violations.append(f"measurement_preview_failed:{preview.reason}")
        strict_crop = image_after[(slice(None),) + slices].astype(np.float32, copy=True)
        strict_segmentation = segmentation_after[(0,) + slices].astype(np.int16, copy=True)
        label_support = result.placement.support.astype(bool, copy=False)
        image_support = np.asarray(
            preview.evidence.get("image_support", label_support), dtype=bool
        )
        if np.any(strict_crop[:, ~image_support] != original_crop[:, ~image_support]):
            violations.append("strict_crop_changed_outside_image_support")
        if result.state == "COMMITTED" and not np.array_equal(strict_crop, preview.image):
            violations.append("committed_strict_and_preview_candidates_differ")

        blind_code = _blind_code(args.blind_seed, args.stage, result.event_id)
        artifact_path = artifacts_dir / f"{blind_code}.npz"
        montage_path = montages_dir / f"{blind_code}.png"
        np.savez_compressed(
            artifact_path,
            original=original_crop,
            original_segmentation=original_segmentation,
            inserted_label=result.placement.label_cube,
            valid_mask=valid_crop,
            raw_generation=raw,
            preview_candidate=preview.image,
            preview_segmentation=preview.segmentation,
            strict_candidate=strict_crop,
            strict_segmentation=strict_segmentation,
            label_support=label_support,
            image_support=image_support,
            event_json=np.asarray(json.dumps(entry, ensure_ascii=True, sort_keys=True)),
            transaction_json=np.asarray(
                json.dumps(result.audit_mapping(), ensure_ascii=True, sort_keys=True)
            ),
            preview_metadata_json=np.asarray(
                json.dumps(preview.metadata, ensure_ascii=True, sort_keys=True)
            ),
        )
        if preview.reason is None:
            render_montage(
                image_before=original_crop,
                image_after=preview.image,
                segmentation_after=preview.segmentation,
                support=image_support,
                output=montage_path,
                raw_generation=np.asarray(preview.evidence["raw_generation"]),
                harmonized_generation=np.asarray(
                    preview.evidence["harmonized_generation"]
                ),
                pre_harmonization=np.asarray(preview.evidence["pre_harmonization"]),
                label_support=label_support,
                boundary_masks=tuple(
                    np.asarray(preview.evidence[f"boundary_label_{value}"], dtype=bool)
                    for value in (1, 2, 3)
                ),
                boundary_policy="label_only_qc_v1",
                qc_metadata=dict(preview.metadata),
                display_label=blind_code,
            )
        if not montage_path.is_file():
            violations.append("montage_missing")
        row = {
            "event_index": event_index,
            "smoke_id": entry["smoke_id"],
            "event_id": result.event_id,
            "event_seed": result.event_seed,
            "target_case_id": case_id,
            "target_patient_group": entry["target_patient_group"],
            "donor_component_id": entry["donor_component_id"],
            "donor_patient_group": entry["donor_patient_group"],
            "core_volume_bin": entry["core_volume_bin"],
            "blind_code": blind_code,
            "transaction_state": result.state,
            "transaction_reason": result.reason,
            "artifact_path": str(artifact_path.relative_to(output_dir)),
            "artifact_sha256": sha256_file(artifact_path),
            "montage_path": str(montage_path.relative_to(output_dir)),
            "montage_sha256": sha256_file(montage_path) if montage_path.is_file() else "",
            "violations": violations,
        }
        row["row_sha256"] = canonical_json_sha256(row, exclude=("row_sha256",))
        rows.append(row)
        review_rows.append(
            {
                "blind_code": blind_code,
                "event_id": result.event_id,
                "core_volume_bin": entry["core_volume_bin"],
                "montage_path": row["montage_path"],
                "artifact_path": row["artifact_path"],
                "evidence_sha256": row["artifact_sha256"],
                "review_decision": "pending",
                "reviewer": "",
                "reviewed_at_utc": "",
                "notes": "",
            }
        )
        private_entries.append(
            {
                "blind_code": blind_code,
                "event_id": result.event_id,
                "transaction_state": result.state,
                "transaction_reason": result.reason,
            }
        )
        state_counts[result.state] += 1
        reason_counts[result.reason or "COMMITTED"] += 1
        global_violations.extend(
            f"{entry['smoke_id']}:{violation}" for violation in violations
        )

    results_path = output_dir / "fixed_event_results.jsonl"
    results_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    review_path = output_dir / "blinded_manual_review_template.csv"
    with review_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(review_rows, key=lambda row: row["blind_code"]))
    private_path = output_dir / "PRIVATE_BLINDING_MAP.json"
    _exclusive_json(
        private_path,
        {
            "schema_version": 1,
            "stage": args.stage,
            "blind_seed": args.blind_seed,
            "entries": private_entries,
        },
        mode=0o400,
    )
    committed = int(state_counts["COMMITTED"])
    generation_pass_rate = committed / expected_count
    effective_aug_rate = config.p_select * generation_pass_rate
    rate_contract = strict_processor.calibration.payload["effective_rate_contract"]
    if generation_pass_rate < float(rate_contract["minimum_generation_pass_rate"]):
        global_violations.append("generation_pass_rate_below_frozen_minimum")
    if effective_aug_rate < float(rate_contract["minimum_effective_aug_rate"]):
        global_violations.append("effective_aug_rate_below_frozen_minimum")
    report = {
        "schema_version": 1,
        "status": "hold_for_blinded_manual_review" if not global_violations else "fail",
        "stage": args.stage,
        "calibration_partition": expected_partition,
        "attempt_count": expected_count,
        "reviewable_count": len(review_rows),
        "committed_count": committed,
        "rejected_count": expected_count - committed,
        "generation_pass_rate": generation_pass_rate,
        "effective_aug_rate": effective_aug_rate,
        "state_counts": dict(sorted(state_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "per_volume_bin": {
            value: sum(row["core_volume_bin"] == value for row in rows)
            for value in ("27_49", "50_275", "gt_275")
        },
        "violations": global_violations,
        "component_manifest_sha256": manifest.identity_sha256,
        "route_config_sha256": sha256_file(config.path),
        "strict_calibration_sha256": sha256_file(strict_calibration_path),
        "measurement_calibration_sha256": sha256_file(measurement_calibration_path),
        "valid_mask_manifest_sha256": sha256_file(args.valid_mask_manifest),
        "event_manifest_sha256": sha256_file(event_manifest_path),
        "g1_checkpoint_selection_sha256": sha256_file(args.g1_checkpoint_selection),
        "g2_parent_gate_sha256": sha256_file(args.g2_parent_gate),
        "g1_runtime_code": backend.runtime_code,
        "runner_sha256": sha256_file(Path(__file__)),
        "results_file": results_path.name,
        "results_sha256": sha256_file(results_path),
        "review_template": review_path.name,
        "review_template_sha256": sha256_file(review_path),
        "private_blinding_map_sha256": sha256_file(private_path),
        "transaction_audit_sha256": sha256_file(event_audit_path),
    }
    report["report_sha256"] = canonical_json_sha256(report, exclude=("report_sha256",))
    _exclusive_json(output_dir / "FIXED_EVENT_REPORT.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if report["status"] != "hold_for_blinded_manual_review":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
