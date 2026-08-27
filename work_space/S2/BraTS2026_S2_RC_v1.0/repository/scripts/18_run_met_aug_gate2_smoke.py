#!/usr/bin/env python3
"""Execute only the pre-registered Route A Gate 2 four-modality smoke set.

This command never trains nnU-Net and never writes Dataset264.  It loads each
preprocessed training case read-only, runs the exact immutable transaction from
the smoke manifest, emits cropped evidence, and stops in manual-review hold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import (
    ALLOWED_LABELS,
    ComponentManifest,
    EventContext,
    JsonlAuditSink,
    MetAugEngine,
    RouteConfig,
    canonical_json_sha256,
    sha256_file,
)
from custom_nnunet.met_aug_diffusion import G1FourModalityInpaintingBackend
from custom_nnunet.met_aug_fix_v2 import FixV2CandidateProcessor
from custom_nnunet.met_aug_gate2 import (
    GATE2_AUTOMATIC_REPORT_SCHEMA,
    GATE2_VOLUME_BINS,
    gate2_runtime_code_snapshot,
    load_smoke_manifest,
    load_valid_mask_assets,
    write_review_template,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-manifest", required=True)
    parser.add_argument("--route-config", required=True)
    parser.add_argument("--valid-mask-manifest", required=True)
    parser.add_argument("--smoke-manifest", required=True)
    parser.add_argument("--preprocessed-dir", required=True)
    parser.add_argument("--g1-code-dir", required=True)
    parser.add_argument("--g1-checkpoint-root", required=True)
    parser.add_argument("--g1-checkpoint-selection", required=True)
    parser.add_argument("--g2-parent-gate", required=True)
    parser.add_argument("--fix-v2-calibration")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_preprocessed_case(dataset, case_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Read one case through the pinned nnU-Net 2.8 four-value API."""
    data, segmentation, previous_stage_segmentation, _properties = dataset.load_case(case_id)
    if previous_stage_segmentation is not None:
        raise ValueError(f"{case_id}: MET-AUG Gate 2 does not support cascaded data")
    image = np.asarray(data, dtype=np.float32).copy()
    seg = np.asarray(segmentation, dtype=np.int16).copy()
    if image.ndim != 4 or image.shape[0] != 4:
        raise ValueError(f"{case_id}: expected preprocessed four-channel image, got {image.shape}")
    if seg.ndim != 4 or seg.shape[0] != 1 or seg.shape[1:] != image.shape[1:]:
        raise ValueError(f"{case_id}: expected aligned one-channel segmentation, got {seg.shape}")
    return image, seg


def context_from_entry(entry: dict[str, Any]) -> EventContext:
    event = entry["event"]
    return EventContext(
        epoch=int(event["epoch"]),
        rank=int(event["rank"]),
        worker=int(event["worker"]),
        case_id=str(event["case_id"]),
        patch_index=int(event["patch_index"]),
        patch_origin=tuple(int(value) for value in event["patch_origin"]),
        full_shape=tuple(int(value) for value in event["full_shape"]),
    )


def support_in_full_shape(
    shape: tuple[int, int, int], placement, local_support: np.ndarray
) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    start = placement.crop_start
    stop = tuple(value + 64 for value in start)
    result[tuple(slice(begin, end) for begin, end in zip(start, stop))] = local_support
    return result


def event_violations(
    *,
    entry: dict[str, Any],
    result,
    image_before: np.ndarray,
    seg_before: np.ndarray,
    image_after: np.ndarray,
    seg_after: np.ndarray,
    valid_mask: np.ndarray,
) -> list[str]:
    violations: list[str] = []
    if result.state != "COMMITTED":
        violations.append(f"transaction_not_committed:{result.state}:{result.reason}")
        return violations
    if result.event_id != entry["event_id"] or result.event_seed != int(entry["event_seed"]):
        violations.append("event_identity_drift")
    if result.record is None or result.record.component_id != entry["donor_component_id"]:
        violations.append("donor_component_drift")
    if result.placement is None:
        violations.append("missing_committed_placement")
        return violations
    planned = entry["planned"]
    if list(result.placement.crop_start) != planned["crop_start"]:
        violations.append("crop_start_drift")
    if int(np.count_nonzero(result.placement.support)) != int(planned["support_voxels"]):
        violations.append("support_size_drift")
    if image_after.shape != image_before.shape or seg_after.shape != seg_before.shape:
        violations.append("output_shape_drift")
    if image_after.dtype != image_before.dtype or seg_after.dtype != seg_before.dtype:
        violations.append("output_dtype_drift")
    if not np.all(np.isfinite(image_after)):
        violations.append("non_finite_modality_output")
    labels = set(int(value) for value in np.unique(seg_after))
    if not labels.issubset(ALLOWED_LABELS | {-1}):
        violations.append("illegal_output_label")

    local_image_support = np.asarray(
        result.evidence.get("image_support", result.placement.support), dtype=bool
    )
    if local_image_support.shape != result.placement.support.shape:
        violations.append("image_support_shape_drift")
        return violations
    image_support = support_in_full_shape(
        tuple(int(value) for value in seg_before.shape[1:]),
        result.placement,
        local_image_support,
    )
    label_support = support_in_full_shape(
        tuple(int(value) for value in seg_before.shape[1:]),
        result.placement,
        result.placement.support,
    )
    changed_image = np.any(image_after != image_before, axis=0)
    changed_segmentation = seg_after[0] != seg_before[0]
    if np.any(changed_image & ~image_support):
        violations.append("image_changed_outside_support")
    if np.any(changed_segmentation & ~label_support):
        violations.append("segmentation_changed_outside_support")
    if not np.all(valid_mask[image_support]):
        violations.append("support_outside_valid_mask")
    if np.any(seg_before[0][image_support] != 0):
        violations.append("support_overlaps_existing_or_padding_label")

    start = result.placement.crop_start
    stop = tuple(value + 64 for value in start)
    slices = tuple(slice(begin, end) for begin, end in zip(start, stop))
    expected = result.placement.label_cube[result.placement.support]
    actual = seg_after[(0,) + slices][result.placement.support]
    if not np.array_equal(expected, actual):
        violations.append("committed_segmentation_does_not_match_planned_label")
    return violations


def crop_evidence(
    image: np.ndarray,
    segmentation: np.ndarray,
    placement,
) -> tuple[np.ndarray, np.ndarray]:
    start = placement.crop_start
    stop = tuple(value + 64 for value in start)
    slices = tuple(slice(begin, end) for begin, end in zip(start, stop))
    return image[(slice(None),) + slices].copy(), segmentation[(slice(None),) + slices].copy()


def render_montage(
    *,
    image_before: np.ndarray,
    image_after: np.ndarray,
    segmentation_after: np.ndarray,
    support: np.ndarray,
    output: Path,
    raw_generation: np.ndarray | None = None,
    harmonized_generation: np.ndarray | None = None,
    pre_harmonization: np.ndarray | None = None,
    label_support: np.ndarray | None = None,
    boundary_masks: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    boundary_policy: str | None = None,
    qc_metadata: dict[str, Any] | None = None,
    display_label: str | None = None,
) -> None:
    """Render all four modalities in three planes for independent manual review."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    points = np.argwhere(support)
    if points.size == 0:
        raise ValueError("cannot render Gate 2 evidence without a committed support")
    focus = tuple(int(round(value)) for value in points.mean(axis=0))
    labels = ("t1n", "t1c", "t2w", "t2f")

    def planes(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return volume[focus[0], :, :], volume[:, focus[1], :], volume[:, :, focus[2]]

    support_planes = planes(support.astype(np.uint8))
    label_support = support if label_support is None else label_support
    label_planes = planes(label_support.astype(np.uint8))
    fix_v2 = raw_generation is not None
    state_count = 6 if fix_v2 else 2
    if fix_v2:
        figure, axes = plt.subplots(
            12,
            state_count,
            figsize=(18, 30),
            constrained_layout=True,
            squeeze=False,
        )
    else:
        figure, axes = plt.subplots(
            4,
            3 * state_count,
            figsize=(3.2 * 3 * state_count, 12),
            constrained_layout=True,
            squeeze=False,
        )
    boundary_plane_sets = []
    if boundary_masks is not None:
        boundary_plane_sets = [planes(mask.astype(np.uint8)) for mask in boundary_masks]
    boundary_colors = ("#E69F00", "#CC79A7", "#009E73")
    for channel, name in enumerate(labels):
        before_planes = planes(image_before[channel])
        after_planes = planes(image_after[channel])
        if fix_v2:
            assert raw_generation is not None
            assert harmonized_generation is not None
            assert pre_harmonization is not None
            raw_planes = planes(raw_generation[channel])
            harmonized_planes = planes(harmonized_generation[channel])
            pre_planes = planes(pre_harmonization[channel])
            difference_planes = planes(image_after[channel] - image_before[channel])
        brain_values = image_before[channel][np.isfinite(image_before[channel])]
        low, high = np.percentile(brain_values, (1, 99))
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            low, high = float(brain_values.min()), float(brain_values.max())
            if low >= high:
                high = low + 1.0
        difference_values = np.abs(image_after[channel] - image_before[channel])[support]
        difference_limit = float(np.quantile(difference_values, 0.99))
        difference_limit = max(difference_limit, np.finfo(np.float32).eps)
        for plane_index, (before_plane, after_plane) in enumerate(
            zip(before_planes, after_planes)
        ):
            if fix_v2:
                states = (
                    (before_plane, "original", "gray", low, high),
                    (raw_planes[plane_index], "raw", "gray", low, high),
                    (
                        harmonized_planes[plane_index],
                        "harmonized",
                        "gray",
                        low,
                        high,
                    ),
                    (
                        pre_planes[plane_index],
                        "unharmonized blend",
                        "gray",
                        low,
                        high,
                    ),
                    (after_plane, "final", "gray", low, high),
                    (
                        difference_planes[plane_index],
                        "difference",
                        "coolwarm",
                        -difference_limit,
                        difference_limit,
                    ),
                )
            else:
                states = (
                    (before_plane, "before", "gray", low, high),
                    (after_plane, "after", "gray", low, high),
                )
            for offset, (image, title, cmap, state_low, state_high) in enumerate(states):
                if fix_v2:
                    axis = axes[channel * 3 + plane_index, offset]
                else:
                    axis = axes[channel, plane_index * state_count + offset]
                axis.imshow(
                    image.T,
                    cmap=cmap,
                    origin="lower",
                    vmin=state_low,
                    vmax=state_high,
                )
                axis.contour(support_planes[plane_index].T, levels=[0.5], colors="red", linewidths=0.7)
                axis.contour(label_planes[plane_index].T, levels=[0.5], colors="cyan", linewidths=0.7)
                for boundary_index, boundary_planes in enumerate(boundary_plane_sets):
                    if np.any(boundary_planes[plane_index]):
                        axis.contour(
                            boundary_planes[plane_index].T,
                            levels=[0.5],
                            colors=boundary_colors[boundary_index],
                            linewidths=0.6,
                        )
                axis.set_axis_off()
                if fix_v2 and channel == 0 and plane_index == 0:
                    axis.set_title(title)
                elif not fix_v2 and channel == 0:
                    axis.set_title(
                        f"{('axial', 'coronal', 'sagittal')[plane_index]} {title}"
                    )
                if fix_v2 and offset == 0:
                    axis.text(
                        -0.10,
                        0.5,
                        f"{name}\n{('axial', 'coronal', 'sagittal')[plane_index]}",
                        transform=axis.transAxes,
                        ha="right",
                        va="center",
                        fontsize=9,
                        clip_on=False,
                    )
                elif not fix_v2 and plane_index == 0 and offset == 0:
                    axis.set_ylabel(name)
    if fix_v2:
        summary = [
            f"candidate={display_label}"
            if display_label is not None
            else f"policy={boundary_policy}"
        ]
        if qc_metadata:
            geometry = qc_metadata.get("geometry", {})
            boundary = (
                qc_metadata.get("candidate_qc", {}).get("boundary", {})
            )
            summary.extend(
                (
                    f"L={geometry.get('label_support_voxels', 'NA')}",
                    f"H={geometry.get('image_support_voxels', 'NA')}",
                    f"boundary_max={boundary.get('event_max_ratio', 'NA')}",
                )
            )
        figure.suptitle(" | ".join(summary), fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Gate 2 output is immutable and already exists: {output_dir}")
    preprocessed_dir = Path(args.preprocessed_dir).expanduser().resolve()
    if not preprocessed_dir.is_dir():
        raise FileNotFoundError(f"missing nnU-Net preprocessed directory: {preprocessed_dir}")

    manifest = ComponentManifest.load(args.component_manifest)
    config = RouteConfig.load(args.route_config, manifest)
    candidate_processor = None
    calibration_path = None
    if config.fix_v2 is not None:
        if not args.fix_v2_calibration:
            raise ValueError("schema-4 Gate 2 requires --fix-v2-calibration")
        calibration_path = Path(args.fix_v2_calibration).expanduser().resolve()
        candidate_processor = FixV2CandidateProcessor.load(
            calibration_path,
            expected_sha256=config.fix_v2.calibration_sha256,
            expected_policy=config.fix_v2.boundary_policy,
        )
    elif args.fix_v2_calibration:
        raise ValueError("legacy Gate 2 must not receive --fix-v2-calibration")
    assets = load_valid_mask_assets(args.valid_mask_manifest, expected_ids=set(manifest.target_groups))
    smoke_manifest = load_smoke_manifest(
        args.smoke_manifest,
        manifest=manifest,
        config=config,
        valid_mask_manifest_path=args.valid_mask_manifest,
    )
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

    smoke_case_ids = [str(entry["target_case_id"]) for entry in smoke_manifest["smoke_cases"]]
    dataset_class = infer_dataset_class(str(preprocessed_dir))
    preprocessed_dataset = dataset_class(str(preprocessed_dir), smoke_case_ids)
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts_dir = output_dir / "artifacts"
    montages_dir = output_dir / "montages"
    artifacts_dir.mkdir()
    montages_dir.mkdir()
    audit_path = output_dir / "gate2_events.jsonl"
    backend = G1FourModalityInpaintingBackend(
        g1_code_dir=args.g1_code_dir,
        checkpoint_root=args.g1_checkpoint_root,
        checkpoint_selection=args.g1_checkpoint_selection,
        qc_gate=args.g2_parent_gate,
        device=args.device,
    )
    engine = MetAugEngine(
        manifest=manifest,
        config=config,
        backend=backend,
        audit_sink=JsonlAuditSink(audit_path),
        candidate_processor=candidate_processor,
    )
    case_results: list[dict[str, Any]] = []
    global_violations: list[str] = []
    for entry in smoke_manifest["smoke_cases"]:
        case_id = str(entry["target_case_id"])
        context = context_from_entry(entry)
        image_before, segmentation_before = load_preprocessed_case(preprocessed_dataset, case_id)
        valid_mask, _foreground = assets[case_id]
        if tuple(image_before.shape[1:]) != context.full_shape or valid_mask.shape != context.full_shape:
            raise RuntimeError(f"{case_id}: Gate 2 smoke geometry differs from frozen manifest")
        image_after, segmentation_after, result = engine.apply(
            image=image_before,
            segmentation=segmentation_before,
            valid_mask=valid_mask,
            context=context,
        )
        violations = event_violations(
            entry=entry,
            result=result,
            image_before=image_before,
            seg_before=segmentation_before,
            image_after=image_after,
            seg_after=segmentation_after,
            valid_mask=valid_mask,
        )
        placement = result.placement
        artifact_path = artifacts_dir / f"{entry['smoke_id']}.npz"
        montage_path = montages_dir / f"{entry['smoke_id']}.png"
        label_support = np.zeros((64, 64, 64), dtype=bool)
        image_support = np.zeros((64, 64, 64), dtype=bool)
        if placement is not None:
            before_crop, seg_before_crop = crop_evidence(image_before, segmentation_before, placement)
            after_crop, seg_after_crop = crop_evidence(image_after, segmentation_after, placement)
            label_support = placement.support.astype(bool, copy=False)
            image_support = np.asarray(
                result.evidence.get("image_support", label_support), dtype=bool
            )
        else:
            before_crop = after_crop = np.zeros((4, 64, 64, 64), dtype=np.float32)
            seg_before_crop = seg_after_crop = np.zeros((1, 64, 64, 64), dtype=np.int16)
            violations.append("missing_placement_evidence")
        fix_v2_evidence: dict[str, np.ndarray] = {}
        if config.fix_v2 is not None:
            required_evidence = {
                "raw_generation",
                "pre_harmonization",
                "harmonized_generation",
                "candidate",
                "alpha",
                "image_support",
                "label_support",
                "reference_ring",
                "harmonization_ring",
                "boundary_label_1",
                "boundary_label_2",
                "boundary_label_3",
            }
            missing_evidence = sorted(required_evidence - set(result.evidence))
            if missing_evidence:
                violations.append(f"fix_v2_evidence_missing:{','.join(missing_evidence)}")
            else:
                fix_v2_evidence = {
                    key: np.asarray(result.evidence[key]) for key in sorted(required_evidence)
                }
        artifact_payload = {
            "image_before": before_crop,
            "image_after": after_crop,
            "segmentation_before": seg_before_crop,
            "segmentation_after": seg_after_crop,
            "support": label_support.astype(np.uint8),
            "label_support": label_support.astype(np.uint8),
            "image_support": image_support.astype(np.uint8),
            **fix_v2_evidence,
            "smoke_entry_json": np.asarray(
                json.dumps(entry, ensure_ascii=True, sort_keys=True)
            ),
            "event_json": np.asarray(
                json.dumps(result.audit_mapping(), ensure_ascii=True, sort_keys=True)
            ),
        }
        np.savez_compressed(
            artifact_path,
            **artifact_payload,
        )
        if placement is not None:
            try:
                render_montage(
                    image_before=before_crop,
                    image_after=after_crop,
                    segmentation_after=seg_after_crop[0],
                    support=image_support,
                    output=montage_path,
                    raw_generation=fix_v2_evidence.get("raw_generation"),
                    harmonized_generation=fix_v2_evidence.get(
                        "harmonized_generation"
                    ),
                    pre_harmonization=fix_v2_evidence.get("pre_harmonization"),
                    label_support=label_support,
                    boundary_masks=(
                        fix_v2_evidence["boundary_label_1"],
                        fix_v2_evidence["boundary_label_2"],
                        fix_v2_evidence["boundary_label_3"],
                    )
                    if fix_v2_evidence
                    else None,
                    boundary_policy=(
                        config.fix_v2.boundary_policy if config.fix_v2 is not None else None
                    ),
                    qc_metadata=(
                        dict(result.metadata.get("fix_v2", {}))
                        if config.fix_v2 is not None
                        else None
                    ),
                )
            except Exception as exc:  # A missing renderer blocks manual approval, never the transaction audit.
                violations.append(f"montage_render_failed:{type(exc).__name__}")
        if not montage_path.is_file():
            violations.append("montage_missing")
        case_result: dict[str, Any] = {
            "smoke_id": entry["smoke_id"],
            "entry_sha256": entry["entry_sha256"],
            "event_id": result.event_id,
            "event_seed": result.event_seed,
            "target_case_id": case_id,
            "donor_component_id": entry["donor_component_id"],
            "core_volume_bin": entry["core_volume_bin"],
            "core_volume_mm3": entry["core_volume_mm3"],
            "transaction_state": result.state,
            "transaction_reason": result.reason,
            "artifact_path": str(artifact_path.relative_to(output_dir)),
            "artifact_sha256": sha256_file(artifact_path),
            "montage_path": str(montage_path.relative_to(output_dir)) if montage_path.is_file() else "",
            "montage_sha256": sha256_file(montage_path) if montage_path.is_file() else "",
            "automatic_qc_status": "pass" if not violations else "fail",
            "violations": violations,
        }
        case_result["evidence_fingerprint"] = canonical_json_sha256(case_result)
        case_results.append(case_result)
        for violation in violations:
            global_violations.append(f"{entry['smoke_id']}:{violation}")

    case_results_path = output_dir / "automatic_case_results.jsonl"
    case_results_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in case_results),
        encoding="utf-8",
    )
    review_template_path = output_dir / "manual_review_template.csv"
    write_review_template(review_template_path, case_results)
    per_volume_bin = {
        volume_bin: sum(1 for row in case_results if row["core_volume_bin"] == volume_bin)
        for volume_bin in GATE2_VOLUME_BINS
    }
    report: dict[str, Any] = {
        "schema_version": GATE2_AUTOMATIC_REPORT_SCHEMA,
        "route_id": config.route_id,
        "status": "hold_for_manual_review" if not global_violations else "fail",
        "automatic_status": "pass" if not global_violations else "fail",
        "manual_review_status": "hold_for_manual_review" if not global_violations else "not_started",
        "component_manifest_sha256": manifest.identity_sha256,
        "route_config_sha256": sha256_file(config.path),
        "valid_mask_manifest_sha256": sha256_file(args.valid_mask_manifest),
        "smoke_manifest_sha256": smoke_manifest["smoke_manifest_sha256"],
        "g1_checkpoint_selection_sha256": sha256_file(args.g1_checkpoint_selection),
        "g2_parent_gate_sha256": sha256_file(args.g2_parent_gate),
        "g1_runtime_code": backend.runtime_code,
        "runtime_code": gate2_runtime_code_snapshot(REPOSITORY_ROOT),
        "smoke_count": len(case_results),
        "per_volume_bin": per_volume_bin,
        "case_results_file": case_results_path.name,
        "case_results_sha256": sha256_file(case_results_path),
        "manual_review_template": review_template_path.name,
        "manual_review_template_sha256": sha256_file(review_template_path),
        "event_audit_file": audit_path.name,
        "event_audit_sha256": sha256_file(audit_path),
        "violations": global_violations,
    }
    if config.fix_v2 is not None:
        assert calibration_path is not None
        report["fix_v2"] = {
            "boundary_policy": config.fix_v2.boundary_policy,
            "calibration_sha256": sha256_file(calibration_path),
        }
    report["automatic_report_sha256"] = canonical_json_sha256(report, exclude=("automatic_report_sha256",))
    report_path = output_dir / "gate2_automatic_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "automatic_status": report["automatic_status"],
        "manual_review_status": report["manual_review_status"],
        "report": str(report_path),
        "smoke_count": report["smoke_count"],
        "per_volume_bin": report["per_volume_bin"],
        "violations": report["violations"],
    }, ensure_ascii=True, indent=2))
    if report["automatic_status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
