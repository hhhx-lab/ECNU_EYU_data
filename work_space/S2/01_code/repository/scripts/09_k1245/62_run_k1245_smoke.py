#!/usr/bin/env python3
"""Run fixed RC24 and X3-small24 four-modality smoke sets for K1245."""

from __future__ import annotations

import argparse
from dataclasses import replace
import datetime as dt
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import EventContext, JsonlAuditSink, sha256_file  # noqa: E402
from custom_nnunet.met_aug_data_loader import PreprocessedValidMaskStore  # noqa: E402
from custom_nnunet.met_aug_diffusion import G1FourModalityInpaintingBackend  # noqa: E402
from custom_nnunet.met_aug_k1245 import (  # noqa: E402
    K1245Config,
    K1245Engine,
    K1245Manifest,
    RCAwareFixV3CandidateProcessor,
)


def _exclusive_json(path: Path, payload: dict) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def _load_case(dataset, case_id: str) -> tuple[np.ndarray, np.ndarray]:
    data, segmentation, previous, _properties = dataset.load_case(case_id)
    if previous is not None:
        raise RuntimeError("K1245 smoke does not support cascaded data")
    image = np.asarray(data, dtype=np.float32).copy()
    seg = np.asarray(segmentation, dtype=np.int16).copy()
    if image.ndim != 4 or image.shape[0] != 4 or seg.shape != (1,) + image.shape[1:]:
        raise RuntimeError(f"invalid smoke source shape: {case_id}/{image.shape}/{seg.shape}")
    return image, seg


def _render(before: np.ndarray, after: np.ndarray, label: np.ndarray, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    points = np.argwhere(label != 0)
    focus = tuple(int(round(value)) for value in points.mean(axis=0))
    figure, axes = plt.subplots(4, 6, figsize=(18, 12), constrained_layout=True)
    names = ("t1n", "t1c", "t2w", "t2f")
    for channel, name in enumerate(names):
        values = before[channel][np.isfinite(before[channel])]
        low, high = np.quantile(values, (0.01, 0.99))
        planes_before = (before[channel][focus[0]], before[channel][:, focus[1]], before[channel][:, :, focus[2]])
        planes_after = (after[channel][focus[0]], after[channel][:, focus[1]], after[channel][:, :, focus[2]])
        label_planes = (label[focus[0]], label[:, focus[1]], label[:, :, focus[2]])
        for plane in range(3):
            for offset, volume in enumerate((planes_before[plane], planes_after[plane])):
                axis = axes[channel, plane * 2 + offset]
                axis.imshow(volume.T, cmap="gray", origin="lower", vmin=low, vmax=high)
                axis.contour(label_planes[plane].T, levels=[0.5], colors="red", linewidths=0.7)
                axis.set_axis_off()
                if channel == 0:
                    axis.set_title(f"{('axial','coronal','sagittal')[plane]} {'before' if offset == 0 else 'after'}")
                if plane == 0 and offset == 0:
                    axis.set_ylabel(name)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _selected_records(manifest: K1245Manifest, kind: str) -> list[Any]:
    if kind == "rc24":
        records = [record for record in manifest.records if 4 in record.classes_present]
        records.sort(key=lambda record: (record.classes_present, record.core_volume_mm3, record.component_id))
        if len(records) < 24:
            raise RuntimeError(f"RC smoke needs 24 donors, observed {len(records)}")
        positions = np.linspace(0, len(records) - 1, 24, dtype=int)
        return [records[int(index)] for index in positions]
    bins = ((8, 12), (13, 19), (20, 26))
    selected = []
    for lower, upper in bins:
        values = [record for record in manifest.records if lower <= record.core_volume_mm3 <= upper]
        values.sort(key=lambda record: (record.core_volume_mm3, record.component_id))
        if len(values) < 8:
            raise RuntimeError(f"X3 smoke bin {lower}-{upper} has only {len(values)} donors")
        positions = np.linspace(0, len(values) - 1, 8, dtype=int)
        selected.extend(values[int(index)] for index in positions)
    return selected


def _run_set(
    *,
    kind: str,
    manifest: K1245Manifest,
    config: K1245Config,
    backend: Any,
    processor: RCAwareFixV3CandidateProcessor,
    dataset: Any,
    masks: PreprocessedValidMaskStore,
    output: Path,
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    (output / "montages").mkdir()
    engine = K1245Engine(
        manifest=manifest,
        config=config,
        backend=backend,
        audit_sink=JsonlAuditSink(output / "events.jsonl"),
        candidate_processor=processor,
    )
    # Smoke forces selection and the frozen donor, but retains all geometry,
    # generation, QC, and commit logic used by training.
    engine.config = replace(
        config,
        use_d=False,
        use_b=(kind == "rc24"),
        use_e=False,
        use_f=False,
        use_x2=False,
        x1_rc_probability=1.0 if kind == "rc24" else 0.0,
        second_tumour_probability=0.0,
    )
    records = _selected_records(manifest, kind)
    cases = sorted(manifest.target_groups)
    rows = []
    attempts_path = output / "candidate_attempts.jsonl"
    for index, record in enumerate(records):
        candidates = [case for case in cases if manifest.target_groups[case] != record.patient_group]
        committed = None
        for candidate_index in range(min(32, len(candidates))):
            case_id = candidates[(index * 17 + candidate_index) % len(candidates)]
            image, segmentation = _load_case(dataset, case_id)
            valid, _foreground = masks.load(case_id)
            engine.sampler.choose = lambda rng, target_group, request_rc, fixed=record: (fixed, False)
            engine._force_second_tumour = True
            try:
                image_after, seg_after, result = engine.apply(
                    image=image,
                    segmentation=segmentation,
                    valid_mask=valid,
                    context=EventContext(
                        epoch=0,
                        rank=0,
                        worker=0,
                        case_id=case_id,
                        patch_index=index * 100 + candidate_index,
                        full_shape=tuple(int(value) for value in valid.shape),
                    ),
                )
            finally:
                engine._force_second_tumour = False
            attempt = result.audit_mapping()
            attempt.update({
                "smoke_kind": kind,
                "donor_index": index + 1,
                "candidate_index": candidate_index,
                "candidate_case_id": case_id,
            })
            with attempts_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(attempt, ensure_ascii=True, sort_keys=True) + "\n")
            if result.state == "COMMITTED":
                committed = (case_id, image, image_after, seg_after, result)
                break
        if committed is None:
            raise RuntimeError(
                f"{kind} donor did not produce a committed smoke event: "
                f"{record.component_id}; see {attempts_path.name}"
            )
        case_id, image, image_after, seg_after, result = committed
        placement = result.placement
        assert placement is not None
        slices = tuple(slice(start, start + 64) for start in placement.crop_start)
        before_crop = image[(slice(None),) + slices]
        after_crop = image_after[(slice(None),) + slices]
        label_crop = seg_after[(0,) + slices]
        montage = output / "montages" / f"{kind}_{index + 1:02d}.png"
        _render(before_crop, after_crop, label_crop, montage)
        rows.append({
            "smoke_id": f"{kind}_{index + 1:02d}",
            "target_case_id": case_id,
            "component_id": record.component_id,
            "classes_present": list(record.classes_present),
            "anchor_volume_mm3": record.core_volume_mm3,
            "event": result.audit_mapping(),
            "montage": str(montage.relative_to(output)),
            "montage_sha256": sha256_file(montage),
            "automatic_status": "pass",
        })
    rows_path = output / "automatic_cases.jsonl"
    rows_path.write_text("".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    template = output / "manual_review.csv"
    template.write_text(
        "smoke_id,acceptable,reviewer,reviewed_at_utc,notes\n" +
        "".join(f"{row['smoke_id']},,,,\n" for row in rows), encoding="utf-8"
    )
    report = {
        "artifact_status": "experimental_unvalidated",
        "operator_approved": False,
        "formal_gate_status": "not_run_not_passed",
        "kind": kind,
        "status": "hold_for_manual_review",
        "automatic_status": "pass",
        "automatic_pass_count": len(rows),
        "required_manual_acceptance_count": 24,
        "manifest_sha256": manifest.identity_sha256,
        "config_sha256": sha256_file(config.path),
        "cases_sha256": sha256_file(rows_path),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    _exclusive_json(output / "AUTOMATIC_VALIDATION.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--preprocessed-dir", required=True, type=Path)
    parser.add_argument("--valid-mask-manifest", required=True, type=Path)
    parser.add_argument("--legacy-manifest", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--g1-code-dir", required=True, type=Path)
    parser.add_argument("--g1-checkpoint-root", required=True, type=Path)
    parser.add_argument("--g1-checkpoint-selection", required=True, type=Path)
    parser.add_argument("--g2-parent-gate", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = args.root.resolve()
    plan = root / "K1245_RC_AWARE_EXECUTION_PLAN_20260921_R1.json"
    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    legacy_identity = plan_payload["parent_contract"]["legacy_component_manifest_identity_sha256"]
    processor = RCAwareFixV3CandidateProcessor.load(
        args.calibration,
        expected_sha256=plan_payload["parent_contract"]["fix_v3_calibration_file_sha256"],
        expected_policy="label_only_qc_v1",
    )
    backend = G1FourModalityInpaintingBackend(
        g1_code_dir=args.g1_code_dir,
        checkpoint_root=args.g1_checkpoint_root,
        checkpoint_selection=args.g1_checkpoint_selection,
        qc_gate=args.g2_parent_gate,
        device=args.device,
    )
    masks = PreprocessedValidMaskStore(args.valid_mask_manifest)
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    dataset_class = infer_dataset_class(str(args.preprocessed_dir))
    dataset = dataset_class(str(args.preprocessed_dir), sorted(masks.records))
    reports = {}
    for kind, pool, route in (("rc24", "rc27", "K2"), ("small24", "small8", "K4")):
        manifest = K1245Manifest.load(root / "pools" / pool / "component_manifest.json")
        config = K1245Config.load(root / "config" / f"K1245_ROUTE_{route}.json", manifest=manifest, plan_path=plan)
        processor.bind_extended_manifest(
            legacy_manifest_sha256=legacy_identity,
            extended_manifest_sha256=manifest.identity_sha256,
        )
        reports[kind] = _run_set(
            kind=kind,
            manifest=manifest,
            config=config,
            backend=backend,
            processor=processor,
            dataset=dataset,
            masks=masks,
            output=root / "smoke" / kind,
        )
        processor.component_manifest_sha256 = legacy_identity
    _exclusive_json(root / "evidence/K1245_SMOKE_AUTOMATIC.json", {
        "artifact_status": "experimental_unvalidated",
        "operator_approved": False,
        "formal_gate_status": "not_run_not_passed",
        "status": "hold_for_manual_review",
        "plan_sha256": sha256_file(plan),
        "reports": reports,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    })


if __name__ == "__main__":
    main()
