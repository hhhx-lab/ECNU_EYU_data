#!/usr/bin/env python3
"""Run the isolated 24-case K1 RC smoke; manual review remains a separate gate."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT / "runtime/repository"
sys.path.insert(0, str(REPOSITORY))
SMOKE_SOURCE = REPOSITORY / "scripts/09_k1245/62_run_k1245_smoke.py"
spec = importlib.util.spec_from_file_location("k1245_smoke_shared", SMOKE_SOURCE)
shared = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(shared)
from custom_nnunet.met_aug_data_loader import PreprocessedValidMaskStore
from custom_nnunet.met_aug_diffusion import G1FourModalityInpaintingBackend
from custom_nnunet.met_aug_k1245 import K1245Config, K1245Engine, K1245Manifest, RCAwareFixV3CandidateProcessor
from custom_nnunet.met_aug_core import EventContext, JsonlAuditSink, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-plan", required=True, type=Path)
    parser.add_argument("--pool-root", required=True, type=Path)
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
    plan = args.smoke_plan.resolve()
    config_path = ROOT / "config/K1245_ROUTE_K1_FALLBACK.json"
    manifest_path = args.pool_root / "pools/rc27/component_manifest.json"
    output = ROOT / ("smoke/k1_rc24_" + plan.stem.rsplit("_", 1)[-1].lower())
    manifest = K1245Manifest.load(manifest_path)
    fallback_plan = ROOT / "K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json"
    config = K1245Config.load(config_path, manifest=manifest, plan_path=fallback_plan)
    if config.route_id != "K1" or not config.use_f:
        raise SystemExit("K1 fallback config contract mismatch")
    plan_payload = json.loads(plan.read_text())
    processor = RCAwareFixV3CandidateProcessor.load(
        args.calibration,
        expected_sha256=plan_payload["parent_contract"]["fix_v3_calibration_file_sha256"],
        expected_policy="label_only_qc_v1",
    )
    legacy_payload = json.loads(args.legacy_manifest.read_text())
    legacy_identity = legacy_payload["manifest_sha256"]
    processor.bind_extended_manifest(
        legacy_manifest_sha256=legacy_identity,
        extended_manifest_sha256=manifest.identity_sha256,
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
    dataset = infer_dataset_class(str(args.preprocessed_dir))(str(args.preprocessed_dir), sorted(masks.records))
    output.mkdir(parents=True, exist_ok=False)
    (output / "montages").mkdir()
    engine = K1245Engine(
        manifest=manifest,
        config=config,
        backend=backend,
        audit_sink=JsonlAuditSink(output / "events.jsonl"),
        candidate_processor=processor,
    )
    engine.config = __import__("dataclasses").replace(
        config, use_d=False, use_b=True, use_e=False, use_f=False, use_x2=False,
        x1_rc_probability=1.0, second_tumour_probability=0.0,
    )
    records = shared._selected_records(manifest, "rc24")
    cases = sorted(manifest.target_groups)
    rows = []
    attempts = output / "candidate_attempts.jsonl"
    for index, record in enumerate(records):
        candidates = [case for case in cases if manifest.target_groups[case] != record.patient_group]
        committed = None
        for candidate_index in range(min(32, len(candidates))):
            case_id = candidates[(index * 17 + candidate_index) % len(candidates)]
            image, segmentation = shared._load_case(dataset, case_id)
            valid, _ = masks.load(case_id)
            engine.sampler.choose = lambda rng, target_group, request_rc, fixed=record: (fixed, False)
            engine._force_second_tumour = True
            try:
                image_after, seg_after, event = engine.apply(
                    image=image,
                    segmentation=segmentation,
                    valid_mask=valid,
                    context=EventContext(
                        epoch=0, rank=0, worker=0, case_id=case_id,
                        patch_index=index * 100 + candidate_index,
                        full_shape=tuple(int(value) for value in valid.shape),
                    ),
                )
            finally:
                engine._force_second_tumour = False
            attempt = event.audit_mapping()
            attempt.update({"smoke_kind": "k1_rc24", "donor_index": index + 1,
                            "candidate_index": candidate_index, "candidate_case_id": case_id})
            with attempts.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(attempt, ensure_ascii=True, sort_keys=True) + "\n")
            if event.state == "COMMITTED":
                committed = (case_id, image, image_after, seg_after, event)
                break
        if committed is None:
            raise RuntimeError(f"K1 RC donor did not commit: {record.component_id}")
        case_id, image, image_after, seg_after, event = committed
        placement = event.placement
        slices = tuple(slice(start, start + 64) for start in placement.crop_start)
        montage = output / "montages" / f"k1_rc_{index + 1:02d}.png"
        shared._render(image[(slice(None),) + slices], image_after[(slice(None),) + slices],
                       seg_after[(0,) + slices], montage)
        rows.append({"smoke_id": f"k1_rc_{index + 1:02d}", "target_case_id": case_id,
                     "component_id": record.component_id, "classes_present": list(record.classes_present),
                     "event": event.audit_mapping(), "montage": montage.relative_to(output).as_posix(),
                     "montage_sha256": sha256_file(montage), "automatic_status": "pass"})
    cases_path = output / "automatic_cases.jsonl"
    cases_path.write_text("".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows))
    (output / "manual_review.csv").write_text(
        "smoke_id,acceptable,reviewer,reviewed_at_utc,notes\n" +
        "".join(f"{row['smoke_id']},,,,\n" for row in rows), encoding="utf-8")
    report = {
        "artifact_status": "experimental_unvalidated", "operator_approved": False,
        "formal_gate_status": "not_run_not_passed", "status": "hold_for_manual_review",
        "automatic_status": "pass", "automatic_pass_count": len(rows),
        "manual_review_required": 24, "route_id": "K1",
        "F_STRICT_PREREG_GATE": False, "fallback_gate": "F_FALLBACK_ENGINEERING_ELIGIBILITY_GATE",
        "plan_sha256": sha256_file(plan), "fallback_plan_sha256": sha256_file(ROOT / "K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json"),
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": manifest.identity_sha256, "cases_sha256": sha256_file(cases_path),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    fd = os.open(output / "AUTOMATIC_VALIDATION.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
