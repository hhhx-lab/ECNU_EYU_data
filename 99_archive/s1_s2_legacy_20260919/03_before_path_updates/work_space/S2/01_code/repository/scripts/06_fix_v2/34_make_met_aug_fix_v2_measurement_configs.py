#!/usr/bin/env python3
"""Create non-gating A/B/C measurement calibrations from real Reference data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import (  # noqa: E402
    ComponentManifest,
    canonical_json_sha256,
    sha256_file,
)
from custom_nnunet.met_aug_fix_v2 import FixV2Calibration  # noqa: E402
from custom_nnunet.met_aug_fix_v2_calibration import (  # noqa: E402
    build_measurement_calibration,
    build_measurement_route_config,
)


CANDIDATES = (
    ("A_label_only", "label_only_qc_v1", 0.0),
    ("B_halo_1p5mm", "halo_cosine_v1", 1.5),
    ("B_halo_2mm", "halo_cosine_v1", 2.0),
    ("B_halo_3mm", "halo_cosine_v1", 3.0),
    ("B_halo_4mm", "halo_cosine_v1", 4.0),
    ("C_halo_harmonized_1p5mm", "halo_cosine_harmonized_v1", 1.5),
    ("C_halo_harmonized_2mm", "halo_cosine_harmonized_v1", 2.0),
    ("C_halo_harmonized_3mm", "halo_cosine_harmonized_v1", 3.0),
    ("C_halo_harmonized_4mm", "halo_cosine_harmonized_v1", 4.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-manifest", required=True)
    parser.add_argument("--partition-audit", required=True)
    parser.add_argument("--reference-cdf", required=True)
    parser.add_argument("--reference-validation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def _exclusive_write(path: Path, payload: dict, mode: int = 0o444) -> None:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"measurement config directory already exists: {output_dir}")
    manifest = ComponentManifest.load(args.component_manifest)
    partition_path = Path(args.partition_audit).expanduser().resolve()
    reference_path = Path(args.reference_cdf).expanduser().resolve()
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    validation_path = Path(args.reference_validation).expanduser().resolve()
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "pass":
        raise ValueError("Reference validation is not passing")
    if validation.get("validation_audit_sha256") != canonical_json_sha256(
        validation, exclude=("validation_audit_sha256",)
    ):
        raise ValueError("Reference validation audit drifted")
    if validation.get("reference_file_sha256") != sha256_file(reference_path):
        raise ValueError("Reference validation binds another Reference file")
    if validation.get("reference_cdf_audit_sha256") != reference.get(
        "reference_cdf_audit_sha256"
    ):
        raise ValueError("Reference validation audit identity drifted")
    output_dir.mkdir(parents=True, exist_ok=False)
    index: list[dict] = []
    for candidate_id, policy, radius in CANDIDATES:
        calibration = build_measurement_calibration(
            reference=reference,
            reference_path=reference_path,
            partition_path=partition_path,
            manifest=manifest,
            boundary_policy=policy,
            halo_radius_mm=radius,
        )
        calibration_path = output_dir / f"{candidate_id}.measurement_calibration.json"
        _exclusive_write(calibration_path, calibration)
        loaded = FixV2Calibration.load(calibration_path, expected_policy=policy)
        config = build_measurement_route_config(
            manifest,
            calibration_sha256=loaded.sha256,
            boundary_policy=policy,
            seed=args.seed,
        )
        config_path = output_dir / f"{candidate_id}.route_config.json"
        _exclusive_write(config_path, config)
        index.append(
            {
                "candidate_id": candidate_id,
                "boundary_policy": policy,
                "halo_radius_mm": radius,
                "calibration_file": calibration_path.name,
                "calibration_sha256": sha256_file(calibration_path),
                "route_config_file": config_path.name,
                "route_config_sha256": sha256_file(config_path),
            }
        )
    index_payload = {
        "schema_version": 1,
        "status": "measurement_only_not_gate_eligible",
        "component_manifest_sha256": manifest.identity_sha256,
        "partition_sha256": sha256_file(partition_path),
        "reference_cdf_sha256": sha256_file(reference_path),
        "reference_validation_sha256": sha256_file(validation_path),
        "candidates": index,
    }
    _exclusive_write(output_dir / "MEASUREMENT_CONFIG_INDEX.json", index_payload)
    print(json.dumps(index_payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
