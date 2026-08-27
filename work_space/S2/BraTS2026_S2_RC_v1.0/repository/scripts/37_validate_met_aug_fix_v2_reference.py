#!/usr/bin/env python3
"""Strictly validate completed train-only Fix-v2 Reference evidence."""

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
from custom_nnunet.met_aug_fix_v2_calibration import (  # noqa: E402
    validate_reference_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-manifest", required=True)
    parser.add_argument("--partition-audit", required=True)
    parser.add_argument("--valid-mask-manifest", required=True)
    parser.add_argument("--preprocessed-dir", required=True)
    parser.add_argument("--reference-cdf", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _preprocessed_contract(preprocessed: Path, manifest: ComponentManifest) -> str:
    plans_path = preprocessed.parent / "nnUNetPlans.json"
    payload = json.loads(plans_path.read_text(encoding="utf-8"))
    matches = [
        (str(name), config)
        for name, config in payload.get("configurations", {}).items()
        if isinstance(config, dict)
        and str(config.get("data_identifier", "")) == preprocessed.name
    ]
    if len(matches) != 1:
        raise ValueError("preprocessed directory does not uniquely match nnU-Net plans")
    name, config = matches[0]
    spacing = [float(value) for value in config.get("spacing", ())]
    if spacing != [1.0, 1.0, 1.0]:
        raise ValueError(f"preprocessed plans are not true 1 mm: {spacing}")
    return canonical_json_sha256(
        {
            "directory": str(preprocessed),
            "plans_sha256": sha256_file(plans_path),
            "plans_configuration": name,
            "data_identifier": preprocessed.name,
            "post_resampling_spacing_mm": spacing,
            "spacing_source": "nnUNetPlans.configuration.spacing",
            "component_manifest_sha256": manifest.identity_sha256,
        }
    )


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Reference validation output already exists: {output}")
    manifest = ComponentManifest.load(args.component_manifest)
    reference_path = Path(args.reference_cdf).expanduser().resolve()
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    preprocessed = Path(args.preprocessed_dir).expanduser().resolve()
    summary = validate_reference_evidence(
        reference,
        reference_path=reference_path,
        partition_path=args.partition_audit,
        manifest=manifest,
        expected_valid_mask_manifest_sha256=sha256_file(args.valid_mask_manifest),
        expected_preprocessed_contract_sha256=_preprocessed_contract(
            preprocessed, manifest
        ),
    )
    summary["validator_sha256"] = sha256_file(Path(__file__))
    summary["validation_audit_sha256"] = canonical_json_sha256(
        summary, exclude=("validation_audit_sha256",)
    )
    encoded = json.dumps(summary, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        output.unlink(missing_ok=True)
        raise
    print(encoded, end="")


if __name__ == "__main__":
    main()
