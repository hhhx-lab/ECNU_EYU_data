#!/usr/bin/env python3
"""Extract immutable empirical Fix-v2 Reference evidence from train-only data."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import (  # noqa: E402
    ComponentManifest,
    VALID_MASK_MANIFEST_SCHEMA,
    canonical_json_sha256,
    sha256_file,
)
from custom_nnunet.met_aug_fix_v2_calibration import (  # noqa: E402
    ReferenceCase,
    build_reference_evidence,
    normalize_preprocessed_segmentation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-manifest", required=True)
    parser.add_argument("--partition-audit", required=True)
    parser.add_argument("--valid-mask-manifest", required=True)
    parser.add_argument("--preprocessed-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def _load_b2nd(path: Path) -> np.ndarray:
    try:
        import blosc2
    except ImportError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("the configured nnU-Net Conda environment lacks blosc2") from exc
    return np.asarray(blosc2.open(str(path), mode="r")[:])


def _valid_mask_index(path: Path) -> tuple[dict[str, dict], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != VALID_MASK_MANIFEST_SCHEMA:
        raise ValueError("valid-mask manifest schema drifted")
    if payload.get("manifest_sha256") != canonical_json_sha256(
        payload, exclude=("manifest_sha256",)
    ):
        raise ValueError("valid-mask manifest identity drifted")
    records_path = path.parent / str(payload.get("records_file", ""))
    if not records_path.is_file() or sha256_file(records_path) != payload.get("records_sha256"):
        raise ValueError("valid-mask records drifted")
    records = {
        str(row["case_id"]): row
        for row in (
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    return records, sha256_file(path)


def _preprocessed_spacing(
    plans_path: Path, preprocessed_dir: Path
) -> tuple[tuple[float, float, float], str]:
    """Return the post-resampling spacing bound to this data directory."""
    payload = json.loads(plans_path.read_text(encoding="utf-8"))
    configurations = payload.get("configurations")
    if not isinstance(configurations, dict):
        raise ValueError("nnU-Net plans have no configurations")
    matches = [
        (str(name), value)
        for name, value in configurations.items()
        if isinstance(value, dict)
        and str(value.get("data_identifier", "")) == preprocessed_dir.name
    ]
    if len(matches) != 1:
        raise ValueError(
            "preprocessed directory must match exactly one nnU-Net plans configuration; "
            f"directory={preprocessed_dir.name}, matches={[name for name, _ in matches]}"
        )
    configuration_name, configuration = matches[0]
    spacing = tuple(float(value) for value in configuration.get("spacing", ()))
    if len(spacing) != 3 or not np.allclose(spacing, (1.0, 1.0, 1.0), atol=1e-6):
        raise ValueError(
            "Fix-v2 Reference requires post-resampling 1 mm spacing in nnU-Net plans, "
            f"got {spacing}"
        )
    return spacing, configuration_name


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Reference evidence already exists: {output}")
    manifest = ComponentManifest.load(args.component_manifest)
    partition_path = Path(args.partition_audit).expanduser().resolve()
    valid_manifest_path = Path(args.valid_mask_manifest).expanduser().resolve()
    preprocessed = Path(args.preprocessed_dir).expanduser().resolve()
    if not preprocessed.is_dir():
        raise FileNotFoundError(f"preprocessed directory is missing: {preprocessed}")
    valid_records, valid_manifest_sha = _valid_mask_index(valid_manifest_path)
    plans_path = preprocessed.parent / "nnUNetPlans.json"
    if not plans_path.is_file():
        raise FileNotFoundError(f"nnU-Net plans are missing: {plans_path}")
    spacing, configuration_name = _preprocessed_spacing(plans_path, preprocessed)
    preprocessed_contract_sha = canonical_json_sha256(
        {
            "directory": str(preprocessed),
            "plans_sha256": sha256_file(plans_path),
            "plans_configuration": configuration_name,
            "data_identifier": preprocessed.name,
            "post_resampling_spacing_mm": list(spacing),
            "spacing_source": "nnUNetPlans.configuration.spacing",
            "component_manifest_sha256": manifest.identity_sha256,
        }
    )

    def load_case(case_id: str, patient_group: str) -> ReferenceCase:
        image_path = preprocessed / f"{case_id}.b2nd"
        segmentation_path = preprocessed / f"{case_id}_seg.b2nd"
        properties_path = preprocessed / f"{case_id}.pkl"
        for path in (image_path, segmentation_path, properties_path):
            if not path.is_file():
                raise FileNotFoundError(f"{case_id}: missing preprocessed asset {path.name}")
        image = _load_b2nd(image_path).astype(np.float32, copy=False)
        segmentation = normalize_preprocessed_segmentation(_load_b2nd(segmentation_path))
        with properties_path.open("rb") as handle:
            properties = pickle.load(handle)
        source_spacing = tuple(float(value) for value in properties.get("spacing", ()))
        if len(source_spacing) != 3 or any(
            not np.isfinite(value) or value <= 0 for value in source_spacing
        ):
            raise ValueError(f"{case_id}: invalid source-image spacing {source_spacing}")
        record = valid_records.get(case_id)
        if record is None:
            raise ValueError(f"{case_id}: valid-mask record is missing")
        mask_path = valid_manifest_path.parent / str(record["mask_path"])
        if not mask_path.is_file() or sha256_file(mask_path) != record.get("sha256"):
            raise ValueError(f"{case_id}: valid-mask artifact drifted")
        with np.load(mask_path, allow_pickle=False) as payload:
            valid_mask = np.asarray(payload["valid_mask"], dtype=bool)
        return ReferenceCase(
            case_id=case_id,
            patient_group=patient_group,
            image=image,
            segmentation=segmentation,
            valid_mask=valid_mask,
            spacing_mm=spacing,
        )

    evidence = build_reference_evidence(
        manifest=manifest,
        partition_path=partition_path,
        valid_mask_manifest_sha256=valid_manifest_sha,
        preprocessed_contract_sha256=preprocessed_contract_sha,
        case_loader=load_case,
        workers=args.workers,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(evidence, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        output.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "output": str(output),
                "output_sha256": sha256_file(output),
                "reference_cdf_audit_sha256": evidence["reference_cdf_audit_sha256"],
                "patient_group_count": evidence["patient_group_count"],
                "component_count": evidence["component_count"],
                "usable_component_count": evidence["usable_component_count"],
                "excluded_component_count": evidence["excluded_component_count"],
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
