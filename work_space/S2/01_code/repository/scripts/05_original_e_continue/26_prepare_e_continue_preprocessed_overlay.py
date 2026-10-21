#!/usr/bin/env python3
"""Create an immutable true-1mm training overlay with nnU-Net provenance metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping


DATASET_NAME = "Dataset264_BraTS2026_MET_Completion"
EXPECTED_CACHE_AUDIT_IDENTITY = "17b7fc946528f68c5f9da7157cd1d80135edb6fcc1a6d1e9ecd2a450d17e056f"
EXPECTED_PLANS_SHA256 = "c20ac311f0b3db0f0710e98b0b56e65e8bb38c13b95094b6d6f9966ac529ffa5"
EXPECTED_FINGERPRINT_SHA256 = "09ef24f4564382184033a2fcdbc3bc1862cd33b93d45961ce9718df3db386102"
EXPECTED_SOURCE_ENTRIES = frozenset({
    "dataset.json",
    "gt_segmentations",
    "nnUNetPlans.json",
    "nnUNetPlans_3d_fullres",
    "splits_final.json",
})
AUDIT_NAME = "PREPROCESSED_OVERLAY_AUDIT.json"
AUDIT_SCHEMA = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-preprocessed-root", required=True)
    parser.add_argument("--fingerprint-source", required=True)
    parser.add_argument("--cache-audit", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any], *, exclude: tuple[str, ...] = ()) -> str:
    filtered = {key: value for key, value in payload.items() if key not in exclude}
    encoded = json.dumps(filtered, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _existing(path: str | Path, *, label: str, directory: bool) -> Path:
    resolved = Path(path).expanduser().resolve()
    valid = resolved.is_dir() if directory else resolved.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"missing {label} {kind}: {resolved}")
    return resolved


def inspect_sources(
    source_preprocessed_root: str | Path,
    fingerprint_source: str | Path,
    cache_audit: str | Path,
) -> dict[str, Any]:
    source_root = _existing(source_preprocessed_root, label="true-1mm root", directory=True)
    source_dataset = _existing(source_root / DATASET_NAME, label="true-1mm dataset", directory=True)
    observed_entries = {path.name for path in source_dataset.iterdir()}
    if observed_entries != EXPECTED_SOURCE_ENTRIES:
        raise RuntimeError(
            "true-1mm dataset top-level entries drifted: "
            f"observed={sorted(observed_entries)}, expected={sorted(EXPECTED_SOURCE_ENTRIES)}"
        )
    plans_path = _existing(source_dataset / "nnUNetPlans.json", label="true-1mm plans", directory=False)
    if sha256_file(plans_path) != EXPECTED_PLANS_SHA256:
        raise RuntimeError("true-1mm plans SHA256 drifted")

    fingerprint = _existing(fingerprint_source, label="Dataset264 fingerprint", directory=False)
    if sha256_file(fingerprint) != EXPECTED_FINGERPRINT_SHA256:
        raise RuntimeError("Dataset264 fingerprint SHA256 drifted")
    fingerprint_payload = json.loads(fingerprint.read_text(encoding="utf-8"))
    required_fingerprint_keys = {
        "foreground_intensity_properties_per_channel",
        "median_relative_size_after_cropping",
        "shapes_after_crop",
        "spacings",
    }
    if set(fingerprint_payload) != required_fingerprint_keys:
        raise RuntimeError("Dataset264 fingerprint schema drifted")
    if len(fingerprint_payload["shapes_after_crop"]) != 1138 or len(fingerprint_payload["spacings"]) != 1138:
        raise RuntimeError("Dataset264 fingerprint case count drifted")

    audit_path = _existing(cache_audit, label="true-1mm cache audit", directory=False)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or audit.get("audit_identity_sha256") != EXPECTED_CACHE_AUDIT_IDENTITY:
        raise RuntimeError("true-1mm cache audit identity drifted")
    if audit.get("plans", {}).get("file_sha256") != EXPECTED_PLANS_SHA256:
        raise RuntimeError("true-1mm cache audit plans binding drifted")
    return {
        "source_root": source_root,
        "source_dataset": source_dataset,
        "fingerprint": fingerprint,
        "cache_audit": audit_path,
        "cache_audit_sha256": sha256_file(audit_path),
    }


def expected_audit(sources: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    source_dataset = Path(sources["source_dataset"])
    entries: dict[str, Any] = {}
    for name in sorted(EXPECTED_SOURCE_ENTRIES):
        source = (source_dataset / name).resolve()
        row: dict[str, Any] = {
            "kind": "directory_symlink" if source.is_dir() else "file_symlink",
            "source": str(source),
        }
        if source.is_file():
            row["source_sha256"] = sha256_file(source)
        entries[name] = row
    payload: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "status": "pass",
        "dataset": DATASET_NAME,
        "output_root": str(output_root),
        "source_preprocessed_root": str(sources["source_root"]),
        "source_dataset": str(source_dataset),
        "source_cache_audit": str(sources["cache_audit"]),
        "source_cache_audit_sha256": sources["cache_audit_sha256"],
        "source_cache_audit_identity": EXPECTED_CACHE_AUDIT_IDENTITY,
        "source_plans_sha256": EXPECTED_PLANS_SHA256,
        "fingerprint_snapshot": {
            "source": str(sources["fingerprint"]),
            "source_sha256": EXPECTED_FINGERPRINT_SHA256,
            "destination": str(output_root / DATASET_NAME / "dataset_fingerprint.json"),
            "role": "raw Dataset264 provenance only; preprocessing and normalization remain bound to frozen E plans",
        },
        "entries": entries,
    }
    payload["audit_sha256"] = canonical_sha256(payload, exclude=("audit_sha256",))
    return payload


def validate_overlay(output_root: Path, expected: Mapping[str, Any]) -> None:
    audit_path = output_root / AUDIT_NAME
    observed = json.loads(_existing(audit_path, label="overlay audit", directory=False).read_text(encoding="utf-8"))
    if observed != expected:
        raise RuntimeError("immutable preprocessed overlay audit drifted")
    if observed.get("audit_sha256") != canonical_sha256(observed, exclude=("audit_sha256",)):
        raise RuntimeError("preprocessed overlay audit identity is invalid")
    dataset = _existing(output_root / DATASET_NAME, label="overlay dataset", directory=True)
    expected_names = set(EXPECTED_SOURCE_ENTRIES) | {"dataset_fingerprint.json"}
    if {path.name for path in dataset.iterdir()} != expected_names:
        raise RuntimeError("preprocessed overlay entries drifted")
    for name, row in expected["entries"].items():
        destination = dataset / name
        if not destination.is_symlink():
            raise RuntimeError(f"overlay source entry is not a symlink: {destination}")
        if destination.resolve() != Path(row["source"]):
            raise RuntimeError(f"overlay source symlink drifted: {destination}")
        if "source_sha256" in row and sha256_file(destination) != row["source_sha256"]:
            raise RuntimeError(f"overlay source file SHA256 drifted: {destination}")
    fingerprint = dataset / "dataset_fingerprint.json"
    if fingerprint.is_symlink() or sha256_file(fingerprint) != EXPECTED_FINGERPRINT_SHA256:
        raise RuntimeError("overlay fingerprint snapshot drifted")


def create_or_validate_overlay(
    source_preprocessed_root: str | Path,
    fingerprint_source: str | Path,
    cache_audit: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    sources = inspect_sources(source_preprocessed_root, fingerprint_source, cache_audit)
    output = Path(output_root).expanduser().resolve()
    if output == sources["source_root"] or sources["source_root"] in output.parents:
        raise RuntimeError("overlay must not be created inside the true-1mm cache")
    expected = expected_audit(sources, output)
    if output.exists():
        validate_overlay(output, expected)
        return {"state": "validated", "audit": expected}

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        dataset = temporary / DATASET_NAME
        dataset.mkdir()
        for name in sorted(EXPECTED_SOURCE_ENTRIES):
            source = (Path(sources["source_dataset"]) / name).resolve()
            os.symlink(source, dataset / name, target_is_directory=source.is_dir())
        shutil.copy2(sources["fingerprint"], dataset / "dataset_fingerprint.json")
        (temporary / AUDIT_NAME).write_text(
            json.dumps(expected, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    validate_overlay(output, expected)
    return {"state": "created", "audit": expected}


def main() -> None:
    args = parse_args()
    result = create_or_validate_overlay(
        args.source_preprocessed_root,
        args.fingerprint_source,
        args.cache_audit,
        args.output_root,
    )
    print(json.dumps({
        "status": "pass",
        "state": result["state"],
        "output_root": result["audit"]["output_root"],
        "audit_sha256": result["audit"]["audit_sha256"],
    }, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
