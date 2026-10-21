#!/usr/bin/env python3
"""Statically validate the isolated original E Docker build context."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STATIC_SHA256_MANIFEST = ROOT / "STATIC_CONTEXT_SHA256.json"
EXPECTED_SHA256 = {
    "app/inference_frozen.py": "fa367326d18d6e03ef3b313daca7213af43a96ecfa818fe29e64c52d75df9b38",
    "app/custom_nnunet/nnUNetTrainerBraTS2026RC_inference.py": "c7c464c9ec283d75ab7d2f353d50025dc52e8b59dcc0d1f0256abdff7b78df13",
    "requirements-inference.txt": "e33e7207c30418d68b7abeeec1df1826f080172ccdaa1ab01e42109f88366b24",
    "model/dataset.json": "e6e77a97ed8f43c44b3ae647e4cbd51606cf75393bb1241cb9f862e9579a43e8",
    "model/plans.json": "c20ac311f0b3db0f0710e98b0b56e65e8bb38c13b95094b6d6f9966ac529ffa5",
}
EXPECTED_CHECKPOINT_SHA256 = "4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267"
EXPECTED_STATIC_FILES = {
    ".dockerignore",
    ".env.example",
    ".gitignore",
    "CONTEXT_MANIFEST.json",
    "Dockerfile",
    "README.md",
    "app/custom_nnunet/__init__.py",
    "app/custom_nnunet/nnUNetTrainerBraTS2026RC_inference.py",
    "app/inference_frozen.py",
    "app/run_container.py",
    "build_local.sh",
    "compose.yaml",
    "model/dataset.json",
    "model/plans.json",
    "requirements-inference.txt",
    "tests/test_run_container.py",
    "validate_context.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    default_checkpoint = (
        ROOT.parents[1]
        / "results/s2_small_lesion_ablation_20260721"
        / "remote_snapshot_complete_20260724T0343/focal/fold_0/checkpoint_final.pth"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    args = parser.parse_args()

    checked: dict[str, str] = {}
    for relative, expected in EXPECTED_SHA256.items():
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"Missing frozen context file: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise SystemExit(f"SHA256 drift for {relative}: {actual} != {expected}")
        checked[relative] = actual

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Missing original E checkpoint: {checkpoint}")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit("Original E checkpoint SHA256 drift")

    manifest = json.loads((ROOT / "CONTEXT_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_status") != "experimental_unvalidated":
        raise SystemExit("Context status marker is missing")
    if manifest.get("checkpoint", {}).get("sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit("Context manifest checkpoint binding drift")

    static_manifest = json.loads(STATIC_SHA256_MANIFEST.read_text(encoding="utf-8"))
    if static_manifest.get("artifact_status") != "experimental_unvalidated":
        raise SystemExit("Static SHA256 manifest status marker is missing")
    static_files = static_manifest.get("files")
    if not isinstance(static_files, dict) or set(static_files) != EXPECTED_STATIC_FILES:
        raise SystemExit("Static SHA256 manifest file coverage drift")
    for relative, expected in sorted(static_files.items()):
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise SystemExit(f"Static SHA256 drift for {relative}: {actual} != {expected}")

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    required_fragments = (
        "experimental_unvalidated",
        "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime",
        "COPY --from=checkpoint checkpoint_final.pth",
        "ENTRYPOINT",
    )
    for fragment in required_fragments:
        if fragment not in dockerfile:
            raise SystemExit(f"Dockerfile contract missing: {fragment}")
    if (ROOT / "model/fold_0/checkpoint_final.pth").exists():
        raise SystemExit("Checkpoint must remain outside the build context")

    build_script = (ROOT / "build_local.sh").read_text(encoding="utf-8")
    if '--build-context "checkpoint=${CHECKPOINT_CONTEXT}"' not in build_script:
        raise SystemExit("Build script must inject the temporary checkpoint-only context")
    if '--build-context "checkpoint=${CHECKPOINT_DIR}"' in build_script:
        raise SystemExit("Build script exposes the complete original fold directory")
    if "python3 " in build_script:
        raise SystemExit("Build script bypasses the configured Conda-first Python")

    print(
        json.dumps(
            {
                "status": "pass",
                "artifact_status": "experimental_unvalidated",
                "checkpoint_sha256": checkpoint_sha256,
                "frozen_file_count": len(checked),
                "static_file_count": len(static_files),
                "registry_push_performed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
