#!/usr/bin/env python3
"""Validate and intake one G1 V3 missing-T2W completion run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from g2_synthetic_raw_intake_qc import DEFAULT_RESULTS_ROOT, run_intake


REQUIRED_FIELDS = (
    "generation_run_id",
    "generator_name",
    "seed",
    "source_csv",
    "vae_weights",
    "encdec_checkpoint",
    "bbdm_checkpoint",
    "bbdm_s",
    "validation_run",
)


def load_metadata(run_root: Path) -> tuple[Path, dict[str, object]]:
    for name in ("generation_config.json", "inference_run.json"):
        path = run_root / name
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"{path} must contain a JSON object")
            return path, data
    raise FileNotFoundError("V3 run is missing generation_config.json/inference_run.json")


def validate_v3_metadata(metadata: dict[str, object]) -> None:
    missing = [field for field in REQUIRED_FIELDS if metadata.get(field) in (None, "")]
    if missing:
        raise ValueError("V3 completion metadata is incomplete: " + ", ".join(missing))
    generation_mode = str(metadata.get("generation_mode") or metadata.get("generator_io") or "")
    if "completion" not in generation_mode:
        raise ValueError("V3 generation metadata must declare completion mode")


def validate_v3_delivery(run_root: Path) -> None:
    missing = [
        name
        for name in ("synthetic_generation_manifest.csv", "generation_log.jsonl")
        if not (run_root / name).is_file()
    ]
    if missing:
        raise ValueError("V3 completion delivery is incomplete: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion-run-root", required=True)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument(
        "--data-root",
        default=os.environ.get("G2_DATA_ROOT", ""),
        help=(
            "External BraTS data root containing the Training, corrected-labels, "
            "and optional Validation directories. This remaps versioned "
            "work_space/G1/data/raw paths without copying the dataset."
        ),
    )
    parser.add_argument("--synthetic-run-id", default="")
    parser.add_argument("--refresh-templates", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.completion_run_root).expanduser().resolve()
    if not run_root.is_dir():
        raise SystemExit(f"V3 completion run not found: {run_root}")
    if args.data_root:
        data_root = Path(args.data_root).expanduser().resolve()
        if not data_root.is_dir():
            raise SystemExit(f"BraTS data root not found: {data_root}")
        os.environ["G2_DATA_ROOT"] = str(data_root)
    try:
        _, metadata = load_metadata(run_root)
        validate_v3_metadata(metadata)
        validate_v3_delivery(run_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    run_intake(
        argparse.Namespace(
            synthetic_run_root=str(run_root),
            results_root=args.results_root,
            synthetic_run_id=args.synthetic_run_id,
            generation_mode="completion",
            refresh_templates=args.refresh_templates,
        )
    )


if __name__ == "__main__":
    main()
