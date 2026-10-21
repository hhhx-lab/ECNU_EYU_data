#!/usr/bin/env python3
"""Run the experimental_unvalidated BraTS 2026 Task 1 container contract."""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


CASE_ID_PATTERN = re.compile(r"^BraTS-MET-[0-9]{5}-[0-9]{3}$")
CHANNELS = (
    ("t1n", "0000"),
    ("t1c", "0001"),
    ("t2w", "0002"),
    ("t2f", "0003"),
)
ALLOWED_LABELS = {0, 1, 2, 3, 4}


class ContractError(RuntimeError):
    """Raised when mounted inputs or generated outputs violate the task contract."""


@dataclass(frozen=True)
class Case:
    case_id: str
    directory: Path
    sources: dict[str, Path]


def env_positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ContractError(f"{name} must be positive, got {value}")
    return value


def discover_cases(input_root: Path) -> list[Case]:
    if not input_root.is_dir():
        raise ContractError(f"Input directory does not exist: {input_root}")

    case_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    invalid_dirs = [path.name for path in case_dirs if not CASE_ID_PATTERN.fullmatch(path.name)]
    if invalid_dirs:
        raise ContractError(f"Unexpected case directories: {invalid_dirs[:10]}")
    if not case_dirs:
        raise ContractError(f"No Task 1 case directories found in {input_root}")

    cases: list[Case] = []
    for case_dir in case_dirs:
        expected_names = {
            f"{case_dir.name}-{modality}.nii.gz" for modality, _ in CHANNELS
        }
        actual_names = {
            path.name
            for path in case_dir.iterdir()
            if path.is_file() and path.name.endswith(".nii.gz")
        }
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        if missing or unexpected:
            raise ContractError(
                f"Invalid modalities for {case_dir.name}: "
                f"missing={missing}, unexpected={unexpected}"
            )

        sources: dict[str, Path] = {}
        for modality, _ in CHANNELS:
            source = case_dir / f"{case_dir.name}-{modality}.nii.gz"
            if source.stat().st_size <= 0:
                raise ContractError(f"Empty input volume: {source}")
            sources[modality] = source
        cases.append(Case(case_dir.name, case_dir, sources))
    return cases


def require_empty_output(output_root: Path) -> None:
    if not output_root.is_dir():
        raise ContractError(f"Output directory is not mounted: {output_root}")
    existing = sorted(path.name for path in output_root.iterdir() if not path.name.startswith("."))
    if existing:
        raise ContractError(f"Output directory must be empty: {existing[:10]}")


def materialize_nnunet_input(cases: list[Case], prepared_root: Path) -> None:
    prepared_root.mkdir(parents=True, exist_ok=False)
    for case in cases:
        for modality, channel in CHANNELS:
            target = prepared_root / f"{case.case_id}_{channel}.nii.gz"
            target.symlink_to(case.sources[modality].resolve(strict=True))


def validate_label_values(unique_values: Iterable[object], case_id: str) -> set[int]:
    numeric_values = [float(value) for value in unique_values]
    if not all(math.isfinite(value) for value in numeric_values):
        raise ContractError(f"Non-finite prediction values for {case_id}")
    if any(value != round(value) for value in numeric_values):
        raise ContractError(f"Non-integer prediction values for {case_id}")

    labels = {int(value) for value in numeric_values}
    if not labels.issubset(ALLOWED_LABELS):
        raise ContractError(f"Unexpected labels for {case_id}: {sorted(labels)}")
    return labels


def validate_prediction(case: Case, prediction: Path) -> None:
    import nibabel as nib
    import numpy as np

    if not prediction.is_file() or prediction.stat().st_size <= 0:
        raise ContractError(f"Missing or empty prediction: {prediction}")

    source_image = nib.load(str(case.sources["t1n"]))
    prediction_image = nib.load(str(prediction))
    if tuple(prediction_image.shape) != tuple(source_image.shape):
        raise ContractError(f"Shape mismatch for {case.case_id}")
    if not np.allclose(
        prediction_image.header.get_zooms()[:3],
        source_image.header.get_zooms()[:3],
        rtol=0,
        atol=1e-5,
    ):
        raise ContractError(f"Spacing mismatch for {case.case_id}")
    if not np.allclose(prediction_image.affine, source_image.affine, rtol=0, atol=1e-4):
        raise ContractError(f"Affine mismatch for {case.case_id}")

    values = np.asanyarray(prediction_image.dataobj)
    validate_label_values(np.unique(values), case.case_id)


def run() -> None:
    input_root = Path(os.environ.get("S2_INPUT_ROOT", "/input")).resolve()
    output_root = Path(os.environ.get("S2_OUTPUT_ROOT", "/output")).resolve()
    model_root = Path(os.environ.get("S2_MODEL_ROOT", "/opt/model")).resolve()
    preprocess_workers = env_positive_int("S2_PREPROCESS_WORKERS", 4)
    export_workers = env_positive_int("S2_EXPORT_WORKERS", 4)

    cases = discover_cases(input_root)
    require_empty_output(output_root)
    expected_names = {f"{case.case_id}.nii.gz" for case in cases}

    for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        os.environ.setdefault(name, f"/tmp/{name}")

    with tempfile.TemporaryDirectory(prefix="s2-original-e-") as temporary:
        temporary_root = Path(temporary)
        prepared_root = temporary_root / "input"
        raw_output_root = temporary_root / "output"
        materialize_nnunet_input(cases, prepared_root)
        raw_output_root.mkdir()

        from inference_frozen import run as run_frozen_inference

        run_frozen_inference(
            SimpleNamespace(
                input=str(prepared_root),
                output=str(raw_output_root),
                model_root=str(model_root),
                fold=0,
                preprocess_workers=preprocess_workers,
                export_workers=export_workers,
            )
        )

        actual_names = {
            path.name
            for path in raw_output_root.iterdir()
            if path.is_file() and path.name.endswith(".nii.gz")
        }
        if actual_names != expected_names:
            raise ContractError(
                "Prediction coverage mismatch: "
                f"missing={sorted(expected_names - actual_names)[:10]}, "
                f"unexpected={sorted(actual_names - expected_names)[:10]}"
            )

        for case in cases:
            source_prediction = raw_output_root / f"{case.case_id}.nii.gz"
            validate_prediction(case, source_prediction)
            shutil.copy2(source_prediction, output_root / source_prediction.name)

    final_names = {
        path.name
        for path in output_root.iterdir()
        if path.is_file() and path.name.endswith(".nii.gz")
    }
    if final_names != expected_names:
        raise ContractError("Final output coverage changed during publication")
    print(
        "S2_ORIGINAL_E_CONTAINER_PASS "
        f"status=experimental_unvalidated cases={len(cases)} outputs={len(final_names)}"
    )


if __name__ == "__main__":
    run()
