#!/usr/bin/env python3
"""Run paired Run-L/Run-H G1 sampling for Fix-v2 Gate-0 calibration evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom_nnunet.met_aug_core import (  # noqa: E402
    S2_MODALITIES,
    canonical_json_sha256,
    sha256_file,
)
from custom_nnunet.met_aug_diffusion import (  # noqa: E402
    G1FourModalityInpaintingBackend,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--g1-code-dir", required=True)
    parser.add_argument("--g1-checkpoint-root", required=True)
    parser.add_argument("--g1-checkpoint-selection", required=True)
    parser.add_argument("--g2-parent-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _ncc(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= np.finfo(np.float64).eps:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "q50": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_npz).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"paired ablation output already exists: {output_dir}")
    if not input_path.is_file():
        raise FileNotFoundError(f"paired ablation input is missing: {input_path}")
    with np.load(input_path, allow_pickle=False) as payload:
        required = {"image_crop", "label_crop", "halo_support", "spacing_mm"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"paired ablation input misses arrays: {missing}")
        image = np.asarray(payload["image_crop"], dtype=np.float32)
        label = np.asarray(payload["label_crop"], dtype=np.int16)
        halo_support = np.asarray(payload["halo_support"])
        spacing = tuple(float(value) for value in np.asarray(payload["spacing_mm"]).tolist())
    if image.shape != (4, 64, 64, 64) or label.shape != (64, 64, 64):
        raise ValueError("paired ablation requires one 4x64^3 crop and one 64^3 label")
    if halo_support.shape != label.shape or halo_support.dtype != np.bool_:
        raise ValueError("halo_support must be a boolean 64^3 array")
    if len(spacing) != 3 or any(not np.isfinite(value) or value <= 0 for value in spacing):
        raise ValueError("paired ablation spacing is invalid")
    if not np.all(np.isfinite(image)):
        raise ValueError("paired ablation input image is non-finite")
    label_values = set(int(value) for value in np.unique(label))
    if not label_values.issubset({0, 1, 2, 3}) or not np.any(label != 0):
        raise ValueError("paired ablation label contains RC/invalid classes or is empty")
    label_support = label != 0
    if np.any(label_support & ~halo_support) or not np.any(halo_support & ~label_support):
        raise ValueError("paired ablation requires strict L subset H")

    backend = G1FourModalityInpaintingBackend(
        g1_code_dir=args.g1_code_dir,
        checkpoint_root=args.g1_checkpoint_root,
        checkpoint_selection=args.g1_checkpoint_selection,
        qc_gate=args.g2_parent_gate,
        device=args.device,
    )
    run_l = backend.generate(
        image,
        label,
        seed=args.seed,
        inpaint_support=label_support,
    )
    run_h = backend.generate(
        image,
        label,
        seed=args.seed,
        inpaint_support=halo_support,
    )
    if not np.all(np.isfinite(run_l)) or not np.all(np.isfinite(run_h)):
        raise RuntimeError("paired G1 output is non-finite")
    halo_only = halo_support & ~label_support
    outside_h = ~halo_support
    violations: list[str] = []
    if not np.array_equal(run_l[:, halo_only], image[:, halo_only]):
        violations.append("run_l_changed_halo_only_region")
    if not np.array_equal(run_l[:, outside_h], image[:, outside_h]):
        violations.append("run_l_changed_outside_h")
    if not np.array_equal(run_h[:, outside_h], image[:, outside_h]):
        violations.append("run_h_changed_outside_h")

    modality_metrics: dict[str, dict] = {}
    for channel, modality in enumerate(S2_MODALITIES):
        halo_residual = np.abs(run_h[channel][halo_only] - image[channel][halo_only])
        lesion_pair_drift = np.abs(run_h[channel][label_support] - run_l[channel][label_support])
        original_gradients = np.stack(np.gradient(image[channel], *spacing), axis=0)
        generated_gradients = np.stack(np.gradient(run_h[channel], *spacing), axis=0)
        dot = np.sum(original_gradients * generated_gradients, axis=0)
        denominator = np.linalg.norm(original_gradients, axis=0) * np.linalg.norm(
            generated_gradients, axis=0
        )
        informative = halo_only & (denominator > np.finfo(np.float32).eps)
        gradient_cosine_q05 = 1.0
        if np.any(informative):
            gradient_cosine_q05 = float(
                np.quantile(dot[informative] / denominator[informative], 0.05)
            )
        modality_metrics[modality] = {
            "run_h_halo_residual": _quantiles(halo_residual),
            "run_h_halo_nonzero_fraction": float(np.count_nonzero(halo_residual) / halo_residual.size),
            "run_l_run_h_lesion_drift": _quantiles(lesion_pair_drift),
            "halo_ncc": _ncc(
                image[channel][halo_only].astype(np.float64),
                run_h[channel][halo_only].astype(np.float64),
            ),
            "halo_gradient_cosine_q05": gradient_cosine_q05,
        }
        if not np.any(halo_residual != 0):
            violations.append(f"{modality}:run_h_has_no_halo_generation")

    if violations:
        raise RuntimeError(f"paired backend contract failed: {sorted(violations)}")
    output_dir.mkdir(parents=True, exist_ok=False)
    arrays_path = output_dir / "paired_outputs.npz"
    np.savez_compressed(
        arrays_path,
        original=image,
        label=label,
        label_support=label_support,
        halo_support=halo_support,
        run_l=run_l,
        run_h=run_h,
        spacing_mm=np.asarray(spacing, dtype=np.float64),
    )
    report = {
        "schema_version": 1,
        "status": "complete_for_calibration",
        "seed": args.seed,
        "input_npz_sha256": sha256_file(input_path),
        "g1_checkpoint_selection_sha256": sha256_file(args.g1_checkpoint_selection),
        "g2_parent_gate_sha256": sha256_file(args.g2_parent_gate),
        "g1_runtime_code": backend.runtime_code,
        "runner_sha256": sha256_file(Path(__file__)),
        "label_support_voxels": int(np.count_nonzero(label_support)),
        "halo_support_voxels": int(np.count_nonzero(halo_support)),
        "modalities": modality_metrics,
        "arrays_file": arrays_path.name,
        "arrays_sha256": sha256_file(arrays_path),
        "violations": [],
    }
    report["report_sha256"] = canonical_json_sha256(report, exclude=("report_sha256",))
    report_path = output_dir / "paired_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
