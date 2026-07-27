#!/usr/bin/env python3
"""Prepare the mandatory manual-review queue for the full Diffusion gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from g2_finalize_diffusion_gate import required_review_ids


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def prepare_review(
    qc_root: Path,
    output_root: Path,
    *,
    low_score_count: int = 10,
    batch_size: int = 12,
) -> dict[str, object]:
    review_index_path = qc_root / "review_index.csv"
    montage_root = qc_root / "montages"
    if not review_index_path.is_file() or not montage_root.is_dir():
        raise FileNotFoundError("Full QC review_index.csv or montages/ is missing")
    output_csv = output_root / "mandatory_review_template.csv"
    output_json = output_root / "review_batches.json"
    output_readme = output_root / "MANDATORY_REVIEW_README.md"
    existing = [path for path in (output_csv, output_json, output_readme) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite review queue: {existing}")

    review_rows = read_csv(review_index_path)
    if len(review_rows) != 94:
        raise ValueError(f"Expected 94 review-index rows, found {len(review_rows)}")
    mandatory_ids, mandatory_reasons = required_review_ids(
        review_rows, low_score_count=low_score_count
    )
    row_by_id = {row["source_case_id"]: row for row in review_rows}
    missing_montages = sorted(
        case_id
        for case_id in mandatory_ids
        if not (montage_root / f"{case_id}.png").is_file()
    )
    if missing_montages:
        raise FileNotFoundError(f"Mandatory montages are missing: {missing_montages[:20]}")

    queue = sorted(
        mandatory_ids,
        key=lambda case_id: (
            -len(mandatory_reasons[case_id]),
            float(row_by_id[case_id]["min_tumour_ssim"]),
            case_id,
        ),
    )
    rows = []
    for index, case_id in enumerate(queue, start=1):
        source = row_by_id[case_id]
        rows.append(
            {
                "review_order": index,
                "review_batch": (index - 1) // batch_size + 1,
                "source_case_id": case_id,
                "mandatory_reasons": ";".join(mandatory_reasons[case_id]),
                "has_rc": source.get("has_rc", ""),
                "tiny_count": source.get("tiny_count", ""),
                "small_count": source.get("small_count", ""),
                "large_count": source.get("large_count", ""),
                "min_tumour_ssim": source.get("min_tumour_ssim", ""),
                "mean_support_ssim": source.get("mean_support_ssim", ""),
                "artifact_flags": source.get("artifact_flags", ""),
                "montage_path": str((montage_root / f"{case_id}.png").resolve()),
                "manual_decision": "",
                "risk_accepted": "",
                "risk_code": "",
                "manual_observation": "",
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    batches = [queue[index : index + batch_size] for index in range(0, len(queue), batch_size)]
    summary = {
        "full_case_count": len(review_rows),
        "mandatory_review_count": len(queue),
        "low_score_count": low_score_count,
        "batch_size": batch_size,
        "batch_count": len(batches),
        "batches": [
            {"batch": index, "source_case_ids": values}
            for index, values in enumerate(batches, start=1)
        ],
    }
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    output_readme.write_text(
        "# Diffusion Full94 Mandatory Manual Review\n\n"
        f"- Full lesion-positive cases: 94\n"
        f"- Mandatory review cases: {len(queue)}\n"
        f"- Review batches: {len(batches)} x up to {batch_size}\n\n"
        "The queue is the union of all RC, tiny, large/tiled, artifact-flagged, "
        "lowest-score and smoke-risk cases. Allowed decisions are "
        "`pass_technical_visual`, `pass_with_documented_risk`, "
        "`needs_regeneration`, and `reject`. Every documented risk requires "
        "`risk_accepted=True` before the final gate can be approved.\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--low-score-count", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()
    summary = prepare_review(
        args.qc_root,
        args.output_root,
        low_score_count=args.low_score_count,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
