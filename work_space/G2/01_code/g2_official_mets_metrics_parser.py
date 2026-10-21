#!/usr/bin/env python3
"""Parse or validate BraTS 2026 Task1 MET leaderboard metrics.

The parser mirrors the public BraTS_evaluation MET parser logic and emits the
leaderboard columns visible for Task1: lesionwise DSC/NSD for ET/RC/TC/WT and
small-instance TP/FN/FP/F1 for ET/TC/WT/RC.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median


REGIONS = ["et", "rc", "tc", "wt"]
SMALL_REGIONS = ["et", "tc", "wt", "rc"]
LEADERBOARD_COLUMNS = [
    "lesionwise_dsc_mean_et",
    "lesionwise_nsd_mean_et",
    "lesionwise_dsc_mean_rc",
    "lesionwise_nsd_mean_rc",
    "lesionwise_dsc_mean_tc",
    "lesionwise_nsd_mean_tc",
    "lesionwise_dsc_mean_wt",
    "lesionwise_nsd_mean_wt",
    "small_instance_tp_et",
    "small_instance_fn_et",
    "small_instance_fp_et",
    "small_instance_f1_et",
    "small_instance_tp_tc",
    "small_instance_fn_tc",
    "small_instance_fp_tc",
    "small_instance_f1_tc",
    "small_instance_tp_wt",
    "small_instance_fn_wt",
    "small_instance_fp_wt",
    "small_instance_f1_wt",
    "small_instance_tp_rc",
    "small_instance_fn_rc",
    "small_instance_fp_rc",
    "small_instance_f1_rc",
]


def is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def mean(values: list[float]) -> float:
    clean = [v for v in values if not is_nan(v)]
    return sum(clean) / len(clean) if clean else math.nan


def std(values: list[float]) -> float:
    clean = [v for v in values if not is_nan(v)]
    if len(clean) <= 1:
        return math.nan
    avg = sum(clean) / len(clean)
    return math.sqrt(sum((v - avg) ** 2 for v in clean) / (len(clean) - 1))


def med(values: list[float]) -> float:
    clean = [v for v in values if not is_nan(v)]
    return float(median(clean)) if clean else math.nan


def f1(tp: float, fp: float, fn: float) -> float:
    den = 2 * tp + fp + fn
    return (2 * tp) / den if den > 0 else 0.0


def parse_subject(subject_data: dict[str, object], vol_threshold: float, overlap_threshold: float) -> dict[str, object]:
    subject_id = str(subject_data.get("subject_name", ""))
    row: dict[str, object] = {"subject_id": subject_id}

    for region in REGIONS:
        region_data = subject_data.get(region)
        if not isinstance(region_data, dict):
            continue

        large_dsc: list[float] = []
        large_nsd: list[float] = []
        large_hd95: list[float] = []
        large_tp = 0
        large_fn = 0
        small_tp = 0
        small_fn = 0
        small_found = False
        large_present = False

        lesion_instances = region_data.get("reference_instances", [])
        if not isinstance(lesion_instances, list):
            lesion_instances = []
        for lesion_data in lesion_instances:
            if not isinstance(lesion_data, dict):
                continue
            volume = lesion_data.get("volume")
            if volume is None:
                continue
            volume = float(volume)
            is_large = volume >= vol_threshold
            if is_large:
                large_present = True
            else:
                small_found = True

            matched = lesion_data.get("is_matched") == 1
            sq_dsc = lesion_data.get("sq_dsc")
            sq_dsc_f = float(sq_dsc) if sq_dsc is not None else None
            if matched:
                if is_large:
                    large_dsc.append(sq_dsc_f if sq_dsc_f is not None else 0.0)
                    large_nsd.append(float(lesion_data.get("sq_nsd") or 0.0))
                    hd95 = lesion_data.get("sq_hd95")
                    large_hd95.append(float(hd95) if hd95 is not None and not math.isinf(float(hd95)) else 373.0)
                    if sq_dsc_f is not None and sq_dsc_f >= overlap_threshold:
                        large_tp += 1
                    else:
                        large_fn += 1
                elif sq_dsc_f is not None:
                    if sq_dsc_f >= overlap_threshold:
                        small_tp += 1
                    else:
                        small_fn += 1
            elif is_large:
                large_fn += 1
                large_dsc.append(0.0)
                large_nsd.append(0.0)
                large_hd95.append(373.0)
            else:
                small_fn += 1

        num_fp = int(region_data.get("fp", 0) or 0)
        if num_fp > 0:
            large_dsc.extend([0.0] * num_fp)
            large_nsd.extend([0.0] * num_fp)
            large_hd95.extend([373.0] * num_fp)

        if not large_present:
            row[f"lesionwise_dsc_mean_{region}"] = math.nan
            row[f"lesionwise_nsd_mean_{region}"] = math.nan
            row[f"lesionwise_hd95_mean_{region}"] = math.nan
        else:
            row[f"lesionwise_dsc_mean_{region}"] = mean(large_dsc)
            row[f"lesionwise_nsd_mean_{region}"] = mean(large_nsd)
            row[f"lesionwise_hd95_mean_{region}"] = mean(large_hd95)

        if small_found:
            row[f"small_instance_tp_{region}"] = small_tp
            row[f"small_instance_fn_{region}"] = small_fn
            row[f"small_instance_fp_{region}"] = num_fp
            row[f"small_instance_f1_{region}"] = f1(small_tp, num_fp, small_fn)
        else:
            row[f"small_instance_tp_{region}"] = math.nan
            row[f"small_instance_fn_{region}"] = math.nan
            row[f"small_instance_fp_{region}"] = math.nan
            row[f"small_instance_f1_{region}"] = math.nan

        row[f"large_instance_tp_{region}"] = large_tp
        row[f"large_instance_fn_{region}"] = large_fn
        row[f"large_instance_fp_{region}"] = num_fp
        row[f"large_instance_f1_{region}"] = f1(large_tp, num_fp, large_fn)

    return row


def write_rows(rows: list[dict[str, object]], output_csv: Path) -> None:
    extra = []
    for key in rows[0].keys() if rows else []:
        if key not in {"subject_id", *LEADERBOARD_COLUMNS}:
            extra.append(key)
    fieldnames = ["subject_id", *LEADERBOARD_COLUMNS, *sorted(extra)]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_json(json_path: Path, output_csv: Path, vol_threshold: float, overlap_threshold: float) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for subject_data in data.get("metrics", []):
        if not isinstance(subject_data, dict) or "error" in subject_data:
            continue
        rows.append(parse_subject(subject_data, vol_threshold, overlap_threshold))

    if not rows:
        raise SystemExit("no valid subject metrics found")

    summary_rows = []
    numeric_columns = [key for row in rows for key, value in row.items() if key != "subject_id" and isinstance(value, (int, float))]
    numeric_columns = sorted(set(numeric_columns))
    for name, func in [("mean", mean), ("std", std), ("median", med)]:
        summary = {"subject_id": name}
        for column in numeric_columns:
            summary[column] = func([float(row.get(column, math.nan)) for row in rows])
        summary_rows.append(summary)

    write_rows(rows + summary_rows, output_csv)


def validate_csv(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
    missing = [column for column in LEADERBOARD_COLUMNS if column not in columns]
    if missing:
        raise SystemExit("missing official Task1 leaderboard columns:\n" + "\n".join(missing))
    print("CSV contains all BraTS 2026 Task1 leaderboard columns.")


def main() -> None:
    parser = argparse.ArgumentParser(description="BraTS 2026 Task1 MET leaderboard parser/validator.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_cmd = subparsers.add_parser("parse-json", help="Parse BraTS_evaluation Panoptica JSON into Task1 leaderboard columns.")
    parse_cmd.add_argument("--json-path", required=True)
    parse_cmd.add_argument("--output-csv", required=True)
    parse_cmd.add_argument("--vol-threshold", type=float, default=27.0)
    parse_cmd.add_argument("--overlap-threshold", type=float, default=0.2)

    validate_cmd = subparsers.add_parser("validate-csv", help="Check that a CSV has all official Task1 leaderboard columns.")
    validate_cmd.add_argument("--csv-path", required=True)

    args = parser.parse_args()
    if args.command == "parse-json":
        parse_json(Path(args.json_path), Path(args.output_csv), args.vol_threshold, args.overlap_threshold)
    elif args.command == "validate-csv":
        validate_csv(Path(args.csv_path))


if __name__ == "__main__":
    main()
