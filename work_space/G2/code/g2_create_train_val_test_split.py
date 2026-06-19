#!/usr/bin/env python3
"""Create the locked G2 train/val/test split for BraTS 2026 Task1.

Default policy:
1. Use the existing G2 two-way fold as the anchor.
2. Treat its old `val` list as the locked internal test set.
3. Split the old `train` pool into train and dev/val by a stable hash.

This keeps the historical 259-case holdout untouched while giving G1/S1/S2 a
separate validation set for tuning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable


DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
DEFAULT_SEED = "20260619"


def read_mapping(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"nnunet_case_id", "source_case_id"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"mapping CSV missing required columns: {sorted(missing)}")
    return rows


def read_split(path: Path) -> list[dict[str, list[str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise ValueError(f"split JSON must be a dict or a list with one dict: {path}")
    return data


def stable_score(source_case_id: str, seed: str) -> float:
    digest = hashlib.sha256(f"{seed}::{source_case_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def sorted_unique(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value).strip()})


def _validate_known_ids(ids: set[str], known_ids: set[str], label: str) -> None:
    unknown = sorted(ids - known_ids)
    if unknown:
        preview = ", ".join(unknown[:10])
        raise ValueError(f"{label} contains {len(unknown)} ids not present in mapping CSV: {preview}")


def create_train_val_test_split(
    mapping_rows: list[dict[str, str]],
    base_split: list[dict[str, list[str]]] | None = None,
    val_fraction_of_train_pool: float = 0.2,
    test_fraction: float = 0.2,
    seed: str = DEFAULT_SEED,
) -> dict[str, object]:
    if not 0 <= val_fraction_of_train_pool < 1:
        raise ValueError("--val-fraction-of-train-pool must be in [0, 1)")
    if not 0 <= test_fraction < 1:
        raise ValueError("--test-fraction must be in [0, 1)")

    nn_to_source = {row["nnunet_case_id"]: row["source_case_id"] for row in mapping_rows}
    all_ids = set(nn_to_source)
    if not all_ids:
        raise ValueError("mapping CSV contains no cases")

    source_split_json = ""
    if base_split:
        anchor = base_split[0]
        base_train = set(anchor.get("train", []))
        _validate_known_ids(base_train, all_ids, "base train split")
        if "test" in anchor:
            test_ids = set(anchor.get("test", []))
            policy = "existing_test_locked_then_hash_val_from_train"
        else:
            test_ids = set(anchor.get("val", []))
            policy = "legacy_fixed_val_locked_as_internal_test_then_hash_val_from_train"
        _validate_known_ids(test_ids, all_ids, "base test/val split")
        train_pool = base_train - test_ids
        missing_from_anchor = all_ids - base_train - test_ids
        if missing_from_anchor:
            # Keep newly added cases in the tunable training pool instead of silently dropping them.
            train_pool |= missing_from_anchor
    else:
        scored_all = sorted(
            (stable_score(nn_to_source[nn_id], f"{seed}:test"), nn_id)
            for nn_id in all_ids
        )
        test_count = int(round(len(scored_all) * test_fraction))
        test_ids = {nn_id for _, nn_id in scored_all[:test_count]}
        train_pool = all_ids - test_ids
        policy = "hash_test_then_hash_val_from_remaining"

    scored_train_pool = sorted(
        (stable_score(nn_to_source[nn_id], f"{seed}:val"), nn_id)
        for nn_id in train_pool
    )
    val_count = int(round(len(scored_train_pool) * val_fraction_of_train_pool))
    val_ids = {nn_id for _, nn_id in scored_train_pool[:val_count]}
    train_ids = set(train_pool) - val_ids

    overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    if overlap:
        raise ValueError(f"split overlap detected: {sorted(overlap)[:10]}")
    coverage = train_ids | val_ids | test_ids
    if coverage != all_ids:
        missing = sorted(all_ids - coverage)
        extra = sorted(coverage - all_ids)
        raise ValueError(f"split coverage mismatch; missing={missing[:10]}, extra={extra[:10]}")

    return {
        "name": "fold0_train_val_test",
        "policy": policy,
        "seed": seed,
        "val_fraction_of_train_pool": val_fraction_of_train_pool,
        "test_fraction": "" if base_split else test_fraction,
        "source_split_json": source_split_json,
        "mapping_csv": "",
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "train": sorted_unique(train_ids),
        "val": sorted_unique(val_ids),
        "test": sorted_unique(test_ids),
    }


def membership_rows(split: dict[str, object], mapping_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    split_by_case: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for nn_id in split[split_name]:  # type: ignore[index]
            split_by_case[str(nn_id)] = split_name

    rows: list[dict[str, object]] = []
    for row in sorted(mapping_rows, key=lambda item: item["nnunet_case_id"]):
        nn_id = row["nnunet_case_id"]
        source_case_id = row["source_case_id"]
        rows.append({
            "nnunet_case_id": nn_id,
            "source_case_id": source_case_id,
            "split": split_by_case[nn_id],
            "stable_score_val": f"{stable_score(source_case_id, str(split['seed']) + ':val'):.12f}",
            "stable_score_test": f"{stable_score(source_case_id, str(split['seed']) + ':test'):.12f}",
            "split_policy": split["policy"],
            "split_seed": split["seed"],
        })
    return rows


def write_split_outputs(
    split: dict[str, object],
    mapping_rows: list[dict[str, str]],
    output_json: Path,
    membership_csv: Path,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    membership_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps([split], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rows = membership_rows(split, mapping_rows)
    fieldnames = [
        "nnunet_case_id",
        "source_case_id",
        "split",
        "stable_score_val",
        "stable_score_test",
        "split_policy",
        "split_seed",
    ]
    with membership_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create G2 train/val/test split artifacts.")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--mapping-csv", default="")
    parser.add_argument("--base-split-json", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--membership-csv", default="")
    parser.add_argument("--val-fraction-of-train-pool", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    mapping_csv = Path(args.mapping_csv) if args.mapping_csv else results_root / "manifests" / "nnunet_case_mapping_realonly.csv"
    base_split_json = Path(args.base_split_json) if args.base_split_json else results_root / "splits" / "splits_final_fold0_realval.json"
    output_json = Path(args.output_json) if args.output_json else results_root / "splits" / "splits_final_train_val_test.json"
    membership_csv = Path(args.membership_csv) if args.membership_csv else results_root / "splits" / "splits_final_train_val_test_membership.csv"

    mapping_rows = read_mapping(mapping_csv)
    base_split = read_split(base_split_json) if base_split_json.exists() else None
    split = create_train_val_test_split(
        mapping_rows,
        base_split=base_split,
        val_fraction_of_train_pool=args.val_fraction_of_train_pool,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    split["source_split_json"] = str(base_split_json) if base_split_json.exists() else ""
    split["mapping_csv"] = str(mapping_csv)
    write_split_outputs(split, mapping_rows, output_json, membership_csv)

    counts = split["counts"]
    print(f"train={counts['train']}")  # type: ignore[index]
    print(f"val={counts['val']}")  # type: ignore[index]
    print(f"test={counts['test']}")  # type: ignore[index]
    print(f"split_json={output_json}")
    print(f"membership_csv={membership_csv}")


if __name__ == "__main__":
    main()
