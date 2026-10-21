#!/usr/bin/env python3
"""Create deterministic patient-grouped G2 train/val/test splits.

The authoritative split unit is the patient group obtained by removing the
final numeric suffix from a BraTS-MET case ID. When authentic T2W case IDs are
provided as the anchor, their assignment reproduces the G1 V3 seed-42 split;
groups containing only fake/broken T2W cases are then assigned without moving
the anchored groups.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
DEFAULT_SEED = "42"


def boolish(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def patient_group(case_id: str) -> str:
    """Keep all BraTS-MET-xxxxx-yyy records from one patient together."""
    prefix, separator, suffix = str(case_id).rpartition("-")
    return prefix if separator and suffix.isdigit() else str(case_id)


def seed_value(seed: str) -> int | str:
    text = str(seed).strip()
    return int(text) if text.lstrip("-").isdigit() else text


def read_mapping(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
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


def display_result_path(path: Path, results_root: Path) -> str:
    try:
        return path.relative_to(results_root).as_posix()
    except ValueError:
        return path.as_posix()


def grouped_source_ids(source_ids: Iterable[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for source_id in sorted_unique(source_ids):
        grouped[patient_group(source_id)].append(source_id)
    return dict(grouped)


def assign_g1_v3_anchor_groups(
    grouped_ids: dict[str, list[str]],
    seed: str,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, str]:
    """Mirror G1 V3: shuffled groups, test first, then val, then train."""
    groups = sorted(grouped_ids)
    random.Random(seed_value(seed)).shuffle(groups)
    total_cases = sum(len(grouped_ids[group]) for group in groups)
    target_test = max(1, round(total_cases * test_fraction)) if test_fraction else 0
    target_val = max(1, round(total_cases * val_fraction)) if val_fraction else 0
    counts: Counter[str] = Counter()
    assignments: dict[str, str] = {}
    for group in groups:
        if counts["test"] < target_test:
            split_name = "test"
        elif counts["val"] < target_val:
            split_name = "val"
        else:
            split_name = "train"
        assignments[group] = split_name
        counts[split_name] += len(grouped_ids[group])
    return assignments


def assign_unanchored_groups(
    grouped_ids: dict[str, list[str]],
    assignments: dict[str, str],
    seed: str,
    val_fraction: float,
    test_fraction: float,
) -> None:
    """Assign missing-only groups while preserving every anchored assignment."""
    total_cases = sum(len(case_ids) for case_ids in grouped_ids.values())
    target_test = round(total_cases * test_fraction)
    target_val = round(total_cases * val_fraction)
    counts: Counter[str] = Counter()
    for group, split_name in assignments.items():
        counts[split_name] += len(grouped_ids[group])

    unassigned = sorted(set(grouped_ids) - set(assignments))
    random.Random(f"{seed}:unanchored").shuffle(unassigned)
    for group in unassigned:
        if counts["test"] < target_test:
            split_name = "test"
        elif counts["val"] < target_val:
            split_name = "val"
        else:
            split_name = "train"
        assignments[group] = split_name
        counts[split_name] += len(grouped_ids[group])


def assignments_from_base_split(
    base_split: list[dict[str, list[str]]],
    nn_to_source: dict[str, str],
) -> dict[str, str]:
    """Convert a legacy case split to groups, giving holdout membership priority."""
    anchor = base_split[0]
    holdout_name = "test" if "test" in anchor else "val"
    holdout_ids = set(anchor.get(holdout_name, []))
    unknown = sorted(holdout_ids - set(nn_to_source))
    if unknown:
        raise ValueError(f"base holdout contains unknown IDs: {unknown[:10]}")
    assignments = {
        patient_group(nn_to_source[nn_id]): "test"
        for nn_id in holdout_ids
    }
    return assignments


def validate_patient_group_split(
    split_by_source: dict[str, str],
) -> dict[str, int]:
    group_splits: dict[str, set[str]] = defaultdict(set)
    for source_id, split_name in split_by_source.items():
        group_splits[patient_group(source_id)].add(split_name)
    leaking = {group: values for group, values in group_splits.items() if len(values) > 1}
    if leaking:
        preview = ", ".join(f"{group}:{sorted(values)}" for group, values in list(leaking.items())[:10])
        raise ValueError(f"patient-group split leakage detected: {preview}")
    return dict(Counter(next(iter(values)) for values in group_splits.values()))


def create_train_val_test_split(
    mapping_rows: list[dict[str, str]],
    base_split: list[dict[str, list[str]]] | None = None,
    val_fraction_of_train_pool: float = 0.10,
    test_fraction: float = 0.10,
    seed: str = DEFAULT_SEED,
    anchor_case_ids: set[str] | None = None,
) -> dict[str, object]:
    """Create a split with zero patient-group overlap.

    ``val_fraction_of_train_pool`` is retained for CLI compatibility, but in the
    patient-group master policy it is the requested fraction of all cases.
    """
    val_fraction = val_fraction_of_train_pool
    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("validation and test fractions must be in [0, 1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError("validation fraction + test fraction must be less than 1")

    nn_to_source = {row["nnunet_case_id"]: row["source_case_id"] for row in mapping_rows}
    if len(nn_to_source) != len(mapping_rows):
        raise ValueError("mapping CSV contains duplicate nnU-Net IDs")
    source_to_nn = {source_id: nn_id for nn_id, source_id in nn_to_source.items()}
    if len(source_to_nn) != len(mapping_rows):
        raise ValueError("mapping CSV contains duplicate source case IDs")
    if not source_to_nn:
        raise ValueError("mapping CSV contains no cases")

    grouped_all = grouped_source_ids(source_to_nn)
    if base_split:
        assignments = assignments_from_base_split(base_split, nn_to_source)
        policy = "legacy_holdout_expanded_to_patient_groups_then_balanced"
    else:
        anchor_sources = set(anchor_case_ids) if anchor_case_ids is not None else set(source_to_nn)
        unknown_anchor = sorted(anchor_sources - set(source_to_nn))
        if unknown_anchor:
            raise ValueError(f"anchor contains unknown source case IDs: {unknown_anchor[:10]}")
        anchor_groups = grouped_source_ids(anchor_sources)
        assignments = assign_g1_v3_anchor_groups(
            anchor_groups,
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
        policy = (
            "g1_v3_seed42_authentic_anchor_then_patient_group_balanced_missing"
            if anchor_case_ids is not None
            else "patient_group_seeded_train_val_test"
        )

    assign_unanchored_groups(
        grouped_all,
        assignments,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    split_by_source = {
        source_id: assignments[patient_group(source_id)]
        for source_id in source_to_nn
    }
    group_counts = validate_patient_group_split(split_by_source)
    split_ids = {
        name: sorted(source_to_nn[source_id] for source_id, value in split_by_source.items() if value == name)
        for name in ("train", "val", "test")
    }
    coverage = set().union(*map(set, split_ids.values()))
    if coverage != set(nn_to_source):
        raise ValueError("split coverage mismatch")
    if any(not split_ids[name] for name in ("train", "val", "test")):
        raise ValueError("train, val, and test must all be non-empty")

    return {
        "name": "master_patient_group_train_val_test",
        "policy": policy,
        "seed": seed,
        "patient_group_rule": "remove_final_numeric_case_suffix",
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "anchor_case_count": len(anchor_case_ids) if anchor_case_ids is not None else len(mapping_rows),
        "counts": {name: len(split_ids[name]) for name in ("train", "val", "test")},
        "patient_group_counts": {name: group_counts.get(name, 0) for name in ("train", "val", "test")},
        "train": split_ids["train"],
        "val": split_ids["val"],
        "test": split_ids["test"],
    }


def filter_split(
    split: dict[str, object],
    allowed_nnunet_ids: set[str],
    name: str,
) -> dict[str, object]:
    filtered = dict(split)
    filtered["name"] = name
    for split_name in ("train", "val", "test"):
        filtered[split_name] = [
            nn_id for nn_id in split[split_name]  # type: ignore[index]
            if str(nn_id) in allowed_nnunet_ids
        ]
    filtered["counts"] = {
        split_name: len(filtered[split_name])  # type: ignore[arg-type]
        for split_name in ("train", "val", "test")
    }
    filtered["derived_from"] = split.get("name", "")
    return filtered


def membership_rows(split: dict[str, object], mapping_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    split_by_case: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for nn_id in split[split_name]:  # type: ignore[index]
            split_by_case[str(nn_id)] = split_name

    rows: list[dict[str, object]] = []
    for row in sorted(mapping_rows, key=lambda item: item["nnunet_case_id"]):
        nn_id = row["nnunet_case_id"]
        if nn_id not in split_by_case:
            continue
        source_case_id = row["source_case_id"]
        rows.append({
            "nnunet_case_id": nn_id,
            "source_case_id": source_case_id,
            "patient_group": patient_group(source_case_id),
            "split": split_by_case[nn_id],
            "t2w_status": row.get("t2w_status", "authentic"),
            "eligible_for_realonly": row.get("eligible_for_realonly", "True"),
            "stable_score_val": f"{stable_score(patient_group(source_case_id), str(split['seed']) + ':val'):.12f}",
            "stable_score_test": f"{stable_score(patient_group(source_case_id), str(split['seed']) + ':test'):.12f}",
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
        "patient_group",
        "split",
        "t2w_status",
        "eligible_for_realonly",
        "stable_score_val",
        "stable_score_test",
        "split_policy",
        "split_seed",
    ]
    with membership_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--mapping-csv", default="")
    parser.add_argument("--base-split-json", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--membership-csv", default="")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    mapping_csv = Path(args.mapping_csv) if args.mapping_csv else results_root / "manifests" / "nnunet_case_mapping_master.csv"
    base_split_json = Path(args.base_split_json) if args.base_split_json else None
    output_json = Path(args.output_json) if args.output_json else results_root / "splits" / "splits_master_train_val_test.json"
    membership_csv = Path(args.membership_csv) if args.membership_csv else results_root / "splits" / "splits_master_train_val_test_membership.csv"

    mapping_rows = read_mapping(mapping_csv)
    base_split = read_split(base_split_json) if base_split_json and base_split_json.exists() else None
    anchor_case_ids = {
        row["source_case_id"]
        for row in mapping_rows
        if boolish(row.get("eligible_for_realonly", "True"))
    }
    split = create_train_val_test_split(
        mapping_rows,
        base_split=base_split,
        val_fraction_of_train_pool=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        anchor_case_ids=anchor_case_ids,
    )
    split["mapping_csv"] = display_result_path(mapping_csv, results_root)
    write_split_outputs(split, mapping_rows, output_json, membership_csv)
    print(json.dumps({"counts": split["counts"], "patient_group_counts": split["patient_group_counts"]}, indent=2))
    print(f"split_json={output_json}")
    print(f"membership_csv={membership_csv}")


if __name__ == "__main__":
    main()
