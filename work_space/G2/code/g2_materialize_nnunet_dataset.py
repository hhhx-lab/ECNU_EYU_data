#!/usr/bin/env python3
"""Materialize G2 real/synthetic manifests into an nnU-Net raw dataset.

Default mode is manifest-only so this script is safe to run on a laptop. For
G1 T2W completion rows, the default policy replaces the original fake/broken
T2W channel in the matching real case instead of appending a duplicate case. By
default it only uses completion/synthetic rows accepted for training; pass
--include-ablation-only explicitly if you want the controlled ablation rows too.
On the training machine, use --mode symlink or --mode copy after checking disk
space.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
DEFAULT_FAKE_T2W = DEFAULT_RESULTS_ROOT / "qc" / "official_fake_t2w_cases_by_gzip_header_2026-06-15.csv"
PROJECT_ROOT_NAME = "ECNU_EYU_data"
CHANNEL_ORDERS = {
    "g2_official": ["t1n", "t1c", "t2w", "t2f"],
    "s2_current": ["t1c", "t1n", "t2f", "t2w"],
}
LABELS = {"background": 0, "NETC": 1, "SNFH": 2, "ET": 3, "RC": 4}


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "work_space" / "G1").exists() and (parent / "work_space" / "G2").exists():
            return parent
    raise RuntimeError(f"Could not locate ECNU_EYU_data project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())


def parse_workspace_path(path_str: str | Path | None, anchor: Path | None = None) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    if anchor is not None:
        candidate = (anchor / path).resolve()
        if candidate.exists():
            return candidate
    candidate = (PROJECT_ROOT / path).resolve()
    if candidate.exists():
        return candidate
    return candidate


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def boolish(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_fake_t2w_cases(path: Path) -> set[str]:
    if not path.exists():
        return set()
    rows = read_csv(path)
    cases: set[str] = set()
    for row in rows:
        case_id = row.get("case_id") or row.get("subject_id") or row.get("id")
        if case_id:
            cases.add(case_id)
    return cases


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def source_for(row: dict[str, str], modality: str) -> str | None:
    candidates = [
        f"{modality}_source_path",
        f"normalized_{modality}_path",
        f"raw_{modality}_path",
    ]
    if modality == "seg":
        candidates = ["seg_source_path", "normalized_seg_path", "raw_seg_path"]
    values = [row.get(key, "") for key in candidates if row.get(key, "")]
    for value in values:
        resolved = parse_workspace_path(value)
        if resolved is not None and resolved.exists():
            return value
    return values[0] if values else None


def completion_replacement_source(row: dict[str, str], modality: str) -> str | None:
    if modality == "t2w":
        return source_for(row, "t2w")
    return None


def link_or_copy(src: Path | None, dst: Path, mode: str, overwrite: bool) -> str:
    if src is None:
        return "missing_source"
    if mode == "manifest-only":
        return "planned"
    if not src.exists():
        return "missing_source"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return "exists"
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"unsupported mode: {mode}")
    return mode


def is_completion_row(row: dict[str, str]) -> bool:
    return boolish(row.get("source_completion_mode", "")) or row.get("label_kind", "") == "completion"


def row_is_usable(row: dict[str, str], include_ablation_only: bool) -> bool:
    accepted = boolish(row.get("accepted_for_training", ""))
    ablation = boolish(row.get("accepted_for_ablation_only", ""))
    return accepted or (include_ablation_only and ablation)


def selected_completion_replacements(
    synthetic_rows: list[dict[str, str]],
    include_ablation_only: bool,
    completion_policy: str,
) -> dict[str, dict[str, str]]:
    if completion_policy == "append":
        return {}

    replacements: dict[str, dict[str, str]] = {}
    for row in synthetic_rows:
        if not is_completion_row(row) or not row_is_usable(row, include_ablation_only):
            continue
        source_case_id = row.get("source_case_id", "")
        if not source_case_id:
            continue
        t2w_source = source_for(row, "t2w")
        if not t2w_source:
            continue
        current = replacements.get(source_case_id)
        if current is None:
            replacements[source_case_id] = row
            continue
        # Prefer training-accepted rows over ablation-only rows if both exist.
        if boolish(row.get("accepted_for_training", "")) and not boolish(current.get("accepted_for_training", "")):
            replacements[source_case_id] = row
    return replacements


def write_case_files(
    row: dict[str, str],
    row_type: str,
    case_id: str,
    source_case_id: str,
    dataset_dir: Path,
    channel_order: list[str],
    mode: str,
    overwrite: bool,
    materialized: list[dict[str, object]],
    replacement_row: dict[str, str] | None = None,
) -> None:
    for channel_idx, modality in enumerate(channel_order):
        source_path = completion_replacement_source(replacement_row, modality) if replacement_row else None
        source_path = source_path or source_for(row, modality)
        src = parse_workspace_path(source_path, PROJECT_ROOT) if source_path else None
        dst = dataset_dir / "imagesTr" / f"{case_id}_{channel_idx:04d}.nii.gz"
        action = link_or_copy(src, dst, mode, overwrite)
        materialized.append(
            {
                "case_id": case_id,
                "source_case_id": source_case_id,
                "row_type": row_type,
                "modality": modality,
                "nnunet_channel": f"{channel_idx:04d}",
                "source_path": source_path or "",
                "target_path": str(dst),
                "action": action,
                "replacement_synthetic_raw_id": replacement_row.get("synthetic_raw_id", "") if replacement_row else "",
                "replacement_synthetic_final_id": replacement_row.get("synthetic_final_id", "") if replacement_row else "",
            }
        )

    source_path = source_for(row, "seg")
    src = parse_workspace_path(source_path, PROJECT_ROOT) if source_path else None
    dst = dataset_dir / "labelsTr" / f"{case_id}.nii.gz"
    action = link_or_copy(src, dst, mode, overwrite)
    materialized.append(
        {
            "case_id": case_id,
            "source_case_id": source_case_id,
            "row_type": row_type,
            "modality": "seg",
            "nnunet_channel": "label",
            "source_path": source_path or "",
            "target_path": str(dst),
            "action": action,
            "replacement_synthetic_raw_id": replacement_row.get("synthetic_raw_id", "") if replacement_row else "",
            "replacement_synthetic_final_id": replacement_row.get("synthetic_final_id", "") if replacement_row else "",
        }
    )


def build_rows(
    real_rows: list[dict[str, str]],
    synthetic_rows: list[dict[str, str]],
    fake_t2w_cases: set[str],
    dataset_dir: Path,
    channel_order: list[str],
    mode: str,
    overwrite: bool,
    include_ablation_only: bool,
    completion_policy: str,
    include_unreplaced_fake_t2w: bool,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    materialized: list[dict[str, object]] = []
    stats = {
        "included_cases": 0,
        "real_cases": 0,
        "replaced_completion_cases": 0,
        "appended_synthetic_cases": 0,
        "skipped_unreplaced_fake_t2w_cases": 0,
        "skipped_synthetic_rows": 0,
    }
    used_case_ids: set[str] = set()
    replacements = selected_completion_replacements(synthetic_rows, include_ablation_only, completion_policy)

    for row in real_rows:
        case_id = row.get("nnunet_case_id") or row.get("case_id") or row.get("synthetic_final_id") or row.get("source_case_id")
        if not case_id:
            continue
        source_case_id = row.get("source_case_id") or row.get("case_id") or case_id
        replacement_row = replacements.get(source_case_id)
        source_has_fake_t2w = source_case_id in fake_t2w_cases
        if source_has_fake_t2w and replacement_row is None and not include_unreplaced_fake_t2w:
            stats["skipped_unreplaced_fake_t2w_cases"] += 1
            continue

        row_type = "real_with_completion_t2w" if replacement_row else "real"
        write_case_files(row, row_type, case_id, source_case_id, dataset_dir, channel_order, mode, overwrite, materialized, replacement_row)
        stats["included_cases"] += 1
        stats["real_cases"] += 1
        if replacement_row:
            stats["replaced_completion_cases"] += 1
        used_case_ids.add(case_id)

    for row in synthetic_rows:
        if not row_is_usable(row, include_ablation_only):
            stats["skipped_synthetic_rows"] += 1
            continue
        if is_completion_row(row) and completion_policy in {"auto", "replace"}:
            continue
        case_id = row.get("nnunet_case_id") or row.get("case_id") or row.get("synthetic_final_id") or row.get("source_case_id")
        if not case_id:
            stats["skipped_synthetic_rows"] += 1
            continue
        if case_id in used_case_ids:
            raise SystemExit(f"duplicate nnU-Net case id during materialization: {case_id}")
        source_case_id = row.get("source_case_id") or case_id
        write_case_files(row, "synthetic", case_id, source_case_id, dataset_dir, channel_order, mode, overwrite, materialized)
        stats["included_cases"] += 1
        stats["appended_synthetic_cases"] += 1
        used_case_ids.add(case_id)

    return materialized, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an nnU-Net raw dataset from G2 real mapping plus accepted synthetic manifest."
    )
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--real-mapping", default="", help="real-only mapping CSV. Defaults to results/manifests/nnunet_case_mapping_realonly.csv")
    parser.add_argument("--synthetic-accepted-manifest", default="", help="Optional synthetic_accepted_manifest_<run_id>.csv")
    parser.add_argument("--fake-t2w-cases", default="", help="CSV listing source case ids whose real T2W is fake/broken. Defaults to G2 QC fake list.")
    parser.add_argument("--output-root", required=True, help="nnUNet_raw root or any destination root on the training machine.")
    parser.add_argument("--dataset-id", default="261")
    parser.add_argument("--dataset-name", default="BraTS2026_MET_RealSynth_G1")
    parser.add_argument("--channel-order", choices=sorted(CHANNEL_ORDERS), default="g2_official")
    parser.add_argument("--mode", choices=["manifest-only", "symlink", "hardlink", "copy"], default="manifest-only")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--completion-policy",
        choices=["auto", "replace", "append"],
        default="auto",
        help="auto/replace uses accepted G1 completion T2W to replace fake real T2W; append keeps completion rows as separate synthetic cases.",
    )
    parser.add_argument(
        "--include-unreplaced-fake-t2w",
        action="store_true",
        help="Also include real rows whose T2W is flagged fake/broken and has no accepted completion replacement. Default excludes them.",
    )
    parser.add_argument(
        "--allow-missing-fake-t2w-list",
        action="store_true",
        help="Continue if the fake/broken T2W case list is missing. Default fails fast to avoid leaking raw fake T2W.",
    )
    parser.add_argument(
        "--allow-missing-sources",
        action="store_true",
        help="Write manifest even if source NIfTI files are missing. Default fails fast for symlink/hardlink/copy modes.",
    )
    parser.add_argument(
        "--include-ablation-only",
        action="store_true",
        help="Also include accepted_for_ablation_only synthetic rows. Default excludes them from the main training dataset.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    real_mapping = Path(args.real_mapping) if args.real_mapping else results_root / "manifests" / "nnunet_case_mapping_realonly.csv"
    synthetic_manifest = Path(args.synthetic_accepted_manifest) if args.synthetic_accepted_manifest else None
    fake_t2w_path = Path(args.fake_t2w_cases) if args.fake_t2w_cases else results_root / "qc" / DEFAULT_FAKE_T2W.name
    output_root = Path(args.output_root).expanduser().resolve()
    dataset_dir = output_root / f"Dataset{args.dataset_id}_{args.dataset_name}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "imagesTr").mkdir(exist_ok=True)
    (dataset_dir / "labelsTr").mkdir(exist_ok=True)

    real_rows = read_csv(real_mapping)
    synthetic_rows = read_csv(synthetic_manifest) if synthetic_manifest else []
    if not fake_t2w_path.exists() and not args.allow_missing_fake_t2w_list:
        raise SystemExit(
            f"fake/broken T2W case list not found: {fake_t2w_path}. "
            "Refresh G2 audit or pass --allow-missing-fake-t2w-list for a controlled diagnostic run."
        )
    fake_t2w_cases = load_fake_t2w_cases(fake_t2w_path)
    channel_order = CHANNEL_ORDERS[args.channel_order]
    rows, stats = build_rows(
        real_rows,
        synthetic_rows,
        fake_t2w_cases,
        dataset_dir,
        channel_order,
        args.mode,
        args.overwrite,
        args.include_ablation_only,
        args.completion_policy,
        args.include_unreplaced_fake_t2w,
    )

    dataset_json = {
        "channel_names": {str(idx): modality for idx, modality in enumerate(channel_order)},
        "labels": LABELS,
        "numTraining": stats["included_cases"],
        "file_ending": ".nii.gz",
        "g2_channel_order": args.channel_order,
        "g2_materialization_mode": args.mode,
        "g2_include_ablation_only": args.include_ablation_only,
        "g2_completion_policy": args.completion_policy,
        "g2_include_unreplaced_fake_t2w": args.include_unreplaced_fake_t2w,
        "g2_real_mapping": str(real_mapping),
        "g2_synthetic_manifest": str(synthetic_manifest) if synthetic_manifest else "",
        "g2_fake_t2w_cases": str(fake_t2w_path),
        "g2_materialization_stats": stats,
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fieldnames = [
        "case_id",
        "source_case_id",
        "row_type",
        "modality",
        "nnunet_channel",
        "source_path",
        "target_path",
        "action",
        "replacement_synthetic_raw_id",
        "replacement_synthetic_final_id",
    ]
    write_csv(dataset_dir / "g2_materialization_manifest.csv", rows, fieldnames)
    missing_source_rows = [row for row in rows if row["action"] == "missing_source"]
    if missing_source_rows and args.mode != "manifest-only" and not args.allow_missing_sources:
        preview = ", ".join(
            f"{row['case_id']}:{row['modality']}->{row['source_path']}" for row in missing_source_rows[:10]
        )
        raise SystemExit(
            "missing source NIfTI files detected during nnU-Net materialization; "
            f"fix paths or pass --allow-missing-sources for diagnostics. examples: {preview}"
        )

    print(f"dataset_dir={dataset_dir}")
    print(f"numTraining={stats['included_cases']}")
    print(f"channel_order={args.channel_order}:{','.join(channel_order)}")
    print(f"mode={args.mode}")
    print(f"include_ablation_only={args.include_ablation_only}")
    print(f"completion_policy={args.completion_policy}")
    print(f"fake_t2w_cases={len(fake_t2w_cases)}")
    print(f"replaced_completion_cases={stats['replaced_completion_cases']}")
    print(f"skipped_unreplaced_fake_t2w_cases={stats['skipped_unreplaced_fake_t2w_cases']}")
    print(f"appended_synthetic_cases={stats['appended_synthetic_cases']}")
    print(f"missing_source_files={len(missing_source_rows)}")


if __name__ == "__main__":
    main()
