import argparse
import csv
import json
import random
from pathlib import Path


MODALITY_COLUMNS = [
    "t1n_source_path",
    "t1c_source_path",
    "t2w_source_path",
    "t2f_source_path",
]


def resolve_project_path(path, project_root):
    path = Path(path).expanduser()
    if path.is_absolute():
        return str(path)
    return str(Path(project_root) / path)


def row_to_record(row, fold, project_root):
    return {
        "image": [resolve_project_path(row[column], project_root) for column in MODALITY_COLUMNS],
        "label": resolve_project_path(row["seg_source_path"], project_root),
        "fold": fold,
        "case_id": row["source_case_id"],
        "nnunet_case_id": row["nnunet_case_id"],
    }


def load_g2_split(split_json, mapping_csv, project_root):
    with open(split_json) as f:
        data = json.load(f)
    split = data[0] if isinstance(data, list) else data

    with open(mapping_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    nnunet_to_row = {row["nnunet_case_id"]: row for row in rows}

    def convert(ids, split_name, fold):
        missing = [case_id for case_id in ids if case_id not in nnunet_to_row]
        if missing:
            raise SystemExit(
                f"{split_name} ids missing from mapping: {missing[:10]} ... total={len(missing)}"
            )
        return [row_to_record(nnunet_to_row[case_id], fold, project_root) for case_id in ids]

    test_cases = [
        nnunet_to_row[case_id]["source_case_id"]
        for case_id in split.get("test", [])
        if case_id in nnunet_to_row
    ]
    return convert(split["train"], "train", 1), convert(split["val"], "val", 0), test_cases


def make_random_split(data_root, val_ratio, seed):
    data_root = Path(data_root)
    cases = sorted(
        d.name
        for d in data_root.iterdir()
        if d.is_dir() and d.name.startswith("BraTS-MET-")
    )
    random.seed(seed)
    random.shuffle(cases)
    split_idx = int((1.0 - val_ratio) * len(cases))
    return cases[:split_idx], cases[split_idx:], []


def main():
    parser = argparse.ArgumentParser(
        description="Create S3 Swin UNETR train/val split JSON."
    )
    parser.add_argument("--data-dir", default="", help="Raw BraTS-MET case root. Required for random mode.")
    parser.add_argument("--output", default="full_split.json", help="Output JSON path.")
    parser.add_argument("--g2-split-json", default="", help="G2 fixed train/val/test split JSON.")
    parser.add_argument("--g2-mapping-csv", default="", help="G2 nnU-Net case mapping CSV.")
    parser.add_argument("--project-root", default="", help="Project root used to resolve relative mapping paths.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio for random fallback.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for random fallback.")
    args = parser.parse_args()

    if args.g2_split_json and args.g2_mapping_csv:
        project_root = args.project_root or str(Path(args.g2_mapping_csv).resolve().parents[4])
        train_records, val_records, test_cases = load_g2_split(args.g2_split_json, args.g2_mapping_csv, project_root)
        mode = "g2_fixed_split"
    else:
        if not args.data_dir:
            raise SystemExit("--data-dir is required when G2 split/mapping are not provided.")
        train_cases, val_cases, test_cases = make_random_split(args.data_dir, args.val_ratio, args.seed)
        train_records = [{"image": c, "label": c, "fold": 1} for c in train_cases]
        val_records = [{"image": c, "label": c, "fold": 0} for c in val_cases]
        mode = "random_fallback"

    # data_utils.py uses fold=0 as validation and all other folds as training.
    data = train_records + val_records
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(
            {
                "mode": mode,
                "training": data,
                "internal_locked_test": test_cases,
            },
            f,
            indent=2,
        )

    print(f"mode: {mode}")
    print(f"train: {len(train_records)}")
    print(f"val: {len(val_records)}")
    print(f"internal locked test: {len(test_cases)}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
