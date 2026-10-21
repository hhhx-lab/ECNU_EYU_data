"""Export train / val / test case lists (source IDs such as BraTS-MET-00001-000) from the S2 split.

The S2 split JSON (used by S2 scripts/04_build_fixed_split.py) has the form
    {"train": [...], "val": [...], "test": [...]}      (or a one-element list of such an object)
with nnU-Net case IDs. The S2 mapping CSV (columns nnunet_case_id, source_case_id) maps them back
to the raw BraTS case folder names. If the split already contains source IDs, omit --mapping_csv.

Usage (run from TumourDiffusion_1mm/):
    python scripts/make_case_lists.py --split_json /path/to/split.json \
        --mapping_csv /path/to/mapping.csv --out_dir lists
Writes lists/train_cases.txt, lists/val_cases.txt, lists/test_cases.txt
"""

import argparse
import csv
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--mapping_csv", default="")
    ap.add_argument("--out_dir", default="lists")
    args = ap.parse_args()

    with open(args.split_json, encoding="utf-8") as f:
        data = json.load(f)
    split = data[0] if isinstance(data, list) else data
    mapping = None
    if args.mapping_csv:
        with open(args.mapping_csv, encoding="utf-8-sig", newline="") as f:
            mapping = {r["nnunet_case_id"].strip(): r["source_case_id"].strip() for r in csv.DictReader(f)}

    os.makedirs(args.out_dir, exist_ok=True)
    seen = {}
    for name in ("train", "val", "test"):
        ids = [str(v).strip() for v in split.get(name, []) if str(v).strip()]
        if mapping is not None:
            missing = [i for i in ids if i not in mapping]
            if missing:
                raise SystemExit(f"{len(missing)} {name} IDs missing from the mapping, e.g. {missing[:3]}")
            ids = [mapping[i] for i in ids]
        for i in ids:
            if i in seen:
                raise SystemExit(f"{i} appears in both '{seen[i]}' and '{name}'")
            seen[i] = name
        path = os.path.join(args.out_dir, f"{name}_cases.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {name} cases exported from {os.path.basename(args.split_json)}\n")
            f.write("\n".join(sorted(ids)) + "\n")
        print(f"{name}: {len(ids)} cases -> {path}")


if __name__ == "__main__":
    main()
