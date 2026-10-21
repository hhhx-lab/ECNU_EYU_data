"""Fail-closed split selection shared by training, evaluation and list export."""
import re


def patient_group(case_id):
    value = str(case_id).strip()
    # BraTS-MET-00001-000 and 00001-001 must resolve to the same patient.
    match = re.fullmatch(r"(?:BraTS-[A-Za-z]+-)?(\d+)-\d{3}", value, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.fullmatch(r"BraTS-[A-Za-z]+-(\d+)", value, re.IGNORECASE)
    return match.group(1) if match else value


def validate_group_splits(groups_by_split):
    owners = {}
    for split, cases in groups_by_split.items():
        for case in cases:
            group = patient_group(case)
            if not group or group.lower() == "nan":
                raise ValueError("Missing patient identity")
            if group in owners and owners[group] != split:
                raise ValueError(f"Patient {group} occurs in both {owners[group]} and {split}")
            owners[group] = split


def select_split(df, split):
    if "split" not in df.columns:
        if split != "all":
            raise ValueError("CSV requires a split column; rebuild it with csv_creator.py")
        return df.copy()
    if "patient_id" not in df.columns or df["patient_id"].isna().any():
        raise ValueError("CSV requires non-empty patient_id values")
    if df["split"].isna().any() or not df["split"].isin(["train", "val", "test"]).all():
        raise ValueError("CSV split values must be train, val or test")
    validate_group_splits({key: rows["patient_id"].tolist() for key, rows in df.groupby("split")})
    selected = df if split == "all" else df[df["split"] == split]
    if selected.empty:
        raise ValueError(f"No rows with split == {split!r}")
    return selected.copy().reset_index(drop=True)
