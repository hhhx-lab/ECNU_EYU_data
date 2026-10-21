"""Reconstruct frozen G1 VAE images into a separate native-space dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.bundle import ConfigParser


MODALITIES = ("t1n", "t1c", "t2w", "t2f")
EXPECTED_CHECKPOINT_SHA = "5d86a0b3cbc5df34d64d5a0ff376eae453634a609bd24dec1a1191fde800b13b"
META = {
    "artifact_status": "experimental_unvalidated",
    "operator_approved": False,
    "formal_gate_status": "not_run_not_passed",
}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def read_rows(path: Path, selected_ids: tuple[str, ...] | None) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1030 and len({row["id"] for row in rows}) == 1030
    assert Counter(row["split"] for row in rows) == {"train": 823, "val": 103, "test": 104}
    if selected_ids is not None:
        rows = [row for row in rows if row["id"] in selected_ids]
        if {row["id"] for row in rows} != set(selected_ids):
            raise ValueError("smoke case ids not in frozen CSV")
    return rows


def load_vae(checkpoint_path: Path, code_root: Path):
    if digest(checkpoint_path) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("epoch004 checkpoint SHA drift")
    sys.path.insert(0, str(code_root))
    import configs

    parser = ConfigParser(configs.NETWORKS_CONFIG)
    parser.parse(True)
    model = parser.get_parsed_content("autoencoder_def")
    trusted = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(trusted, dict) or trusted.get("epoch") != 4:
        raise RuntimeError("epoch004 checkpoint structure drift")
    model.load_state_dict(trusted["model_state_dict"], strict=True)
    model.eval().to("cuda:0")
    return model


def reconstruct_one(row: dict, data_root: Path, output_root: Path, model) -> dict:
    import synthesis.utils as utils
    from synthesis.spatial import build_spatial_transform, resample_to_model, restore_to_native

    case = row["id"]
    source_dir = data_root / case
    sources = {m: source_dir / Path(row[m]).name for m in MODALITIES}
    sources["seg"] = source_dir / Path(row["seg"]).name
    if not all(path.is_file() for path in sources.values()):
        raise FileNotFoundError(f"source missing for {case}")
    final_dir = output_root / "cases" / case
    temporary_dir = output_root / "cases" / ("." + case + ".partial")
    if final_dir.exists() or temporary_dir.exists():
        raise FileExistsError(f"case already exists: {case}")
    temporary_dir.mkdir(parents=True)
    images = {m: nib.load(str(sources[m])) for m in MODALITIES}
    ref = images["t1n"]
    source_seg = nib.load(str(sources["seg"]))
    for image in [*images.values(), source_seg]:
        if image.shape != ref.shape or not np.allclose(image.affine, ref.affine, atol=1e-4, rtol=0):
            raise ValueError(f"source geometry drift: {case}")
    seg = np.asanyarray(source_seg.dataobj)
    if not np.isfinite(seg).all() or not np.allclose(seg, np.rint(seg), atol=1e-6):
        raise ValueError(f"invalid source label: {case}")
    label_values = sorted(int(x) for x in np.unique(seg))
    if not set(label_values).issubset({0, 1, 2, 3, 4}):
        raise ValueError(f"unexpected label values: {case}: {label_values}")
    raw = {m: images[m].get_fdata(dtype=np.float32) for m in MODALITIES}
    if any(not np.isfinite(x).all() for x in raw.values()):
        raise ValueError(f"nonfinite source image: {case}")
    normalized = {m: utils.robust_normalize(raw[m]) for m in MODALITIES}
    transform = build_spatial_transform(
        [normalized[m] for m in ("t1n", "t1c", "t2f")],
        ref.affine, segmentation=np.rint(seg).astype(np.int16),
        target_shape=(256, 256, 160),
    )
    output_files = {}
    with torch.inference_mode():
        for m in MODALITIES:
            image = resample_to_model(normalized[m], transform, order=1)
            tensor = torch.from_numpy(image).to("cuda:0").unsqueeze(0).unsqueeze(0)
            mu, _ = model.encode(tensor)
            reconstructed = model.decode(mu).squeeze().float().cpu().numpy()
            del tensor, mu
            native = restore_to_native(reconstructed, transform, order=1)
            native = utils.postprocess_intensity(native, m).astype(np.float32)
            if native.shape != ref.shape or not np.isfinite(native).all() or not np.any(native):
                raise ValueError(f"invalid reconstruction: {case}/{m}")
            header = images[m].header.copy()
            header.set_data_dtype(np.float32)
            target = temporary_dir / f"{case}-{m}.nii.gz"
            nib.save(nib.Nifti1Image(native, images[m].affine, header), str(target))
            reopened = nib.load(str(target))
            if reopened.shape != ref.shape or not np.allclose(reopened.affine, images[m].affine, atol=1e-4, rtol=0):
                raise ValueError(f"output geometry drift: {case}/{m}")
            output_files[m] = {"name": target.name, "sha256": digest(target), "bytes": target.stat().st_size}
    label_target = temporary_dir / f"{case}-seg.nii.gz"
    shutil.copy2(sources["seg"], label_target)
    if digest(label_target) != digest(sources["seg"]):
        raise ValueError(f"label copy drift: {case}")
    output_files["seg"] = {"name": label_target.name, "sha256": digest(label_target), "bytes": label_target.stat().st_size}
    record = {
        **META, "case_id": case, "split": row["split"],
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "normalization": "G1_V3_robust_normalize_then_deployed_modality_postprocess",
        "latent_sampling": "posterior_mean", "geometry": "native",
        "native_shape": list(ref.shape), "label_values": label_values,
        "source_sha256": {m: digest(p) for m, p in sources.items()},
        "files": output_files,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(temporary_dir / "case_manifest.json", record)
    os.rename(temporary_dir, final_dir)
    print(json.dumps({"case": case, "split": row["split"], "status": "pass", "files": 5}), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-id", action="append")
    args = parser.parse_args()
    selected = tuple(args.case_id) if args.case_id else None
    rows = read_rows(args.source_csv, selected)
    if args.output_root.exists():
        raise FileExistsError(f"output root already exists: {args.output_root}")
    args.output_root.mkdir(parents=True)
    atomic_json(args.output_root / "RUN_CONTRACT.json", {
        **META, "status": "started", "source_csv_sha256": digest(args.source_csv),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "cases_expected": len(rows), "smoke": selected is not None,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    model = load_vae(args.checkpoint, args.code_root)
    records = [reconstruct_one(row, args.data_root, args.output_root, model) for row in rows]
    expected_splits = Counter(row["split"] for row in rows)
    observed = Counter(record["split"] for record in records)
    if observed != expected_splits or len(records) != len(rows):
        raise RuntimeError("cohort count drift")
    summary = {
        **META, "status": "pass", "cases": len(records),
        "images": 4 * len(records), "labels": len(records),
        "split_counts": dict(observed),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "manifest_sha256": {r["case_id"]: digest(args.output_root / "cases" / r["case_id"] / "case_manifest.json") for r in records},
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.output_root / "DATASET_VALIDATION.json", summary)
    atomic_json(args.output_root / "COMPLETE.ok", summary)


if __name__ == "__main__":
    main()
