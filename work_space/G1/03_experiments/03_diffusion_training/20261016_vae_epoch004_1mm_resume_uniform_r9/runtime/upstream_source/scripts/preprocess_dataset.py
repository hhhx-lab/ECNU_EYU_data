"""Convert raw BraTS-MET cases into the nnU-Net 3d_fullres representation (1 mm, masked z-score).

Each case directory must contain  <case>-{t1c,t1n,t2w,t2f,seg}.nii.gz .
Output (one folder per case, same file names, same layout as the original DataSet/):
    <out_dir>/<case>/<case>-t1c.nii.gz ... float32, nnU-Net masked z-score, 1 x 1 x 1 mm
    <out_dir>/<case>/<case>-seg.nii.gz      uint8, labels 0-4 (nnU-Net's -1 "outside brain" -> 0)
    The sampling coordinates and header both use exactly --target_spacing (physical_spacing_v2).
    Shape rounding changes the far field-of-view edge, not the voxel spacing.
    <out_dir>/preprocess_report.csv         per-case shapes / spacings / lesion counts before & after
    <out_dir>/preprocess_config.json        backend + arguments

Usage (run from TumourDiffusion_1mm/):
    python scripts/preprocess_dataset.py \
        --raw_dirs /data/raw/1 /data/raw/2 ... /data/raw/u9 \
        --out_dir DataSet --case_list lists/train_cases.txt --num_workers 8
"""

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # package root
from src.preprocess.nnunet_like import BACKEND, GRID_VERSION, physical_grid, preprocess_case
from src.utils.volume_validation import validate_case, sha256_file

MODALITIES = ["t1c", "t1n", "t2w", "t2f"]
CC_STRUCTURE = np.ones((3, 3, 3), dtype=np.int8)


def read_list(path):
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip() and not line.startswith("#")}


def find_cases(raw_dirs):
    cases = {}
    for root in raw_dirs:
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if not os.path.isdir(d) or not name.startswith("BraTS-"):
                continue  # skips __MACOSX, stray archives, etc.
            if name in cases:
                raise ValueError(f"Duplicate case {name} in {root} and {cases[name]}")
            cases[name] = d
    return cases


def new_affine(affine, bbox, shape_cropped, new_shape, spacing, target_spacing):
    """Voxel-to-world matrix of the resampled grid.

    Uses the same physical_grid mapping as resample_physical. The low voxel-cell boundary
    remains fixed and voxel size is exactly target_spacing. Returns (affine, true_spacing).
    """
    expected_shape, f_exact, offset = physical_grid(shape_cropped, spacing, target_spacing)
    if not np.array_equal(expected_shape, new_shape):
        raise ValueError("Output shape differs from the physical sampling grid")
    lo = np.array([b[0] for b in bbox], float)
    m = np.eye(4)
    m[:3, :3] = np.diag(f_exact)
    m[:3, 3] = lo + offset
    return affine @ m, np.asarray(target_spacing, float)


def corner_shift_mm(a, b, shape):
    """Max world-space distance between the 8 volume corners under two voxel-to-world matrices."""
    c = np.array([[i, j, k, 1.0] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1)
                  for k in (0, shape[2] - 1)]).T
    return float(np.linalg.norm((a @ c - b @ c)[:3], axis=0).max())


def save_atomic(img, path):
    """Write next to the target and rename, so an interrupted run never leaves a truncated file."""
    tmp = path[:-len(".nii.gz")] + ".tmp.nii.gz"
    nib.save(img, tmp)
    os.replace(tmp, path)


def count_lesions(mask, voxel_volume):
    lab, n = ndimage.label(mask, structure=CC_STRUCTURE)
    if n == 0:
        return 0, 0
    sizes = np.bincount(lab.ravel())[1:] * voxel_volume
    return int(n), int((sizes < 27.0).sum())


def process(job):
    name, case_dir, out_dir, target_spacing, overwrite, allowed_labels, affine_tol = job
    out_case = os.path.join(out_dir, name)
    outputs = [os.path.join(out_case, f"{name}-{m}.nii.gz") for m in MODALITIES + ["seg"]]
    t0 = time.time()
    try:
        if os.path.realpath(case_dir) == os.path.realpath(out_case):
            raise ValueError("Output case directory must differ from the raw input directory")
        paths = [os.path.join(case_dir, f"{name}-{m}.nii.gz") for m in MODALITIES + ["seg"]]
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            return {"case": name, "status": f"missing {', '.join(os.path.basename(p) for p in missing)}"}
        images, arrays = validate_case(paths, allowed_labels, affine_tol)
        contract = {"grid_version": GRID_VERSION, "backend": BACKEND,
                    "target_spacing": list(target_spacing), "allowed_labels": list(allowed_labels),
                    "affine_tol_mm": affine_tol,
                    "sources": {os.path.basename(p): sha256_file(p) for p in paths}}
        marker_path = os.path.join(out_case, "preprocess_complete.json")
        if not overwrite and any(os.path.exists(p) for p in outputs):
            try:
                with open(marker_path, encoding="utf-8") as stream:
                    marker = json.load(stream)
                if marker.get("contract") != contract or marker.get("status") != "complete":
                    raise ValueError("Missing/stale completion contract")
                validate_case(outputs, allowed_labels, affine_tol)
                if marker["outputs"] != {os.path.basename(p): sha256_file(p) for p in outputs}:
                    raise ValueError("Output checksum mismatch")
            except Exception as exc:
                raise ValueError(f"Existing output cannot be reused; rerun with --overwrite: {exc}") from exc
            return {"case": name, "status": "skipped (verified)"}
        seg_img = images[-1]
        spacing = [float(v) for v in seg_img.header.get_zooms()[:3]]
        data = np.stack(arrays[:-1])
        seg = arrays[-1].astype(np.int16)[None]
        del images, arrays

        vv0 = float(np.prod(spacing))
        n0, small0 = count_lesions(seg[0] > 0, vv0)
        data_out, seg_out, info = preprocess_case(data, seg, spacing, target_spacing)
        n1, small1 = count_lesions(seg_out[0] > 0, float(np.prod(target_spacing)))
        if not np.isfinite(data_out).all():
            return {"case": name, "status": "error: non-finite values after preprocessing"}

        affine, true_spacing = new_affine(seg_img.affine, info["bbox_used_for_cropping"],
                                          info["shape_after_cropping"], info["shape_after_resampling"],
                                          spacing, target_spacing)
        os.makedirs(out_case, exist_ok=True)
        with open(marker_path, "w", encoding="utf-8") as stream:
            json.dump({"status": "in_progress", "contract": contract}, stream)
        for c, m in enumerate(MODALITIES):
            save_atomic(nib.Nifti1Image(data_out[c], affine), outputs[c])
        seg_save = np.clip(seg_out[0], 0, None).astype(np.uint8)  # -1 (outside brain) -> 0
        save_atomic(nib.Nifti1Image(seg_save, affine), outputs[-1])  # written last: marks the case complete
        marker = {"status": "complete", "contract": contract,
                  "outputs": {os.path.basename(p): sha256_file(p) for p in outputs}}
        with open(marker_path + ".tmp", "w", encoding="utf-8") as stream:
            json.dump(marker, stream, indent=2)
        os.replace(marker_path + ".tmp", marker_path)
        return {
            "case": name, "status": "ok", "backend": info["backend"],
            "orig_shape": "x".join(map(str, data.shape[1:])),
            "orig_spacing": "x".join(f"{s:.3f}" for s in spacing),
            "new_shape": "x".join(map(str, data_out.shape[1:])),
            "true_spacing": "x".join(f"{s:.4f}" for s in true_spacing),
            "lesions_before": n0, "lesions_after": n1, "lesions_lost": n0 - n1,
            "small_lt27mm3_before": small0, "small_lt27mm3_after": small1,
            "seconds": round(time.time() - t0, 1),
        }
    except Exception as e:  # noqa: BLE001
        return {"case": name, "status": f"error: {type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dirs", nargs="+", required=True, help="Folders that contain BraTS-MET-* case folders")
    ap.add_argument("--out_dir", default="DataSet")
    ap.add_argument("--case_list", default="", help="Optional txt with case IDs to process (one per line)")
    ap.add_argument("--exclude_list", default="", help="Optional txt with case IDs to skip")
    ap.add_argument("--target_spacing", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    ap.add_argument("--num_workers", type=int, default=4, help="Parallel cases (~10 GB RAM each for 512x512x400 volumes)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allowed_labels", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                    help="Label values accepted in seg; cases with other values are rejected (see report)")
    ap.add_argument("--affine_tol_mm", type=float, default=0.1,
                    help="Max allowed world-space offset between a modality's voxel grid and the seg's")
    args = ap.parse_args()
    if not np.isfinite(args.target_spacing).all() or min(args.target_spacing) <= 0:
        ap.error("--target_spacing must be finite and positive")
    if not np.isfinite(args.affine_tol_mm) or args.affine_tol_mm < 0:
        ap.error("--affine_tol_mm must be finite and non-negative")
    if not set(args.allowed_labels).issubset({0, 1, 2, 3, 4}):
        ap.error("Only labels 0-4 are supported")

    cases = find_cases(args.raw_dirs)
    keep, drop = read_list(args.case_list), read_list(args.exclude_list) or set()
    absent = []
    if keep is not None:
        absent = sorted(keep - drop - set(cases))
        if absent:
            print(f"[WARN] {len(absent)} listed cases not found, e.g. {absent[:5]}")
        cases = {k: v for k, v in cases.items() if k in keep}
    cases = {k: v for k, v in cases.items() if k not in drop}
    if not cases and not absent:
        raise SystemExit("No cases selected for preprocessing")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Backend: {BACKEND} | cases to process: {len(cases)} | target spacing: {args.target_spacing}")
    if BACKEND != "nnunetv2":
        print("[NOTE] Using fallback cropping/normalization. Resampling uses the exact physical grid.")

    with open(os.path.join(args.out_dir, "preprocess_config.json"), "w", encoding="utf-8") as f:
        json.dump({"backend": BACKEND, "grid_version": GRID_VERSION, **vars(args)}, f, indent=2)

    jobs = [(k, v, args.out_dir, tuple(args.target_spacing), args.overwrite, tuple(args.allowed_labels),
             args.affine_tol_mm) for k, v in sorted(cases.items())]
    fields = ["case", "status", "backend", "orig_shape", "orig_spacing", "new_shape", "true_spacing",
              "lesions_before",
              "lesions_after", "lesions_lost", "small_lt27mm3_before", "small_lt27mm3_after", "seconds"]
    report_path = os.path.join(args.out_dir, "preprocess_report.csv")
    t0, done, failed, rejected = time.time(), 0, 0, []
    with open(report_path, "w", newline="", encoding="utf-8") as fr, Pool(max(1, args.num_workers)) as pool:
        writer = csv.DictWriter(fr, fieldnames=fields)
        writer.writeheader()
        for case in absent:
            writer.writerow({"case": case, "status": "missing: listed case not found"})
            failed += 1
        for row in pool.imap_unordered(process, jobs):
            writer.writerow(row)
            fr.flush()
            done += 1
            if row["status"].startswith("rejected"):
                rejected.append(f"{row['case']}: {row['status']}")
            elif row["status"] != "ok" and not row["status"].startswith("skipped"):
                failed += 1
            if done % 20 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"[{done}/{len(jobs)}] {el / 60:.1f} min elapsed, ETA {el / done * (len(jobs) - done) / 60:.1f} min, failed {failed}")
    print(f"Done. Report: {report_path}")
    if rejected:
        print(f"[WARN] {len(rejected)} case(s) rejected by the data checks (not written to {args.out_dir}):")
        for r in sorted(rejected):
            print("   ", r)
    if failed:
        print(f"[WARN] {failed} cases failed - see 'status' column of the report.")
    if failed or rejected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
