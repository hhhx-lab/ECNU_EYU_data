"""Convert raw BraTS-MET cases into the nnU-Net 3d_fullres representation (1 mm, masked z-score).

Each case directory must contain  <case>-{t1c,t1n,t2w,t2f,seg}.nii.gz .
Output (one folder per case, same file names, same layout as the original DataSet/):
    <out_dir>/<case>/<case>-t1c.nii.gz ... float32, nnU-Net masked z-score, 1 x 1 x 1 mm
    <out_dir>/<case>/<case>-seg.nii.gz      uint8, labels 0-4 (nnU-Net's -1 "outside brain" -> 0)
    The header spacing is exactly --target_spacing (nnU-Net convention); the true spacing after
    rounding the new shape (within a fraction of a percent) is in the report column true_spacing.
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
from src.preprocess.nnunet_like import BACKEND, preprocess_case  # noqa: E402

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
                print(f"[WARN] duplicate case {name} in {root} (keeping {cases[name]})")
                continue
            cases[name] = d
    return cases


def new_affine(affine, bbox, shape_cropped, new_shape, spacing, target_spacing):
    """Voxel-to-world matrix of the resampled grid.

    Origin: centre of the first new voxel under the pixel-centre mapping used by nnU-Net/skimage.
    [1mm] Voxel size: exactly target_spacing (nnU-Net treats the resampled array as target_spacing);
    the true size shape_cropped / new_shape * spacing differs by a fraction of a percent because the
    new shape is rounded. Returns (affine, true_spacing).
    """
    f_true = np.asarray(shape_cropped, float) / np.asarray(new_shape, float)
    f_exact = np.asarray(target_spacing, float) / np.asarray(spacing, float)
    lo = np.array([b[0] for b in bbox], float)
    m = np.eye(4)
    m[:3, :3] = np.diag(f_exact)
    m[:3, 3] = lo + 0.5 * f_true - 0.5
    return affine @ m, f_true * np.asarray(spacing, float)


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
    if not overwrite and all(os.path.isfile(p) for p in outputs):
        return {"case": name, "status": "skipped (exists)"}
    t0 = time.time()
    try:
        paths = [os.path.join(case_dir, f"{name}-{m}.nii.gz") for m in MODALITIES + ["seg"]]
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            return {"case": name, "status": f"missing {', '.join(os.path.basename(p) for p in missing)}"}
        seg_img = nib.load(paths[-1])
        spacing = [float(v) for v in seg_img.header.get_zooms()[:3]]
        data = []
        for m, p in zip(MODALITIES, paths[:-1]):
            img = nib.load(p)
            if img.shape[:3] != seg_img.shape[:3]:
                return {"case": name, "status": f"rejected: shape mismatch {m} {img.shape} vs seg {seg_img.shape}"}
            # [1mm] Same voxel grid in world space (modalities and seg must be co-registered)
            shift = corner_shift_mm(img.affine, seg_img.affine, seg_img.shape)
            if shift > affine_tol:
                return {"case": name, "status": f"rejected: {m} grid differs from seg by {shift:.2f} mm "
                                                f"(tolerance {affine_tol} mm)"}
            arr = img.get_fdata(dtype=np.float32)
            if not np.isfinite(arr).all():  # [1mm] NaN/Inf would turn the whole z-scored modality into NaN
                return {"case": name, "status": f"rejected: NaN/Inf in {m}"}
            data.append(arr)
        data = np.stack(data)
        seg_raw = seg_img.get_fdata(dtype=np.float32)
        # [1mm] Label validation: finite, integer, and within the allowed label set
        if not np.isfinite(seg_raw).all():
            return {"case": name, "status": "rejected: NaN/Inf in seg"}
        if np.abs(seg_raw - np.rint(seg_raw)).max() > 1e-3:
            return {"case": name, "status": "rejected: non-integer label values in seg"}
        bad = sorted(set(np.unique(np.rint(seg_raw)).astype(int).tolist()) - set(allowed_labels))
        if bad:
            return {"case": name, "status": f"rejected: label values {bad} not in {sorted(allowed_labels)}"}
        seg = np.rint(seg_raw).astype(np.int16)[None]

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
        for c, m in enumerate(MODALITIES):
            save_atomic(nib.Nifti1Image(data_out[c], affine), outputs[c])
        seg_save = np.clip(seg_out[0], 0, None).astype(np.uint8)  # -1 (outside brain) -> 0
        save_atomic(nib.Nifti1Image(seg_save, affine), outputs[-1])  # written last: marks the case complete
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

    cases = find_cases(args.raw_dirs)
    keep, drop = read_list(args.case_list), read_list(args.exclude_list) or set()
    if keep is not None:
        absent = sorted(keep - set(cases))
        if absent:
            print(f"[WARN] {len(absent)} listed cases not found, e.g. {absent[:5]}")
        cases = {k: v for k, v in cases.items() if k in keep}
    cases = {k: v for k, v in cases.items() if k not in drop}
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Backend: {BACKEND} | cases to process: {len(cases)} | target spacing: {args.target_spacing}")
    if BACKEND != "nnunetv2":
        print("[NOTE] nnunetv2 not importable - using the built-in replica of its resampling. "
              "Install nnunetv2 (same version as S2) for bit-identical arrays.")

    with open(os.path.join(args.out_dir, "preprocess_config.json"), "w", encoding="utf-8") as f:
        json.dump({"backend": BACKEND, **vars(args)}, f, indent=2)

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


if __name__ == "__main__":
    main()
