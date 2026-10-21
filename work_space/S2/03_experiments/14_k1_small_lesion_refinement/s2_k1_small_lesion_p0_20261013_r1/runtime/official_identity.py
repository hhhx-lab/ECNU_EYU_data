"""Stable full-reference identity using the frozen official approximator and matcher."""
import csv
import hashlib
import json
import struct
import sys
from pathlib import Path
from cpu_roundtrip import ROOT, emit, require, sha


def stable_id(case, region, label_sha, geometry, coordinates):
    coords = sorted(tuple(int(v) for v in row) for row in coordinates)
    require(coords and all(len(row) == 3 for row in coords), "empty/invalid instance")
    require(len(set(coords)) == len(coords), "duplicate instance coordinates")
    coordinate_bytes = b"".join(struct.pack("<qqq", *row) for row in coords)
    identity = dict(case=case, region=region, label_sha256=label_sha, geometry=geometry,
                    coordinates_sha256=hashlib.sha256(coordinate_bytes).hexdigest())
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def match_arrays(prediction, reference):
    import numpy as np
    from panoptica.utils.processing_pair import SemanticPair
    from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
    from panoptica.instance_matcher import MaxBipartiteMatching
    from panoptica.metrics import Metric
    from panoptica._functionals import _get_paired_crop
    crop = _get_paired_crop(prediction, reference)
    pair = SemanticPair(prediction[crop], reference[crop])
    approximate = ConnectedComponentsInstanceApproximator().approximate_instances(pair)
    matcher = MaxBipartiteMatching(matching_metric=Metric.DSC, matching_threshold=1e-6)
    matched = matcher.match_instances(approximate, matching_threshold=1e-6)
    origin = np.asarray([s.start or 0 for s in crop])
    rows = []
    for ref_label in matched.ref_labels:
        coords = np.argwhere(matched.reference_arr == ref_label)
        is_matched = ref_label in matched.matched_instances
        dsc = float(Metric.DSC(matched.reference_arr, matched.prediction_arr, ref_label, ref_label)) if is_matched else None
        rows.append(dict(coords=coords + origin, voxel_count=len(coords), is_matched=int(is_matched), sq_dsc=dsc))
    return rows, len(matched.missed_prediction_labels)


def preflight():
    import numpy as np
    import importlib.metadata
    import SimpleITK as sitk
    from panoptica import Panoptica_Evaluator
    from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
    from panoptica.utils.processing_pair import SemanticPair
    from importlib.resources import files
    require(importlib.metadata.version("panoptica") == "2.1.0", "panoptica version drift")
    config = Path(str(files("brats_evaluation").joinpath("configs/config_mets.yaml")))
    evaluator = Panoptica_Evaluator.load_from_config(str(config))
    label = np.zeros((6,6,6), dtype=np.uint8)
    label[1,1,1] = label[2,2,2] = 4
    approx = ConnectedComponentsInstanceApproximator().approximate_instances(SemanticPair(label, label))
    require(approx.n_ref_instances == 1, "26-connectivity contract failed")
    rows, fp = match_arrays(label, label)
    require(len(rows) == 1 and rows[0]["sq_dsc"] == 1 and fp == 0, "matcher identity test failed")
    image = sitk.GetImageFromArray(label)
    image.SetSpacing((.5,.5,.5))
    result = evaluator.evaluate(image, image)["rc"].to_dict(True)
    require(result["reference_instances"][0]["volume"] == .25, "physical volume test failed")
    geom = dict(shape=[6,6,6], spacing=[1,1,1])
    a = stable_id("case", "rc", "a"*64, geom, [[1,1,1]])
    b = stable_id("case", "rc", "a"*64, geom, [[2,2,2]])
    require(a != b, "equal-size identity collision")
    return evaluator


def write_csv(path, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run():
    import numpy as np
    import SimpleITK as sitk
    from panoptica.instance_approximator import ConnectedComponentsInstanceApproximator
    from panoptica.utils.processing_pair import SemanticPair
    from brats_evaluation.metrics_parser import parse_mets_results
    evaluator = preflight()
    plan = json.loads((ROOT / "CPU_EXECUTION_PLAN.json").read_text())
    cohort = Path(plan["cohort_root"])
    output = ROOT / "results/cpu_identity"
    output.mkdir(parents=True, exist_ok=False)
    all_rows, official, comparison = [], [], []
    regions = dict(wt=[1,2,3], tc=[1,3], et=[3], rc=[4])
    prior = {m: {c["subject_name"]: c for c in json.loads((ROOT / "sources" / m / "panoptica_evaluation_summary.json").read_text())["metrics"]} for m in ("K1", "K4")}
    for case_id in plan["case_ids"]:
        ref_path = cohort / "reference" / (case_id + ".nii.gz")
        image = sitk.ReadImage(str(ref_path))
        reference = sitk.GetArrayFromImage(image)
        geometry = dict(shape=list(reference.shape), spacing=list(image.GetSpacing()),
                        origin=list(image.GetOrigin()), direction=list(image.GetDirection()))
        label_sha = sha(ref_path)
        predictions = {}
        for method in ("K1", "K4", "label_roundtrip"):
            p = (ROOT / "roundtrip/prediction" if method == "label_roundtrip" else Path(plan["prediction_roots"][method])) / (case_id + ".nii.gz")
            if method != "label_roundtrip":
                require(sha(p) == plan["prediction_sha256"][method][p.name], "prediction SHA drift")
            pred_image = sitk.ReadImage(str(p))
            for getter in ("GetSpacing", "GetOrigin", "GetDirection", "GetSize"):
                require(np.allclose(getattr(pred_image,getter)(), getattr(image,getter)(), atol=1e-6, rtol=0), "prediction geometry drift")
            predictions[method] = sitk.GetArrayFromImage(pred_image)
            require(set(np.unique(predictions[method])) <= set(range(5)), "illegal prediction label")
        for region, labels in regions.items():
            binary_ref = np.isin(reference, labels).astype(np.uint8)
            full = ConnectedComponentsInstanceApproximator().approximate_instances(SemanticPair(binary_ref,binary_ref))
            identity = {}
            for instance in full.ref_labels:
                coords = np.argwhere(full.reference_arr == instance)
                key = stable_id(case_id,region,label_sha,geometry,coords)
                require(key not in identity, "reference identity collision")
                identity[key] = dict(coords=coords, count=len(coords), volume=float(len(coords)*np.prod(image.GetSpacing())))
            for method, prediction in predictions.items():
                records, fp = match_arrays(np.isin(prediction,labels).astype(np.uint8), binary_ref)
                emitted = set()
                fingerprint = []
                small_tp = small_fn = 0
                for record in records:
                    key = stable_id(case_id,region,label_sha,geometry,record["coords"])
                    require(key in identity and key not in emitted, "cropped/full reference identity mismatch")
                    emitted.add(key)
                    volume = identity[key]["volume"]
                    detected = record["is_matched"] == 1 and record["sq_dsc"] >= .2
                    if volume < 27:
                        small_tp += int(detected)
                        small_fn += int(not detected)
                    fingerprint.append((record["voxel_count"],volume,record["is_matched"],record["sq_dsc"] or 0))
                    all_rows.append(dict(subject_id=case_id,region=region,method=method,
                        stable_reference_id=key,reference_sha256=label_sha,original_voxel_count=record["voxel_count"],
                        original_volume_mm3=volume,small=volume<27,is_matched=record["is_matched"],
                        sq_dsc=record["sq_dsc"],detected=detected,geometry_json=json.dumps(geometry,sort_keys=True),
                        transport_category="not_evaluated_do_not_infer_disappearance"))
                require(emitted == set(identity), "incomplete reference matrix")
                if method != "label_roundtrip":
                    existing = prior[method][case_id + ".nii.gz"][region]
                    expected = sorted((r["voxel_count"],r["volume"],r.get("is_matched",0),r.get("sq_dsc",0) or 0) for r in existing["reference_instances"])
                    observed = sorted(fingerprint)
                    require(len(expected)==len(observed) and all(np.allclose(a,b,atol=1e-10,rtol=1e-10) for a,b in zip(expected,observed)), "official reference match fingerprint mismatch")
                    require(fp == existing["fp"], "official FP mismatch")
                comparison.append(dict(subject_id=case_id,region=region,method=method,small_tp=small_tp,small_fn=small_fn,all_fp=fp))
        evaluated = evaluator.evaluate(str(ROOT / "roundtrip/prediction" / (case_id+".nii.gz")), str(ref_path))
        official.append(dict(subject_name=case_id+".nii.gz", **{g:r.to_dict(True) for g,r in evaluated.items()}))
        print("P0_IDENTITY_CASE",case_id,flush=True)
    write_csv(output / "REFERENCE_DETECTION_MATRIX.csv",all_rows)
    write_csv(output / "CASE_COUNTS.csv",comparison)
    # Preserve official undefined NaN semantics in the official summary only.
    with (output / "panoptica_evaluation_summary.json").open("x") as handle:
        json.dump(dict(metrics=official,missings=[]),handle)
    parse_mets_results(str(output / "panoptica_evaluation_summary.json"),27,.2,str(output / "leaderboard_metrics.csv"))
    paired = {}
    for row in all_rows:
        if row["region"]=="wt" and row["small"]:
            paired.setdefault(row["stable_reference_id"],{})[row["method"]]=row["detected"]
    require(len(paired)==286 and all(set(v)=={"K1","K4","label_roundtrip"} for v in paired.values()),"small286 matrix mismatch")
    require(sum(v["K1"] for v in paired.values())==46 and sum(v["K4"] for v in paired.values())==52,"WT small historical count drift")
    stats=dict(k1_tp=46,k4_tp=52,common_missed_k1_k4=sum(not v["K1"] and not v["K4"] for v in paired.values()),
        new_k4_vs_k1=sum(v["K4"] and not v["K1"] for v in paired.values()),lost_k4_vs_k1=sum(v["K1"] and not v["K4"] for v in paired.values()),
        roundtrip_tp=sum(v["label_roundtrip"] for v in paired.values()))
    emit(ROOT / "evidence/OFFICIAL_IDENTITY_VALIDATION.json",dict(status="pass",case_count=103,
        detection_record_count=len(all_rows),wt_small_reference_count=286,statistics=stats,
        scientific_scope="two-route intersection only; label roundtrip is not a theoretical upper bound"))


if __name__ == "__main__":
    if sys.argv[1] == "preflight":
        preflight()
        print("OFFICIAL_IDENTITY_PREFLIGHT_PASS")
    elif sys.argv[1] == "run":
        run()
    else:
        raise ValueError("unknown action")
