"""Frozen label-only nnU-Net round trip, followed by official instance diagnosis."""
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = dict(artifact_status="experimental_unvalidated", operator_approved=False,
            formal_gate_status="not_run_not_passed")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def emit(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(**META, remote_utc=datetime.now(timezone.utc).isoformat(), **payload),
                  handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def label_roundtrip(images, seg, properties, pm, cm, dataset):
    import numpy as np
    from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
    from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape

    props = copy.deepcopy(properties)
    require(images.shape[1:] == seg.shape[1:], "image/label shape mismatch")
    require(np.isfinite(images).all() and np.isfinite(seg).all(), "nonfinite image/label")
    require(set(np.unique(seg)) <= set(range(5)), "illegal original label")
    require(properties["spacing"] == properties["seg_spacing"], "image/label spacing mismatch")
    data = images.transpose([0, *[i + 1 for i in pm.transpose_forward]])
    label = seg.transpose([0, *[i + 1 for i in pm.transpose_forward]])
    spacing = [properties["spacing"][i] for i in pm.transpose_forward]
    props["shape_before_cropping"] = list(data.shape[1:])
    cropped, label, bbox = crop_to_nonzero(data, label)
    props["bbox_used_for_cropping"] = bbox
    props["shape_after_cropping_and_before_resampling"] = list(cropped.shape[1:])
    del data, cropped
    shape = compute_new_shape(label.shape[1:], spacing, cm.spacing)
    label = cm.resampling_fn_seg(label, shape, spacing, cm.spacing)
    label = np.maximum(label[0], 0).astype(np.uint8)
    require(set(np.unique(label)) <= set(range(5)), "illegal resampled label")
    scores = np.stack([label == value for value in range(5)]).astype(np.float32)
    output = convert_predicted_logits_to_segmentation_with_correct_shape(
        scores, pm, cm, pm.get_label_manager(dataset), props, False, num_threads_torch=1)
    require(output.shape == seg.shape[1:], "roundtrip shape drift")
    require(set(np.unique(output)) <= set(range(5)), "illegal roundtrip label")
    return output, props, list(label.shape)


def preflight():
    import numpy as np
    import torch
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    import nnunetv2.inference.export_prediction
    plan = json.loads((ROOT / "CPU_EXECUTION_PLAN.json").read_text())
    for path, digest in plan["input_reference_sha256"].items():
        require(sha(Path(path)) == digest, "input/reference SHA drift: " + path)
    for path, digest in plan["software_sha256"].items():
        require(sha(Path(path)) == digest, "software source SHA drift: " + path)
    for path, digest in plan["dependency_sha256"].items():
        require(sha(Path(path)) == digest, "dependency SHA drift: " + path)
    for relative, digest in plan["static_sha256"].items():
        require(sha(ROOT / relative) == digest, "runtime/manifest SHA drift")
    pm = PlansManager(plan["plans_path"])
    cm = pm.get_configuration("3d_fullres")
    dataset = json.loads(Path(plan["dataset_path"]).read_text())
    require(list(cm.spacing) == [1.0, 1.0, 1.0], "spacing contract drift")
    require(pm.image_reader_writer_class.__name__ == "SimpleITKIO", "reader contract drift")
    label = np.zeros((1, 8, 8, 8), dtype=np.int8)
    label[0, 3:5, 3:5, 3:5] = 4
    images = np.ones((4, 8, 8, 8), dtype=np.float32)
    props = dict(spacing=[1.0]*3, seg_spacing=[1.0]*3,
                 sitk_stuff=dict(spacing=[1.0]*3, origin=[0.0]*3,
                                 direction=[1,0,0,0,1,0,0,0,1]))
    output, _, _ = label_roundtrip(images, label, props, pm, cm, dataset)
    require(np.array_equal(output, label[0]), "identity spacing roundtrip failed")
    empty_output, _, _ = label_roundtrip(images, np.zeros_like(label), props, pm, cm, dataset)
    require(not np.any(empty_output), "empty label roundtrip failed")
    cropped_images = np.zeros_like(images)
    cropped_images[:, 2:6, 2:6, 2:6] = 1
    crop_output, crop_properties, _ = label_roundtrip(cropped_images, label, props, pm, cm, dataset)
    require(np.array_equal(crop_output, label[0]), "nonzero crop roundtrip failed")
    require(crop_properties["bbox_used_for_cropping"] == [[2,6],[2,6],[2,6]], "crop bbox mismatch")
    require(not torch.cuda.is_available(), "CPU diagnosis must not expose GPU")
    subprocess.run([plan["eval_python"], "-B", str(ROOT / "runtime/official_identity.py"), "preflight"], check=True, timeout=180)
    return plan, pm, cm, dataset


def run():
    plan, pm, cm, dataset = preflight()
    emit(ROOT / "evidence/P0_CPU_STARTED.json", dict(status="started", job_id=os.environ.get("SLURM_JOB_ID"),
         hostname=os.uname().nodename, plan_sha256=sha(ROOT / "CPU_EXECUTION_PLAN.json")))
    prediction = ROOT / "roundtrip/prediction"
    prediction.mkdir(parents=True, exist_ok=False)
    cases = list(csv.DictReader((ROOT / "manifests/FIXED103_REAL_T2W_CASES.csv").open()))
    require(len(cases) == 103, "case count")
    rw = pm.image_reader_writer_class()
    geometry = []
    import numpy as np
    for index, case in enumerate(cases):
        case_id = case["source_case_id"]
        cohort = Path(plan["cohort_root"])
        files = [str(cohort / "input" / (case_id + f"_{i:04d}.nii.gz")) for i in range(4)]
        images, props = rw.read_images(files)
        seg, seg_props = rw.read_seg(str(cohort / "reference" / (case_id + ".nii.gz")))
        for key in ("origin", "direction", "spacing"):
            require(np.allclose(props["sitk_stuff"][key], seg_props["sitk_stuff"][key], rtol=0, atol=1e-6), "image/GT geometry mismatch")
        props["seg_spacing"] = seg_props["spacing"]
        output, transformed, training_shape = label_roundtrip(images, seg, props, pm, cm, dataset)
        target = prediction / (case_id + ".nii.gz")
        rw.write_seg(output, str(target), transformed)
        reread, reread_props = rw.read_seg(str(target))
        require(np.array_equal(reread[0], output), "write/read label drift")
        require(reread_props["spacing"] == props["spacing"], "write/read spacing drift")
        geometry.append(dict(subject_id=case_id, original_shape=list(seg.shape[1:]),
            original_properties=props, training_shape=training_shape,
            bbox=transformed["bbox_used_for_cropping"], prediction_sha256=sha(target)))
        print(f"P0_ROUNDTRIP_CASE {index+1}/103 {case_id}", flush=True)
    emit(ROOT / "roundtrip/GEOMETRY.json", dict(status="pass", cases=geometry,
         score_policy="hard-label one-hot bounded scores, frozen probability resampling and argmax; not a theoretical upper bound"))
    subprocess.run([plan["eval_python"], "-B", str(ROOT / "runtime/official_identity.py"), "run"], check=True)
    require((ROOT / "evidence/OFFICIAL_IDENTITY_VALIDATION.json").is_file(), "identity validation missing")
    emit(ROOT / "evidence/P0_CPU_VALIDATION.json", dict(status="pass", case_count=103,
        plan_sha256=sha(ROOT / "CPU_EXECUTION_PLAN.json"), instance_transport_classification="not_yet_evaluated",
        scope="fixed103 full-volume identity, K1/K4 official matching, fixed label roundtrip; not all P0"))
    (ROOT / "evidence/P0_CPU_COMPLETE.ok").write_text("status=pass\nartifact_status=experimental_unvalidated\noperator_approved=false\nformal_gate_status=not_run_not_passed\n")
    lightweight = [p for parent in (ROOT/"results", ROOT/"evidence") for p in parent.rglob("*") if p.is_file() and p.suffix in (".json", ".csv", ".md", ".ok")]
    emit(ROOT / "evidence/LIGHTWEIGHT_SHA256.json", dict(status="pass", files={str(p.relative_to(ROOT)):sha(p) for p in lightweight}))


if __name__ == "__main__":
    try:
        if sys.argv[1] == "preflight":
            preflight()
            emit(ROOT / "evidence/CPU_STATIC_PREFLIGHT.json", dict(status="pass", plan_sha256=sha(ROOT / "CPU_EXECUTION_PLAN.json"), actual_imports=True,
                 empty_label_roundtrip_pass=True, nonzero_crop_roundtrip_pass=True,
                 official_boundary_report_sha256=sha(ROOT / "evidence/OFFICIAL_BOUNDARY_TESTS.json")))
            print("P0_CPU_STATIC_PASS")
        elif sys.argv[1] == "run":
            run()
        else:
            raise ValueError("unknown action")
    except Exception:
        name = "P0_CPU_FAILED.json" if sys.argv[1] == "run" else "CPU_STATIC_PREFLIGHT_FAILED.json"
        emit(ROOT / "evidence" / name, dict(status="failed", error=traceback.format_exc()))
        raise
