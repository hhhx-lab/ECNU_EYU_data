"""Read-only post-completion audit and lightweight evidence transport."""
import base64
import csv
import gc
import hashlib
import itertools
import json
import math
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

BASE = Path("/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2")
ROOTS = {
    "extend": BASE / "experimental_unvalidated/s2_met_aug_bef_extend175_20260907_r4",
    "met_r2": BASE / "results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r2",
    "met_r4": BASE / "results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r4",
}
PLANS = {
    "extend": ("EXTEND175_EXECUTION_PLAN_20260907_R4.json", "cdc19fe9b3d4d3373822c0aa0a80f594f8718d284dc12424be68114314f963ae"),
    "met_r2": ("SELECTED103_OFFICIAL_MET_EXECUTION_PLAN_20260907_R2.json", "1767372636d82e5d02bbd0e9d8ce4b982b1777a60c36c27b655c3b3b5b429c3b"),
    "met_r4": ("SELECTED103_OFFICIAL_MET_RECOVERY_PLAN_20260907_R4.json", "b733d54fc32391143baf1c3fde30f8004b1d507878d7b05be65dbc3112a46513"),
}
METHODS = ("B", "C40", "D", "E_SNFH", "F")
META = {"artifact_status": "experimental_unvalidated", "operator_approved": False, "formal_gate_status": "not_run_not_passed"}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def contract(data):
    for key, value in META.items():
        require(data.get(key) == value, "artifact metadata mismatch: " + key)


def finite_tree(value, context):
    count = 0
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), "nonfinite tensor: " + context)
        return value.numel()
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            require(bool(np.isfinite(value).all()), "nonfinite array: " + context)
            return value.size
        return 0
    if isinstance(value, dict):
        for k, v in value.items():
            count += finite_tree(v, context + "." + str(k))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            count += finite_tree(v, context + "." + str(i))
    elif isinstance(value, (float, int, np.number)):
        require(math.isfinite(float(value)), "nonfinite scalar: " + context)
        return 1
    return count


def selected_light_file(root, path):
    rel = path.relative_to(root)
    if path.is_symlink() or any(x in rel.parts for x in ("uploads", "model_snapshots", "cache", "prediction", "input", "reference", "__pycache__")):
        return False
    if path.suffix.lower() not in (".json", ".csv", ".md", ".txt", ".ok", ".log", ".out", ".err"):
        return False
    if len(rel.parts) == 1:
        return path.name == "README.md" or "PLAN" in path.name or "STATIC_SHA256SUMS" in path.name
    return rel.parts[0] in ("evidence", "logs", "manifests", "configs", "cohort", "results")


def main():
    torch.set_num_threads(2)
    report = {**META, "status": "pass", "remote_verified_at_utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname()}
    plans = {}
    for name, root in ROOTS.items():
        file, expected = PLANS[name]
        require(sha(root / file) == expected, name + " plan SHA drift")
        plans[name] = read_json(root / file)
        contract(plans[name])
        subprocess.run(["sha256sum", "-c", "STATIC_SHA256SUMS.txt"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        require(read_json(root / "evidence/REMOTE_STATIC_PREFLIGHT.json")["status"] == "pass", name + " static preflight failed")
    for name in ("extend", "met_r4"):
        require(not list((ROOTS[name] / "evidence").glob("*FAILED*")), name + " failure marker exists")
    report["plan_sha256"] = {k: v[1] for k, v in PLANS.items()}
    report["extension"] = {}
    for route in plans["extend"]["routes"]:
        method = route["route_token"]
        root = ROOTS["extend"]
        val = read_json(root / f"evidence/{method}_EXTENSION_TRAINING_VALIDATION.json")
        started = read_json(root / f"evidence/{method}_EXTENSION_TRAINING_STARTED.json")
        contract(val)
        require(val["status"] == "pass" and val["completed_total_epochs"] == 175, method + " training validation")
        require((root / f"evidence/{method}_EXTENSION_TRAINING_COMPLETE.ok").is_file(), method + " missing complete")
        require(started["source_total_epoch"] == 125 and started["target_total_epoch"] == 175 and "A100" in started["device_name"], method + " started contract")
        require(sha(Path(started["source_checkpoint"])) == route["source_checkpoint_sha256"], method + " original source SHA")
        rows = []
        for row in val["candidate_checkpoints"]:
            epoch = row["total_epoch"]
            path = Path(row["path"])
            require(path.is_relative_to(root), "checkpoint outside current extension root")
            require(path.name == f"checkpoint_epoch_{epoch:04d}.pth", "checkpoint filename")
            require(sha(path) == row["sha256"], method + " candidate SHA")
            ck = torch.load(path, map_location="cpu", weights_only=False)
            require(ck["current_epoch"] == epoch, method + " current_epoch")
            counts = {}
            for key in ("network_weights", "optimizer_state", "grad_scaler_state", "logging"):
                require(key in ck, method + " missing " + key)
                counts[key] = finite_tree(ck[key], method + "." + key)
            require(counts["network_weights"] > 0 and counts["optimizer_state"] > 0 and counts["logging"] > 0, method + " empty checkpoint state")
            score = float(ck["logging"]["ema_fg_dice"][epoch - 1])
            require(score == row["ema_fg_dice"], method + " candidate score drift")
            if epoch == 125:
                require(row["sha256"] == route["source_checkpoint_sha256"], method + " source125 SHA")
                require(path.stat().st_mode & 0o777 == 0o400, method + " source125 permissions")
            rows.append({**row, "numeric_elements_checked": counts, "amp_state_is_none": ck["grad_scaler_state"] is None, "finite": True})
            del ck
            gc.collect()
        require([r["total_epoch"] for r in rows] == [125, 150, 175], method + " candidate epoch set")
        best = min(rows, key=lambda r: (-r["ema_fg_dice"], r["total_epoch"]))
        require(best["sha256"] == val["best_checkpoint"]["sha256"], method + " selection rule")
        report["extension"][method] = {"candidates": rows, "selected_epoch": best["total_epoch"], "selected_sha256": best["sha256"], "selected_ema_fg_dice": best["ema_fg_dice"], "delta_from125": best["ema_fg_dice"] - rows[0]["ema_fg_dice"], "job_id": val["job_id"]}

    r2, r4 = ROOTS["met_r2"], ROOTS["met_r4"]
    input_audit = read_json(r2 / "cohort/fixed103/input_audit.json")
    require(input_audit["status"] == "pass" and input_audit["case_count"] == 103 and input_audit["source_t2w_used"] is True, "input audit")
    require(input_audit["geometry_mismatch_count"] == input_audit["geometry_repaired_count"] == 0, "input geometry")
    manifest = r4 / "manifests/FIXED103_REAL_T2W_CASES.csv"
    require(sha(manifest) == "86ba853dadc64ffc45357895ba0bd543bf5b6b14c0c83eb789f1e875d4aa2e7e", "manifest SHA")
    with manifest.open() as f:
        ids = {r["source_case_id"] for r in csv.DictReader(f)}
    require(len(ids) == 103, "manifest cardinality")
    reference = r2 / "cohort/fixed103/reference"
    require({p.name.removesuffix(".nii.gz") for p in reference.glob("*.nii.gz")} == ids, "reference IDs")
    require(len(list((r2 / "cohort/fixed103/input").glob("*.nii.gz"))) == 412, "input count")
    require(sha(r2 / "runtime/run_official_met_eval.sh") == sha(r4 / "runtime/run_official_met_eval.sh"), "official MET pipeline drift")
    report["official_met"] = {}
    csv_data = {}
    for method in METHODS:
        root = r2 if method in ("B", "E_SNFH", "F") else r4
        tag = "EVALUATION" if root == r2 else "RECOVERY"
        val = read_json(root / f"evidence/{method}_{tag}_VALIDATION.json")
        started = read_json(root / f"evidence/{method}_{tag}_STARTED.json")
        require(not (root / f"evidence/{method}_{tag}_FAILED.json").exists(), method + " failure")
        require((root / f"evidence/{method}_{tag}_COMPLETE.ok").is_file(), method + " complete")
        contract(val)
        expected = plans["met_r4"]["routes"][method]
        require(val["status"] == "pass" and val["case_count"] == val["prediction_count"] == 103, method + " validation")
        require(val["selected_epoch"] == expected["checkpoint_epoch"], method + " selected epoch")
        require(val["checkpoint_sha256"] == expected["checkpoint_sha256"], method + " frozen SHA")
        checkpoint = r2 / f"model_snapshots/{method}/fold_0/checkpoint_final.pth"
        require(sha(checkpoint) == expected["checkpoint_sha256"], method + " checkpoint content SHA")
        require(started["single_gpu"] is True and "A100" in started["device_name"], method + " single GPU")
        out = root / f"cohort/fixed103/models/{method}"
        inf = read_json(out / "inference_audit.json")
        require(inf["status"] == "pass" and inf["prediction_count"] == inf["case_count"] == 103 and inf["checkpoint_sha256"] == expected["checkpoint_sha256"], method + " inference audit")
        if root == r4:
            require(inf["execution_mode"] == "sequential_in_process" and inf["preprocess_workers"] == inf["export_workers"] == 0 and inf["multiprocessing_queue_used"] is False, method + " sequential contract")
        predictions = list((out / "prediction").glob("*.nii.gz"))
        require({p.name.removesuffix(".nii.gz") for p in predictions} == ids, method + " prediction IDs")
        for pred in predictions:
            image, ref = nib.load(pred), nib.load(reference / pred.name)
            require(image.shape == ref.shape and np.allclose(image.affine, ref.affine, rtol=0, atol=1e-5), method + " prediction geometry " + pred.name)
        ev = read_json(out / "evaluation/EVALUATION_VALIDATION.json")
        require(ev["status"] == "pass" and ev["case_count"] == 103, method + " official eval validation")
        require((out / "evaluation/EVALUATION_COMPLETE.ok").is_file(), method + " official eval complete")
        summary_path = out / "evaluation/panoptica_evaluation_summary.json"
        require(sha(summary_path) == ev["summary_json_sha256"], method + " evaluator summary SHA")
        summary = read_json(summary_path)
        require(summary["missings"] == [] and len(summary["metrics"]) == 103, method + " evaluator coverage")
        require(not any(isinstance(x, dict) and "error" in x for x in summary["metrics"]), method + " evaluator errors")
        csv_path = out / "evaluation/leaderboard_metrics.csv"
        require(sha(csv_path) == val["metrics_csv_sha256"] == ev["metrics_csv_sha256"], method + " metrics SHA")
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fields = reader.fieldnames
        require(len(rows) == 106 and [x["subject_id"] for x in rows[-3:]] == ["mean", "std", "median"], method + " CSV rows")
        # Official MET retains the NIfTI suffix; the manifest stores bare case IDs.
        require({x["subject_id"].removesuffix(".nii.gz") for x in rows[:-3]} == ids, method + " CSV IDs")
        csv_data[method] = (rows, fields)
        report["official_met"][method] = {"job_id": val["job_id"], "selected_epoch": val["selected_epoch"], "checkpoint_sha256": expected["checkpoint_sha256"], "prediction_count": 103, "geometry_mismatch_count": 0, "metrics_csv_sha256": sha(csv_path), "means": {k:numeric(v) if math.isfinite(numeric(v)) else None for k,v in rows[-3].items() if k != "subject_id"}}
    for method in METHODS:
        require(csv_data[method][1] == csv_data["B"][1] and [r["subject_id"] for r in csv_data[method][0]] == [r["subject_id"] for r in csv_data["B"][0]], method + " paired case order")

    paired_path = r4 / "cohort/fixed103/paired_comparison/PAIRED_COMPARISON.json"
    paired = read_json(paired_path)
    val = read_json(r4 / "evidence/PAIRED_COMPARISON_VALIDATION.json")
    contract(val)
    require(val["status"] == "pass" and val["case_count"] == 103 and val["pair_count"] == 10 and val["method_count"] == 5, "paired validation")
    require((r4 / "evidence/PAIRED_COMPARISON_COMPLETE.ok").exists(), "paired complete")
    require(sha(paired_path) == val["paired_comparison_sha256"], "paired SHA")
    require(paired["status"] == "pass" and paired["case_count"] == 103 and list(paired["methods"]) == list(METHODS) and len(paired["pairs"]) == 10, "paired shape")
    require(paired["bootstrap"] == {"replicates":20000,"seed":20260831}, "bootstrap contract")
    rng = np.random.default_rng(20260831)
    verified_metrics = 0
    exclusions = {}
    for pair, names in zip(paired["pairs"], itertools.combinations(METHODS, 2)):
        a_name, b_name = names
        require((pair["method_a"], pair["method_b"]) == names, "pair order")
        for record, metric in zip(pair["metrics"], csv_data["B"][1][1:]):
            require(record["metric"] == metric, "paired metric order")
            a = np.asarray([numeric(r[metric]) for r in csv_data[a_name][0][:-3]])
            b = np.asarray([numeric(r[metric]) for r in csv_data[b_name][0][:-3]])
            mask = np.isfinite(a) & np.isfinite(b)
            n = int(mask.sum())
            require(record["paired_complete_case_count"] == n and record["excluded_nonfinite_pair_count"] == 103 - n, "paired finite count")
            if n < 2:
                require(record["status"] == "unavailable_insufficient_complete_pairs", "paired unavailable metric")
                continue
            sign = -1 if record["direction"] == "lower_is_better" else 1
            benefit = (a[mask] - b[mask]) * sign
            require(record["wins_a"] == int((benefit > 0).sum()) and record["ties"] == int((benefit == 0).sum()) and record["wins_b"] == int((benefit < 0).sum()), "paired wins/ties/losses")
            samples = benefit[rng.integers(0, n, size=(20000, n))].mean(axis=1)
            ci = np.quantile(samples, (0.025, 0.975))
            p = min(1., 2 * min(float(np.mean(samples <= 0)), float(np.mean(samples >= 0))))
            require(np.allclose([record["benefit_mean_a_over_b"],record["benefit_ci95_low"],record["benefit_ci95_high"],record["paired_two_sided_bootstrap_p"]],[float(benefit.mean()),float(ci[0]),float(ci[1]),p],rtol=0,atol=1e-12), "bootstrap reproduction mismatch")
            verified_metrics += 1
            if n != 103:
                exclusions[a_name + "/" + b_name + "/" + metric] = 103 - n
    report["paired"] = {"methods":5,"pairs":10,"cases":103,"bootstrap_replicates":20000,"bootstrap_seed":20260831,"numeric_records_reproduced":verified_metrics,"per_metric_nonfinite_exclusions":exclusions,"json_sha256":sha(paired_path)}
    report["remote_verified_completed_utc"] = datetime.now(timezone.utc).isoformat()
    inventory = []
    total = 0
    for name, root in ROOTS.items():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or not selected_light_file(root, path):
                continue
            size = path.stat().st_size
            require(size <= 16 * 1024 * 1024, "unexpected large light artifact " + str(path))
            total += size
            require(total <= 100 * 1024 * 1024, "light payload exceeds 100 MiB")
            data = path.read_bytes()
            inventory.append({"root_key":name,"relative_path":str(path.relative_to(root)),"size":size,"sha256":hashlib.sha256(data).hexdigest(),"content_base64":base64.b64encode(data).decode("ascii")})
    report["light_file_count"] = len(inventory)
    report["light_total_bytes"] = total
    return {"report":report,"files":inventory}


try:
    result = main()
except Exception as exc:
    result = {"report":{**META,"status":"failed","error_type":type(exc).__name__,"error":str(exc),"remote_utc":datetime.now(timezone.utc).isoformat()},"files":[]}
print("ECNU_AUDIT_JSON_BEGIN")
print(json.dumps(result,allow_nan=False))
print("ECNU_AUDIT_JSON_END")
