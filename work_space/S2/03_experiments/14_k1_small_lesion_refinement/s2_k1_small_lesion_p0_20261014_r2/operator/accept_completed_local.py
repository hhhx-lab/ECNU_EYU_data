"""Independent local count/identity/SHA acceptance of the completed CPU scope."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "operator/pull_observed_lightweight_20261015T070910Z"
PULLED = AUDIT / "remote_exact"
SNAPSHOT = ROOT / "operator/snapshot_once_20261015T070755Z/PROGRESS_SNAPSHOT.json"
PLAN_SHA = "e3169bf735a39d05c066ca34c5568285b48ba060c79007331a850c903c936beb"
META = dict(artifact_status="experimental_unvalidated", operator_approved=False,
            formal_gate_status="not_run_not_passed")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    return json.loads((PULLED/name).read_text())


def csv_rows(name):
    with (PULLED/name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    snapshot = json.loads(SNAPSHOT.read_text())
    require("4068477|s2k1p0cpu2|COMPLETED|0:0|" in snapshot["sacct"]["stdout"], "terminal not successful")
    require(not snapshot["critical_errors"] and not snapshot["static_sha_drift"], "error or drift")
    require(not any("FAILED" in name for name in snapshot["evidence"]), "FAILED evidence present")
    require(sha(PULLED/"CPU_EXECUTION_PLAN.json") == PLAN_SHA, "plan drift")
    for name, record in snapshot["lightweight_files"].items():
        require(sha(PULLED/name) == record["sha256"], "snapshot transfer SHA drift: " + name)
    for name, digest in load("evidence/LIGHTWEIGHT_SHA256.json")["files"].items():
        require(sha(PULLED/name) == digest, "completion manifest drift: " + name)
    require(load("evidence/P0_CPU_VALIDATION.json")["status"] == "pass", "CPU validation missing")
    require((PULLED/"evidence/P0_CPU_COMPLETE.ok").is_file(), "COMPLETE missing")
    plan = load("CPU_EXECUTION_PLAN.json")
    cases = set(plan["case_ids"])
    require(len(cases) == 103 and snapshot["roundtrip_nifti_count"] == 103, "case count")
    geometry = load("roundtrip/GEOMETRY.json")["cases"]
    require(len(geometry) == 103 and {r["subject_id"] for r in geometry} == cases, "geometry coverage")
    for row in geometry:
        require(len(row["prediction_sha256"]) == 64, "roundtrip prediction digest missing")
    matrix = csv_rows("results/cpu_identity/REFERENCE_DETECTION_MATRIX.csv")
    counts = csv_rows("results/cpu_identity/CASE_COUNTS.csv")
    require(len(matrix) == 6762 and len(counts) == 1236, "matrix/count rows")
    regions = {"wt", "tc", "et", "rc"}
    methods = {"K1", "K4", "label_roundtrip"}
    require(Counter(r["method"] for r in matrix) == Counter({m:2254 for m in methods}), "method denominator")
    identities = defaultdict(dict)
    derived = defaultdict(lambda: dict(small_tp=0, small_fn=0))
    fingerprints = defaultdict(list)
    for row in matrix:
        case, region, method = row["subject_id"], row["region"], row["method"]
        require(case in cases and region in regions and method in methods, "unexpected matrix key")
        identity = row["stable_reference_id"]
        key = (case, region, identity)
        require(method not in identities[key], "duplicate reference method")
        identities[key][method] = row
        volume = float(row["original_volume_mm3"])
        count = int(row["original_voxel_count"])
        geom = json.loads(row["geometry_json"])
        require(math.isclose(volume, count * math.prod(geom["spacing"]), rel_tol=1e-12), "physical volume wrong")
        small = row["small"] == "True"
        require(small == (volume < 27), "small eligibility wrong")
        matched = int(row["is_matched"])
        dsc = float(row["sq_dsc"]) if row["sq_dsc"] else None
        require(matched in (0,1) and (not matched or (dsc is not None and math.isfinite(dsc) and 0<=dsc<=1)), "invalid DSC")
        detected = matched == 1 and dsc >= .2
        require((row["detected"] == "True") == detected, "threshold wrong")
        if small:
            derived[(case,region,method)]["small_tp"] += int(detected)
            derived[(case,region,method)]["small_fn"] += int(not detected)
        fingerprints[(case,region,method)].append((count,volume,matched,dsc or 0))
        require(row["transport_category"] == "not_evaluated_do_not_infer_disappearance", "unsupported transport claim")
    require(len(identities) == 2254 and all(set(v) == methods for v in identities.values()), "full identity coverage")
    for values in identities.values():
        sample = next(iter(values.values()))
        for row in values.values():
            require(all(row[k] == sample[k] for k in ("reference_sha256","original_voxel_count","original_volume_mm3","small","geometry_json")), "cross-method identity drift")
    prior = {m:{c["subject_name"].removesuffix('.nii.gz'):c for c in load('sources/'+m+'/panoptica_evaluation_summary.json')["metrics"]} for m in ("K1","K4")}
    official = load("results/cpu_identity/panoptica_evaluation_summary.json")
    require(not official["missings"] and len(official["metrics"]) == 103, "official missing cases")
    prior["label_roundtrip"] = {c["subject_name"].removesuffix('.nii.gz'):c for c in official["metrics"]}
    require(all(set(v) == cases for v in prior.values()), "official cohort differs")
    keyed_counts = {}
    for row in counts:
        key = (row["subject_id"],row["region"],row["method"])
        require(key not in keyed_counts, "duplicate case count")
        keyed_counts[key] = row
    require(set(keyed_counts) == {(c,r,m) for c in cases for r in regions for m in methods}, "case count coverage")
    for key,row in keyed_counts.items():
        case,region,method = key
        require(all(int(row[k]) == derived[key][k] for k in ("small_tp","small_fn")), "case TP/FN drift")
        original = prior[method][case][region]
        require(int(row["all_fp"]) == original["fp"], "case FP drift")
        expected = sorted((r["voxel_count"],r["volume"],r.get("is_matched",0),r.get("sq_dsc",0) or 0) for r in original["reference_instances"])
        observed = sorted(fingerprints[key])
        require(len(expected) == len(observed), "reference count mismatch")
        require(all(all(math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-10) for a,b in zip(x,y)) for x,y in zip(expected,observed)), "official fingerprint mismatch")
    small = [v for (c,r,i),v in identities.items() if r == "wt" and v["K1"]["small"] == "True"]
    small_cases = {v["K1"]["subject_id"] for v in small}
    require(len(small) == 286 and len(small_cases) == 46, "WT small denominator")
    def hit(value,method):
        return value[method]["detected"] == "True"
    stats = dict(k1_tp=sum(hit(v,'K1') for v in small),k4_tp=sum(hit(v,'K4') for v in small),
                 common_missed_k1_k4=sum(not hit(v,'K1') and not hit(v,'K4') for v in small),
                 new_k4_vs_k1=sum(hit(v,'K4') and not hit(v,'K1') for v in small),
                 lost_k4_vs_k1=sum(hit(v,'K1') and not hit(v,'K4') for v in small),
                 roundtrip_tp=sum(hit(v,'label_roundtrip') for v in small))
    require(stats == load("evidence/OFFICIAL_IDENTITY_VALIDATION.json")["statistics"], "remote statistics mismatch")
    table = []
    for method in ('K1','K4','label_roundtrip'):
        tp = sum(hit(v,method) for v in small)
        fp = sum(int(keyed_counts[(case,'wt',method)]["all_fp"]) for case in small_cases)
        table.append(dict(method=method,small_tp=tp,small_fn=286-tp,all_wt_fp_in46=fp,recall=tp/286))
    leaderboard = csv_rows("results/cpu_identity/leaderboard_metrics.csv")
    require(len(leaderboard)==106 and {r['subject_id'].removesuffix('.nii.gz') for r in leaderboard if r['subject_id'] not in ('mean','std','median')}==cases,"MET CSV count")
    require(len(leaderboard[0]) == 73, "official metric columns")
    for method in methods:
        local_csv = leaderboard if method=='label_roundtrip' else csv_rows('sources/'+method+'/leaderboard_metrics.csv')
        mean = next(r for r in local_csv if r['subject_id']=='mean')
        for row in table:
            if row['method']==method:
                row['case_mean_small_f1_wt'] = float(mean['small_instance_f1_wt'])
    result = dict(**META,status="pass_cpu_scope_only",local_utc=datetime.now(timezone.utc).isoformat(),
        source_snapshot_sha256=sha(SNAPSHOT),remote_snapshot_utc=snapshot['remote_utc'],job_id='4068477',
        plan_sha256=PLAN_SHA,case_count=103,reference_count=2254,detection_record_count=6762,
        case_count_rows=1236,official_metric_columns=72,wt_small_cases=46,wt_small_instances=286,
        statistics=stats,small_wt_comparison=table,sha_checked=len(snapshot['lightweight_files']),
        full_p0_complete=False,instance_transport_classification='not_evaluated',scientific_scope='exploratory_cpu_fidelity_and_two_route_identity_only')
    with (AUDIT / 'LOCAL_CPU_ACCEPTANCE.json').open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write('\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
