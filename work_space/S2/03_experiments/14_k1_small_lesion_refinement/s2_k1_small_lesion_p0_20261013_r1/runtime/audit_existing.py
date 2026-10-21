"""Audit existing official summaries without inventing cross-route instance IDs."""
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S2 = ROOT.parents[2]
META = dict(artifact_status="experimental_unvalidated", operator_approved=False,
            formal_gate_status="not_run_not_passed")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def count_region(region):
    refs = region["reference_instances"]
    require(len(refs) == region["n_ref_instances"], "reference count mismatch")
    small_tp = small_fn = 0
    for ref in refs:
        volume = ref["volume"]
        require(math.isfinite(volume) and volume > 0, "invalid reference volume")
        matched = ref.get("is_matched", 0)
        require(matched in (0, 1), "invalid matched flag")
        if matched:
            dsc = ref.get("sq_dsc")
            require(dsc is not None and math.isfinite(dsc) and 0 <= dsc <= 1,
                    "matched instance has undefined/invalid Dice")
        if volume < 27:
            if matched and ref["sq_dsc"] >= 0.2:
                small_tp += 1
            else:
                small_fn += 1
    fp = region["fp"]
    require(isinstance(fp, int) and fp >= 0, "invalid FP")
    return small_tp, small_fn, fp


def unique_cases(rows, key):
    result = {}
    for row in rows:
        case = row[key]
        require(case not in result, "duplicate case: " + case)
        result[case] = row
    return result


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    kroot = S2 / "03_experiments/12_k1245_rc_aware"
    ext = S2 / "03_experiments/09_epoch175_official_met/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact/sources/B/epoch175"
    sources = {
        "K1": kroot / "s2_met_aug_k1_selected103_official_met_20261008_r1/cohort/fixed103/models/K1/evaluation",
        "K4": kroot / "s2_met_aug_k24_selected103_official_met_20261002_r1/cohort/fixed103/models/K4/evaluation",
        "original_E": S2 / "07_submissions/original_e_20260727_20260728/s2_official_validation_inference_20260727_r1/selection_evidence/E",
        "B175": ext / "evaluation",
        "D50": S2 / "03_experiments/07_screen125_official_met/s2_met_aug_btof_selected103_official_met_20260907_r4/cohort/fixed103/models/D/evaluation",
    }
    nine = json.loads((S2 / "03_experiments/13_nine_route_selected103_official_met/results/VALIDATION.json").read_text())
    require(nine["status"] == "pass", "nine-route validation failed")
    identities = {r["method"]: r for r in nine["input_identities"]}
    rows, summary, inputs = [], [], []
    baseline = None
    for route, directory in sources.items():
        source = directory / "panoptica_evaluation_summary.json"
        csv_path = Path(identities[route]["metrics_csv"])
        require(sha(csv_path) == identities[route]["metrics_csv_sha256"], "frozen CSV SHA drift")
        document = json.loads(source.read_text())
        require(not document.get("missings"), "missing official cases")
        cases = unique_cases(document["metrics"], "subject_name")
        with csv_path.open(newline="") as handle:
            csv_rows = [r for r in csv.DictReader(handle) if r["subject_id"] not in ("mean", "std", "median")]
        csv_cases = unique_cases(csv_rows, "subject_id")
        require(len(cases) == 103 and set(cases) == set(csv_cases), "103-case mismatch")
        # This fingerprint checks equal volume distributions, not instance identity.
        distribution = {}
        totals = dict(small_tp=0, small_fn=0, fp_in_small_cases=0, small_cases=0)
        for case, record in cases.items():
            require("error" not in record, "official case error")
            for region_name in ("wt", "et", "tc", "rc"):
                region = record[region_name]
                tp, fn, fp = count_region(region)
                distribution[(case, region_name)] = sorted((r["voxel_count"], r["volume"]) for r in region["reference_instances"])
                expected = csv_cases[case]
                for suffix, value in (("tp", tp), ("fn", fn), ("fp", fp)):
                    raw = expected["small_instance_" + suffix + "_" + region_name]
                    require((not raw and tp + fn == 0) or (raw and float(raw) == value), "official small count mismatch")
                require(float(expected["all_instance_fp_" + region_name]) == fp, "all FP mismatch")
                for index, ref in enumerate(region["reference_instances"]):
                    rows.append(dict(route=route, subject_id=case, region=region_name,
                                     summary_record_index=index, stable_reference_id="",
                                     identity_status="unresolved_not_pairable",
                                     volume=ref["volume"], voxel_count=ref["voxel_count"],
                                     small=ref["volume"] < 27,
                                     is_matched=ref.get("is_matched", 0),
                                     sq_dsc=ref.get("sq_dsc") if ref.get("is_matched") else None))
                if region_name == "wt" and tp + fn:
                    totals["small_tp"] += tp
                    totals["small_fn"] += fn
                    totals["fp_in_small_cases"] += fp
                    totals["small_cases"] += 1
        if baseline is None:
            baseline = distribution
        require(distribution == baseline, "reference volume multiset drift")
        require(totals["small_cases"] == 46 and totals["small_tp"] + totals["small_fn"] == 286, "WT reference denominator drift")
        summary.append(dict(route=route, **totals, micro_recall=totals["small_tp"] / 286,
                            fp_per_small_case=totals["fp_in_small_cases"] / 46))
        inputs.append(dict(route=route, summary_path=str(source), summary_sha256=sha(source),
                           csv_path=str(csv_path), csv_sha256=sha(csv_path),
                           checkpoint_sha256=identities[route]["checkpoint_sha256"]))
    output = ROOT / "results/existing_metrics"
    output.mkdir(parents=True, exist_ok=False)
    with (output / "UNPAIRED_REFERENCE_RECORDS.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "VALIDATION.json", dict(**META, status="pass", phase="P0-A",
        local_utc=datetime.now(timezone.utc).isoformat(), case_count=103, small_wt_case_count=46,
        small_wt_reference_count=286, routes=summary, inputs=inputs, record_count=len(rows),
        reference_volume_multisets_consistent=True, cross_route_reference_identity="unresolved",
        common_missed_intersection=None, new_lost_reference_instances=None,
        scope="Independent counts only; equal volumes/order do not establish identity."))
    (output / "COMPLETE.ok").write_text("P0-A counts pass; reference identity unresolved\nartifact_status=experimental_unvalidated\noperator_approved=false\nformal_gate_status=not_run_not_passed\n")
    write_json(output / "LIGHTWEIGHT_SHA256.json", dict(**META, files={p.name: sha(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(dict(status="pass", rows=len(rows), routes=summary, identity="unresolved")))


if __name__ == "__main__":
    main()
