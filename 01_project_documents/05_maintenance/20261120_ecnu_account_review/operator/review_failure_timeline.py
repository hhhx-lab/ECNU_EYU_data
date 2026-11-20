"""Validate the saved snapshot locally; never connect or change remote jobs."""
import csv
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "operator/failure_timeline_20261120T104954Z"
FIELDS = ["job_id", "name", "state", "exit_code", "submit_beijing", "start_beijing",
          "end_beijing", "elapsed", "nodes", "requested_memory", "max_rss", "max_vm_size"]
BEIJING = timezone(timedelta(hours=8))


def parse_accounting(raw):
    rows = []
    for cells in csv.reader(io.StringIO(raw), delimiter="|"):
        if not cells:
            continue
        # --parsable2 retains an empty final value; --parsable adds a delimiter.
        if len(cells) == len(FIELDS) + 1 and cells[-1] == "":
            cells.pop()
        if len(cells) != len(FIELDS):
            raise ValueError(f"Expected {len(FIELDS)} fields, got {len(cells)}")
        rows.append(dict(zip(FIELDS, (cell.strip() for cell in cells))))
    return rows


def self_test():
    tests = 0
    for values in (["x"] * 12, ["x"] * 10 + ["", ""], ["x"] * 11 + [""]):
        for extra in ("", "|"):
            assert list(parse_accounting("|".join(values) + extra)[0].values()) == values
            tests += 1
    for values in (["x"] * 11, ["x"] * 13, ["x"] * 12 + ["", ""]):
        try:
            parse_accounting("|".join(values))
        except ValueError:
            tests += 1
        else:
            raise AssertionError("Malformed field count accepted")
    assert parse_accounting("\n") == []
    return tests + 1


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def main():
    tests = self_test()
    result = json.loads((AUDIT / "RESULT.json").read_text())
    assert result["query_timezone"] == result["query_tz_env"] == "Asia/Shanghai"
    assert datetime.fromisoformat(result["query_local_clock"]).utcoffset() == timedelta(hours=8)
    assert result["accounting"]["returncode"] == 0 and not result["accounting"]["stderr"]
    assert result["errors"] == ["Unparsed accounting row"] * 72
    rows = parse_accounting(result["accounting"]["stdout"])
    by_id = {r["job_id"]: r for r in rows}
    assert len(rows) == len(by_id) == 144
    with (ROOT / "NOV12_JOBS.csv").open(newline="") as handle:
        prior = list(csv.DictReader(handle))
    assert len(prior) == 72
    allocations = [by_id[r["job_id"]] for r in prior]
    for old, new in zip(prior, allocations):
        for old_key, new_key in (("job_id", "job_id"), ("name", "name"), ("state", "state"),
                                 ("exit_code", "exit_code"), ("submit", "submit_beijing"),
                                 ("start", "start_beijing"), ("end", "end_beijing")):
            assert old[old_key] == new[new_key], (old["job_id"], old_key)
        batch = by_id[new["job_id"] + ".batch"]
        if new["state"] == "COMPLETED":
            assert new["exit_code"] == batch["exit_code"] == "0:0" and batch["state"] == "COMPLETED"
    assert Counter(r["state"] for r in allocations) == {"COMPLETED": 54, "FAILED": 17, "TIMEOUT": 1}
    failure_evidence = []
    manifests = set()
    expected_nodes = {"B0p": "gpu15", "BL": "gpu12", "BU": "gpu13", "BUL": "gpu13"}
    assert len(result["failed_training"]) == 17
    for run, record in result["failed_training"].items():
        allocation, batch = by_id[record["job_id"]], by_id[record["job_id"] + ".batch"]
        assert allocation["state"] == batch["state"] == "FAILED"
        assert allocation["exit_code"] == batch["exit_code"] == "75:0"
        assert allocation["name"] == "rcb_" + run
        assert len(record["evidence"]) == 1
        entry = record["evidence"][0]
        assert entry["exists"] and Path(entry["path"]).name == f"RUN_{run}_NOSHM_{record['job_id']}.json"
        marker = entry["value"]
        assert marker["run"] == run and marker["job_id"] == record["job_id"]
        assert marker["status"] == "noshm" and marker["attempt"] == 0
        assert marker["node"] == allocation["nodes"] == expected_nodes[run.split("_")[0]]
        assert marker["cleanup"]["child_returncode"] is None
        assert "NoShm" in marker["error"] and "1200 s; need 42 GB + 20 GB margin" in marker["error"]
        assert "21 checks" in marker["error"]
        manifests.add(marker["manifest_sha256"])
        start = datetime.fromisoformat(allocation["start_beijing"]).replace(tzinfo=BEIJING)
        end = datetime.fromisoformat(allocation["end_beijing"]).replace(tzinfo=BEIJING)
        assert start <= datetime.fromisoformat(marker["started_utc"]) < end
        assert abs((datetime.fromisoformat(marker["ended_utc"]) - end).total_seconds()) < 1
        failure_evidence.append(dict(run=run, job_id=record["job_id"], node=marker["node"],
            start_beijing=allocation["start_beijing"], end_beijing=allocation["end_beijing"],
            elapsed=allocation["elapsed"], error=marker["error"], marker_path=entry["path"],
            remote_marker_sha256=entry["sha256"], marker_started_utc=marker["started_utc"],
            marker_ended_utc=marker["ended_utc"], training_not_started=True))
    assert len(manifests) == 1
    logs = "\n".join(r["text"] for r in result["rc_scan_failed_logs"])
    assert "Detected 12 oom-kill event(s)" in logs and "DUE TO TIME LIMIT" in logs
    assert by_id["4388852"]["state"] == "TIMEOUT" and by_id["4388852"]["exit_code"] == "0:0"
    assert by_id["4388852.batch"]["state"] == "OUT_OF_MEMORY" and by_id["4388852.batch"]["exit_code"] == "0:125"
    for entry in result["sources"]:
        if entry.get("exists") and "text" in entry:
            assert hashlib.sha256(entry["text"].encode()).hexdigest() == entry["sha256"]
    for entry in result["rc_scan_failed_logs"] + [log for r in result["failed_training"].values() for log in r["logs"]]:
        assert hashlib.sha256(entry["text"].encode()).hexdigest() == entry["excerpt_sha256"]
    receipt = json.loads((AUDIT / "LOCAL_OPERATION_RECEIPT.json").read_text())
    assert receipt["preexisting_vpn_preserved"] and not receipt["vpn_owned"]
    with (ROOT / "NOV12_TIMELINE_BEIJING.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(allocations)
    write_json(ROOT / "FAILURE_TIMELINE_REVIEW.json", dict(
        status="pass", observation_kind="local_replay_of_saved_read_only_snapshot",
        reviewed_utc=datetime.now(timezone.utc).isoformat(), query_timezone="Asia/Shanghai",
        artifact_status="experimental_unvalidated", operator_approved=False, formal_gate_status="not_run_not_passed",
        original_snapshot_status=result["status"], original_parse_errors_preserved=72,
        correction="Preserve empty last field with csv.reader; no SSH or remote retry",
        self_tests=tests, accounting_rows=144, allocations=72, allocation_states=dict(Counter(r["state"] for r in allocations)),
        prior_72_allocation_times_unchanged=True, failed_training=failure_evidence,
        observed_training_marker_manifest_sha256=sorted(manifests),
        scan=dict(job_id="4388852", allocation=by_id["4388852"], batch=by_id["4388852.batch"], oom_events=12),
        source_snapshot=str((AUDIT / "RESULT.json").relative_to(ROOT)), source_snapshot_sha256=sha(AUDIT / "RESULT.json"),
        local_operator_sha256=sha(Path(__file__)), timeline_csv_sha256=sha(ROOT / "NOV12_TIMELINE_BEIJING.csv"),
        remote_utc=result["remote_utc"], local_received_utc=result["local_received_utc"],
        identity_boundary="Marker JSON contains parsed values and remote SHA, not byte-for-byte original JSON; full package/checkpoints not revalidated",
        boundaries=dict(no_new_ssh=True, no_job_changes=True, no_image_or_model_reads=True, preexisting_vpn_preserved=True)))
    print(json.dumps(dict(status="pass", allocations=72, original_noshm_markers=17, self_tests=tests)))


if __name__ == "__main__":
    main()
