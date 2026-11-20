"""Verify downloaded Markdown identities and the focused Nov12 accounting table."""
import csv
import importlib.util
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("review_fetch", ROOT / "operator/fetch_nov12_markdown.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)
m = fetch.m


def main():
    audit = ROOT / "operator/nov12_markdown_20261120T092929Z"
    acceptance = json.loads((audit / "DOWNLOAD_ACCEPTANCE.json").read_text())
    result = json.loads((audit / "RESULT.json").read_text())
    assert acceptance["file_count"] == len(acceptance["files"]) == 57
    assert acceptance["total_bytes"] == sum(r["bytes"] for r in acceptance["files"]) == 324533
    for r in acceptance["files"]:
        p = Path(r["local_path"])
        assert p.is_file() and not p.is_symlink() and p.stat().st_size == r["bytes"]
        assert m.sha(p) == r["sha256"]
    rows = []
    for line in result["accounting"]["stdout"].splitlines():
        cells = line.split("|")
        if cells[-1] == "":
            cells.pop()
        assert len(cells) == 8
        rows.append(dict(zip(("job_id", "name", "state", "exit_code", "elapsed", "submit", "start", "end"), cells)))
    by_id = {r["job_id"]: r for r in rows}
    assert len(by_id) == len(rows)
    allocations = [r for r in rows if "." not in r["job_id"]]
    step_state_differences = []
    for r in allocations:
        assert r["submit"].startswith("2026-11-12")
        batch = by_id[r["job_id"] + ".batch"]
        if r["state"] == "COMPLETED":
            assert r["exit_code"] == batch["exit_code"] == "0:0" and batch["state"] == "COMPLETED"
        elif (batch["state"], batch["exit_code"]) != (r["state"], r["exit_code"]):
            step_state_differences.append({"job_id": r["job_id"],
                "allocation": {k:r[k] for k in ("state", "exit_code")},
                "batch": {k:batch[k] for k in ("state", "exit_code")}})
    assert step_state_differences == [{"job_id": "4388852", "allocation": {"state": "TIMEOUT", "exit_code": "0:0"},
        "batch": {"state": "OUT_OF_MEMORY", "exit_code": "0:125"}}]
    rc_train = [r for r in allocations if r["name"].startswith("rcb_B")]
    assert len(rc_train) == 25
    assert Counter((r["state"], r["exit_code"]) for r in rc_train) == Counter({("COMPLETED", "0:0"): 8, ("FAILED", "75:0"): 17})
    versions = defaultdict(Counter)
    for r in rc_train:
        versions[r["name"].removeprefix("rcb_").rsplit("_f", 1)[0]][r["state"]] += 1
    assert dict(versions) == {"B0": Counter(COMPLETED=5), "B0p": Counter(FAILED=5),
        "BL": Counter(FAILED=5), "BU": Counter(FAILED=5), "BUL": Counter(COMPLETED=3, FAILED=2)}
    directory = Path(acceptance["destination"])
    train_report = directory / "s2_rc_train_batch1_20261112_r1/results/batch_summary_4389576/REPORT.md"
    with train_report.open() as f:
        table = [line.strip().split("|")[1:-1] for line in f if line.startswith("| ")]
    statuses = {r[0].strip(): r[3].strip() for r in table if len(r) >= 5 and "_f" in r[0]}
    assert len(statuses) == 25 and Counter(statuses.values()) == Counter({"pass": 8, "not_started": 17})
    for r in rc_train:
        run = r["name"].removeprefix("rcb_")
        assert statuses[run] == ("pass" if r["state"] == "COMPLETED" else "not_started")
    csv_path = ROOT / "NOV12_JOBS.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(allocations[0]))
        writer.writeheader()
        writer.writerows(allocations)
    review = {**m.META, "status": "metadata_and_markdown_review_pass",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "verification_level": "scheduler_allocation_and_batch_plus_downloaded_markdown_not_runtime_or_metric_replay",
        "batches": result["roots"], "allocation_count": len(allocations),
        "allocation_states": dict(Counter(r["state"] for r in allocations)),
        "failed_allocation_batch_differences": step_state_differences,
        "rc_train": {"planned": 25, "completed0": 8, "failed75": 17,
            "version_states": {k:dict(v) for k,v in versions.items()},
            "not_started_reason": "node_shared_memory_shortage_as_recorded_in_server_summary"},
        "download": {k:acceptance[k] for k in ("destination", "file_count", "total_bytes", "local_accepted_utc", "remote_checked_utc")},
        "acceptance_sha256": m.sha(audit / "DOWNLOAD_ACCEPTANCE.json"),
        "accounting_snapshot_sha256": m.sha(audit / "RESULT.json"),
        "jobs_csv_sha256": m.sha(csv_path), "no_job_changes": True,
        "models_downloaded": False, "image_data_downloaded": False,
        "project_completion_not_accepted_from_scheduler_alone": True}
    m.exclusive(ROOT / "REVIEW.json", review)
    print(json.dumps(review, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
