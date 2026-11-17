"""Verify archived identities and derive round availability without submitting."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "2023": ("syn51156910", "9615344", "2023-08-14T23:59:00.000Z"),
    "2024": ("syn53708249", "9615545", "2024-08-07T23:59:00.000Z"),
    "2025": ("syn64153130", "9615901", "2025-08-07T23:59:00.000Z"),
}


def read(relative):
    return json.loads((ROOT / relative).read_text())


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def save(relative, value):
    path = ROOT / relative
    if path.exists():
        raise ValueError("derived evidence already exists; preserve the earlier record")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    now = datetime.now(timezone.utc)
    sources = []
    for receipt_path in sorted(ROOT.rglob("*.source.json")):
        receipt = json.loads(receipt_path.read_text())
        response = ROOT / receipt["response_path"]
        if response.stat().st_size != receipt["bytes"]:
            raise ValueError("response size changed")
        if hashlib.sha256(response.read_bytes()).hexdigest() != receipt["sha256"]:
            raise ValueError("response SHA changed")
        sources.append(receipt)
    rows = []
    for year, (project, evaluation_id, end) in EXPECTED.items():
        prefix = "sources/" + year + "/"
        evaluation = read(prefix + "evaluation.json")
        rounds = read(prefix + "rounds.json")
        dashboard = read(prefix + "dashboard.json")
        mets = read(prefix + "mets.json")
        if evaluation["id"] != evaluation_id or evaluation["contentSource"] != project:
            raise ValueError("wrong year or task")
        if "evaluation_" + evaluation_id not in unquote(dashboard["markdown"]):
            raise ValueError("queue is not bound to the official METS dashboard")
        if rounds.get("nextPageToken") or len(rounds["page"]) != 1:
            raise ValueError("round list is incomplete or differs from the reviewed snapshot")
        round_ = rounds["page"][0]
        if round_["evaluationId"] != evaluation_id or round_["roundEnd"] != end:
            raise ValueError("round identity differs from the reviewed snapshot")
        if timestamp(round_["roundStart"]) >= timestamp(end):
            raise ValueError("invalid round interval")
        active = timestamp(round_["roundStart"]) <= now < timestamp(end)
        if active:
            raise ValueError("round has become active; revise the report before concluding closure")
        explicit_closed = "Challenge Closed!" in mets["markdown"]
        if year in ("2023", "2024") and not explicit_closed:
            raise ValueError("explicit closure banner is missing")
        rows.append(dict(year=year, project=project, evaluation_id=evaluation_id,
                         name=evaluation["name"], complete_round_list=True,
                         round=round_, active_round_count=0,
                         explicit_closed_banner=explicit_closed,
                         current_standard_validation_submission="closed_by_round_configuration",
                         actual_upload_attempted=False, scoring_runner_liveness="not_tested",
                         authenticated_user_eligibility="not_checked"))
    controls = [read("sources/" + n + ".json") for n in ("active_only_exact_ids", "all_exact_ids")]
    control_ids = [[x["id"] for x in c["results"]] for c in controls]
    if control_ids[0] != control_ids[1] or len(control_ids[0]) != 3:
        raise ValueError("activeOnly control response changed; review it independently")
    browser = []
    for path in sorted((ROOT / "browser").glob("*.json")):
        entry = json.loads(path.read_text())
        body = entry.get("text", entry.get("excerpt", ""))
        if not body or "observedAtUTC" not in entry:
            raise ValueError("empty or undated browser observation")
        browser.append(dict(path=str(path.relative_to(ROOT)), url=entry["url"],
                            received_utc=entry["observedAtUTC"], bytes=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    question = read("browser/2025_reopen_question.json")["text"]
    if "timeline for opening the validation queue submission has not been set yet" not in question:
        raise ValueError("moderator reopening statement not observed")
    for year in EXPECTED:
        observed = read("browser/" + year + "_results.json")
        body = observed.get("text", observed.get("excerpt", ""))
        if not re.search(r"\b97[0-9]{5}\b", body):
            raise ValueError("leaderboard rows were not actually rendered")
    result = dict(reviewed_at_utc=now.isoformat(), scope="BraTS METS validation 2023-2025",
                  rows=rows, active_only_control=dict(same_ids_for_true_and_false=control_ids[0],
                  used_as_open_evidence=False, cause="not_investigated"),
                  source_count=len(sources), successful_sources=sum(x["success"] for x in sources),
                  unsuccessful_sources=[x for x in sources if not x["success"]],
                  browser_observations=browser, identity_checks="pass",
                  no_upload_no_submission_no_ecnu_no_vpn_no_install=True,
                  artifact_status="experimental_unvalidated", operator_approved=False,
                  formal_gate_status="not_run_not_passed")
    save("AVAILABILITY_REVIEW.json", result)
    save("SOURCE_INDEX.json", dict(sources=sources, browser_observations=browser))
    print(json.dumps(dict(identity_checks="pass", sources=len(sources),
                          successful_sources=result["successful_sources"],
                          active_mets_rounds=0, years=list(EXPECTED)), ensure_ascii=False))


if __name__ == "__main__":
    main()
