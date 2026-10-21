"""Mechanically copy immutable r1 inputs into a new independent candidate."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "s2_k1_small_lesion_p0_20261013_r1"
OLD_PLAN_SHA = "d9d353513e5ce6e95e905c843f6174db17c3e651f8a060fff07876b2edbcf387"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan_path = SOURCE / "CPU_EXECUTION_PLAN.json"
    if sha(plan_path) != OLD_PLAN_SHA:
        raise RuntimeError("r1 plan drift")
    plan = json.loads(plan_path.read_text())
    paths = list(plan["static_sha256"]) + ["runtime/audit_existing.py"]
    paths += [str(p.relative_to(SOURCE)) for p in (SOURCE / "tests").glob("*.py")]
    for name in paths:
        old, new = SOURCE / name, ROOT / name
        if name in plan["static_sha256"] and sha(old) != plan["static_sha256"][name]:
            raise RuntimeError("r1 static drift: " + name)
        if new.exists():
            raise RuntimeError("new target already exists: " + name)
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old, new)
    provenance = ROOT / "sources/r1_preserved"
    provenance.mkdir()
    for old in (plan_path,
                SOURCE / "operator/stop_known_unsafe_cpu_20261013T130539Z/STOP_RECEIPT.json",
                SOURCE / "operator/stop_known_unsafe_cpu_20261013T130539Z/LOCAL_OPERATION_RECEIPT.json"):
        shutil.copyfile(old, provenance / old.name)
    receipt = dict(artifact_status="experimental_unvalidated", operator_approved=False,
                   formal_gate_status="not_run_not_passed", status="copied_candidate_not_frozen",
                   local_utc=datetime.now(timezone.utc).isoformat(), source_root=str(SOURCE),
                   r1_plan_sha256=OLD_PLAN_SHA, source_sha256={n: sha(SOURCE/n) for n in paths},
                   old_runtime_modified=False, remote_operation=False)
    with (ROOT / "operator/PREPARATION_RECEIPT.json").open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps(dict(status=receipt["status"], files=len(paths))))


if __name__ == "__main__":
    main()
