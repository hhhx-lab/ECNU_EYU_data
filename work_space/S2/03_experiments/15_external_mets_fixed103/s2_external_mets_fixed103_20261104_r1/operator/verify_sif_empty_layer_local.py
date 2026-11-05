"""Python 3.10 local regression only; no SSH, installs, models, or submission."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from sif_empty_layer_sources import ROOT, ORIGINAL_SHA256, candidate_source


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("This reproduction requires existing Conda Python 3.10")
    audit = ROOT / "operator" / ("verify_sif_empty_layer_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    audit.mkdir(exist_ok=False)
    source = candidate_source()
    with (audit / "CANDIDATE_SOURCE.json").open("x") as out:
        json.dump(dict(source=source, sha256=hashlib.sha256(source.encode()).hexdigest(),
                       original_sha256=ORIGINAL_SHA256, remotely_executed=False), out, indent=2, sort_keys=True)
        out.write("\n")
    with (audit / "UNITTEST.log").open("x") as log:
        result = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", str(ROOT / "tests"),
                                 "-p", "test_sif_empty_layer.py", "-v"], stdout=log, stderr=subprocess.STDOUT, timeout=60)
    report = dict(status="pass" if result.returncode == 0 else "failed",
                  artifact_status="experimental_unvalidated", operator_approved=False,
                  formal_gate_status="not_run_not_passed", local_utc=datetime.now(timezone.utc).isoformat(),
                  python=sys.executable, python_version=sys.version, test_log_sha256=sha(audit / "UNITTEST.log"),
                  candidate_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  frozen_original_sha256=sha(ROOT / "runtime/normalize_oci_hardlinks.py"),
                  source_sha256={str(p.relative_to(ROOT)): sha(p) for p in (
                      Path(__file__).resolve(), ROOT / "operator/sif_empty_layer_sources.py",
                      ROOT / "tests/test_sif_empty_layer.py", ROOT / "tests/test_normalize_oci_hardlinks.py")},
                  remote_writes=False, jobs_created=False, packages_installed=False,
                  original_failed_job_preserved="4304987", full_SIF_pass=False, GPU_pass=False)
    with (audit / "VERIFICATION.json").open("x") as out:
        json.dump(report, out, indent=2, sort_keys=True)
        out.write("\n")
    print(json.dumps(dict(audit=str(audit), status=report["status"], python_version=sys.version)), flush=True)
    if result.returncode:
        raise RuntimeError("Local Python 3.10 regression failed; no automatic retry")


if __name__ == "__main__":
    main()
