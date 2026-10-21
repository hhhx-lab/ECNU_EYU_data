"""Read-only single-source SCP calls using an existing terminal SHA snapshot."""
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "operator/snapshot_once_20261015T070755Z/PROGRESS_SNAPSHOT.json"
HELPER = ROOT.parents[2] / "03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260923_r3_fallback/operator/access.py"


def main():
    spec = importlib.util.spec_from_file_location("transport", HELPER)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    transport.ROOT = ROOT
    access = transport.Access("pull_observed_lightweight")
    start = datetime.now(timezone.utc).isoformat()
    outcome = "failed_preserved"
    try:
        observed = json.loads(SOURCE.read_text())
        assert "4068477|s2k1p0cpu2|COMPLETED|0:0|" in observed["sacct"]["stdout"]
        remote = json.loads((ROOT / "CPU_EXECUTION_PLAN.json").read_text())["remote_root"]
        pulled = access.audit / "remote_exact"
        pulled.mkdir()
        access.start_vpn()
        parents = sorted({name.split('/')[0] for name in observed["lightweight_files"] if '/' in name} - {"roundtrip"})
        assert set(parents) <= {"evidence", "results", "logs", "manifests", "slurm", "sources"}
        # Each scp has only one source and therefore one authentication exchange.
        for parent in parents:
            access.call(["scp", "-r", "-P", access.env["HPC_LOGIN_PORT"], "-o", "StrictHostKeyChecking=yes",
                         "-o", "ConnectTimeout=15", f'{access.env["HPC_SSH_USER"]}@{access.env["HPC_LOGIN_HOST"]}:{remote}/{parent}', str(pulled)], timeout=120)
        for name in ("CPU_EXECUTION_PLAN.json", "roundtrip/GEOMETRY.json"):
            if name in observed["lightweight_files"]:
                target = pulled / name
                target.parent.mkdir(parents=True, exist_ok=True)
                access.call(["scp", "-P", access.env["HPC_LOGIN_PORT"], "-o", "StrictHostKeyChecking=yes",
                             "-o", "ConnectTimeout=15", f'{access.env["HPC_SSH_USER"]}@{access.env["HPC_LOGIN_HOST"]}:{remote}/{name}', str(target)], timeout=120)
        mismatches, json_count = [], 0
        for name, expected in observed["lightweight_files"].items():
            p = pulled / name
            if not p.is_file() or transport.sha(p) != expected["sha256"]:
                mismatches.append(name)
            if p.is_file() and p.suffix == ".json":
                json.loads(p.read_text())
                json_count += 1
        forbidden = [str(p.relative_to(pulled)) for p in pulled.rglob('*') if p.is_file() and p.name.endswith(('.nii.gz','.pth','.npz','.jsonl'))]
        assert not mismatches and not forbidden, "transfer SHA/exclusion mismatch"
        outcome = "pass"
        transport.exclusive(access.audit / "LIGHTWEIGHT_PULL_VALIDATION.json",dict(**transport.META,status=outcome,
            source_snapshot_sha256=transport.sha(SOURCE),source_snapshot=str(SOURCE),file_count=len(observed["lightweight_files"]),
            json_parsed=json_count,sha_mismatches=mismatches,forbidden_files=forbidden,progress_snapshots=0,remote_writes=False))
        print(json.dumps(dict(status=outcome,file_count=len(observed["lightweight_files"]),json_parsed=json_count,audit=str(access.audit))))
    finally:
        access.close_vpn()
        transport.exclusive(access.audit / "LOCAL_OPERATION_RECEIPT.json",dict(**transport.META,status=outcome,
            local_start_utc=start,local_end_utc=datetime.now(timezone.utc).isoformat(),vpn_closed=access.owns_vpn,
            preexisting_vpn_preserved=not access.owns_vpn,readonly=True,progress_snapshots=0))


if __name__ == "__main__":
    main()
