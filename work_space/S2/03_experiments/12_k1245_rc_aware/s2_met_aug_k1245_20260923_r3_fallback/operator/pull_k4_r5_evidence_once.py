"""One-shot, read-only pull of K4 R5 lightweight evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path

OPERATOR = Path(__file__).resolve().parent
sys.path.insert(0, str(OPERATOR))
import access  # noqa: E402


REMOTE_ROOT = access.REMOTE / "attempts/k4_small_smoke_r5"
ALLOWED_NAMES = {
    "K4_SMALL_SMOKE_AUTOMATIC.json",
    "K4_SMALL_SMOKE_R5_VALIDATION.json",
    "manual_review.csv",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing existing output: {output}")
    output.mkdir(parents=True)
    pull = output / "pulled"
    pull.mkdir()
    op = access.Access("pull_k4_r5_evidence")
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    result = None
    error = None
    try:
        op.start_vpn()
        listing_code = (
            "import pathlib,json;"
            f"root=pathlib.Path({str(REMOTE_ROOT)!r});"
            "allowed={'K4_SMALL_SMOKE_AUTOMATIC.json','K4_SMALL_SMOKE_R5_VALIDATION.json','manual_review.csv'};"
            "files=[];"
            "[files.append(str(p.relative_to(root))) for p in root.rglob('*') if p.is_file() and (p.name in allowed or p.suffix.lower() in {'.png','.jpg','.jpeg','.out','.err','.log'})];"
            "print(json.dumps({'files':sorted(files)},sort_keys=True))"
        )
        listing = json.loads(op.ssh(
            "env PYTHONDONTWRITEBYTECODE=1 " + access.PYTHON + " -B -c " + shlex.quote(listing_code)
        ).splitlines()[-1])
        files = listing["files"]
        required = [name for name in files if Path(name).name in ALLOWED_NAMES]
        missing = sorted(ALLOWED_NAMES - {Path(name).name for name in files})
        if missing:
            raise RuntimeError("missing required remote evidence: " + ", ".join(missing))
        for rel in files:
            rel_path = Path(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                raise RuntimeError(f"unsafe remote relative path: {rel}")
            target = pull / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            op.call([
                "scp", "-P", op.env["HPC_LOGIN_PORT"], "-o", "StrictHostKeyChecking=yes",
                "-o", "ConnectTimeout=15",
                f"{op.env['HPC_SSH_USER']}@{op.env['HPC_LOGIN_HOST']}:{REMOTE_ROOT / rel_path}",
                str(target),
            ])
        hashes = {str(p.relative_to(pull)): digest(p) for p in sorted(pull.rglob("*")) if p.is_file()}
        result = {"remote_root": str(REMOTE_ROOT), "files": sorted(hashes), "sha256": hashes,
                  "required": sorted(required), "missing_required": missing}
    except Exception as exc:
        error = repr(exc)
    finally:
        try:
            op.close_vpn()
        finally:
            receipt = {
                **access.META,
                "action": "pull_k4_r5_evidence_once",
                "local_started_utc": started,
                "local_completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "vpn_closed": op.owns_vpn,
                "preexisting_vpn_left_unchanged": not op.owns_vpn,
                "remote_root": str(REMOTE_ROOT),
                "result": result,
                "error": error,
            }
            access.exclusive(op.audit / "LOCAL_OPERATION_RECEIPT.json", receipt)
    if error:
        raise SystemExit(error)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
