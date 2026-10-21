"""Bounded, one-shot staging and publication of the user-provided dated revision."""
import argparse
import hashlib
import json
import shutil
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[4]
ACTIVE = PROJECT / "work_space/G1/01_code/tumour_diffusion_1mm"
ARCHIVE = Path("/Users/hwaigc/Downloads/TumourDiffusion_1mm_20261013.zip")
STAGED = HERE / "incoming_source"
PREVIOUS = HERE / "previous_source_revision3"
STATE = dict(artifact_status="experimental_unvalidated", operator_approved=False,
             formal_gate_status="not_run_not_passed")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(root):
    paths = sorted(root.rglob("*"))
    if any(p.is_symlink() for p in paths):
        raise ValueError("Symlink in source tree")
    return {p.relative_to(root).as_posix(): dict(sha256=digest(p), bytes=p.stat().st_size)
            for p in paths if p.is_file()}


def record(name, data):
    with (HERE / name).open("x") as output:
        json.dump(dict(**STATE, recorded_at=datetime.now(timezone.utc).isoformat(), **data),
                  output, ensure_ascii=False, indent=2)
        output.write("\n")


def stage():
    if STAGED.exists() or (HERE / "STAGING_RECEIPT.json").exists():
        raise ValueError("Refuse existing staging/evidence")
    previous = manifest(ACTIVE)
    baseline = json.loads((HERE.parent / "20261013_tumour_diffusion_1mm_revision3_readiness/REPLACEMENT_RECEIPT.json").read_text())["revision"]
    if previous != baseline:
        raise ValueError("Active source drift; do not replace")
    entries, seen = {}, set()
    with zipfile.ZipFile(ARCHIVE) as archive:
        infos = archive.infolist()
        if len(infos) > 1000 or sum(i.file_size for i in infos) > 100_000_000:
            raise ValueError("Archive size limit exceeded")
        for info in infos:
            p = PurePosixPath(info.filename)
            if p.is_absolute() or ".." in p.parts or "\\" in info.filename or not p.parts or p.parts[0] != "TumourDiffusion_1mm":
                raise ValueError("Unsafe archive path")
            if info.filename in seen or stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1:
                raise ValueError("Duplicate/symlink/encrypted entry")
            seen.add(info.filename)
            if not info.is_dir():
                blob = archive.read(info)
                entries[PurePosixPath(*p.parts[1:]).as_posix()] = dict(
                    sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob))
        STAGED.mkdir()
        for info in infos:
            dest = STAGED.joinpath(*PurePosixPath(info.filename).parts[1:])
            if info.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("xb") as output:
                    output.write(archive.read(info))
    if manifest(STAGED) != entries:
        raise ValueError("Staged SHA mismatch")
    copy = HERE / ARCHIVE.name
    shutil.copy2(ARCHIVE, copy)
    if digest(copy) != digest(ARCHIVE):
        raise ValueError("Preserved ZIP mismatch")
    comparison = dict(added=sorted(set(entries)-set(previous)),
                      removed=sorted(set(previous)-set(entries)),
                      changed=sorted(p for p in set(entries)&set(previous) if entries[p] != previous[p]),
                      unchanged=sorted(p for p in set(entries)&set(previous) if entries[p] == previous[p]))
    record("STAGING_RECEIPT.json", dict(archive=str(ARCHIVE), archive_sha256=digest(ARCHIVE),
           active_root=str(ACTIVE), staged_root=str(STAGED), previous=previous, revision=entries,
           comparison=comparison, active_unchanged=True))
    print(json.dumps(dict(staged_files=len(entries), previous_files=len(previous), comparison=comparison), indent=2))


def publish():
    receipt = json.loads((HERE / "STAGING_RECEIPT.json").read_text())
    audit = json.loads((HERE / "DATED_REVISION_READINESS_EVIDENCE.json").read_text())
    if PREVIOUS.exists() or (HERE / "REPLACEMENT_RECEIPT.json").exists():
        raise ValueError("Refuse duplicate publication")
    if manifest(ACTIVE) != receipt["previous"] or manifest(STAGED) != receipt["revision"]:
        raise ValueError("Source drift before publication")
    if not audit["source_unchanged"] or any(c["status"] == "probe_error" for c in audit["checks"]):
        raise ValueError("Incomplete audit; do not publish")
    ACTIVE.rename(PREVIOUS)
    try:
        STAGED.rename(ACTIVE)
    except BaseException:
        PREVIOUS.rename(ACTIVE)
        raise
    if manifest(ACTIVE) != receipt["revision"] or manifest(PREVIOUS) != receipt["previous"]:
        raise ValueError("Post-publication mismatch; preserve evidence")
    record("REPLACEMENT_RECEIPT.json", dict(archive_sha256=receipt["archive_sha256"],
        active_root=str(ACTIVE), previous_root=str(PREVIOUS), revision=receipt["revision"],
        comparison=receipt["comparison"], source_files=len(receipt["revision"]),
        active_sha_verified=True, previous_preserved_sha_verified=True, source_edits=False,
        readiness=audit["status"]))
    print("Published exact dated revision; prior revision preserved with verified SHA")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["stage", "publish"])
    action = parser.parse_args().action
    stage() if action == "stage" else publish()
