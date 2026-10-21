#!/usr/bin/env python3
"""Preserve the old source and install the exact, bounded ZIP revision once."""

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
ARCHIVE = Path("/Users/hwaigc/Downloads/TumourDiffusion_1mm(3).zip")
PREVIOUS = HERE / "previous_source_20261009"
STAGING = HERE / "incoming_source"
STATUS = dict(artifact_status="experimental_unvalidated", operator_approved=False,
              formal_gate_status="not_run_not_passed")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(root):
    return {path.relative_to(root).as_posix(): dict(sha256=digest(path), bytes=path.stat().st_size)
            for path in sorted(root.rglob("*")) if path.is_file()}


def main():
    if PREVIOUS.exists() or STAGING.exists() or (HERE / "REPLACEMENT_RECEIPT.json").exists():
        raise SystemExit("Replacement already started; refuse to overwrite evidence")
    previous = manifest(ACTIVE)
    original_manifest = json.loads((HERE.parent / "20261009_tumour_diffusion_1mm_readiness/IMPORT_MANIFEST.json").read_text())
    baseline = {row["path"]: dict(sha256=row["sha256"], bytes=row["bytes"]) for row in original_manifest["files"]}
    stamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    zip_sha = digest(ARCHIVE)
    with zipfile.ZipFile(ARCHIVE) as archive:
        infos = archive.infolist()
        if len(infos) > 1000 or sum(i.file_size for i in infos) > 100_000_000:
            raise SystemExit("Archive exceeds bounded import limits")
        seen, entries = set(), {}
        for info in infos:
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
                raise SystemExit("Unsafe archive path")
            if not path.parts or path.parts[0] != "TumourDiffusion_1mm":
                raise SystemExit("Unexpected archive root")
            if info.filename in seen or stat.S_ISLNK(info.external_attr >> 16):
                raise SystemExit("Duplicate or symlink archive entry")
            seen.add(info.filename)
            if info.flag_bits & 1:
                raise SystemExit("Encrypted archive not supported")
            if not info.is_dir():
                blob = archive.read(info)
                entries[str(PurePosixPath(*path.parts[1:]))] = dict(
                    sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob))
        STAGING.mkdir()
        for info in infos:
            path = PurePosixPath(info.filename)
            dest = STAGING.joinpath(*path.parts[1:])
            if info.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("xb") as stream:
                    stream.write(archive.read(info))
    if manifest(STAGING) != entries:
        raise SystemExit("Extracted source mismatch; active source untouched")
    shutil.copy2(ARCHIVE, HERE / "TumourDiffusion_1mm_revision3_original.zip")
    if digest(HERE / "TumourDiffusion_1mm_revision3_original.zip") != zip_sha:
        raise SystemExit("Preserved ZIP SHA mismatch")
    ACTIVE.rename(PREVIOUS)
    try:
        STAGING.rename(ACTIVE)
    except BaseException:
        PREVIOUS.rename(ACTIVE)
        raise
    if manifest(PREVIOUS) != previous or manifest(ACTIVE) != entries:
        raise SystemExit("Post-move mismatch; preserve all evidence for investigation")
    comparison = dict(added=sorted(set(entries) - set(previous)),
                      removed_from_active_but_preserved=sorted(set(previous) - set(entries)),
                      changed=sorted(p for p in set(entries) & set(previous) if entries[p] != previous[p]),
                      unchanged=sorted(p for p in set(entries) & set(previous) if entries[p] == previous[p]))
    receipt = dict(**STATUS, recorded_at=stamp, archive_origin=str(ARCHIVE), archive_sha256=zip_sha,
                   active_root=str(ACTIVE), preserved_previous_root=str(PREVIOUS),
                   previous_matches_20261009_baseline=previous == baseline,
                   previous_preservation_verified=True, archive_to_active_sha_verified=True,
                   source_edits=False, source_file_count=len(entries), previous_file_count=len(previous),
                   previous=previous, revision=entries, comparison=comparison)
    (HERE / "REPLACEMENT_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("previous", "revision")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
