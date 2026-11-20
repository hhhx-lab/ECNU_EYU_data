"""Download only bounded Markdown explanations for the user's Nov12 batches."""
import importlib.util
import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("review_snapshot", ROOT / "operator/account_snapshot.py")
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)
m = snapshot.transport.m


def remote_scan(roots, job_ids):
    return r'''
import hashlib,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOTS=ROOTS_VALUE
IDS=IDS_VALUE
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
files=[]
errors=[]
for name in ROOTS:
    root=Path(name)
    if not root.is_dir() or root.is_symlink() or root.resolve()!=root:
        errors.append('Root missing or redirected: '+name)
        continue
    for directory,dirs,names in os.walk(root,followlinks=False):
        d=Path(directory)
        depth=len(d.relative_to(root).parts)
        dirs[:]=[x for x in dirs if not x.startswith('.') and not (d/x).is_symlink()
            and x not in {'data','dataset','datasets','weights','checkpoints','models','predictions','probabilities','cache','__pycache__'}]
        if depth>=3:dirs[:]=[]
        for n in sorted(names):
            p=d/n
            if p.suffix.lower()!='.md' or p.is_symlink() or not p.is_file():continue
            size=p.stat().st_size
            if size>2097152:
                errors.append('Markdown exceeds per-file budget: '+str(p));continue
            files.append(dict(root=name,relative_path=p.relative_to(root).as_posix(),path=str(p),bytes=size,sha256=sha(p)))
files.sort(key=lambda x:x['path'])
if len(files)>200 or sum(f['bytes'] for f in files)>20971520:errors.append('Markdown total budget exceeded')
p=subprocess.run(['sacct','-n','-P','-j',','.join(IDS),'--format=JobID,JobName%120,State,ExitCode,Elapsed,Submit,Start,End'],capture_output=True,text=True,timeout=45)
if p.returncode:errors.append('Accounting query failed')
print(json.dumps(dict(status='pass' if not errors else 'review_incomplete',remote_utc=datetime.now(timezone.utc).isoformat(),
    roots=ROOTS,files=files,total_bytes=sum(f['bytes'] for f in files),errors=errors,
    accounting=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),
    scope='Markdown only; no dataset, model, script execution or job changes'),sort_keys=True))
'''.replace("ROOTS_VALUE", repr(roots)).replace("IDS_VALUE", repr(job_ids))


def main():
    prior = ROOT / "operator/account_metadata_20261120T092623Z/RESULT.json"
    history = json.loads(prior.read_text())["history_rows"]
    selected = [r for r in history if r["submit"].startswith("2026-11-12")
        and r["work_dir"].startswith("/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/")
        and "disk_cleanup" not in r["work_dir"] and "speed_compile" not in r["work_dir"]]
    roots = sorted({r["work_dir"] for r in selected})
    ids = sorted({r["job_id"] for r in selected})
    assert len(roots) == 7 and ids
    code = remote_scan(roots, ids)
    access = m.Access("nov12_markdown")
    started = m.datetime.now(m.timezone.utc).isoformat()
    stop_ok = False
    try:
        m.exclusive(access.audit / "EXECUTED_SOURCE.json", {**m.META, "source": code,
            "local_operator_sha256": m.sha(Path(__file__)), "prior_snapshot_sha256": m.sha(prior)})
        p = subprocess.run([m.VPN, "start"], env=access.env, capture_output=True, timeout=100)
        flags = re.findall(rb"VPN_SESSION_OWNED=([01])", p.stdout)
        if len(flags) != 1:
            raise RuntimeError("VPN ownership unconfirmed; private output suppressed")
        access.owns_vpn = flags[0] == b"1"
        if p.returncode:
            raise RuntimeError("VPN start failed; no SSH; private output suppressed")
        p = subprocess.run([m.VPN, "check"], env=access.env, capture_output=True, timeout=40)
        if p.returncode:
            raise RuntimeError("VPN check failed; no SSH")
        result = json.loads(access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + m.PYTHON + " -B -c " + shlex.quote(code), timeout=140).strip().splitlines()[-1])
        result["local_received_utc"] = m.datetime.now(m.timezone.utc).isoformat()
        m.exclusive(access.audit / "RESULT.json", result)
        if result["status"] != "pass":
            raise RuntimeError("Markdown manifest/accounting incomplete; no transfer")
        if shutil.disk_usage(ROOT).free < result["total_bytes"] + 1073741824:
            raise RuntimeError("Local download storage budget failed")
        destination = ROOT / ("markdown_" + access.audit.name.removeprefix("nov12_markdown_"))
        destination.mkdir(exist_ok=False)
        downloads = []
        for record in result["files"]:
            path = destination / Path(record["root"]).name / record["relative_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise RuntimeError("Refusing local overwrite")
            access.call(["scp", "-P", access.env["HPC_LOGIN_PORT"], "-o", "StrictHostKeyChecking=yes",
                "-o", "ConnectTimeout=15", access.env["HPC_SSH_USER"] + "@" + access.env["HPC_LOGIN_HOST"] + ":" + record["path"], str(path)], timeout=180)
            if path.stat().st_size != record["bytes"] or m.sha(path) != record["sha256"]:
                raise RuntimeError("Download identity mismatch: " + str(path))
            downloads.append({**record, "local_path": str(path)})
        verify = r'''
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
expected=EXPECTED_VALUE
bad=[]
for r in expected:
    p=Path(r['path'])
    if not p.is_file() or p.is_symlink() or p.stat().st_size!=r['bytes']:
        bad.append(r['path']);continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    if h.hexdigest()!=r['sha256']:bad.append(r['path'])
print(json.dumps(dict(status='pass' if not bad else 'drift',bad=bad,remote_utc=datetime.now(timezone.utc).isoformat())))
'''.replace("EXPECTED_VALUE", repr(result["files"]))
        checked = json.loads(access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + m.PYTHON + " -B -c " + shlex.quote(verify), timeout=90).strip().splitlines()[-1])
        m.exclusive(access.audit / "POST_TRANSFER_REMOTE_CHECK.json", checked)
        if checked["status"] != "pass":
            raise RuntimeError("Remote Markdown changed during transfer; preserve files unaccepted")
        acceptance = {**m.META, "status": "download_identity_pass_not_project_completion",
            "files": downloads, "file_count": len(downloads), "total_bytes": result["total_bytes"],
            "remote_checked_utc": checked["remote_utc"], "local_accepted_utc": m.datetime.now(m.timezone.utc).isoformat(),
            "destination": str(destination), "no_remote_writes": True, "no_job_changes": True}
        m.exclusive(access.audit / "DOWNLOAD_ACCEPTANCE.json", acceptance)
        print(json.dumps({k:acceptance[k] for k in ("status", "file_count", "total_bytes", "destination")}, ensure_ascii=False))
    except BaseException as error:
        m.exclusive(access.audit / "OPERATION_STOPPED.json", {**m.META, "message": str(error),
            "utc": m.datetime.now(m.timezone.utc).isoformat(), "no_automatic_retry": True})
        raise
    finally:
        if access.owns_vpn:
            p = subprocess.run([m.VPN, "stop"], env=access.env, capture_output=True, timeout=40)
            stop_ok = p.returncode == 0
        m.exclusive(access.audit / "LOCAL_OPERATION_RECEIPT.json", {**m.META, "started_utc": started,
            "completed_utc": m.datetime.now(m.timezone.utc).isoformat(), "vpn_owned": access.owns_vpn,
            "own_vpn_stop_completed": stop_ok, "preexisting_vpn_preserved": not access.owns_vpn})
        if access.owns_vpn and not stop_ok:
            raise RuntimeError("Own VPN stop unconfirmed; do not stop unrelated sessions")


if __name__ == "__main__":
    main()
