"""Bounded deployment, launch, smoke submission, and snapshot operations for K1245 r1."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parents[2] / "01_code/repository"
REMOTE = Path(
    "/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/"
    "s2_met_aug_k1245_20260921_r1"
)
PARENT = Path(
    "/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/"
    "s2_met_aug_fix_v3_full200_20260810_r1"
)
PYTHON = "/public/home/zqchen/.conda/envs/segmamba/bin/python"
ENVFILE = "/Users/hwaigc/比赛+课题/服务器登陆/服务器登陆的env"
VPN = "/Users/hwaigc/比赛+课题/服务器登陆/ecnu-vpn-split.sh"
PLAN = "K1245_RC_AWARE_EXECUTION_PLAN_20260921_R1.json"
META = {
    "artifact_status": "experimental_unvalidated",
    "operator_approved": False,
    "formal_gate_status": "not_run_not_passed",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exclusive_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


class RemoteAccess:
    def __init__(self, action: str) -> None:
        self.action = action
        self.env = dict(os.environ)
        if self.env.get("ECNU_SERVER_ENV_FILE") != ENVFILE or not self.env.get("HPC_SSH_PASSWORD"):
            raise RuntimeError("authorized ECNU environment was not loaded")
        for key, default in (
            ("HPC_LOGIN_HOST", "59.78.189.132"),
            ("HPC_LOGIN_PORT", "2323"),
            ("HPC_SSH_USER", "zqchen"),
        ):
            self.env.setdefault(key, default)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.audit = ROOT / "operator" / f"{action}_{stamp}"
        self.audit.mkdir(parents=True, exist_ok=False)
        self.owns_vpn = False

    def call(self, argv: list[str], timeout: int = 180) -> str:
        self.env["REMOTE_ARGV_B64"] = " ".join(
            base64.b64encode(value.encode("utf-8")).decode("ascii") for value in argv
        )
        script = r'''
log_user 1
set auth 0
set timeout ''' + str(timeout) + r'''
set argv {}
foreach item [split $env(REMOTE_ARGV_B64) " "] {
  lappend argv [exec python3 -c {import base64,sys; print(base64.b64decode(sys.argv[1]).decode("utf-8"), end="")} $item]
}
spawn -noecho {*}$argv
expect {
  -re "(P|p)assword:" {
    incr auth
    if {$auth > 1} {close; exit 125}
    log_user 0
    send -- "$env(HPC_SSH_PASSWORD)\r"
    log_user 1
    exp_continue
  }
  eof {exit [lindex [wait] 3]}
  timeout {close; exit 124}
}
'''
        reply = subprocess.run(
            ["expect"], input=script, text=True, capture_output=True,
            env=self.env, timeout=timeout + 20,
        )
        if reply.returncode:
            diagnostic = (reply.stdout + reply.stderr).replace(
                self.env["HPC_SSH_PASSWORD"], "[REDACTED]"
            )
            exclusive_json(
                self.audit / "FAILED.json",
                {**META, "action": self.action, "exit_code": reply.returncode,
                 "diagnostic_tail": diagnostic[-5000:]},
            )
            raise RuntimeError(f"bounded remote operation failed with exit {reply.returncode}")
        return reply.stdout

    def ssh(self, command: str, timeout: int = 180) -> str:
        return self.call(
            [
                "ssh", "-p", self.env["HPC_LOGIN_PORT"],
                "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15",
                f'{self.env["HPC_SSH_USER"]}@{self.env["HPC_LOGIN_HOST"]}', command,
            ],
            timeout=timeout,
        )

    def scp(self, sources: list[Path], destination: str) -> None:
        self.call(
            [
                "scp", "-P", self.env["HPC_LOGIN_PORT"],
                "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15",
                *[str(path) for path in sources],
                f'{self.env["HPC_SSH_USER"]}@{self.env["HPC_LOGIN_HOST"]}:{destination}',
            ]
        )

    def start_vpn(self) -> None:
        result = subprocess.run([VPN, "start"], env=self.env, capture_output=True, timeout=100)
        self.owns_vpn = b"OpenConnect is already running" not in result.stdout
        if result.returncode:
            raise RuntimeError("VPN startup failed; private output suppressed")

    def close_vpn(self) -> None:
        if not self.owns_vpn:
            return
        result = subprocess.run([VPN, "stop"], env=self.env, capture_output=True, timeout=40)
        if result.returncode:
            raise RuntimeError("VPN cleanup could not be confirmed")


def deploy(access: RemoteAccess) -> dict:
    remote = shlex.quote(str(REMOTE))
    access.ssh(
        "set -e; root=" + remote + "; "
        "test ! -e \"$root/evidence/REMOTE_STATIC_PREFLIGHT.json\"; "
        "test ! -e \"$root/evidence/POOL_BUILD_LAUNCH.json\"; "
        "test ! -e \"$root/pools\"; "
        "mkdir -p \"$root/runtime/repository/custom_nnunet\" "
        "\"$root/runtime/repository/scripts/09_k1245\" \"$root/runtime/repository/tests\" "
        "\"$root/evidence\" \"$root/logs\" \"$root/slurm\""
    )
    access.scp([ROOT / PLAN, ROOT / "README.md"], str(REMOTE) + "/")
    access.scp(sorted((ROOT / "runtime").glob("*.py")), str(REMOTE / "runtime") + "/")
    access.scp(sorted((ROOT / "slurm").glob("*.slurm")), str(REMOTE / "slurm") + "/")
    access.scp(sorted((REPOSITORY / "custom_nnunet").glob("*.py")), str(REMOTE / "runtime/repository/custom_nnunet") + "/")
    access.scp(sorted((REPOSITORY / "scripts/09_k1245").glob("*.py")), str(REMOTE / "runtime/repository/scripts/09_k1245") + "/")
    access.scp([REPOSITORY / "tests/test_met_aug_k1245.py"], str(REMOTE / "runtime/repository/tests") + "/")
    files = {
        PLAN: sha256(ROOT / PLAN),
        "runtime/run_remote_static_preflight.py": sha256(ROOT / "runtime/run_remote_static_preflight.py"),
        "runtime/repository/custom_nnunet/met_aug_k1245.py": sha256(REPOSITORY / "custom_nnunet/met_aug_k1245.py"),
        "runtime/repository/custom_nnunet/met_aug_diffusion.py": sha256(REPOSITORY / "custom_nnunet/met_aug_diffusion.py"),
        "runtime/repository/custom_nnunet/nnUNetTrainerBraTS2026RCMetAugK1245FocalCompletionFineTune.py": sha256(REPOSITORY / "custom_nnunet/nnUNetTrainerBraTS2026RCMetAugK1245FocalCompletionFineTune.py"),
    }
    code = (
        "import hashlib,json,pathlib; root=pathlib.Path(" + repr(str(REMOTE)) + "); "
        "expected=" + repr(files) + "; actual={k:hashlib.sha256((root/k).read_bytes()).hexdigest() for k in expected}; "
        "assert actual==expected,(actual,expected); print(json.dumps({'status':'deployed','sha256':actual},sort_keys=True))"
    )
    output = access.ssh(
        "env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -c " + shlex.quote(code)
    )
    return {"status": "deployed", "plan_sha256": files[PLAN], "remote_output": output[-4000:]}


def preflight(access: RemoteAccess) -> dict:
    command = (
        "env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=" + shlex.quote(str(REMOTE / "runtime/repository"))
        + " " + PYTHON + " -B " + shlex.quote(str(REMOTE / "runtime/run_remote_static_preflight.py"))
        + " --root " + shlex.quote(str(REMOTE)) + " --parent " + shlex.quote(str(PARENT))
    )
    output = access.ssh(command, timeout=240)
    check = (
        "import json,pathlib; p=pathlib.Path(" + repr(str(REMOTE / "evidence/REMOTE_STATIC_PREFLIGHT.json"))
        + "); x=json.loads(p.read_text()); assert x['status']=='pass'; print(json.dumps(x,sort_keys=True))"
    )
    verified = access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -c " + shlex.quote(check))
    return {"status": "preflight_pass", "remote_output": (output + verified)[-8000:]}


def launch(access: RemoteAccess) -> dict:
    command = (
        "env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B "
        + shlex.quote(str(REMOTE / "runtime/launch_pool_build.py"))
        + " --root " + shlex.quote(str(REMOTE))
    )
    output = access.ssh(command)
    return {"status": "pool_build_launched", "remote_output": output[-4000:]}


def snapshot(access: RemoteAccess) -> dict:
    code = r'''
import hashlib, json, pathlib, subprocess
from datetime import datetime, timezone
root=pathlib.Path(''' + repr(str(REMOTE)) + r''')
def load(name):
    p=root/'evidence'/name
    return json.loads(p.read_text()) if p.is_file() else None
launch=load('POOL_BUILD_LAUNCH.json')
job_id=str((launch or {}).get('job_id',''))
sq=subprocess.run(['squeue','-h','-j',job_id,'-o','%i|%T|%M|%R'],capture_output=True,text=True).stdout.strip() if job_id else ''
sa=subprocess.run(['sacct','-n','-P','-j',job_id,'--format=JobIDRaw,JobName,State,ExitCode,Elapsed'],capture_output=True,text=True).stdout.strip() if job_id else ''
smoke_launch=load('K1245_SMOKE_LAUNCH.json')
smoke_job_id=str((smoke_launch or {}).get('job_id',''))
smoke_sq=subprocess.run(['squeue','-h','-j',smoke_job_id,'-o','%i|%T|%M|%R'],capture_output=True,text=True).stdout.strip() if smoke_job_id else ''
smoke_sa=subprocess.run(['sacct','-n','-P','-j',smoke_job_id,'--format=JobIDRaw,JobName,State,ExitCode,Elapsed'],capture_output=True,text=True).stdout.strip() if smoke_job_id else ''
def tail(pattern):
    paths=sorted((root/'logs').glob(pattern))
    return {str(p.name): p.read_text(errors='replace')[-4000:] for p in paths}
files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and '.pools.staging' not in p.parts and 'pools' not in p.parts}
staging_counts={name:sum(1 for _ in (root/'.pools.staging'/name).glob('*.npz')) for name in ('rc27','small8')}
print('K1245_SNAPSHOT_BEGIN')
print(json.dumps({'remote_utc':datetime.now(timezone.utc).isoformat(),'hostname':subprocess.check_output(['hostname'],text=True).strip(),'root_exists':root.is_dir(),'launch':launch,'started':load('POOL_BUILD_STARTED.json'),'failed':load('POOL_BUILD_FAILED.json'),'complete':(root/'evidence/POOL_BUILD_COMPLETE.ok').is_file(),'squeue':sq,'sacct':sa,'smoke':{'launch':smoke_launch,'started':load('K1245_SMOKE_STARTED.json'),'failed':load('K1245_SMOKE_FAILED.json'),'validation':load('K1245_SMOKE_VALIDATION.json'),'complete':(root/'evidence/K1245_SMOKE_COMPLETE.ok').is_file(),'squeue':smoke_sq,'sacct':smoke_sa,'stdout_tail':tail('s2k1245smkr1_*.out'),'stderr_tail':tail('s2k1245smkr1_*.err')},'pool_validation':load('../pools/K1245_POOL_AND_EMPIRICAL_VALIDATION.json'),'staging_counts':staging_counts,'files':files},sort_keys=True))
print('K1245_SNAPSHOT_END')
'''
    output = access.ssh(
        "env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -c " + shlex.quote(code)
    )
    payload = output.split("K1245_SNAPSHOT_BEGIN", 1)[1].split("K1245_SNAPSHOT_END", 1)[0].strip()
    return json.loads(payload)


def launch_smoke(access: RemoteAccess) -> dict:
    smoke_script = ROOT / "slurm/02_run_k1245_smoke.slurm"
    launcher = ROOT / "runtime/launch_k1245_smoke.py"
    access.scp([smoke_script], str(REMOTE / "slurm") + "/")
    access.scp([launcher], str(REMOTE / "runtime") + "/")
    output = access.ssh(
        "env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B "
        + shlex.quote(str(REMOTE / "runtime/launch_k1245_smoke.py"))
        + " --root " + shlex.quote(str(REMOTE))
    )
    return {"status": "smoke_launched", "remote_output": output[-4000:]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("deploy", "preflight", "launch", "launch_smoke", "snapshot"))
    args = parser.parse_args()
    access = RemoteAccess(args.action)
    started = datetime.now(timezone.utc).isoformat()
    try:
        access.start_vpn()
        result = globals()[args.action](access)
    finally:
        access.close_vpn()
    receipt = {
        **META,
        "action": args.action,
        "local_started_utc": started,
        "local_completed_utc": datetime.now(timezone.utc).isoformat(),
        "vpn_closed": access.owns_vpn,
        "preexisting_vpn_left_unchanged": not access.owns_vpn,
        "result": result,
    }
    exclusive_json(access.audit / "LOCAL_OPERATION_RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
