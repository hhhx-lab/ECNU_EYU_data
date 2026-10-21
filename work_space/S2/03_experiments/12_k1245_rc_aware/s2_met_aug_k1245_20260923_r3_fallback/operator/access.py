"""One-shot, audited ECNU operations for the isolated K1245 fallback attempt."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R2_LOCAL = ROOT.parent / "s2_met_aug_k1245_20260923_r2"
R2_REMOTE = Path("/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1245_20260923_r2")
REMOTE = Path("/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1245_20260923_r3_fallback")
REMOTE_R3D = REMOTE / "attempts/k1_rc_smoke_r3d"
REMOTE_R3F = REMOTE / "attempts/k1_rc_smoke_r3f"
REMOTE_R3G = REMOTE / "attempts/k1_rc_smoke_r3g"
REMOTE_K4_R3 = REMOTE / "attempts/k4_small_smoke_r3"
PARENT_REMOTE = Path("/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1")
PYTHON = "/public/home/zqchen/.conda/envs/segmamba/bin/python"
ENVFILE = "/Users/hwaigc/比赛+课题/服务器登陆/服务器登陆的env"
VPN = "/Users/hwaigc/比赛+课题/服务器登陆/ecnu-vpn-split.sh"
R2_RECEIPT = R2_LOCAL / "operator/snapshot_20260923T070821Z/LOCAL_OPERATION_RECEIPT.json"
R1_RECEIPT = ROOT.parent / "s2_met_aug_k1245_20260921_r1/operator/snapshot_20260922T154321Z/LOCAL_OPERATION_RECEIPT.json"
META = {"artifact_status": "experimental_unvalidated", "operator_approved": False, "formal_gate_status": "not_run_not_passed"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


class Access:
    def __init__(self, action: str) -> None:
        self.env = dict(os.environ)
        if self.env.get("ECNU_SERVER_ENV_FILE") != ENVFILE or not self.env.get("HPC_SSH_PASSWORD"):
            raise RuntimeError("ECNU credentials were not loaded from the authorized env file")
        self.env.setdefault("HPC_LOGIN_HOST", "59.78.189.132")
        self.env.setdefault("HPC_LOGIN_PORT", "2323")
        self.env.setdefault("HPC_SSH_USER", "zqchen")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.audit = ROOT / "operator" / f"{action}_{stamp}"
        self.audit.mkdir(parents=True, exist_ok=False)
        self.owns_vpn = False

    def call(self, argv: list[str], timeout: int = 240) -> str:
        self.env["REMOTE_ARGV_B64"] = " ".join(base64.b64encode(x.encode()).decode("ascii") for x in argv)
        expect_script = r'''
log_user 1
set auth 0
set timeout ''' + str(timeout) + r'''
set argv {}
foreach item [split $env(REMOTE_ARGV_B64) " "] {
  lappend argv [exec python3 -c {import base64,sys; print(base64.b64decode(sys.argv[1]).decode(), end="")} $item]
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
        result = subprocess.run(["expect"], input=expect_script, text=True, capture_output=True,
                                env=self.env, timeout=timeout + 30)
        if result.returncode:
            diagnostic = (result.stdout + result.stderr).replace(self.env["HPC_SSH_PASSWORD"], "[REDACTED]")
            exclusive(self.audit / "FAILED.json", {**META, "status": "failed", "exit_code": result.returncode,
                                                     "diagnostic_tail": diagnostic[-3000:]})
            raise RuntimeError(f"bounded remote operation failed ({result.returncode})")
        return result.stdout

    def ssh(self, command: str, timeout: int = 240) -> str:
        return self.call(["ssh", "-p", self.env["HPC_LOGIN_PORT"], "-o", "StrictHostKeyChecking=yes",
                          "-o", "ConnectTimeout=15", f'{self.env["HPC_SSH_USER"]}@{self.env["HPC_LOGIN_HOST"]}', command], timeout)

    def scp(self, source: Path, destination: str) -> None:
        self.call(["scp", "-P", self.env["HPC_LOGIN_PORT"], "-o", "StrictHostKeyChecking=yes",
                   "-o", "ConnectTimeout=15", str(source),
                   f'{self.env["HPC_SSH_USER"]}@{self.env["HPC_LOGIN_HOST"]}:{destination}'])

    def scp_immutable(self, source: Path, destination_dir: str) -> None:
        remote_path = destination_dir.rstrip("/") + "/" + source.name
        probe = "if test -f " + shlex.quote(remote_path) + "; then sha256sum " + shlex.quote(remote_path) + "; else printf MISSING; fi"
        observed = self.ssh(probe).strip().splitlines()[-1].strip()
        local_sha = sha(source)
        if observed != "MISSING":
            remote_sha = observed.split()[0]
            if remote_sha != local_sha:
                raise RuntimeError(f"immutable remote file differs; refusing overwrite: {source.name}")
            return
        self.scp(source, destination_dir.rstrip("/") + "/")

    def start_vpn(self) -> None:
        result = subprocess.run([VPN, "start"], env=self.env, capture_output=True, timeout=100)
        self.owns_vpn = b"OpenConnect is already running" not in result.stdout
        if result.returncode:
            raise RuntimeError("split VPN startup failed; private output suppressed")

    def close_vpn(self) -> None:
        if self.owns_vpn:
            result = subprocess.run([VPN, "stop"], env=self.env, capture_output=True, timeout=40)
            if result.returncode:
                raise RuntimeError("could not confirm shutdown of this operation's VPN")


def prepare_validate(access: Access) -> dict:
    plan = ROOT / "K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json"
    prereg = ROOT.parent.parent.parent / "05_documents/02_plans/20260921_k1245_rc_aware_execution/F_FALLBACK_GATE_PREREGISTRATION_20260923_R3.md"
    evaluator = ROOT / "runtime/evaluate_f_fallback.py"
    readme = ROOT / "README.md"
    r2_plan_local = R2_LOCAL / "K1245_RC_AWARE_EXECUTION_PLAN_20260923_R2.json"
    r2_receipt = R2_RECEIPT
    expected = {
        "K1245_RC_AWARE_EXECUTION_PLAN_20260923_R2.json": sha(r2_plan_local),
        "evidence/REMOTE_STATIC_PREFLIGHT.json": "1848ffe601f0749122d144938595d42fe9c78bf0f6bf71114ba7e096dd45a350",
        "evidence/POOL_BUILD_COMPLETE.ok": "37a40f08d8548dba289b9b0bb35bcf63b359f6d37ee86044ebc6b6da080b9ec1",
        "operator/snapshot_20260923T070821Z/LOCAL_OPERATION_RECEIPT.json": sha(r2_receipt),
    }
    if expected["K1245_RC_AWARE_EXECUTION_PLAN_20260923_R2.json"] != "815494ccba511570afa4c53c23034807ce8f351d06446f25d1c2c4fc4e364018":
        raise RuntimeError("local frozen r2 plan changed")
    if sha(R1_RECEIPT) != "de1f231c46bc61de4c91ecd12983690aac3f6bb9765f4e1ea4d0df83cfe47785":
        raise RuntimeError("local frozen r1 strict-failure receipt changed")
    remote_root = shlex.quote(str(REMOTE))
    access.ssh("set -euo pipefail; test ! -e " + remote_root + "; mkdir -m 700 " + remote_root +
                "; mkdir -m 700 " + remote_root + "/evidence " + remote_root + "/runtime " + remote_root + "/sources " + remote_root + "/operator")
    for path, dest in (
        (plan, str(REMOTE) + "/"), (readme, str(REMOTE) + "/"), (prereg, str(REMOTE) + "/sources/"),
        (evaluator, str(REMOTE) + "/runtime/"),
        (r2_receipt, str(REMOTE) + "/sources/LOCAL_OPERATION_RECEIPT.json"),
        (R1_RECEIPT, str(REMOTE) + "/sources/R1_STRICT_FAILURE_RECEIPT.json"),
        (r2_plan_local, str(REMOTE) + "/sources/"),
    ):
        access.scp(path, dest)
    static_code = (
        "import hashlib,json,pathlib,os,datetime;"
        f"root=pathlib.Path({str(REMOTE)!r}); r2=pathlib.Path({str(R2_REMOTE)!r});"
        "sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();"
        "plan=root/'K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json'; p=json.loads(plan.read_text());"
        "assert sha(r2/'K1245_RC_AWARE_EXECUTION_PLAN_20260923_R2.json')==p['parent_contract']['r2_experiment_plan_sha256'];"
        "assert sha(r2/'evidence/REMOTE_STATIC_PREFLIGHT.json')==p['fallback_gate']['source_remote_static_preflight_sha256'];"
        "assert sha(r2/'evidence/POOL_BUILD_COMPLETE.ok')==p['fallback_gate']['source_pool_complete_marker_sha256'];"
        "e=json.loads((r2/'pools/K1245_POOL_AND_EMPIRICAL_VALIDATION.json').read_text());"
        "assert e['status']=='blocked_f_gate' and e['empirical']['f_gate_pass'] is False;"
        "assert e['manifests']['rc27']['sha256']==p['fallback_gate']['source_rc27_manifest_identity_sha256'];"
        "assert e['manifests']['small8']['sha256']==p['fallback_gate']['source_small8_manifest_identity_sha256'];"
        "assert sha(root/'sources/K1245_RC_AWARE_EXECUTION_PLAN_20260923_R2.json')==p['parent_contract']['r2_experiment_plan_sha256'];"
        "assert sha(root/'sources/LOCAL_OPERATION_RECEIPT.json')==p['fallback_gate']['source_snapshot_receipt_sha256'];"
        "assert sha(root/'sources/R1_STRICT_FAILURE_RECEIPT.json')==p['strict_gate']['source_evidence_receipt_sha256'];"
        "out={'artifact_status':'experimental_unvalidated','operator_approved':False,'formal_gate_status':'not_run_not_passed','status':'pass','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,'plan_sha256':sha(plan),'r2_plan_sha256':sha(r2/'K1245_RC_AWARE_EXECUTION_PLAN_20260923_R2.json'),'r2_pool_status':e['status'],'F_STRICT_PREREG_GATE':False};"
        "dest=root/'evidence/REMOTE_FALLBACK_STATIC_PREFLIGHT.json';"
        "fd=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400); h=os.fdopen(fd,'w');json.dump(out,h,sort_keys=True,indent=2);h.write('\\n');h.close();print(json.dumps(out,sort_keys=True))"
    )
    access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B -c " + shlex.quote(static_code))
    command = ("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B " +
               shlex.quote(str(REMOTE / "runtime/evaluate_f_fallback.py")) +
               " --r2-receipt " + shlex.quote(str(REMOTE / "sources/LOCAL_OPERATION_RECEIPT.json")) +
               " --output " + shlex.quote(str(REMOTE / "evidence/F_FALLBACK_GATE_VALIDATION.json")))
    output = access.ssh(command)
    local_gate = access.audit / "F_FALLBACK_GATE_VALIDATION.json"
    access.call(["scp", "-P", access.env["HPC_LOGIN_PORT"], "-o", "StrictHostKeyChecking=yes",
                 "-o", "ConnectTimeout=15", f'{access.env["HPC_SSH_USER"]}@{access.env["HPC_LOGIN_HOST"]}:{REMOTE}/evidence/F_FALLBACK_GATE_VALIDATION.json', str(local_gate)])
    payload = json.loads(local_gate.read_text())
    if payload.get("status") != "pass" or payload.get("F_STRICT_PREREG_GATE") is not False:
        raise RuntimeError("remote fallback gate did not pass with strict=false")
    return {"status": "pass", "remote_output_tail": output[-2500:], "fallback_gate_sha256": sha(local_gate),
            "remote_preflight_sha256": "captured remotely; included in remote root"}


def configure_k1(access: Access) -> dict:
    repo = ROOT.parents[2] / "01_code/repository"
    plan = ROOT / "K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json"
    config_plan = ROOT / "K1_FALLBACK_CONFIG_PLAN_20260923_R3A.json"
    smoke_plan = ROOT / "K1_RC_SMOKE_EXECUTION_PLAN_20260923_R3A.json"
    local_code = [
        *sorted((repo / "custom_nnunet").glob("*.py")),
        repo / "scripts/09_k1245/60_build_k1245_component_pools.py",
        repo / "scripts/09_k1245/61_prepare_k1245_configs.py",
        repo / "scripts/09_k1245/62_run_k1245_smoke.py",
        repo / "tests/test_met_aug_k1245.py",
        ROOT / "runtime/prepare_k1_fallback_config_r3a.py",
    ]
    dirs = ("runtime/repository/custom_nnunet", "runtime/repository/scripts/09_k1245", "runtime/repository/tests")
    access.ssh("set -e; root=" + shlex.quote(str(REMOTE)) + "; " +
                "test -f \"$root/evidence/F_FALLBACK_GATE_VALIDATION.json\"; " +
                "test ! -e \"$root/config/K1245_ROUTE_K1_FALLBACK.json\"; " +
                "test ! -e \"$root/evidence/K1_FALLBACK_CONFIG_PREPARATION.json\"; " +
                "mkdir -p -m 700 \"$root/config\" \"$root/runtime/repository/custom_nnunet\" \"$root/runtime/repository/scripts/09_k1245\" \"$root/runtime/repository/tests\"")
    for src in local_code:
        rel = src.relative_to(repo) if repo in src.parents else Path("runtime") / src.name
        target = str(REMOTE / "runtime/repository" / rel.parent) + "/" if repo in src.parents else str(REMOTE / "runtime") + "/"
        access.scp(src, target)
    access.scp(config_plan, str(REMOTE) + "/")
    access.scp(smoke_plan, str(REMOTE) + "/")
    command = (
        "set -euo pipefail; root=" + shlex.quote(str(REMOTE)) + "; parent=" + shlex.quote(str(PARENT_REMOTE)) + "; "
        "export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=\"$root/runtime/repository\"; "
        "test -f \"$root/evidence/REMOTE_FALLBACK_STATIC_PREFLIGHT.json\"; "
        "test -f \"$root/evidence/F_FALLBACK_GATE_VALIDATION.json\"; "
        "test ! -e \"$root/config/K1245_ROUTE_K1_FALLBACK.json\"; "
        "test \"$(sha256sum \"$root/K1_FALLBACK_CONFIG_PLAN_20260923_R3A.json\" | cut -d' ' -f1)\" = \"0a674b1e1d9bb6cf9f7355005f137f0a919f86588687bd6d2a99dd1ec979ece6\"; "
        + shlex.quote(PYTHON) + " -B \"$root/runtime/prepare_k1_fallback_config_r3a.py\" "
        + shlex.quote(str(R2_REMOTE)) + " " + shlex.quote(str(PARENT_REMOTE))
    )
    output = access.ssh(command, timeout=240)
    return {"status": "K1_fallback_config_prepared", "output_tail": output[-3000:],
            "fallback_plan_sha256": sha(plan), "smoke_plan_sha256": sha(smoke_plan)}


def deploy_smoke_r3b(access: Access) -> dict:
    repo = ROOT.parents[2] / "01_code/repository"
    prereg = ROOT.parents[2] / "05_documents/02_plans/20260921_k1245_rc_aware_execution/F_FALLBACK_GATE_PREREGISTRATION_20260923_R3.md"
    config_plan = ROOT / "K1_FALLBACK_CONFIG_PLAN_20260923_R3A.json"
    smoke_plan = ROOT / "K1_RC_SMOKE_EXECUTION_PLAN_20260923_R3B.json"
    runtime_files = [ROOT / "runtime/run_k1_rc_smoke.py", ROOT / "runtime/launch_k1_rc_smoke.py"]
    static_files = [
        ROOT / "K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json",
        config_plan,
        smoke_plan,
        ROOT / "README.md",
        prereg,
        *runtime_files,
        ROOT / "slurm/02_run_k1_rc_smoke_r3b.slurm",
        *sorted((repo / "custom_nnunet").glob("*.py")),
        repo / "scripts/09_k1245/60_build_k1245_component_pools.py",
        repo / "scripts/09_k1245/61_prepare_k1245_configs.py",
        repo / "scripts/09_k1245/62_run_k1245_smoke.py",
        repo / "tests/test_met_aug_k1245.py",
    ]
    access.ssh("mkdir -p " + " ".join(shlex.quote(str(REMOTE / rel)) for rel in
                ("runtime/repository/custom_nnunet", "runtime/repository/scripts/09_k1245", "runtime/repository/tests", "slurm", "logs")))
    expected = {}
    source_map = {}
    for src in static_files:
        if ROOT in src.parents:
            relative = src.relative_to(ROOT)
            remote_path = REMOTE / relative
        elif repo in src.parents:
            relative = src.relative_to(repo)
            remote_path = REMOTE / "runtime/repository" / relative
        elif src == prereg:
            remote_path = REMOTE / "sources" / src.name
        else:
            raise RuntimeError(f"unmapped immutable smoke source: {src}")
        destination = str(remote_path.parent)
        expected[str(remote_path)] = sha(src)
        source_map[str(remote_path)] = (src, destination)
    check_code = (
        "import pathlib,hashlib,json,os;expected=" + repr(expected) + ";"
        "actual={p:(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() if pathlib.Path(p).is_file() else None) for p in expected};"
        "bad={p:(actual[p],h) for p,h in expected.items() if actual[p] is not None and actual[p]!=h};"
        "assert not bad,('immutable source mismatch',bad);"
        "print(json.dumps({'missing':[p for p in expected if actual[p] is None],'verified_existing':sum(v is not None for v in actual.values())},sort_keys=True))"
    )
    observed = json.loads(access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B -c " + shlex.quote(check_code)).splitlines()[-1])
    missing = set(observed["missing"])
    for remote_path, (src, destination) in source_map.items():
        if remote_path in missing:
            access.scp(src, destination.rstrip("/") + "/")
    static_code = (
        "import hashlib,json,pathlib,os,datetime;root=pathlib.Path(" + repr(str(REMOTE)) + ");"
        "expected=" + repr(expected) + ";sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();"
        "actual={p:sha(pathlib.Path(p)) for p in expected};"
        "assert actual==expected,(actual,expected);"
        "import py_compile;"
        "py_compile.compile(str(root/'runtime/run_k1_rc_smoke.py'),doraise=True);"
        "py_compile.compile(str(root/'runtime/launch_k1_rc_smoke.py'),doraise=True);"
        "plan=json.loads((root/'K1_RC_SMOKE_EXECUTION_PLAN_20260923_R3B.json').read_text());"
        "assert plan['job_name']=='s2k1rcsmkr3b' and plan['training_authorization'] is False and plan['k5_authorization'] is False;"
        "assert plan['F_STRICT_PREREG_GATE'] is False;"
        "out={'artifact_status':'experimental_unvalidated','operator_approved':False,'formal_gate_status':'not_run_not_passed','status':'pass','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,'plan_sha256':sha(root/'K1_RC_SMOKE_EXECUTION_PLAN_20260923_R3B.json'),'runtime_sha256':sha(root/'runtime/run_k1_rc_smoke.py'),'slurm_sha256':sha(root/'slurm/02_run_k1_rc_smoke_r3b.slurm'),'sources_sha256':actual};"
        "dest=root/'evidence/K1_SMOKE_R3B_REMOTE_STATIC_PREFLIGHT.json';"
        "fd=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400);h=os.fdopen(fd,'w');json.dump(out,h,indent=2,sort_keys=True);h.write('\\n');h.close();print(json.dumps(out,sort_keys=True))"
    )
    result = access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B -c " + shlex.quote(static_code), timeout=240)
    return {"status": "static_preflight_pass", "output_tail": result[-3000:],
            "source_file_count": len(expected), "smoke_plan_sha256": sha(smoke_plan)}


def deploy_smoke_r3c(access: Access) -> dict:
    repo = ROOT.parents[2] / "01_code/repository"
    prereg = ROOT.parents[2] / "05_documents/02_plans/20260921_k1245_rc_aware_execution/F_FALLBACK_GATE_PREREGISTRATION_20260923_R3.md"
    smoke_plan = ROOT / "K1_RC_SMOKE_EXECUTION_PLAN_20260923_R3C.json"
    sources = [
        ROOT / "K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json",
        ROOT / "K1_FALLBACK_CONFIG_PLAN_20260923_R3A.json",
        smoke_plan, ROOT / "README.md", prereg,
        ROOT / "runtime/run_k1_rc_smoke.py", ROOT / "slurm/03_run_k1_rc_smoke_r3c.slurm",
        *sorted((repo / "custom_nnunet").glob("*.py")),
        repo / "scripts/09_k1245/62_run_k1245_smoke.py",
    ]
    access.ssh("mkdir -p " + " ".join(shlex.quote(str(REMOTE / p)) for p in
                ("sources", "runtime", "slurm", "logs", "runtime/repository/custom_nnunet", "runtime/repository/scripts/09_k1245")))
    expected = {}
    upload = []
    for src in sources:
        if ROOT in src.parents:
            dest = REMOTE / src.relative_to(ROOT)
        elif repo in src.parents:
            dest = REMOTE / "runtime/repository" / src.relative_to(repo)
        elif src == prereg:
            dest = REMOTE / "sources" / src.name
        else:
            raise RuntimeError(f"unmapped R3C static source: {src}")
        expected[str(dest)] = sha(src)
        upload.append((src, dest))
    check = (
        "import pathlib,hashlib,json;expected=" + repr(expected) + ";"
        "sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() if pathlib.Path(p).is_file() else None;"
        "actual={p:sha(p) for p in expected};bad={p:(actual[p],v) for p,v in expected.items() if actual[p] not in (None,v)};"
        "assert not bad,('immutable mismatch',bad);print(json.dumps({'missing':[p for p,v in actual.items() if v is None],'existing':sum(v is not None for v in actual.values())},sort_keys=True))"
    )
    observed = json.loads(access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B -c " + shlex.quote(check)).splitlines()[-1])
    missing = set(observed["missing"])
    for src, dest in upload:
        if str(dest) in missing:
            access.scp(src, str(dest.parent) + "/")
    static = (
        "import pathlib,hashlib,json,datetime,os,py_compile;root=pathlib.Path(" + repr(str(REMOTE)) + ");expected=" + repr(expected) + ";"
        "actual={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in expected};assert actual==expected;"
        "py_compile.compile(str(root/'runtime/run_k1_rc_smoke.py'),doraise=True);"
        "plan=json.loads((root/'K1_RC_SMOKE_EXECUTION_PLAN_20260923_R3C.json').read_text());"
        "assert plan['job_name']=='s2k1rcsmkr3c' and plan['strict_gate_result']['result'] is False;"
        "assert plan['training_authorization'] is False and plan['k5_authorization'] is False;"
        "pool=pathlib.Path(" + repr(str(R2_REMOTE)) + ")/'pools/rc27/component_manifest.json';m=json.loads(pool.read_text());"
        "assert hashlib.sha256(pool.read_bytes()).hexdigest()==plan['smoke']['pool_manifest_file_sha256'];"
        "assert m['manifest_sha256']==plan['smoke']['pool_manifest_identity_sha256'];"
        "cfg=json.loads((root/'config/K1245_ROUTE_K1_FALLBACK.json').read_text());"
        "assert hashlib.sha256((root/'config/K1245_ROUTE_K1_FALLBACK.json').read_bytes()).hexdigest()==plan['k1_config_sha256'];"
        "assert cfg['F_STRICT_PREREG_GATE'] is False;"
        "out={'artifact_status':'experimental_unvalidated','operator_approved':False,'formal_gate_status':'not_run_not_passed','status':'pass','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,'job_name':plan['job_name'],'strict_gate':False,'files_sha256':actual};"
        "dest=root/'evidence/K1_SMOKE_R3C_STATIC_PREFLIGHT.json';fd=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400);h=os.fdopen(fd,'w');json.dump(out,h,sort_keys=True,indent=2);h.write('\\n');h.close();print(json.dumps(out,sort_keys=True))"
    )
    result = access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B -c " + shlex.quote(static), timeout=240)
    return {"status": "static_preflight_pass", "remote_output_tail": result[-3000:], "source_count": len(expected)}


def deploy_smoke_r3d(access: Access) -> dict:
    attempt = ROOT / "attempts/k1_rc_smoke_r3d"
    repo = ROOT.parents[2] / "01_code/repository"
    prereg = ROOT.parents[2] / "05_documents/02_plans/20260921_k1245_rc_aware_execution/F_FALLBACK_GATE_PREREGISTRATION_20260923_R3.md"
    plan = attempt / "K1_RC_SMOKE_EXECUTION_PLAN_20260923_R3D.json"
    runner = attempt / "runtime/run_k1_rc_smoke.py"
    slurm = attempt / "run_k1_rc_smoke_r3d.slurm"
    files = [plan, slurm, runner, prereg,
             ROOT / "K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json",
             ROOT / "K1_FALLBACK_CONFIG_PLAN_20260923_R3A.json",
             ROOT / "runtime/prepare_k1_fallback_config_r3a.py",
             *sorted((repo / "custom_nnunet").glob("*.py")),
             repo / "scripts/09_k1245/62_run_k1245_smoke.py"]
    access.ssh("set -euo pipefail; mkdir -m 700 -p " +
               " ".join(shlex.quote(str(REMOTE_R3D / p)) for p in
                        ("evidence", "config", "runtime", "runtime/repository/custom_nnunet",
                         "runtime/repository/scripts/09_k1245", "sources", "logs", "operator", "smoke")))
    expected = {}
    for src in files:
        if repo in src.parents:
            dest = REMOTE_R3D / "runtime/repository" / src.relative_to(repo)
        elif src == prereg:
            dest = REMOTE_R3D / "sources" / src.name
        elif src == ROOT / "K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json":
            dest = REMOTE_R3D / src.name
        elif src == ROOT / "runtime/prepare_k1_fallback_config_r3a.py":
            dest = REMOTE_R3D / "runtime" / src.name
        elif attempt in src.parents:
            dest = REMOTE_R3D / src.relative_to(attempt)
        elif ROOT in src.parents:
            dest = REMOTE_R3D / src.relative_to(ROOT)
        else:
            dest = REMOTE_R3D / "sources" / src.name
        expected[str(dest)] = sha(src)
        access.scp_immutable(src, str(dest.parent))
    stage_command = (
        "set -euo pipefail; src=" + shlex.quote(str(REMOTE)) + "; dst=" + shlex.quote(str(REMOTE_R3D)) + "; "
        "if test -e \"$dst/evidence/F_FALLBACK_GATE_VALIDATION.json\"; then cmp -s \"$src/evidence/F_FALLBACK_GATE_VALIDATION.json\" \"$dst/evidence/F_FALLBACK_GATE_VALIDATION.json\"; else cp -p \"$src/evidence/F_FALLBACK_GATE_VALIDATION.json\" \"$dst/evidence/\"; fi; "
        "if test -e \"$dst/K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json\"; then cmp -s \"$src/K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json\" \"$dst/K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json\"; else cp -p \"$src/K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json\" \"$dst/\"; fi; "
        "if test -e \"$dst/evidence/K1_FALLBACK_CONFIG_PREPARATION.json\"; then cmp -s \"$src/evidence/K1_FALLBACK_CONFIG_PREPARATION.json\" \"$dst/evidence/K1_FALLBACK_CONFIG_PREPARATION.json\"; else cp -p \"$src/evidence/K1_FALLBACK_CONFIG_PREPARATION.json\" \"$dst/evidence/\"; fi; "
        "if test -e \"$dst/config/K1245_ROUTE_K1_FALLBACK.json\"; then cmp -s \"$src/config/K1245_ROUTE_K1_FALLBACK.json\" \"$dst/config/K1245_ROUTE_K1_FALLBACK.json\"; else cp -p \"$src/config/K1245_ROUTE_K1_FALLBACK.json\" \"$dst/config/\"; fi; "
        "test -f \"$dst/config/K1245_ROUTE_K1_FALLBACK.json\"; "
        "test -f \"$dst/evidence/K1_FALLBACK_CONFIG_PREPARATION.json\"; "
        "cmp -s \"$src/config/K1245_ROUTE_K1_FALLBACK.json\" \"$dst/config/K1245_ROUTE_K1_FALLBACK.json\"; "
        "cmp -s \"$src/evidence/K1_FALLBACK_CONFIG_PREPARATION.json\" \"$dst/evidence/K1_FALLBACK_CONFIG_PREPARATION.json\""
    )
    access.ssh(stage_command, timeout=240)
    preflight = (
        "import pathlib,hashlib,json,datetime,os,py_compile,subprocess;root=pathlib.Path(" + repr(str(REMOTE_R3D)) + ");"
        "expected=" + repr(expected) + ";sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();"
        "actual={p:sha(p) for p in expected};assert actual==expected,(actual,expected);"
        "plan=json.loads((root/'K1_RC_SMOKE_EXECUTION_PLAN_20260923_R3D.json').read_text());"
        "fallback=root/'K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json';"
        "cfg=json.loads((root/'config/K1245_ROUTE_K1_FALLBACK.json').read_text());gate=json.loads((root/'evidence/F_FALLBACK_GATE_VALIDATION.json').read_text());"
        "assert plan['job_name']=='s2k1rcsmkr3d' and plan['strict_gate_result']['result'] is False;"
        "assert plan['training_authorization'] is False and plan['k5_authorization'] is False;"
        "assert cfg['plan_sha256']==sha(fallback) and sha(root/'config/K1245_ROUTE_K1_FALLBACK.json')==plan['k1_config_sha256'];"
        "assert cfg['F_STRICT_PREREG_GATE'] is False and gate['F_STRICT_PREREG_GATE'] is False and gate['status']=='pass';"
        "assert sha(root/'runtime/run_k1_rc_smoke.py')==plan['smoke_runtime_sha256'];"
        "pool=pathlib.Path(" + repr(str(R2_REMOTE / "pools/rc27/component_manifest.json")) + ");"
        "assert sha(pool)==plan['smoke']['pool_manifest_file_sha256'];"
        "assert not subprocess.run(['squeue','-h','-n','s2k1rcsmkr3d','-o','%i'],capture_output=True,text=True,check=True).stdout.strip();"
        "subprocess.run(['bash','-n',str(root/'run_k1_rc_smoke_r3d.slurm')],check=True);"
        "py_compile.compile(str(root/'runtime/run_k1_rc_smoke.py'),doraise=True);"
        "out={'artifact_status':'experimental_unvalidated','operator_approved':False,'formal_gate_status':'not_run_not_passed','status':'pass','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,'job_name':plan['job_name'],'files_sha256':actual,'F_STRICT_PREREG_GATE':False,'training_authorization':False,'k5_authorization':False};"
        "e=root/'evidence';fd=os.open(e/'K1_SMOKE_R3D_STATIC_PREFLIGHT.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400);h=os.fdopen(fd,'w');json.dump(out,h,sort_keys=True,indent=2);h.write('\\n');h.close();"
        "emit=lambda n,x:(lambda fd:(lambda f:(json.dump(x,f,sort_keys=True,indent=2),f.write('\\n'),f.close()))(os.fdopen(fd,'w')))(os.open(e/n,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400));"
        "emit('K1_SMOKE_R3D_RESERVATION.json',{**out,'status':'reserved'});"
        "res=subprocess.run(['sbatch','--hold','--parsable',str(root/'run_k1_rc_smoke_r3d.slurm')],capture_output=True,text=True,check=True);jid=res.stdout.strip().split(';')[0];assert jid.isdigit();"
        "emit('K1_SMOKE_R3D_LAUNCH.json',{**out,'status':'submitted_held','job_id':jid,'slurm_sha256':sha(root/'run_k1_rc_smoke_r3d.slurm')});"
        "subprocess.run(['scontrol','release',jid],check=True);emit('K1_SMOKE_R3D_RELEASED.json',{**out,'status':'released','job_id':jid});"
        "print(json.dumps({'job_id':jid,'job_name':plan['job_name'],'status':'released','F_STRICT_PREREG_GATE':False}))"
    )
    result = access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B -c " + shlex.quote(preflight), timeout=300)
    return {"status": "released", "plan_sha256": sha(plan), "runner_sha256": sha(runner), "remote": result[-1500:]}


def inspect_r3(access: Access) -> dict:
    code = (
        "import json,pathlib,subprocess,hashlib;root=pathlib.Path(" + repr(str(REMOTE)) + ");r2=pathlib.Path(" + repr(str(R2_REMOTE)) + ");"
        "files=sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file());"
        "dirs=sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_dir());"
        "mp=r2/'pools/rc27/component_manifest.json';m=json.loads(mp.read_text());"
        "manifest={'file_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),'identity_sha256':m['manifest_sha256']};"
        "q=subprocess.run(['squeue','-h','-n','s2k1rcsmkr3','-o','%i|%T|%M|%R'],capture_output=True,text=True);"
        "print(json.dumps({'root_exists':root.is_dir(),'files':files,'directories':dirs,'r2_rc27_manifest':manifest,'smoke_squeue':q.stdout.strip(),'squeue_rc':q.returncode},sort_keys=True))"
    )
    return {"remote_snapshot": access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B -c " + shlex.quote(code))}


def deploy_smoke_r3e(access: Access) -> dict:
    attempt = ROOT / "attempts/k1_rc_smoke_r3e"
    plan = attempt / "K1_RC_SMOKE_EXECUTION_PLAN_20260923_R3E.json"
    runner = attempt / "runtime/run_k1_rc_smoke.py"
    slurm = attempt / "run_k1_rc_smoke_r3e.slurm"
    prereg = ROOT.parents[2] / "05_documents/02_plans/20260921_k1245_rc_aware_execution/F_FALLBACK_GATE_PREREGISTRATION_20260923_R3.md"
    remote = REMOTE / "attempts/k1_rc_smoke_r3e"
    sources = [plan, runner, slurm, prereg]
    expected = {}
    access.ssh("set -euo pipefail; mkdir -m 700 -p " +
               " ".join(shlex.quote(str(remote / p)) for p in ("evidence", "config", "runtime", "sources", "logs", "smoke")))
    for src in sources:
        if src == prereg:
            dest = remote / "sources" / src.name
        elif src == runner:
            dest = remote / "runtime" / src.name
        else:
            dest = remote / src.name
        expected[str(dest)] = sha(src)
        access.scp(src, str(dest.parent) + "/")
    access.ssh("set -euo pipefail; src=" + shlex.quote(str(REMOTE)) + "; dst=" + shlex.quote(str(remote)) + "; "
               "for pair in 'K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json' 'config/K1245_ROUTE_K1_FALLBACK.json' 'evidence/F_FALLBACK_GATE_VALIDATION.json' 'evidence/K1_FALLBACK_CONFIG_PREPARATION.json'; do "
               "if test -e \"$dst/$pair\"; then cmp -s \"$src/$pair\" \"$dst/$pair\"; else cp -p \"$src/$pair\" \"$dst/$pair\"; fi; done")
    static = (
        "import pathlib,hashlib,json,datetime,os,subprocess,py_compile;root=pathlib.Path(" + repr(str(remote)) + ");"
        "sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();plan=json.loads((root/'K1_RC_SMOKE_EXECUTION_PLAN_20260923_R3E.json').read_text());"
        "cfg=json.loads((root/'config/K1245_ROUTE_K1_FALLBACK.json').read_text());gate=json.loads((root/'evidence/F_FALLBACK_GATE_VALIDATION.json').read_text());"
        "assert plan['job_name']=='s2k1rcsmkr3e' and plan['strict_gate_result']['result'] is False and not plan['training_authorization'] and not plan['k5_authorization'];"
        "assert cfg['F_STRICT_PREREG_GATE'] is False and sha(root/'config/K1245_ROUTE_K1_FALLBACK.json')==plan['k1_config_sha256'];"
        "assert gate['status']=='pass' and gate['F_STRICT_PREREG_GATE'] is False;"
        "assert sha(root/'runtime/run_k1_rc_smoke.py')==plan['smoke_runtime_sha256'];"
        "pool=pathlib.Path(" + repr(str(R2_REMOTE / "pools/rc27/component_manifest.json")) + ");assert sha(pool)==plan['smoke']['pool_manifest_file_sha256'];"
        "assert not subprocess.run(['squeue','-h','-n','s2k1rcsmkr3e','-o','%i'],capture_output=True,text=True,check=True).stdout.strip();"
        "subprocess.run(['bash','-n',str(root/'run_k1_rc_smoke_r3e.slurm')],check=True);py_compile.compile(str(root/'runtime/run_k1_rc_smoke.py'),doraise=True);"
        "out={'artifact_status':'experimental_unvalidated','operator_approved':False,'formal_gate_status':'not_run_not_passed','status':'pass','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,'job_name':'s2k1rcsmkr3e','F_STRICT_PREREG_GATE':False,'training_authorization':False,'k5_authorization':False};"
        "e=root/'evidence';fd=os.open(e/'K1_SMOKE_R3E_STATIC_PREFLIGHT.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400);h=os.fdopen(fd,'w');json.dump(out,h,sort_keys=True,indent=2);h.write('\\n');h.close();"
        "fd=os.open(e/'K1_SMOKE_R3E_RESERVATION.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400);h=os.fdopen(fd,'w');json.dump({**out,'status':'reserved'},h,sort_keys=True,indent=2);h.write('\\n');h.close();"
        "res=subprocess.run(['sbatch','--hold','--parsable',str(root/'run_k1_rc_smoke_r3e.slurm')],capture_output=True,text=True,check=True);jid=res.stdout.strip().split(';')[0];assert jid.isdigit();"
        "fd=os.open(e/'K1_SMOKE_R3E_LAUNCH.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400);h=os.fdopen(fd,'w');json.dump({**out,'status':'submitted_held','job_id':jid},h,sort_keys=True,indent=2);h.write('\\n');h.close();"
        "subprocess.run(['scontrol','release',jid],check=True);fd=os.open(e/'K1_SMOKE_R3E_RELEASED.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400);h=os.fdopen(fd,'w');json.dump({**out,'status':'released','job_id':jid},h,sort_keys=True,indent=2);h.write('\\n');h.close();print(json.dumps({'job_id':jid,'status':'released','F_STRICT_PREREG_GATE':False}))"
    )
    result = access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B -c " + shlex.quote(static), timeout=300)
    return {"status": "released", "remote_output": result[-1500:]}


def deploy_smoke_r3f(access: Access) -> dict:
    """Deploy one isolated K1 smoke with the complete shared runtime bundle."""
    attempt = ROOT / "attempts/k1_rc_smoke_r3f"
    repo = ROOT.parents[2] / "01_code/repository"
    prereg = ROOT.parents[2] / "05_documents/02_plans/20260921_k1245_rc_aware_execution/F_FALLBACK_GATE_PREREGISTRATION_20260923_R3.md"
    plan = attempt / "K1_RC_SMOKE_EXECUTION_PLAN_20260923_R3F.json"
    runner = attempt / "runtime/run_k1_rc_smoke.py"
    slurm = attempt / "run_k1_rc_smoke_r3f.slurm"
    files = [
        plan, runner, slurm, prereg,
        ROOT / "K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json",
        ROOT / "K1_FALLBACK_CONFIG_PLAN_20260923_R3A.json",
        ROOT / "runtime/prepare_k1_fallback_config_r3a.py",
        repo / "scripts/09_k1245/62_run_k1245_smoke.py",
        *sorted((repo / "custom_nnunet").glob("*.py")),
    ]
    access.ssh(
        "set -euo pipefail; mkdir -m 700 -p "
        + " ".join(shlex.quote(str(REMOTE_R3F / p)) for p in (
            "evidence", "config", "runtime", "runtime/repository/custom_nnunet",
            "runtime/repository/scripts/09_k1245", "sources", "logs", "operator", "smoke"
        )) + "; "
        + "for marker in K1_SMOKE_R3F_LAUNCH.json K1_SMOKE_R3F_RELEASED.json K1_SMOKE_R3F_STARTED.json K1_SMOKE_R3F_FAILED.json K1_SMOKE_R3F_VALIDATION.json; do test ! -e "
        + shlex.quote(str(REMOTE_R3F / "evidence")) + "/$marker; done"
    )
    expected = {}
    for src in files:
        if repo in src.parents:
            dest = REMOTE_R3F / "runtime/repository" / src.relative_to(repo)
        elif src == prereg:
            dest = REMOTE_R3F / "sources" / src.name
        elif src == ROOT / "K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json":
            dest = REMOTE_R3F / src.name
        elif src == ROOT / "K1_FALLBACK_CONFIG_PLAN_20260923_R3A.json":
            dest = REMOTE_R3F / src.name
        elif src == ROOT / "runtime/prepare_k1_fallback_config_r3a.py":
            dest = REMOTE_R3F / "runtime" / src.name
        elif attempt in src.parents:
            dest = REMOTE_R3F / src.relative_to(attempt)
        else:
            raise RuntimeError(f"unclassified R3F source: {src}")
        expected[str(dest)] = sha(src)
        access.scp_immutable(src, str(dest.parent))

    stage = (
        "set -euo pipefail; src=" + shlex.quote(str(REMOTE)) + "; dst=" + shlex.quote(str(REMOTE_R3F)) + "; "
        "for pair in 'K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json' 'config/K1245_ROUTE_K1_FALLBACK.json' 'evidence/F_FALLBACK_GATE_VALIDATION.json' 'evidence/K1_FALLBACK_CONFIG_PREPARATION.json'; do "
        "if test -e \"$dst/$pair\"; then cmp -s \"$src/$pair\" \"$dst/$pair\"; else cp -p \"$src/$pair\" \"$dst/$pair\"; fi; done"
    )
    access.ssh(stage, timeout=240)
    expected_literal = repr(expected)
    static = (
        "import pathlib,hashlib,json,datetime,os,py_compile,subprocess,sys,importlib.util;"
        "root=pathlib.Path(" + repr(str(REMOTE_R3F)) + ");expected=" + expected_literal + ";"
        "sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();"
        "actual={p:sha(p) for p in expected};assert actual==expected,(actual,expected);"
        "plan=json.loads((root/'K1_RC_SMOKE_EXECUTION_PLAN_20260923_R3F.json').read_text());"
        "cfg=json.loads((root/'config/K1245_ROUTE_K1_FALLBACK.json').read_text());gate=json.loads((root/'evidence/F_FALLBACK_GATE_VALIDATION.json').read_text());"
        "assert plan['job_name']=='s2k1rcsmkr3f' and plan['strict_gate_result']['result'] is False;"
        "assert plan['training_authorization'] is False and plan['k5_authorization'] is False;"
        "assert cfg['F_STRICT_PREREG_GATE'] is False and gate['status']=='pass' and gate['F_STRICT_PREREG_GATE'] is False;"
        "assert sha(root/'config/K1245_ROUTE_K1_FALLBACK.json')==plan['k1_config_sha256'];"
        "pool=pathlib.Path(" + repr(str(R2_REMOTE / "pools/rc27/component_manifest.json")) + ");"
        "pm=json.loads(pool.read_text());assert sha(pool)==plan['smoke']['pool_manifest_file_sha256'] and pm['manifest_sha256']==plan['smoke']['pool_manifest_identity_sha256'];"
        "helper=root/'runtime/repository/scripts/09_k1245/62_run_k1245_smoke.py';assert helper.is_file();"
        "assert 'r3d' not in (root/'run_k1_rc_smoke_r3f.slurm').read_text() and 'r3e' not in (root/'run_k1_rc_smoke_r3f.slurm').read_text();"
        "compile_files=[helper,*sorted((root/'runtime/repository/custom_nnunet').glob('*.py'))];[py_compile.compile(str(p),doraise=True) for p in compile_files];"
        "sys.path.insert(0,str(root/'runtime/repository'));spec=importlib.util.spec_from_file_location('k1245_smoke_shared',helper);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);import custom_nnunet.met_aug_k1245;"
        "assert not subprocess.run(['squeue','-h','-n','s2k1rcsmkr3f','-o','%i'],capture_output=True,text=True,check=True).stdout.strip();"
        "subprocess.run(['bash','-n',str(root/'run_k1_rc_smoke_r3f.slurm')],check=True);"
        "out={'artifact_status':'experimental_unvalidated','operator_approved':False,'formal_gate_status':'not_run_not_passed','status':'pass','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,'job_name':'s2k1rcsmkr3f','F_STRICT_PREREG_GATE':False,'training_authorization':False,'k5_authorization':False,'runtime_helper_sha256':sha(helper),'custom_nnunet_file_count':len(list((root/'runtime/repository/custom_nnunet').glob('*.py'))),'files_sha256':actual};"
        "e=root/'evidence';emit=lambda n,x:(lambda fd:(lambda f:(json.dump(x,f,sort_keys=True,indent=2),f.write('\\n'),f.close()))(os.fdopen(fd,'w')))(os.open(e/n,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400));"
        "emit('K1_SMOKE_R3F_STATIC_PREFLIGHT.json',out);emit('K1_SMOKE_R3F_RESERVATION.json',{**out,'status':'reserved'});"
        "res=subprocess.run(['sbatch','--hold','--parsable',str(root/'run_k1_rc_smoke_r3f.slurm')],capture_output=True,text=True,check=True);jid=res.stdout.strip().split(';')[0];assert jid.isdigit();"
        "emit('K1_SMOKE_R3F_LAUNCH.json',{**out,'status':'submitted_held','job_id':jid,'slurm_sha256':sha(root/'run_k1_rc_smoke_r3f.slurm')});subprocess.run(['scontrol','release',jid],check=True);emit('K1_SMOKE_R3F_RELEASED.json',{**out,'status':'released','job_id':jid});print(json.dumps({'job_id':jid,'job_name':'s2k1rcsmkr3f','status':'released','F_STRICT_PREREG_GATE':False},sort_keys=True))"
    )
    result = access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B -c " + shlex.quote(static), timeout=360)
    return {"status": "released", "plan_sha256": sha(plan), "runner_sha256": sha(runner), "helper_sha256": sha(repo / "scripts/09_k1245/62_run_k1245_smoke.py"), "source_count": len(expected), "remote": result[-2000:]}


def inspect_r3f(access: Access) -> dict:
    root = REMOTE_R3F
    code = (
        "import pathlib,subprocess,json,datetime,os;root=pathlib.Path(" + repr(str(root)) + ");"
        "marks={n:(root/'evidence'/n).is_file() for n in ('K1_SMOKE_R3F_STATIC_PREFLIGHT.json','K1_SMOKE_R3F_RESERVATION.json','K1_SMOKE_R3F_LAUNCH.json','K1_SMOKE_R3F_RELEASED.json','K1_SMOKE_R3F_STARTED.json','K1_SMOKE_R3F_FAILED.json','K1_SMOKE_R3F_VALIDATION.json')};"
        "q=subprocess.run(['squeue','-h','-n','s2k1rcsmkr3f','-o','%i|%T|%M|%R'],capture_output=True,text=True);"
        "launch=root/'evidence/K1_SMOKE_R3F_LAUNCH.json';jid=json.loads(launch.read_text()).get('job_id') if launch.is_file() else None;"
        "a=subprocess.run(['sacct','-n','-X','-j',str(jid),'-o','JobIDRaw,JobName,State,ExitCode,Submit,Start,End'],capture_output=True,text=True) if jid else None;"
        "print(json.dumps({'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,'markers':marks,'squeue':q.stdout.strip(),'squeue_rc':q.returncode,'job_id':jid,'sacct':a.stdout.strip() if a else None,'sacct_rc':a.returncode if a else None},sort_keys=True))"
    )
    return {"remote_snapshot": access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B -c " + shlex.quote(code), timeout=180)}


def deploy_smoke_r3g(access: Access) -> dict:
    attempt = ROOT / "attempts/k1_rc_smoke_r3g"
    repo = ROOT.parents[2] / "01_code/repository"
    prereg = ROOT.parents[2] / "05_documents/02_plans/20260921_k1245_rc_aware_execution/F_FALLBACK_GATE_PREREGISTRATION_20260923_R3.md"
    plan = attempt / "K1_RC_SMOKE_EXECUTION_PLAN_20260923_R3G.json"
    runner = attempt / "runtime/run_k1_rc_smoke.py"
    slurm = attempt / "run_k1_rc_smoke_r3g.slurm"
    files = [plan, runner, slurm, prereg,
             ROOT / "K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json",
             ROOT / "K1_FALLBACK_CONFIG_PLAN_20260923_R3A.json",
             ROOT / "runtime/prepare_k1_fallback_config_r3a.py",
             repo / "scripts/09_k1245/62_run_k1245_smoke.py",
             *sorted((repo / "custom_nnunet").glob("*.py"))]
    expected = {}
    access.ssh("set -euo pipefail; test ! -e " + shlex.quote(str(REMOTE_R3G)) + "; mkdir -m 700 -p " +
               " ".join(shlex.quote(str(REMOTE_R3G / p)) for p in
                        ("evidence", "config", "runtime", "runtime/repository/custom_nnunet",
                         "runtime/repository/scripts/09_k1245", "sources", "logs", "operator", "smoke")))
    for src in files:
        if repo in src.parents:
            dest = REMOTE_R3G / "runtime/repository" / src.relative_to(repo)
        elif src == prereg:
            dest = REMOTE_R3G / "sources" / src.name
        elif src == ROOT / "K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json":
            dest = REMOTE_R3G / src.name
        elif src == ROOT / "K1_FALLBACK_CONFIG_PLAN_20260923_R3A.json":
            dest = REMOTE_R3G / src.name
        elif src == ROOT / "runtime/prepare_k1_fallback_config_r3a.py":
            dest = REMOTE_R3G / "runtime" / src.name
        elif attempt in src.parents:
            dest = REMOTE_R3G / src.relative_to(attempt)
        else:
            raise RuntimeError(f"unclassified R3G source: {src}")
        expected[str(dest)] = sha(src)
        access.scp_immutable(src, str(dest.parent))
    stage = (
        "set -euo pipefail; src=" + shlex.quote(str(REMOTE)) + "; dst=" + shlex.quote(str(REMOTE_R3G)) + "; "
        "for pair in 'K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json' 'config/K1245_ROUTE_K1_FALLBACK.json' 'evidence/F_FALLBACK_GATE_VALIDATION.json' 'evidence/K1_FALLBACK_CONFIG_PREPARATION.json'; do "
        "if test -e \"$dst/$pair\"; then cmp -s \"$src/$pair\" \"$dst/$pair\"; else cp -p \"$src/$pair\" \"$dst/$pair\"; fi; done"
    )
    access.ssh(stage, timeout=240)
    static = (
        "import pathlib,hashlib,json,datetime,os,py_compile,subprocess,sys,importlib.util;root=pathlib.Path(" + repr(str(REMOTE_R3G)) + ");expected=" + repr(expected) + ";sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();actual={p:sha(p) for p in expected};assert actual==expected,(actual,expected);"
        "plan=json.loads((root/'K1_RC_SMOKE_EXECUTION_PLAN_20260923_R3G.json').read_text());cfg=json.loads((root/'config/K1245_ROUTE_K1_FALLBACK.json').read_text());gate=json.loads((root/'evidence/F_FALLBACK_GATE_VALIDATION.json').read_text());"
        "assert plan['job_name']=='s2k1rcsmkr3g' and plan['strict_gate_result']['result'] is False and plan['training_authorization'] is False and plan['k5_authorization'] is False;"
        "assert plan['parent_contract']['fix_v3_calibration_file_sha256']=='68a5d8a6c501804b7a47d626e43a58ff71b781154c0cc955a9c5a5d5b786d37f';"
        "assert cfg['F_STRICT_PREREG_GATE'] is False and gate['status']=='pass' and gate['F_STRICT_PREREG_GATE'] is False;assert sha(root/'config/K1245_ROUTE_K1_FALLBACK.json')==plan['k1_config_sha256'];"
        "pool=pathlib.Path(" + repr(str(R2_REMOTE / "pools/rc27/component_manifest.json")) + ");pm=json.loads(pool.read_text());assert sha(pool)==plan['smoke']['pool_manifest_file_sha256'] and pm['manifest_sha256']==plan['smoke']['pool_manifest_identity_sha256'];"
        "helper=root/'runtime/repository/scripts/09_k1245/62_run_k1245_smoke.py';runner=root/'runtime/run_k1_rc_smoke.py';assert helper.is_file() and sha(runner)==plan['smoke_runtime_sha256'];"
        "compile_files=[runner,helper,*sorted((root/'runtime/repository/custom_nnunet').glob('*.py'))];[py_compile.compile(str(p),doraise=True) for p in compile_files];sys.path.insert(0,str(root/'runtime/repository'));spec=importlib.util.spec_from_file_location('k1245_smoke_shared',helper);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);import custom_nnunet.met_aug_k1245;"
        "assert not subprocess.run(['squeue','-h','-n','s2k1rcsmkr3g','-o','%i'],capture_output=True,text=True,check=True).stdout.strip();subprocess.run(['bash','-n',str(root/'run_k1_rc_smoke_r3g.slurm')],check=True);"
        "out={'artifact_status':'experimental_unvalidated','operator_approved':False,'formal_gate_status':'not_run_not_passed','status':'pass','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,'job_name':'s2k1rcsmkr3g','F_STRICT_PREREG_GATE':False,'training_authorization':False,'k5_authorization':False,'runtime_helper_sha256':sha(helper),'custom_nnunet_file_count':len(list((root/'runtime/repository/custom_nnunet').glob('*.py'))),'files_sha256':actual};e=root/'evidence';emit=lambda n,x:(lambda fd:(lambda f:(json.dump(x,f,sort_keys=True,indent=2),f.write('\\n'),f.close()))(os.fdopen(fd,'w')))(os.open(e/n,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400));emit('K1_SMOKE_R3G_STATIC_PREFLIGHT.json',out);emit('K1_SMOKE_R3G_RESERVATION.json',{**out,'status':'reserved'});res=subprocess.run(['sbatch','--hold','--parsable',str(root/'run_k1_rc_smoke_r3g.slurm')],capture_output=True,text=True,check=True);jid=res.stdout.strip().split(';')[0];assert jid.isdigit();emit('K1_SMOKE_R3G_LAUNCH.json',{**out,'status':'submitted_held','job_id':jid,'slurm_sha256':sha(root/'run_k1_rc_smoke_r3g.slurm')});subprocess.run(['scontrol','release',jid],check=True);emit('K1_SMOKE_R3G_RELEASED.json',{**out,'status':'released','job_id':jid});print(json.dumps({'job_id':jid,'job_name':'s2k1rcsmkr3g','status':'released','F_STRICT_PREREG_GATE':False},sort_keys=True))"
    )
    result = access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B -c " + shlex.quote(static), timeout=360)
    return {"status":"released","plan_sha256":sha(plan),"runner_sha256":sha(runner),"source_count":len(expected),"remote":result[-2000:]}


def inspect_r3g(access: Access) -> dict:
    root = REMOTE_R3G
    code = (
        "import pathlib,subprocess,json,datetime,os;root=pathlib.Path(" + repr(str(root)) + ");marks={n:(root/'evidence'/n).is_file() for n in ('K1_SMOKE_R3G_STATIC_PREFLIGHT.json','K1_SMOKE_R3G_RESERVATION.json','K1_SMOKE_R3G_LAUNCH.json','K1_SMOKE_R3G_RELEASED.json','K1_SMOKE_R3G_STARTED.json','K1_SMOKE_R3G_FAILED.json','K1_SMOKE_R3G_VALIDATION.json')};q=subprocess.run(['squeue','-h','-n','s2k1rcsmkr3g','-o','%i|%T|%M|%R'],capture_output=True,text=True);launch=root/'evidence/K1_SMOKE_R3G_LAUNCH.json';jid=json.loads(launch.read_text()).get('job_id') if launch.is_file() else None;a=subprocess.run(['sacct','-n','-X','-j',str(jid),'-o','JobIDRaw,JobName,State,ExitCode,Submit,Start,End'],capture_output=True,text=True) if jid else None;print(json.dumps({'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,'markers':marks,'squeue':q.stdout.strip(),'job_id':jid,'sacct':a.stdout.strip() if a else None,'squeue_rc':q.returncode,'sacct_rc':a.returncode if a else None},sort_keys=True))"
    )
    return {"remote_snapshot":access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B -c " + shlex.quote(code),timeout=180)}


def inspect_r3d(access: Access) -> dict:
    code = (
        "import pathlib,subprocess,json,datetime;root=pathlib.Path(" + repr(str(REMOTE_R3D)) + ");"
        "files=sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()) if root.exists() else [];"
        "q=subprocess.run(['squeue','-h','-n','s2k1rcsmkr3d','-o','%i|%T|%M|%R'],capture_output=True,text=True);"
        "launch=root/'evidence/K1_SMOKE_R3D_LAUNCH.json';jid=json.loads(launch.read_text()).get('job_id') if launch.is_file() else None;"
        "a=subprocess.run(['sacct','-n','-X','-j',str(jid),'-o','JobIDRaw,JobName,State,ExitCode,Submit,Start,End'],capture_output=True,text=True) if jid else None;"
        "marks={n:(root/'evidence'/n).is_file() for n in ('K1_SMOKE_R3D_STATIC_PREFLIGHT.json','K1_SMOKE_R3D_RESERVATION.json','K1_SMOKE_R3D_LAUNCH.json','K1_SMOKE_R3D_RELEASED.json','K1_SMOKE_R3D_STARTED.json','K1_SMOKE_R3D_FAILED.json','K1_SMOKE_R3D_VALIDATION.json')};"
        "print(json.dumps({'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':__import__('os').uname().nodename,'root_exists':root.exists(),'files':files,'markers':marks,'squeue':q.stdout.strip(),'squeue_rc':q.returncode,'job_id':jid,'sacct':a.stdout.strip() if a else None,'sacct_rc':a.returncode if a else None},sort_keys=True))"
    )
    return {"remote_snapshot": access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B -c " + shlex.quote(code), timeout=180)}


def compare_r3d(access: Access) -> dict:
    code = (
        "import pathlib,hashlib,json;src=pathlib.Path(" + repr(str(REMOTE)) + ");dst=pathlib.Path(" + repr(str(REMOTE_R3D)) + ");"
        "names=('config/K1245_ROUTE_K1_FALLBACK.json','evidence/F_FALLBACK_GATE_VALIDATION.json','evidence/K1_FALLBACK_CONFIG_PREPARATION.json','K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json');"
        "h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None;"
        "print(json.dumps({n:{'parent':h(src/n),'r3d':h(dst/n),'same':h(src/n)==h(dst/n)} for n in names},sort_keys=True))"
    )
    return {"comparison": access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B -c " + shlex.quote(code), timeout=180)}


def inspect_r3e(access: Access) -> dict:
    root = REMOTE / "attempts/k1_rc_smoke_r3e"
    code = (
        "import pathlib,subprocess,json,datetime,os;root=pathlib.Path(" + repr(str(root)) + ");"
        "marks={n:(root/'evidence'/n).is_file() for n in ('K1_SMOKE_R3E_STATIC_PREFLIGHT.json','K1_SMOKE_R3E_RESERVATION.json','K1_SMOKE_R3E_LAUNCH.json','K1_SMOKE_R3E_RELEASED.json','K1_SMOKE_R3E_STARTED.json','K1_SMOKE_R3E_FAILED.json','K1_SMOKE_R3E_VALIDATION.json')};"
        "q=subprocess.run(['squeue','-h','-n','s2k1rcsmkr3e','-o','%i|%T|%M|%R'],capture_output=True,text=True);"
        "jid='3736556';a=subprocess.run(['sacct','-n','-X','-j',jid,'-o','JobIDRaw,JobName,State,ExitCode,Submit,Start,End'],capture_output=True,text=True);"
        "print(json.dumps({'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,'markers':marks,'squeue':q.stdout.strip(),'sacct':a.stdout.strip(),'squeue_rc':q.returncode,'sacct_rc':a.returncode},sort_keys=True))"
    )
    return {"remote_snapshot": access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B -c " + shlex.quote(code), timeout=180)}


def deploy_k4_small_r2(access: Access) -> dict:
    attempt = ROOT / "attempts/k4_small_smoke_r2"
    remote = REMOTE / "attempts/k4_small_smoke_r2"
    plan = attempt / "K4_SMALL_SMOKE_EXECUTION_PLAN_20260923_R2.json"
    runner = attempt / "runtime/run_k4_small_smoke.py"
    slurm = attempt / "run_k4_small_smoke_r2.slurm"
    repo = ROOT.parents[2] / "01_code/repository"
    parent_plan = ROOT.parent / "s2_met_aug_k1245_20260923_r2/K1245_RC_AWARE_EXECUTION_PLAN_20260923_R2.json"
    sources = [plan, runner, slurm, parent_plan,
               repo / "scripts/09_k1245/62_run_k1245_smoke.py",
               *sorted((repo / "custom_nnunet").glob("*.py"))]
    expected = {}
    access.ssh("set -euo pipefail; mkdir -m 700 -p " + " ".join(shlex.quote(str(remote / p)) for p in
               ("evidence", "runtime", "runtime/repository/custom_nnunet", "runtime/repository/scripts/09_k1245", "logs", "smoke", "sources")))
    for src in sources:
        if src == parent_plan:
            dest = remote / "sources" / src.name
        elif repo in src.parents:
            dest = remote / "runtime/repository" / src.relative_to(repo)
        elif src == runner:
            dest = remote / "runtime" / src.name
        else:
            dest = remote / src.name
        expected[str(dest)] = sha(src)
        access.scp_immutable(src, str(dest.parent))
    stage = (
        "set -euo pipefail; src=" + shlex.quote(str(R2_REMOTE)) + "; dst=" + shlex.quote(str(remote)) + "; "
        "for pair in 'config/K1245_ROUTE_K4.json' 'pools/small8/component_manifest.json' 'evidence/REMOTE_STATIC_PREFLIGHT.json' 'evidence/POOL_BUILD_COMPLETE.ok'; do "
        "mkdir -p \"$dst/$(dirname \"$pair\")\"; if test -e \"$dst/$pair\"; then cmp -s \"$src/$pair\" \"$dst/$pair\"; else cp -p \"$src/$pair\" \"$dst/$pair\"; fi; done"
    )
    access.ssh(stage)
    static = (
        "import pathlib,hashlib,json,datetime,os,subprocess,py_compile;root=pathlib.Path(" + repr(str(remote)) + ");r2=pathlib.Path(" + repr(str(R2_REMOTE)) + ");"
        "sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();plan=json.loads((root/'K4_SMALL_SMOKE_EXECUTION_PLAN_20260923_R2.json').read_text());"
        "cfg=json.loads((root/'config/K1245_ROUTE_K4.json').read_text());manifest=json.loads((root/'pools/small8/component_manifest.json').read_text());"
        "assert plan['route_id']=='K4' and plan['fallback_context']['F_STRICT_PREREG_GATE'] is False and not plan['training_authorization'] and not plan['k5_authorization'];"
        "assert sha(root/'sources/K1245_RC_AWARE_EXECUTION_PLAN_20260923_R2.json')==plan['parent_plan_sha256'];"
        "assert sha(root/'config/K1245_ROUTE_K4.json')==plan['config_sha256'] and cfg['route_id']=='K4' and cfg['manifest_sha256']==manifest['manifest_sha256'];"
        "assert sha(root/'pools/small8/component_manifest.json')==plan['pool']['manifest_file_sha256'] and manifest['manifest_sha256']==plan['pool']['manifest_identity_sha256'];"
        "assert manifest['component_count']>=24 and manifest['constraints']['include_rc'] is False;"
        "assert sha(root/'runtime/run_k4_small_smoke.py')==plan['smoke_runtime_sha256'];"
        "assert not subprocess.run(['squeue','-h','-n','s2k4smallsmkr2','-o','%i'],capture_output=True,text=True,check=True).stdout.strip();"
        "subprocess.run(['bash','-n',str(root/'run_k4_small_smoke_r2.slurm')],check=True);py_compile.compile(str(root/'runtime/run_k4_small_smoke.py'),doraise=True);"
        "out={'artifact_status':'experimental_unvalidated','operator_approved':False,'formal_gate_status':'not_run_not_passed','status':'pass','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,'job_name':'s2k4smallsmkr2','route_id':'K4','F_STRICT_PREREG_GATE':False,'training_authorization':False,'k5_authorization':False,'small8_manifest_sha256':sha(root/'pools/small8/component_manifest.json')};"
        "e=root/'evidence';emit=lambda n,x:(lambda fd:(lambda f:(json.dump(x,f,sort_keys=True,indent=2),f.write('\\n'),f.close()))(os.fdopen(fd,'w')))(os.open(e/n,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400));"
        "emit('K4_SMALL_SMOKE_STATIC_PREFLIGHT.json',out);emit('K4_SMALL_SMOKE_RESERVATION.json',{**out,'status':'reserved'});"
        "res=subprocess.run(['sbatch','--hold','--parsable',str(root/'run_k4_small_smoke_r2.slurm')],capture_output=True,text=True,check=True);jid=res.stdout.strip().split(';')[0];assert jid.isdigit();"
        "emit('K4_SMALL_SMOKE_LAUNCH.json',{**out,'status':'submitted_held','job_id':jid});subprocess.run(['scontrol','release',jid],check=True);emit('K4_SMALL_SMOKE_RELEASED.json',{**out,'status':'released','job_id':jid});print(json.dumps({'job_id':jid,'status':'released','F_STRICT_PREREG_GATE':False}))"
    )
    result = access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B -c " + shlex.quote(static), timeout=300)
    return {"status": "released", "remote_output": result[-1500:]}


def deploy_k4_small_r3(access: Access) -> dict:
    attempt = ROOT / "attempts/k4_small_smoke_r3"
    remote = REMOTE_K4_R3
    plan = attempt / "K4_SMALL_SMOKE_EXECUTION_PLAN_20260923_R3.json"
    runner = attempt / "runtime/run_k4_small_smoke.py"
    slurm = attempt / "run_k4_small_smoke_r3.slurm"
    repo = ROOT.parents[2] / "01_code/repository"
    parent_plan = ROOT.parent / "s2_met_aug_k1245_20260923_r2/K1245_RC_AWARE_EXECUTION_PLAN_20260923_R2.json"
    sources = [plan, runner, slurm, parent_plan,
               repo / "scripts/09_k1245/62_run_k1245_smoke.py",
               *sorted((repo / "custom_nnunet").glob("*.py"))]
    expected = {}
    access.ssh("set -euo pipefail; test ! -e " + shlex.quote(str(remote)) + "; mkdir -m 700 -p " +
               " ".join(shlex.quote(str(remote / p)) for p in
                        ("evidence", "runtime", "runtime/repository/custom_nnunet",
                         "runtime/repository/scripts/09_k1245", "logs", "smoke", "sources")))
    for src in sources:
        if src == parent_plan:
            dest = remote / "sources" / src.name
        elif repo in src.parents:
            dest = remote / "runtime/repository" / src.relative_to(repo)
        elif src == runner:
            dest = remote / "runtime" / src.name
        else:
            dest = remote / src.name
        expected[str(dest)] = sha(src)
        access.scp_immutable(src, str(dest.parent))
    stage = (
        "set -euo pipefail; src=" + shlex.quote(str(R2_REMOTE)) + "; dst=" + shlex.quote(str(remote)) + "; "
        "for pair in 'config/K1245_ROUTE_K4.json' 'pools/small8/component_manifest.json' 'evidence/REMOTE_STATIC_PREFLIGHT.json' 'evidence/POOL_BUILD_COMPLETE.ok'; do "
        "mkdir -p \"$dst/$(dirname \"$pair\")\"; if test -e \"$dst/$pair\"; then cmp -s \"$src/$pair\" \"$dst/$pair\"; else cp -p \"$src/$pair\" \"$dst/$pair\"; fi; done"
    )
    access.ssh(stage, timeout=240)
    static = (
        "import pathlib,hashlib,json,datetime,os,subprocess,py_compile,sys,importlib.util;root=pathlib.Path(" + repr(str(remote)) + ");expected=" + repr(expected) + ";"
        "sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();actual={p:sha(p) for p in expected};assert actual==expected,(actual,expected);"
        "plan=json.loads((root/'K4_SMALL_SMOKE_EXECUTION_PLAN_20260923_R3.json').read_text());cfg=json.loads((root/'config/K1245_ROUTE_K4.json').read_text());manifest=json.loads((root/'pools/small8/component_manifest.json').read_text());"
        "assert plan['job_name']=='s2k4smallsmkr3' and plan['route_id']=='K4' and plan['fallback_context']['F_STRICT_PREREG_GATE'] is False;"
        "assert plan['training_authorization'] is False and plan['k5_authorization'] is False;assert sha(root/'sources/K1245_RC_AWARE_EXECUTION_PLAN_20260923_R2.json')==plan['parent_plan_sha256'];"
        "assert sha(root/'config/K1245_ROUTE_K4.json')==plan['config_sha256'] and cfg['route_id']=='K4' and cfg['manifest_sha256']==manifest['manifest_sha256'];"
        "assert sha(root/'pools/small8/component_manifest.json')==plan['pool']['manifest_file_sha256'] and manifest['manifest_sha256']==plan['pool']['manifest_identity_sha256'];"
        "assert manifest['component_count']>=24 and manifest['constraints']['include_rc'] is False;runner=root/'runtime/run_k4_small_smoke.py';assert sha(runner)==plan['smoke_runtime_sha256'];"
        "assert 'render_without_contour' in runner.read_text() and '.contour(' not in runner.read_text();"
        "compile_files=[runner,root/'runtime/repository/scripts/09_k1245/62_run_k1245_smoke.py',*sorted((root/'runtime/repository/custom_nnunet').glob('*.py'))];[py_compile.compile(str(p),doraise=True) for p in compile_files];"
        "sys.path.insert(0,str(root/'runtime/repository'));spec=importlib.util.spec_from_file_location('k4_runner',runner);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);"
        "assert not subprocess.run(['squeue','-h','-n','s2k4smallsmkr3','-o','%i'],capture_output=True,text=True,check=True).stdout.strip();subprocess.run(['bash','-n',str(root/'run_k4_small_smoke_r3.slurm')],check=True);"
        "out={'artifact_status':'experimental_unvalidated','operator_approved':False,'formal_gate_status':'not_run_not_passed','status':'pass','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,'job_name':'s2k4smallsmkr3','route_id':'K4','F_STRICT_PREREG_GATE':False,'training_authorization':False,'k5_authorization':False,'runtime_sha256':sha(runner),'small8_manifest_sha256':sha(root/'pools/small8/component_manifest.json'),'files_sha256':actual};"
        "e=root/'evidence';emit=lambda n,x:(lambda fd:(lambda f:(json.dump(x,f,sort_keys=True,indent=2),f.write('\\n'),f.close()))(os.fdopen(fd,'w')))(os.open(e/n,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400));emit('K4_SMALL_SMOKE_STATIC_PREFLIGHT.json',out);emit('K4_SMALL_SMOKE_RESERVATION.json',{**out,'status':'reserved'});"
        "res=subprocess.run(['sbatch','--hold','--parsable',str(root/'run_k4_small_smoke_r3.slurm')],capture_output=True,text=True,check=True);jid=res.stdout.strip().split(';')[0];assert jid.isdigit();emit('K4_SMALL_SMOKE_LAUNCH.json',{**out,'status':'submitted_held','job_id':jid,'slurm_sha256':sha(root/'run_k4_small_smoke_r3.slurm')});subprocess.run(['scontrol','release',jid],check=True);emit('K4_SMALL_SMOKE_RELEASED.json',{**out,'status':'released','job_id':jid});print(json.dumps({'job_id':jid,'job_name':'s2k4smallsmkr3','status':'released','F_STRICT_PREREG_GATE':False},sort_keys=True))"
    )
    result = access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B -c " + shlex.quote(static), timeout=360)
    return {"status":"released","plan_sha256":sha(plan),"runner_sha256":sha(runner),"source_count":len(expected),"remote":result[-2000:]}


def inspect_k4_small_r3(access: Access) -> dict:
    root = REMOTE_K4_R3
    code = (
        "import pathlib,subprocess,json,datetime,os;root=pathlib.Path(" + repr(str(root)) + ");marks={n:(root/'evidence'/n).is_file() for n in ('K4_SMALL_SMOKE_STATIC_PREFLIGHT.json','K4_SMALL_SMOKE_RESERVATION.json','K4_SMALL_SMOKE_LAUNCH.json','K4_SMALL_SMOKE_RELEASED.json','K4_SMALL_SMOKE_STARTED.json','K4_SMALL_SMOKE_FAILED.json','K4_SMALL_SMOKE_VALIDATION.json')};"
        "q=subprocess.run(['squeue','-h','-n','s2k4smallsmkr3','-o','%i|%T|%M|%R'],capture_output=True,text=True);launch=root/'evidence/K4_SMALL_SMOKE_LAUNCH.json';jid=json.loads(launch.read_text()).get('job_id') if launch.is_file() else None;a=subprocess.run(['sacct','-n','-X','-j',str(jid),'-o','JobIDRaw,JobName,State,ExitCode,Submit,Start,End'],capture_output=True,text=True) if jid else None;print(json.dumps({'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,'markers':marks,'squeue':q.stdout.strip(),'job_id':jid,'sacct':a.stdout.strip() if a else None,'squeue_rc':q.returncode,'sacct_rc':a.returncode if a else None},sort_keys=True))"
    )
    return {"remote_snapshot":access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B -c " + shlex.quote(code),timeout=180)}


def diagnose_k4_small_r2(access: Access) -> dict:
    code = (
        "import pathlib,hashlib,json;root=pathlib.Path(" + repr(str(R2_REMOTE)) + ");"
        "h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();"
        "mp=root/'pools/small8/component_manifest.json';cp=root/'config/K1245_ROUTE_K4.json';"
        "m=json.loads(mp.read_text());c=json.loads(cp.read_text());"
        "print(json.dumps({'manifest_file_sha256':h(mp),'manifest_identity_sha256':m.get('manifest_sha256'),'config_file_sha256':h(cp),'config_manifest_binding':c.get('manifest_sha256'),'config_route':c.get('route_id'),'pool_complete':(root/'evidence/POOL_BUILD_COMPLETE.ok').is_file(),'smoke_queue':__import__('subprocess').run(['squeue','-h','-n','s2k4smallsmkr2','-o','%i|%T'],capture_output=True,text=True).stdout.strip()},sort_keys=True))"
    )
    return {"diagnostic": access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B -c " + shlex.quote(code), timeout=180)}


def inspect_k4_small_r2(access: Access) -> dict:
    root = REMOTE / "attempts/k4_small_smoke_r2"
    code = (
        "import pathlib,subprocess,json,datetime,os;root=pathlib.Path(" + repr(str(root)) + ");"
        "marks={n:(root/'evidence'/n).is_file() for n in ('K4_SMALL_SMOKE_STATIC_PREFLIGHT.json','K4_SMALL_SMOKE_RESERVATION.json','K4_SMALL_SMOKE_LAUNCH.json','K4_SMALL_SMOKE_RELEASED.json','K4_SMALL_SMOKE_STARTED.json','K4_SMALL_SMOKE_FAILED.json','K4_SMALL_SMOKE_VALIDATION.json')};"
        "q=subprocess.run(['squeue','-h','-n','s2k4smallsmkr2','-o','%i|%T|%M|%R'],capture_output=True,text=True);"
        "a=subprocess.run(['sacct','-n','-X','-j','3737611','-o','JobIDRaw,JobName,State,ExitCode,Submit,Start,End'],capture_output=True,text=True);"
        "print(json.dumps({'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,'markers':marks,'squeue':q.stdout.strip(),'sacct':a.stdout.strip(),'squeue_rc':q.returncode,'sacct_rc':a.returncode},sort_keys=True))"
    )
    return {"remote_snapshot": access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B -c " + shlex.quote(code), timeout=180)}


def diagnose_r3d(access: Access) -> dict:
    attempt = ROOT / "attempts/k1_rc_smoke_r3d"
    plan = attempt / "K1_RC_SMOKE_EXECUTION_PLAN_20260923_R3D.json"
    runner = attempt / "runtime/run_k1_rc_smoke.py"
    slurm = attempt / "run_k1_rc_smoke_r3d.slurm"
    repo = ROOT.parents[2] / "01_code/repository"
    sources = [plan, slurm, runner,
               ROOT / "K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json",
               ROOT / "K1_FALLBACK_CONFIG_PLAN_20260923_R3A.json",
               ROOT / "runtime/prepare_k1_fallback_config_r3a.py",
               *sorted((repo / "custom_nnunet").glob("*.py")), repo / "scripts/09_k1245/62_run_k1245_smoke.py"]
    expected = {}
    for src in sources:
        if repo in src.parents:
            dest = REMOTE_R3D / "runtime/repository" / src.relative_to(repo)
        elif src.parent == attempt:
            dest = REMOTE_R3D / src.name
        elif attempt in src.parents:
            dest = REMOTE_R3D / src.relative_to(attempt)
        else:
            dest = REMOTE_R3D / "sources" / src.name
        expected[str(dest)] = sha(src)
    code = (
        "import pathlib,hashlib,json,subprocess;root=pathlib.Path(" + repr(str(REMOTE_R3D)) + ");expected=" + repr(expected) + ";sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();"
        "actual={p:(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() if pathlib.Path(p).is_file() else None) for p in expected};"
        "plan=json.loads((root/'K1_RC_SMOKE_EXECUTION_PLAN_20260923_R3D.json').read_text());"
        "fallback=root/'K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json';cfgp=root/'config/K1245_ROUTE_K1_FALLBACK.json';"
        "cfg=json.loads(cfgp.read_text());gate=json.loads((root/'evidence/F_FALLBACK_GATE_VALIDATION.json').read_text());"
        "pool=pathlib.Path(" + repr(str(R2_REMOTE / "pools/rc27/component_manifest.json")) + ");"
        "print(json.dumps({'hash_mismatch_or_missing':{p:{'actual':actual[p],'expected':h} for p,h in expected.items() if actual[p]!=h},"
        "'config_plan_hash_ok':cfg['plan_sha256']==hashlib.sha256(fallback.read_bytes()).hexdigest(),"
        "'config_sha_ok':cfg['config_sha256']==plan['k1_config_sha256'],'remote_config_sha256':sha(root/'config/K1245_ROUTE_K1_FALLBACK.json'),'plan_config_sha256':plan['k1_config_sha256'],'gate_ok':gate['status']=='pass' and gate['F_STRICT_PREREG_GATE'] is False,"
        "'runner_sha_ok':hashlib.sha256((root/'runtime/run_k1_rc_smoke.py').read_bytes()).hexdigest()==plan['smoke_runtime_sha256'],"
        "'pool_sha_ok':hashlib.sha256(pool.read_bytes()).hexdigest()==plan['smoke']['pool_manifest_file_sha256'],"
        "'training_authorization':plan['training_authorization'],'k5_authorization':plan['k5_authorization'],"
        "'queue':subprocess.run(['squeue','-h','-n','s2k1rcsmkr3d','-o','%i|%T'],capture_output=True,text=True).stdout.strip()},sort_keys=True))"
    )
    return {"diagnostic": access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B -c " + shlex.quote(code), timeout=180)}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare_validate", "configure_k1", "deploy_smoke_r3b", "deploy_smoke_r3c", "deploy_smoke_r3d", "deploy_smoke_r3e", "deploy_smoke_r3f", "deploy_smoke_r3g", "deploy_k4_small_r2", "deploy_k4_small_r3", "diagnose_k4_small_r2", "inspect_k4_small_r2", "inspect_k4_small_r3", "inspect_r3", "inspect_r3d", "inspect_r3e", "inspect_r3f", "inspect_r3g", "diagnose_r3d", "compare_r3d"))
    args = parser.parse_args()
    access = Access(args.action)
    started = datetime.now(timezone.utc).isoformat()
    result = None
    try:
        access.start_vpn()
        result = globals()[args.action](access)
    finally:
        access.close_vpn()
    exclusive(access.audit / "LOCAL_OPERATION_RECEIPT.json", {
        **META, "action": args.action, "local_started_utc": started,
        "local_completed_utc": datetime.now(timezone.utc).isoformat(), "vpn_closed": access.owns_vpn,
        "preexisting_vpn_left_unchanged": not access.owns_vpn, "result": result,
    })
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
