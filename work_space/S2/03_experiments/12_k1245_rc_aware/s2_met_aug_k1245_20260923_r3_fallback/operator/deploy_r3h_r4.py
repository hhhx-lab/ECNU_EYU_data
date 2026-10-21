from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import py_compile
import shlex
import subprocess
from pathlib import Path

ACCESS_PATH = Path(__file__).with_name("access.py")
SPEC = importlib.util.spec_from_file_location("k1245_access", ACCESS_PATH)
ACCESS = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(ACCESS)

ROOT = ACCESS.ROOT
REPO = ROOT.parents[2] / "01_code/repository"
REMOTE_BASE = ACCESS.REMOTE
R2_REMOTE = ACCESS.R2_REMOTE
PARENT_REMOTE = ACCESS.PARENT_REMOTE
PYTHON = ACCESS.PYTHON
META = ACCESS.META


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_emit_code(root: Path, job_name: str, plan_name: str, slurm_name: str,
                     runner_rel: str, preflight_name: str, reservation_name: str,
                     launch_name: str, released_name: str, required: list[str],
                     stage_pairs: list[str], stage_source: Path, extra_asserts: str) -> str:
    required_repr = repr(required)
    stage_repr = repr(stage_pairs)
    return (
        "import datetime as dt,hashlib,json,os,pathlib,py_compile,subprocess,sys;"
        f"root=pathlib.Path({str(root)!r});src=pathlib.Path({str(stage_source)!r});"
        f"sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();"
        f"required={required_repr};actual={{p:sha(root/p) for p in required}};assert all((root/p).is_file() for p in required),actual;"
        f"plan=json.loads((root/{plan_name!r}).read_text());assert plan['job_name']=={job_name!r};"
        f"{extra_asserts}"
        "import sys,importlib.util;sys.path.insert(0,str(root/'runtime/repository'));"
        "helper=root/'runtime/repository/scripts/09_k1245/62_run_k1245_smoke.py';hs=importlib.util.spec_from_file_location('k1245_smoke_shared',helper);hm=importlib.util.module_from_spec(hs);assert hs.loader is not None;hs.loader.exec_module(hm);"
        "import custom_nnunet.met_aug_k1245;"
        f"[py_compile.compile(str(root/p),doraise=True) for p in required if p.endswith('.py')];"
        f"q=subprocess.run(['squeue','-h','-n',{job_name!r},'-o','%i'],capture_output=True,text=True,check=True);assert not q.stdout.strip(),q.stdout;"
        f"subprocess.run(['bash','-n',str(root/{slurm_name!r})],check=True);"
        f"out={{**{META!r},'status':'pass','timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00','Z'),'hostname':os.uname().nodename,'job_name':{job_name!r},'plan_sha256':sha(root/{plan_name!r}),'runner_sha256':sha(root/{runner_rel!r}),'files_sha256':actual}};"
        f"e=root/'evidence';emit=lambda n,x:(lambda fd:(lambda f:(json.dump(x,f,sort_keys=True,indent=2),f.write('\\n'),f.close()))(os.fdopen(fd,'w')))(os.open(e/n,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400));"
        f"emit({preflight_name!r},out);emit({reservation_name!r},{{**out,'status':'reserved'}});"
        f"res=subprocess.run(['sbatch','--hold','--parsable',str(root/{slurm_name!r})],capture_output=True,text=True);"
        "msg=(res.stdout+' '+res.stderr).strip();emit('QOS_BLOCKED.json',{**out,'status':'qos_blocked','message':msg}) if res.returncode and 'QOSMaxSubmitJobPerUserLimit' in msg else None;assert res.returncode==0,msg;"
        "jid=res.stdout.strip().split(';')[0];assert jid.isdigit();"
        f"emit({launch_name!r},{{**out,'status':'submitted_held','job_id':jid,'slurm_sha256':sha(root/{slurm_name!r})}});"
        "subprocess.run(['scontrol','release',jid],check=True);"
        f"emit({released_name!r},{{**out,'status':'released','job_id':jid}});print(json.dumps({{'job_id':jid,'job_name':{job_name!r},'status':'released'}},sort_keys=True))"
    )


def deploy_k1(access: ACCESS.Access) -> dict:
    attempt = ROOT / "attempts/k1_rc_smoke_r3i"
    remote = REMOTE_BASE / "attempts/k1_rc_smoke_r3i"
    plan = attempt / "K1_RC_SMOKE_EXECUTION_PLAN_20260923_R3I.json"
    runner = attempt / "runtime/run_k1_rc_smoke.py"
    slurm = attempt / "run_k1_rc_smoke_r3i.slurm"
    prereg = ROOT.parents[2] / "05_documents/02_plans/20260921_k1245_rc_aware_execution/F_FALLBACK_GATE_PREREGISTRATION_20260923_R3.md"
    files = [plan, slurm, runner, prereg,
             ROOT / "K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json",
             ROOT / "K1_FALLBACK_CONFIG_PLAN_20260923_R3A.json",
             ROOT / "runtime/prepare_k1_fallback_config_r3a.py",
             REPO / "scripts/09_k1245/62_run_k1245_smoke.py",
             *sorted((REPO / "custom_nnunet").glob("*.py"))]
    access.ssh("set -euo pipefail; if test -e " + str(remote) + "; then test -f " + str(remote / "evidence/QOS_BLOCKED.json") + "; test ! -e " + str(remote / "evidence/K1_SMOKE_R3I_REVALIDATED_LAUNCH.json") + "; test ! -e " + str(remote / "evidence/K1_SMOKE_R3I_REVALIDATED_RELEASED.json") + "; else mkdir -m 700 " + str(remote) + "; fi; mkdir -m 700 -p " + " ".join(str(remote / p) for p in ("evidence", "config", "runtime", "runtime/repository/custom_nnunet", "runtime/repository/scripts/09_k1245", "sources", "logs", "smoke")))
    for src in files:
        if REPO in src.parents:
            dest = remote / "runtime/repository" / src.relative_to(REPO)
        elif src == prereg:
            dest = remote / "sources" / src.name
        elif src == ROOT / "runtime/prepare_k1_fallback_config_r3a.py":
            dest = remote / "runtime" / src.name
        elif attempt in src.parents:
            dest = remote / src.relative_to(attempt)
        else:
            dest = remote / src.name
        access.scp_immutable(src, str(dest.parent))
    access.ssh("set -euo pipefail; src=" + str(REMOTE_BASE) + "; dst=" + str(remote) + "; for pair in K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json config/K1245_ROUTE_K1_FALLBACK.json evidence/F_FALLBACK_GATE_VALIDATION.json evidence/K1_FALLBACK_CONFIG_PREPARATION.json; do mkdir -p \"$dst/$(dirname \"$pair\")\"; if test -e \"$dst/$pair\"; then cmp -s \"$src/$pair\" \"$dst/$pair\"; else cp -p \"$src/$pair\" \"$dst/$pair\"; fi; done")
    custom_required = ["runtime/repository/custom_nnunet/" + p.name for p in sorted((REPO / "custom_nnunet").glob("*.py"))]
    required = [p.relative_to(attempt).as_posix() for p in (plan, slurm, runner)] + ["runtime/repository/scripts/09_k1245/62_run_k1245_smoke.py"] + custom_required
    extra = "assert plan['strict_gate_result']['result'] is False and plan['training_authorization'] is False and plan['k5_authorization'] is False;assert plan['smoke_runtime_sha256']==sha(root/'runtime/run_k1_rc_smoke.py');assert 'r3i' in (root/'run_k1_rc_smoke_r3i.slurm').read_text();assert (root/'runtime/repository/scripts/09_k1245/62_run_k1245_smoke.py').is_file();"
    code = remote_emit_code(remote, "s2k1rcsmkr3i", plan.name, slurm.name, "runtime/run_k1_rc_smoke.py", "K1_SMOKE_R3I_REVALIDATED_STATIC_PREFLIGHT.json", "K1_SMOKE_R3I_REVALIDATED_RESERVATION.json", "K1_SMOKE_R3I_REVALIDATED_LAUNCH.json", "K1_SMOKE_R3I_REVALIDATED_RELEASED.json", required, [], REMOTE_BASE, extra)
    return {"route": "K1", "remote": access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B -c " + shlex.quote(code), timeout=360)[-3000:]}


def deploy_k4(access: ACCESS.Access) -> dict:
    attempt = ROOT / "attempts/k4_small_smoke_r5"
    remote = REMOTE_BASE / "attempts/k4_small_smoke_r5"
    plan = attempt / "K4_SMALL_SMOKE_EXECUTION_PLAN_20260923_R5.json"
    runner = attempt / "runtime/run_k4_small_smoke.py"
    slurm = attempt / "run_k4_small_smoke_r5.slurm"
    parent_plan = ROOT.parent / "s2_met_aug_k1245_20260923_r2/K1245_RC_AWARE_EXECUTION_PLAN_20260923_R2.json"
    files = [plan, runner, slurm, parent_plan,
             REPO / "scripts/09_k1245/62_run_k1245_smoke.py",
             *sorted((REPO / "custom_nnunet").glob("*.py"))]
    access.ssh("set -euo pipefail; test ! -e " + str(remote) + "; mkdir -m 700 -p " + " ".join(str(remote / p) for p in ("evidence", "runtime", "runtime/repository/custom_nnunet", "runtime/repository/scripts/09_k1245", "sources", "logs", "smoke")))
    for src in files:
        if src == parent_plan:
            dest = remote / "sources" / src.name
        elif REPO in src.parents:
            dest = remote / "runtime/repository" / src.relative_to(REPO)
        elif attempt in src.parents:
            dest = remote / src.relative_to(attempt)
        else:
            dest = remote / src.name
        access.scp_immutable(src, str(dest.parent))
    access.ssh("set -euo pipefail; src=" + str(R2_REMOTE) + "; dst=" + str(remote) + "; for pair in config/K1245_ROUTE_K4.json pools/small8/component_manifest.json evidence/REMOTE_STATIC_PREFLIGHT.json evidence/POOL_BUILD_COMPLETE.ok; do mkdir -p \"$dst/$(dirname \"$pair\")\"; if test -e \"$dst/$pair\"; then cmp -s \"$src/$pair\" \"$dst/$pair\"; else cp -p \"$src/$pair\" \"$dst/$pair\"; fi; done")
    custom_required = ["runtime/repository/custom_nnunet/" + p.name for p in sorted((REPO / "custom_nnunet").glob("*.py"))]
    required = [p.relative_to(attempt).as_posix() for p in (plan, slurm, runner)] + ["runtime/repository/scripts/09_k1245/62_run_k1245_smoke.py"] + custom_required
    extra = "assert plan['route_id']=='K4' and plan['fallback_context']['F_STRICT_PREREG_GATE'] is False and plan['training_authorization'] is False and plan['k5_authorization'] is False;assert plan['smoke_runtime_sha256']==sha(root/'runtime/run_k4_small_smoke.py');assert 'r5' in (root/'run_k4_small_smoke_r5.slurm').read_text();assert 'render_without_contour' in (root/'runtime/run_k4_small_smoke.py').read_text();assert '.contour(' not in (root/'runtime/run_k4_small_smoke.py').read_text();"
    code = remote_emit_code(remote, "s2k4smallsmkr5", plan.name, slurm.name, "runtime/run_k4_small_smoke.py", "K4_SMALL_SMOKE_R5_STATIC_PREFLIGHT.json", "K4_SMALL_SMOKE_R5_RESERVATION.json", "K4_SMALL_SMOKE_R5_LAUNCH.json", "K4_SMALL_SMOKE_R5_RELEASED.json", required, [], R2_REMOTE, extra)
    return {"route": "K4", "remote": access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B -c " + shlex.quote(code), timeout=360)[-3000:]}


def deploy_k1_r3j(access: ACCESS.Access) -> dict:
    attempt = ROOT / "attempts/k1_rc_smoke_r3j"
    remote = REMOTE_BASE / "attempts/k1_rc_smoke_r3j"
    plan = attempt / "K1_RC_SMOKE_EXECUTION_PLAN_20260924_R3J.json"
    runner = attempt / "runtime/run_k1_rc_smoke.py"
    slurm = attempt / "run_k1_rc_smoke_r3j.slurm"
    prereg = ROOT.parents[2] / "05_documents/02_plans/20260921_k1245_rc_aware_execution/F_FALLBACK_GATE_PREREGISTRATION_20260923_R3.md"
    files = [plan, slurm, runner, prereg,
             ROOT / "K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json",
             ROOT / "K1_FALLBACK_CONFIG_PLAN_20260923_R3A.json",
             ROOT / "runtime/prepare_k1_fallback_config_r3a.py",
             REPO / "scripts/09_k1245/62_run_k1245_smoke.py",
             *sorted((REPO / "custom_nnunet").glob("*.py"))]
    access.ssh("set -euo pipefail; test ! -e " + str(remote) + "; mkdir -m 700 -p " + " ".join(str(remote / p) for p in ("evidence", "config", "runtime", "runtime/repository/custom_nnunet", "runtime/repository/scripts/09_k1245", "sources", "logs", "smoke")))
    for src in files:
        if REPO in src.parents:
            dest = remote / "runtime/repository" / src.relative_to(REPO)
        elif src == prereg:
            dest = remote / "sources" / src.name
        elif src == ROOT / "runtime/prepare_k1_fallback_config_r3a.py":
            dest = remote / "runtime" / src.name
        elif attempt in src.parents:
            dest = remote / src.relative_to(attempt)
        else:
            dest = remote / src.name
        access.scp_immutable(src, str(dest.parent))
    access.ssh("set -euo pipefail; src=" + str(REMOTE_BASE) + "; dst=" + str(remote) + "; for pair in K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json config/K1245_ROUTE_K1_FALLBACK.json evidence/F_FALLBACK_GATE_VALIDATION.json evidence/K1_FALLBACK_CONFIG_PREPARATION.json; do mkdir -p \"$dst/$(dirname \"$pair\")\"; if test -e \"$dst/$pair\"; then cmp -s \"$src/$pair\" \"$dst/$pair\"; else cp -p \"$src/$pair\" \"$dst/$pair\"; fi; done")
    custom_required = ["runtime/repository/custom_nnunet/" + p.name for p in sorted((REPO / "custom_nnunet").glob("*.py"))]
    required = [p.relative_to(attempt).as_posix() for p in (plan, slurm, runner)] + ["runtime/repository/scripts/09_k1245/62_run_k1245_smoke.py"] + custom_required
    extra = "assert plan['job_name']=='s2k1rcsmkr3j' and plan['strict_gate_result']['result'] is False and plan['training_authorization'] is False and plan['k5_authorization'] is False;assert plan['smoke_runtime_sha256']==sha(root/'runtime/run_k1_rc_smoke.py');assert 'r3j' in (root/'run_k1_rc_smoke_r3j.slurm').read_text();assert '.contour(' not in (root/'runtime/run_k1_rc_smoke.py').read_text();"
    code = remote_emit_code(remote, "s2k1rcsmkr3j", plan.name, slurm.name, "runtime/run_k1_rc_smoke.py", "K1_SMOKE_R3J_STATIC_PREFLIGHT.json", "K1_SMOKE_R3J_RESERVATION.json", "K1_SMOKE_R3J_LAUNCH.json", "K1_SMOKE_R3J_RELEASED.json", required, [], REMOTE_BASE, extra)
    return {"route": "K1_R3J", "remote": access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B -c " + shlex.quote(code), timeout=360)[-3000:]}


def main() -> None:
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    result = {}
    access = ACCESS.Access("deploy_k1_r3j")
    try:
        access.start_vpn()
        result["K1_R3J"] = deploy_k1_r3j(access)
    except Exception as exc:
        result["K1_R3J"] = {"status": "failed", "error": str(exc), "audit": str(access.audit)}
    finally:
        access.close_vpn()
    ACCESS.exclusive(access.audit / "LOCAL_OPERATION_RECEIPT.json", {
        **META, "action": "deploy_k1_r3j", "local_started_utc": started,
        "local_completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "vpn_closed": access.owns_vpn, "result": result,
    })
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
