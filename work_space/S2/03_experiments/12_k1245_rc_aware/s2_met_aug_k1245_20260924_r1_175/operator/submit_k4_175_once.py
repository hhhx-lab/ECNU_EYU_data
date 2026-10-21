"""Deploy and submit exactly one independent K4 175-epoch training job."""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
FALLBACK_OPERATOR = HERE.parent / "s2_met_aug_k1245_20260923_r3_fallback/operator"
sys.path.insert(0, str(FALLBACK_OPERATOR))
import access as audited_access  # noqa: E402

REMOTE = Path("/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1245_20260924_r2_175")
SOURCE = Path("/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1245_20260923_r3_fallback/attempts/k4_small_smoke_r5")
PYTHON = "/public/home/zqchen/.conda/envs/segmamba/bin/python"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def q(path: Path | str) -> str:
    return shlex.quote(str(path))


def main() -> None:
    plan = HERE / "K1245_RC_AWARE_175_EXECUTION_PLAN_20260924_R1.json"
    auth_json = HERE / "evidence/K4_175_TRAINING_AUTHORIZATION.json"
    auth_md = HERE / "evidence/K4_175_TRAINING_AUTHORIZATION_20260924.md"
    slurm = HERE / "slurm/01_train_k4_175.slurm"
    trainer = HERE / "runtime/nnUNetTrainerBraTS2026RCMetAugK1245FocalCompletionFineTune175.py"
    local_evidence = HERE / "evidence"
    access = audited_access.Access("submit_k4_175")
    local_started = dt.datetime.now(dt.timezone.utc).isoformat()
    result: dict[str, object] = {}
    try:
        access.start_vpn()
        remote = q(REMOTE)
        source = q(SOURCE)
        # One read-only remote preflight before creating the new attempt.
        readonly = (
            "set -euo pipefail; "
            f"if test -e {remote}; then test ! -e {remote}/evidence/K4_175_TRAINING_LAUNCH.json; test ! -e {remote}/evidence/K4_175_TRAINING_STARTED.json; fi; "
            f"test -d {source}/runtime/repository; "
            f"test -f {source}/config/K1245_ROUTE_K4.json; "
            f"test -f {source}/pools/small8/component_manifest.json; "
            f"test -f {source}/runtime/repository/custom_nnunet/nnUNetTrainerBraTS2026RCMetAugK1245FocalCompletionFineTune.py; "
            "test -z \"$(squeue -h -n s2k4train175r1 -o %i)\"; "
            f"{PYTHON} -B -c " + q(
                "import json,hashlib,pathlib; "
                f"r=pathlib.Path({str(SOURCE)!r}); "
                "m=json.loads((r/'pools/small8/component_manifest.json').read_text()); "
                "assert m['component_count'] >= 24; "
                "print(json.dumps({'source_runtime':True,'small8_count':m['component_count'],'trainer_sha256':hashlib.sha256((r/'runtime/repository/custom_nnunet/nnUNetTrainerBraTS2026RCMetAugK1245FocalCompletionFineTune.py').read_bytes()).hexdigest()},sort_keys=True))"
            )
        )
        result["remote_readonly_preflight"] = access.ssh(readonly, timeout=240).strip()

        access.ssh(f"set -euo pipefail; mkdir -m 700 -p {remote}/{{evidence,logs,slurm,runtime,results}}")
        for source_path, destination in (
            (plan, str(REMOTE) + "/"),
            (auth_json, str(REMOTE / "evidence") + "/"),
            (auth_md, str(REMOTE / "evidence") + "/"),
            (slurm, str(REMOTE / "slurm") + "/"),
        ):
            access.scp(source_path, destination)

        prepare = (
            "set -euo pipefail; "
            f"src={source}; dst={remote}; "
            "cp -a \"$src/runtime/repository\" \"$dst/runtime/\"; "
            "cp -a \"$src/config\" \"$dst/\"; cp -a \"$src/pools\" \"$dst/\"; chmod -R u+rwX \"$dst/config\" \"$dst/pools\" \"$dst/evidence\"; "
            "mkdir -p \"$dst/evidence\"; cp -p \"$src/evidence/REMOTE_STATIC_PREFLIGHT.json\" \"$dst/evidence/\"; printf 'complete\\n' > \"$dst/evidence/POOL_BUILD_COMPLETE.ok\"; "
            f"{PYTHON} -B - <<'PY'\n"
            "import hashlib,json,sys\n"
            "from pathlib import Path\n"
            f"root=Path({str(REMOTE)!r}); repo=root/'runtime/repository'; sys.path.insert(0,str(repo))\n"
            "from custom_nnunet.met_aug_core import canonical_json_sha256\n"
            "plan=root/'K1245_RC_AWARE_175_EXECUTION_PLAN_20260924_R1.json'\n"
            "cfgp=root/'config/K1245_ROUTE_K4.json'; cfg=json.loads(cfgp.read_text())\n"
            "cfg['plan_sha256']=hashlib.sha256(plan.read_bytes()).hexdigest(); cfg['config_sha256']=canonical_json_sha256(cfg,exclude=('config_sha256',)); cfgp.write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\\n')\n"
            "manifest=json.loads((root/'pools/small8/component_manifest.json').read_text())\n"
            "authp=root/'evidence/K4_175_TRAINING_AUTHORIZATION.json'; auth=json.loads(authp.read_text())\n"
            "auth.update({'plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest(),'manifest_sha256':manifest['manifest_sha256'],'config_sha256':hashlib.sha256(cfgp.read_bytes()).hexdigest()}); authp.write_text(json.dumps(auth,indent=2,sort_keys=True)+'\\n')\n"
            "PY"
        )
        access.ssh(prepare, timeout=300)
        access.ssh(f"set -euo pipefail; mkdir -p {q(REMOTE / 'runtime/repository/custom_nnunet')}")
        access.scp(trainer, str(REMOTE / "runtime/repository/custom_nnunet") + "/")

        static_and_submit = (
            "set -euo pipefail; "
            f"root={remote}; "
            f"{PYTHON} -B - <<'PY'\n"
            "import datetime as dt,hashlib,importlib,importlib.util,json,os,py_compile,subprocess,sys\n"
            "from pathlib import Path\n"
            f"root=Path({str(REMOTE)!r}); repo=root/'runtime/repository'; sys.path.insert(0,str(repo))\n"
            "trainer='nnUNetTrainerBraTS2026RCMetAugK1245FocalCompletionFineTune175'\n"
            "trainer_path=repo/'custom_nnunet'/f'{trainer}.py'; plan=root/'K1245_RC_AWARE_175_EXECUTION_PLAN_20260924_R1.json'; cfg=root/'config/K1245_ROUTE_K4.json'; man=root/'pools/small8/component_manifest.json'; auth=root/'evidence/K4_175_TRAINING_AUTHORIZATION.json'; slurm=root/'slurm/01_train_k4_175.slurm'\n"
            "assert json.loads(cfg.read_text())['route_id']=='K4'; assert json.loads(man.read_text())['component_count']>=24\n"
            "assert json.loads(auth.read_text())['status']=='pass' and json.loads(auth.read_text())['plan_sha256']==hashlib.sha256(plan.read_bytes()).hexdigest()\n"
            "assert json.loads(auth.read_text())['manifest_sha256']==json.loads(man.read_text())['manifest_sha256']; assert json.loads(cfg.read_text())['plan_sha256']==hashlib.sha256(plan.read_bytes()).hexdigest()\n"
            "for path in sorted((repo/'custom_nnunet').glob('*.py')): py_compile.compile(str(path),doraise=True)\n"
            "mod=importlib.import_module('custom_nnunet.'+trainer); assert getattr(mod,trainer).__name__==trainer\n"
            "assert 'K1245 revision requires epochs=175' in trainer_path.read_text() and 'MET_AUG_K1245_175_RUNTIME_PASS' in trainer_path.read_text()\n"
            "subprocess.run(['bash','-n',str(slurm)],check=True)\n"
            "pre={'artifact_status':'experimental_unvalidated','operator_approved':False,'formal_gate_status':'not_run_not_passed','status':'pass','route_id':'K4','job_name':'s2k4train175r1','epochs':175,'checkpoint_epochs':[25,50,75,100,125,150,175],'plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest(),'config_sha256':hashlib.sha256(cfg.read_bytes()).hexdigest(),'manifest_sha256':json.loads(man.read_text())['manifest_sha256'],'trainer_sha256':hashlib.sha256(trainer_path.read_bytes()).hexdigest(),'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00','Z')}\n"
            "e=root/'evidence'; fd=os.open(e/'K4_175_TRAINING_STATIC_PREFLIGHT.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400); f=os.fdopen(fd,'w'); json.dump(pre,f,indent=2,sort_keys=True); f.write('\\n'); f.close()\n"
            "assert not subprocess.run(['squeue','-h','-n','s2k4train175r1','-o','%i'],capture_output=True,text=True,check=True).stdout.strip()\n"
            "res=subprocess.run(['sbatch','--hold','--parsable',str(slurm)],capture_output=True,text=True,check=True); jid=res.stdout.strip().split(';')[0]; assert jid.isdigit()\n"
            "for name,payload in [('K4_175_TRAINING_LAUNCH.json',{**pre,'status':'submitted_held','job_id':jid}),('K4_175_TRAINING_RELEASED.json',{**pre,'status':'released','job_id':jid})]:\n"
            " fd=os.open(e/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400); f=os.fdopen(fd,'w'); json.dump(payload,f,indent=2,sort_keys=True); f.write('\\n'); f.close()\n"
            "subprocess.run(['scontrol','release',jid],check=True); print(json.dumps({'job_id':jid,'job_name':'s2k4train175r1','status':'released'},sort_keys=True))\n"
            "PY"
        )
        result["launch"] = access.ssh(static_and_submit, timeout=360).strip()
        jid = json.loads(result["launch"].splitlines()[-1])["job_id"]
        snapshot = access.ssh(
            "set -e; "
            f"{PYTHON} -B -c " + q(
                "import datetime as dt,json,os,subprocess; "
                f"jid={jid!r}; "
                "q=subprocess.run(['squeue','-h','-j',jid,'-o','%i|%T|%M|%R'],capture_output=True,text=True); "
                "a=subprocess.run(['sacct','-n','-X','-j',jid,'-o','JobIDRaw,JobName,State,ExitCode,Submit,Start,End'],capture_output=True,text=True); "
                "print(json.dumps({'remote_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'hostname':os.uname().nodename,'job_id':jid,'squeue':q.stdout.strip(),'sacct':a.stdout.strip(),'squeue_rc':q.returncode,'sacct_rc':a.returncode},sort_keys=True))"
            ),
            timeout=180,
        ).strip()
        result["post_submit_snapshot"] = snapshot
        local_evidence.mkdir(parents=True, exist_ok=True)
        receipt = {"artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed","local_started_utc":local_started,"local_completed_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"vpn_closed":False,"result":result}
        (local_evidence / "K4_175_SUBMISSION_RECEIPT.json").write_text(json.dumps(receipt,ensure_ascii=True,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    finally:
        access.close_vpn()
        receipt_path = local_evidence / "K4_175_SUBMISSION_RECEIPT.json"
        if receipt_path.exists():
            data = json.loads(receipt_path.read_text(encoding="utf-8")); data["vpn_closed"] = True; receipt_path.write_text(json.dumps(data,ensure_ascii=True,indent=2,sort_keys=True)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
