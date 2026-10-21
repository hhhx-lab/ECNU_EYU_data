"""Submit the independent K4 calibration-recovery attempt once."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "s2_met_aug_k1245_20260923_r3_fallback/operator"))
from access import Access, exclusive  # noqa: E402

REMOTE = Path("/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1245_20260925_r3_175")
SOURCE = Path("/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1245_20260923_r3_fallback/attempts/k4_small_smoke_r5")
PARENT = Path("/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1")
PYTHON = "/public/home/zqchen/.conda/envs/segmamba/bin/python"
CALIBRATION = PARENT / "calibration_rebuild_20260819_r10/FIX_V2_CALIBRATION_ECNU_FROZEN.json"
CALIBRATION_SHA = "68a5d8a6c501804b7a47d626e43a58ff71b781154c0cc955a9c5a5d5b786d37f"
TRAINER = "nnUNetTrainerBraTS2026RCMetAugK1245FocalCompletionFineTune175"
META = {"artifact_status": "experimental_unvalidated", "operator_approved": False,
        "formal_gate_status": "not_run_not_passed", "F_STRICT_PREREG_GATE": False}


def q(value: object) -> str:
    return shlex.quote(str(value))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remote_python(access: Access, code: str) -> dict:
    output = access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + PYTHON + " -B -c " + q(code), timeout=360)
    return json.loads(output.strip().splitlines()[-1])


def main() -> None:
    access = Access("submit_k4_175_r3")
    plan = HERE / "K1245_RC_AWARE_175_EXECUTION_PLAN_20260925_R3.json"
    auth = HERE / "evidence/K4_175_TRAINING_AUTHORIZATION.json"
    slurm = HERE / "slurm/01_train_k4_175_r3.slurm"
    trainer = HERE / "runtime" / f"{TRAINER}.py"
    receipt = HERE / "evidence/K4_175_R3_SUBMISSION_RECEIPT.json"
    result = {**META, "local_started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "remote_root": str(REMOTE), "prior_failed_job_id": "3741523"}
    for path in (plan, auth, slurm, trainer):
        assert path.is_file(), path
    assert not receipt.exists()
    try:
        access.start_vpn()
        readonly = remote_python(access, f"""
import hashlib,json,pathlib,subprocess
r=pathlib.Path({str(REMOTE)!r}); s=pathlib.Path({str(SOURCE)!r}); c=pathlib.Path({str(CALIBRATION)!r})
assert not r.exists(), 'new attempt root already exists'
assert s.is_dir() and (s/'runtime/repository').is_dir()
assert (s/'config/K1245_ROUTE_K4.json').is_file()
assert (s/'pools/small8/component_manifest.json').is_file()
assert hashlib.sha256(c.read_bytes()).hexdigest()=={CALIBRATION_SHA!r}
assert json.loads((s/'config/K1245_ROUTE_K4.json').read_text())['legacy_fix_v3_calibration_sha256']=={CALIBRATION_SHA!r}
assert not subprocess.run(['squeue','-h','-n','s2k4train175r3','-o','%i'],capture_output=True,text=True,check=True).stdout.strip()
print(json.dumps({{'status':'pass','calibration_sha256':{CALIBRATION_SHA!r}}}))
""")
        result["readonly_preflight"] = readonly
        access.ssh("set -euo pipefail; mkdir -m 700 " + q(REMOTE) + "; mkdir -m 700 " +
                   " ".join(q(REMOTE / name) for name in ("evidence", "logs", "slurm", "runtime", "results")))
        for local, destination in ((plan, REMOTE), (auth, REMOTE / "evidence"),
                                   (slurm, REMOTE / "slurm")):
            access.scp_immutable(local, str(destination))
        access.ssh("set -euo pipefail; src=" + q(SOURCE) + "; dst=" + q(REMOTE) + "; "
                   "cp -a \"$src/runtime/repository\" \"$dst/runtime/\"; "
                   "cp -a \"$src/config\" \"$dst/\"; cp -a \"$src/pools\" \"$dst/\"; "
                   "cp -p \"$src/evidence/REMOTE_STATIC_PREFLIGHT.json\" \"$dst/evidence/\"; "
                   "chmod -R u+rwX \"$dst/config\" \"$dst/pools\" \"$dst/evidence\"; "
                   "printf 'complete\\n' > \"$dst/evidence/POOL_BUILD_COMPLETE.ok\"")
        access.scp_immutable(trainer, str(REMOTE / "runtime/repository/custom_nnunet"))
        launched = remote_python(access, f"""
import datetime as dt,hashlib,importlib,json,os,pathlib,subprocess,sys
r=pathlib.Path({str(REMOTE)!r}); repo=r/'runtime/repository'; sys.path.insert(0,str(repo))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=r/{plan.name!r}; cfg=r/'config/K1245_ROUTE_K4.json'; man=r/'pools/small8/component_manifest.json'
auth=r/'evidence/K4_175_TRAINING_AUTHORIZATION.json'; slurm=r/'slurm/{slurm.name}'
trainer=repo/'custom_nnunet/{trainer.name}'; cal=pathlib.Path({str(CALIBRATION)!r})
assert sha(plan)=={sha(plan)!r} and sha(slurm)=={sha(slurm)!r} and sha(trainer)=={sha(trainer)!r}
assert sha(cal)=={CALIBRATION_SHA!r}
from custom_nnunet.met_aug_core import canonical_json_sha256
x=json.loads(cfg.read_text()); assert x['route_id']=='K4' and x['legacy_fix_v3_calibration_sha256']==sha(cal)
x['plan_sha256']=sha(plan); x['config_sha256']=canonical_json_sha256(x,exclude=('config_sha256',))
cfg.write_text(json.dumps(x,indent=2,sort_keys=True)+'\\n')
m=json.loads(man.read_text()); assert m['component_count']>=24
a=json.loads(auth.read_text()); assert a['route_id']=='K4' and a['status']=='pass'
a.update({{'plan_sha256':sha(plan),'config_sha256':sha(cfg),'manifest_sha256':m['manifest_sha256']}})
auth.write_text(json.dumps(a,indent=2,sort_keys=True)+'\\n')
for p in (repo/'custom_nnunet').glob('*.py'): compile(p.read_bytes(),str(p),'exec')
mod=importlib.import_module('custom_nnunet.{TRAINER}'); assert getattr(mod,{TRAINER!r}).__name__=={TRAINER!r}
subprocess.run(['bash','-n',str(slurm)],check=True)
assert 'export S2_MET_AUG_FIX_V3_CALIBRATION=' in slurm.read_text()
assert '{TRAINER}__nnUNetPlans__3d_fullres/fold_0' in slurm.read_text()
assert not subprocess.run(['squeue','-h','-n','s2k4train175r3','-o','%i'],capture_output=True,text=True,check=True).stdout.strip()
e=r/'evidence'; stamp=dt.datetime.now(dt.timezone.utc).isoformat()
pre={{'artifact_status':'experimental_unvalidated','operator_approved':False,'formal_gate_status':'not_run_not_passed','F_STRICT_PREREG_GATE':False,'status':'pass','job_name':'s2k4train175r3','epochs':175,'plan_sha256':sha(plan),'config_sha256':sha(cfg),'manifest_sha256':m['manifest_sha256'],'trainer_sha256':sha(trainer),'slurm_sha256':sha(slurm),'calibration_path':str(cal),'calibration_sha256':sha(cal),'remote_utc':stamp}}
def put(name,payload):
 fd=os.open(e/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
 with os.fdopen(fd,'w') as h: json.dump(payload,h,indent=2,sort_keys=True); h.write('\\n')
put('K4_175_R3_STATIC_PREFLIGHT.json',pre)
put('K4_175_R3_RESERVATION.json',{{**pre,'status':'reserved'}})
job=subprocess.run(['sbatch','--hold','--parsable',str(slurm)],capture_output=True,text=True,check=True).stdout.strip().split(';')[0]
assert job.isdigit()
put('K4_175_R3_LAUNCH.json',{{**pre,'status':'submitted_held','job_id':job}})
subprocess.run(['scontrol','release',job],check=True)
put('K4_175_R3_RELEASED.json',{{**pre,'status':'released','job_id':job}})
print(json.dumps({{'status':'released','job_id':job,'job_name':'s2k4train175r3','remote_utc':stamp}}))
""")
        result["launch"] = launched
        exclusive(receipt, {**result, "local_completed_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        print(json.dumps(launched, sort_keys=True))
    finally:
        access.close_vpn()


if __name__ == "__main__":
    main()
