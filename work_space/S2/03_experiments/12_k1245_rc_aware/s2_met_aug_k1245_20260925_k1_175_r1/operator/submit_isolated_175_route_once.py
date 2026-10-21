"""One-shot, fail-closed submission for the isolated K1/K2/K4 175 routes."""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "s2_met_aug_k1245_20260923_r3_fallback/operator"))
from access import Access, exclusive  # noqa: E402

REMOTE_BASE = Path("/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated")
PARENT = REMOTE_BASE / "s2_met_aug_fix_v3_full200_20260810_r1"
R2 = REMOTE_BASE / "s2_met_aug_k1245_20260923_r2"
FALLBACK = REMOTE_BASE / "s2_met_aug_k1245_20260923_r3_fallback/attempts"
PYTHON = "/public/home/zqchen/.conda/envs/segmamba/bin/python"
TRAINER = "nnUNetTrainerBraTS2026RCMetAugK1245FocalCompletionFineTune175.py"
CALIBRATION = PARENT / "calibration_rebuild_20260819_r10/FIX_V2_CALIBRATION_ECNU_FROZEN.json"
ORIGINAL_E = PARENT / "frozen/original_e/checkpoint_final.pth"
RAW = Path("/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/data/ecnu_completion_emergency/nnUNet_raw")
META = {"artifact_status": "experimental_unvalidated", "operator_approved": False,
        "formal_gate_status": "not_run_not_passed", "F_STRICT_PREREG_GATE": False}
ROUTES = {
    "K1": {
        "root": "s2_met_aug_k1245_20260929_k1_175_r3",
        "plan": "K1245_K1_175_EXECUTION_PLAN_20260929_R3.json",
        "slurm": "01_train_k1_175_r3.slurm",
        "job_name": "s2k1train175r3",
        "source": FALLBACK / "k1_rc_smoke_r3j",
        "config": FALLBACK / "k1_rc_smoke_r3j/config/K1245_ROUTE_K1_FALLBACK.json",
        "config_sha256": "9d96ee03058d49fe7598169f1f80edb4db1d70945914405e36250f3dd7ecd188",
        "pool": R2 / "pools/rc27",
        "manifest_identity": "a9b74b410838f8fcba50d66d52c7751bef2b07ddc94f2f774eabaed8c98701ac",
        "static": R2 / "evidence/REMOTE_STATIC_PREFLIGHT.json",
        "auto": FALLBACK / "k1_rc_smoke_r3j/smoke/k1_rc24_r3j/AUTOMATIC_VALIDATION.json",
    },
    "K2": {
        "root": "s2_met_aug_k1245_20260929_k2_175_r2",
        "plan": "K1245_K2_175_EXECUTION_PLAN_20260929_R2.json",
        "slurm": "01_train_k2_175_r2.slurm",
        "job_name": "s2k2train175r2",
        "source": FALLBACK / "k1_rc_smoke_r3j",
        "config": R2 / "config/K1245_ROUTE_K2.json",
        "config_sha256": "ab4e1001e4e2b57f64acaaf8340558a3694add6fb9f78df5c2d36bbbc7aec815",
        "pool": R2 / "pools/rc27",
        "manifest_identity": "a9b74b410838f8fcba50d66d52c7751bef2b07ddc94f2f774eabaed8c98701ac",
        "static": R2 / "evidence/REMOTE_STATIC_PREFLIGHT.json",
        "auto": FALLBACK / "k1_rc_smoke_r3j/smoke/k1_rc24_r3j/AUTOMATIC_VALIDATION.json",
    },
    "K4": {
        "root": "s2_met_aug_k1245_20260929_k4_175_r6",
        "plan": "K1245_K4_175_EXECUTION_PLAN_20260929_R6.json",
        "slurm": "01_train_k4_175_r6.slurm",
        "job_name": "s2k4train175r6",
        "source": FALLBACK / "k4_small_smoke_r5",
        "config": FALLBACK / "k4_small_smoke_r5/config/K1245_ROUTE_K4.json",
        "config_sha256": "a0d98585e51a05b07da6f92323b668a191e65882097040ef98206d0deea5eb19",
        "pool": R2 / "pools/small8",
        "manifest_identity": "956e6384d47216e41b79905ad9c3c776dbc47422f445dadd9bd8e9c3b7a03244",
        "static": R2 / "evidence/REMOTE_STATIC_PREFLIGHT.json",
        "auto": FALLBACK / "k4_small_smoke_r5/evidence/K4_SMALL_SMOKE_AUTOMATIC.json",
    },
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def q(value: object) -> str:
    return shlex.quote(str(value))


def remote_python(access: Access, code: str, *, timeout: int = 360) -> dict:
    output = access.ssh("env PYTHONDONTWRITEBYTECODE=1 MKL_THREADING_LAYER=GNU " + PYTHON + " -B -c " + q(code), timeout=timeout)
    return json.loads(output.strip().splitlines()[-1])


def local_check(route: str, cfg: dict) -> tuple[Path, dict]:
    root = BASE / cfg["root"]
    plan = root / cfg["plan"]
    auth = root / f"evidence/{route}_175_TRAINING_AUTHORIZATION.json"
    slurm = root / "slurm" / cfg["slurm"]
    trainer = root / "runtime" / TRAINER
    receipt = root / "evidence/SUBMISSION_RECEIPT.json"
    for path in (plan, auth, slurm, trainer):
        if not path.is_file():
            raise RuntimeError(f"missing local contract: {path}")
    if receipt.exists():
        raise RuntimeError("submission receipt already exists")
    data = json.loads(plan.read_text())
    approval = json.loads(auth.read_text())
    script = slurm.read_text()
    if data["epochs"] != 175 or approval["route_id"] != route or not approval["training_authorization"]:
        raise RuntimeError("route authorization or epoch contract failed")
    if data["formal_gate_status"] != "not_run_not_passed" or approval["operator_approved"] is not False:
        raise RuntimeError("experimental status drift")
    required = (cfg["root"], cfg["job_name"], cfg["plan"], f"K1245_ROUTE_{route}.json",
                "--gres=gpu:1", "--ntasks=1", "S2_K1245_EPOCHS=175",
                "S2_MET_AUG_FIX_V3_CALIBRATION", "export MKL_THREADING_LAYER=GNU")
    if any(value not in script for value in required) or "--exclusive" in script:
        raise RuntimeError("Slurm path/resource/environment contract drift")
    raw_contract = str(RAW) in script or (route == "K4" and
                    str(RAW.parent) in script and 'nnUNet_raw="$SOURCE/nnUNet_raw"' in script)
    if not raw_contract:
        raise RuntimeError("Slurm raw-data path drift")
    subprocess.run(["bash", "-n", str(slurm)], check=True)
    ast.parse(trainer.read_text(), filename=str(trainer))
    return root, {"plan_sha256": sha(plan), "slurm_sha256": sha(slurm),
                  "trainer_sha256": sha(trainer), "auth_sha256": sha(auth)}


def main(route: str) -> None:
    cfg = ROUTES[route]
    root, hashes = local_check(route, cfg)
    remote = REMOTE_BASE / cfg["root"]
    audit = {**META, "route_id": route, "local_started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "remote_root": str(remote), **hashes}
    access = Access(f"submit_{route.lower()}_175_{cfg['root']}")
    parameters = {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()}
    parameters.update({"route": route, "remote": str(remote), "parent": str(PARENT),
                       "calibration": str(CALIBRATION), "original_e": str(ORIGINAL_E),
                       "raw": str(RAW), "hashes": hashes, "trainer": TRAINER})
    try:
        access.start_vpn()
        readonly = remote_python(access, "C=" + repr(parameters) + "\n" + r'''
import datetime,hashlib,json,pathlib,subprocess,sys
p=lambda value:pathlib.Path(value)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
r=p(C['remote']); source=p(C['source']); config=p(C['config']); pool=p(C['pool'])
assert not r.exists(), 'attempt root already exists'
assert source.is_dir() and (source/'runtime/repository/custom_nnunet').is_dir()
assert sha(config)==C['config_sha256']
assert json.loads(config.read_text())['route_id']==C['route']
manifest=json.loads((pool/'component_manifest.json').read_text())
assert manifest['manifest_sha256']==C['manifest_identity'] and manifest['component_count']>=24
sys.path.insert(0,str(source/'runtime/repository'))
from custom_nnunet.met_aug_k1245 import K1245Manifest
assert K1245Manifest.load(pool/'component_manifest.json').identity_sha256==C['manifest_identity']
auto=json.loads(p(C['auto']).read_text())
assert auto['automatic_pass_count']==24 and auto['manifest_sha256']==C['manifest_identity']
assert sha(p(C['calibration']))=='68a5d8a6c501804b7a47d626e43a58ff71b781154c0cc955a9c5a5d5b786d37f'
assert sha(p(C['original_e']))=='4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267'
assert (p(C['raw'])/'Dataset264_BraTS2026_MET_Completion').is_dir()
assert (p(C['parent'])/'data/nnUNet_preprocessed/Dataset264_BraTS2026_MET_Completion').is_dir()
assert p(C['static']).is_file()
assert pathlib.Path('/public/home/zqchen/.conda/envs/segmamba/bin/nnUNetv2_train').is_file()
runtime=subprocess.run(['/usr/bin/env','MKL_THREADING_LAYER=GNU','/public/home/zqchen/.conda/envs/segmamba/bin/python','-B','-c','import numpy,torch'],capture_output=True,text=True)
assert runtime.returncode==0, runtime.stderr[-500:]
if C['route']=='K2':
    d=json.loads(config.read_text())
    empirical=d['empirical']
    assert d['features']=={'b':True,'d':True,'e':True,'f':False,'x2':True,'x3':False}
    assert d['second_tumour_probability']==0.20
    assert len(empirical['rank_volume_samples'])==7 and len(empirical['nn_distance_samples_mm'])==6887
assert not subprocess.run(['squeue','-h','-n',C['job_name'],'-o','%i'],capture_output=True,text=True,check=True).stdout.strip()
test=subprocess.run(['sbatch','--test-only','--partition=a100','--gres=gpu:1','--ntasks=1','--cpus-per-task=8','--mem=64G','--time=5-00:00:00','--wrap=true'],capture_output=True,text=True)
assert test.returncode==0, test.stderr[-500:]
print(json.dumps({'status':'pass','remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_name':C['job_name'],'pool_manifest_identity':C['manifest_identity']}))
''')
        audit["readonly_preflight"] = readonly
        access.ssh("set -euo pipefail; mkdir -m 700 " + q(remote) + "; mkdir -m 700 " +
                   " ".join(q(remote / part) for part in ("evidence", "logs", "slurm", "runtime", "results", "config", "pools")))
        access.scp_immutable(root / cfg["plan"], str(remote))
        access.scp_immutable(root / f"evidence/{route}_175_TRAINING_AUTHORIZATION.json", str(remote / "evidence"))
        access.scp_immutable(root / "slurm" / cfg["slurm"], str(remote / "slurm"))
        access.ssh("set -euo pipefail; src=" + q(cfg["source"]) + "; dst=" + q(remote) + "; "
                   "cp -a \"$src/runtime/repository\" \"$dst/runtime/\"; "
                   "cp -p " + q(cfg["config"]) + " \"$dst/config/K1245_ROUTE_" + route + ".json\"; "
                   "cp -p " + q(cfg["static"]) + " \"$dst/evidence/REMOTE_STATIC_PREFLIGHT.json\"; "
                   "cp -p " + q(R2 / "evidence/POOL_BUILD_COMPLETE.ok") + " \"$dst/evidence/POOL_BUILD_COMPLETE.ok\"; "
                   "ln -s " + q(cfg["pool"]) + " \"$dst/pools/" + Path(cfg["pool"]).name + "\"; "
                   "chmod u+w \"$dst/config/K1245_ROUTE_" + route + ".json\" \"$dst/evidence/" + route + "_175_TRAINING_AUTHORIZATION.json\""
                   , timeout=180)
        access.scp_immutable(root / "runtime" / TRAINER, str(remote / "runtime/repository/custom_nnunet"))
        launched = remote_python(access, "C=" + repr(parameters) + "\n" + r'''
import datetime,hashlib,importlib,json,math,os,pathlib,subprocess,sys
p=lambda value:pathlib.Path(value)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
r=p(C['remote']); repo=r/'runtime/repository'; sys.path.insert(0,str(repo))
plan=r/C['plan']; slurm=r/'slurm'/C['slurm']; trainer=repo/'custom_nnunet'/C['trainer']
cfg=r/'config'/('K1245_ROUTE_'+C['route']+'.json')
pool=r/'pools'/p(C['pool']).name; man=pool/'component_manifest.json'
auth=r/'evidence'/(C['route']+'_175_TRAINING_AUTHORIZATION.json')
assert sha(plan)==C['hashes']['plan_sha256'] and sha(slurm)==C['hashes']['slurm_sha256']
assert sha(trainer)==C['hashes']['trainer_sha256'] and sha(auth)==C['hashes']['auth_sha256']
from custom_nnunet.met_aug_core import canonical_json_sha256
from custom_nnunet.met_aug_k1245 import K1245Config,K1245Manifest
x=json.loads(cfg.read_text()); assert x['route_id']==C['route']
x['plan_sha256']=sha(plan)
x['config_sha256']=canonical_json_sha256(x,exclude=('config_sha256',))
cfg.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
manifest=K1245Manifest.load(man)
assert manifest.identity_sha256==C['manifest_identity']
loaded=K1245Config.load(cfg,manifest=manifest,plan_path=plan)
assert loaded.route_id==C['route']
a=json.loads(auth.read_text()); assert a['status']=='pass' and a['route_id']==C['route'] and a['training_authorization'] is True
a.update({'plan_sha256':sha(plan),'config_sha256':sha(cfg),'manifest_sha256':manifest.identity_sha256})
auth.write_text(json.dumps(a,indent=2,sort_keys=True)+'\n')
for file in (repo/'custom_nnunet').glob('*.py'): compile(file.read_bytes(),str(file),'exec')
name=C['trainer'][:-3]
module=importlib.import_module('custom_nnunet.'+name)
assert getattr(module,name).__name__==name
subprocess.run(['bash','-n',str(slurm)],check=True)
body=slurm.read_text()
assert C['remote'] in body and C['job_name'] in body and C['plan'] in body
assert '--gres=gpu:1' in body and '--exclusive' not in body
assert 'export S2_MET_AUG_FIX_V3_CALIBRATION=' in body
assert 'export MKL_THREADING_LAYER=GNU' in body
assert str(p(C['raw'])) in body or (C['route']=='K4' and str(p(C['raw']).parent) in body and 'nnUNet_raw="$SOURCE/nnUNet_raw"' in body)
assert not subprocess.run(['squeue','-h','-n',C['job_name'],'-o','%i'],capture_output=True,text=True,check=True).stdout.strip()
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
e=r/'evidence'; meta={'artifact_status':'experimental_unvalidated','operator_approved':False,'formal_gate_status':'not_run_not_passed','F_STRICT_PREREG_GATE':False}
pre={**meta,'status':'pass','route_id':C['route'],'job_name':C['job_name'],'epochs':175,'plan_sha256':sha(plan),'config_sha256':sha(cfg),'manifest_sha256':manifest.identity_sha256,'trainer_sha256':sha(trainer),'slurm_sha256':sha(slurm),'calibration_sha256':sha(p(C['calibration'])),'original_e_sha256':sha(p(C['original_e'])),'remote_utc':stamp}
def put(name,payload):
 fd=os.open(e/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
 with os.fdopen(fd,'w') as handle: json.dump(payload,handle,indent=2,sort_keys=True); handle.write('\n')
put(C['route']+'_175_STATIC_PREFLIGHT.json',pre)
put(C['route']+'_175_RESERVATION.json',{**pre,'status':'reserved'})
submit=subprocess.run(['sbatch','--hold','--parsable',str(slurm)],capture_output=True,text=True)
if submit.returncode:
 put(C['route']+'_175_SUBMISSION_REJECTED.json',{**pre,'status':'submission_rejected','exit_code':submit.returncode,'stderr_tail':submit.stderr[-600:]})
 raise RuntimeError('held sbatch rejected: '+submit.stderr[-300:])
job=submit.stdout.strip().split(';')[0]
assert job.isdigit()
put(C['route']+'_175_LAUNCH.json',{**pre,'status':'submitted_held','job_id':job})
release=subprocess.run(['scontrol','release',job],capture_output=True,text=True)
if release.returncode:
 put(C['route']+'_175_RELEASE_FAILED.json',{**pre,'status':'release_failed','job_id':job,'stderr_tail':release.stderr[-600:]})
 raise RuntimeError('held job accepted but release failed')
put(C['route']+'_175_RELEASED.json',{**pre,'status':'released','job_id':job})
queue=subprocess.run(['squeue','-h','-j',job,'-o','%i|%j|%T|%R'],capture_output=True,text=True,check=True).stdout.strip()
accounting=subprocess.run(['sacct','-X','-n','-P','-j',job,'--format=JobID,JobName,State,ExitCode'],capture_output=True,text=True,check=True).stdout.strip()
print(json.dumps({'status':'released','job_id':job,'job_name':C['job_name'],'queue':queue,'sacct':accounting,'remote_utc':stamp}))
''', timeout=360)
        audit["launch"] = launched
        audit["local_completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        exclusive(root / "evidence/SUBMISSION_RECEIPT.json", audit)
        print(json.dumps(launched, sort_keys=True))
    finally:
        access.close_vpn()


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ROUTES:
        raise SystemExit("usage: submit_isolated_175_route_once.py K1|K2|K4")
    main(sys.argv[1])
