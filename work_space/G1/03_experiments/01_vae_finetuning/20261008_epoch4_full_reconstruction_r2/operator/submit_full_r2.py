"""Deploy and submit one isolated full reconstruction recovery attempt."""

import datetime
import importlib.util
import json
import pathlib
import re
import shlex


ROOT = pathlib.Path(__file__).resolve().parents[1]
R1 = ROOT.parent / "20261008_epoch4_full_reconstruction"
PROJECT = ROOT.parents[4]
REMOTE = pathlib.Path(
    "/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/"
    "experimental_unvalidated/g1_vae_epoch004_full_reconstruction_20261008_r2"
)
PARENT = pathlib.Path(
    "/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/"
    "experimental_unvalidated/g1_vae_epoch004_full_reconstruction_20261008_r1"
)
CODE = pathlib.Path(
    "/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/code/"
    "brats2025-latent-ensemble-synthesis-main-v3"
)
CHECKPOINT = pathlib.Path(
    "/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/"
    "experimental_unvalidated/g1_vae_extended_finetune_20260901_r3_attempt03/"
    "training/checkpoints/checkpoint_epoch_004.pt"
)
spec = importlib.util.spec_from_file_location(
    "ecnu_access", PROJECT / "work_space/S2/03_experiments/12_k1245_rc_aware/"
    "s2_met_aug_k1245_20260923_r3_fallback/operator/access.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.ROOT = ROOT

PLAN = ROOT / "RECONSTRUCTION_RECOVERY_PLAN.json"
SLURM = ROOT / "slurm/run_full_reconstruction_r2.slurm"
RUNTIME = R1 / "runtime/reconstruct_epoch004.py"
SMOKE_STATUS = R1 / "operator/smoke_snapshot_20261008T051424Z/SMOKE_STATUS.json"
R1_FAILURE = R1 / "operator/readonly_full_20261008T054740Z/FULL_STATUS.json"
EXPECTED_PARENT = {
    "plan": "d081764c30103d9f3d9010604a4fb0d78613066a6196d7f995ce0f1e6b54441a",
    "runtime": "525b2dd071c956a2d24d6ad9885118a2f45bfe5f7401998e6ea553d96e33a5da",
    "smoke_status": "b708524c1264be721081cf1584e9f7c22007095e373421997c0e0994857dc5a2",
    "failure_status": "17455ad1ff2bfb08d2181eb5fbc601ec6dff9e691dddfe0105cb3a6ec90b1738",
    "checkpoint": "5d86a0b3cbc5df34d64d5a0ff376eae453634a609bd24dec1a1191fde800b13b",
    "source_csv": "0d1765e12cdad7d8a5df57a5d2c46c79414ed99a2015a4d8fed57cd4c4fc836b",
}


def local_gate():
    if m.sha(R1 / "RECONSTRUCTION_PLAN.json") != EXPECTED_PARENT["plan"]:
        raise RuntimeError("r1 plan changed")
    for name, path in (("runtime", RUNTIME), ("smoke_status", SMOKE_STATUS),
                       ("failure_status", R1_FAILURE)):
        if m.sha(path) != EXPECTED_PARENT[name]:
            raise RuntimeError(f"r1 {name} changed")
    plan = json.loads(PLAN.read_text())
    if (plan["parent_plan_sha256"] != EXPECTED_PARENT["plan"]
            or plan["parent_runtime_sha256"] != EXPECTED_PARENT["runtime"]
            or plan["parent_smoke_status_sha256"] != EXPECTED_PARENT["smoke_status"]
            or plan["parent_failed_full_status_sha256"] != EXPECTED_PARENT["failure_status"]
            or plan["checkpoint_sha256"] != EXPECTED_PARENT["checkpoint"]
            or plan["source_csv_sha256"] != EXPECTED_PARENT["source_csv"]):
        raise RuntimeError("r2 plan parent identities differ")
    smoke = json.loads(SMOKE_STATUS.read_text())
    failed = json.loads(R1_FAILURE.read_text())
    if ("3940942|COMPLETED|0:0|" not in smoke["sacct"]["stdout"]
            or smoke["files"]["COMPLETE.ok"]["content"]["status"] != "pass"
            or len(smoke["case_validation"]) != 2):
        raise RuntimeError("r1 smoke was not accepted")
    if ("3940993|FAILED|1:0|" not in failed["sacct"]["stdout"]
            or "CASES[@]: unbound variable" not in failed["logs"]["g1vae4recon_3940993.err"]["tail"]
            or failed["case_dirs"] != 0):
        raise RuntimeError("r1 failure identity differs")
    script = SLURM.read_text()
    if ("CASES[@]" in script or "RECON_PREFLIGHT_ONLY" not in script
            or "--output-root \"$ROOT/dataset\"" not in script):
        raise RuntimeError("r2 script did not remove empty-array contract")
    return plan


def remote_preflight(access, sources):
    expected = {str(REMOTE / relative): m.sha(source) for source, relative in sources}
    code = f'''
import ast,collections,csv,datetime,hashlib,importlib.util,json,os,pathlib,subprocess,sys
import torch
from monai.bundle import ConfigParser
root=pathlib.Path({str(REMOTE)!r})
parent=pathlib.Path({str(PARENT)!r})
code_root=pathlib.Path({str(CODE)!r})
checkpoint=pathlib.Path({str(CHECKPOINT)!r})
expected={expected!r}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(pathlib.Path(p))==digest for p,digest in expected.items())
plan=json.loads((root/'RECONSTRUCTION_RECOVERY_PLAN.json').read_text())
assert plan['plan_id']=='G1_VAE_EPOCH004_FULL_RECONSTRUCTION_20261008_R2'
assert plan['artifact_status']=='experimental_unvalidated'
assert plan['operator_approved'] is False and plan['formal_gate_status']=='not_run_not_passed'
assert sha(parent/'RECONSTRUCTION_PLAN.json')==plan['parent_plan_sha256']
assert sha(parent/'runtime/reconstruct_epoch004.py')==plan['parent_runtime_sha256']
assert sha(root/'runtime/reconstruct_epoch004.py')==plan['parent_runtime_sha256']
assert sha(checkpoint)==plan['checkpoint_sha256']=={EXPECTED_PARENT['checkpoint']!r}
assert sha(code_root/'data/data_csv.csv')==plan['source_csv_sha256']=={EXPECTED_PARENT['source_csv']!r}
assert not (root/'dataset').exists() and not (root/'evidence/FULL_LAUNCH_AUDIT.json').exists()
assert not (root/'evidence/REMOTE_STATIC_PREFLIGHT.json').exists()
smoke=json.loads((parent/'smoke/COMPLETE.ok').read_text())
assert smoke['status']=='pass' and smoke['cases']==2 and smoke['images']==8 and smoke['labels']==2
assert smoke['checkpoint_sha256']==plan['checkpoint_sha256']
assert len(smoke['manifest_sha256'])==2
assert 'CASES[@]: unbound variable' in (parent/'logs/g1vae4recon_3940993.err').read_text()
assert not (parent/'dataset').exists()
with (code_root/'data/data_csv.csv').open(newline='') as stream: rows=list(csv.DictReader(stream))
assert len(rows)==1030 and collections.Counter(r['split'] for r in rows)=={{'train':823,'val':103,'test':104}}
ast.parse((root/'runtime/reconstruct_epoch004.py').read_text())
runtime_spec=importlib.util.spec_from_file_location('g1_epoch004_runtime',root/'runtime/reconstruct_epoch004.py')
runtime_module=importlib.util.module_from_spec(runtime_spec)
runtime_spec.loader.exec_module(runtime_module)
sys.path.insert(0,str(code_root))
import configs
import synthesis.utils
import synthesis.spatial
parser=ConfigParser(configs.NETWORKS_CONFIG)
parser.parse(True)
model=parser.get_parsed_content('autoencoder_def')
ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert ck.get('epoch')==4
model.load_state_dict(ck['model_state_dict'],strict=True)
slurm=root/'slurm/run_full_reconstruction_r2.slurm'
subprocess.run(['bash','-n',str(slurm)],check=True)
env=os.environ.copy()
env['RECON_PHASE']='full'
env['RECON_PREFLIGHT_ONLY']='1'
dry=subprocess.run(['bash','-u',str(slurm)],env=env,text=True,capture_output=True,check=True)
assert 'FULL_R2_BASH_U_PREFLIGHT_PASS' in dry.stdout
out={{'artifact_status':'experimental_unvalidated','operator_approved':False,
     'formal_gate_status':'not_run_not_passed','status':'pass',
     'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'hostname':os.uname().nodename,'plan_sha256':sha(root/'RECONSTRUCTION_RECOVERY_PLAN.json'),
     'slurm_sha256':sha(slurm),'runtime_sha256':sha(root/'runtime/reconstruct_epoch004.py'),
     'checkpoint_sha256':sha(checkpoint),'source_csv_sha256':sha(code_root/'data/data_csv.csv'),
     'parent_failed_job_id':3940993,'parent_smoke_job_id':3940942,
     'bash_n':'pass','bash_u_full_dry_run':'pass','model_strict_load':'pass',
     'cases':len(rows),'split_counts':dict(collections.Counter(r['split'] for r in rows))}}
dest=root/'evidence/REMOTE_STATIC_PREFLIGHT.json'
fd=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
with os.fdopen(fd,'w') as stream:
    json.dump(out,stream,sort_keys=True,indent=2)
    stream.write('\\n')
print(json.dumps(out,sort_keys=True))
'''
    output = access.ssh(
        "env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=" + shlex.quote(str(CODE))
        + " " + m.PYTHON + " -B -c " + shlex.quote(code), timeout=300
    )
    result = json.loads(output.strip().splitlines()[-1])
    if (result.get("status") != "pass" or result.get("bash_u_full_dry_run") != "pass"
            or result.get("model_strict_load") != "pass"):
        raise RuntimeError("remote static preflight did not pass")
    return result


def main():
    local_gate()
    sources = [
        (PLAN, "RECONSTRUCTION_RECOVERY_PLAN.json"),
        (ROOT / "README.md", "README.md"),
        (SLURM, "slurm/run_full_reconstruction_r2.slurm"),
        (RUNTIME, "runtime/reconstruct_epoch004.py"),
        (SMOKE_STATUS, "sources/R1_SMOKE_STATUS.json"),
        (R1_FAILURE, "sources/R1_FULL_STATUS.json"),
    ]
    if not all(source.is_file() for source, _ in sources):
        raise FileNotFoundError("r2 package incomplete")
    access = m.Access("submit_full_r2")
    vpn_started = False
    try:
        access.start_vpn()
        vpn_started = True
        root = shlex.quote(str(REMOTE))
        access.ssh("set -euo pipefail; test ! -e " + root + "; mkdir -m 700 " + root
                   + "; mkdir -m 700 " + " ".join(shlex.quote(str(REMOTE / name))
                       for name in ("runtime", "slurm", "logs", "evidence", "sources")))
        for source, relative in sources:
            access.scp(source, str(REMOTE / relative))
        preflight = remote_preflight(access, sources)
        submission = access.ssh(
            "sbatch --parsable --hold --export=ALL,RECON_PHASE=full "
            + shlex.quote(str(REMOTE / "slurm/run_full_reconstruction_r2.slurm"))
        )
        job_id = submission.strip().splitlines()[-1].split(";", 1)[0]
        if not re.fullmatch(r"[0-9]+", job_id):
            raise RuntimeError("sbatch did not return a canonical job id")
        held = access.ssh("squeue -h -j " + job_id + " -o '%i|%T|%j|%b'").strip()
        if job_id not in held or "g1vae4fullr2" not in held:
            raise RuntimeError("held r2 job missing from scheduler")
        audit = {
            **m.META, "status": "held_audited", "job_id": job_id,
            "name": "g1vae4fullr2", "phase": "full", "parent_failed_job_id": 3940993,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "preflight": preflight, "held_squeue": held,
        }
        code = (
            "import json,pathlib,os; p=pathlib.Path(" + repr(str(REMOTE / "evidence/FULL_LAUNCH_AUDIT.json"))
            + ");v=" + repr(json.dumps(audit, ensure_ascii=True))
            + ";fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400);"
            + "h=os.fdopen(fd,'w');h.write(v+'\\n');h.close();print('AUDIT_PASS')"
        )
        access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + m.PYTHON
                   + " -B -c " + shlex.quote(code))
        access.ssh("scontrol release " + job_id)
        released = access.ssh("squeue -h -j " + job_id + " -o '%i|%T|%j|%b'").strip()
        m.exclusive(access.audit / "SUBMISSION_RECEIPT.json", {
            **m.META, "status": "released", "job_id": job_id,
            "remote_root": str(REMOTE), "held_squeue": held,
            "released_squeue": released, "preflight": preflight,
            "local_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
        print(json.dumps({"job_id": job_id, "queue": released,
                          "audit": str(access.audit)}, ensure_ascii=False))
    finally:
        if vpn_started:
            access.close_vpn()
            print("VPN_CLOSED_OR_PREEXISTING_PRESERVED")


if __name__ == "__main__":
    main()
