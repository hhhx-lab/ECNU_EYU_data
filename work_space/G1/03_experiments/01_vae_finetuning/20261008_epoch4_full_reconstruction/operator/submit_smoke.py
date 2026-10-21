"""Deploy frozen epoch004 reconstruction assets and submit one audited smoke job."""

import datetime
import importlib.util
import json
import pathlib
import re
import shlex


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT = pathlib.Path(__file__).resolve().parents[6]
REMOTE = pathlib.Path(
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


def static_check(a, sources):
    expected = {str(REMOTE / destination): m.sha(source) for source, destination in sources}
    code_sha = {
        "synthesis/utils.py": "5d5dd28de7dfed96fe6c3988da52563f37940e89850fc2b84accafd8bd55d3ab",
        "synthesis/spatial.py": "b16b80697fd0a9394eca81833356c79b2ba81a1cfe92669d1b414ce3b6569afb",
        "synthesis/pipeline.py": "0097c284d5d636dfc93f1dfd8461f5d02d54b666aa337a8f635a1c5b4853d396",
        "configs.py": "105da207a98a3195fa131679ba7ba3e960e5fb03d2441f878e57a40f58917863",
    }
    code = f'''
import pathlib,json,hashlib,ast,csv,collections,os,datetime,sys
import torch
from monai.bundle import ConfigParser
root=pathlib.Path({str(REMOTE)!r})
source=pathlib.Path({str(CODE)!r})
checkpoint=pathlib.Path({str(CHECKPOINT)!r})
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected={expected!r}
code_sha={code_sha!r}
assert all(sha(pathlib.Path(p))==s for p,s in expected.items())
assert all(sha(source/p)==s for p,s in code_sha.items())
ast.parse((root/'runtime/reconstruct_epoch004.py').read_text())
sys.path.insert(0,str(source))
import configs
import synthesis.utils
import synthesis.spatial
parser=ConfigParser(configs.NETWORKS_CONFIG)
parser.parse(True)
model=parser.get_parsed_content('autoencoder_def')
assert sha(checkpoint)=='5d86a0b3cbc5df34d64d5a0ff376eae453634a609bd24dec1a1191fde800b13b'
ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert ck.get('epoch')==4
model.load_state_dict(ck['model_state_dict'],strict=True)
with (source/'data/data_csv.csv').open(newline='') as stream: rows=list(csv.DictReader(stream))
assert len(rows)==1030 and collections.Counter(r['split'] for r in rows)=={{'train':823,'val':103,'test':104}}
assert {{'BraTS-MET-00001-000','BraTS-MET-01117-003'}}.issubset({{r['id'] for r in rows}})
out={{'status':'pass','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,'artifact_status':'experimental_unvalidated','operator_approved':False,'formal_gate_status':'not_run_not_passed','source_csv_sha256':sha(source/'data/data_csv.csv'),'plan_sha256':sha(root/'RECONSTRUCTION_PLAN.json'),'runtime_sha256':sha(root/'runtime/reconstruct_epoch004.py'),'slurm_sha256':sha(root/'slurm/run_reconstruction.slurm'),'checkpoint_sha256':sha(checkpoint),'source_code_sha256':code_sha,'model_strict_load':'pass','case_count':len(rows)}}
dest=root/'evidence/REMOTE_STATIC_PREFLIGHT.json'
fd=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
with os.fdopen(fd,'w') as stream: json.dump(out,stream,sort_keys=True,indent=2)
print(json.dumps(out,sort_keys=True))
'''
    output = a.ssh(
        "env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=" + shlex.quote(str(CODE))
        + " " + m.PYTHON + " -B -c " + shlex.quote(code), timeout=300
    )
    preflight = json.loads(output.strip().splitlines()[-1])
    if preflight["status"] != "pass" or preflight["model_strict_load"] != "pass":
        raise RuntimeError("remote static preflight did not pass")
    return preflight


def main():
    sources = [
        (ROOT / "RECONSTRUCTION_PLAN.json", "RECONSTRUCTION_PLAN.json"),
        (ROOT / "README.md", "README.md"),
        (ROOT / "runtime/reconstruct_epoch004.py", "runtime/reconstruct_epoch004.py"),
        (ROOT / "slurm/run_reconstruction.slurm", "slurm/run_reconstruction.slurm"),
    ]
    if not all(path.is_file() for path, _ in sources):
        raise FileNotFoundError("local runtime package incomplete")
    a = m.Access("submit_smoke")
    try:
        a.start_vpn()
        a.ssh(
            "set -euo pipefail; test ! -e " + shlex.quote(str(REMOTE))
            + "; mkdir -m 700 " + shlex.quote(str(REMOTE))
            + "; mkdir -m 700 " + " ".join(shlex.quote(str(REMOTE / name))
                                     for name in ("runtime", "slurm", "logs", "evidence"))
        )
        for source, destination in sources:
            a.scp(source, str(REMOTE / destination))
        preflight = static_check(a, sources)
        submission = a.ssh(
            "sbatch --parsable --hold --export=ALL,RECON_PHASE=smoke "
            + shlex.quote(str(REMOTE / "slurm/run_reconstruction.slurm"))
        )
        job_id = submission.strip().splitlines()[-1].split(";", 1)[0]
        if not re.fullmatch(r"[0-9]+", job_id):
            raise RuntimeError("sbatch did not return a canonical job id")
        queue = a.ssh("squeue -h -j " + job_id + " -o '%i|%T|%j|%b'").strip()
        if job_id not in queue:
            raise RuntimeError("held smoke job missing from scheduler")
        audit = {
            **m.META, "status": "held_audited", "job_id": job_id,
            "name": "g1vae4recon", "phase": "smoke",
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "preflight": preflight, "held_squeue": queue,
        }
        code = (
            "import json,pathlib,os; p=pathlib.Path(" + repr(str(REMOTE / "evidence/SMOKE_LAUNCH_AUDIT.json"))
            + ");v=" + repr(json.dumps(audit, ensure_ascii=True))
            + ";fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400);"
            + "h=os.fdopen(fd,'w');h.write(v+'\\n');h.close();print('AUDIT_PASS')"
        )
        a.ssh("env PYTHONDONTWRITEBYTECODE=1 " + m.PYTHON
              + " -B -c " + shlex.quote(code))
        a.ssh("scontrol release " + job_id)
        released = a.ssh("squeue -h -j " + job_id + " -o '%i|%T|%j|%b'").strip()
        m.exclusive(a.audit / "SUBMISSION_RECEIPT.json", {
            **m.META, "status": "released", "job_id": job_id,
            "remote_root": str(REMOTE), "held_squeue": queue,
            "released_squeue": released,
            "plan_sha256": m.sha(ROOT / "RECONSTRUCTION_PLAN.json"),
            "runtime_sha256": m.sha(ROOT / "runtime/reconstruct_epoch004.py"),
            "local_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
        print(json.dumps({"job_id": job_id, "queue": released,
                          "audit": str(a.audit)}, ensure_ascii=False))
    finally:
        a.close_vpn()
        print("VPN_CLOSED_OR_PREEXISTING_PRESERVED")


if __name__ == "__main__":
    main()
