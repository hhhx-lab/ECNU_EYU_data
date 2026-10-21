"""Gate and submit the immutable full epoch004 reconstruction exactly once."""

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

EXPECTED = {
    "RECONSTRUCTION_PLAN.json": "d081764c30103d9f3d9010604a4fb0d78613066a6196d7f995ce0f1e6b54441a",
    "runtime/reconstruct_epoch004.py": "525b2dd071c956a2d24d6ad9885118a2f45bfe5f7401998e6ea553d96e33a5da",
    "slurm/run_reconstruction.slurm": "61d879bf2d9c437a6e0dd2194936ae89355912f864f0163e2f6af1fe01bef23b",
}
CHECKPOINT_SHA = "5d86a0b3cbc5df34d64d5a0ff376eae453634a609bd24dec1a1191fde800b13b"
CSV_SHA = "0d1765e12cdad7d8a5df57a5d2c46c79414ed99a2015a4d8fed57cd4c4fc836b"
SMOKE_JOB = "3940942"


def remote_gate(access):
    code = f'''
import csv,datetime,hashlib,json,os,pathlib
import nibabel as nib
import numpy as np
root=pathlib.Path({str(REMOTE)!r})
code_root=pathlib.Path({str(CODE)!r})
checkpoint=pathlib.Path({str(CHECKPOINT)!r})
expected={EXPECTED!r}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(root/p)==digest for p,digest in expected.items())
assert sha(checkpoint)=={CHECKPOINT_SHA!r}
assert sha(code_root/'data/data_csv.csv')=={CSV_SHA!r}
assert not (root/'dataset').exists()
assert not (root/'evidence/FULL_LAUNCH_AUDIT.json').exists()
assert not (root/'evidence/FULL_STATIC_PREFLIGHT.json').exists()
pre=json.loads((root/'evidence/REMOTE_STATIC_PREFLIGHT.json').read_text())
assert pre['status']=='pass' and pre['model_strict_load']=='pass'
assert pre['checkpoint_sha256']=={CHECKPOINT_SHA!r}
smoke=root/'smoke'
summary=json.loads((smoke/'DATASET_VALIDATION.json').read_text())
complete=json.loads((smoke/'COMPLETE.ok').read_text())
assert summary==complete and summary['status']=='pass'
assert summary['cases']==2 and summary['images']==8 and summary['labels']==2
assert summary['split_counts']=={{'train':1,'val':1}}
assert summary['checkpoint_sha256']=={CHECKPOINT_SHA!r}
assert summary['artifact_status']=='experimental_unvalidated'
assert summary['operator_approved'] is False and summary['formal_gate_status']=='not_run_not_passed'
with (code_root/'data/data_csv.csv').open(newline='') as stream:
    rows={{r['id']:r for r in csv.DictReader(stream)}}
assert len(rows)==1030
assert set(summary['manifest_sha256'])=={{'BraTS-MET-00001-000','BraTS-MET-01117-003'}}
for case,manifest_sha in summary['manifest_sha256'].items():
    case_dir=smoke/'cases'/case
    manifest_path=case_dir/'case_manifest.json'
    assert sha(manifest_path)==manifest_sha
    manifest=json.loads(manifest_path.read_text())
    assert manifest['case_id']==case and manifest['split']==rows[case]['split']
    assert manifest['checkpoint_sha256']=={CHECKPOINT_SHA!r}
    assert set(manifest['files'])=={{'t1n','t1c','t2w','t2f','seg'}}
    source_dir=code_root/'data/input'/case
    reference=nib.load(str(source_dir/pathlib.Path(rows[case]['t1n']).name))
    for modality,entry in manifest['files'].items():
        output=case_dir/entry['name']
        source=source_dir/pathlib.Path(rows[case][modality]).name
        assert output.stat().st_size==entry['bytes'] and sha(output)==entry['sha256']
        assert sha(source)==manifest['source_sha256'][modality]
        image=nib.load(str(output))
        assert image.shape==reference.shape and np.allclose(image.affine,reference.affine,atol=1e-4,rtol=0)
        assert np.isfinite(np.asanyarray(image.dataobj)).all()
        if modality=='seg':
            assert sha(output)==sha(source)
            assert set(np.unique(np.asanyarray(image.dataobj))).issubset({{0,1,2,3,4}})
out={{'artifact_status':'experimental_unvalidated','operator_approved':False,
     'formal_gate_status':'not_run_not_passed','status':'pass',
     'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'hostname':os.uname().nodename,'smoke_job_id':{SMOKE_JOB!r},
     'smoke_cases':2,'smoke_images':8,'smoke_labels':2,
     'plan_sha256':expected['RECONSTRUCTION_PLAN.json'],
     'runtime_sha256':expected['runtime/reconstruct_epoch004.py'],
     'slurm_sha256':expected['slurm/run_reconstruction.slurm'],
     'checkpoint_sha256':{CHECKPOINT_SHA!r},'source_csv_sha256':{CSV_SHA!r}}}
target=root/'evidence/FULL_STATIC_PREFLIGHT.json'
fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
with os.fdopen(fd,'w') as stream:
    json.dump(out,stream,sort_keys=True,indent=2)
    stream.write('\\n')
print(json.dumps(out,sort_keys=True))
'''
    output = access.ssh(
        "env PYTHONDONTWRITEBYTECODE=1 " + m.PYTHON + " -B -c "
        + shlex.quote(code), timeout=300
    )
    result = json.loads(output.strip().splitlines()[-1])
    if result.get("status") != "pass":
        raise RuntimeError("full static preflight did not pass")
    return result


def main():
    for name, digest in EXPECTED.items():
        if m.sha(ROOT / name) != digest:
            raise RuntimeError(f"frozen local asset drift: {name}")
    snapshot = ROOT / "operator/smoke_snapshot_20261008T051424Z/SMOKE_STATUS.json"
    status = json.loads(snapshot.read_text())
    if ("3940942|COMPLETED|0:0|" not in status["sacct"]["stdout"]
            or not status["files"]["COMPLETE.ok"]["exists"]
            or status["files"]["DATASET_VALIDATION.json"]["content"]["status"] != "pass"
            or len(status["case_validation"]) != 2):
        raise RuntimeError("local smoke completion gate failed")
    access = m.Access("submit_full")
    try:
        access.start_vpn()
        preflight = remote_gate(access)
        queue_before = access.ssh("squeue -h -u zqchen -o '%i|%T|%j|%b'").strip()
        if "g1vae4recon" in queue_before:
            raise RuntimeError("a reconstruction job is already queued")
        submission = access.ssh(
            "sbatch --parsable --hold --export=ALL,RECON_PHASE=full "
            + shlex.quote(str(REMOTE / "slurm/run_reconstruction.slurm"))
        )
        job_id = submission.strip().splitlines()[-1].split(";", 1)[0]
        if not re.fullmatch(r"[0-9]+", job_id):
            raise RuntimeError("sbatch did not return a canonical job id")
        queue = access.ssh("squeue -h -j " + job_id + " -o '%i|%T|%j|%b'").strip()
        if job_id not in queue or "g1vae4recon" not in queue:
            raise RuntimeError("held full job missing from scheduler")
        audit = {
            **m.META, "status": "held_audited", "job_id": job_id,
            "name": "g1vae4recon", "phase": "full",
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "preflight": preflight, "held_squeue": queue,
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
            "remote_root": str(REMOTE), "held_squeue": queue,
            "released_squeue": released, "preflight": preflight,
            "local_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
        print(json.dumps({"job_id": job_id, "queue": released,
                          "audit": str(access.audit)}, ensure_ascii=False))
    finally:
        access.close_vpn()
        print("VPN_CLOSED_OR_PREEXISTING_PRESERVED")


if __name__ == "__main__":
    main()
