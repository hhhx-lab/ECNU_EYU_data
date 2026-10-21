"""One bounded read-only ECNU inventory; transport only, never deploy/submit."""
import base64
import importlib.util
import json
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S2 = ROOT.parents[2]
HELPER = S2 / "03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260923_r3_fallback/operator/access.py"

REMOTE_CODE = r'''
import csv, hashlib, importlib, importlib.metadata, inspect, json, os, subprocess
from datetime import datetime, timezone
from pathlib import Path
base=Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2')
parent=base/'experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1'
resume=base/'experimental_unvalidated/s2_met_aug_k1_175_resume_20261002_r1'
train=base/'experimental_unvalidated/s2_met_aug_k1245_20260929_k1_175_r3'
cohort=base/'results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r2/cohort/fixed103'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def command(argv):
 try:
  p=subprocess.run(argv,capture_output=True,text=True,timeout=25)
  return dict(rc=p.returncode,out=p.stdout,err=p.stderr)
 except subprocess.TimeoutExpired:
  return dict(rc=124,out='',err='bounded subprocess timeout; no retry')
def files(p):
 return sorted(str(x.relative_to(p)) for x in p.iterdir()) if p.is_dir() else []
out=dict(remote_utc=datetime.now(timezone.utc).isoformat(),hostname=os.uname().nodename,
 disk=command(['df','-h','/public']),squeue=command(['squeue','-h','-j','3919543,3940964','-o','%i|%j|%T|%R']),
 sacct=command(['sacct','-n','-X','-j','3919543,3940964','--format=JobID,JobName,State,ExitCode','-P']))
out['frozen_files']=[]
dependencies=json.loads((resume/'evidence/REMOTE_FINAL_PREFLIGHT.json').read_text())['dependency_sha256']
for path,expected in dependencies.items():
 p=Path(path)
 out['frozen_files'].append(dict(path=path,exists=p.is_file(),sha256=sha(p) if p.is_file() else None,expected_sha256=expected))
out['input_audit_sha256']=sha(cohort/'input_audit.csv')
with (cohort/'input_audit.csv').open() as f: audits=list(csv.DictReader(f))
checks=[]
for row in audits:
 for i,mod in enumerate(['t1n','t1c','t2w','t2f','seg']):
  p=cohort/('reference' if mod=='seg' else 'input')/(row['source_case_id']+('.nii.gz' if mod=='seg' else '_%04d.nii.gz'%i))
  # Audit column names are inspected rather than guessed if a contract differs.
  key=mod+'_sha256'
  if key not in row:
   checks.append(dict(path=str(p),error='audit column unavailable',columns=list(row)));continue
  checks.append(dict(path=str(p),sha256=sha(p),expected_sha256=row[key]))
out['input_reference_checks']=checks
out['cohort_counts']=dict(input=len(list((cohort/'input').glob('*.nii.gz'))),reference=len(list((cohort/'reference').glob('*.nii.gz'))))
prep=parent/'data/nnUNet_preprocessed/Dataset264_BraTS2026_MET_Completion'
out['preprocessed_dirs']=files(prep)
out['parent_inputs']=files(parent/'inputs')
out['parent_data']=files(parent/'data')
out['train_root']=files(train)
out['train_configs']=files(train/'config')
out['train_runtime']=files(train/'runtime')
out['raw_dirs']={str(p):files(p)[:15] for p in (parent/'data/nnUNet_raw',parent/'inputs',prep/'nnUNetPlans_3d_fullres')}
out['raw_dataset_dirs']={str(p):dict(children=files(p),labels_count=len(list((p/'labelsTr').glob('*.nii.gz')))) for p in (parent/'data/nnUNet_raw').glob('Dataset*') if p.is_dir()}
out['split_lists']={name:(parent/'inputs/splits'/name).read_text().splitlines() for name in ['train_fixed.txt','val_fixed.txt']}
out['resume_validation']=json.loads((resume/'evidence/TRAINING_VALIDATION.json').read_text())
out['plans']=json.loads((prep/'nnUNetPlans.json').read_text())
out['dataset']=json.loads((prep/'dataset.json').read_text())
out['g1_selection']=json.loads((parent/'frozen/g1/checkpoint_selection.json').read_text())
out['software']={}
for module in ['nnunetv2.preprocessing.resampling.default_resampling','nnunetv2.inference.export_prediction','nnunetv2.preprocessing.preprocessors.default_preprocessor','nnunetv2.utilities.plans_handling.plans_handler','nnunetv2.utilities.label_handling.label_handling','nnunetv2.preprocessing.cropping.cropping','nnunetv2.imageio.simpleitk_reader_writer']:
 try:
  p=Path(importlib.metadata.distribution('nnunetv2').locate_file(module.replace('.','/')+'.py'))
  out['software'][module]=dict(path=str(p),sha256=sha(p),source=p.read_text(),inspection='installed_source_only_not_runtime_import')
 except Exception as e:out['software'][module]=dict(error=type(e).__name__+': '+str(e))
out['versions']={n:importlib.metadata.version(n) for n in ['nnunetv2','numpy','torch','scipy','nibabel']}
fold=resume/'results/K1/Dataset264_BraTS2026_MET_Completion/nnUNetTrainerBraTS2026RCMetAugK1245FocalCompletionFineTune175__nnUNetPlans__3d_fullres/fold_0'
out['checkpoints']=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in (fold/'checkpoint_epoch_0150.pth',fold/'checkpoint_epoch_0175.pth') if p.is_file()]
out['prediction_assets']={}
for root,route in [('s2_met_aug_k1_selected103_official_met_20261008_r1','K1'),('s2_met_aug_k24_selected103_official_met_20261002_r1','K4')]:
 p=base/'results/experimental_unvalidated'/root/'cohort/fixed103/models'/route
 out['prediction_assets'][route]=dict(root=str(p),children=files(p),predictions=len(list((p/'prediction').glob('*.nii.gz'))),probabilities=len(list((p/'prediction').glob('*.npz'))))
eval_code="import importlib.metadata,json,pathlib,hashlib; p=pathlib.Path(importlib.metadata.distribution('panoptica').locate_file('panoptica')); b=pathlib.Path(importlib.metadata.distribution('BraTS-evaluation').locate_file('brats_evaluation')); print(json.dumps({'versions':{n:importlib.metadata.version(n) for n in ['panoptica','BraTS-evaluation','numpy']},'root':str(p),'inspection':'installed_source_only_not_runtime_import','python_sources':{str(x.relative_to(p)):x.read_text() for x in p.rglob('*.py')},'brats_sources':{str(x.relative_to(b)):x.read_text() for x in b.rglob('*') if x.is_file() and x.suffix in ['.py','.yaml']}}))"
e=command(['/public/home/zqchen/.conda/envs/brats_eval/bin/python','-B','-c',eval_code])
out['official_evaluator']=json.loads(e['out']) if e['rc']==0 else e
out['snapshot_end_remote_utc']=datetime.now(timezone.utc).isoformat()
print('P0_JSON_BEGIN')
print(json.dumps(out,allow_nan=False))
print('P0_JSON_END')
'''


def main():
    spec = importlib.util.spec_from_file_location("transport", HELPER)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    transport.ROOT = ROOT
    access = transport.Access("readonly_inventory")
    local_start = datetime.now(timezone.utc).isoformat()
    transport.exclusive(access.audit / "SNAPSHOT_REQUEST.json", dict(**transport.META,
        local_start_utc=local_start, remote_code_sha256=transport.sha(Path(__file__)),
        encoded_readonly_script_sha256=__import__('hashlib').sha256(REMOTE_CODE.encode()).hexdigest(),
        remote_code=REMOTE_CODE, action="read_only_no_deploy_no_sbatch"))
    outcome = "failed"
    try:
        access.start_vpn()
        encoded = base64.b64encode(REMOTE_CODE.encode()).decode()
        command = "env PYTHONDONTWRITEBYTECODE=1 MKL_THREADING_LAYER=GNU OPENBLAS_NUM_THREADS=1 " + transport.PYTHON + " -B -c " + shlex.quote("import base64;exec(base64.b64decode(" + repr(encoded) + "))")
        output = access.ssh(command, timeout=300)
        # Redact all secret-like env values before preserving arbitrary source text.
        for key, value in access.env.items():
            if any(word in key.upper() for word in ("PASSWORD", "TOKEN", "SECRET")) and value:
                output = output.replace(value, "[REDACTED]")
        payload = json.loads(output.split("P0_JSON_BEGIN\n", 1)[1].split("\nP0_JSON_END", 1)[0])
        transport.exclusive(access.audit / "REMOTE_INVENTORY.json", {**transport.META, **payload})
        outcome = "inventory_received"
        print(json.dumps(dict(status=outcome, audit=str(access.audit), remote_utc=payload["remote_utc"],
                              cohort_counts=payload["cohort_counts"], predictions=payload["prediction_assets"])))
    finally:
        access.close_vpn()
        transport.exclusive(access.audit / "LOCAL_OPERATION_RECEIPT.json", dict(**transport.META,
            status=outcome, local_start_utc=local_start, local_end_utc=datetime.now(timezone.utc).isoformat(),
            owns_vpn=access.owns_vpn, vpn_closed=access.owns_vpn,
            preexisting_vpn_preserved=not access.owns_vpn, action="single_readonly_inventory_no_remote_writes"))


if __name__ == "__main__":
    main()
