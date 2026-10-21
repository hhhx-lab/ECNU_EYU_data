"""One read-only asset snapshot for the independent follow-up diagnostics."""
import base64
import hashlib
import importlib.util
import json
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S2 = ROOT.parents[2]
HELPER = S2 / '03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260923_r3_fallback/operator/access.py'
PARENT = ROOT.parent / 's2_k1_small_lesion_p0_20261014_r2'
CODE = r'''
import csv,hashlib,json,os,subprocess,importlib.metadata
from pathlib import Path
from datetime import datetime,timezone
base=Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2')
parent=base/'experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1'
r2=base/'experimental_unvalidated/s2_k1_small_lesion_p0_20261014_r2'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def cmd(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=30)
 return dict(rc=r.returncode,stdout=r.stdout,stderr=r.stderr)
plan=json.loads((r2/'CPU_EXECUTION_PLAN.json').read_text())
assert sha(r2/'CPU_EXECUTION_PLAN.json')=='e3169bf735a39d05c066ca34c5568285b48ba060c79007331a850c903c936beb'
drift={p:sha(Path(p)) for group in ('dependency_sha256','software_sha256','input_reference_sha256') for p,d in plan[group].items() if sha(Path(p))!=d}
out=dict(remote_utc=datetime.now(timezone.utc).isoformat(),hostname=os.uname().nodename,
 squeue=cmd(['squeue','-h','-j','4068477','-o','%i|%j|%T|%R']),
 sacct=cmd(['sacct','-n','-X','-j','4068477','--format=JobID,JobName,State,ExitCode','-P']),
 disk=cmd(['df','-B1',str(r2)]),dependency_drift=drift)
mapping=parent/'inputs/mapping/nnunet_case_mapping_master.csv'
train_path=parent/'inputs/splits/train_fixed.txt'
val_path=parent/'inputs/splits/val_fixed.txt'
rows=list(csv.DictReader(mapping.open()))
train=train_path.read_text().splitlines(); val=val_path.read_text().splitlines()
raw=base/'data/ecnu_completion_emergency/nnUNet_raw/Dataset264_BraTS2026_MET_Completion'
out['mapping']=dict(path=str(mapping),sha256=sha(mapping),rows=rows)
out['train']=dict(path=str(train_path),sha256=sha(train_path),ids=train)
out['val']=dict(path=str(val_path),sha256=sha(val_path),ids=val)
out['raw_dataset']=str(raw)
out['train_labels']=[dict(case=c,path=str(raw/'labelsTr'/(c+'.nii.gz')),sha256=sha(raw/'labelsTr'/(c+'.nii.gz'))) for c in train]
out['raw_dataset_json']=dict(path=str(raw/'dataset.json'),sha256=sha(raw/'dataset.json'),content=json.loads((raw/'dataset.json').read_text()))
out['r2_completion']={n:dict(sha256=sha(r2/'evidence'/n),content=(r2/'evidence'/n).read_text()) for n in ('P0_CPU_VALIDATION.json','OFFICIAL_IDENTITY_VALIDATION.json','P0_CPU_COMPLETE.ok')}
out['r2_geometry_sha256']=sha(r2/'roundtrip/GEOMETRY.json')
out['r2_matrix_sha256']=sha(r2/'results/cpu_identity/REFERENCE_DETECTION_MATRIX.csv')
out['r2_roundtrip']={p.name:sha(p) for p in (r2/'roundtrip/prediction').glob('*.nii.gz')}
routes={
 'B175':base/'results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260910_r1/cohort/fixed103/models/B',
 'D50':base/'results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r4/cohort/fixed103/models/D'}
out['routes']={}
for method,r in routes.items():
 out['routes'][method]=dict(root=str(r),predictions={p.name:sha(p) for p in (r/'prediction').glob('*.nii.gz')},
  evaluation={p.name:dict(path=str(p),sha256=sha(p),content=p.read_text()) for p in (r/'evaluation').glob('*') if p.is_file() and p.suffix in ('.json','.csv','.ok','.txt')})
out['original_e_candidates']={}
for r in (parent/'frozen/original_e',base/'results/s2_e_baseline_true1mm_eval_20260726_r1',base/'results/s2_official_validation_inference_20260727_r1',base/'experimental_unvalidated/s2_e_baseline_true1mm_eval_20260726_r1'):
 out['original_e_candidates'][str(r)]=[str(p.relative_to(r)) for p in r.glob('*/*')][:80] if r.exists() else None
resume=base/'experimental_unvalidated/s2_met_aug_k1_175_resume_20261002_r1/results/K1/Dataset264_BraTS2026_MET_Completion/nnUNetTrainerBraTS2026RCMetAugK1245FocalCompletionFineTune175__nnUNetPlans__3d_fullres/fold_0'
out['checkpoints']={str(epoch):dict(path=str(resume/('checkpoint_epoch_%04d.pth'%epoch)),sha256=sha(resume/('checkpoint_epoch_%04d.pth'%epoch))) for epoch in (150,175)}
dist=importlib.metadata.distribution('nnunetv2')
out['installed_sources']={}
for name in ('inference/predict_from_raw_data.py','training/dataloading/data_loader.py','training/dataloading/nnunet_dataset.py','training/nnUNetTrainer/nnUNetTrainer.py','training/data_augmentation/custom_transforms/deep_supervision_donwsampling.py'):
 p=Path(dist.locate_file('nnunetv2/'+name))
 out['installed_sources'][name]=dict(path=str(p),sha256=sha(p),source=p.read_text()) if p.is_file() else dict(missing=True,path=str(p))
out['completed_remote_utc']=datetime.now(timezone.utc).isoformat()
print('ASSET_SNAPSHOT='+json.dumps(out))
'''


def main():
    spec = importlib.util.spec_from_file_location('transport', HELPER)
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)
    t.ROOT = ROOT
    access = t.Access('inventory_once')
    started = datetime.now(timezone.utc).isoformat()
    status = 'failed'
    try:
        access.start_vpn()
        encoded = base64.b64encode(CODE.encode()).decode()
        raw = access.ssh('env PYTHONDONTWRITEBYTECODE=1 MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 ' + t.PYTHON + ' -B -c ' + shlex.quote('import base64;exec(base64.b64decode('+repr(encoded)+'))'), timeout=300)
        for k,v in access.env.items():
            if v and any(x in k.upper() for x in ('PASSWORD','TOKEN','SECRET')):
                raw = raw.replace(v,'[REDACTED]')
        d = json.loads(next(x.split('=',1)[1] for x in raw.splitlines() if x.startswith('ASSET_SNAPSHOT=')))
        t.exclusive(access.audit/'ASSET_SNAPSHOT.json',dict(**t.META,**d))
        assert not d['dependency_drift'], 'frozen asset drift; stop'
        assert len(d['train_labels']) == len(d['train']['ids']) == 1035
        assert len(d['val']['ids']) == len(d['r2_roundtrip']) == 103
        assert d['checkpoints']['175']['sha256']=='28e81950b675cd0f5ca26cc1509f9271b2fa827b524f379024b417c0a5402204'
        assert d['checkpoints']['150']['sha256']=='ba7ec030030913ace2939091d07f500ee7041d0f9e615a1fe16f535d805c3338'
        status='received_readonly_assets'
        print(json.dumps(dict(status=status,audit=str(access.audit),remote_utc=d['remote_utc'],train_labels=len(d['train_labels']),routes={m:len(v['predictions']) for m,v in d['routes'].items()},original_e_candidates=d['original_e_candidates'])))
    finally:
        access.close_vpn()
        t.exclusive(access.audit/'LOCAL_OPERATION_RECEIPT.json',dict(**t.META,status=status,local_start_utc=started,local_end_utc=datetime.now(timezone.utc).isoformat(),vpn_closed=access.owns_vpn,preexisting_vpn_preserved=not access.owns_vpn,progress_snapshots=1,remote_writes=False))


if __name__ == '__main__':
    main()
