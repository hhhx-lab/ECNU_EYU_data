"""Read one source-job/checkpoint snapshot using the existing VPN access helper."""

import base64
import importlib.util
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCESS = ROOT.parent / 's2_met_aug_k1245_20260923_r3_fallback/operator/access.py'

REMOTE_CODE = r'''
import base64, datetime as dt, hashlib, inspect, json, math, os, pathlib, re, subprocess
import torch
torch.set_num_threads(2)
source=pathlib.Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1245_20260929_k1_175_r3')
trainer='nnUNetTrainerBraTS2026RCMetAugK1245FocalCompletionFineTune175'
fold=source/f'results/K1/Dataset264_BraTS2026_MET_Completion/{trainer}__nnUNetPlans__3d_fullres/fold_0'
meta=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed')
def cmd(argv):
 r=subprocess.run(argv,capture_output=True,text=True,timeout=20)
 return dict(rc=r.returncode,out=r.stdout.strip(),err=r.stderr.strip())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def finite(v):
 if torch.is_tensor(v): return bool(torch.isfinite(v).all())
 if isinstance(v,dict): return all(finite(x) for x in v.values())
 if isinstance(v,(list,tuple)): return all(finite(x) for x in v)
 if isinstance(v,float): return math.isfinite(v)
 return True
def stat(p):
 s=p.stat(); return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino)
def require(v,m):
 if not v: raise RuntimeError(m)
report={**meta,'remote_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'hostname':os.uname().nodename,'disk':cmd(['df','-h','/public']),
 'squeue':cmd(['squeue','-h','-j','3827543','-o','%i|%j|%T|%M|%l|%N|%R']),
 'sacct':cmd(['sacct','-n','-P','-X','-j','3827543','-o','JobIDRaw,JobName,State,ExitCode,Elapsed,Timelimit,End']),
 'markers':{n:(source/f'evidence/K1_175_TRAINING_{n}.json').exists() for n in ('STARTED','FAILED','VALIDATION')},'files':{}}
report['markers']['COMPLETE']=(source/'evidence/K1_175_TRAINING_COMPLETE.ok').exists()
started=json.loads((source/'evidence/K1_175_TRAINING_STARTED.json').read_text())
plan=source/'K1245_K1_175_EXECUTION_PLAN_20260929_R3.json'
cfg=source/'config/K1245_ROUTE_K1.json'
manifest=source/'pools/rc27/component_manifest.json'
repo=source/'runtime/repository'
for name,p in [('plan',plan),('config',cfg),('manifest',manifest),('trainer',repo/f'custom_nnunet/{trainer}.py')]:
 require(sha(p)==started[name+'_sha256'],name+' STARTED SHA drift')
report['source_contract_sha256']={str(p.relative_to(source)):sha(p) for p in [plan,cfg,manifest,source/'evidence/K1_175_TRAINING_AUTHORIZATION.json',*sorted((repo/'custom_nnunet').glob('*.py'))]}
ckp=fold/'checkpoint_latest.pth'
require(ckp.is_file(),'checkpoint_latest.pth absent')
before=stat(ckp)
ck=torch.load(ckp,map_location='cpu',weights_only=False)
digest=sha(ckp)
after=stat(ckp)
if before!=after:
 report['checkpoint']={'status':'deferred_source_changed_during_read','before':before,'after':after}
else:
 require(ck['trainer_name']==trainer,'checkpoint trainer drift')
 epoch=int(ck['current_epoch']); require(0<epoch<175 and epoch%25==0,'latest checkpoint epoch contract')
 require(bool(ck['network_weights']) and bool(ck['optimizer_state'].get('state')),'network/optimizer state absent')
 require('grad_scaler_state' in ck and ck['grad_scaler_state'] is not None,'CUDA AMP state absent')
 require(all(finite(ck[k]) for k in ['network_weights','optimizer_state','grad_scaler_state','logging']),'nonfinite state')
 logging=ck['logging']
 for k in ('ema_fg_dice','train_losses','val_losses','lrs'):
  require(k in logging and len(logging[k])==epoch,k+' length drift')
 for group in ck['optimizer_state']['param_groups']:
  require(math.isclose(group['lr'],0.001*(1-(epoch-1)/175)**0.9,rel_tol=1e-9), 'saved LR drift')
 from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
 param=torch.nn.Parameter(torch.zeros(1))
 optimizer=torch.optim.SGD([param],lr=0.001)
 scheduler=PolyLRScheduler(optimizer,0.001,175)
 scheduler.step(epoch)
 require(math.isclose(optimizer.param_groups[0]['lr'],0.001*(1-epoch/175)**0.9,rel_tol=1e-9),'resumed schedule drift')
 report['checkpoint']={'status':'pass','path':str(ckp),'sha256':digest,'stat':after,'current_epoch':epoch,'trainer_name':ck['trainer_name'],
 'network_finite':True,'optimizer_finite':True,'amp_finite':True,'logging_finite':True,'network_tensor_count':len(ck['network_weights']),
 'optimizer_state_count':len(ck['optimizer_state']['state']),'amp_state':ck['grad_scaler_state'],
 'logging_lengths':{k:len(v) for k,v in logging.items() if isinstance(v,(list,tuple))},
 'last_saved_lr':ck['optimizer_state']['param_groups'][0]['lr'],'next_epoch_lr':optimizer.param_groups[0]['lr'],
 'ema_at_checkpoint':logging['ema_fg_dice'][-1],'model_contract':{'plans':ck['init_args']['plans'],'dataset_json':ck['init_args']['dataset_json']},
 'init_args_keys':list(ck['init_args']),'saved_mirroring_axes':ck.get('inference_allowed_mirroring_axes')}
del ck
logs=sorted(fold.glob('training_log_*.txt'),key=lambda p:p.stat().st_mtime)
text=logs[-1].read_text(errors='replace') if logs else ''
epochs=re.findall(r'(?m)^.*Epoch\s+(\d+)\s*$',text)
report['latest_log_epoch']=int(epochs[-1]) if epochs else None
report['training_tail']=text[-3000:]
report['checkpoint_names']=[p.name for p in sorted(fold.glob('checkpoint_*.pth'))]
report['critical_errors']=[]
for p in sorted((source/'logs').glob('*3827543*')):
 if p.is_file():
  t=p.read_text(errors='replace'); report['critical_errors'] += [l[-400:] for l in t.splitlines() if re.search(r'Traceback|CUDA error|out.of.memory|ImportError|nonfinite',l,re.I)]
  report.setdefault('logs',{})[p.name]=t[-2000:]
import nnunetv2.run.run_training as runmod
runpath=pathlib.Path(inspect.getsourcefile(runmod))
paths=[plan,cfg,source/'slurm/01_train_k1_175_r3.slurm',source/'evidence/K1_175_TRAINING_STARTED.json',source/'evidence/K1_175_TRAINING_AUTHORIZATION.json',
 repo/'custom_nnunet/nnUNetTrainerBraTS2026RC.py',repo/f'custom_nnunet/{trainer}.py',repo/'custom_nnunet/met_aug_paired_training.py',repo/'custom_nnunet/met_aug_transform.py',runpath]
for p in paths:
 b=p.read_bytes(); rel=str(p.relative_to(source)) if source in p.parents else 'installed/run_training.py'
 report['files'][rel]={'base64':base64.b64encode(b).decode(),'sha256':hashlib.sha256(b).hexdigest()}
report['status']='pass'
print(json.dumps(report,allow_nan=False))
'''


def main():
    spec=importlib.util.spec_from_file_location('k1245_access',ACCESS)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    access=mod.Access('k1_resume_source_inspection_20261002')
    started=datetime.now(timezone.utc).isoformat()
    try:
        access.start_vpn()
        raw=access.ssh('env PYTHONDONTWRITEBYTECODE=1 MKL_THREADING_LAYER=GNU '+mod.PYTHON+' -B -c '+shlex.quote(REMOTE_CODE),timeout=180)
        report=json.loads(raw[raw.find('{'):]); files=report.pop('files')
        for rel,entry in files.items():
            import hashlib
            b=base64.b64decode(entry['base64']); assert hashlib.sha256(b).hexdigest()==entry['sha256']
            target=access.audit/'pulled'/rel; target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f: f.write(b)
        report['local_utc']=datetime.now(timezone.utc).isoformat()
        mod.exclusive(access.audit/'SOURCE_CHECKPOINT_INSPECTION.json',report)
        print(json.dumps({'audit':str(access.audit),'sacct':report['sacct'],'checkpoint':{k:v for k,v in report['checkpoint'].items() if k!='model_contract'},'latest_log_epoch':report['latest_log_epoch'],'critical_errors':report['critical_errors']},ensure_ascii=False))
    finally:
        access.close_vpn()
        mod.exclusive(access.audit/'LOCAL_OPERATION_RECEIPT.json',dict(mod.META,local_started_utc=started,local_completed_utc=datetime.now(timezone.utc).isoformat(),vpn_closed=access.owns_vpn,preexisting_vpn_left_unchanged=not access.owns_vpn))


if __name__=='__main__': main()
