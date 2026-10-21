"""Bounded K1 time extension and read-only K2/K4 training acceptance."""

import base64
import importlib.util
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path


REMOTE_CODE = r'''
import base64, datetime as dt, gc, hashlib, json, math, os, pathlib, subprocess
import torch
torch.set_num_threads(2)
META=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed')
base=pathlib.Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated')
TRAINER='nnUNetTrainerBraTS2026RCMetAugK1245FocalCompletionFineTune175'
def cmd(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=30)
    return dict(rc=p.returncode,out=p.stdout.strip(),err=p.stderr.strip())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(p.read_text())
def require(ok,msg):
    if not ok: raise RuntimeError(msg)
def finite(x):
    if torch.is_tensor(x): return bool(torch.isfinite(x).all())
    if isinstance(x,dict): return all(finite(v) for v in x.values())
    if isinstance(x,(list,tuple)): return all(finite(v) for v in x)
    if isinstance(x,float): return math.isfinite(x)
    return True
report={**META,'remote_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'hostname':os.uname().nodename,'disk':cmd(['df','-h','/public']),'routes':{},'files':{}}
k1=cmd(['scontrol','show','job','3827543','-o'])
report['k1_before']=k1
require(k1['rc']==0 and 'JobName=s2k1train175r3' in k1['out'],'K1 job identity unavailable')
require('UserId=zqchen(' in k1['out'],'K1 job ownership drift')
if 'JobState=RUNNING' in k1['out']:
    report['k1_extension_command']=cmd(['scontrol','update','JobId=3827543','TimeLimit=9-00:00:00'])
    report['k1_after']=cmd(['scontrol','show','job','3827543','-o'])
    report['k1_extension_confirmed']=report['k1_extension_command']['rc']==0 and 'TimeLimit=9-00:00:00' in report['k1_after']['out']
else:
    report['k1_extension_confirmed']=False
    report['k1_extension_reason']='job no longer RUNNING; no mutation attempted'
for route,jid,dirname,planname in (
    ('K2','3827555','s2_met_aug_k1245_20260929_k2_175_r2','K1245_K2_175_EXECUTION_PLAN_20260929_R2.json'),
    ('K4','3827580','s2_met_aug_k1245_20260929_k4_175_r6','K1245_K4_175_EXECUTION_PLAN_20260929_R6.json')):
    root=base/dirname
    schedule=cmd(['sacct','-n','-P','-X','-j',jid,'-o','JobIDRaw,State,ExitCode,Elapsed,Start,End'])
    require(schedule['rc']==0 and any(line.split('|')[:3]==[jid,'COMPLETED','0:0'] for line in schedule['out'].splitlines()),route+' scheduler not COMPLETED 0:0')
    require(not (root/f'evidence/{route}_175_TRAINING_FAILED.json').exists(),route+' FAILED marker')
    vp=root/f'evidence/{route}_175_TRAINING_VALIDATION.json'
    val=read(vp); plan=read(root/planname); started=read(root/f'evidence/{route}_175_TRAINING_STARTED.json')
    require(val['status']=='pass' and val['completed_epochs']==175 and val['job_id']==jid,route+' training validation drift')
    require((root/f'evidence/{route}_175_TRAINING_COMPLETE.ok').is_file(),route+' COMPLETE absent')
    require(started['job_id']==jid and started['route']==route and started['epochs']==175,route+' STARTED drift')
    require(started['plan_sha256']==sha(root/planname),route+' plan SHA drift')
    require(started['config_sha256']==sha(root/f'config/K1245_ROUTE_{route}.json'),route+' config SHA drift')
    require(started['trainer_sha256']==sha(root/f'runtime/repository/custom_nnunet/{TRAINER}.py'),route+' trainer SHA drift')
    require(plan['pretrained_checkpoint_sha256']=='4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267',route+' original E contract')
    fold=root/f'results/{route}/Dataset264_BraTS2026_MET_Completion/{TRAINER}__nnUNetPlans__3d_fullres/fold_0'
    recorded={r['epoch']:r for r in val['candidate_checkpoints']}; require(set(recorded)=={25,50,75,100,125,150,175},route+' candidate set')
    rows=[]; model_contract=None
    for epoch in (25,50,75,100,125,150,175):
        path=fold/f'checkpoint_epoch_{epoch:04d}.pth'; digest=sha(path)
        require(digest==recorded[epoch]['sha256'],route+' checkpoint SHA drift')
        ck=torch.load(path,map_location='cpu',weights_only=False)
        require(ck['current_epoch']==epoch and ck['trainer_name']==TRAINER,route+' checkpoint epoch/trainer drift')
        require(bool(ck.get('network_weights')) and bool(ck.get('optimizer_state')),route+' incomplete network/optimizer')
        require('grad_scaler_state' in ck and 'logging' in ck,route+' incomplete AMP/logging')
        require(finite(ck['network_weights']) and finite(ck['optimizer_state']) and finite(ck['grad_scaler_state']) and finite(ck['logging']),route+' nonfinite checkpoint')
        ema=ck['logging']['ema_fg_dice']; require(len(ema)>=epoch,route+' logging length')
        score=float(ema[epoch-1]); require(score==recorded[epoch]['ema_fg_dice'],route+' EMA mismatch')
        contract={'plans':ck['init_args']['plans'],'dataset_json':ck['init_args']['dataset_json']}
        if model_contract is None: model_contract=contract
        else: require(contract==model_contract,route+' candidate model contract drift')
        rows.append(dict(epoch=epoch,ema_fg_dice=score,sha256=digest,path=str(path),current_epoch=epoch,network_finite=True,optimizer_finite=True,amp_finite=True,logging_finite=True))
        del ck; gc.collect()
    best=sorted(rows,key=lambda r:(-r['ema_fg_dice'],r['epoch']))[0]
    require(all(best[k]==val['best_checkpoint'][k] for k in ('epoch','sha256','ema_fg_dice')),route+' selected checkpoint drift')
    labels=set()
    for x in model_contract['dataset_json']['labels'].values(): labels.update(x if isinstance(x,(tuple,list)) else [x])
    require(labels=={0,1,2,3,4},route+' dataset labels drift')
    report['routes'][route]={**META,'status':'pass','root':str(root),'job_id':jid,'scheduler':schedule,'training_validation_sha256':sha(vp),'completed_epochs':175,'candidates':rows,'selected':best,'model_contract':model_contract,'allowed_labels':sorted(labels)}
    paths=[root/planname,*sorted((root/'config').glob('*.json')),*sorted((root/'evidence').glob('*.json')),*sorted((root/'evidence').glob('*.ok')),*sorted((root/'slurm').glob('*.slurm')),*sorted((root/'logs').glob('*.out')),*sorted((root/'logs').glob('*.err')),*sorted(fold.glob('training_log_*.txt')),*sorted(fold.glob('summary.json'))]
    for p in paths:
        if p.stat().st_size>3000000: continue
        content=p.read_bytes(); rel=route+'/'+str(p.relative_to(root))
        report['files'][rel]={'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content),'base64':base64.b64encode(content).decode()}
    report['routes'][route]['nnunet_validation_mean_dice']=None
    for p in fold.glob('training_log_*.txt'):
        import re
        matches=re.findall(r'Mean Validation Dice:\s*([0-9.eE+-]+)',p.read_text())
        if matches: report['routes'][route]['nnunet_validation_mean_dice']=float(matches[-1])
report['status']='pass'
print(json.dumps(report,allow_nan=False))
'''


def main():
    spec = importlib.util.spec_from_file_location('k1245_access', Path(__file__).with_name('access.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    access = mod.Access('extend_k1_accept_k2_k4_20261002')
    started = datetime.now(timezone.utc).isoformat()
    try:
        access.start_vpn()
        raw = access.ssh('env MKL_THREADING_LAYER=GNU PYTHONDONTWRITEBYTECODE=1 ' + mod.PYTHON + ' -B -c ' + shlex.quote(REMOTE_CODE), timeout=240)
        data = json.loads(raw[raw.find('{'):])
        files = data.pop('files')
        index = {}
        for rel, item in files.items():
            target = access.audit / 'pulled' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            content = base64.b64decode(item['base64'])
            import hashlib
            assert hashlib.sha256(content).hexdigest() == item['sha256']
            with target.open('xb') as handle:
                handle.write(content)
            if target.suffix == '.json': json.loads(content)
            index[rel] = {k: item[k] for k in ('sha256', 'bytes')}
        data['local_utc'] = datetime.now(timezone.utc).isoformat()
        mod.exclusive(access.audit / 'ACCEPTANCE_AND_EXTENSION.json', data)
        mod.exclusive(access.audit / 'PULLED_SHA256.json', {**mod.META, 'status': 'pass', 'files': index})
        print(json.dumps({'audit':str(access.audit),'k1_extension_confirmed':data['k1_extension_confirmed'],'extension_response':data.get('k1_extension_command'),'routes':{k:{'selected':v['selected'],'nnunet_validation_mean_dice':v['nnunet_validation_mean_dice']} for k,v in data['routes'].items()},'pulled_files':len(index)},ensure_ascii=False))
    finally:
        access.close_vpn()
        mod.exclusive(access.audit / 'LOCAL_OPERATION_RECEIPT.json', {**mod.META,'action':'extend_k1_accept_k2_k4','local_started_utc':started,'local_completed_utc':datetime.now(timezone.utc).isoformat(),'vpn_closed':access.owns_vpn,'preexisting_vpn_left_unchanged':not access.owns_vpn})


if __name__ == '__main__':
    main()
