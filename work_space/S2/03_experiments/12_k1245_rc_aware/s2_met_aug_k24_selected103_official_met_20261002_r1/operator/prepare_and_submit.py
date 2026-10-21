"""Prepare, verify, and submit each frozen K2/K4 MET job once."""

import argparse
import ast
import hashlib
import importlib.util
import json
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXPERIMENTS=ROOT.parent
S2=ROOT.parents[2]
SOURCE=S2/'03_experiments/07_screen125_official_met/s2_met_aug_btof_selected103_official_met_20260907_r4/runtime'
OLD=S2/'03_experiments/09_epoch175_official_met/s2_met_aug_bef_epoch175_official_met_20260910_r1'
ACCESS=EXPERIMENTS/'s2_met_aug_k1245_20260923_r3_fallback/operator/access.py'
ACCEPTANCE=EXPERIMENTS/'s2_met_aug_k1245_20260923_r3_fallback/operator/extend_k1_accept_k2_k4_20261002_20261002T032211Z/ACCEPTANCE_AND_EXTENSION.json'
REMOTE='/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/'+ROOT.name
META=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed')


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')


def prepare():
    for src,dest in (
        (SOURCE/'frozen_s2_inference.py',ROOT/'runtime/frozen_s2_inference.py'),
        (SOURCE/'run_official_met_eval.sh',ROOT/'runtime/run_official_met_eval.sh'),
        (SOURCE/'custom_nnunet/nnUNetTrainerBraTS2026RC_inference.py',ROOT/'runtime/custom_nnunet/nnUNetTrainerBraTS2026RC_inference.py'),
        (SOURCE/'custom_nnunet/__init__.py',ROOT/'runtime/custom_nnunet/__init__.py'),
        (OLD/'manifests/FIXED103_REAL_T2W_CASES.csv',ROOT/'manifests/FIXED103_REAL_T2W_CASES.csv')):
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists(): assert sha(dest)==sha(src)
        else: shutil.copyfile(src,dest)
    accepted=json.loads(ACCEPTANCE.read_text())
    contract=json.loads((OLD/'EPOCH175_OFFICIAL_MET_EXECUTION_PLAN_20260910_R1.json').read_text())['case_contract']
    routes={}
    for route,r in accepted['routes'].items():
        assert r['status']=='pass'
        routes[route]=dict(training_job_id=r['job_id'],training_root=r['root'],training_validation_sha256=r['training_validation_sha256'],
                           checkpoint=r['selected']['path'],checkpoint_sha256=r['selected']['sha256'],selected_epoch=r['selected']['epoch'],
                           selection_ema_fg_dice=r['selected']['ema_fg_dice'])
    plan=dict(META,experiment_id=ROOT.name,remote_root=REMOTE,routes=routes,case_contract=contract,
              training_split_root='/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1/inputs/splits',
              bootstrap=dict(replicates=20000,seed=20260831),exploratory_only=True,fixed103_for_selection_and_report=True,
              F_STRICT_PREREG_GATE=False,training_budget=175,preprocess_workers=0,export_workers=0,multiprocessing_queue_used=False,
              exclusive_node=False,job_names=dict(K2='s2k2met103r1',K4='s2k4met103r1'),
              acceptance_sha256=sha(ACCEPTANCE),inference_scope='selected checkpoint official MET; no online augmentation during inference',
              paired_status='await official MET outputs; no scientific gate inferred from training completion')
    write(ROOT/'EXECUTION_PLAN.json',plan)
    write(ROOT/'evidence/TRAINING_ACCEPTANCE_SOURCE.json',accepted)
    files={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and 'operator' not in p.relative_to(ROOT).parts}
    write(ROOT/'STATIC_SHA256.json',dict(META,files=files))
    for p in (ROOT/'runtime').rglob('*.py'): ast.parse(p.read_text(),str(p))
    subprocess.run(['bash','-n',str(ROOT/'slurm/evaluate.slurm')],check=True)
    print(json.dumps({'status':'prepared','files':len(files),'plan_sha256':sha(ROOT/'EXECUTION_PLAN.json')}))


def deploy():
    spec=importlib.util.spec_from_file_location('k1245_access',ACCESS)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    access=mod.Access('deploy_k24_met103_20261002')
    start=datetime.now(timezone.utc).isoformat()
    try:
        access.start_vpn()
        files=[p for p in ROOT.rglob('*') if p.is_file() and 'operator' not in p.relative_to(ROOT).parts]
        hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
        stage=access.audit/'transfer_bundle'
        stage.mkdir()
        for src in files:
            dest=stage/src.relative_to(ROOT)
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,dest)
        (stage/'logs').mkdir()
        access.ssh('test ! -e '+shlex.quote(REMOTE),timeout=60)
        access.call(['scp','-r','-P',access.env['HPC_LOGIN_PORT'],'-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',str(stage),
                     f"{access.env['HPC_SSH_USER']}@{access.env['HPC_LOGIN_HOST']}:{REMOTE}"],timeout=120)
        verify="import hashlib,pathlib; root=pathlib.Path("+repr(REMOTE)+"); expected="+repr(hashes)+"; assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in expected.items()); print('STATIC_SHA_PASS')"
        access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+mod.PYTHON+' -B -c '+shlex.quote(verify),timeout=60)
        command='env PYTHONDONTWRITEBYTECODE=1 MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=2 PYTHONPATH='+shlex.quote(REMOTE+'/runtime')+' '+mod.PYTHON+' -B '+shlex.quote(REMOTE+'/runtime/workflow.py')+' preflight'
        pre=access.ssh(command,timeout=240)
        mod.exclusive(access.audit/'PREFLIGHT_RECEIPT.json',dict(META,status='pass',remote_output_tail=pre[-4000:],plan_sha256=sha(ROOT/'EXECUTION_PLAN.json')))
        launch=r'''
import datetime as dt,hashlib,json,os,pathlib,subprocess
root=pathlib.Path(REMOTE_ROOT)
plan=json.loads((root/'EXECUTION_PLAN.json').read_text())
META=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed')
def emit(name,p):
 p={**META,'remote_utc':dt.datetime.now(dt.timezone.utc).isoformat(),**p}
 fd=os.open(root/'evidence'/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o440)
 with os.fdopen(fd,'w') as f: json.dump(p,f,indent=2,sort_keys=True); f.write('\n')
pre=json.loads((root/'evidence/REMOTE_STATIC_PREFLIGHT.json').read_text()); assert pre['status']=='pass'
assert not (root/'evidence/REMOTE_STATIC_PREFLIGHT_FAILED.json').exists()
jobs={}
for route in ('K2','K4'):
 name=plan['job_names'][route]
 q=subprocess.run(['squeue','-h','-n',name,'-o','%i'],capture_output=True,text=True,check=True)
 assert not q.stdout.strip(),'job already queued'
 assert not (root/f'evidence/{route}_RESERVATION.json').exists(),'route already attempted'
 emit(route+'_RESERVATION.json',dict(status='reserved',route=route,job_name=name,plan_sha256=hashlib.sha256((root/'EXECUTION_PLAN.json').read_bytes()).hexdigest()))
 r=subprocess.run(['sbatch','--hold','--parsable','--job-name='+name,str(root/'slurm/evaluate.slurm'),route],capture_output=True,text=True)
 if r.returncode:
  emit(route+'_SUBMISSION_FAILED.json',dict(status='failed',return_code=r.returncode,stderr=r.stderr))
  print(json.dumps(dict(status='submission_failed',route=route,jobs=jobs,stderr=r.stderr))); raise SystemExit(0)
 jid=r.stdout.strip().split(';')[0]; assert jid.isdigit()
 emit(route+'_LAUNCH.json',dict(status='submitted_held',route=route,job_id=jid,job_name=name))
 release=subprocess.run(['scontrol','release',jid],capture_output=True,text=True)
 if release.returncode:
  emit(route+'_RELEASE_FAILED.json',dict(status='failed',job_id=jid,stderr=release.stderr)); print(json.dumps(dict(status='release_failed',jobs=jobs,job_id=jid))); raise SystemExit(0)
 emit(route+'_RELEASED.json',dict(status='released',job_id=jid,route=route)); jobs[route]=jid
emit('LAUNCH_SUMMARY.json',dict(status='released',jobs=jobs))
print(json.dumps(dict(status='released',jobs=jobs)))
'''.replace('REMOTE_ROOT',repr(REMOTE))
        output=access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+mod.PYTHON+' -B -c '+shlex.quote(launch),timeout=120)
        result=json.loads(output[output.find('{'):]); mod.exclusive(access.audit/'LAUNCH_RECEIPT.json',dict(META,**result,remote_root=REMOTE))
        print(json.dumps(dict(audit=str(access.audit),**result)))
    finally:
        access.close_vpn()
        mod.exclusive(access.audit/'LOCAL_OPERATION_RECEIPT.json',dict(META,local_started_utc=start,local_completed_utc=datetime.now(timezone.utc).isoformat(),vpn_closed=access.owns_vpn,preexisting_vpn_left_unchanged=not access.owns_vpn))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=('prepare','deploy'))
    globals()[parser.parse_args().action]()
