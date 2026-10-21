"""Explicit, bounded stage launch or one read-only status snapshot."""
import argparse
import ast
import base64
import json
import os
import shlex
import subprocess
from common import ROOT, REMOTE, m, transport


def freeze(access):
    assert not (ROOT/'PACKAGE_MANIFEST.json').exists(), 'Frozen smoke launch cannot be retried'
    test=subprocess.run(['conda','run','--no-capture-output','-n','g1_t2w_bbdm','python','-B','-m','unittest','discover','-s','tests','-v'],
            cwd=ROOT,capture_output=True,text=True,timeout=180,
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''})
    assert test.returncode==0 and 'Ran 0 tests' not in test.stderr,test.stdout+test.stderr
    m.exclusive(access.audit/'LOCAL_TESTS.json',dict(m.META,status='pass',stdout=test.stdout,stderr=test.stderr,
            scope='synthetic numeric and trusted complete final checkpoint CPU tests; not A100/quality pass'))
    files={}
    for folder in ('runtime','tests','config','slurm','operator'):
        for p in (ROOT/folder).glob('*'):
            if p.is_file() and p.suffix in ('.py','.json','.slurm'):
                if p.suffix=='.py':ast.parse(p.read_text())
                if p.suffix=='.slurm':
                    subprocess.run(['bash','-n',str(p)],check=True,timeout=10)
                    dry=subprocess.run(['bash','-u',str(p)],env={**os.environ,'G1_STATIC_DRY_RUN':'1'},capture_output=True,text=True,timeout=10)
                    assert dry.returncode==0 and 'G1_R9_EVAL_BASH_U_DRY_RUN_PASS' in dry.stdout
                files[str(p.relative_to(ROOT))]=p
    for name in ('README.md','PROTOCOL.md'):files[name]=ROOT/name
    m.exclusive(ROOT/'PACKAGE_MANIFEST.json',dict(m.META,training_package_sha256='80f432b454ea80cf3d4142cb87d194c7913c0eabd2da473b3821b06d10df63f6',
            files={n:m.sha(p) for n,p in files.items()},authorization='2026-10-18 user: generation quality evaluation and best model',
            frozen_before_candidate_scores=True,no_automatic_retry=True))
    files['PACKAGE_MANIFEST.json']=ROOT/'PACKAGE_MANIFEST.json'
    payload=access.audit/'PAYLOAD.json'
    m.exclusive(payload,dict(files_b64={n:base64.b64encode(p.read_bytes()).decode() for n,p in files.items()}))
    return payload


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('smoke','full','snapshot'));a=parser.parse_args()
    access=transport(a.action);started=m.datetime.now(m.timezone.utc).isoformat();result=None
    if a.action=='smoke':payload=freeze(access)
    elif a.action=='full':
        assert (ROOT/'PACKAGE_MANIFEST.json').exists()
        m.exclusive(ROOT/'FULL_LAUNCH_RESERVATION.json',dict(m.META,utc=started,no_retry=True))
    try:
        access.start_vpn()
        if a.action=='smoke':
            reserve='import pathlib; r=pathlib.Path('+repr(str(REMOTE))+');r.mkdir(mode=0o700,exist_ok=False);(r/"evidence").mkdir();(r/"logs").mkdir();print("EVAL_RESERVED")'
            access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+m.PYTHON+' -B -c '+shlex.quote(reserve),timeout=30)
            access.scp(payload,str(REMOTE/'evidence/PAYLOAD.json'))
            deploy='''import pathlib,json,base64,hashlib,os
r=pathlib.Path(__ROOT__);p=r/'evidence/PAYLOAD.json'
assert hashlib.sha256(p.read_bytes()).hexdigest()==__PAYLOAD_SHA__
for n,b in json.loads(p.read_text())['files_b64'].items():
 assert not pathlib.Path(n).is_absolute() and '..' not in pathlib.Path(n).parts
 target=r/n;target.parent.mkdir(parents=True,exist_ok=True)
 with os.fdopen(os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o440),'wb') as f:f.write(base64.b64decode(b))
assert hashlib.sha256((r/'PACKAGE_MANIFEST.json').read_bytes()).hexdigest()==__PACKAGE_SHA__
print('EVAL_DEPLOYED')
'''.replace('__ROOT__',repr(str(REMOTE))).replace('__PAYLOAD_SHA__',repr(m.sha(payload))).replace('__PACKAGE_SHA__',repr(m.sha(ROOT/'PACKAGE_MANIFEST.json')))
            access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+m.PYTHON+' -B -c '+shlex.quote(deploy),timeout=45)
        if a.action in ('smoke','full'):
            env='env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= R9_TEST_ROOT='+shlex.quote(json.loads((ROOT/'config/RUN.json').read_text())['training_root'])
            out=access.ssh(env+' '+m.PYTHON+' -B -u '+shlex.quote(str(REMOTE/'operator/remote_launch.py'))+' --stage '+a.action,timeout=480)
        else:
            out=access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+m.PYTHON+' -B '+shlex.quote(str(REMOTE/'operator/remote_snapshot.py')),timeout=120)
        result=json.loads(out.strip().splitlines()[-1]);result['local_utc']=m.datetime.now(m.timezone.utc).isoformat()
        m.exclusive(access.audit/('SNAPSHOT.json' if a.action=='snapshot' else 'SUBMISSION_RECEIPT.json'),result)
        print(json.dumps(dict(audit=str(access.audit),**{k:v for k,v in result.items() if k!='evidence'})))
        assert result.get('status')!='stopped', 'Stage stopped, preserve evidence; no retry'
    finally:
        access.close_vpn()
        m.exclusive(access.audit/'LOCAL_OPERATION_RECEIPT.json',dict(m.META,started_utc=started,
            local_completed_utc=m.datetime.now(m.timezone.utc).isoformat(),vpn_owned=access.owns_vpn,
            own_vpn_stop_completed=access.owns_vpn,preexisting_vpn_left_unchanged=not access.owns_vpn,
            result_status=result.get('status') if result else 'unconfirmed'))


if __name__=='__main__':main()
