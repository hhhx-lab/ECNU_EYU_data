"""Freeze and deploy once, then audit one held comparison DAG."""
import ast
import base64
import importlib.util
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[4]
spec=importlib.util.spec_from_file_location('shared_launch_access',PROJECT/'work_space/G1/03_experiments/03_diffusion_training/20261020_raw_real_modality_specific_1mm_r2/operator/common.py')
access=importlib.util.module_from_spec(spec);spec.loader.exec_module(access);access.m.ROOT=ROOT
m=access.m


def main():
    assert not (ROOT/'PACKAGE_MANIFEST.json').exists(),'Frozen launch cannot be retried'
    tests=subprocess.run(['conda','run','--no-capture-output','-n','g1_t2w_bbdm','python','-B','-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],
        env={**os.environ,'SHARED_LOCAL_MODEL_TEST':'1','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=180)
    count=re.search(r'Ran (\d+) tests',tests.stderr)
    assert tests.returncode==0 and count and int(count[1])>=12,tests.stderr
    m.exclusive(ROOT/'LOCAL_TESTS.json',dict(m.META,status='pass',test_count=int(count[1]),stdout=tests.stdout,stderr=tests.stderr,scope='Eight trusted strict CPU model loads/forward and synthetic metrics; not real A100 quality'))
    files=[ROOT/'README.md',ROOT/'PROTOCOL.md',ROOT/'RUN.json',ROOT/'LOCAL_TESTS.json']
    for folder in ('runtime','tests','slurm','operator'):
        for p in (ROOT/folder).glob('*'):
            if p.is_file() and p.suffix in ('.py','.slurm'):
                if p.suffix=='.py':ast.parse(p.read_text())
                if p.suffix=='.slurm':subprocess.run(['bash','-n',str(p)],check=True,timeout=10)
                files.append(p)
    manifest=dict(m.META,files={str(p.relative_to(ROOT)):m.sha(p) for p in files},authorization='2026-10-21 user uniform old/new evaluation',
        no_automatic_retry=True,frozen_before_new_shared_scores=True,prior_incompatible_scores_already_known=True)
    m.exclusive(ROOT/'PACKAGE_MANIFEST.json',manifest);files.append(ROOT/'PACKAGE_MANIFEST.json')
    payload={str(p.relative_to(ROOT)):base64.b64encode(p.read_bytes()).decode() for p in files}
    payload_file=ROOT/'operator/DEPLOY_PAYLOAD.json'
    m.exclusive(payload_file,payload)
    remote=json.loads((ROOT/'RUN.json').read_text())['remote_root'];cfg=json.loads((ROOT/'RUN.json').read_text())
    def deploy(a):
        reserve='import pathlib;r=pathlib.Path('+repr(remote)+');r.mkdir(mode=0o700,exist_ok=False);(r/"logs").mkdir();(r/"evidence").mkdir();print("RESERVED")'
        a.ssh(m.PYTHON+' -B -c '+shlex.quote(reserve),timeout=45)
        a.scp(payload_file,remote+'/evidence/DEPLOY_PAYLOAD.json')
        code='''import pathlib,json,base64,os,hashlib
r=pathlib.Path(__ROOT__);p=r/'evidence/DEPLOY_PAYLOAD.json'
assert hashlib.sha256(p.read_bytes()).hexdigest()==__PAYLOAD_SHA__
os.chmod(p,0o440);files=json.loads(p.read_text())
for n,b in files.items():
 assert not pathlib.Path(n).is_absolute() and '..' not in pathlib.Path(n).parts
 p=r/n;p.parent.mkdir(parents=True,exist_ok=True)
 with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o440),'wb') as f:f.write(base64.b64decode(b))
assert hashlib.sha256((r/'PACKAGE_MANIFEST.json').read_bytes()).hexdigest()==__SHA__
for mod in ('t1c','t1n','t2w','t2f'):(r/'models/july150k'/mod).mkdir(parents=True)
print('DEPLOY_PASS')
'''.replace('__ROOT__',repr(remote)).replace('__PAYLOAD_SHA__',repr(m.sha(payload_file))).replace('__SHA__',repr(m.sha(ROOT/'PACKAGE_MANIFEST.json')))
        a.ssh(m.PYTHON+' -B -c '+shlex.quote(code),timeout=120)
        local=json.loads((ROOT/'LOCAL_MODELS.json').read_text())
        for mod,info in local['july150k'].items():
            destination=cfg['models']['july150k'][mod]['path']
            assert m.sha(Path(info['path']))==info['sha256']
            a.scp(Path(info['path']),destination)
    result,audit=access.run_remote('submit',
        'import os,runpy;os.environ["G1_PAIR_PACKAGE_SHA256"]='+repr(m.sha(ROOT/'PACKAGE_MANIFEST.json'))+';runpy.run_path('+repr(remote+'/operator/remote_launch.py')+',run_name="__main__")',timeout=480,setup=deploy)
    assert result['status']=='pass' and result['submission_status']=='released' and len(result['jobs'])==10 and len(result['released'])==10
    assert result['package_sha256']==m.sha(ROOT/'PACKAGE_MANIFEST.json')
    m.exclusive(audit/'LOCAL_SUBMISSION_ACCEPTANCE.json',dict(m.META,status='pass',utc=m.datetime.now(m.timezone.utc).isoformat(),
        jobs=result['jobs'],release_count=10,package_sha256=result['package_sha256'],remote_utc=result['remote_utc'],
        local_received_utc=result['local_received_utc'],note='Submitted/released only; no training or scoring completion inference'))


if __name__=='__main__':main()
