"""User-authorized connection retry; preserve the immutable comparison package."""
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
EXPECTED='23d63cf999aa28cda627181275dfc9ccd323cb9dc72d9aa0a9aeed296457b0de'
spec=importlib.util.spec_from_file_location('pair_original_launch',ROOT/'operator/launch.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
m=original.m


def validate():
    assert m.sha(ROOT/'PACKAGE_MANIFEST.json')==EXPECTED
    manifest=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text())
    assert len(manifest['files'])==27
    for name,h in manifest['files'].items():
        assert m.sha(ROOT/name)==h,name
    previous=json.loads((ROOT/'operator/submit_20261021T061053Z/OPERATION_STOPPED.json').read_text())
    assert previous['stage']=='vpn_start' and previous['remote_attempted'] is False
    payload=json.loads((ROOT/'operator/DEPLOY_PAYLOAD.json').read_text())
    assert set(payload)==set(manifest['files'])|{'PACKAGE_MANIFEST.json'}
    for name,encoded in payload.items():assert base64.b64decode(encoded)==(ROOT/name).read_bytes()
    tests=subprocess.run(['conda','run','--no-capture-output','-n','g1_t2w_bbdm','python','-B','-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],
        env={**os.environ,'SHARED_LOCAL_MODEL_TEST':'1','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
        capture_output=True,text=True,timeout=180)
    assert tests.returncode==0 and 'Ran 13 tests' in tests.stderr,tests.stderr
    for name in manifest['files']:
        p=ROOT/name
        if p.suffix=='.py':ast.parse(p.read_text())
        if p.suffix=='.slurm':subprocess.run(['bash','-n',str(p)],check=True,timeout=10)
    return tests


def deploy(a,c):
    remote=c['remote_root'];payload_file=ROOT/'operator/DEPLOY_PAYLOAD.json'
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
'''.replace('__ROOT__',repr(remote)).replace('__PAYLOAD_SHA__',repr(m.sha(payload_file))).replace('__SHA__',repr(EXPECTED))
    a.ssh(m.PYTHON+' -B -c '+shlex.quote(code),timeout=120)
    local=json.loads((ROOT/'LOCAL_MODELS.json').read_text())
    for mod,info in local['july150k'].items():
        assert m.sha(Path(info['path']))==info['sha256']
        a.scp(Path(info['path']),c['models']['july150k'][mod]['path'])


def sanitized_errors(a,value):
    lines=[line[:1000] for line in value.splitlines() if line.startswith('ERROR:')]
    result='\n'.join(lines)
    for key,val in a.env.items():
        if val and any(s in key.upper() for s in ('PASSWORD','TOKEN','SECRET','USERNAME')):result=result.replace(val,'[REDACTED]')
    return result


def main():
    tests=validate();c=json.loads((ROOT/'RUN.json').read_text());a=m.Access('retry_submit')
    started=m.datetime.now(m.timezone.utc).isoformat();stage='vpn_start';result=None;stopped=False
    m.exclusive(a.audit/'AUTHORIZATION_AND_LOCAL_TESTS.json',dict(m.META,utc=started,user_request='提交作业',
        package_sha256=EXPECTED,operator_sha256=m.sha(Path(__file__)),original_no_remote_attempt=True,
        test_count=13,stdout=tests.stdout,stderr=tests.stderr))
    try:
        start=subprocess.run([m.VPN,'start'],env=a.env,capture_output=True,text=True,timeout=100)
        flags=re.findall(r'VPN_SESSION_OWNED=([01])',start.stdout)
        m.exclusive(a.audit/'VPN_START_SUMMARY.json',dict(m.META,utc=m.datetime.now(m.timezone.utc).isoformat(),
            returncode=start.returncode,ownership_flags=flags,error=sanitized_errors(a,start.stderr)))
        assert len(flags)==1,'VPN ownership unconfirmed; no SSH'
        a.owns_vpn=flags[0]=='1';assert start.returncode==0,'VPN start failed; no SSH'
        stage='vpn_check';check=subprocess.run([m.VPN,'check'],env=a.env,capture_output=True,timeout=40)
        assert check.returncode==0,'VPN check failed; no SSH'
        stage='deployment';deploy(a,c)
        stage='remote_submission'
        code='import os,runpy;os.environ["G1_PAIR_PACKAGE_SHA256"]='+repr(EXPECTED)+';runpy.run_path('+repr(c['remote_root']+'/operator/remote_launch.py')+',run_name="__main__")'
        out=a.ssh('env PYTHONDONTWRITEBYTECODE=1 '+m.PYTHON+' -B -c '+shlex.quote(code),timeout=480)
        result=json.loads(out.strip().splitlines()[-1]);result['local_received_utc']=m.datetime.now(m.timezone.utc).isoformat()
        m.exclusive(a.audit/'RESULT.json',result)
        assert result['status']=='pass' and result['submission_status']=='released'
        assert result['package_sha256']==EXPECTED and len(result['jobs'])==len(result['released'])==10
        for phase in result['jobs']:
            assert result['evidence'][phase+'_RELEASE.json']['returncode']==0
            assert result['evidence'][phase+'_HELD_AUDIT.json']['status']=='held_audited'
        m.exclusive(a.audit/'LOCAL_SUBMISSION_ACCEPTANCE.json',dict(m.META,status='pass',utc=m.datetime.now(m.timezone.utc).isoformat(),
            jobs=result['jobs'],release_count=10,package_sha256=EXPECTED,remote_utc=result['remote_utc'],
            local_received_utc=result['local_received_utc'],note='Submission only; no completion or generation quality inference'))
        print(json.dumps(dict(audit=str(a.audit),status='released',jobs=result['jobs'],queue=result['queue'],remote_utc=result['remote_utc'])))
    except BaseException as error:
        m.exclusive(a.audit/'OPERATION_STOPPED.json',dict(m.META,utc=m.datetime.now(m.timezone.utc).isoformat(),
            stage=stage,message=str(error),remote_attempted=stage in ('deployment','remote_submission'),no_automatic_retry=True))
        raise
    finally:
        if a.owns_vpn:
            stop=subprocess.run([m.VPN,'stop'],env=a.env,capture_output=True,text=True,timeout=40)
            stopped=stop.returncode==0
        if not stopped:
            status=subprocess.run([m.VPN,'status'],env=a.env,capture_output=True,text=True,timeout=40)
            states=[json.loads(line[line.index('{'):]) for line in status.stdout.splitlines() if '{' in line]
            logged_out=status.returncode==0 and len(states)==1 and states[0].get('status')=='logout' and states[0].get('tunnel_status')=='tunnelUnavailable'
            stopped=logged_out if a.owns_vpn else False
            m.exclusive(a.audit/'INDEPENDENT_VPN_STATUS.json',dict(m.META,utc=m.datetime.now(m.timezone.utc).isoformat(),
                returncode=status.returncode,confirmed_logout=logged_out,ownership_confirmed=a.owns_vpn))
        m.exclusive(a.audit/'LOCAL_OPERATION_RECEIPT.json',dict(m.META,started_utc=started,
            completed_utc=m.datetime.now(m.timezone.utc).isoformat(),stage=stage,vpn_owned=a.owns_vpn,
            own_vpn_stop_completed=stopped,preexisting_vpn_preserved=not a.owns_vpn,result_status=result.get('status') if result else 'unconfirmed'))
        assert not a.owns_vpn or stopped,'Own VPN shutdown not confirmed'


if __name__=='__main__':main()
