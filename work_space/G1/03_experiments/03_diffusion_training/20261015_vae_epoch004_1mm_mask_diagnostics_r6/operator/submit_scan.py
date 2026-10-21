"""One independent diagnostic CPU held submission; never submit training."""
import ast
import base64
import datetime
import json
import os
import shlex
import subprocess
from common import ROOT, REMOTE, PARENT, PARENT_REMOTE, m, transport


def main():
    access=transport('submit_scan')
    assert not (ROOT/'plan/PACKAGE_MANIFEST.json').exists(), 'Frozen scan cannot be resubmitted'
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    test=subprocess.run(['conda','run','--no-capture-output','-n','g1_t2w_bbdm','python','-B',str(ROOT/'tests/test_support_audit.py')],
                        env=env,capture_output=True,text=True,timeout=60)
    assert test.returncode==0,test.stderr
    script=ROOT/'slurm/scan.slurm'
    subprocess.run(['bash','-n',str(script)],check=True,timeout=10)
    dry=subprocess.run(['bash','-u',str(script)],env={'G1_STATIC_DRY_RUN':'1','PATH':'/usr/bin:/bin'},capture_output=True,text=True,timeout=10)
    assert dry.returncode==0 and 'G1_MASK_SCAN_R6_BASH_U_PASS' in dry.stdout
    sources={'README.md':ROOT/'README.md'}
    for folder in ('runtime','tests','slurm','config','plan'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:sources[str(p.relative_to(ROOT))]=p
    for p in (ROOT/'runtime').glob('*.py'):ast.parse(p.read_text())
    auth={**m.META,'status':'diagnostic_scan_authorized','scope':'926 train/val authentic support coverage scan only',
          'user_authorization':'先集中检查全部926例，汇总覆盖异常；三模态都无有效影像不能膨胀mask假造支持',
          'training_jobs_to_submit':0,'resource':{'partition':'cpu_192G','cpus':8,'memory':'90G','walltime':'24:00:00'},
          'known_case_diagnostic_sha256':m.sha(PARENT/'operator/diagnose_mask_case_20261015T091414Z/CASE_DIAGNOSTIC.json')}
    m.exclusive(ROOT/'plan/AUTHORIZATION.json',auth);sources['plan/AUTHORIZATION.json']=ROOT/'plan/AUTHORIZATION.json'
    package={**m.META,'files':{n:m.sha(p) for n,p in sources.items()},'scope':'diagnostic_no_source_repair','local_tests':'5 pass'}
    m.exclusive(ROOT/'plan/PACKAGE_MANIFEST.json',package);sources['plan/PACKAGE_MANIFEST.json']=ROOT/'plan/PACKAGE_MANIFEST.json'
    m.exclusive(access.audit/'LOCAL_TESTS.json',{**m.META,'status':'pass','output':test.stdout+test.stderr,'bash_n':'pass','bash_u':'pass'})
    payload=access.audit/'PAYLOAD.json'
    m.exclusive(payload,{'files_b64':{n:base64.b64encode(p.read_bytes()).decode() for n,p in sources.items()}})
    code=r'''
import ast,base64,datetime,hashlib,json,os,pathlib,py_compile,re,subprocess,tempfile,traceback
root=pathlib.Path(__ROOT__);parent=pathlib.Path(__PARENT__)
meta={'artifact_status':'experimental_unvalidated','operator_approved':False,'formal_gate_status':'not_run_not_passed'}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o440),'w') as h:
  json.dump(v,h,sort_keys=True,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
def fields(s):return dict(re.findall(r'([A-Za-z][A-Za-z0-9_/:]*)=([^\s]+)',s))
def call(a,timeout=30,env=None):
 r=subprocess.run(a,capture_output=True,text=True,timeout=timeout,env=env,cwd=root)
 return {'argv':a,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'utc':now()}
def checked(a,timeout=30,env=None):
 r=call(a,timeout,env);assert r['returncode']==0,json.dumps(r);return r
job=None;stage='deploy';mutations=[]
try:
 payload=root/'evidence/PAYLOAD.json';assert sha(payload)==__PAYLOAD_SHA__
 for n,b in json.loads(payload.read_text())['files_b64'].items():
  p=root/n;p.parent.mkdir(parents=True,exist_ok=True)
  with os.fdopen(os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o440),'wb') as h:h.write(base64.b64decode(b))
 stage='precheck'
 package=json.loads((root/'plan/PACKAGE_MANIFEST.json').read_text())
 assert sha(root/'plan/PACKAGE_MANIFEST.json')==__PACKAGE_SHA__
 assert all(sha(root/p)==v for p,v in package['files'].items())
 cfg=json.loads((root/'config/SCAN.json').read_text())
 assert sha(parent/'plan/PACKAGE_MANIFEST.json')==cfg['parent_package_sha256']
 pp=json.loads((parent/'plan/PACKAGE_MANIFEST.json').read_text())
 assert all(sha(parent/p)==v for p,v in pp['files'].items())
 assert (parent/'dataset/FAILED.json').exists() and not (parent/'dataset/COMPLETE.ok').exists()
 assert sha(pathlib.Path(cfg['source_code'])/'data/data_csv.csv')==cfg['source_csv_sha256']
 accounting=checked(['sacct','-n','-P','-j','4086739,4069916,4069917,4069918,4069919','--format=JobID,State,ExitCode,Elapsed'])
 assert '4086739|FAILED|1:0' in accounting['stdout']
 assert all(j+'|CANCELLED|0:0|00:00:00' in accounting['stdout'] for j in ('4069916','4069917','4069918','4069919'))
 queue=checked(['squeue','-h','-u','zqchen','-o','%i|%T|%j|%P|%r'])
 assert 'g1dmaskr6' not in queue['stdout']
 partition=checked(['scontrol','show','partition','cpu_192G','-o']);f=fields(partition['stdout'])
 assert f['State']=='UP' and f['ReqResv']=='NO' and f['MaxTime']=='UNLIMITED'
 assert f['AllowAccounts']=='ALL' or 'zqchen' in f['AllowAccounts'].split(',')
 assert f['AllowGroups']=='ALL' and f['AllowQos']=='ALL'
 names=checked(['scontrol','show','hostnames',f['Nodes']])['stdout'].splitlines()
 nodes=checked(['scontrol','show','node',','.join(names),'-o'])
 suitable=[]
 for line in nodes['stdout'].splitlines():
  nf=fields(line);cap=int(nf['RealMemory'])-int(nf.get('MemSpecLimit','0'))
  if cap>=92160 and int(nf['CPUTot'])>=8 and not any(s in nf['State'] for s in ('DOWN','DRAIN','FAIL','NO_RESPOND','FUTURE')):
   suitable.append({'node':nf['NodeName'],'allocatable_mib':cap,'state':nf['State']})
 assert suitable,'No unchanged-resource-capable CPU node'
 qos=checked(['sacctmgr','-n','-P','show','qos','where','name=user_zqchen,'+f['QoS'],
              'format=Name,MaxWall,MaxTRESPerJob,MaxTRESPerNode,MinTRESPerJob,MaxJobsPU,MaxSubmitPU'])
 qr=[s.split('|') for s in qos['stdout'].splitlines() if s.strip()]
 assert any(s[0].lower()=='user_zqchen' for s in qr) and all(not s[i] for s in qr for i in (1,2,3,4))
 available=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize;assert available>=1073741824
 disk=checked(['df','-B1',str(root)]);public=checked(['df','-h','/public'])
 script=root/'slurm/scan.slurm';checked(['bash','-n',str(script)])
 env=os.environ.copy();env.update(G1_STATIC_DRY_RUN='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
 assert 'G1_MASK_SCAN_R6_BASH_U_PASS' in checked(['bash','-u',str(script)],env=env)['stdout']
 for p in (root/'runtime').glob('*.py'):
  ast.parse(p.read_text())
  with tempfile.TemporaryDirectory() as d:py_compile.compile(str(p),cfile=str(pathlib.Path(d)/'probe.pyc'),doraise=True)
 tests=checked([__PYTHON__,'-B',str(root/'tests/test_support_audit.py')],timeout=90,env=env)
 import_probe=checked([__PYTHON__,'-B','-c','import sys;sys.path.insert(0,"runtime");import scan_coverage;assert scan_coverage.MODS==("t1n","t1c","t2f");print("REAL_IMPORT_PASS")'],timeout=90,env=env)
 static={**meta,'status':'pass','remote_utc':now(),'hostname':os.uname().nodename,'package_sha256':sha(root/'plan/PACKAGE_MANIFEST.json'),
  'script_sha256':sha(script),'parent_package_sha256':cfg['parent_package_sha256'],'accounting':accounting,'queue':queue,
  'partition':partition,'suitable_nodes':suitable,'qos':qos,'account_disk':disk,'public_disk':public,'available_bytes':available,
  'cpu_tests':tests,'real_import':import_probe,'scope':'926 diagnostic only; no NIfTI/checkpoint; no training'}
 write(root/'evidence/STATIC_PREFLIGHT.json',static)
 write(root/'evidence/SCAN_RESERVATION.json',{**meta,'status':'exclusive_reserved','static_sha256':sha(root/'evidence/STATIC_PREFLIGHT.json'),'script_sha256':sha(script),'utc':now()})
 stage='submit'
 response=call(['sbatch','--parsable','--hold','--export=NONE',str(script)]);mutations.append(response);write(root/'evidence/SUBMISSION_RESPONSE.json',response)
 assert response['returncode']==0,'Unique scan submission rejected; no retry'
 job=response['stdout'].strip().split(';')[0];assert re.fullmatch(r'[0-9]+',job)
 write(root/'evidence/CANONICAL_JOB.json',{**meta,'job_id':job,'status':'held','script_sha256':sha(script)})
 held=checked(['scontrol','show','job',job,'-o']);f=fields(held['stdout'])
 assert f['JobName']=='g1dmaskr6' and f['JobState']=='PENDING' and f['Reason']=='JobHeldUser'
 assert f['Partition']=='cpu_192G' and f['NumCPUs']=='8' and f['CPUs/Task']=='8' and f['MinMemoryNode']=='90G' and f['TimeLimit']=='1-00:00:00'
 assert f['Dependency']=='(null)' and f['NumTasks']=='1' and f['Account']=='zqchen'
 assert f['Command']==str(script) and f['StdOut']==str(root/'logs'/('g1dmaskr6_'+job+'.out')) and f['StdErr']==str(root/'logs'/('g1dmaskr6_'+job+'.err'))
 assert sha(script)==package['files']['slurm/scan.slurm']
 write(root/'evidence/HELD_LAUNCH_AUDIT.json',{**meta,'status':'held_audited','job':held,'package_sha256':sha(root/'plan/PACKAGE_MANIFEST.json'),'script_sha256':sha(script),'utc':now()})
 stage='release'
 response=call(['scontrol','release',job]);mutations.append(response);write(root/'evidence/RELEASE_RESPONSE.json',response)
 assert response['returncode']==0,'Release rejected; stop no retry'
 released=checked(['scontrol','show','job',job,'-o']);rf=fields(released['stdout'])
 assert rf['JobState'] in ('PENDING','RUNNING') and rf['Reason'] not in ('JobHeldAdmin','JobHeldUser','BadConstraints')
 queue=checked(['squeue','-h','-j',job,'-o','%i|%T|%j|%P|%r|%M'])
 receipt={**meta,'status':'scan_submitted_released','job_id':job,'remote_root':str(root),'remote_utc':now(),
  'package_sha256':sha(root/'plan/PACKAGE_MANIFEST.json'),'script_sha256':sha(script),'queue_after_release':queue,
  'released_job':released,'training_jobs_submitted':[],'evidence':{p.name:json.loads(p.read_text()) for p in (root/'evidence').glob('*.json') if p.name!='PAYLOAD.json'}}
 write(root/'evidence/SUBMISSION_RECEIPT.json',receipt)
 print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(receipt),flags=re.I))
except BaseException as error:
 failure={**meta,'status':'stopped','stage':stage,'job_id':job,'message':str(error),'traceback':traceback.format_exc(),'mutations':mutations,'remote_utc':now()}
 write(root/'evidence/OPERATION_FAILED.json',failure)
 print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(failure),flags=re.I))
'''
    for k,v in {'__ROOT__':str(REMOTE),'__PARENT__':str(PARENT_REMOTE),'__PAYLOAD_SHA__':m.sha(payload),
                '__PACKAGE_SHA__':m.sha(ROOT/'plan/PACKAGE_MANIFEST.json'),'__PYTHON__':m.PYTHON}.items():code=code.replace(k,repr(v))
    compile(code,'scan_submit','exec')
    try:
        access.start_vpn()
        reserve='import pathlib;root=pathlib.Path('+repr(str(REMOTE))+');root.mkdir(mode=0o700,exist_ok=False);(root/"evidence").mkdir();(root/"logs").mkdir();print("SCAN_ROOT_RESERVED")'
        access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+m.PYTHON+' -B -c '+shlex.quote(reserve),timeout=30)
        access.scp(payload,str(REMOTE/'evidence/PAYLOAD.json'))
        out=access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+m.PYTHON+' -B -u -c '+shlex.quote(code),timeout=420)
        receipt=json.loads(out.strip().splitlines()[-1]);receipt['local_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit/'SUBMISSION_RECEIPT.json',receipt)
        print(json.dumps({'audit':str(access.audit),**{k:v for k,v in receipt.items() if k!='evidence'}},ensure_ascii=False))
        if receipt['status']=='stopped':raise RuntimeError('Scan submission stopped; preserve evidence no retry')
    finally:
        access.close_vpn()


if __name__=='__main__':main()
