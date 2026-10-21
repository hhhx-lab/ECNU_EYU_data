"""Exclusive immutable deployment and one held submission per dependency-linked job."""
import base64
import datetime
import json
import shlex
import subprocess
from common import ROOT, REMOTE, PARENT_REMOTE, RECON, SOURCE_CODE, m, transport


def main():
    import sys
    complete_preflight = False
    assert not sys.argv[1:] or complete_preflight
    assert complete_preflight or not (ROOT / 'plan/PACKAGE_MANIFEST.json').exists(), 'Already frozen; no repeated submission'
    import os
    test_env = os.environ.copy()
    test_env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                    CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    test = subprocess.run(['conda', 'run', '--no-capture-output', '-n', 'g1_t2w_bbdm', 'python', '-B', '-u',
                           str(ROOT / 'tests/test_runtime.py'), '-v'], capture_output=True, text=True, timeout=300, env=test_env)
    assert test.returncode == 0, test.stdout + test.stderr
    sources = {'README.md': ROOT / 'README.md'}
    for folder in ('plan', 'config', 'runtime', 'tests', 'slurm'):
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path != ROOT / 'plan/PACKAGE_MANIFEST.json':
                sources[str(path.relative_to(ROOT))] = path
    for path in (ROOT / 'slurm').glob('*.slurm'):
        subprocess.run(['bash', '-n', str(path)], check=True, timeout=10)
        dry = subprocess.run(['bash', '-u', str(path)], env={'G1_STATIC_DRY_RUN': '1', 'PATH': '/usr/bin:/bin'},
                             capture_output=True, text=True, timeout=10)
        assert dry.returncode == 0 and 'G1_R5_BASH_U_DRY_RUN_PASS' in dry.stdout
        if path.stem != 'prepare':
            assert '--signal=B:USR1@300' in path.read_text(), 'Pre-timeout signal must target exec batch process'
    if complete_preflight:
        package = json.loads((ROOT / 'plan/PACKAGE_MANIFEST.json').read_text())
        assert package['files'] == {p: m.sha(f) for p, f in sources.items()}, 'Runtime package changed; cannot complete original preflight'
    else:
        m.exclusive(ROOT / 'plan/PACKAGE_MANIFEST.json', {**m.META, 'files': {p: m.sha(f) for p, f in sources.items()},
                    'acceptance': 'waived_by_user_not_pass', 'cpu_regression_tests': '7 pass', 'attempt':'manual_r5'})
    sources['plan/PACKAGE_MANIFEST.json'] = ROOT / 'plan/PACKAGE_MANIFEST.json'
    access = transport('complete_unexecuted_preflight' if complete_preflight else 'submit')
    payload = access.audit / 'PAYLOAD.json'
    m.exclusive(payload, {'files_b64': {p: base64.b64encode(f.read_bytes()).decode() for p, f in sources.items()}})
    m.exclusive(access.audit / 'LOCAL_TESTS.json', {**m.META, 'status': 'pass', 'output': test.stdout + test.stderr,
                 'scope': 'seven synthetic CPU tests and five bash dry-runs; not GPU or manual quality acceptance'})
    code = r'''
import ast,base64,datetime,hashlib,importlib.util,json,os,pathlib,py_compile,re,subprocess,sys,tempfile,time,traceback
root=pathlib.Path(__REMOTE__);meta={'artifact_status':'experimental_unvalidated','operator_approved':False,'formal_gate_status':'not_run_not_passed'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(path,p):
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o440)
 with os.fdopen(fd,'w') as h:json.dump(p,h,sort_keys=True,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
def cmd(args,timeout=30,env=None):
 r=subprocess.run(args,text=True,capture_output=True,timeout=timeout,env=env)
 if r.returncode:raise RuntimeError('Command rejected: '+repr(args)+': '+r.stderr[-2000:])
 return r.stdout
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def run_probe(name,argv,timeout=300):
 env=os.environ.copy()
 env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
  CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
 started=time.monotonic();out=root/'evidence'/(name+'.stdout.log');err=root/'evidence'/(name+'.stderr.log')
 with out.open('x') as oh,err.open('x') as eh:
  try:
   r=subprocess.run(argv,env=env,stdout=oh,stderr=eh,text=True,timeout=timeout,cwd=root)
   result={'returncode':r.returncode,'bounded_timeout':False}
  except subprocess.TimeoutExpired:
   result={'returncode':124,'bounded_timeout':True}
 record={**meta,**result,'utc':now(),'elapsed_seconds':time.monotonic()-started,'timeout_seconds':timeout,
  'stdout_sha256':sha(out),'stderr_sha256':sha(err),'thread_policy':'OMP/MKL/OPENBLAS/NUMEXPR=1; CPU only CUDA hidden'}
 write(root/'evidence'/(name+'.json'),record)
 if record['returncode']:raise RuntimeError(name+' failed: '+json.dumps(record)+': '+err.read_text()[-3000:])
 return out.read_text()
jobs={}
try:
 payload=root/'evidence'/__PAYLOAD_NAME__
 assert sha(payload)==__PAYLOAD_SHA__
 if __COMPLETE_PREFLIGHT__:
  failure=json.loads((root/'evidence/OPERATION_FAILED.json').read_text())
  assert failure['jobs_already_held_or_released']=={} and "'str' object is not callable" in failure['message']
  assert not list((root/'evidence').glob('*_CANONICAL_JOB.json')) and not (root/'evidence/STATIC_PREFLIGHT.json').exists()
  assert not (root/'evidence/CPU_REGRESSION.json').exists() and not (root/'evidence/CPU_REGRESSION.stdout.log').exists()
  write(root/'evidence/OPERATOR_REPAIR_AUTHORIZATION.json',{**meta,'utc':now(),
   'scope':'complete unexecuted preflight; no deployment overwrite; no prior sbatch',
   'user_authorization':'修复预检超时并建立独立新批次提交','preserved_failure_sha256':sha(root/'evidence/OPERATION_FAILED.json'),
   'fix':'rename shadowed phase function to run_probe'})
 else:
  for relative,data in json.loads(payload.read_text())['files_b64'].items():
   path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
   with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o440),'wb') as h:h.write(base64.b64decode(data))
 package=json.loads((root/'plan/PACKAGE_MANIFEST.json').read_text())
 assert all(sha(root/p)==v for p,v in package['files'].items())
 auth=json.loads((root/'plan/EXECUTION_AUTHORIZATION.json').read_text())
 assert auth['status']=='execution_authorized_with_user_waiver' and auth['operator_approved'] is False
 assert auth['attempt']=='manual_r5' and auth['retry_authorization']=='修复预检超时并建立独立新批次提交'
 parent=pathlib.Path(__PARENT__)
 assert sha(parent/'plan/PACKAGE_MANIFEST.json')==auth['immediate_parent_package_sha256']
 prior_failure=json.loads((parent/'evidence/REPAIR_OPERATION_FAILED.json').read_text())
 assert prior_failure['jobs_already_held_or_released']=={} and 'Insufficient measured input/checkpoint capacity' in prior_failure['message']
 write(root/'evidence/PARENT_R4_FAILURE_IDENTITY.json',{**meta,'utc':now(),'sha256':sha(parent/'evidence/REPAIR_OPERATION_FAILED.json'),
  'parent_package_sha256':sha(parent/'plan/PACKAGE_MANIFEST.json'),'jobs_already_submitted':[],'failure':prior_failure})
 assert not list((parent/'evidence').glob('*_CANONICAL_JOB.json'))
 assert auth['P0_STRICT_ACCEPTANCE'] is False and auth['independent_manual_review_performed'] is False
 assert sha(root/'plan/USER_WAIVER_20261014.md')==auth['waiver_sha256']
 assert sha(root/'config/RUN.json')==auth['training_config_sha256']
 assert sha(root/'plan/splits_final_train_val_test_membership.csv')==auth['membership_sha256']
 assert sha(root/'plan/corrected_label_overlay.csv')==auth['overlay_sha256']
 upstream=json.loads((root/'plan/UPSTREAM_SOURCE_MANIFEST.json').read_text())
 assert len(upstream['files'])==37 and all(sha(root/'runtime/upstream_source'/p)==v for p,v in upstream['files'].items())
 recon=pathlib.Path(__RECON__)
 complete=json.loads((recon/'dataset/COMPLETE.ok').read_text())
 assert complete==json.loads((recon/'dataset/DATASET_VALIDATION.json').read_text())
 assert complete['status']=='pass' and complete['cases']==1030 and complete['images']==4120 and complete['labels']==1030
 assert not (recon/'dataset/FAILED.json').exists()
 assert '3941042|COMPLETED|0:0' in cmd(['sacct','-n','-P','-j','3941042','--format=JobID,State,ExitCode'])
 assert sha(pathlib.Path(__SOURCE__)/'data/data_csv.csv')=='0d1765e12cdad7d8a5df57a5d2c46c79414ed99a2015a4d8fed57cd4c4fc836b'
 partitions={p:cmd(['scontrol','show','partition',p]) for p in ('cpu_96G','a100')}
 assert all('State=UP' in v and 'zqchen' in v for v in partitions.values())
 queue=cmd(['squeue','-h','-u','zqchen','-o','%i|%T|%j|%P|%r'])
 assert 'g1d1m5' not in queue, 'Existing matching job; refuse duplicate'
 for phase in ('prepare','t1c','t1n','t2w','t2f'):
  script=root/'slurm'/(phase+'.slurm')
  cmd(['bash','-n',str(script)])
  env=os.environ.copy();env['G1_STATIC_DRY_RUN']='1'
  assert 'G1_R5_BASH_U_DRY_RUN_PASS' in cmd(['bash','-u',str(script)],env=env)
 for path in (root/'runtime').glob('*.py'):
  ast.parse(path.read_text())
  with tempfile.TemporaryDirectory() as d:py_compile.compile(str(path),cfile=str(pathlib.Path(d)/'test.pyc'),doraise=True)
 tests=run_probe('CPU_REGRESSION',[__PYTHON__,'-B','-u','-X','faulthandler',str(root/'tests/test_runtime.py'),'-v'])
 probe="import sys,json,faulthandler;faulthandler.enable();faulthandler.dump_traceback_later(180,repeat=False);sys.path.insert(0,"+repr(str(root/'runtime'))+");import prepare;cfg=json.loads((prepare.ROOT/'config/RUN.json').read_text());print(json.dumps(prepare.estimate(cfg)),flush=True)"
 storage=json.loads(run_probe('STORAGE_HEADERS',[__PYTHON__,'-B','-u','-c',probe],timeout=420).strip().splitlines()[-1])
 probe="import sys,json;sys.path.insert(0,"+repr(str(root/'runtime/upstream_source'))+");from src.networks.DiffusionNetwork import get_diffusion_network;from types import SimpleNamespace;cfg=json.load(open("+repr(str(root/'config/RUN.json'))+"));n=get_diffusion_network(SimpleNamespace(**cfg['training']),n_steps=1000);p=sum(x.numel() for x in n.parameters());assert p==4475999;print(json.dumps({'parameters':p}),flush=True)"
 run_probe('NETWORK_IMPORT',[__PYTHON__,'-B','-u','-c',probe])
 static={**meta,'status':'pass','utc':now(),'hostname':os.uname().nodename,'package_sha256':sha(root/'plan/PACKAGE_MANIFEST.json'),
  'bash_n':'5 pass','bash_u':'5 pass','py_compile':'pass','real_import_and_CPU_tests':'7 pass','network_parameters':4475999,
  'storage_header_upper_bound':storage,'user_queue':queue,'partitions':partitions,'acceptance_status':'waived_by_user_not_pass',
  'training_contract':'4 independent A100, batch16, eager/nonDDP, each200000 successful updates',
  'preprocessing_contract':'926 train/val, authentic common mask, strict 1mm, test104 locked'}
 write(root/'evidence/STATIC_PREFLIGHT.json',static)
 for phase in ('prepare','t1c','t1n','t2w','t2f'):
  stage=phase.upper();script=root/'slurm'/(phase+'.slurm');name='g1d1m5p1' if phase=='prepare' else 'g1d1m5'+phase
  dependency=None if phase=='prepare' else 'afterok:'+jobs['prepare']
  write(root/'evidence'/(stage+'_RESERVATION.json'),{**meta,'utc':now(),'script_sha256':sha(script),'dependency':dependency})
  arguments=['sbatch','--parsable','--hold','--export=NONE']
  if dependency:arguments+=['--dependency='+dependency,'--kill-on-invalid-dep=yes']
  response=subprocess.run(arguments+[str(script)],capture_output=True,text=True,timeout=30)
  write(root/'evidence'/(stage+'_SUBMISSION_RESPONSE.json'),{**meta,'utc':now(),'returncode':response.returncode,'stdout':response.stdout,'stderr':response.stderr})
  if response.returncode:raise RuntimeError('Unique '+stage+' submission rejected: '+response.stderr[-2000:])
  job=response.stdout.strip().split(';')[0];assert re.fullmatch('[0-9]+',job)
  jobs[phase]=job
  write(root/'evidence'/(stage+'_CANONICAL_JOB.json'),{**meta,'status':'held','job_id':job,'script_sha256':sha(script),'dependency':dependency})
  held=cmd(['scontrol','show','job',job])
  assert 'JobState=PENDING' in held and 'Reason=JobHeldUser' in held and 'JobName='+name in held
  assert 'Partition='+('cpu_96G' if phase=='prepare' else 'a100') in held
  assert str(root/'logs'/ (name+'_'+job+'.out')) in held and str(root/'logs'/(name+'_'+job+'.err')) in held
  assert ('NumTasks=1' in held and 'NumCPUs=8' in held)
  if phase!='prepare':assert 'gpu:1' in held and ('afterok:'+jobs['prepare']) in held
  write(root/'evidence'/(stage+'_LAUNCH_AUDIT.json'),{**meta,'utc':now(),'status':'held_audited','job_id':job,
   'held_job':held,'dependency':dependency,'script_sha256':sha(script),'package_sha256':sha(root/'plan/PACKAGE_MANIFEST.json'),
   'static_preflight_sha256':sha(root/'evidence/STATIC_PREFLIGHT.json')})
  assert sha(script)==package['files']['slurm/'+phase+'.slurm']
 # Release only after all five canonical audits exist; partial QOS failure leaves accepted jobs held.
 for phase,job in jobs.items():
  cmd(['scontrol','release',job])
  write(root/'evidence'/(phase.upper()+'_RELEASE_RECEIPT.json'),{**meta,'utc':now(),'job_id':job,'status':'released'})
 receipt={**meta,'status':'all_five_released','jobs':jobs,'remote_root':str(root),'remote_utc':now(),'hostname':os.uname().nodename,
  'package_sha256':sha(root/'plan/PACKAGE_MANIFEST.json'),'storage':storage,'acceptance':'waived_by_user_not_pass',
  'queue_after_release':cmd(['squeue','-h','-j',','.join(jobs.values()),'-o','%i|%T|%j|%P|%r|%b'])}
 write(root/'evidence/SUBMISSION_RECEIPT.json',receipt)
 print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(receipt,sort_keys=True),flags=re.I))
except BaseException as error:
 write(root/'evidence'/__FAILURE_NAME__,{**meta,'utc':now(),'jobs_already_held_or_released':jobs,'message':str(error),'traceback':traceback.format_exc()})
 raise
'''
    payload_name = 'REPAIR_PAYLOAD.json' if complete_preflight else 'PAYLOAD.json'
    for key, value in {'__REMOTE__': str(REMOTE), '__PARENT__':str(PARENT_REMOTE), '__RECON__': str(RECON), '__SOURCE__': str(SOURCE_CODE),
                       '__PAYLOAD_SHA__': m.sha(payload), '__PYTHON__': m.PYTHON, '__PAYLOAD_NAME__':payload_name,
                       '__FAILURE_NAME__':'REPAIR_OPERATION_FAILED.json' if complete_preflight else 'OPERATION_FAILED.json'}.items():
        code = code.replace(key, repr(value))
    code = code.replace('__COMPLETE_PREFLIGHT__', repr(complete_preflight))
    compile(code, '<r5_remote_submit>', 'exec')
    try:
        access.start_vpn()
        reservation = ("import pathlib,os,json;root=pathlib.Path(" + repr(str(REMOTE)) + ");root.mkdir(mode=0o700,exist_ok=False);"
                       "(root/'evidence').mkdir();(root/'logs').mkdir();"
                       "h=os.fdopen(os.open(root/'evidence/ROOT_RESERVATION.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o440),'w');"
                       "json.dump(" + repr({**m.META, 'payload_sha256': m.sha(payload), 'local_audit': str(access.audit)}) + ",h);h.close();print('R5_ROOT_RESERVED')")
        if not complete_preflight:
            access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -c ' + shlex.quote(reservation), timeout=40)
        access.scp(payload, str(REMOTE / 'evidence' / payload_name))
        result = access.ssh('env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= ' + m.PYTHON + ' -B -u -c ' + shlex.quote(code), timeout=1100)
        receipt = json.loads(result.strip().splitlines()[-1])
        receipt['local_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit / 'SUBMISSION_RECEIPT.json', receipt)
        print(json.dumps({'audit': str(access.audit), **receipt}, ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
