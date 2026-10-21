"""Freeze independent R9, audit its inputs, and submit each dependency job once."""
import ast
import base64
import datetime
import json
import os
import shlex
import subprocess

from common import ROOT, REMOTE, R6_REMOTE, RECON, SOURCE_CODE, m, transport

ATTEMPT = 'resume_uniform_r9'
LOSS_POLICY = 'uniform_original_quantization_warning_v1'
LABEL_ALGORITHM = 'original_physical_spacing_v2_labelwise_threshold_no_case_exception'
R8_PACKAGE_SHA = 'a73f614ae36e2b237f8c65929faa5547aa191a9701bcc00e689e23bbb806a745'
R8_FAILED_SHA = 'fc254bde644be02374cf0714c0029bda8fd874fdc77a34c46bfd2ae94ca7482d'
R7_PACKAGE_SHA = '41c47c0516e9818c82319da2a370507ebc3fd773ef121f4baed786159e5a4158'


def main():
    assert not (ROOT / 'plan/PACKAGE_MANIFEST.json').exists(), 'Frozen attempt cannot be submitted again'
    assert not (ROOT / 'plan/EXECUTION_AUTHORIZATION.json').exists(), 'Reserved attempt cannot be retried'
    assert not (ROOT / 'plan/LOCAL_LABEL_EXCEPTION.json').exists(), 'R9 forbids case exceptions'
    assert (ROOT / 'README.md').is_file() and (ROOT / 'plan/USER_AUTHORIZATION_20261016.md').is_file()
    cfg = json.loads((ROOT / 'config/RUN.json').read_text())
    reuse = json.loads((ROOT / 'plan/REUSE_CANDIDATES.json').read_text())
    parent = REMOTE.parent / 'g1_diffusion_1mm_vae_epoch004_20261016_resume_localfix_r8'
    r7 = REMOTE.parent / 'g1_diffusion_1mm_vae_epoch004_20261015_support_domain_r7'
    assert cfg['attempt'] == ATTEMPT and cfg['remote_root'] == str(REMOTE)
    assert REMOTE.name == 'g1_diffusion_1mm_vae_epoch004_20261016_resume_uniform_r9'
    assert cfg['reuse_parent'] == str(parent) and cfg['reuse_parent_package_sha256'] == R8_PACKAGE_SHA
    assert cfg['recon'] == str(RECON) and cfg['source_code'] == str(SOURCE_CODE)
    assert cfg['loss_policy'] == LOSS_POLICY and cfg['label_resampling_algorithm'] == LABEL_ALGORITHM
    assert reuse['parents'] == {str(parent): R8_PACKAGE_SHA, str(r7): R7_PACKAGE_SHA}
    candidates = reuse['candidates']
    assert len(candidates) == len({r['case_id'] for r in candidates}) == 514
    assert {r['mode'] for r in candidates} == {'all_outputs_reference', 'label_only_rebuild'}
    assert [r['case_id'] for r in candidates if r['mode'] == 'label_only_rebuild'] == ['BraTS-MET-01110-002']
    for row in candidates:
        assert row['split'] in ('train', 'val')
        assert row['directory'] in [str(p / 'dataset/cases' / row['case_id']) for p in (parent, r7)]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    test = subprocess.run(['conda', 'run', '--no-capture-output', '-n', 'g1_t2w_bbdm', 'python', '-B',
                           '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), '-p', 'test*.py', '-v'],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert test.returncode == 0, test.stdout + test.stderr
    assert 'Ran 0 tests' not in test.stderr and list((ROOT / 'tests').rglob('test*.py')), 'Empty test discovery'
    assert {p.name for p in (ROOT / 'slurm').glob('*.slurm')} == {n + '.slurm' for n in ('prepare', 't1c', 't1n', 't2w', 't2f')}
    for p in (ROOT / 'slurm').glob('*.slurm'):
        subprocess.run(['bash', '-n', str(p)], check=True, timeout=10)
        dry = subprocess.run(['bash', '-u', str(p)], env={**env, 'G1_STATIC_DRY_RUN': '1'}, capture_output=True, text=True, timeout=10)
        assert dry.returncode == 0 and 'G1_R9_BASH_U_DRY_RUN_PASS' in dry.stdout
    upstream = json.loads((ROOT / 'plan/UPSTREAM_SOURCE_MANIFEST.json').read_text())
    assert len(upstream['files']) == 37
    assert all(m.sha(ROOT / 'runtime/upstream_source' / p) == v for p, v in upstream['files'].items())
    auth = {**m.META, 'status': 'execution_authorized_with_user_waiver', 'attempt': ATTEMPT,
            'user_request': '采用上述统一策略，同时修好训练入口，再接续。',
            'scope': 'one CPU preprocess and four afterok single-A100 training submissions',
            'support_contract': 'signal_statistics_vs_full_native_FOV_geometry_v1',
            'STRICT_ORIGINAL_GATE': False, 'NONZERO_COMPARISON_GATE': False,
            'independent_full_revalidation': 'waived_by_user_not_pass',
            'independent_manual_review': 'waived_by_user_not_pass',
            'independent_P2_smoke': 'waived_by_user_not_pass',
            'runtime_integrity_checks': 'mandatory_during_input_construction_and_consumption',
            'loss_policy': LOSS_POLICY, 'label_resampling_algorithm': LABEL_ALGORITHM,
            'parent_r8_package_sha256': R8_PACKAGE_SHA, 'parent_r8_failed_sha256': R8_FAILED_SHA,
            'reuse_parent': str(parent), 'reuse_parents': reuse['parents'],
            'parent_jobs': {'prepare': '4124867', 't1c': '4124868', 't1n': '4124869', 't2w': '4124870', 't2f': '4124871'},
            'reuse_candidate_count': 514, 'label_only_rebuild_count': 1, 'new_case_count': 412,
            'r6_package_sha256': 'b17c632444a3bc5af2fd1877ba6c444916b42275818c4e041235ac5eceae7d90',
            'r6_diagnostic_sha256': m.sha(ROOT / 'plan/R6_DIAGNOSTIC_COMPLETE.json'),
            'training_config_sha256': m.sha(ROOT / 'config/RUN.json'),
            'authorization_md_sha256': m.sha(ROOT / 'plan/USER_AUTHORIZATION_20261016.md'),
            'reuse_candidates_sha256': m.sha(ROOT / 'plan/REUSE_CANDIDATES.json'),
            'original_plan_sha256': m.sha(ROOT / 'plan/G1_EPOCH004_RECON_1MM_DIFFUSION_ECNU_PLAN_20261013_R1.md'),
            'membership_sha256': m.sha(ROOT / 'plan/splits_final_train_val_test_membership.csv')}
    access = transport('submit')
    m.exclusive(ROOT / 'plan/EXECUTION_AUTHORIZATION.json', auth)
    files = {'README.md': ROOT / 'README.md'}
    for name in ('submit.py', 'common.py'):
        files['operator/' + name] = ROOT / 'operator' / name
    for folder in ('runtime', 'tests', 'config', 'plan', 'slurm'):
        for p in (ROOT / folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                files[str(p.relative_to(ROOT))] = p
    for p in (ROOT / 'runtime').glob('*.py'):
        ast.parse(p.read_text())
    m.exclusive(ROOT / 'plan/PACKAGE_MANIFEST.json', {**m.META, 'attempt': ATTEMPT,
                'loss_policy': LOSS_POLICY, 'label_resampling_algorithm': LABEL_ALGORITHM,
                'files': {n: m.sha(p) for n, p in files.items()}, 'independent_acceptance': 'waived_by_user_not_pass'})
    files['plan/PACKAGE_MANIFEST.json'] = ROOT / 'plan/PACKAGE_MANIFEST.json'
    m.exclusive(access.audit / 'LOCAL_TESTS.json', {**m.META, 'status': 'pass', 'output': test.stdout + test.stderr,
                'discovery': 'tests/test*.py', 'timeout_seconds': 180,
                'scope': 'discovered synthetic CPU regressions including uniform quantization warnings and parent reference/loader; not full cohort/GPU quality'})
    payload = access.audit / 'PAYLOAD.json'
    m.exclusive(payload, {'files_b64': {n: base64.b64encode(p.read_bytes()).decode() for n, p in files.items()}})
    code = r'''
import ast,base64,datetime,hashlib,json,os,pathlib,py_compile,re,subprocess,tempfile,time,traceback
root=pathlib.Path(__ROOT__);parent=pathlib.Path(__PARENT__);r6=pathlib.Path(__R6__);recon=pathlib.Path(__RECON__)
meta={'artifact_status':'experimental_unvalidated','operator_approved':False,'formal_gate_status':'not_run_not_passed'}
deadline=time.monotonic()+540
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o440),'w') as f:
  json.dump(v,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def call(argv,timeout=30,env=None):
 remaining=deadline-time.monotonic()
 if remaining<=0:raise TimeoutError('R9 bounded operation deadline exceeded; no retry')
 r=subprocess.run(argv,capture_output=True,text=True,timeout=min(timeout,remaining),env=env,cwd=root)
 return {'argv':argv,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'utc':now()}
def checked(argv,timeout=30,env=None):
 r=call(argv,timeout,env)
 if r['returncode']:raise RuntimeError(json.dumps(r))
 return r
def fields(s):return dict(re.findall(r'([A-Za-z][A-Za-z0-9_/:]*)=([^\s]+)',s))
jobs={};stage='deploy'
try:
 payload=root/'evidence/PAYLOAD.json';assert sha(payload)==__PAYLOAD_SHA__
 for n,b in json.loads(payload.read_text())['files_b64'].items():
  p=root/n;p.parent.mkdir(parents=True,exist_ok=True)
  with os.fdopen(os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o440),'wb') as f:f.write(base64.b64decode(b))
 stage='static_preflight'
 package=json.loads((root/'plan/PACKAGE_MANIFEST.json').read_text())
 assert sha(root/'plan/PACKAGE_MANIFEST.json')==__PACKAGE_SHA__
 assert all(sha(root/p)==v for p,v in package['files'].items())
 auth=json.loads((root/'plan/EXECUTION_AUTHORIZATION.json').read_text())
 assert package['attempt']==auth['attempt']=='resume_uniform_r9' and auth['STRICT_ORIGINAL_GATE'] is False
 assert auth['NONZERO_COMPARISON_GATE'] is False and not (root/'plan/LOCAL_LABEL_EXCEPTION.json').exists()
 reuse=json.loads((root/'plan/REUSE_CANDIDATES.json').read_text())
 assert reuse['parents']==auth['reuse_parents']
 assert reuse['parents'][str(parent)]==auth['parent_r8_package_sha256']
 parent_audits={}
 for name,digest in auth['reuse_parents'].items():
  source=pathlib.Path(name);assert source.is_absolute() and source.resolve()==source
  assert sha(source/'plan/PACKAGE_MANIFEST.json')==digest
  pp=json.loads((source/'plan/PACKAGE_MANIFEST.json').read_text());assert pp['files']
  for n,v in pp['files'].items():
   assert not pathlib.Path(n).is_absolute() and (source/n).resolve().is_relative_to(source)
   assert sha(source/n)==v
  assert (source/'dataset/FAILED.json').is_file() and not (source/'dataset/COMPLETE.ok').exists()
  parent_audits[name]={'package_sha256':digest,'failed_sha256':sha(source/'dataset/FAILED.json')}
 assert parent_audits[str(parent)]['failed_sha256']==auth['parent_r8_failed_sha256']
 r7=parent.parent/'g1_diffusion_1mm_vae_epoch004_20261015_support_domain_r7'
 assert parent_audits[str(r7)]['failed_sha256']=='ad6368a7a7a4a7a469b0a78003ddd33f914db0a9a70b02c65eba9373e7e91a5d'
 assert sha(r6/'plan/PACKAGE_MANIFEST.json')==auth['r6_package_sha256']
 assert sha(r6/'scan/COMPLETE.ok')==auth['r6_diagnostic_sha256']==sha(root/'plan/R6_DIAGNOSTIC_COMPLETE.json')
 diagnostic=json.loads((root/'plan/R6_DIAGNOSTIC_COMPLETE.json').read_text())
 assert diagnostic['cases']==926 and diagnostic['identity_error_cases']==[] and diagnostic['STRICT_ORIGINAL_GATE'] is False
 for n,key in [('config/RUN.json','training_config_sha256'),('plan/USER_AUTHORIZATION_20261016.md','authorization_md_sha256'),('plan/splits_final_train_val_test_membership.csv','membership_sha256'),('plan/REUSE_CANDIDATES.json','reuse_candidates_sha256'),('plan/G1_EPOCH004_RECON_1MM_DIFFUSION_ECNU_PLAN_20261013_R1.md','original_plan_sha256')]:assert sha(root/n)==auth[key]
 cfg=json.loads((root/'config/RUN.json').read_text())
 assert cfg['attempt']==auth['attempt'] and cfg['remote_root']==str(root) and cfg['support_contract']==auth['support_contract']
 assert cfg['reuse_parent']==auth['reuse_parent']==str(parent) and cfg['reuse_parent_package_sha256']==auth['parent_r8_package_sha256']
 assert cfg['recon']==str(recon) and cfg['source_code']==__SOURCE__
 for k in ('loss_policy','label_resampling_algorithm'):assert cfg[k]==auth[k]==package[k]
 assert cfg['seed']==20261013 and cfg['training']['batch_size']==16 and cfg['training']['num_steps']==200000
 upstream=json.loads((root/'plan/UPSTREAM_SOURCE_MANIFEST.json').read_text())
 assert len(upstream['files'])==37 and all(sha(root/'runtime/upstream_source'/p)==v for p,v in upstream['files'].items())
 complete=json.loads((recon/'dataset/COMPLETE.ok').read_text())
 assert complete==json.loads((recon/'dataset/DATASET_VALIDATION.json').read_text())
 assert complete['status']=='pass' and complete['cases']==1030 and complete['images']==4120 and complete['labels']==1030
 assert not (recon/'dataset/FAILED.json').exists()
 assert sha(pathlib.Path(__SOURCE__)/'data/data_csv.csv')==cfg['source_csv_sha256']
 candidates=reuse['candidates'];assert len(candidates)==len({c['case_id'] for c in candidates})==514
 assert [c['case_id'] for c in candidates if c['mode']=='label_only_rebuild']==['BraTS-MET-01110-002']
 for c in candidates:
  directory=pathlib.Path(c['directory']);assert c['split'] in ('train','val')
  assert directory.parent.parent.parent.as_posix() in auth['reuse_parents']
  assert directory==directory.parent.parent.parent/'dataset/cases'/c['case_id'] and directory.resolve()==directory
  assert c['mode'] in ('all_outputs_reference','label_only_rebuild')
  assert sha(directory/'preprocess_complete.json')==c['marker_sha256']
 accounting=checked(['sacct','-n','-P','-j',','.join(auth['parent_jobs'].values()),'--format=JobID,State,ExitCode,Elapsed'])
 records={r.split('|')[0]:r.split('|')[1:4] for r in accounting['stdout'].splitlines() if len(r.split('|'))>=4}
 assert records['4124867'][:2]==['FAILED','1:0']
 assert all(records[j]==['CANCELLED','0:0','00:00:00'] for j in ('4124868','4124869','4124870','4124871'))
 queue=checked(['squeue','-h','-u','zqchen','-o','%i|%T|%j|%P|%r'])
 assert 'g1d1m9' not in queue['stdout'],'Duplicate R9 job'
 partitions={p:checked(['scontrol','show','partition',p,'-o']) for p in ('cpu_192G','a100')}
 suitable={}
 for name,result in partitions.items():
  f=fields(result['stdout']);assert f['State']=='UP' and f['ReqResv']=='NO'
  assert f['AllowAccounts']=='ALL' or 'zqchen' in f['AllowAccounts'].split(',')
  assert f['AllowGroups']=='ALL' and (f['AllowQos']=='ALL' or 'user_zqchen' in f['AllowQos'].split(','))
  names=checked(['scontrol','show','hostnames',f['Nodes']])['stdout'].splitlines()
  nodes=checked(['scontrol','show','node',','.join(names),'-o'])
  suitable[name]=[]
  for line in nodes['stdout'].splitlines():
   nf=fields(line);cap=int(nf['RealMemory'])-int(nf.get('MemSpecLimit','0'))
   valid=cap>=(92160 if name=='cpu_192G' else 65536) and int(nf['CPUTot'])>=8
   if name=='a100':valid=valid and 'gpu' in nf.get('Gres','').lower()
   if valid and not any(s in nf['State'] for s in ('DOWN','DRAIN','FAIL','NO_RESPOND','FUTURE')):suitable[name].append({'node':nf['NodeName'],'memory_mib':cap,'state':nf['State'],'gres':nf.get('Gres')})
  assert suitable[name],'No compatible node in '+name
 qos=checked(['sacctmgr','-n','-P','show','qos','where','name=user_zqchen,partition_a100','format=Name,MaxWall,MaxJobsPU,MaxSubmitPU'])
 env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',G1_STATIC_DRY_RUN='1')
 for p in (root/'slurm').glob('*.slurm'):
  checked(['bash','-n',str(p)]);assert 'G1_R9_BASH_U_DRY_RUN_PASS' in checked(['bash','-u',str(p)],env=env)['stdout']
 for p in (root/'runtime').glob('*.py'):
  ast.parse(p.read_text())
  with tempfile.TemporaryDirectory() as d:py_compile.compile(str(p),cfile=str(pathlib.Path(d)/'probe.pyc'),doraise=True)
 tests=checked([__PYTHON__,'-B','-m','unittest','discover','-s',str(root/'tests'),'-p','test*.py','-v'],timeout=180,env=env)
 assert 'Ran 0 tests' not in tests['stderr'] and list((root/'tests').rglob('test*.py'))
 probe="""import sys,json,pathlib,numpy as np,nibabel as nib
sys.path.insert(0,"runtime")
import prepare as p
c=json.load(open("config/RUN.json"));case="BraTS-MET-01132-002"
assert case in {r["source_case_id"] for r in p.members()}
path=pathlib.Path(c["recon"])/"dataset/cases"/case/(case+"-seg.nii.gz")
manifest=json.loads((path.parent/"case_manifest.json").read_text())
assert manifest["case_id"]==case and manifest["geometry"]=="native" and manifest["checkpoint_sha256"]==p.CHECKPOINT
assert p.sha(path)==manifest["files"]["seg"]["sha256"]==manifest["source_sha256"]["seg"]
im=nib.load(str(path));a=p.load_volume(path,im,label=True).astype(np.uint8)
s=np.asarray(im.header.get_zooms()[:3],float);n,f,o=p.physical_grid(a.shape,s,(1.,1.,1.))
b=p.resample_physical(a[None],n,f,o,s,(1.,1.,1.),is_seg=True)[0]
native=p.label_summary(a,s);target=p.label_summary(b,(1.,1.,1.));r=p.loss_record(native,target)
assert native["2"]["voxels"]==1 and np.isclose(native["2"]["volume_mm3"],.25,rtol=0,atol=1e-6)
assert target["2"]["voxels"]==0 and "2" in r["lost_classes"] and r["resolution_warning"] is True
assert r["local_label_exception"] is None and r["loss_policy"]==c["loss_policy"] and r["label_resampling_algorithm"]==c["label_resampling_algorithm"]
print(json.dumps(dict(case_id=case,native_label_sha256=p.sha(path),native_manifest_sha256=p.sha(path.parent/"case_manifest.json"),native=native,target=target,record=r,status="expected_quantization_warning",scope="one_native_seg_only_no_full_images_no_quality_pass")))"""
 actual_warning=checked([__PYTHON__,'-B','-c',probe],timeout=130,env=env)
 probe='import sys,json;sys.path.insert(0,"runtime");import prepare;cfg=json.loads((prepare.ROOT/"config/RUN.json").read_text());print(json.dumps(prepare.estimate(cfg)))'
 storage=json.loads(checked([__PYTHON__,'-B','-c',probe],timeout=120,env=env)['stdout'].strip().splitlines()[-1])
 assert storage['cases']==926 and storage['reused_candidates']==514 and storage['label_only_rebuilds']==1 and storage['new_cases']==412
 probe='import sys,json;sys.path.insert(0,"runtime/upstream_source");from src.networks.DiffusionNetwork import get_diffusion_network;from types import SimpleNamespace;c=json.load(open("config/RUN.json"));n=get_diffusion_network(SimpleNamespace(**c["training"]),n_steps=1000);p=sum(v.numel() for v in n.parameters());assert p==4475999;print(p)'
 imported=checked([__PYTHON__,'-B','-c',probe],timeout=90,env=env)
 static={**meta,'status':'pass','remote_utc':now(),'hostname':os.uname().nodename,'package_sha256':sha(root/'plan/PACKAGE_MANIFEST.json'),
  'queue':queue,'accounting':accounting,'reuse_parents':parent_audits,'partitions':partitions,'suitable_nodes':suitable,'qos':qos,'storage':storage,
  'account_disk':checked(['df','-B1',str(root)]),'public_disk':checked(['df','-h','/public']),
  'synthetic_cpu_tests':tests,'actual_one_case_quantization_warning':actual_warning,'network_import':imported,'independent_full_revalidation':'waived_by_user_not_pass',
  'loss_policy':auth['loss_policy'],'label_resampling_algorithm':auth['label_resampling_algorithm'],
  'scope':'engineering_only_storage_headers_and_one_native_seg_no_full_images_no_scientific_pass'}
 write(root/'evidence/STATIC_PREFLIGHT.json',static)
 for phase in ('prepare','t1c','t1n','t2w','t2f'):
  stage='submit_'+phase;script=root/'slurm'/(phase+'.slurm');name='g1d1m9p1' if phase=='prepare' else 'g1d1m9'+phase
  dependency=None if phase=='prepare' else 'afterok:'+jobs['prepare']
  write(root/'evidence'/(phase.upper()+'_RESERVATION.json'),{**meta,'utc':now(),'script_sha256':sha(script),'dependency':dependency})
  argv=['sbatch','--parsable','--hold','--export=NONE']
  if dependency:argv+=['--dependency='+dependency,'--kill-on-invalid-dep=yes']
  response=call(argv+[str(script)]);write(root/'evidence'/(phase.upper()+'_SUBMISSION_RESPONSE.json'),response)
  assert response['returncode']==0,'Unique submission rejected; no retry: '+response['stderr']
  job=response['stdout'].strip().split(';')[0];assert re.fullmatch(r'[0-9]+',job);jobs[phase]=job
  write(root/'evidence'/(phase.upper()+'_CANONICAL_JOB.json'),{**meta,'job_id':job,'script_sha256':sha(script),'dependency':dependency,'status':'held'})
  held=checked(['scontrol','show','job',job,'-o']);f=fields(held['stdout'])
  assert f['JobName']==name and f['JobState']=='PENDING' and f['Reason']=='JobHeldUser'
  assert f['Partition']==('cpu_192G' if phase=='prepare' else 'a100') and f['NumCPUs']=='8' and f['NumTasks']=='1'
  assert f['MinMemoryNode']==('90G' if phase=='prepare' else '64G')
  assert f['TimeLimit']==('1-00:00:00' if phase=='prepare' else '2-00:00:00')
  assert f['Command']==str(script) and f['StdOut']==str(root/'logs'/(name+'_'+job+'.out')) and f['StdErr']==str(root/'logs'/(name+'_'+job+'.err'))
  if dependency:assert dependency in f['Dependency'] and ('gpu:1' in f.get('TRES','') or 'gres/gpu=1' in f.get('TRES',''))
  else:assert f['Dependency']=='(null)'
  assert sha(script)==package['files']['slurm/'+phase+'.slurm']
  write(root/'evidence'/(phase.upper()+'_LAUNCH_AUDIT.json'),{**meta,'utc':now(),'status':'held_audited','job':held,'package_sha256':sha(root/'plan/PACKAGE_MANIFEST.json'),'dependency':dependency,'static_sha256':sha(root/'evidence/STATIC_PREFLIGHT.json')})
 # All jobs are audited before releasing any. Release the producer last.
 for phase in ('t1c','t1n','t2w','t2f','prepare'):
  stage='release_'+phase;r=call(['scontrol','release',jobs[phase]])
  write(root/'evidence'/(phase.upper()+'_RELEASE_RECEIPT.json'),r);assert r['returncode']==0,'Release rejected; no retry'
 receipt={**meta,'status':'all_five_released','jobs':jobs,'remote_root':str(root),'remote_utc':now(),'hostname':os.uname().nodename,
  'package_sha256':sha(root/'plan/PACKAGE_MANIFEST.json'),'storage':storage,
  'queue_after_release':checked(['squeue','-h','-j',','.join(jobs.values()),'-o','%i|%T|%j|%P|%r|%b|%M']),
  'STRICT_ORIGINAL_GATE':False,'NONZERO_COMPARISON_GATE':False,'independent_full_revalidation':'waived_by_user_not_pass'}
 write(root/'evidence/SUBMISSION_RECEIPT.json',receipt)
except BaseException as error:
 receipt={**meta,'status':'stopped','stage':stage,'jobs_already_held_or_released':jobs,'remote_utc':now(),'message':str(error),'traceback':traceback.format_exc()}
 write(root/'evidence/OPERATION_FAILED.json',receipt)
receipt['evidence']={p.name:json.loads(p.read_text()) for p in (root/'evidence').glob('*.json') if p.name!='PAYLOAD.json'}
print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(receipt),flags=re.I))
'''
    for key, value in {'__ROOT__': str(REMOTE), '__PARENT__': str(parent), '__R6__': str(R6_REMOTE),
                       '__RECON__': str(RECON), '__SOURCE__': str(SOURCE_CODE), '__PYTHON__': m.PYTHON,
                       '__PAYLOAD_SHA__': m.sha(payload), '__PACKAGE_SHA__': m.sha(ROOT / 'plan/PACKAGE_MANIFEST.json')}.items():
        code = code.replace(key, repr(value))
    compile(code, 'r9_unique_submit', 'exec')
    try:
        access.start_vpn()
        reserve = 'import pathlib;root=pathlib.Path(' + repr(str(REMOTE)) + ');root.mkdir(mode=0o700,exist_ok=False);(root/"evidence").mkdir();(root/"logs").mkdir();print("R9_RESERVED")'
        access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -c ' + shlex.quote(reserve), timeout=30)
        access.scp(payload, str(REMOTE / 'evidence/PAYLOAD.json'))
        out = access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -u -c ' + shlex.quote(code), timeout=570)
        receipt = json.loads(out.strip().splitlines()[-1])
        receipt['local_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit / 'SUBMISSION_RECEIPT.json', receipt)
        print(json.dumps({'audit': str(access.audit), **{k: v for k, v in receipt.items() if k != 'evidence'}}, ensure_ascii=False))
        if receipt['status'] == 'stopped':
            raise RuntimeError('R9 stopped; preserve evidence and do not retry')
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
