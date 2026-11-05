"""One explicitly selected full103 held job and same-ID release; no retries."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

from session import session, access_module
from s0_slurm_fields_r2 import job_fields

ROOT = Path(__file__).resolve().parents[1]
MODELS = {'EXT2025_MET_GLI': ('full103_2025_r1', 's2ext25full'),
          'EXT2026_BIO_INTEL_FULL': ('full103_bio_r1', 's2ext26full')}
CONTROLLER = '/public/home/zqchen/.conda/envs/segmamba'
VERSIONS = {'torch': '2.6.0', 'nibabel': '5.4.2', 'numpy': '1.26.4'}
META = access_module.META


def encode(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def render_slurm(attempt, model, contract_sha):
    assert model in MODELS and Path(attempt).is_absolute()
    name = MODELS[model][1]
    lines = ['#!/bin/bash', '#SBATCH --job-name=' + name, '#SBATCH --partition=a100',
             '#SBATCH --account=zqchen', '#SBATCH --qos=user_zqchen', '#SBATCH --nodes=1',
             '#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=8', '#SBATCH --gres=gpu:1',
             '#SBATCH --mem=64G', '#SBATCH --time=48:00:00', '#SBATCH --no-requeue',
             '#SBATCH --export=NONE', '#SBATCH --output=' + attempt + '/logs/' + name + '_%j.out',
             '#SBATCH --error=' + attempt + '/logs/' + name + '_%j.err',
             'set -e', 'set -o pipefail', 'umask 077', 'source /etc/profile',
             'module load singularity-ce/4.1.1', 'set -u',
             'unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY',
             'export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 LC_ALL=C',
             'export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8',
             'export FULL_EXPECTED_CONTRACT_SHA256=' + shlex.quote(contract_sha),
             'cd ' + shlex.quote(attempt),
             'exec ' + shlex.quote(CONTROLLER + '/bin/python') + ' -B ' + shlex.quote(attempt + '/runtime/run_full_inference.py') +
             ' --attempt ' + shlex.quote(attempt) + ' --model ' + shlex.quote(model), '']
    return '\n'.join(lines)


def validate_held(text, identity, contract):
    fields = job_fields(text)
    attempt, model = contract['attempt'], contract['selected_model']
    name = MODELS[model][1]
    expected = dict(JobId=identity, JobName=name, JobState='PENDING', Reason='JobHeldUser', Priority='0',
                    Partition='a100', Account='zqchen', QOS='user_zqchen', NumCPUs='8', NumTasks='1',
                    Requeue='0', Command=attempt + '/slurm/full.slurm', WorkDir=attempt)
    expected['CPUs/Task'] = '8'
    for key, value in expected.items():
        assert fields.get(key) == value, 'full held identity drift: ' + key
    assert fields.get('UserId', '').split('(')[0] == 'zqchen', 'full job owner drift'
    assert fields.get('NumNodes') in ('1', '1-1'), 'full nodes drift'
    assert fields.get('MinMemoryNode') in ('64G', '65536M', '65536'), 'full memory drift'
    assert fields.get('TimeLimit') in ('2-00:00:00', '48:00:00'), 'full walltime drift'
    assert fields.get('Dependency') in ('(null)', 'None'), 'unexpected full dependency'
    gpu_entries = fields.get('TresPerNode', '').split(',')
    assert gpu_entries and set(gpu_entries) <= {'gpu:1', 'gres/gpu:1'}, 'full GPU specification drift'
    tres = dict(x.split('=',1) for x in fields.get('TRES',fields.get('ReqTRES','')).split(',') if '=' in x)
    assert tres.get('gres/gpu') == '1', 'full single GPU aggregate request drift'
    for key, extension in (('StdOut', 'out'), ('StdErr', 'err')):
        assert fields.get(key) in (attempt + '/logs/' + name + '_' + identity + '.' + extension,
                                  attempt + '/logs/' + name + '_%j.' + extension), 'full log identity drift'
    return fields


def payload_for(model, metadata_path):
    metadata = json.loads(Path(metadata_path).read_text())
    remote = os.environ['EXTERNAL_MET_REMOTE_ROOT']
    attempt = remote + '/attempts/' + MODELS[model][0]
    authorization = json.loads((ROOT / 'DIRECT_FULL103_AUTHORIZATION_20261106.json').read_text())
    assert authorization['create_two_full_inference_GPU_jobs_after_environment_preparation']
    source = metadata['author_identity']
    bio = metadata['bio_identity']
    contract = dict(META, attempt=attempt, remote_root=remote, selected_model=model, cases=103, patient_groups=69,
        stage='direct_full103_user_waived_smoke', smoke_gate='waived_by_user_not_pass',
        measured_runtime_peak_gate='waived_by_user_not_pass', separate_J01_J04='waived_by_user_not_pass',
        controller_environment_prefix=CONTROLLER, controller_versions=VERSIONS,
        backend=metadata['backend'], sif=metadata['sif'], author_revision=source['commit'],
        bio_folds={'A': [0, 1, 2, 3, 4], 'B': [1, 3]},
        bio_files=[x for m in ('A', 'B') for x in bio['models'][m]['files']] + [bio['brainiac']],
        bio_source_files=[dict(path=remote + '/assets/bio_source/' + n, bytes=v['bytes'], sha256=v['sha256']) for n,v in source['files'].items()],
        model_IDs=authorization['selected_models'], incremental_reserved_bytes=150 * 1024**3,
        reservation_not_measured_peak_pass=True, models_downloaded=False, images_returned_to_Mac=False,
        scientific_quality_pass=False, automatic_retry=False,
        image2025_scratch_path_adapter=dict(original='/input_tmp',temporary='/tmp/input_tmp',model_logic_changed=False),
        **{'2025_main_sha256': '7bbbe5c9be854df5a6df1db19bf5f1dd2daaa0348282c382a5d018ee6a5294f7'})
    if model == 'EXT2026_BIO_INTEL_FULL':
        # The independently accepted runtime must be supplied, never inferred from a directory.
        from accept_runtime_r5 import validate_result
        runtime = metadata['accepted_bio_runtime']
        validate_result(runtime, remote)
        acceptance = metadata['independent_runtime_acceptance']
        assert acceptance['status'] == 'independent_R5_environment_acceptance_not_GPU_or_quality_pass'
        assert acceptance['accepted_bio_runtime'] == runtime and len(acceptance['files']) == 5
        assert acceptance['isolated_CPU_imports']['versions'] == runtime['versions']['versions']
        prefix = runtime['environment']
        source_root = remote + '/assets/bio_source'
        contract.update(bio_environment_prefix=prefix, bio_runtime_acceptance=runtime,
            bio_runtime_identity_files=acceptance['files'],
            bio_mounts=[dict(host=prefix,container=prefix),dict(host=source_root,container=source_root),
                        dict(host=bio['nnunet_results'],container=bio['nnunet_results']),
                        dict(host=bio['brainiac']['path'],container=bio['brainiac']['path'])],
            bio_config=dict(source=source_root + '/src', model_A_dir=bio['models']['A']['directory'], model_B_dir=bio['models']['B']['directory']),
            bio_environment=dict(nnUNet_results=bio['nnunet_results'], nnUNet_raw='/tmp/nnUNet_raw',
                 nnUNet_preprocessed='/tmp/nnUNet_preprocessed', BRATS_WORK_DIR='/tmp/brats_work',
                 BRATS_INPUT_DIR='/input', BRATS_OUTPUT_DIR='/output', BRAINIAC_CKPT=bio['brainiac']['path'],
                 BRAINIAC_MODALITY_INDICES='0,1,2,3', BRAINIAC_FEATURE_CHANNELS='4', BRAINIAC_FREEZE='1',
                 BRATS_DISABLE_TTA='1', BRATS_ENABLE_PP='1', BRATS_STEP_SIZE='0.5',
                 BRATS_CHECKPOINT='checkpoint_final.pth', PATH=prefix + '/bin:/usr/bin:/bin'))
    files = {'contract/EXECUTION_CONTRACT.json': encode(contract),
             'contract/USER_AUTHORIZATION.json': encode(authorization)}
    project = ROOT.parents[4]
    data = project / 'work_space/S2/03_experiments/12_k1245_rc_aware/s2_met_aug_k1_selected103_official_met_20261008_r1/manifests'
    expected = {'FIXED103_REAL_T2W_CASES.csv': '86ba853dadc64ffc45357895ba0bd543bf5b6b14c0c83eb789f1e875d4aa2e7e',
                'INPUT_REFERENCE_SHA256.json': '5f52bfdc3a957d848063850460226a7c3d9c25f0926e827f728f21e525fbd9df'}
    for name, sha256 in expected.items():
        text = (data / name).read_bytes().decode()
        assert digest(text) == sha256, 'local fixed103 identity drift'
        files['manifests/' + name] = text
    for name in ('contract.py', 'full_contract.py', 'isolation_check.py', 'gpu_isolation.py',
                 'checked_2025.py', 'checked_bio.py', 'run_full_inference.py'):
        text = (ROOT / 'runtime' / name).read_text()
        ast.parse(text, feature_version=(3,10))
        files['runtime/' + name] = text
    script = render_slurm(attempt, model, digest(files['contract/EXECUTION_CONTRACT.json']))
    check = subprocess.run(['/bin/bash','-n'], input=script, text=True, capture_output=True)
    assert check.returncode == 0, 'local full Slurm syntax error'
    files['slurm/full.slurm'] = script
    files['contract/PACKAGE_SHA256.json'] = encode(dict(META, files={n:digest(t) for n,t in files.items()}))
    return dict(model=model, contract=contract, files=files)


REMOTE = r'''
import ast,csv,hashlib,io,json,os,re,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
payload=json.loads(Path(PAYLOAD_PATH).read_text())
assert hashlib.sha256(Path(PAYLOAD_PATH).read_bytes()).hexdigest()==PAYLOAD_SHA,'payload drift'
contract=payload['contract'];attempt=Path(contract['attempt']);root=Path(contract['remote_root'])
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
 path.chmod(0o440)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def command(argv):
 p=subprocess.run(argv,capture_output=True,text=True,timeout=60)
 assert len(p.stdout)+len(p.stderr)<131072,'command output budget'
 if p.returncode:
  write(attempt/('operator/COMMAND_FAILED_'+ACTION+'_'+stage+'.json'),dict(argv=argv,returncode=p.returncode,
       stdout=p.stdout,stderr=p.stderr,remote_utc=datetime.now(timezone.utc).isoformat()))
  raise RuntimeError('command failed ('+str(p.returncode)+'): '+repr(argv)+'; stderr='+p.stderr)
 return p.stdout
def static():
 for n,d in json.loads((attempt/'contract/PACKAGE_SHA256.json').read_text())['files'].items():assert sha(attempt/n)==d,'package drift: '+n
 sys.path.insert(0,str(attempt/'runtime'))
 from full_contract import verify_assets
 verify_assets(contract,contract['selected_model'])
 assert shutil.disk_usage(root).free>contract['incremental_reserved_bytes'],'fresh storage reservation unavailable'
def held(identity):
 raw=command(['scontrol','show','job',identity,'-o'])
 fields=validate_held(raw,identity,contract)
 return dict(raw_scontrol=raw,fields=fields)
stage='before_new_attempt'
try:
 if ACTION=='hold':
  assert not attempt.exists(),'full inference attempt already exists; do not retry'
  assert attempt.parent==root/'attempts' and payload['model']==contract['selected_model'],'attempt identity drift'
  attempt.mkdir(mode=0o700)
  stage='freeze_full103_package'
  for n,t in payload['files'].items():
   p=attempt/n;assert p.resolve().is_relative_to(attempt.resolve()),'package path escape'
   p.parent.mkdir(parents=True,exist_ok=True)
   with p.open('x') as f:f.write(t)
   p.chmod(0o550 if n.endswith('.slurm') else 0o440)
   if n.endswith('.py'):ast.parse(t,feature_version=(3,10))
  for n in ('logs','operator','reports','scratch','predictions'):(attempt/n).mkdir(mode=0o700)
  static()
  command(['/bin/bash','-n',str(attempt/'slurm/full.slurm')])
  probe="import importlib.metadata as m,json; import torch,nibabel,numpy; print(json.dumps({n:m.version(n) for n in "+repr(contract['controller_versions'])+"}))"
  observed=json.loads(command([contract['controller_environment_prefix']+'/bin/python','-B','-c',probe]))
  assert observed==contract['controller_versions'],'controller imports/versions drift'
  bootstrap="import json,torch,numpy; from pathlib import Path; assert not Path('/public/home/zqchen/.ssh').exists(); print(json.dumps(dict(torch=torch.__version__,numpy=numpy.__version__,models_loaded=False,images_read=False,GPU_smoke='waived_by_user_not_pass')))"
  bootstrap_output=command([contract['backend'],'exec','--cleanenv','--containall','--no-home','--no-mount','hostfs,bind-paths','--net','--network','none','--pwd','/tmp',contract['sif']['path'],'python3','-B','-c',bootstrap])
  write(attempt/'operator/IMAGE_CPU_IMPORT_CHECK.json',dict(stdout=bootstrap_output,scope='CPU_imports_no_model_no_GPU_smoke'))
  stage='fresh_scheduling_and_capacity'
  partition=job_fields(command(['scontrol','show','partition','a100','-o']))
  assert partition['State']=='UP' and ('zqchen' in partition['AllowAccounts'].split(',') or partition['AllowAccounts']=='ALL'),'partition account unavailable'
  assoc=list(csv.DictReader(io.StringIO(command(['sacctmgr','-P','show','assoc','where','user=zqchen','format=User,Account,Partition,QOS,MaxJobs,MaxSubmitJobs'])),delimiter='|'))
  assert any(x['Account']=='zqchen' and 'user_zqchen' in x['QOS'].split(',') for x in assoc),'account QoS drift'
  qos=list(csv.DictReader(io.StringIO(command(['sacctmgr','-P','show','qos'])),delimiter='|'))
  selected=[x for x in qos if x['Name'] in ('user_zqchen','partition_a100')]
  assert len(selected)==2,'applicable QoS missing'
  queue=command(['squeue','-h','-u','zqchen','-o','%i|%T|%P|%j|%r|%b'])
  jobs=[x.split('|') for x in queue.splitlines() if x]
  assert not any(x[3] in ('s2ext25full','s2ext26full') and x[3]==JOB_NAME for x in jobs),'full inference already queued'
  usage=sum(int(job_fields(command(['scontrol','show','job',x[0],'-o']))['NumCPUs']) for x in jobs)
  for q in selected:
   limit=q.get('MaxTRESPU','')
   if limit:
    assert re.fullmatch(r'cpu=\d+',limit) and usage+8<=int(limit.split('=')[1]),'CPU QoS capacity insufficient'
   for key in ('MaxJobsPU','MaxSubmitPU','MaxSubmitJobsPU'):
    if q.get(key):assert len(jobs)+1<=int(q[key]),'QoS job limit insufficient'
  resource=dict(partition=partition,qos=selected,association=assoc,own_queue=queue,
       prospective_CPU_upper_bound=usage+8,free_bytes=shutil.disk_usage(root).free,
       quota_stdout=command(['quota','-s']),remote_utc=datetime.now(timezone.utc).isoformat(),
       reserved_bytes=contract['incremental_reserved_bytes'],measured_runtime_peak_gate='waived_by_user_not_pass')
  write(attempt/'operator/RESOURCE_SNAPSHOT.json',resource)
  write(attempt/'operator/SUBMISSION_RESERVED.json',dict(model=payload['model'],contract_sha256=sha(attempt/'contract/EXECUTION_CONTRACT.json')))
  stage='single_sbatch_held'
  raw=command(['sbatch','--hold','--parsable','--export=NONE','--chdir='+str(attempt),str(attempt/'slurm/full.slurm')])
  write(attempt/'operator/SBATCH_RESULT.json',dict(stdout=raw,returncode=0))
  match=re.fullmatch(r'(\d+)(?:;[\w.-]+)?',raw.strip());assert match,'ambiguous JobID; inspect only'
  identity=match.group(1)
  write(attempt/'operator/HELD_CREATED.json',dict(job_id=identity,model=payload['model']))
  stage='independent_held_audit'
  audit=held(identity);write(attempt/'operator/HELD_AUDIT.json',audit)
  mapping=dict(jobs={payload['model']:dict(job_id=identity)},contract_sha256=sha(attempt/'contract/EXECUTION_CONTRACT.json'),package_sha256=sha(attempt/'contract/PACKAGE_SHA256.json'))
  write(attempt/'operator/HELD_JOB_MAP.json',mapping)
  result=dict(status='full103_submitted_held_not_running',job_id=identity,model=payload['model'],held_audit=audit,
       held_audit_sha256=sha(attempt/'operator/HELD_AUDIT.json'),mapping=mapping,resource=resource,
       remote_utc=datetime.now(timezone.utc).isoformat())
 elif ACTION=='release':
  assert not (attempt/'operator/RELEASE_RESERVED.json').exists(),'release already attempted'
  created=json.loads((attempt/'operator/HELD_CREATED.json').read_text());identity=created['job_id']
  assert VERIFIED['job_id']==identity and VERIFIED['local_held_verified'],'local held verification missing'
  assert VERIFIED['held_audit_sha256']==sha(attempt/'operator/HELD_AUDIT.json'),'held audit drift'
  static();held(identity)
  write(attempt/'operator/LOCAL_HELD_VERIFICATION.json',VERIFIED)
  write(attempt/'operator/RELEASE_RESERVED.json',dict(job_id=identity))
  stage='release_same_ID_once'
  raw=command(['scontrol','release',identity]);write(attempt/'operator/RELEASE_RESULT.json',dict(job_id=identity,returncode=0,stdout=raw))
  observed=job_fields(command(['scontrol','show','job',identity,'-o']))
  result=dict(status='full103_released_not_completion_or_quality_pass',job_id=identity,model=payload['model'],
       observed_job=observed,release_returncode=0,contract_sha256=sha(attempt/'contract/EXECUTION_CONTRACT.json'),
       package_sha256=sha(attempt/'contract/PACKAGE_SHA256.json'),free_bytes=shutil.disk_usage(root).free,
       remote_utc=datetime.now(timezone.utc).isoformat(),smoke_gate='waived_by_user_not_pass')
  write(attempt/'operator/SUBMISSION_RECEIPT.json',result)
 else:raise RuntimeError('unknown direct submission action')
 print(json.dumps(result))
except BaseException as error:
 result=dict(status='direct_full103_submission_stopped',stage=stage,error_type=type(error).__name__,
       message=str(error),job_id=locals().get('identity'),remote_utc=datetime.now(timezone.utc).isoformat(),automatic_retry=False)
 if attempt.exists():write(attempt/('operator/SUBMISSION_FAILED_'+ACTION+'.json'),result)
 print(json.dumps(result))
'''


def code_for(action, payload_path, payload_sha, model, verified=None):
    import inspect
    code = 'MODELS=' + repr(MODELS) + '\n' + inspect.getsource(job_fields) + '\n' + inspect.getsource(validate_held)
    code = 'import re\n' + code
    code += '\nACTION=' + repr(action) + '\nPAYLOAD_PATH=' + repr(payload_path) + '\nPAYLOAD_SHA=' + repr(payload_sha)
    code += '\nJOB_NAME=' + repr(MODELS[model][1]) + '\nVERIFIED=' + repr(verified) + '\n' + REMOTE
    ast.parse(code, feature_version=(3,10))
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    args = parser.parse_args()
    payload = payload_for(args.model, args.metadata)
    with session('submit_direct_full103_' + args.model.lower()) as access:
        payload_path = access.audit / 'SUBMISSION_PAYLOAD.json'
        access_module.exclusive(payload_path, payload)
        remote_audit = os.environ['EXTERNAL_MET_REMOTE_ROOT'] + '/operator/' + access.audit.name
        access.ssh('mkdir -m 700 ' + shlex.quote(remote_audit))
        access.scp(payload_path, remote_audit + '/')
        remote_path = remote_audit + '/SUBMISSION_PAYLOAD.json'
        def invoke(action, verified=None):
            code = code_for(action, remote_path, access_module.sha(payload_path), args.model, verified)
            access_module.exclusive(access.audit / (action.upper() + '_EXECUTED_SOURCE.json'), dict(source=code,sha256=digest(code)))
            command = 'env LC_ALL=C PYTHONDONTWRITEBYTECODE=1 ' + shlex.quote(os.environ['ECNU_REMOTE_PYTHON']) + ' -B -c ' + shlex.quote(code)
            result = json.loads(access.ssh(command,timeout=900).strip().splitlines()[-1])
            access_module.exclusive(access.audit / (action.upper() + '_RESULT.json'), result)
            if result['status']=='direct_full103_submission_stopped':raise RuntimeError(encode(result))
            return result
        held = invoke('hold')
        fields = validate_held(held['held_audit']['raw_scontrol'], held['job_id'], payload['contract'])
        assert fields == held['held_audit']['fields'], 'held parsed fields drift'
        assert held['mapping']['contract_sha256'] == digest(payload['files']['contract/EXECUTION_CONTRACT.json'])
        assert held['mapping']['package_sha256'] == digest(payload['files']['contract/PACKAGE_SHA256.json'])
        assert digest(encode(held['held_audit'])) == held['held_audit_sha256'], 'held evidence transport drift'
        verified=dict(local_held_verified=True,job_id=held['job_id'],held_audit_sha256=held['held_audit_sha256'])
        access_module.exclusive(access.audit/'LOCAL_HELD_VERIFICATION.json',verified)
        released=invoke('release',verified)
        print(json.dumps(dict(audit=str(access.audit),result=released)),flush=True)


if __name__ == '__main__':
    main()
