"""Authorized installation-method fix in a new environment, preserving R1."""
import json
import os
from pathlib import Path
import shlex
from session import session,access_module

ROOT=Path(__file__).resolve().parents[1]
CODE=r'''
import hashlib,json,os,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
environment=Path(os.environ['EXTERNAL_MET_BIO_ENV'])
assert environment.parent==root/'envs' and environment.name=='bio_intel_py310_r2' and not environment.exists()
assert (root/'operator/ENV_INSTALL_FAILED.json').is_file()
assert json.loads((root/'operator/ENV_INSTALL_FAILED.json').read_text())['stage']=='torch_cu124_install'
audit=root/'operator/environment_r2';audit.mkdir(mode=0o700,exist_ok=False)
logs=root/'logs/environment_r2';logs.mkdir(mode=0o700,exist_ok=False)
free=shutil.disk_usage(root).free;assert free>100*1024**3
meta=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed')
def write(name,data):
    with (audit/name).open('x') as f:json.dump(data,f,indent=2);f.write('\n')
source=root/'assets/bio_source';identity=json.loads((root/'operator/BIO_SOURCE_IDENTITY.json').read_text())
assert identity['commit']=='83c73e0cac0b5a3c77a298547040ea3ea7f99854'
for name,item in identity['files'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==item['sha256']
write('STARTED.json',dict(meta,status='started',utc=datetime.now(timezone.utc).isoformat(),environment=str(environment),
    old_failure_preserved=True,free_bytes_before=free,only_install_method_changed=True))
os.environ.update(CONDA_PKGS_DIRS=str(root/'cache/conda'),PIP_CONFIG_FILE='/dev/null',PIP_DISABLE_PIP_VERSION_CHECK='1',
    PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
stage='conda_create'
def run(argv,name,timeout=1200,extra=None):
    env=dict(os.environ);env.update(extra or {})
    with (logs/name).open('x') as f:p=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=timeout)
    assert p.returncode==0,'install stage failed: '+name
try:
    run(['/public/software/apps/miniconda3/bin/conda','create','--prefix',str(environment),'--override-channels','-c','conda-forge',
        '--repodata-fn','current_repodata.json','-y','python=3.10','pip'],'conda_create.log',timeout=900)
    py=str(environment/'bin/python')
    constraints=audit/'constraints.txt'
    with constraints.open('x') as f:f.write('torch==2.6.0+cu124\ntorchvision==0.21.0+cu124\npillow==10.4.0\n')
    stage='torch_and_pillow_binary_install'
    run([py,'-B','-m','pip','install','--no-cache-dir','--only-binary=:all:','--index-url','https://pypi.org/simple',
         '--extra-index-url','https://download.pytorch.org/whl/cu124','-c',str(constraints),
         'torch==2.6.0+cu124','torchvision==0.21.0+cu124','pillow==10.4.0'],'torch_binary_install.log',timeout=1800)
    stage='author_dependencies'
    run([py,'-B','-m','pip','install','--no-cache-dir','--index-url','https://pypi.org/simple','-c',str(constraints),
         '-r',str(source/'requirements.txt'),'matplotlib==3.10.9'],'author_dependencies.log',timeout=1200)
    stage='trainer_registration'
    run([py,'-B',str(source/'docker/install_trainers.py')],'trainer_registration.log',timeout=180,
        extra={'TRAINER_SOURCE_DIR':str(source/'trainers')})
    stage='versions_and_pip_check'
    expected={'torch':'2.6.0','nnunetv2':'2.7.0','monai':'1.5.2','SimpleITK':'2.5.5','nibabel':'5.4.0','scipy':'1.15.2',
        'numpy':'2.2.6','pandas':'2.3.3','timm':'1.0.22','einops':'0.8.2','threadpoolctl':'3.6.0','matplotlib':'3.10.9','pillow':'10.4.0'}
    code='import importlib.metadata as m,json,torch; print(json.dumps(dict(versions={n:m.version(n) for n in '+repr(list(expected))+'},cuda_build=torch.version.cuda)))'
    observed=json.loads(subprocess.check_output([py,'-B','-c',code],text=True))
    assert all(observed['versions'][n].split('+')[0]==v for n,v in expected.items()) and observed['cuda_build']=='12.4'
    run([py,'-B','-m','pip','check'],'pip_check.log',timeout=60)
    freeze=subprocess.check_output([py,'-B','-m','pip','freeze','--all'],text=True)
    with (audit/'ENVIRONMENT_FREEZE.txt').open('x') as f:f.write(freeze)
    result=dict(meta,status='environment_r2_preparation_pass_not_model_runtime_pass',environment=str(environment),
        versions=observed,remote_utc=datetime.now(timezone.utc).isoformat(),free_bytes_after=shutil.disk_usage(root).free,
        jobs_created=False,old_environment_modified=False,model_loaded=False)
    write('COMPLETE.json',result);print(json.dumps(result))
except BaseException as error:
    result=dict(meta,status='environment_r2_preparation_failed',stage=stage,error_type=type(error).__name__,
        message=str(error) if isinstance(error,AssertionError) else 'bounded independent install failed',
        remote_utc=datetime.now(timezone.utc).isoformat(),no_automatic_retry=True,jobs_created=False)
    write('FAILED.json',result);print(json.dumps(result))
'''


def main():
    assert json.loads((ROOT/'ENVIRONMENT_R2_AUTHORIZATION.json').read_text())['preserve_original_install_failure']
    with session('install_environment_r2') as access:
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' EXTERNAL_MET_BIO_ENV='+shlex.quote(os.environ['EXTERNAL_MET_BIO_ENV'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(CODE)
        result=json.loads(access.ssh(command,timeout=4200).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'])),flush=True)
        if result['status']=='environment_r2_preparation_failed':raise RuntimeError('Environment R2 preparation failed; no automatic retry')


if __name__=='__main__':main()
