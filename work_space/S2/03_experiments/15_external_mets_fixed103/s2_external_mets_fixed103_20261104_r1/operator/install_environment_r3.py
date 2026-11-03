"""Authorized compatible-binary method, copied R3, original R2 left intact."""
import json
import os
import shlex
from session import session,access_module
from download_assets import ROOT

CODE=r'''
import hashlib,json,os,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
old=root/'envs/bio_intel_py310_r2'
environment=Path(os.environ['EXTERNAL_MET_BIO_ENV'])
assert environment.parent==root/'envs' and environment.name=='bio_intel_py310_r3' and not environment.exists()
failed=json.loads((root/'operator/environment_r2/FAILED.json').read_text())
assert failed['stage']=='author_dependencies'
audit=root/'operator/environment_r3';audit.mkdir(mode=0o700,exist_ok=False)
logs=root/'logs/environment_r3';logs.mkdir(mode=0o700,exist_ok=False)
meta=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed')
def write(name,data):
    with (audit/name).open('x') as f:json.dump(data,f,indent=2);f.write('\n')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
source=root/'assets/bio_source'
identity=json.loads((root/'operator/BIO_SOURCE_IDENTITY.json').read_text())
assert identity['commit']=='83c73e0cac0b5a3c77a298547040ea3ea7f99854'
for name,item in identity['files'].items():assert digest(source/name)==item['sha256']
old_guard={str(p.relative_to(old)):digest(p) for p in (old/'conda-meta').glob('*') if p.is_file()}
old_guard['bin/python_sha256']=digest(old/'bin/python')
old_failure_sha=digest(root/'operator/environment_r2/FAILED.json')
os.environ.update(CONDA_PKGS_DIRS=str(root/'cache/conda'),PIP_CONFIG_FILE='/dev/null',PIP_DISABLE_PIP_VERSION_CHECK='1',
    PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',LC_ALL='C')
for key in list(os.environ):
    if key.startswith('CONDA_OVERRIDE_'):del os.environ[key]
conda='/public/software/apps/miniconda3/bin/conda'
stage='compatible_binary_solver_preflight'
def run(argv,name,timeout=1200,extra=None):
    env=dict(os.environ);env.update(extra or {})
    with (logs/name).open('x') as f:result=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=timeout)
    assert result.returncode==0,'environment stage failed: '+name
    return logs/name
try:
    free=shutil.disk_usage(root).free;assert free>100*1024**3
    host_libc=subprocess.check_output(['getconf','GNU_LIBC_VERSION'],text=True).strip()
    assert tuple(map(int,host_libc.split()[-1].split('.'))) >= (2,17)
    specs=['python=3.10.22','pandas=2.3.3','numpy=2.2.6']
    preflight=run([conda,'create','--prefix',str(environment),'--dry-run','--json','--override-channels','-c','conda-forge',
                   '--repodata-fn','current_repodata.json',*specs],'compatible_solver_preflight.json',900)
    plan=json.loads(preflight.read_text());assert plan['success'] is True
    selected=plan['actions']['LINK']
    assert any(x['name']=='pandas' and x['version']=='2.3.3' for x in selected)
    assert any(x['name']=='numpy' and x['version']=='2.2.6' for x in selected)
    assert not any(x['name'] in ['pytorch','torch','torchvision','gcc','gxx','gcc_linux-64','gxx_linux-64'] for x in selected)
    write('STARTED.json',dict(meta,status='compatible_binary_plan_pass_install_started',host_libc=host_libc,
        selected_packages=selected,preflight_sha256=digest(preflight),free_bytes_before=free,
        remote_utc=datetime.now(timezone.utc).isoformat(),original_r2_guard=old_guard,old_failure_sha256=old_failure_sha,
        reuse_method='conda_copy_clone_no_shared_environment_files',no_system_compiler_install=True))
    stage='independent_r3_copy_clone'
    run([conda,'create','--prefix',str(environment),'--clone',str(old),'--copy','-y'],'r2_copy_clone.log',1200)
    stage='compatible_binary_install'
    run([conda,'install','--prefix',str(environment),'--copy','--freeze-installed','--override-channels','-c','conda-forge',
         '--repodata-fn','current_repodata.json','-y',*specs],'pandas_compatible_install.log',900)
    py=str(environment/'bin/python')
    constraints=audit/'constraints.txt'
    with constraints.open('x') as f:f.write('torch==2.6.0+cu124\ntorchvision==0.21.0+cu124\npillow==10.4.0\npandas==2.3.3\nnumpy==2.2.6\n')
    stage='author_dependencies'
    run([py,'-B','-m','pip','install','--no-cache-dir','--index-url','https://pypi.org/simple','-c',str(constraints),
         '-r',str(source/'requirements.txt'),'matplotlib==3.10.9'],'author_dependencies.log',1200)
    stage='trainer_registration'
    run([py,'-B',str(source/'docker/install_trainers.py')],'trainer_registration.log',180,
        extra={'TRAINER_SOURCE_DIR':str(source/'trainers')})
    stage='versions_and_pip_check'
    expected={'torch':'2.6.0','nnunetv2':'2.7.0','monai':'1.5.2','SimpleITK':'2.5.5','nibabel':'5.4.0','scipy':'1.15.2',
        'numpy':'2.2.6','pandas':'2.3.3','timm':'1.0.22','einops':'0.8.2','threadpoolctl':'3.6.0','matplotlib':'3.10.9','pillow':'10.4.0'}
    code='import importlib.metadata as m,json,torch,pandas,numpy; print(json.dumps(dict(versions={n:m.version(n) for n in '+repr(list(expected))+'},cuda_build=torch.version.cuda)))'
    observed=json.loads(subprocess.check_output([py,'-B','-c',code],text=True))
    assert all(observed['versions'][name].split('+')[0]==value for name,value in expected.items())
    assert observed['versions']['torch']=='2.6.0+cu124' and observed['cuda_build']=='12.4'
    run([py,'-B','-m','pip','check'],'pip_check.log',60)
    freeze=subprocess.check_output([py,'-B','-m','pip','freeze','--all'],text=True)
    with (audit/'ENVIRONMENT_FREEZE.txt').open('x') as f:f.write(freeze)
    explicit=subprocess.check_output([conda,'list','--prefix',str(environment),'--explicit','--sha256'],text=True)
    with (audit/'CONDA_EXPLICIT.txt').open('x') as f:f.write(explicit)
    stage='original_r2_preservation_check'
    for name,value in old_guard.items():assert digest(old/'bin/python' if name=='bin/python_sha256' else old/name)==value
    assert digest(root/'operator/environment_r2/FAILED.json')==old_failure_sha
    result=dict(meta,status='environment_r3_preparation_pass_not_model_runtime_pass',environment=str(environment),versions=observed,
        host_libc=host_libc,remote_utc=datetime.now(timezone.utc).isoformat(),free_bytes_after=shutil.disk_usage(root).free,
        jobs_created=False,old_r1_r2_modified=False,model_loaded=False,system_compiler_installed=False)
    write('COMPLETE.json',result);print(json.dumps(result))
except BaseException as error:
    result=dict(meta,status='environment_r3_preparation_failed',stage=stage,error_type=type(error).__name__,
        message=str(error) if isinstance(error,AssertionError) else 'bounded independent environment preparation failed',
        remote_utc=datetime.now(timezone.utc).isoformat(),no_automatic_retry=True,jobs_created=False)
    write('FAILED.json',result);print(json.dumps(result))
'''


def main():
    assert json.loads((ROOT/'ENVIRONMENT_R3_AUTHORIZATION.json').read_text())['old_r1_r2_environments_failures_preserved']
    with session('install_environment_r3') as access:
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' EXTERNAL_MET_BIO_ENV='+shlex.quote(os.environ['EXTERNAL_MET_BIO_ENV'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(CODE)
        result=json.loads(access.ssh(command,timeout=5400).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'])),flush=True)
        if result['status']=='environment_r3_preparation_failed':raise RuntimeError('Environment R3 failed; preserve audit; no automatic retry')


if __name__=='__main__':main()
