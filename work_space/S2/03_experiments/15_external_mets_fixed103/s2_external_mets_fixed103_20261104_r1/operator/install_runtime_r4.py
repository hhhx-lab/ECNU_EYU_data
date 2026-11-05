"""One approved R4 runtime preparation consuming the three accepted wheels."""
import hashlib
import json
import os
from pathlib import Path
import shlex
from urllib.parse import unquote, urlparse

from session import session, access_module

ROOT = Path(__file__).resolve().parents[1]
DIRECT = {'nnunetv2': '2.7.0', 'monai': '1.5.2', 'SimpleITK': '2.5.5',
          'nibabel': '5.4.0', 'scipy': '1.15.2', 'numpy': '2.2.6', 'pandas': '2.3.3',
          'timm': '1.0.22', 'einops': '0.8.2', 'threadpoolctl': '3.6.0',
          'matplotlib': '3.10.9', 'pillow': '10.4.0',
          'torch': '2.6.0+cu124', 'torchvision': '0.21.0+cu124'}


def validate_pip_plan(plan, installed, wheels):
    def require(ok, message):
        if not ok:
            raise RuntimeError(message)
    canonical = lambda name: name.lower().replace('_', '-').replace('.', '-')
    expected = {canonical(k): v for k, v in DIRECT.items()}
    expected.update({canonical(w['name']): w['version'] for w in wheels})
    selected = {}
    for item in plan['install']:
        name = canonical(item['metadata']['name'])
        require(name not in selected, 'duplicate selected distribution')
        selected[name] = item['metadata']['version']
        require(name not in {'torch', 'torchvision', 'numpy', 'pandas'}, 'existing model/base dependency replacement')
        if name in expected:
            require(selected[name] == expected[name], 'approved version drift: ' + name)
        info = item['download_info']
        url = urlparse(info['url'])
        if url.scheme == 'file':
            matches = [w for w in wheels if canonical(w['name']) == name]
            require(len(matches) == 1, 'unapproved local distribution')
            wheel = matches[0]
            require(Path(unquote(url.path)).resolve() == Path(wheel['path']).resolve(), 'compiled wheel path drift')
            require(info['archive_info']['hashes']['sha256'] == wheel['sha256'], 'compiled wheel resolver SHA drift')
        else:
            require(url.scheme == 'https' and url.hostname == 'files.pythonhosted.org', 'non-official package source')
            require(not url.username and not url.password, 'authenticated package URL forbidden')
            require('sha256' in info['archive_info']['hashes'], 'package SHA absent')
    actual = {canonical(k): v for k, v in installed.items() if v is not None}
    actual.update(selected)
    require(all(actual.get(k) == v for k, v in expected.items()), 'incomplete exact runtime dependency plan')
    require(all(canonical(w['name']) in selected for w in wheels), 'accepted compiled wheel not consumed')
    return selected


BODY = r'''
import hashlib,json,os,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import unquote,urlparse
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
old=root/'envs/bio_intel_py310_r3'
environment=root/'envs/bio_intel_py310_r4'
audit=root/'operator/environment_r4'
logs=root/'logs/environment_r4'
def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):digest.update(block)
    return digest.hexdigest()
def write(path,data):
    with Path(path).open('x') as f:json.dump(data,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    Path(path).chmod(0o440)
meta=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed')
require(not environment.exists() and not audit.exists() and not logs.exists(),'R4 already attempted; no retry')
audit.mkdir(mode=0o700);logs.mkdir(mode=0o700)
stage='immutable_input_acceptance'
env={k:v for k,v in os.environ.items() if not k.upper().endswith('_PROXY') and
     not k.startswith(('CONDA_OVERRIDE_','HPC_','ECNU_','SINGULARITY','APPTAINER'))}
env.update(PIP_CONFIG_FILE='/dev/null',PIP_DISABLE_PIP_VERSION_CHECK='1',PYTHONDONTWRITEBYTECODE='1',
           OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',LC_ALL='C')
conda='/public/software/apps/miniconda3/bin/conda'
def run(argv,name,timeout=1200,extra=None):
    local_env=dict(env);local_env.update(extra or {})
    with (logs/name).open('x') as f:
        p=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,env=local_env,timeout=timeout)
    require(p.returncode==0,'runtime preparation failed: '+name)
    return logs/name
try:
    authorization=json.loads(os.environ['R4_RUNTIME_AUTHORIZATION'])
    require(authorization['consume_previously_approved_dependencies_and_wheels'],'runtime authorization missing')
    write(audit/'AUTHORIZATION.json',authorization)
    write(audit/'EXECUTED_SOURCE.json',dict(source_sha256=os.environ['R4_RUNTIME_SOURCE_SHA256']))
    complete_path=root/'attempts/s0_sif_recovery_r2/reports/COMPLETE.json'
    require(sha(complete_path)=='8d302e878015401bf05904a5b5deea9262b6537c6810cb858835364b8c12d3f2','CPU completion identity drift')
    complete=json.loads(complete_path.read_text())
    require(complete['job_id']=='4309579' and complete['status']=='S0_recovery_artifacts_complete_runtime_and_GPU_still_require_validation','CPU completion contract drift')
    p=subprocess.run(['sacct','-n','-P','-j','4309579','--format=JobID,State,ExitCode'],capture_output=True,text=True,timeout=30)
    rows=[line.strip().split('|') for line in p.stdout.splitlines() if line.strip()]
    require(p.returncode==0 and {r[0] for r in rows}=={'4309579','4309579.batch'} and
            all(r[1:3]==['COMPLETED','0:0'] for r in rows),'CPU successful accounting required')
    require(not (complete_path.parent/'FAILED.json').exists(),'CPU failure conflicts with completion')
    contract=json.loads((root/'attempts/s0_sif_recovery_r2/contract/PREPARATION_CONTRACT.json').read_text())
    wheels=contract['reused_wheels']
    for item in [complete['sif'],*wheels]:
        path=Path(item['path'])
        require(path.is_file() and not path.is_symlink() and path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'accepted artifact drift')
    for wheel in wheels:
        wheel['version']={'numexpr':'2.14.1','connected-components-3d':'4.1.1','ndindex':'1.10.1'}[wheel['name']]
    require(len(wheels)==3 and len({w['name'] for w in wheels})==3,'compiled wheel set drift')
    source=root/'assets/bio_source'
    identity=json.loads((root/'operator/BIO_SOURCE_IDENTITY.json').read_text())
    require(identity['commit']=='83c73e0cac0b5a3c77a298547040ea3ea7f99854','author source revision drift')
    for name,item in identity['files'].items():require(sha(source/name)==item['sha256'],'author source drift')
    failures={str(p.relative_to(root)):sha(p) for p in [root/'operator/ENV_INSTALL_FAILED.json',
        root/'operator/environment_r2/FAILED.json',root/'operator/environment_r3/FAILED.json',
        root/'attempts/s0_cpu_prepare_r2/reports/FAILED.json',root/'attempts/s0_cpu_prepare_r3a/reports/FAILED.json',
        root/'attempts/s0_sif_recovery_r1/reports/FAILED.json'] if p.is_file()}
    require(root.joinpath('operator/environment_r3/FAILED.json').is_file(),'R3 failed audit absent')
    old_guard={str(p.relative_to(old)):sha(p) for p in (old/'conda-meta').glob('*') if p.is_file()}
    old_guard['bin/python']=sha(old/'bin/python')
    require(shutil.disk_usage(root).free>100*1024**3,'fresh runtime capacity gate')
    probe='import importlib.metadata as m,json; names='+repr(list(DIRECT))+'; out={};\nfor n in names:\n try: out[n]=m.version(n)\n except m.PackageNotFoundError: out[n]=None\nprint(json.dumps(out))'
    installed=json.loads(subprocess.check_output([str(old/'bin/python'),'-B','-c',probe],env=env,text=True,timeout=40))
    constraints=audit/'constraints.txt'
    text='\n'.join(k+'=='+v for k,v in {**DIRECT,**{w['name']:w['version'] for w in wheels}}.items())+'\n'
    with constraints.open('x') as f:f.write(text)
    args=['-B','-m','pip','install','--no-cache-dir','--index-url','https://pypi.org/simple','--prefer-binary',
          '-c',str(constraints),'-r',str(source/'requirements.txt'),'matplotlib==3.10.9',*[w['path'] for w in wheels]]
    stage='exact_dependency_plan_before_environment_creation'
    report=audit/'PIP_INSTALL_PLAN.json'
    run([str(old/'bin/python'),*args,'--dry-run','--report',str(report)],'pip_plan.log',1200)
    selected=validate_pip_plan(json.loads(report.read_text()),installed,wheels)
    resolved_constraints=audit/'RESOLVED_CONSTRAINTS.txt'
    with resolved_constraints.open('x') as f:
        f.write(text+'\n'.join(k+'=='+v for k,v in selected.items())+'\n')
    write(audit/'PLAN_ACCEPTANCE.json',dict(meta,selected=selected,installed_base=installed,
        plan_sha256=sha(report),compiled_wheels=wheels,old_guard=old_guard,prior_failure_guard=failures,
        free_bytes_before=shutil.disk_usage(root).free,remote_utc=datetime.now(timezone.utc).isoformat(),
        status='exact_runtime_plan_pass_not_installation_or_GPU_pass'))
    stage='copy_clone_existing_base_no_solver'
    run([conda,'create','--prefix',str(environment),'--clone',str(old),'--copy','--offline','-y'],'copy_clone.log',1200)
    stage='consume_approved_dependencies_and_compiled_wheels'
    py=str(environment/'bin/python')
    install_args=list(args)
    install_args[install_args.index('-c')+1]=str(resolved_constraints)
    run([py,*install_args,'--report',str(audit/'PIP_INSTALL_ACTUAL.json')],'install.log',1800)
    actual_plan=json.loads((audit/'PIP_INSTALL_ACTUAL.json').read_text())
    actual=validate_pip_plan(actual_plan,installed,wheels)
    require(actual==selected,'installed distribution selection differs from accepted plan')
    require({i['metadata']['name']:i['download_info'] for i in actual_plan['install']}==
            {i['metadata']['name']:i['download_info'] for i in json.loads(report.read_text())['install']},'resolved package asset drift')
    stage='author_trainer_registration_and_imports'
    run([py,'-B',str(source/'docker/install_trainers.py')],'trainer_registration.log',180,
        {'TRAINER_SOURCE_DIR':str(source/'trainers')})
    imports='import torch,torchvision,numpy,pandas,scipy,nibabel,SimpleITK,monai,timm,einops,threadpoolctl,matplotlib,numexpr,cc3d,ndindex,nnunetv2; import importlib.metadata as m,json; print(json.dumps(dict(versions={n:m.version(n) for n in '+repr(list(DIRECT))+ '},cuda_build=torch.version.cuda)))'
    observed=json.loads(subprocess.check_output([py,'-B','-c',imports],env=env,text=True,timeout=90))
    require(observed['versions']==DIRECT and observed['cuda_build']=='12.4','exact versions or imports drift')
    run([py,'-B','-m','pip','check'],'pip_check.log',90)
    for name,digest in old_guard.items():require(sha(old/name)==digest,'old R3 environment modified')
    for name,digest in failures.items():require(sha(root/name)==digest,'prior failure modified')
    freeze=subprocess.check_output([py,'-B','-m','pip','freeze','--all'],env=env,text=True,timeout=90)
    with (audit/'ENVIRONMENT_FREEZE.txt').open('x') as f:f.write(freeze)
    result=dict(meta,status='runtime_R4_dependencies_imports_trainers_pass_not_model_or_GPU_pass',
        environment=str(environment),versions=observed,compiled_wheels=wheels,author_revision=identity['commit'],
        plan_sha256=sha(report),actual_install_report_sha256=sha(audit/'PIP_INSTALL_ACTUAL.json'),
        freeze_sha256=sha(audit/'ENVIRONMENT_FREEZE.txt'),prior_failure_guard=failures,old_R3_guard=old_guard,
        remote_utc=datetime.now(timezone.utc).isoformat(),free_bytes_after=shutil.disk_usage(root).free,
        models_loaded=False,images_read=False,jobs_created=False,system_compiler_installed=False)
    write(audit/'COMPLETE.json',result);print(json.dumps(result))
except BaseException as error:
    result=dict(meta,status='runtime_R4_preparation_failed',stage=stage,error_type=type(error).__name__,
        message=str(error),remote_utc=datetime.now(timezone.utc).isoformat(),no_automatic_retry=True,jobs_created=False)
    write(audit/'FAILED.json',result);print(json.dumps(result))
'''


def remote_code():
    import inspect
    code = 'DIRECT=' + repr(DIRECT) + '\n' + inspect.getsource(validate_pip_plan) + '\n' + BODY
    compile(code, 'approved_runtime_R4', 'exec')
    return code


def main():
    authorization = dict(consume_previously_approved_dependencies_and_wheels=True,
        latest_user_request='开始提交推理作业', preserve_old_environments_and_failures=True,
        change_model_or_direct_dependency_versions=False, system_compiler_installation=False,
        reuse_successful_CPU_4309579=True, GPU_submission_before_runtime_acceptance=False)
    code = remote_code()
    with session('install_runtime_r4') as access:
        access_module.exclusive(access.audit / 'EXECUTED_SOURCE.json',
                                dict(source=code, sha256=hashlib.sha256(code.encode()).hexdigest()))
        access_module.exclusive(access.audit / 'AUTHORIZATION.json', authorization)
        command = 'env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT=' + shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command += ' R4_RUNTIME_AUTHORIZATION=' + shlex.quote(json.dumps(authorization))
        command += ' R4_RUNTIME_SOURCE_SHA256=' + shlex.quote(hashlib.sha256(code.encode()).hexdigest())
        command += ' ' + shlex.quote(os.environ['ECNU_REMOTE_PYTHON']) + ' -B -c ' + shlex.quote(code)
        result = json.loads(access.ssh(command, timeout=6000).strip().splitlines()[-1])
        access_module.exclusive(access.audit / 'RESULT.json', result)
        print(json.dumps(dict(audit=str(access.audit), result=result)), flush=True)
        if result['status'] == 'runtime_R4_preparation_failed':
            raise RuntimeError('Independent R4 stopped; preserve original failure; no automatic retry')


if __name__ == '__main__':
    main()
