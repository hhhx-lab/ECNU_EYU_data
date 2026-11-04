"""One exact same-version binary preflight; no R4 environment or jobs created."""
import inspect
import json
import os
import shlex
from pathlib import Path
from environment_binary_contract import validate_plan
from session import session, access_module

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {'python': '3.10.22', 'pandas': '2.3.3', 'numpy': '2.2.6',
            'numexpr': '2.14.1', 'connected-components-3d': '4.1.1', 'ndindex': '1.10.1'}
CODE = r'''
import hashlib,json,os,subprocess,shutil
from datetime import datetime,timezone
from pathlib import Path
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
target=root/'envs/bio_intel_py310_r4'
assert not target.exists(), 'R4 environment already exists; no retry'
prior=root/'operator/environment_r3/FAILED.json'
assert json.loads(prior.read_text())['stage']=='author_dependencies'
audit=root/'operator/environment_r4_preflight'
audit.mkdir(mode=0o700,exist_ok=False)
meta=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def guard():
    result={}
    for prefix in sorted((root/'envs').iterdir()):
        if not prefix.is_dir() or prefix.name==target.name:continue
        result[prefix.name]={str(p.relative_to(prefix)):digest(p) for p in (prefix/'conda-meta').glob('*') if p.is_file()}
        result[prefix.name]['bin/python_sha256']=digest(prefix/'bin/python')
    result['failure_markers']={str(p.relative_to(root)):digest(p) for p in [
        root/'operator/environment/FAILED.json',root/'operator/environment_r2/FAILED.json',prior] if p.is_file()}
    return result
def write(name,payload):
    with (audit/name).open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
    (audit/name).chmod(0o440)
stage='host_and_previous_identity'
before=guard()
try:
    free=shutil.disk_usage(root).free
    assert free>100*1024**3
    host_libc=subprocess.check_output(['getconf','GNU_LIBC_VERSION'],text=True,timeout=20).strip()
    assert tuple(map(int,host_libc.split()[-1].split('.'))) >= (2,17)
    env=dict(os.environ)
    for key in list(env):
        if key.startswith('CONDA_OVERRIDE_'):del env[key]
    env.update(CONDA_PKGS_DIRS=str(root/'cache/conda'),PYTHONDONTWRITEBYTECODE='1',LC_ALL='C',
               OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
    conda=os.environ['EXTERNAL_MET_CONDA_BIN']
    info=json.loads(subprocess.check_output([conda,'info','--json'],text=True,env=env,timeout=60))
    glibc=[item for item in info.get('virtual_pkgs',[]) if item[0]=='__glibc']
    assert len(glibc)==1 and glibc[0][1]==host_libc.split()[-1], 'solver host glibc mismatch'
    write('HOST.json',dict(meta,host_libc=host_libc,virtual_packages=info.get('virtual_pkgs'),free_bytes=free,
                          before_guard=before,remote_utc=datetime.now(timezone.utc).isoformat()))
    stage='same_version_binary_solver'
    command=[conda,'create','--prefix',str(target),'--dry-run','--json','--override-channels',
             '-c','conda-forge','--repodata-fn','current_repodata.json',*[name+'='+version for name,version in expected.items()]]
    log=audit/'SOLVER.json'
    with log.open('x') as f:
        result=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
    assert log.stat().st_size<8*1024**2, 'solver metadata exceeds lightweight budget'
    text=log.read_text()
    try:plan=json.loads(text)
    except json.JSONDecodeError:plan=dict(success=False,error='invalid_solver_json',message=text[-4000:])
    assert result.returncode==0, 'solver failed: '+str(plan.get('exception_name',plan.get('error','unknown')))
    selected=validate_plan(plan,expected)
    stage='prior_environment_preservation'
    assert guard()==before, 'old environment or failure marker drift'
    assert not target.exists(), 'dry-run unexpectedly created R4'
    outcome=dict(meta,status='r4_same_version_binary_preflight_pass_no_environment_created',host_libc=host_libc,
        expected=expected,selected_packages=selected,solver_sha256=digest(log),
        target=str(target),prior_guard=before,prior_environments_preserved=True,free_bytes=shutil.disk_usage(root).free,
        remote_utc=datetime.now(timezone.utc).isoformat(),environment_created=False,models_loaded=False,jobs_created=False)
    write('COMPLETE.json',outcome)
except BaseException as error:
    outcome=dict(meta,status='r4_binary_preflight_failed_no_install_no_jobs',stage=stage,error_type=type(error).__name__,
        message=str(error) if isinstance(error,AssertionError) else 'bounded binary preflight failed',
        remote_utc=datetime.now(timezone.utc).isoformat(),environment_created=target.exists(),
        prior_environments_preserved=guard()==before,no_automatic_retry=True,models_loaded=False,jobs_created=False)
    log=audit/'SOLVER.json'
    if log.is_file():
        outcome.update(solver_sha256=digest(log),solver_tail=log.read_text()[-5000:])
    write('FAILED.json',outcome)
print(json.dumps(outcome))
'''


def remote_code():
    source = inspect.getsource(validate_plan) + '\nexpected=' + repr(EXPECTED) + '\n' + CODE
    compile(source, 'r4_binary_preflight', 'exec')
    return source


def main():
    authorization = json.loads((ROOT/'ENVIRONMENT_R4_AUTHORIZATION.json').read_text())
    assert authorization['new_conda_environment_only_after_preflight_pass']
    code = remote_code()
    with session('preflight_environment_r4') as access:
        command = 'env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT=' + shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command += ' EXTERNAL_MET_CONDA_BIN=' + shlex.quote(os.environ['EXTERNAL_MET_CONDA_BIN'])
        command += ' ' + shlex.quote(os.environ['ECNU_REMOTE_PYTHON']) + ' -B -c ' + shlex.quote(code)
        outcome = json.loads(access.ssh(command,timeout=1080).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',outcome)
        print(json.dumps(dict(audit=str(access.audit),status=outcome['status'],remote_utc=outcome['remote_utc'])),flush=True)
        if outcome['status'] != 'r4_same_version_binary_preflight_pass_no_environment_created':
            raise RuntimeError('R4 binary preflight failed; no environment creation or automatic retry')


if __name__ == '__main__':
    main()
