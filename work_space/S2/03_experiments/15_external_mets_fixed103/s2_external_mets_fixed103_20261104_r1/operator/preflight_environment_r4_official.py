"""One authorized full-URL binary preflight; preserves the earlier mirror failure."""
import hashlib
import inspect
import json
import os
import shlex
from pathlib import Path
from urllib.parse import urlsplit

from environment_binary_contract import validate_plan
from preflight_environment_r4 import EXPECTED
from session import session, access_module

ROOT = Path(__file__).resolve().parents[1]
CHANNEL = 'https://conda.anaconda.org/conda-forge'


def validate_official_plan(plan, expected):
    selected = validate_plan(plan, expected)
    for item in [*selected, *plan['actions'].get('FETCH', [])]:
        origins = [item.get(key) for key in ('url', 'base_url', 'channel')]
        urls = [value for value in origins if isinstance(value, str) and '://' in value]
        assert urls, 'official absolute package origin missing'
        for value in urls:
            parsed = urlsplit(value)
            assert (parsed.scheme == 'https' and parsed.netloc == 'conda.anaconda.org'
                    and not parsed.query and not parsed.fragment
                    and (parsed.path == '/conda-forge' or parsed.path.startswith('/conda-forge/'))), \
                'package not from official Conda-forge origin'
    return selected


CODE = r'''
import hashlib,json,os,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
target=root/'envs/bio_intel_py310_r4'
assert not target.exists(), 'R4 already exists; no retry'
parent=root/'operator/environment_r4_preflight/FAILED.json'
assert json.loads(parent.read_text())['stage']=='same_version_binary_solver'
audit=root/'operator/environment_r4_official_preflight'
audit.mkdir(mode=0o700,exist_ok=False)
meta=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed')
def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()
def guard():
    result={}
    for prefix in sorted((root/'envs').iterdir()):
        if not prefix.is_dir() or prefix.name==target.name:continue
        result[prefix.name]={str(p.relative_to(prefix)):digest(p) for p in (prefix/'conda-meta').glob('*') if p.is_file()}
        result[prefix.name]['bin/python_sha256']=digest(prefix/'bin/python')
    markers=[root/'operator/environment/FAILED.json',root/'operator/environment_r2/FAILED.json',
             root/'operator/environment_r3/FAILED.json',parent]
    assert all(p.is_file() for p in markers), 'prior failure marker missing'
    result['failure_markers']={str(p.relative_to(root)):digest(p) for p in markers}
    return result
def write(name,payload):
    with (audit/name).open('x') as stream:json.dump(payload,stream,indent=2);stream.write('\n')
    (audit/name).chmod(0o440)
before=guard()
stage='host_and_previous_identity'
host_libc=None
virtual=[]
try:
    free=shutil.disk_usage(root).free
    assert free>100*1024**3, 'fresh environment capacity gate failed'
    host_libc=subprocess.check_output(['getconf','GNU_LIBC_VERSION'],text=True,timeout=20).strip()
    assert tuple(map(int,host_libc.split()[-1].split('.'))) >= (2,17)
    env=dict(os.environ)
    for key in list(env):
        if key.startswith('CONDA_OVERRIDE_'):del env[key]
    env.update(CONDA_PKGS_DIRS=str(root/'cache/conda'),PYTHONDONTWRITEBYTECODE='1',LC_ALL='C',
               OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
    conda=os.environ['EXTERNAL_MET_CONDA_BIN']
    info=json.loads(subprocess.check_output([conda,'info','--json'],text=True,env=env,timeout=60))
    virtual=info.get('virtual_pkgs',[])
    glibc=[item for item in virtual if item[0]=='__glibc']
    assert len(glibc)==1 and glibc[0][1]==host_libc.split()[-1], 'real virtual glibc mismatch'
    write('STARTED.json',dict(meta,host_libc=host_libc,virtual_packages=virtual,free_bytes=free,
        expected=expected,channel=channel,source_sha256=source_sha,before_guard=before,
        parent_failure_sha256=digest(parent),remote_utc=datetime.now(timezone.utc).isoformat()))
    stage='official_same_version_binary_solver'
    command=[conda,'create','--prefix',str(target),'--dry-run','--json','--override-channels',
        '-c',channel,'--repodata-fn','repodata.json',*[name+'='+version for name,version in expected.items()]]
    log=audit/'SOLVER.json'
    with log.open('x') as stream:
        result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
    assert log.stat().st_size<8*1024**2, 'solver output exceeds lightweight budget'
    text=log.read_text()
    try:plan=json.loads(text)
    except json.JSONDecodeError:plan=dict(success=False,error='invalid_solver_json',message=text[-4000:])
    assert result.returncode==0, 'solver failed: '+str(plan.get('exception_name',plan.get('error','unknown')))
    stage='official_package_origin_and_exact_version_gate'
    selected=validate_official_plan(plan,expected)
    stage='prior_environment_preservation'
    assert guard()==before, 'old environment or failure drift'
    assert not target.exists(), 'dry-run created R4 unexpectedly'
    outcome=dict(meta,status='r4_official_binary_preflight_pass_no_environment_created',host_libc=host_libc,
        virtual_packages=virtual,expected=expected,channel=channel,selected_packages=selected,
        solver_sha256=digest(log),source_sha256=source_sha,target=str(target),
        prior_environments_preserved=True,free_bytes=shutil.disk_usage(root).free,
        remote_utc=datetime.now(timezone.utc).isoformat(),environment_created=False,models_loaded=False,jobs_created=False)
    write('COMPLETE.json',outcome)
except BaseException as error:
    outcome=dict(meta,status='r4_official_binary_preflight_failed_no_install_no_jobs',stage=stage,
        error_type=type(error).__name__,message=str(error) if isinstance(error,AssertionError) else 'bounded official preflight failed',
        channel=channel,host_libc=host_libc,virtual_packages=virtual,source_sha256=source_sha,
        remote_utc=datetime.now(timezone.utc).isoformat(),free_bytes=shutil.disk_usage(root).free,
        environment_created=target.exists(),prior_environments_preserved=guard()==before,
        no_automatic_retry=True,models_loaded=False,jobs_created=False)
    log=audit/'SOLVER.json'
    if log.is_file():
        outcome['solver_sha256']=digest(log)
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-5000))
            outcome['solver_tail']=stream.read(5000).decode('utf-8',errors='replace')
    write('FAILED.json',outcome)
print(json.dumps(outcome))
'''


def remote_code():
    source = 'from urllib.parse import urlsplit\n' + inspect.getsource(validate_plan)
    source += '\n' + inspect.getsource(validate_official_plan)
    source += '\nexpected=' + repr(EXPECTED) + '\nchannel=' + repr(CHANNEL) + '\n' + CODE
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    source = 'source_sha=' + repr(source_sha) + '\n' + source
    compile(source, 'r4_official_binary_preflight', 'exec')
    return source


def main():
    authorization = json.loads((ROOT/'ENVIRONMENT_R4_OFFICIAL_PREFLIGHT_AUTHORIZATION.json').read_text())
    assert authorization['channel'] == CHANNEL
    assert authorization['scope'] == 'one_independent_official_full_channel_r4_preflight_then_create_r4_only_on_pass'
    code = remote_code()
    with session('preflight_environment_r4_official') as access:
        command = 'env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT=' + shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command += ' EXTERNAL_MET_CONDA_BIN=' + shlex.quote(os.environ['EXTERNAL_MET_CONDA_BIN'])
        command += ' ' + shlex.quote(os.environ['ECNU_REMOTE_PYTHON']) + ' -B -c ' + shlex.quote(code)
        outcome = json.loads(access.ssh(command,timeout=1080).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',outcome)
        print(json.dumps(dict(audit=str(access.audit),status=outcome['status'],remote_utc=outcome['remote_utc'])),flush=True)
        if outcome['status'] != 'r4_official_binary_preflight_pass_no_environment_created':
            raise RuntimeError('Official R4 preflight failed; preserve failure; no installation or automatic retry')


if __name__ == '__main__':
    main()
