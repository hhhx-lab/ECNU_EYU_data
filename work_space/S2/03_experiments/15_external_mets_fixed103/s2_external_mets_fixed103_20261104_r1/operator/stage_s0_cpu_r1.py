"""Stage the authorized offline CPU attempt after real compiler preflight pass."""
import hashlib
import json
import os
from pathlib import Path
import shlex
from session import session, access_module

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_CODE = r'''
import os,shutil,subprocess,urllib.request
from datetime import datetime,timezone
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
attempt=root/'attempts'/ATTEMPT
prefix=root/'envs'/BUILD_ENV
audit=attempt/'reports'
stage='accepted_R3_identity_and_source_requirements'
before=prior_guard(root)
try:
    for name,digest in expected_runtime.items():
        require(sha(attempt/'runtime'/name)==digest,'received runtime SHA drift: '+name)
        compile((attempt/'runtime'/name).read_text(),name,'exec')
    require(not prefix.exists(),'build environment already exists; no install retry')
    preflight_audit=root/'operator/s0_source_build_preflight_r3'
    preflight=json.loads((preflight_audit/'COMPLETE.json').read_text())
    require(preflight['status']=='s0_source_build_preflight_pass_no_install_no_job','compiler preflight incomplete')
    solver=json.loads((preflight_audit/'SOLVER.json').read_text())
    require(sha(preflight_audit/'SOLVER.json')==preflight['solver_sha256'],'solver identity drift')
    selected=validate_build_plan(solver)
    require(shutil.disk_usage(root).free>150*1024**3,'fresh staging capacity failed')
    require(sha(preflight_audit/'BUILD_ENV_EXPLICIT.txt')==preflight['explicit_lock_sha256'],'accepted lock identity drift')
    text=render_explicit_lock(explicit_lock_records(solver))
    require(text==(preflight_audit/'BUILD_ENV_EXPLICIT.txt').read_text(),'explicit lock differs from solver')
    lock=attempt/'contract/BUILD_ENV_EXPLICIT.txt'
    with lock.open('x') as stream:stream.write(text)
    lock.chmod(0o440)
    env=isolated_conda_env(preflight_audit,os.environ)
    conda=os.environ['EXTERNAL_MET_CONDA_BIN']
    observed=json.loads(subprocess.check_output([conda,'config','--show','channel_alias','custom_channels',
        'pkgs_dirs','ssl_verify','add_anaconda_token','--json'],env=env,text=True,timeout=60))
    validate_effective_config(observed,preflight_audit)
    for name,item in SOURCES.items():
        request=urllib.request.Request(item['url'],headers={'User-Agent':'ECNU-authorized-same-version-source-build'})
        with urllib.request.urlopen(request,timeout=60) as response:
            require(response.geturl().startswith('https://files.pythonhosted.org/'),'source redirect origin drift')
            raw=response.read(item['bytes']+1)
        require(len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],'source bytes/SHA drift')
        path=attempt/'sources'/item['filename']
        with path.open('xb') as stream:stream.write(raw)
        path.chmod(0o440)
    configurations={name:read_sdist_build_config(attempt/'sources'/item['filename']) for name,item in SOURCES.items()}
    requirements=validate_solver_build_requirements(configurations,selected)
    write(audit/'SOURCE_BUILD_REQUIREMENTS.json',dict(status='source_requirements_covered_by_exact_solver',
        configurations=configurations,checked=requirements,sources=SOURCES))
    stage='exact_build_environment_install'
    with (attempt/'logs/BUILD_ENV_INSTALL.log').open('x') as log:
        p=subprocess.run([conda,'create','--prefix',str(prefix),'--copy','--file',str(lock),'-y','--json'],
                         env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    require(p.returncode==0,'build environment installation failed')
    stage='exact_environment_identity_and_source_staging'
    metadata=[json.loads(p.read_text()) for p in (prefix/'conda-meta').glob('*.json')]
    installed={(x['name'],x['version'],x['build']) for x in metadata}
    expected={(x['name'],x['version'],x.get('build',x.get('build_string'))) for x in selected}
    require(installed==expected,'installed build environment differs from exact plan')
    py=prefix/'bin/python'
    probe='import sys;sys.path.insert(0,'+repr(str(attempt/'runtime'))+');import prepare_s0_cpu_r1 as p,json;from pathlib import Path;print(json.dumps({name:p.validate_sdist(Path('+repr(str(attempt/'sources'))+')/item["filename"]) for name,item in p.SOURCES.items()}))'
    builds=json.loads(subprocess.check_output([str(py),'-B','-c',probe],env=env,text=True,timeout=60))
    readelf=prefix/'bin/x86_64-conda-linux-gnu-readelf'
    require(readelf.is_file(),'Conda readelf missing')
    build_files={str(p.relative_to(prefix)):sha(p) for p in (prefix/'conda-meta').glob('*') if p.is_file()}
    for name in ['bin/python','bin/x86_64-conda-linux-gnu-cc','bin/x86_64-conda-linux-gnu-c++','bin/x86_64-conda-linux-gnu-readelf']:
        require((prefix/name).is_file(),'build executable missing: '+name)
        build_files[name]=sha(prefix/name)
    require(prior_guard(root)==before,'old environment/failure changed during staging')
    package={str(p.relative_to(attempt)):sha(p) for p in (attempt/'runtime').glob('*.py')}
    contract=dict(META,stage='S0_CPU_preparation_only',remote_root=str(root),attempt=str(attempt),build_env=str(prefix),
        backend='/public/software/singularity-ce/4.1.1/bin/singularity',readelf=str(readelf),
        archive=str(root/'assets/container_conversion/fixed2025.docker-archive.tar'),archive_sha256=ARCHIVE_SHA,
        sources=SOURCES,build_configurations=builds,build_env_files=build_files,package_files=package,prior_guard=before,
        compiler_preflight_sha256=sha(preflight_audit/'COMPLETE.json'),compiler_preflight=str(preflight_audit),
        source_build_requirements_sha256=sha(audit/'SOURCE_BUILD_REQUIREMENTS.json'),
        incremental_budget_bytes=150*1024**3,budget_scope='conservative_CPU_preparation_reservation_not_full_inference',
        resource=dict(partition='cpu_192G',cpus_per_task=8,memory_gib=64,time_limit='04:00:00',gpu=0),
        single_conversion_deadline_seconds=7200,automatic_retry=False,original_inference_jobs_created=False)
    write(attempt/'contract/PREPARATION_CONTRACT.json',contract)
    outcome=dict(status='S0_offline_CPU_input_staging_pass_not_model_runtime_pass',attempt=str(attempt),build_env=str(prefix),
        contract_sha256=sha(attempt/'contract/PREPARATION_CONTRACT.json'),build_env_explicit_sha256=sha(lock),
        build_configurations=builds,prior_environments_preserved=True,free_bytes=shutil.disk_usage(root).free,
        runtime_R4_installed=False,jobs_created=False,models_loaded=False,remote_utc=datetime.now(timezone.utc).isoformat())
    write(audit/'STAGING_COMPLETE.json',outcome)
except BaseException as error:
    outcome=dict(status='S0_input_staging_failed',stage=stage,error_type=type(error).__name__,message=str(error),
        no_automatic_retry=True,environment_created=prefix.exists(),jobs_created=False,models_loaded=False,
        prior_environments_preserved=prior_guard(root)==before,remote_utc=datetime.now(timezone.utc).isoformat())
    write(audit/'STAGING_FAILED.json',outcome)
print(json.dumps(dict(META,**outcome)))
'''


def remote_code():
    identities = {name:hashlib.sha256((ROOT/'runtime'/name).read_bytes()).hexdigest()
                  for name in ('s0_build_contract.py', 'prepare_s0_cpu_r1.py')}
    prefix='\n'.join((ROOT/'operator'/name).read_text() for name in
                      ('s0_source_build_r2_contract.py','s0_source_build_r3_contract.py','s0_source_requirements.py'))
    source=(ROOT/'runtime/s0_build_contract.py').read_text()+'\n'+prefix+'\nexpected_runtime='+repr(identities)+'\n'+ENVIRONMENT_CODE
    compile(source, 'S0_stage_build_environment', 'exec')
    return source


def main():
    authorization = json.loads((ROOT/'S0_BUILD_CONVERSION_AUTHORIZATION.json').read_text())
    assert authorization['new_cpu_preparation_job_authorized']
    assert json.loads((ROOT/'S0_SOURCE_BUILD_R3_AUTHORIZATION.json').read_text())['independent_R3_parser_correction_authorized']
    remote = os.environ['EXTERNAL_MET_REMOTE_ROOT']
    py = os.environ['ECNU_REMOTE_PYTHON']
    with session('stage_s0_cpu_r1') as access:
        bootstrap = '''from pathlib import Path
import json
root=Path('''+repr(remote)+''')
assert json.loads((root/'operator/s0_source_build_preflight_r3/COMPLETE.json').read_text())['status']=='s0_source_build_preflight_pass_no_install_no_job'
attempt=root/'attempts/s0_cpu_prepare_r1'
assert not attempt.exists()
attempt.mkdir(parents=True,mode=0o700)
for name in ['runtime','contract','sources','logs','reports','wheels','wheel_probe','scratch','operator','slurm']:(attempt/name).mkdir(mode=0o700)
print(json.dumps(dict(status='exclusive_CPU_staging_directory_created',attempt=str(attempt))))
'''
        result = json.loads(access.ssh(shlex.quote(py)+' -B -c '+shlex.quote(bootstrap)).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'BOOTSTRAP.json', result)
        for name in ('s0_build_contract.py', 'prepare_s0_cpu_r1.py'):
            access.scp(ROOT/'runtime'/name, remote+'/attempts/s0_cpu_prepare_r1/runtime/'+name)
        access.scp(ROOT/'S0_BUILD_CONVERSION_AUTHORIZATION.json', remote+'/attempts/s0_cpu_prepare_r1/contract/')
        access.scp(ROOT/'S0_SOURCE_BUILD_R3_AUTHORIZATION.json', remote+'/attempts/s0_cpu_prepare_r1/contract/')
        command = 'env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(remote)
        command += ' EXTERNAL_MET_CONDA_BIN='+shlex.quote(os.environ['EXTERNAL_MET_CONDA_BIN'])
        command += ' '+shlex.quote(py)+' -B -c '+shlex.quote(remote_code())
        result = json.loads(access.ssh(command,timeout=2200).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json', result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'],remote_utc=result['remote_utc'])),flush=True)
        if result['status'] != 'S0_offline_CPU_input_staging_pass_not_model_runtime_pass':
            raise RuntimeError('S0 staging failed; preserve new environment and audit; no submission/retry')


if __name__ == '__main__':
    main()
