"""Read approved sdist build metadata after the pre-install parsing failure."""
import json
import os
from pathlib import Path
import shlex
from session import session,access_module

ROOT=Path(__file__).resolve().parents[1]
CODE=r'''
import os,shutil,tarfile
from datetime import datetime,timezone
try:
    import tomllib as toml
except ImportError:
    from pip._vendor import tomli as toml
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
attempt=root/'attempts/s0_cpu_prepare_r1'
failure=json.loads((attempt/'reports/STAGING_FAILED.json').read_text())
require(failure['stage']=='accepted_R3_identity_and_source_requirements' and failure['message']=="'build-backend'", 'unexpected staging failure')
sources={}
for name,item in SOURCES.items():
    path=attempt/'sources'/item['filename']
    require(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'], 'staged sdist identity drift')
    with tarfile.open(path,'r:gz') as archive:
        members=archive.getmembers()
        require(len(members)<10000 and sum(x.size for x in members)<50*1024**2, 'sdist metadata budget')
        configs=[x for x in members if x.name.count('/')==1 and x.name.endswith('/pyproject.toml')]
        require(len(configs)==1 and configs[0].size<65536, 'pyproject metadata ambiguous or oversized')
        raw=archive.extractfile(configs[0]).read()
        build=toml.loads(raw.decode()).get('build-system')
        sources[name]=dict(source_sha256=sha(path),pyproject_sha256=__import__('hashlib').sha256(raw).hexdigest(),
            build_system=build,setup_py_exists=any(x.name.count('/')==1 and x.name.endswith('/setup.py') for x in members),
            setup_cfg_exists=any(x.name.count('/')==1 and x.name.endswith('/setup.cfg') for x in members))
result=dict(META,status='read_only_source_build_failure_diagnosis_not_retry',failure=failure,
    failed_marker_sha256=sha(attempt/'reports/STAGING_FAILED.json'),sources=sources,
    build_environment_exists=(root/'envs'/BUILD_ENV).exists(),runtime_R4_exists=(root/'envs/bio_intel_py310_r4').exists(),
    install_log_exists=(attempt/'logs/BUILD_ENV_INSTALL.log').exists(),
    scheduler_acceptance_sha256=sha(root/'operator/s0_scheduler_acceptance_r1/COMPLETE.json'),
    prior_environments_preserved=prior_guard(root)==json.loads((root/'operator/s0_scheduler_acceptance_r1/COMPLETE.json').read_text())['prior_guard'],
    free_bytes=shutil.disk_usage(root).free,remote_writes=False,solver_rerun=False,jobs_created=False,
    models_loaded=False,images_read=False,remote_utc=datetime.now(timezone.utc).isoformat())
print(json.dumps(result))
'''


def remote_code():
    code=(ROOT/'runtime/s0_build_contract.py').read_text()+'\n'+CODE
    compile(code,'read_only_S0_staging_failure','exec')
    return code


def main():
    with session('inspect_s0_staging_failure') as access:
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(remote_code())
        result=json.loads(access.ssh(command,timeout=240).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'],remote_utc=result['remote_utc'])),flush=True)


if __name__=='__main__':
    main()
