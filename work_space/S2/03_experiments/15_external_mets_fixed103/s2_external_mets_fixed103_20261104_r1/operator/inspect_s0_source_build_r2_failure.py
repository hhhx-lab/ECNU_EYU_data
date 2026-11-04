"""One read-only effective-config diagnosis; no solver, install or job."""
import json
import os
from pathlib import Path
import shlex
from session import session,access_module

ROOT=Path(__file__).resolve().parents[1]
CODE=r'''
import subprocess
from datetime import datetime,timezone
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
audit=root/PREFLIGHT_R2
failure=json.loads((audit/'FAILED.json').read_text())
require(failure['stage']=='effective_official_config','unexpected R2 failure stage')
require(not (audit/'SOLVER.json').exists(),'R2 solver unexpectedly executed')
env=isolated_conda_env(audit,os.environ)
config=json.loads(subprocess.check_output([os.environ['EXTERNAL_MET_CONDA_BIN'],'config','--show',
    'channel_alias','custom_channels','pkgs_dirs','ssl_verify','add_anaconda_token','--json'],
    env=env,text=True,timeout=60))
result=dict(META,status='read_only_R2_configuration_diagnosis',failure_sha256=sha(audit/'FAILED.json'),
    config_sha256=sha(audit/'condarc.yaml'),effective_channel_alias=config['channel_alias'],
    effective_conda_forge=config['custom_channels'].get('conda-forge'),pkgs_dirs=config['pkgs_dirs'],
    ssl_verify=config['ssl_verify'],add_anaconda_token=config['add_anaconda_token'],
    build_environment_exists=(root/'envs'/BUILD_ENV).exists(),runtime_R4_exists=(root/'envs/bio_intel_py310_r4').exists(),
    solver_executed=False,jobs_created=False,remote_writes=False,models_loaded=False,
    remote_utc=datetime.now(timezone.utc).isoformat())
print(json.dumps(result))
'''


def remote_code():
    source=(ROOT/'runtime/s0_build_contract.py').read_text()+'\n'
    source+=(ROOT/'operator/s0_source_build_r2_contract.py').read_text()+'\n'+CODE
    compile(source,'s0_R2_read_only_diagnosis','exec')
    return source


def main():
    with session('inspect_s0_source_build_r2_failure') as access:
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' EXTERNAL_MET_CONDA_BIN='+shlex.quote(os.environ['EXTERNAL_MET_CONDA_BIN'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(remote_code())
        result=json.loads(access.ssh(command,timeout=180).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'],remote_utc=result['remote_utc'])),flush=True)


if __name__=='__main__':main()
