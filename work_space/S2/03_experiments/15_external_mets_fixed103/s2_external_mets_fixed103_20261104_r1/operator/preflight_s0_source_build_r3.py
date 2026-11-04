"""Independent parser correction, not executable without a fresh R3 authorization."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import preflight_s0_source_build_r2 as r2
from session import session,access_module

ROOT=Path(__file__).resolve().parents[1]


def remote_code():
    source=r2.CODE
    anchor="audit=root/PREFLIGHT_R2"
    assert source.count(anchor)==1
    source=source.replace(anchor,"""old_r2=root/'operator/s0_source_build_preflight_r2'
require(json.loads((old_r2/'FAILED.json').read_text())['message']=='effective alias drift','R2 failure contract drift')
require(not (old_r2/'SOLVER.json').exists(),'R2 solver unexpectedly executed')
old_r2_guard={p.name:sha(p) for p in old_r2.iterdir() if p.is_file()}
audit=root/PREFLIGHT_R3""")
    anchor='    effective=validate_effective_config(observed,audit)'
    assert source.count(anchor)==1
    source=source.replace(anchor,"    write(audit/'EFFECTIVE_CONFIG.json',effective_config_summary(observed))\n"+anchor)
    anchor="    require(not (root/'envs'/BUILD_ENV).exists(),'dry-run created environment')"
    assert source.count(anchor)==1
    source=source.replace(anchor,"    require({p.name:sha(p) for p in old_r2.iterdir() if p.is_file()}==old_r2_guard,'R2 audit drift')\n"+anchor)
    code=(ROOT/'runtime/s0_build_contract.py').read_text()+'\n'
    code+=(ROOT/'operator/s0_source_build_r2_contract.py').read_text()+'\n'
    code+=(ROOT/'operator/s0_source_build_r3_contract.py').read_text()+'\n'+source
    ast.parse(code)
    return code


def main():
    authorization=json.loads((ROOT/'S0_SOURCE_BUILD_R3_AUTHORIZATION.json').read_text())
    assert authorization['independent_R3_parser_correction_authorized'] is True
    assert not authorization['global_configuration_modified'] and not authorization['automatic_retry']
    with session('preflight_s0_source_build_r3') as access:
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' EXTERNAL_MET_CONDA_BIN='+shlex.quote(os.environ['EXTERNAL_MET_CONDA_BIN'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(remote_code())
        result=json.loads(access.ssh(command,timeout=1500).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'],remote_utc=result['remote_utc'])),flush=True)
        if result['status']!='s0_source_build_preflight_pass_no_install_no_job':
            raise RuntimeError('R3 compiler preflight failed; preserve evidence; no install/job/retry')


if __name__=='__main__':main()
