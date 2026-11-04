"""One authorized independent R2 staging using the accepted lock and R1 sdists."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import stage_s0_cpu_r1 as r1
from session import session,access_module
from s0_cpu_r2_sources import ROOT,pinned,replace_one,runtime_files


def remote_code():
    pinned('operator/stage_s0_cpu_r1.py')
    files=runtime_files()
    body=replace_one(r1.ENVIRONMENT_CODE,"before=prior_guard(root)","before=prior_guard(root)\nparent_before=parent_staging_guard(root)")
    start=body.index('    for name,item in SOURCES.items():\n')
    end=body.index('    configurations=',start)
    body=body[:start]+'''    for name,item in SOURCES.items():
        source=root/'attempts/s0_cpu_prepare_r1/sources'/item['filename']
        require(source.stat().st_size==item['bytes'] and sha(source)==item['sha256'],'parent sdist identity drift')
        path=attempt/'sources'/item['filename']
        with source.open('rb') as src, path.open('xb') as dst:shutil.copyfileobj(src,dst)
        path.chmod(0o440)
'''+body[end:]
    body=body.replace('import prepare_s0_cpu_r1 as p,json','import prepare_s0_cpu_r2 as p,json')
    body=replace_one(body,"    require(prior_guard(root)==before,'old environment/failure changed during staging')",
                     "    require(prior_guard(root)==before,'old environment/failure changed during staging')\n"
                     "    require(parent_staging_guard(root)==parent_before,'parent CPU staging changed during R2')")
    body=replace_one(body,'package_files=package,prior_guard=before,','package_files=package,prior_guard=before,parent_attempt_files=parent_before,')
    helpers='\n'.join((ROOT/'operator'/name).read_text() for name in
                      ('s0_source_build_r2_contract.py','s0_source_build_r3_contract.py','s0_source_requirements.py','s0_source_requirements_r2.py'))
    identities={name:hashlib.sha256(source.encode()).hexdigest() for name,source in files.items()}
    code=files['s0_build_contract.py']+'\n'+helpers+'\nexpected_runtime='+repr(identities)+'\n'+body
    compile(code,'S0_independent_R2_staging','exec')
    return code


def main():
    authorization=json.loads((ROOT/'S0_STAGING_R2_AUTHORIZATION.json').read_text())
    assert authorization['independent_CPU_preparation_R2_authorized'] and not authorization['solver_rerun_authorized']
    remote=os.environ['EXTERNAL_MET_REMOTE_ROOT']
    files=runtime_files()
    bootstrap=files['s0_build_contract.py']+'\nroot=Path('+repr(remote)+')\nfiles='+repr(files)+'''
require(parent_staging_guard(root),'parent CPU staging unavailable')
require(not (root/'envs/bio_build_glibc217_r1').exists(),'R1 build environment unexpectedly exists')
require(not (root/'envs/bio_build_glibc217_r2').exists(),'R2 build environment already exists; no retry')
require(json.loads((root/'operator/s0_scheduler_acceptance_r1/COMPLETE.json').read_text())['status']=='s0_scheduler_acceptance_pass_same_R3_plan_no_install_no_job','independent acceptance incomplete')
attempt=root/'attempts/s0_cpu_prepare_r2'
require(not attempt.exists(),'R2 CPU attempt already exists; no retry')
attempt.mkdir(mode=0o700)
for name in ['runtime','contract','sources','logs','reports','wheels','wheel_probe','scratch','operator','slurm']:(attempt/name).mkdir(mode=0o700)
for name,source in files.items():
    compile(source,name,'exec')
    path=attempt/'runtime'/name
    with path.open('x') as stream:stream.write(source)
    path.chmod(0o440)
print(json.dumps(dict(status='exclusive_R2_CPU_staging_created',attempt=str(attempt))))
'''
    compile(bootstrap,'S0_R2_exclusive_bootstrap','exec')
    with session('stage_s0_cpu_r2') as access:
        access_module.exclusive(access.audit/'GENERATED_SOURCE_IDENTITY.json',dict(
            parent_R1_sources_pinned=True,files={name:hashlib.sha256(source.encode()).hexdigest() for name,source in files.items()}))
        result=json.loads(access.ssh(shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(bootstrap),timeout=240).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'BOOTSTRAP.json',result)
        for name in ('S0_BUILD_CONVERSION_AUTHORIZATION.json','S0_SCHEDULER_R1_AUTHORIZATION.json','S0_STAGING_R2_AUTHORIZATION.json'):
            access.scp(ROOT/name,remote+'/attempts/s0_cpu_prepare_r2/contract/')
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(remote)
        command+=' EXTERNAL_MET_CONDA_BIN='+shlex.quote(os.environ['EXTERNAL_MET_CONDA_BIN'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(remote_code())
        result=json.loads(access.ssh(command,timeout=2200).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'],remote_utc=result['remote_utc'])),flush=True)
        if result['status']!='S0_offline_CPU_input_staging_pass_not_model_runtime_pass':
            raise RuntimeError('R2 CPU staging failed; preserve all attempts without retry or submission')


if __name__=='__main__':
    main()
