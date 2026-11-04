"""Authorized independent audit consuming the unchanged R2 environment and Slurm."""
import hashlib
import json
import os
import shlex

import submit_s0_cpu_r2 as original
from session import session, access_module
from s0_cpu_r2_sources import ROOT, replace_one

FAILED_SHA = 'df7ca183f829e0c896002896f1b12e54115c75902842f3d14991f8a2b52dac03'
SLURM_SHA = 'd879a1addf9756b3bfa8561b96e33f600deb87c1b14f9483e79217d8506620ff'
CONTRACT_SHA = '8a2246326f58d7351712785e2395c54fb81949ad7a4390f19e18e80980329980'


def remote_code():
    for name, digest in {
        'submit_s0_cpu_r2.py':'1af3357989de934ca6933472d1fe71831cbeaa11330c939742a1aa618088d77a',
        's0_qos_table.py':'578e9639a09224a0ebb330f2b65bd5b32ccc0debf76444df3e725d61fb0713b7',
    }.items():
        assert hashlib.sha256((ROOT/'operator'/name).read_bytes()).hexdigest() == digest, 'original CPU operator drift'
    authorization = json.loads((ROOT/'S0_CPU_USAGE_SUBMISSION_AUTHORIZATION.json').read_text())
    helpers = (ROOT/'operator/s0_cpu_usage.py').read_text().replace('from s0_cpu_submission_contract import job_fields\n', '')
    code = 'USAGE_AUTHORIZATION='+repr(authorization)+'\n'+original.remote_code()
    code = replace_one(code, 'import os,subprocess,shutil\nfrom datetime', helpers+'\nimport os,subprocess,shutil\nfrom datetime')
    code = replace_one(code, "audit=attempt/'operator'", """old_audit=attempt/'operator'
require(sha(old_audit/'SUBMISSION_FAILED.json')=="""+repr(FAILED_SHA)+""",'original pre-sbatch failure drift')
old_failure=json.loads((old_audit/'SUBMISSION_FAILED.json').read_text())
require(old_failure['job_id'] is None and old_failure['stage']=='CPU_package_and_fresh_scheduler_preflight','original submission state changed')
for name in ['RESERVATION.json','SBATCH_RESULT.json','HELD_CREATED.json','HELD_AUDIT.json','RELEASE_RESULT.json','SUBMISSION_RECEIPT.json']:
    require(not (old_audit/name).exists(),'original job may already exist; no duplicate submission')
old_files={p.name:sha(p) for p in old_audit.iterdir() if p.is_file()}
audit=old_audit/'submit_usage_r1'
require(not audit.exists(),'independent CPU submission already attempted; no retry')
audit.mkdir(mode=0o700)
write(audit/'AUTHORIZATION.json',USAGE_AUTHORIZATION)""")
    code = replace_one(code, "    with path.open('x') as stream:stream.write(script)\n    path.chmod(0o440)",
        "    require(sha(path)=="+repr(SLURM_SHA)+" and path.read_text()==script,'existing frozen Slurm drift')\n"
        "    require(sha(contract_path)=="+repr(CONTRACT_SHA)+",'accepted R2 preparation contract drift')")
    code = replace_one(code, "                assert count==0, 'aggregate QoS TRES requires current usage audit'",
        "                validate_cpu_usage_limit(snapshot,row,key)")
    code = replace_one(code, "    validation=validate_cpu_snapshot(dict(scheduler=scheduler,free_bytes=shutil.disk_usage(root).free))",
        "    snapshot=dict(scheduler=scheduler,free_bytes=shutil.disk_usage(root).free,cpu_usage=collect_cpu_usage(scheduler['own_queue']['stdout']))\n"
        "    write(audit/'RESOURCE_SNAPSHOT.json',snapshot)\n"
        "    validation=validate_cpu_snapshot(snapshot)\n"
        "    validation['cpu_usage']=snapshot['cpu_usage']")
    code = replace_one(code, "    stage='release_original_held_job'",
        "    require({p.name:sha(p) for p in old_audit.iterdir() if p.is_file()}==old_files,'original CPU submission audit changed')\n"
        "    require(sha(path)=="+repr(SLURM_SHA)+" and sha(contract_path)=="+repr(CONTRACT_SHA)+",'frozen identity changed while held')\n"
        "    stage='release_original_held_job'")
    code = replace_one(code, 'models_loaded=False,images_read=False,original_inference_jobs_created=False)',
        'models_loaded=False,images_read=False,original_inference_jobs_created=False,submission_audit=str(audit),original_pre_sbatch_failed_sha256='+repr(FAILED_SHA)+')')
    code = replace_one(code, 'print(json.dumps(dict(META,**outcome)))', """outcome['light_audit_files']={}
for path in sorted(audit.glob('*.json')):
    require(path.stat().st_size<131072,'CPU audit light transfer budget')
    outcome['light_audit_files'][path.name]=dict(bytes=path.stat().st_size,sha256=sha(path),content=json.loads(path.read_text()))
outcome['frozen_slurm']=dict(sha256=sha(attempt/'slurm/prepare.slurm'),content=(attempt/'slurm/prepare.slurm').read_text())
print(json.dumps(dict(META,**outcome)))""")
    compile(code, 'authorized_single_CPU_usage_submission', 'exec')
    return code


def main():
    authorization = json.loads((ROOT/'S0_CPU_USAGE_SUBMISSION_AUTHORIZATION.json').read_text())
    assert authorization['independent_current_QoS_usage_acceptance_authorized'] and authorization['single_first_held_submission_and_same_ID_release_authorized']
    assert not authorization['solver_rerun_authorized'] and not authorization['environment_reinstallation_authorized']
    code = remote_code()
    with session('submit_s0_cpu_usage_r1') as access:
        access_module.exclusive(access.audit/'SOURCE_IDENTITY.json',dict(remote_code_sha256=hashlib.sha256(code.encode()).hexdigest(),
            frozen_slurm_sha256=SLURM_SHA,preparation_contract_sha256=CONTRACT_SHA,original_failed_sha256=FAILED_SHA))
        command = 'env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command += ' EXTERNAL_MET_CONDA_BIN='+shlex.quote(os.environ['EXTERNAL_MET_CONDA_BIN'])
        command += ' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(code)
        result = json.loads(access.ssh(command,timeout=600).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'],job_id=result.get('job_id'),remote_utc=result['remote_utc'])),flush=True)
        if result['status']!='CPU_preparation_job_released_not_artifact_or_model_runtime_pass':
            raise RuntimeError('Independent CPU usage submission stopped; preserve any held job and all failures; no retry')


if __name__=='__main__':
    main()
