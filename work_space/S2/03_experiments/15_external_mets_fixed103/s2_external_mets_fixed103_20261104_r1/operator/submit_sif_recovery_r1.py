"""One held CPU recovery submission after real compatibility and local verification."""
import hashlib
import json
import os
import shlex

import submit_s0_cpu_r3 as base
from session import session,access_module
from s0_sif_recovery_sources import ROOT,runtime_files,common_source,protocol_source


def preparation_payload():
    authorization=json.loads((ROOT/'S0_SIF_RECOVERY_AUTHORIZATION_20261105.json').read_text())
    assert authorization['independent_same_image_compatibility_recovery_authorized']
    assert authorization['single_new_CPU_held_submission_after_compatibility_verification_authorized']
    assert not authorization['recompile_or_redownload_authorized']
    authorization.update(single_R3_CPU_held_submission_and_release_authorized=True,
                         environment_reinstallation_authorized=False,solver_rerun_authorized=False)
    proof=base.preparation_payload()['bootstrap_proof']
    directory=ROOT/'operator/verify_sif_hardlink_compatibility_20261105T084241Z'
    result=json.loads((directory/'RESULT.json').read_text())
    assert result['normalized_native_probe_pass'] and result['original_error_reproduced']
    assert not result['original_image_full_conversion_attempted'] and not result['jobs_created']
    normalizer_sha=hashlib.sha256((ROOT/'runtime/normalize_oci_hardlinks.py').read_bytes()).hexdigest()
    assert normalizer_sha==result['normalizer_source_sha256']
    proof['compatibility_verification']=dict(remote_complete_sha256=hashlib.sha256(base.encode_json(result)).hexdigest(),
        local_result_sha256=hashlib.sha256((directory/'RESULT.json').read_bytes()).hexdigest(),
        normalizer_source_sha256=normalizer_sha,minimal_native_pass=True,full_SIF_pass=False,GPU_pass=False)
    return dict(authorization=authorization,bootstrap_proof=proof,files=runtime_files())


def remote_code(action,payload):
    assert action in ('prepare_held','release_held')
    code=common_source()+'\n'+protocol_source()+'\nprint(json.dumps(execute('+repr(action)+','+repr(payload)+')))\n'
    compile(code,'SIF_recovery_'+action,'exec')
    assert len(code.encode())<120000,'remote command source too large'
    return code


def snapshot_code(job_id):
    from snapshot_s0_cpu_r2 import CODE
    from s0_cpu_r2_sources import replace_one
    code=CODE.replace("root/'attempts/s0_cpu_prepare_r2'", "root/'attempts/s0_sif_recovery_r1'")
    code=replace_one(code,"job_id=receipt['job_id']","job_id=receipt['job_id']\nrequire(job_id=="+repr(job_id)+",'recovery JobID drift')")
    # The detailed normalization ledger stays remote; snapshots carry only its identity.
    line=next(line for line in code.splitlines() if 'CPU report too large' in line)
    code=replace_one(code,line,
        "    if path.name=='NORMALIZATION_COMPLETE.json':\n"
        "        reports[path.name]=dict(bytes=path.stat().st_size,sha256=sha(path),content_not_transferred=True)\n"
        "        continue\n"+line)
    start=code.index("    wheels_path=attempt/'reports/WHEELS_COMPLETE.json'")
    end=code.index('result=dict(META',start)
    code=code[:start]+'''    require(sha(attempt/'reports/WHEELS_REUSED.json')==complete['reused_wheels_sha256'],'reused wheels report drift')
    require(sha(attempt/'reports/NORMALIZATION_COMPLETE.json')==complete['sif']['normalization_report_sha256'],'normalization ledger drift')
    for item in [complete['sif'],*contract['reused_wheels']]:
        path=Path(item['path'])
        observed=sha(path) if path.is_file() and not path.is_symlink() else None
        valid=path.is_file() and path.stat().st_size==item['bytes'] and observed==item['sha256']
        artifact_identity[str(path)]=dict(expected_bytes=item['bytes'],observed_bytes=path.stat().st_size if path.is_file() else None,
            expected_sha256=item['sha256'],observed_sha256=observed,valid=valid)
        if not valid:drift.append(str(path))
'''+code[end:]
    code=code.replace('S0_CPU_artifacts_complete_runtime_environment_still_requires_consumption',
                      'S0_recovery_artifacts_complete_runtime_and_GPU_still_require_validation')
    code=code.replace('read_only_CPU_R2_observation_not_inference_pass','read_only_SIF_recovery_observation_not_inference_pass')
    code=replace_one(code,'artifact_identity={}',"recovery_guard(attempt,contract)\nartifact_identity={}")
    source=runtime_files()['s0_build_contract.py']+'\n'+code
    compile(source,'SIF_recovery_snapshot','exec')
    return source


def main():
    payload=preparation_payload()
    with session('submit_sif_recovery_r1') as access:
        held=base.invoke(access,'prepare_held',payload,code_factory=remote_code,runtime_factory=runtime_files)
        if held['status']!='R3_CPU_submitted_held_not_released':raise RuntimeError('recovery stopped before held release; no automatic retry')
        verification=base.verify_local_held(held,runtime_factory=runtime_files,helper_factory=common_source)
        access_module.exclusive(access.audit/'LOCAL_HELD_VERIFICATION.json',verification)
        released=base.invoke(access,'release_held',verification,code_factory=remote_code,runtime_factory=runtime_files)
        if released['status']!='CPU_preparation_job_released_not_artifact_or_model_runtime_pass':raise RuntimeError('recovery release stopped; preserve without retry')
        source=snapshot_code(released['job_id'])
        access_module.exclusive(access.audit/'SNAPSHOT_EXECUTED_SOURCE.json',dict(source=source,sha256=hashlib.sha256(source.encode()).hexdigest()))
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(source)
        result=json.loads(access.ssh(command,timeout=600).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'SNAPSHOT.json',result)
        print(json.dumps(dict(audit=str(access.audit),job_id=released['job_id'],status=released['status'],
            remote_utc=result['remote_utc'],scheduler=result['scheduler'],drift=result['drift'])),flush=True)


if __name__=='__main__':main()
