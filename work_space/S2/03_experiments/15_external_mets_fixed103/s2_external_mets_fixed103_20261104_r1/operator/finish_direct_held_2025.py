"""Complete the FIRST release of 4310575 after independent parser correction."""
import json
import os
from pathlib import Path
import shlex

import submit_direct_full103 as submit
from session import session, access_module

ROOT=submit.ROOT
ORIGINAL='submit_direct_full103_ext2025_met_gli_20261105T164611Z'
INSPECTION='inspect_direct_held_4310575_20261105T165006Z'


def main():
    original=ROOT/'operator'/ORIGINAL
    payload=json.loads((original/'SUBMISSION_PAYLOAD.json').read_text())
    inspected=json.loads((ROOT/'operator'/INSPECTION/'RESULT.json').read_text())
    fields=submit.validate_held(inspected['raw_scontrol'],'4310575',payload['contract'])
    audit=dict(raw_scontrol=inspected['raw_scontrol'],fields=fields)
    verified=dict(local_held_verified=True,job_id='4310575',held_audit_sha256=submit.digest(submit.encode(audit)))
    original_failure=json.loads((original/'FAILED.json').read_text())
    with session('finish_existing_held_4310575') as access:
        access_module.exclusive(access.audit/'LOCAL_HELD_VERIFICATION.json',dict(**verified,held_audit=audit,
            first_release_only=True,no_sbatch=True,original_failure_sha256=access_module.sha(original/'FAILED.json')))
        remote_path=os.environ['EXTERNAL_MET_REMOTE_ROOT']+'/operator/'+ORIGINAL+'/SUBMISSION_PAYLOAD.json'
        code=submit.code_for('release',remote_path,access_module.sha(original/'SUBMISSION_PAYLOAD.json'),'EXT2025_MET_GLI',verified)
        insertion='''
assert attempt.name=='full103_2025_r1' and json.loads((attempt/'operator/HELD_CREATED.json').read_text())['job_id']=='4310575','existing held identity drift'
assert not (attempt/'operator/RELEASE_RESERVED.json').exists() and not (attempt/'operator/HELD_AUDIT.json').exists() and not (attempt/'operator/HELD_JOB_MAP.json').exists(),'existing audit or release already attempted'
static();held('4310575')
write(attempt/'operator/FIRST_HELD_AUDIT_PARSER_FAILURE.json',ORIGINAL_FAILURE)
write(attempt/'operator/HELD_AUDIT.json',INDEPENDENT_AUDIT)
write(attempt/'operator/HELD_JOB_MAP.json',dict(jobs={'EXT2025_MET_GLI':dict(job_id='4310575')},contract_sha256=sha(attempt/'contract/EXECUTION_CONTRACT.json'),package_sha256=sha(attempt/'contract/PACKAGE_SHA256.json')))
'''
        anchor="stage='before_new_attempt'\n"
        assert code.count(anchor)==1
        code=code.replace(anchor,'ORIGINAL_FAILURE='+repr(original_failure)+'\nINDEPENDENT_AUDIT='+repr(audit)+'\n'+insertion+'\n'+anchor)
        compile(code,'finish_existing_held_4310575','exec')
        access_module.exclusive(access.audit/'EXECUTED_SOURCE.json',dict(source=code,sha256=submit.digest(code)))
        command='env LC_ALL=C PYTHONDONTWRITEBYTECODE=1 '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(code)
        result=json.loads(access.ssh(command,timeout=600).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),result=result)),flush=True)
        if result['status']=='direct_full103_submission_stopped':raise RuntimeError('Existing held completion stopped; no retry')


if __name__=='__main__':main()
