"""One read-only snapshot of the independently submitted, unchanged CPU attempt."""
import json
import os
import shlex

import snapshot_s0_cpu_r2 as original
from session import session, access_module
from s0_cpu_r2_sources import replace_one


def remote_code():
    code = replace_one(original.remote_code(), "attempt/'operator/SUBMISSION_RECEIPT.json'",
                       "attempt/'operator/submit_usage_r1/SUBMISSION_RECEIPT.json'")
    compile(code, 'single_read_only_CPU_usage_submission_snapshot', 'exec')
    return code


def main():
    with session('snapshot_s0_cpu_usage_r1') as access:
        command = 'env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command += ' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(remote_code())
        result = json.loads(access.ssh(command,timeout=900).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'],job_id=result['job_id'],remote_utc=result['remote_utc'],drift=result['drift'])),flush=True)


if __name__=='__main__':
    main()
