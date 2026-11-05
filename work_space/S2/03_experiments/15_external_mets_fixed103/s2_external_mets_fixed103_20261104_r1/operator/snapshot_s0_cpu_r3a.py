"""One read-only observation of the accepted R3a job; never resubmit."""
import hashlib
import json
import os
import shlex

from session import session, access_module
from submit_s0_cpu_r3a import snapshot_code


def main():
    source=snapshot_code('4304783')
    compile(source,'read_only_R3a_snapshot','exec')
    with session('snapshot_s0_cpu_r3a') as access:
        access_module.exclusive(access.audit/'EXECUTED_SOURCE.json',dict(
            sha256=hashlib.sha256(source.encode()).hexdigest(),source=source))
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(source)
        result=json.loads(access.ssh(command,timeout=900).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),job_id=result['job_id'],
                              remote_utc=result['remote_utc'],drift=result['drift'])),flush=True)


if __name__=='__main__':
    main()
