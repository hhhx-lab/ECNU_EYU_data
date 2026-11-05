"""One bounded read-only snapshot of the already released recovery job."""
import hashlib
import json
import os
import shlex

from session import session,access_module
from submit_sif_recovery_r1 import snapshot_code


def main():
    source=snapshot_code('4304987')
    with session('snapshot_sif_recovery_r1') as access:
        access_module.exclusive(access.audit/'EXECUTED_SOURCE.json',dict(
            source=source,sha256=hashlib.sha256(source.encode()).hexdigest()))
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(source)
        result=json.loads(access.ssh(command,timeout=600).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),job_id=result['job_id'],remote_utc=result['remote_utc'],
            drift=result['drift'],scheduler=result['scheduler'])),flush=True)


if __name__=='__main__':main()
