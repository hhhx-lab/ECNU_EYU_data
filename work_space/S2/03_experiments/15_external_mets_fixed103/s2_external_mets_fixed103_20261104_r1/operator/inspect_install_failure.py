"""One bounded read of this batch's failed dependency command."""
import json
import os
import shlex
from session import session,access_module

CODE=r'''
import json,os
from pathlib import Path
from datetime import datetime,timezone
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
out=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed',
    status='installation_failure_read_only_inspection',remote_utc=datetime.now(timezone.utc).isoformat(),documents={})
for name in ['operator/ENV_INSTALL_STARTED.json','operator/ENV_INSTALL_FAILED.json','operator/BIO_SOURCE_IDENTITY.json',
    'logs/conda_create.log','logs/torch_cu124_install.log']:
    p=root/name
    if p.is_file():
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-14000));text=f.read().decode(errors='replace')
        out['documents'][name]=dict(bytes=p.stat().st_size,tail=text)
out['scope']=dict(remote_writes=False,jobs_created=False,environment_reinstalled=False)
print(json.dumps(out))
'''


def main():
    with session('inspect_install_failure') as access:
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(CODE)
        result=json.loads(access.ssh(command,timeout=120).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'])))


if __name__=='__main__':main()
