"""Bounded R3 failure diagnosis; no installations, models, or job operations."""
import json
import os
import shlex
from session import session, access_module

CODE = r'''
import hashlib,json,os,re,subprocess,shutil
from datetime import datetime,timezone
from pathlib import Path
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
audit=root/'operator/environment_r3'
failed=json.loads((audit/'FAILED.json').read_text())
assert failed['stage']=='author_dependencies'
log=root/'logs/environment_r3/author_dependencies.log'
assert log.stat().st_size<2*1024**2
text=log.read_text()
pattern=r'numexpr|connected.components.3d|ndindex|ERROR:|error:|Successfully built|Failed to build'
lines=[line for line in text.splitlines() if re.search(pattern,line,re.I)]
env=root/'envs/bio_intel_py310_r3'
probe="import importlib.metadata as m,json; result={};\nfor n in ['pandas','numpy','torch','torchvision','numexpr','connected-components-3d','ndindex']:\n try: result[n]=m.version(n)\n except m.PackageNotFoundError: result[n]=None\nprint(json.dumps(result))"
versions=json.loads(subprocess.check_output([str(env/'bin/python'),'-B','-c',probe],text=True,timeout=30))
out=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed',
    status='environment_r3_failure_diagnosed_no_retry',failure=failed,versions=versions,
    host_libc=subprocess.check_output(['getconf','GNU_LIBC_VERSION'],text=True).strip(),
    error_lines=lines,author_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
    free_bytes=shutil.disk_usage(root).free,remote_utc=datetime.now(timezone.utc).isoformat(),
    models_loaded=False,dependencies_installed=False,jobs_created=False)
print(json.dumps(out))
'''


def main():
    with session('inspect_environment_r3') as access:
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(CODE)
        result=json.loads(access.ssh(command,timeout=90).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'],versions=result['versions'],
                              remote_utc=result['remote_utc'])),flush=True)


if __name__=='__main__':main()
