"""Read installed native packaging interfaces; no build/runtime or policy change."""
import json
import os
import shlex
from session import session,access_module

CODE=r'''
import json,subprocess
from datetime import datetime,timezone
backend='/public/software/singularity-ce/4.1.1/bin/singularity'
out=dict(remote_utc=datetime.now(timezone.utc).isoformat(),commands={})
for name,args in {'version':[backend,'version'],'build_help':[backend,'build','--help'],
    'sif_new_help':[backend,'sif','new','--help'],'sif_add_help':[backend,'sif','add','--help'],
    'mksquashfs_help':['/sbin/mksquashfs','-help']}.items():
    p=subprocess.run(args,capture_output=True,text=True,timeout=20)
    assert len(p.stdout)+len(p.stderr)<32768
    out['commands'][name]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
print(json.dumps(out))
'''


def main():
    with session('inspect_container_build_interface') as access:
        command='env PYTHONDONTWRITEBYTECODE=1 LC_ALL=C '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(CODE)
        result=json.loads(access.ssh(command,timeout=120).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),remote_utc=result['remote_utc'])))


if __name__=='__main__':main()
