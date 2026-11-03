"""One bounded read-only R3 installer/backend inspection, not a polling loop."""
import json
import os
import shlex
from session import session,access_module

CODE=r'''
import json,os,subprocess,shutil
from pathlib import Path
from datetime import datetime,timezone
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
audit=root/'operator/environment_r3'
out=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed',
    remote_utc=datetime.now(timezone.utc).isoformat(),free_bytes=shutil.disk_usage(root).free,markers={},logs={})
for name in ['STARTED.json','COMPLETE.json','FAILED.json']:
    p=audit/name
    if p.exists():
        assert p.stat().st_size<1048576
        data=json.loads(p.read_text())
        out['markers'][name]={k:v for k,v in data.items() if k not in ['selected_packages','original_r2_guard']}
    else:out['markers'][name]=None
for p in sorted((root/'logs/environment_r3').glob('*')):
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-2500));tail=f.read().decode(errors='replace')
    out['logs'][p.name]=dict(bytes=p.stat().st_size,tail=tail)
command="module load singularity-ce/4.1.1; command -v singularity mksquashfs unsquashfs; singularity build --help"
p=subprocess.run(['bash','-lc',command],capture_output=True,text=True,timeout=30)
out['backend_help']=dict(returncode=p.returncode,mksquashfs_args_supported='--mksquashfs-args' in p.stdout,
    docker_archive_documented='docker-archive' in p.stdout,command_paths=p.stdout.splitlines()[:3],stderr=p.stderr[-1000:])
print(json.dumps(out))
'''


def main():
    with session('inspect_s0_progress') as access:
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(CODE)
        result=json.loads(access.ssh(command,timeout=90).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),markers={k:v is not None for k,v in result['markers'].items()},
            logs={k:v['bytes'] for k,v in result['logs'].items()},backend=result['backend_help'],remote_utc=result['remote_utc'])))


if __name__=='__main__':main()
