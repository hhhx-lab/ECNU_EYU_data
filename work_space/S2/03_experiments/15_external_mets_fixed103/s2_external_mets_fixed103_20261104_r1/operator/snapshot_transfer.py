"""One lightweight transfer/capacity snapshot; sizes are not SHA acceptance."""
import json
import os
import shlex
from session import session, access_module
from download_assets import ROOT


def code_for(files):
    return r'''
import json,os,subprocess,shutil
from pathlib import Path
from datetime import datetime,timezone
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
expected='''+repr(files)+r'''
observed=[]
for item in expected:
    p=root/'assets/incoming'/item['logical_path']
    part=Path(str(p)+'.part')
    observed.append(dict(logical_path=item['logical_path'],expected_bytes=item['bytes'],
        target_bytes=p.stat().st_size if p.is_file() else None,
        partial_bytes=part.stat().st_size if part.is_file() else None))
marker=root/'operator/ASSET_TRANSFER_COMPLETE.json'
commands={}
for name,argv in [('df',['df','-B1',str(root)]),('quota',['quota','-s'])]:
    p=subprocess.run(argv,capture_output=True,text=True,timeout=30)
    commands[name]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
print(json.dumps(dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed',
    status='transfer_size_snapshot_not_SHA_acceptance',files=observed,complete_marker_exists=marker.is_file(),
    free_bytes=shutil.disk_usage(root).free,commands=commands,remote_utc=datetime.now(timezone.utc).isoformat(),
    models_loaded=False,images_read=False,remote_writes=False,jobs_created=False)))
'''


def main():
    candidates=list((ROOT/'operator').glob('download_bounded_resume_*/LOCAL_ASSET_MANIFEST.json'))
    assert len(candidates)==1
    manifest=json.loads(candidates[0].read_text())
    files=[{key:item[key] for key in ('logical_path','bytes')} for item in manifest['files']]
    with session('snapshot_transfer') as access:
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(code_for(files))
        result=json.loads(access.ssh(command,timeout=120).strip().splitlines()[-1])
        result['local_received_utc']=access_module.datetime.now(access_module.timezone.utc).isoformat()
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'],free_bytes=result['free_bytes'],
            files=result['files'],remote_utc=result['remote_utc'])),flush=True)


if __name__=='__main__':main()
