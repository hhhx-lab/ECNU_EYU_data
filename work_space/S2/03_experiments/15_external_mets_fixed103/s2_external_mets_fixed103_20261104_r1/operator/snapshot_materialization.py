"""One bounded read-only materialization snapshot; no asset content or retry."""
import json
import os
import shlex
from session import session, access_module


def snapshot_code(root):
    code = '''from pathlib import Path
import json,shutil,os
from datetime import datetime,timezone
root=Path(''' + repr(root) + ''')
audit=root/'operator/asset_materialization'
out=dict(remote_utc=datetime.now(timezone.utc).isoformat(),free_bytes=shutil.disk_usage(root).free,
         markers={},asset_content_read=False,remote_writes=False,jobs_created=False)
for name in ['STARTED.json','FAILED.json','COMPLETE.json','BIO_FULL_MODEL_IDENTITY.json',
             'OCI_EXPANDED_IDENTITIES.json','DOCKER_ARCHIVE_IDENTITY.json','NATIVE_BUILD_COMMANDS.json']:
    p=audit/name
    if p.is_file():
        assert p.stat().st_size<200000
        try:out['markers'][name]=json.loads(p.read_text())
        except json.JSONDecodeError:out['markers'][name]=dict(status='incomplete_metadata_at_snapshot')
    else:out['markers'][name]=None
p=audit/'SIF_BUILD.log'
if p.is_file():
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-8000))
        out['sif_log_tail']=f.read().decode(errors='replace')
out['conversion_paths']={}
for name in ['fixed2025.docker-archive.tar','native_rootfs','native_rootfs.squashfs','fixed2025.part.sif']:
    p=root/'assets/container_conversion'/name
    out['conversion_paths'][name]=dict(exists=p.exists(),is_file=p.is_file(),
                                     bytes=p.stat().st_size if p.is_file() else None)
out['matching_owned_processes']=[]
needles=[str(root/'assets/container_conversion'),str(root/'operator/s0_runtime/materialize_assets.py')]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
    try:
        if proc.stat().st_uid!=os.getuid():continue
        command=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
        if any(needle in command for needle in needles):
            out['matching_owned_processes'].append(dict(pid=int(proc.name),command=command[:2000]))
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
print(json.dumps(out))
'''
    compile(code, 'generated_materialization_snapshot', 'exec')
    return code


def main():
    code = snapshot_code(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
    with session('snapshot_materialization') as access:
        py = shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])
        result = json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + py +
                                      ' -B -c ' + shlex.quote(code), timeout=60).strip().splitlines()[-1])
        result['local_received_utc'] = access_module.datetime.now(access_module.timezone.utc).isoformat()
        access_module.exclusive(access.audit/'RESULT.json', dict(access_module.META, **result))
        print(json.dumps(dict(audit=str(access.audit),remote_utc=result['remote_utc'],
                              free_bytes=result['free_bytes'],markers_present=[
                                  k for k,v in result['markers'].items() if v is not None])),flush=True)


if __name__ == '__main__':
    main()
