"""Bounded tar-header diagnosis only; never extracts or retries conversion."""
import hashlib
import io
import json
import os
import shlex
import tarfile
import time
from pathlib import PurePosixPath

from session import session, access_module

FAILED_MEMBER = 'opt/conda/pkgs/pip-25.1.1-pyh8b19718_0/site-packages/pip-25.1.1.dist-info/INSTALLER'


def scan_headers(stream, wanted, deadline):
    found = []
    count = 0
    with tarfile.open(fileobj=stream, mode='r:') as layer:
        for member in layer:
            count += 1
            if count > 300000 or time.monotonic() > deadline:
                raise RuntimeError('Tar-header diagnosis budget exceeded; no extraction')
            name = str(PurePosixPath(member.name))
            if name in wanted or name.endswith('/pip-25.1.1.dist-info/INSTALLER'):
                found.append(dict(name=name, type=member.type.decode('ascii'),
                    linkname=member.linkname, size=member.size,
                    header_offset=member.offset, data_offset=member.offset_data))
                if len(found) > 64:
                    raise RuntimeError('Relevant tar-header count budget exceeded')
    return dict(header_count=count, relevant=found)


def local_check():
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode='w') as archive:
        regular = tarfile.TarInfo('a/INSTALLER')
        regular.size = 1
        archive.addfile(regular, io.BytesIO(b'x'))
        link = tarfile.TarInfo(FAILED_MEMBER)
        link.type = tarfile.LNKTYPE
        link.linkname = 'a/INSTALLER'
        archive.addfile(link)
    payload.seek(0)
    result = scan_headers(payload, {FAILED_MEMBER, 'a/INSTALLER'}, time.monotonic()+5)
    assert result['header_count'] == 2
    assert result['relevant'][1]['linkname'] == 'a/INSTALLER'
    assert result['relevant'][1]['type'] == '1'


CODE = r'''
from pathlib import Path
import stat
from datetime import datetime, timezone
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
attempt=root/'attempts/s0_cpu_prepare_r3a'
failed=attempt/'reports/FAILED.json'
expected_failure='8a5d33b7f6900d27c8f9c282c82b08cd373330ca3dcb2f321a1520a53c4c4432'
assert hashlib.sha256(failed.read_bytes()).hexdigest()==expected_failure
contract=json.loads((attempt/'contract/PREPARATION_CONTRACT.json').read_text())
archive_path=Path(contract['archive'])
assert archive_path==root/'assets/container_conversion/fixed2025.docker-archive.tar'
assert archive_path.is_file() and not archive_path.is_symlink()
assert archive_path.stat().st_size==8409948160
deadline=time.monotonic()+120
observations=[]
with tarfile.open(archive_path,mode='r:') as outer:
    config=outer.extractfile('manifest.json')
    assert config is not None
    manifest=json.load(config)
    assert len(manifest)==1 and len(manifest[0]['Layers'])==11
    wanted={FAILED_MEMBER}
    for index,name in enumerate(manifest[0]['Layers']):
        assert name==str(index)+'/layer.tar'
        stream=outer.extractfile(name)
        assert stream is not None
        with stream:
            item=scan_headers(stream,wanted,deadline)
        item.update(layer_index=index,layer_name=name)
        observations.append(item)
        for header in item['relevant']:
            if header['name']==FAILED_MEMBER and header['type']=='1':
                target=PurePosixPath(header['linkname'])
                assert not target.is_absolute() and '..' not in target.parts
                wanted.add(str(target))
    # Rescan only if a discovered link target was not in the original filter.
    extra=wanted-{FAILED_MEMBER}
    if extra:
        for index,name in enumerate(manifest[0]['Layers']):
            stream=outer.extractfile(name)
            with stream:
                more=scan_headers(stream,extra,deadline)
            observations[index]['link_target_headers']=[h for h in more['relevant'] if h['name'] in extra]
partial=attempt/'conversion/build-temp-1939842888/rootfs'
paths={FAILED_MEMBER,*extra}
partial_state={}
for name in sorted(paths):
    path=partial/name
    try:
        info=path.lstat()
        item=dict(exists=True,size=info.st_size,mode=stat.S_IFMT(info.st_mode),is_symlink=path.is_symlink())
        if path.is_symlink():item['linkname']=os.readlink(path)
    except FileNotFoundError:
        item=dict(exists=False)
    partial_state[name]=item
assert hashlib.sha256(failed.read_bytes()).hexdigest()==expected_failure
print(json.dumps(dict(status='read_only_container_header_diagnosis_not_conversion_pass',
    remote_utc=datetime.now(timezone.utc).isoformat(),job_id='4304783',
    failure_sha256=expected_failure,archive_bytes=archive_path.stat().st_size,
    current_full_archive_sha_recomputed=False,layers=observations,
    partial_root_exists=partial.is_dir(),partial_state=partial_state,
    conversion_retried=False,remote_writes=False,jobs_created=False,
    model_loaded=False,MRI_or_GT_read=False)))
'''


def main():
    import inspect
    local_check()
    source = '\n'.join((
        'import hashlib,json,os,tarfile,time\nfrom pathlib import PurePosixPath',
        'FAILED_MEMBER='+repr(FAILED_MEMBER), inspect.getsource(scan_headers), CODE))
    compile(source, 'read_only_container_tar_headers', 'exec')
    with session('inspect_sif_hardlink_failure') as access:
        access_module.exclusive(access.audit/'EXECUTED_SOURCE.json', dict(
            sha256=hashlib.sha256(source.encode()).hexdigest(),source=source,
            local_synthetic_header_test_pass=True))
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(source)
        result=json.loads(access.ssh(command,timeout=180).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'],
            remote_utc=result['remote_utc'])),flush=True)


if __name__=='__main__':
    main()
