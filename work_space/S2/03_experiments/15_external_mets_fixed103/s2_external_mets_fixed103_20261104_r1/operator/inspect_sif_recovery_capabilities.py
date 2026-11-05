"""Fresh artifact identities and layer tombstones; no extraction or job creation."""
import hashlib
import inspect
import json
import os
import shlex

from session import session, access_module
from inspect_sif_hardlink_failure import scan_headers, FAILED_MEMBER


CODE = r'''
import shutil,subprocess,tarfile,time
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
parent=root/'attempts/s0_cpu_prepare_r3a'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def command(argv):
    p=subprocess.run(argv,capture_output=True,text=True,timeout=30)
    return dict(returncode=p.returncode,stdout=p.stdout[:12000],stderr=p.stderr[:4000])
assert sha(parent/'reports/FAILED.json')=='8a5d33b7f6900d27c8f9c282c82b08cd373330ca3dcb2f321a1520a53c4c4432'
manifest=parent/'reports/WHEELS_COMPLETE.json'
assert sha(manifest)=='79b288745b7912cdd3b4ddebe4782ee4b8a66d96c8fd159bec6a0440394eb6d5'
wheels=json.loads(manifest.read_text())['wheels']
expected={
 'numexpr':(174132,'f623c397a6b879ae06d8bf034c047fe08833bf2abedb235e626fe493e4871a47'),
 'connected-components-3d':(1030729,'0f5390389f76e21b96d8bb457f357bddd2c6679252876ad6ff581e79e3b96a52'),
 'ndindex':(175585,'66c113d5220590899a8433b766cab7f1df0d2f4098f62ba770241169196e9266')}
assert len(wheels)==3 and {w['name'] for w in wheels}==set(expected)
for item in wheels:
    p=Path(item['path'])
    assert p.is_relative_to(parent/'wheels') and p.is_file() and not p.is_symlink()
    assert (p.stat().st_size,sha(p))==expected[item['name']]==(item['bytes'],item['sha256'])
contract=json.loads((parent/'contract/PREPARATION_CONTRACT.json').read_text())
archive=Path(contract['archive'])
assert archive==root/'assets/container_conversion/fixed2025.docker-archive.tar'
assert archive.stat().st_size==8409948160 and not archive.is_symlink()
archive_sha=sha(archive)
assert archive_sha=='731b17a23514bf2460083d4585a42d5ece4137643960c354469790857af52735'
target='opt/conda/lib/python3.11/site-packages/pip-25.1.1.dist-info/INSTALLER'
wanted={FAILED_MEMBER,target}
for path in list(wanted):
    p=PurePosixPath(path)
    wanted.add(str(p.parent/('.wh.'+p.name)))
    for parent_path in p.parents:
        wanted.add(str(parent_path/'.wh..wh..opq'))
        if str(parent_path)!='.':wanted.add(str(parent_path.parent/('.wh.'+parent_path.name)))
deadline=time.monotonic()+120
layers=[]
with tarfile.open(archive,mode='r:') as outer:
    config=json.load(outer.extractfile('manifest.json'))
    assert len(config)==1 and len(config[0]['Layers'])==11
    for index,name in enumerate(config[0]['Layers']):
        assert name==str(index)+'/layer.tar'
        with outer.extractfile(name) as stream:result=scan_headers(stream,wanted,deadline)
        result.update(index=index)
        layers.append(result)
tools={n:shutil.which(n) for n in ('umoci','skopeo','bsdtar','tar','buildah','podman','runc','go','singularity','apptainer')}
out=dict(status='fresh_wheels_archive_and_container_tools_observed_not_compatibility_pass',
    started_remote_utc=STARTED_UTC,remote_utc=datetime.now(timezone.utc).isoformat(),
    archive_sha256=archive_sha,wheel_files_SHA_verified=True,wheels=wheels,layers=layers,tools=tools,
    backend=contract['backend'],backend_version=command([contract['backend'],'version']),
    available_modules=command(['bash','-lc','module -t avail singularity apptainer umoci skopeo buildah podman 2>&1']),
    free_bytes=shutil.disk_usage(root).free,quota=command(['quota','-s']),
    remote_writes=False,conversion_retried=False,jobs_created=False,model_loaded=False,MRI_or_GT_read=False)
assert sha(parent/'reports/FAILED.json')=='8a5d33b7f6900d27c8f9c282c82b08cd373330ca3dcb2f321a1520a53c4c4432'
print(json.dumps(out))
'''


def source():
    result = '\n'.join(('import hashlib,json,os',
        'from datetime import datetime,timezone\nSTARTED_UTC=datetime.now(timezone.utc).isoformat()',
        'import tarfile,time\nfrom pathlib import PurePosixPath',
        'FAILED_MEMBER='+repr(FAILED_MEMBER), inspect.getsource(scan_headers), CODE))
    compile(result, 'sif_recovery_capabilities', 'exec')
    return result


def main():
    root = access_module.ROOT
    authorization = json.loads((root/'S0_SIF_RECOVERY_AUTHORIZATION_20261105.json').read_text())
    assert authorization['independent_same_image_compatibility_recovery_authorized']
    code = source()
    with session('inspect_sif_recovery_capabilities') as access:
        access_module.exclusive(access.audit/'EXECUTED_SOURCE.json', dict(
            sha256=hashlib.sha256(code.encode()).hexdigest(),source=code))
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(code)
        result=json.loads(access.ssh(command,timeout=300).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'],remote_utc=result['remote_utc'])),flush=True)


if __name__=='__main__':
    main()
