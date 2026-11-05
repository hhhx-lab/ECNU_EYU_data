"""One authorized same-Python compatibility probe, not a full image conversion."""
import hashlib
import json
import os
import shlex

from session import session, access_module
from sif_empty_layer_sources import ROOT, ORIGINAL_SHA256, candidate_source

AUDIT = "sif_empty_layer_verification_r2"
FAILED_SHA256 = "526085de8e5a3209ab43cd38cce22826aad54e1ea5a89f4a8fe338da79ad2e9d"

CODE = r'''
import os, sys, subprocess, shutil, types
from datetime import datetime, timezone
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
parent=root/'attempts/s0_sif_recovery_r1'
require(sha(parent/'reports/FAILED.json')==FAILED_SHA256,'failed recovery marker drift')
require(sha(parent/'contract/PREPARATION_CONTRACT.json')=='d957f9e1240186436b7e8ecb607a0f47ef52e7be2b1303cd384959bffe428784','failed recovery contract drift')
contract=json.loads((parent/'contract/PREPARATION_CONTRACT.json').read_text())
require(sys.version_info[:2]==(3,10),'same Python3.10 required')
require(Path(sys.executable).resolve()==(Path(contract['build_env'])/'bin/python').resolve(),'accepted build interpreter drift')
for name,digest in contract['package_files'].items():require(sha(parent/name)==digest,'failed recovery package drift')
require(sha(parent/'runtime/normalize_oci_hardlinks.py')==ORIGINAL_SHA256,'original normalizer drift')
audit=root/'operator'/AUDIT
require(not audit.exists(),'verification already attempted; no automatic retry')
audit.mkdir(mode=0o700)
stage='same_Python310_empty_layer_verification'
try:
    write(audit/'STARTED.json',dict(remote_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,
        candidate_source_sha256=CANDIDATE_SHA256,models_loaded=False,images_read=False))
    def layer_bytes(members):
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w',format=tarfile.PAX_FORMAT) as archive:
            for name,kind,data in members:
                item=tarfile.TarInfo(name)
                item.mode,item.mtime=0o644,123
                if kind=='link':item.type,item.linkname=tarfile.LNKTYPE,data
                else:item.size=len(data)
                archive.addfile(item,io.BytesIO(data) if kind=='file' else None)
        return buffer.getvalue()
    def fixture(path,layers):
        cfg=dict(os='linux',architecture='amd64',config=dict(Env=['PROBE=1'],Entrypoint=['/bin/sh'],Cmd=['probe']),
            rootfs=dict(type='layers',diff_ids=['sha256:'+hashlib.sha256(x).hexdigest() for x in layers]))
        data=json.dumps(cfg).encode()
        name=hashlib.sha256(data).hexdigest()+'.json'
        with tarfile.open(path,mode='w') as archive:
            add_bytes(archive,name,data)
            add_bytes(archive,'manifest.json',json.dumps([dict(Config=name,RepoTags=['ecnu:empty-layer-probe'],
                Layers=[str(i)+'/layer.tar' for i in range(len(layers))])]).encode())
            for i,block in enumerate(layers):add_bytes(archive,str(i)+'/layer.tar',block)
    original={}
    original_source=(parent/'runtime/normalize_oci_hardlinks.py').read_text().replace('from s0_build_contract import require, sha\n','')
    original.update(require=require,sha=sha)
    exec(compile(original_source,'frozen_original_empty_layer_probe','exec'),original)
    checks=[]
    empty=layer_bytes([])
    empty_path=audit/'empty.tar'
    record=normalize_layer(io.BytesIO(empty),empty_path,time.monotonic()+10,1024**2)
    try:
        original['verify_layer'](io.BytesIO(empty),empty_path,[],time.monotonic()+10)
    except OSError as error:
        require(error.errno==22,'different original failure')
    else:raise RuntimeError('original Python3.10 error was not reproduced')
    checks.append('original_empty_errno22_reproduced')
    require(verify_layer(io.BytesIO(empty),empty_path,[],time.monotonic()+10)==0,'candidate empty layer failed')
    checks.append('candidate_empty_layer_pass')
    empty_archive=audit/'empty.docker-archive.tar'
    fixture(empty_archive,[empty])
    empty_result=normalize_archive(empty_archive,audit/'empty_normalized',time.monotonic()+10,1024**2)
    require(empty_result['layers'][0]['headers']==0,'complete empty Docker layer failed')
    checks.append('complete_empty_Docker_archive_pass')
    layers=[layer_bytes([('layer'+str(i),'file',b'probe')]) for i in range(11)]
    layers[2]=layer_bytes([('pip/INSTALLER','file',b'pip\n'),('cache/INSTALLER','link','pip/INSTALLER')])
    layers[3]=empty
    layers[5]=layer_bytes([('.wh.pip','file',b'')])
    full_fixture=audit/'eleven_layer_fixture.tar'
    fixture(full_fixture,layers)
    before=sha(full_fixture)
    normalized=normalize_archive(full_fixture,audit/'normalized_fixture',time.monotonic()+10,1024**2)
    require(len(normalized['layers'])==11 and normalized['layers'][3]['headers']==0,'eleven-layer empty position drift')
    require(sha(full_fixture)==before and normalized['every_member_bytes_and_metadata_compared'],'fixture source/content drift')
    checks.append('eleven_layers_empty_fourth_hardlinks_whiteout_pass')
    for name,source,derived,message in [
        ('added_member_rejected',empty,layer_bytes([('a','file',b'x')]),'added header'),
        ('missing_member_rejected',layer_bytes([('a','file',b'x')]),empty,'lost header'),
        ('changed_bytes_rejected',layer_bytes([('a','file',b'x')]),layer_bytes([('a','file',b'y')]),'bytes differ')]:
        path=audit/(name+'.tar')
        with path.open('xb') as stream:stream.write(derived)
        try:verify_layer(io.BytesIO(source),path,[],time.monotonic()+10)
        except RuntimeError as error:require(message in str(error),'wrong rejection reason')
        else:raise RuntimeError('candidate weakened verification: '+name)
        checks.append(name)
    expired=audit/'expired.tar'
    with expired.open('xb') as stream:stream.write(layer_bytes([('a','file',b'x')]))
    try:verify_layer(io.BytesIO(layer_bytes([('a','file',b'x')])),expired,[],0)
    except RuntimeError as error:require('time budget' in str(error),'wrong deadline rejection')
    else:raise RuntimeError('candidate deadline guard missing')
    checks.append('expired_deadline_rejected')
    stage='small_same_backend_native_probe'
    backend=contract['backend']
    require(subprocess.check_output([backend,'version'],text=True,timeout=20).strip()=='4.1.1','backend drift')
    env={k:v for k,v in os.environ.items() if not k.startswith(('SINGULARITY','APPTAINER')) and not k.upper().endswith('_PROXY')}
    for name in ('tmp','cache'):(audit/name).mkdir()
    env.update(SINGULARITY_TMPDIR=str(audit/'tmp'),SINGULARITY_CACHEDIR=str(audit/'cache'),LC_ALL='C')
    sandbox=audit/'native_fixture_rootfs'
    command=[backend,'build','--disable-cache','--sandbox','--no-cleanup',str(sandbox),'docker-archive://'+normalized['path']]
    p=subprocess.run(command,env=env,capture_output=True,text=True,timeout=60)
    require(len(p.stdout.encode())+len(p.stderr.encode())<131072,'native probe output budget')
    native=dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    write(audit/'NATIVE_PROBE.json',native)
    require(p.returncode==0,'small same-backend native probe failed')
    require((sandbox/'cache/INSTALLER').read_bytes()==b'pip\n' and not (sandbox/'pip/INSTALLER').exists(),'native hardlink/whiteout content drift')
    require((sandbox/'.singularity.d/runscript').is_file(),'native OCI runscript missing')
    require(sha(parent/'reports/FAILED.json')==FAILED_SHA256,'failed marker changed')
    result=dict(status='same_remote_Python310_empty_layer_and_small_native_probe_pass_not_full_SIF_or_GPU_pass',
        remote_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,python_major_minor=list(sys.version_info[:2]),
        python_executable=sys.executable,checks=checks,all_eight_checks_pass=len(checks)==8,
        original_empty_errno22_reproduced=True,small_same_backend_native_probe_pass=True,
        candidate_source_sha256=CANDIDATE_SHA256,original_source_sha256=ORIGINAL_SHA256,failed_recovery_sha256=FAILED_SHA256,
        fixture_original_sha256=before,fixture_derived_sha256=normalized['sha256'],native_probe=native,
        free_bytes=shutil.disk_usage(root).free,models_loaded=False,images_read=False,jobs_created=False,
        full_image_conversion_attempted=False,packages_installed=False)
    write(audit/'COMPLETE.json',result)
except BaseException as error:
    write(audit/'FAILED.json',dict(stage=stage,error_type=type(error).__name__,message=str(error),
        remote_utc=datetime.now(timezone.utc).isoformat(),no_automatic_retry=True,jobs_created=False))
    raise
print(json.dumps(result))
'''


def source():
    normalizer = candidate_source().replace('from s0_build_contract import require, sha\n', '')
    helpers = r'''
def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()
def write(path,payload):
    with path.open('x') as stream:json.dump(payload,stream,sort_keys=True,indent=2);stream.write('\n')
    path.chmod(0o440)
'''
    code = normalizer + '\n' + helpers
    for name, value in dict(AUDIT=AUDIT, ORIGINAL_SHA256=ORIGINAL_SHA256, FAILED_SHA256=FAILED_SHA256,
                            CANDIDATE_SHA256=hashlib.sha256(candidate_source().encode()).hexdigest()).items():
        code += '\n' + name + '=' + repr(value)
    code += '\n' + CODE
    compile(code, 'same_remote_Python310_verification_r2', 'exec')
    return code


def main():
    authorization = json.loads((ROOT / 'S0_SIF_RECOVERY_R2_AUTHORIZATION_20261105.json').read_text())
    assert authorization['independent_R2_authorized'] and authorization['same_remote_Python310_verification_before_submission_required']
    code = source()
    with session('verify_sif_empty_layer_remote_r2') as access:
        access_module.exclusive(access.audit / 'EXECUTED_SOURCE.json', dict(source=code, sha256=hashlib.sha256(code.encode()).hexdigest()))
        command = 'env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT=' + shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        python = os.environ['EXTERNAL_MET_REMOTE_ROOT'] + '/envs/bio_build_glibc217_r2/bin/python'
        command += ' ' + shlex.quote(python) + ' -B -c ' + shlex.quote(code)
        result = json.loads(access.ssh(command, timeout=240).strip().splitlines()[-1])
        access_module.exclusive(access.audit / 'RESULT.json', result)
        print(json.dumps(dict(audit=str(access.audit), status=result['status'], remote_utc=result['remote_utc'])), flush=True)


if __name__ == '__main__':
    main()
