"""Bounded real native-backend compatibility test, using only five non-model bytes."""
import hashlib
import json
import os
from pathlib import Path
import shlex

from session import session, access_module
from inspect_sif_hardlink_failure import FAILED_MEMBER

ROOT=Path(__file__).resolve().parents[1]
NORMALIZER=ROOT/'runtime/normalize_oci_hardlinks.py'

CODE=r'''
import os,subprocess,shutil
from datetime import datetime,timezone
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
parent=root/'attempts/s0_cpu_prepare_r3a'
assert sha(parent/'reports/FAILED.json')=='8a5d33b7f6900d27c8f9c282c82b08cd373330ca3dcb2f321a1520a53c4c4432'
archive_path=root/'assets/container_conversion/fixed2025.docker-archive.tar'
assert archive_path.stat().st_size==8409948160
audit=root/'operator/sif_hardlink_compatibility_r1'
require(not audit.exists(),'compatibility attempt exists; no repeat')
audit.mkdir(mode=0o700)
stage='full_header_compatibility_scan'
try:
    write(audit/'STARTED.json',dict(remote_utc=datetime.now(timezone.utc).isoformat(),models_loaded=False,images_read=False))
    deadline=time.monotonic()+120
    inventories=[]
    sample={}
    with tarfile.open(archive_path,mode='r:') as outer:
        manifest=json.load(outer.extractfile('manifest.json'))
        config_raw=outer.extractfile(manifest[0]['Config']).read()
        original_config=json.loads(config_raw)
        require(len(manifest)==1 and len(manifest[0]['Layers'])==11,'original layer count')
        for index,name in enumerate(manifest[0]['Layers']):
            earlier={}
            hardlinks=0
            with outer.extractfile(name) as stream,tarfile.open(fileobj=stream,mode='r:') as layer:
                for member in layer:
                    require(time.monotonic()<deadline and len(earlier)<300000,'header budget')
                    name=canonical(member.name)
                    require(name not in earlier and not member.sparse,'duplicate/sparse header')
                    require(member.type in (tarfile.REGTYPE,tarfile.AREGTYPE,tarfile.DIRTYPE,tarfile.SYMTYPE,tarfile.LNKTYPE),'unsupported special file')
                    if member.islnk():
                        resolve_link(member,earlier)
                        require(member.size==0,'hardlink size')
                        hardlinks+=1
                    if name==TARGET:
                        require(index==2 and member.isfile() and member.size==5,'minimal original target identity')
                        sample['regular']=copy.deepcopy(member)
                        sample['data']=layer.extractfile(member).read()
                    if name==FAILED_MEMBER:sample['link']=copy.deepcopy(member)
                    if name==WHITEOUT:
                        require(index==5 and member.isfile() and member.size==0,'minimal whiteout identity')
                        sample['whiteout']=copy.deepcopy(member)
                    earlier[name]=member
            inventories.append(dict(index=index,headers=len(earlier),hardlinks=hardlinks))
    require(set(sample)=={'regular','data','link','whiteout'} and sample['link'].linkname==TARGET,'same-image fixture incomplete')
    stage='minimal_same_image_native_probe'
    layers=[]
    for members in ([sample['regular'],sample['link']],[sample['whiteout']]):
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w',format=tarfile.PAX_FORMAT) as layer:
            for member in members:
                layer.addfile(member,io.BytesIO(sample['data']) if member.size else None)
        layers.append(buffer.getvalue())
    config=copy.deepcopy(original_config)
    config['rootfs']['diff_ids']=['sha256:'+hashlib.sha256(data).hexdigest() for data in layers]
    raw=json.dumps(config,sort_keys=True,separators=(',',':')).encode()
    config_name=hashlib.sha256(raw).hexdigest()+'.json'
    fixture=audit/'original_header_fixture.tar'
    with tarfile.open(fixture,mode='w') as output:
        add_bytes(output,config_name,raw)
        add_bytes(output,'manifest.json',json.dumps([dict(Config=config_name,RepoTags=['ecnu:header-probe'],Layers=['0/layer.tar','1/layer.tar'])]).encode())
        for index,data in enumerate(layers):add_bytes(output,str(index)+'/layer.tar',data)
    normalized=normalize_archive(fixture,audit/'normalized_fixture',time.monotonic()+30,1024**2)
    backend=json.loads((parent/'contract/PREPARATION_CONTRACT.json').read_text())['backend']
    require(subprocess.check_output([backend,'version'],text=True,timeout=20).strip()=='4.1.1','backend drift')
    env={k:v for k,v in os.environ.items() if not k.startswith(('SINGULARITY','APPTAINER')) and not k.upper().endswith('_PROXY')}
    for name in ('tmp','cache'):(audit/name).mkdir()
    env.update(SINGULARITY_TMPDIR=str(audit/'tmp'),SINGULARITY_CACHEDIR=str(audit/'cache'),LC_ALL='C')
    observations={}
    for name,source in [('original',fixture),('normalized',Path(normalized['path']))]:
        command=[backend,'build','--disable-cache','--sandbox','--no-cleanup',str(audit/(name+'_rootfs')),'docker-archive://'+str(source)]
        p=subprocess.run(command,env=env,capture_output=True,text=True,timeout=60)
        require(len(p.stdout.encode())+len(p.stderr.encode())<131072,'probe output budget')
        observations[name]=dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
        write(audit/(name.upper()+'_NATIVE_PROBE.json'),observations[name])
    require(observations['original']['returncode']!=0 and FAILED_MEMBER in observations['original']['stderr']
            and 'no such file or directory' in observations['original']['stderr'],'original failure not reproduced')
    require(observations['normalized']['returncode']==0,'normalized native probe failed')
    sandbox=audit/'normalized_rootfs'
    require((sandbox/FAILED_MEMBER).read_bytes()==sample['data'],'normalized cached content drift')
    require(not (sandbox/TARGET).exists(),'normalized whiteout not honored')
    require((sandbox/'.singularity.d/runscript').is_file(),'native OCI runscript not created')
    require(sha(parent/'reports/FAILED.json')=='8a5d33b7f6900d27c8f9c282c82b08cd373330ca3dcb2f321a1520a53c4c4432','old failed marker drift')
    result=dict(status='same_image_minimal_hardlink_whiteout_native_compatibility_pass_not_full_SIF_or_GPU_pass',
        remote_utc=datetime.now(timezone.utc).isoformat(),inventories=inventories,
        fixture_source_content_bytes=5,fixture_content_sha256=hashlib.sha256(sample['data']).hexdigest(),
        original_error_reproduced=True,normalized_native_probe_pass=True,
        original_archive_bytes=archive_path.stat().st_size,original_full_SHA_recomputed_in_this_probe=False,
        minimal_normalization=normalized,observations=observations,free_bytes=shutil.disk_usage(root).free,
        normalizer_source_sha256=NORMALIZER_SHA,models_loaded=False,images_read=False,
        original_image_full_conversion_attempted=False,jobs_created=False,packages_installed=False)
    write(audit/'COMPLETE.json',result)
except BaseException as error:
    write(audit/'FAILED.json',dict(stage=stage,error_type=type(error).__name__,message=str(error),
        remote_utc=datetime.now(timezone.utc).isoformat(),no_automatic_retry=True,jobs_created=False))
    raise
print(json.dumps(result))
'''


def source():
    normalizer=NORMALIZER.read_text().replace('from s0_build_contract import require, sha\n','')
    helpers='''
def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def write(path,payload):
    with path.open('x') as stream:json.dump(payload,stream,sort_keys=True,indent=2);stream.write('\\n')
    path.chmod(0o440)
'''
    code=normalizer+'\n'+helpers+'\nFAILED_MEMBER='+repr(FAILED_MEMBER)+'\n'
    code+='TARGET='+repr('opt/conda/lib/python3.11/site-packages/pip-25.1.1.dist-info/INSTALLER')+'\n'
    code+='WHITEOUT='+repr('opt/conda/lib/python3.11/site-packages/.wh.pip-25.1.1.dist-info')+'\n'
    code+='NORMALIZER_SHA='+repr(hashlib.sha256(NORMALIZER.read_bytes()).hexdigest())+'\n'+CODE
    compile(code,'sif_hardlink_compatibility','exec')
    return code


def main():
    authorization=json.loads((ROOT/'S0_SIF_RECOVERY_AUTHORIZATION_20261105.json').read_text())
    assert authorization['independent_same_image_compatibility_recovery_authorized']
    code=source()
    with session('verify_sif_hardlink_compatibility') as access:
        access_module.exclusive(access.audit/'EXECUTED_SOURCE.json',dict(source=code,sha256=hashlib.sha256(code.encode()).hexdigest()))
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(code)
        result=json.loads(access.ssh(command,timeout=360).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'],remote_utc=result['remote_utc'])),flush=True)


if __name__=='__main__':main()
