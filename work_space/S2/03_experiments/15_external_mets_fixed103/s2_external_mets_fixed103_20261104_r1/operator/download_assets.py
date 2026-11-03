"""User-authorized temporary Mac staging of exactly the selected model assets."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import time
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
IMAGE='brainles/brats25_met_gli-team'
DIGEST='f83d914d4d7a9d316f7605abaf03fa6adbcf0e86d840d93276e86ad65b2a2a82'
BIO_URL='https://huggingface.co/keramik05/MICCAI-METS_challenge_bi-intel_model/resolve/4a8291024dafb79b4c0cc381438e976a36072201/weights.zip'
BRAIN_URL='https://www.dropbox.com/scl/fo/i51xt63roognvt7vuslbl/AMblt6reQVvlSrORTB3_2lE/BrainIAC.ckpt?rlkey=9w55le6tslwxlfz6c0viylmjb&dl=1'
META=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed')


def utc():return datetime.now(timezone.utc).isoformat()


def write(path,payload):
    with path.open('x',encoding='utf-8') as f:
        json.dump(payload,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def main():
    temp=Path(os.environ['EXTERNAL_MET_LOCAL_TRANSFER_ROOT'])
    parent=Path.home()/'Library/Caches/ECNUExternalMets'
    assert temp.parent==parent and temp.name==ROOT.name and not temp.exists(),'exclusive temporary asset directory required'
    free=shutil.disk_usage(parent if parent.exists() else parent.parent).free
    assert free>30*1024**3,'minimum temporary capacity reservation 30GiB not met'
    temp.mkdir(parents=True,mode=0o700,exist_ok=False)
    temp.chmod(0o700)
    audit=ROOT/'operator'/('download_assets_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    audit.mkdir(exist_ok=False)
    write(temp/'OWNERSHIP.json',dict(META,batch=ROOT.name,temporary_root=str(temp),purpose='authorized_selected_assets_transfer_only'))
    proxy=os.environ.get('EXTERNAL_MET_LOCAL_HTTP_PROXY','')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':proxy,'https':proxy} if proxy else {}))
    records=[]
    def small(url,headers=None):
        with opener.open(urllib.request.Request(url,headers=headers or {}),timeout=40) as r:
            b=r.read(2*1024**2)
            assert len(b)<2*1024**2,'metadata exceeds limit'
            return b
    def download(url,relative,size,digest=None,headers=None):
        p=temp/relative;p.parent.mkdir(parents=True,exist_ok=True)
        start=time.monotonic();h=hashlib.sha256();count=0;reported=start
        print(json.dumps(dict(event='download_started',file=relative,expected_bytes=size,utc=utc())),flush=True)
        with opener.open(urllib.request.Request(url,headers=headers or {}),timeout=60) as r,p.open('xb') as f:
            while True:
                block=r.read(1024**2)
                if not block:break
                count+=len(block)
                assert count<=size,'asset size exceeds frozen identity'
                assert time.monotonic()-start<3600,'bounded asset download exceeded one hour'
                f.write(block);h.update(block)
                if time.monotonic()-reported>=30:
                    print(json.dumps(dict(event='download_progress',file=relative,bytes=count,expected_bytes=size,utc=utc())),flush=True)
                    reported=time.monotonic()
        assert count==size,'asset length mismatch'
        actual=h.hexdigest()
        assert digest is None or actual==digest,'asset SHA256 mismatch'
        p.chmod(0o400)
        records.append(dict(path=relative,bytes=count,sha256=actual))
        print(json.dumps(dict(event='download_verified',file=relative,bytes=count,sha256=actual,utc=utc())),flush=True)
    try:
        token=json.loads(small('https://auth.docker.io/token?service=registry.docker.io&scope=repository:'+IMAGE+':pull'))['token']
        headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json'}
        manifest_bytes=small('https://registry-1.docker.io/v2/'+IMAGE+'/manifests/sha256:'+DIGEST,headers)
        assert hashlib.sha256(manifest_bytes).hexdigest()==DIGEST,'image manifest drift'
        manifest=json.loads(manifest_bytes)
        assert sum(x['size'] for x in manifest['layers'])==4446698630,'image compressed layer budget drift'
        assert all(x['digest'].startswith('sha256:') for x in [manifest['config'],*manifest['layers']])
        config_bytes=small('https://registry-1.docker.io/v2/'+IMAGE+'/blobs/'+manifest['config']['digest'],headers)
        assert hashlib.sha256(config_bytes).hexdigest()==manifest['config']['digest'].split(':')[1]
        config=json.loads(config_bytes)
        assert config['architecture']=='amd64' and config['os']=='linux','image platform drift'
        for name,b in [('docker/manifest.json',manifest_bytes),('docker/config.json',config_bytes)]:
            p=temp/name;p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('xb') as f:f.write(b)
            p.chmod(0o400);records.append(dict(path=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
        for layer in manifest['layers']:
            digest=layer['digest'].split(':')[1]
            download('https://registry-1.docker.io/v2/'+IMAGE+'/blobs/'+layer['digest'],'docker/layers/'+digest+'.tar.gz',layer['size'],digest,headers)
        download(BIO_URL,'weights.zip',2472704349,'d308d3dd5dff4ec8fa899d64f056d8a18651b4d4e56f6618c66a2d8998e334f3')
        with zipfile.ZipFile(temp/'weights.zip') as z:
            entries=[]
            for x in z.infolist():
                p=PurePosixPath(x.filename)
                assert not p.is_absolute() and '..' not in p.parts and '\\' not in x.filename,'ZIP path escape'
                assert (x.external_attr>>16)&0o170000!=0o120000,'ZIP symlinks forbidden'
                entries.append(dict(path=x.filename,bytes=x.file_size,compressed_bytes=x.compress_size))
            assert len({x['path'] for x in entries})==len(entries),'ZIP duplicate paths'
            assert sum(x['bytes'] for x in entries)<40*1024**3,'ZIP expansion exceeds S0 preparation cap'
            write(audit/'BIO_ZIP_INVENTORY.json',dict(META,entries=entries,expanded_bytes=sum(x['bytes'] for x in entries)))
        download(BRAIN_URL,'BrainIAC.ckpt',362119240)
        payload=dict(META,status='local_assets_verified_not_server_transferred',temporary_root=str(temp),files=records,
            created_utc=utc(),free_bytes_before=free,temporary_budget_bytes=30*1024**3,
            image_digest='sha256:'+DIGEST,image_layer_count=len(manifest['layers']),
            docker_runtime_config={k:config.get('config',{}).get(k) for k in ['WorkingDir','Entrypoint','Cmd','User']},
            cleanup_status='not_done_requires_remote_transfer_SHA_acceptance',no_model_loaded=True,no_mri_downloaded=True)
        write(audit/'LOCAL_ASSET_MANIFEST.json',payload)
        write(temp/'TRANSFER_MANIFEST.json',payload)
        print(json.dumps(dict(event='all_local_assets_verified',audit=str(audit),files=len(records),utc=utc())),flush=True)
    except BaseException as error:
        write(audit/'DOWNLOAD_FAILED.json',dict(META,status='failed',error_type=type(error).__name__,
            message='Public asset operation failed; no tokens/redirect URLs printed',verified_files=records,
            temporary_root=str(temp),utc=utc(),no_automatic_retry=True,jobs_created=False,partial_files_preserved=True))
        raise RuntimeError('Selected asset download stopped; see private-free audit, no automatic retry') from None


if __name__=='__main__':main()
