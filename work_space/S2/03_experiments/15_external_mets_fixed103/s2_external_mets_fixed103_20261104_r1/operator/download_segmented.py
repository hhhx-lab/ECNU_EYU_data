"""Explicit controlled continuation of the interrupted asset transport, not jobs."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import time
import urllib.request
import zipfile
from download_assets import IMAGE,DIGEST,BIO_URL,BRAIN_URL,META,write,utc

ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    temp=Path(os.environ['EXTERNAL_MET_LOCAL_TRANSFER_ROOT'])
    assert json.loads((temp/'OWNERSHIP.json').read_text())['batch']==ROOT.name
    stage=temp/'segmented_transfer';stage.mkdir(mode=0o700,exist_ok=False)
    audit=ROOT/'operator'/('download_segmented_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    audit.mkdir(exist_ok=False)
    proxy=os.environ.get('EXTERNAL_MET_LOCAL_HTTP_PROXY','')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':proxy,'https':proxy} if proxy else {}))
    def small(url,headers=None):
        with opener.open(urllib.request.Request(url,headers=headers or {}),timeout=40) as r:return r.read(2*1024**2)
    manifest_bytes=(temp/'docker/manifest.json').read_bytes()
    config_bytes=(temp/'docker/config.json').read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest()==DIGEST
    manifest=json.loads(manifest_bytes)
    assert hashlib.sha256(config_bytes).hexdigest()==manifest['config']['digest'].split(':')[1]
    token=json.loads(small('https://auth.docker.io/token?service=registry.docker.io&scope=repository:'+IMAGE+':pull'))['token']
    auth={'Authorization':'Bearer '+token}
    files=[dict(path='docker/manifest.json',bytes=len(manifest_bytes),sha256=DIGEST),
        dict(path='docker/config.json',bytes=len(config_bytes),sha256=hashlib.sha256(config_bytes).hexdigest())]
    try:
        items=[('https://registry-1.docker.io/v2/'+IMAGE+'/blobs/'+x['digest'],'docker/layers/'+x['digest'].split(':')[1]+'.tar.gz',
                x['size'],x['digest'].split(':')[1],auth) for x in manifest['layers']]
        items += [(BIO_URL,'weights.zip',2472704349,'d308d3dd5dff4ec8fa899d64f056d8a18651b4d4e56f6618c66a2d8998e334f3',{}),
                  (BRAIN_URL,'BrainIAC.ckpt',362119240,None,{})]
        for url,relative,size,digest,headers in items:
            old=temp/relative
            if old.is_file() and old.stat().st_size==size and digest is not None:
                assert sha(old)==digest,'complete cached asset differs'
                files.append(dict(path=relative,bytes=size,sha256=digest,reused_verified_complete=True))
                print(json.dumps(dict(event='reuse_verified_asset',file=relative,bytes=size)),flush=True);continue
            p=stage/relative;p.parent.mkdir(parents=True,exist_ok=True)
            prefix=old.stat().st_size if old.is_file() else 0
            assert 0<=prefix<size
            start=time.monotonic();reported=start;done=prefix
            with p.open('xb') as out:
                if prefix:
                    with old.open('rb') as f:
                        for b in iter(lambda:f.read(1024**2),b''):out.write(b)
                out.truncate(size)
            record=dict(file=str(p.relative_to(temp)),expected_bytes=size,unverified_prefix_bytes=prefix,
                unverified_prefix_sha256=sha(old) if prefix else None)
            print(json.dumps(dict(event='segmented_asset_started',**record,utc=utc())),flush=True)
            fd=os.open(p,os.O_WRONLY)
            def part(a,b):
                h={**headers,'Range':f'bytes={a}-{b}'}
                with opener.open(urllib.request.Request(url,headers=h),timeout=60) as r:
                    assert r.status==206 and r.headers.get('Content-Range')==f'bytes {a}-{b}/{size}','range response identity mismatch'
                    pos=a
                    while True:
                        block=r.read(min(1024**2,b-pos+2))
                        if not block:break
                        assert pos+len(block)<=b+1,'range response overflow'
                        count=os.pwrite(fd,block,pos);assert count==len(block)
                        pos+=len(block)
                    assert pos==b+1,'range response truncated'
                return b-a+1
            try:
                chunk=16*1024**2
                with ThreadPoolExecutor(max_workers=6) as pool:
                    pending=[pool.submit(part,a,min(a+chunk,size)-1) for a in range(prefix,size,chunk)]
                    for future in as_completed(pending,timeout=3600):
                        done+=future.result()
                        if time.monotonic()-reported>=30:
                            print(json.dumps(dict(event='segmented_asset_progress',file=relative,bytes=done,expected_bytes=size,utc=utc())),flush=True)
                            reported=time.monotonic()
            finally:os.close(fd)
            actual=sha(p)
            assert done==size and p.stat().st_size==size and (digest is None or actual==digest),'complete asset digest mismatch'
            p.chmod(0o400)
            files.append(dict(path=str(p.relative_to(temp)),bytes=size,sha256=actual,transport_record=record))
            print(json.dumps(dict(event='segmented_asset_verified',file=relative,sha256=actual,bytes=size,utc=utc())),flush=True)
        bio=next(temp/f['path'] for f in files if f['path'].endswith('weights.zip'))
        with zipfile.ZipFile(bio) as z:
            entries=[]
            for x in z.infolist():
                path=PurePosixPath(x.filename)
                assert not path.is_absolute() and '..' not in path.parts and '\\' not in x.filename
                assert ((x.external_attr>>16)&0o170000)!=0o120000
                entries.append(dict(path=x.filename,bytes=x.file_size,compressed_bytes=x.compress_size))
            assert len(entries)==len({x['path'] for x in entries}) and sum(x['bytes'] for x in entries)<40*1024**3
            write(audit/'BIO_ZIP_INVENTORY.json',dict(META,entries=entries,expanded_bytes=sum(x['bytes'] for x in entries)))
        result=dict(META,status='local_assets_verified_not_server_transferred',temporary_root=str(temp),files=files,utc=utc(),
            image_digest='sha256:'+DIGEST,old_partial_preserved=True,no_job_retry=True,no_model_loaded=True,
            user_mac_transfer_amendment='TRANSFER_AMENDMENT_20261104.md',cleanup_status='requires_remote_complete_SHA_acceptance')
        write(audit/'LOCAL_ASSET_MANIFEST.json',result)
        write(stage/'TRANSFER_MANIFEST.json',result)
        print(json.dumps(dict(event='all_local_assets_verified',audit=str(audit),files=len(files))),flush=True)
    except BaseException as error:
        write(audit/'DOWNLOAD_FAILED.json',dict(META,status='failed',error_type=type(error).__name__,
            message=str(error) if isinstance(error,AssertionError) else 'segmented transfer failed; private URL suppressed',
            utc=utc(),verified_files=files,no_automatic_retry=True,old_partial_preserved=True,jobs_created=False))
        raise RuntimeError('Segmented asset transport stopped; preserve audit; no automatic retry') from None


if __name__=='__main__':main()
