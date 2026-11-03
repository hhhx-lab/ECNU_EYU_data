"""Exclusive server asset transfer, complete SHA acceptance, then owned Mac cleanup."""
import json
import os
from pathlib import Path
import shlex
from session import session,access_module
from download_segmented import sha
from download_assets import DIGEST

ROOT=Path(__file__).resolve().parents[1]


def verification_code(partial,target,item):
    return f'''from pathlib import Path
import hashlib,json,os
from datetime import datetime,timezone
p=Path({partial!r});h=hashlib.sha256()
with p.open('rb') as f:
    for b in iter(lambda:f.read(1048576),b''):h.update(b)
assert p.stat().st_size=={item['bytes']!r} and h.hexdigest()=={item['sha256']!r}
os.link(p,{target!r})
p.unlink()
p=Path({target!r});p.chmod(0o440)
print(json.dumps(dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest(),remote_utc=datetime.now(timezone.utc).isoformat())))
'''


def receipt_code(path,payload):
    return f'''from pathlib import Path
import json
p=Path({path!r});payload={payload!r}
with p.open('x') as f:
    json.dump(payload,f,indent=2)
    f.write('\\n')
p.chmod(0o440)
print(json.dumps(dict(status='server_transfer_marker_written')))
'''


def logical_assets(manifest,temp):
    files=manifest['files'];assert len(files)==15
    registry=json.loads((temp/'docker/manifest.json').read_text())
    assert sha(temp/'docker/manifest.json')==DIGEST
    expected={'docker/manifest.json':(2629,DIGEST),
              'docker/config.json':(registry['config']['size'],registry['config']['digest'].split(':')[1]),
              'weights.zip':(2472704349,'d308d3dd5dff4ec8fa899d64f056d8a18651b4d4e56f6618c66a2d8998e334f3'),
              'BrainIAC.ckpt':(362119240,None)}
    expected.update({'docker/layers/'+x['digest'].split(':')[1]+'.tar.gz':
                     (x['size'],x['digest'].split(':')[1]) for x in registry['layers']})
    out=[]
    for item in files:
        logical=item.get('logical_path',item['path'].removeprefix('segmented_transfer/'))
        assert logical in expected,'unexpected logical asset identity'
        p=temp/item['path']
        assert not Path(item['path']).is_absolute() and '..' not in Path(item['path']).parts
        assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(temp.resolve())
        size,digest=expected[logical]
        assert item['bytes']==size and (digest is None or item['sha256']==digest)
        assert p.stat().st_size==size and sha(p)==item['sha256']
        out.append({**item,'logical_path':logical})
    assert {x['logical_path'] for x in out}==set(expected) and len(set(x['path'] for x in out))==15
    return out


def main():
    candidates=sorted((ROOT/'operator').glob('download_bounded_resume_*/LOCAL_ASSET_MANIFEST.json'))
    assert len(candidates)==1,'one complete explicit transport attempt required'
    source=candidates[0];manifest=json.loads(source.read_text())
    temp=Path(os.environ['EXTERNAL_MET_LOCAL_TRANSFER_ROOT'])
    assert manifest['temporary_root']==str(temp) and manifest['status']=='local_assets_verified_not_server_transferred'
    assert temp.parent==Path.home()/'Library/Caches/ECNUExternalMets' and temp.name==ROOT.name
    assert json.loads((temp/'OWNERSHIP.json').read_text())['batch']==ROOT.name
    remote=os.environ['EXTERNAL_MET_REMOTE_ROOT']
    files=logical_assets(manifest,temp)
    cleanup=[]
    for p in sorted(temp.rglob('*')):
        assert not p.is_symlink(),'temporary symlink refuses cleanup'
        if p.is_file():cleanup.append(dict(path=str(p.relative_to(temp)),bytes=p.stat().st_size,sha256=sha(p)))
    with session('transfer_assets') as access:
        access_module.exclusive(access.audit/'MAC_CLEANUP_CANDIDATES.json',dict(access_module.META,temporary_root=str(temp),files=cleanup))
        py=shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])
        def run(code):return json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+py+' -B -c '+shlex.quote(code),timeout=240).strip().splitlines()[-1])
        prepare="from pathlib import Path; import json,shutil; p=Path("+repr(remote)+"); assert p.is_dir(); free=shutil.disk_usage(p).free; assert free>100*1024**3; q=p/'assets/incoming'; q.mkdir(mode=0o700,exist_ok=False); print(json.dumps(dict(free_bytes=free)))"
        access_module.exclusive(access.audit/'TRANSFER_CAPACITY_GATE.json',run(prepare))
        accepted=[]
        for item in files:
            logical=item['logical_path']
            target=remote+'/assets/incoming/'+logical
            assert not Path(logical).is_absolute() and '..' not in Path(logical).parts
            code="from pathlib import Path; import json; p=Path("+repr(target)+"); assert not p.exists() and not p.with_suffix(p.suffix+'.part').exists(); p.parent.mkdir(parents=True,exist_ok=True); print(json.dumps(dict(status='exclusive_destination_ready')))"
            run(code)
            partial=target+'.part'
            argv=['scp','-P',access.env['HPC_LOGIN_PORT'],'-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',
                  str(temp/item['path']),access.env['HPC_SSH_USER']+'@'+access.env['HPC_LOGIN_HOST']+':'+partial]
            access.call(argv,timeout=1800)
            observation=run(verification_code(partial,target,item));accepted.append(observation)
            access_module.exclusive(access.audit/('FILE_'+str(len(accepted)).zfill(2)+'_ACCEPTED.json'),observation)
            print(json.dumps(dict(event='remote_asset_SHA_verified',file=logical,bytes=item['bytes'],remote_utc=observation['remote_utc'])),flush=True)
        confirm='''from pathlib import Path
import hashlib,json,shutil
from datetime import datetime,timezone
files='''+repr(accepted)+'''
for item in files:
    p=Path(item['path']);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    assert p.stat().st_size==item['bytes'] and h.hexdigest()==item['sha256']
print(json.dumps(dict(status='complete_remote_set_reverified',files=len(files),remote_utc=datetime.now(timezone.utc).isoformat(),free_bytes=shutil.disk_usage(Path(files[0]['path']).parent).free)))
'''
        final_observation=run(confirm)
        result=dict(access_module.META,status='all_server_asset_bytes_SHA_verified',files=accepted,
            local_manifest_sha256=sha(source),temporary_root=str(temp),jobs_created=False,
            remote_final_observation=final_observation,
            local_received_utc=access_module.datetime.now(access_module.timezone.utc).isoformat())
        run(receipt_code(remote+'/operator/ASSET_TRANSFER_COMPLETE.json',result))
        access_module.exclusive(access.audit/'TRANSFER_ACCEPTANCE.json',result)
        for item in cleanup:
            p=temp/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],'temporary asset changed before cleanup'
        assert {str(p.relative_to(temp)) for p in temp.rglob('*') if p.is_file()}=={x['path'] for x in cleanup},'unexpected new temporary file; do not delete'
        for item in cleanup:(temp/item['path']).unlink()
        for p in sorted((p for p in temp.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):p.rmdir()
        temp.rmdir()
        access_module.exclusive(access.audit/'MAC_CLEANUP_COMPLETE.json',dict(access_module.META,status='only_owned_temporary_files_removed',
            files=cleanup,temporary_root=str(temp),temporary_root_absent=not temp.exists(),server_assets_preserved=True))
        print(json.dumps(dict(event='asset_transfer_and_owned_Mac_cleanup_complete',audit=str(access.audit))),flush=True)


if __name__=='__main__':main()
