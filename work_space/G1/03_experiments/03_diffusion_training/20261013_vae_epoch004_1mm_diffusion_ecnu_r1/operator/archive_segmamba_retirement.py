"""Preserve bounded lightweight evidence for the explicitly retired S5 branch."""

import base64
import datetime
import hashlib
import json
import shlex
from pathlib import Path

from common import PROJECT, m, transport

ARCHIVE = PROJECT / 'work_space/S5/03_experiments/20261013_remote_segmamba_retirement'


def main():
    access = transport('archive_segmamba_retirement')
    code = r'''
import base64,datetime,hashlib,json,os,pathlib,subprocess,time
root=pathlib.Path('/public/home/zqchen/SegMamba-main')
assert root.is_dir() and not root.is_symlink() and root.stat().st_uid==os.getuid()
files=[];manifest=[];total=0;log_excerpts=[]
for directory,dirs,names in os.walk(root,followlinks=False):
    for name in dirs+names:
        p=pathlib.Path(directory,name);s=p.lstat()
        assert s.st_dev==root.stat().st_dev and not p.is_symlink(),str(p)
        manifest.append({'path':str(p.relative_to(root)),'directory':p.is_dir(),'size_bytes':s.st_size,
                         'allocated_bytes':s.st_blocks*512,'mtime_ns':s.st_mtime_ns,'nlink':s.st_nlink})
    for name in names:
        p=pathlib.Path(directory,name);rel=p.relative_to(root);s=p.stat()
        public_source=(rel.parts[0] in ('configs','model_segmamba','light_training','slurm','evaluation') or len(rel.parts)==1)
        text=p.suffix.lower() in ('.py','.yaml','.yml','.sh','.slurm','.md','.rst','.txt','.csv','.json') or name in ('LICENSE','LICENSE.txt')
        result=rel.parts[0]=='results' and p.suffix.lower() in ('.csv','.json','.txt','.png')
        data_meta=rel.parts[0]=='data' and len(rel.parts)<=3 and p.suffix in ('.csv','.json','.txt')
        if (public_source and text or result or data_meta) and s.st_size<=2*1024**2:
            raw=p.read_bytes();total+=len(raw);assert total<=30*1024**2
            files.append({'path':str(rel),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                          'base64':base64.b64encode(raw).decode()})
        if rel.parts[0]=='logs' and p.suffix in ('.out','.err','.log'):
            with p.open('rb') as f:
                head=f.read(4096);f.seek(max(0,s.st_size-12000));tail=f.read(12000)
            log_excerpts.append({'path':str(rel),'original_size_bytes':s.st_size,
                                 'head':head.decode(errors='replace'),'tail':tail.decode(errors='replace')})
checkpoints=[x for x in manifest if x['path'].endswith('.pt')]
q=subprocess.run(['squeue','-h','-u','zqchen','-o','%i|%j|%T|%R'],capture_output=True,text=True,timeout=10)
v={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,
   'target':str(root),'target_inode':root.stat().st_ino,'target_dev':root.stat().st_dev,
   'files':files,'preserved_bytes':total,'manifest':manifest,'checkpoint_metadata_only':checkpoints,
   'log_excerpts':log_excerpts,'queue':{'exit_code':q.returncode,'stdout':q.stdout,'stderr':q.stderr},
   'available_before_bytes':os.statvfs(root.parent).f_bavail*os.statvfs(root.parent).f_frsize,
   'boundary':'no dataset/checkpoint archive; public source/config/metrics/plots and log excerpts only; historical scalar filenames are not official MET proof'}
print(json.dumps(v,sort_keys=True))
'''
    compile(code, '<archive_segmamba_retirement>', 'exec')
    try:
        access.start_vpn()
        data = json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -c ' + shlex.quote(code), timeout=90).strip().splitlines()[-1])
        ARCHIVE.mkdir(parents=True, exist_ok=False)
        receipt_files = []
        for row in data.pop('files'):
            relative = Path(row['path'])
            assert not relative.is_absolute() and '..' not in relative.parts
            raw = base64.b64decode(row.pop('base64'), validate=True)
            assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
            dest = ARCHIVE / 'preserved' / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as f:
                f.write(raw)
            receipt_files.append(row)
        data.update(m.META)
        data['local_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        data['preserved_files'] = receipt_files
        data['status'] = 'light_evidence_verified_no_deletion_yet'
        data['full_restore_possible'] = False
        m.exclusive(ARCHIVE / 'LIGHT_ARCHIVE_RECEIPT.json', data)
        m.exclusive(access.audit / 'ARCHIVE_LOCATION.json', {'archive':str(ARCHIVE),'files':len(receipt_files),'bytes':data['preserved_bytes']})
        print(json.dumps({'archive':str(ARCHIVE),'files':len(receipt_files),'bytes':data['preserved_bytes'],
                          'checkpoints_metadata':len(data['checkpoint_metadata_only']),
                          'queue':data['queue']},ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
