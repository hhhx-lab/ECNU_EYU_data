"""Delete only the explicitly authorized retired remote SegMamba tree."""

import datetime
import hashlib
import json
import shlex

from archive_segmamba_retirement import ARCHIVE
from common import m, transport


def main():
    receipt = json.loads((ARCHIVE / 'LIGHT_ARCHIVE_RECEIPT.json').read_text())
    for row in receipt['preserved_files']:
        p = ARCHIVE / 'preserved' / row['path']
        assert p.stat().st_size == row['bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256']
    access = transport('delete_retired_segmamba')
    expected = {x['path']:{k:x[k] for k in ('directory','size_bytes','mtime_ns','nlink')} for x in receipt['manifest']}
    m.exclusive(access.audit / 'DELETION_AUTHORIZATION.json', {
        **m.META, 'local_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'/public/home/zqchen/SegMamba-main',
        'user_authorization':'删了吧，这个路线不做了，在本地留存一下轻量文档记录一下这个模型参数和结果就行，服务器上删了吧空间省出来',
        'light_archive':str(ARCHIVE), 'archive_receipt_sha256':hashlib.sha256((ARCHIVE / 'LIGHT_ARCHIVE_RECEIPT.json').read_bytes()).hexdigest(),
        'checkpoint_and_dataset_backup_requested':False, 'full_restore_possible':False,
        'protected':'S3/S4, project S5 local sources, G1/G2/S1/S2, .conda/envs and shared original data'})
    # Expected metadata is sent through stdin-free SSH as compressed JSON data.
    import base64
    import zlib
    payload = base64.b64encode(zlib.compress(json.dumps(expected,sort_keys=True).encode())).decode()
    code = r'''
import base64,datetime,json,os,pathlib,shutil,subprocess,time,zlib
home=pathlib.Path('/public/home/zqchen');root=home/'SegMamba-main'
expected=json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))
assert str(root)=='/public/home/zqchen/SegMamba-main' and root.parent==home
s=root.lstat();assert root.is_dir() and not root.is_symlink() and s.st_uid==os.getuid()
assert s.st_ino==INODE and s.st_dev==DEV
actual={}
for directory,dirs,names in os.walk(root,followlinks=False):
    for name in dirs+names:
        p=pathlib.Path(directory,name);st=p.lstat()
        assert not p.is_symlink() and st.st_dev==s.st_dev,str(p)
        actual[str(p.relative_to(root))]={'directory':p.is_dir(),'size_bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'nlink':st.st_nlink}
assert actual==expected,'target changed since light evidence archive; no deletion'
q=subprocess.run(['squeue','-h','-u','zqchen','-o','%i|%j|%T|%R'],capture_output=True,text=True,timeout=10)
assert q.returncode==0 and not q.stdout.strip(),'active queue entries; no deletion'
refs=[];errors=[];deadline=time.monotonic()+35;scanned=0
for base in (home/'projects/ECNU_EYU_data/work_space',home/'data'):
    if not base.exists():continue
    for directory,dirs,names in os.walk(base,followlinks=False,onerror=lambda e:errors.append(type(e).__name__)):
        if time.monotonic()>deadline:errors.append('project_dependency_scan_timeout');break
        ds=pathlib.Path(directory)
        for name in dirs+names:
            p=ds/name
            try:
                if p.is_symlink() and (str(p.resolve())==str(root) or str(p.resolve()).startswith(str(root)+'/')):
                    refs.append({'path':str(p),'kind':'symlink'})
            except OSError as e:errors.append(type(e).__name__)
        dirs[:]=[n for n in dirs if n not in ('.git','__pycache__','node_modules') and not (ds/n).is_symlink()]
        for name in names:
            p=ds/name
            if p.suffix not in ('.py','.sh','.slurm','.pth'):continue
            try:
                if p.is_symlink() or p.stat().st_size>500000:continue
                scanned+=1
                if str(root) in p.read_text(errors='replace'):refs.append({'path':str(p),'kind':'executable_source'})
            except OSError as e:errors.append(type(e).__name__)
for env in (home/'.conda/envs').iterdir():
    for site in env.glob('lib/python*/site-packages'):
        for p in list(site.glob('*.pth'))+[site/'mamba_ssm',site/'causal_conv1d']:
            if p.is_symlink() and str(p.resolve()).startswith(str(root)):refs.append({'path':str(p),'kind':'environment_symlink'})
            if p.suffix=='.pth' and p.is_file() and p.stat().st_size<=500000 and str(root) in p.read_text(errors='replace'):
                refs.append({'path':str(p),'kind':'environment_pth'})
assert not errors and not refs,json.dumps({'errors':errors,'references':refs})
processes=[]
for p in pathlib.Path('/proc').iterdir():
    if not p.name.isdigit() or p.name==str(os.getpid()):continue
    try:
        if p.stat().st_uid!=os.getuid():continue
        cmd=(p/'cmdline').read_bytes().decode(errors='replace')
        if 'target changed since light evidence archive' in cmd:continue
        cwd=os.readlink(p/'cwd');maps=(p/'maps').read_text(errors='replace')
        if cwd==str(root) or cwd.startswith(str(root)+'/') or str(root) in cmd or str(root) in maps:
            processes.append(p.name)
    except OSError:continue
assert not processes,'login process references target; no deletion'
before=os.statvfs(home).f_bavail*os.statvfs(home).f_frsize
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
# User retired this exact branch and requested only lightweight local evidence.
shutil.rmtree(root)
assert not os.path.lexists(root)
after=os.statvfs(home).f_bavail*os.statvfs(home).f_frsize
v={'status':'deleted_exact_authorized_tree','remote_utc_started':started,
   'remote_utc_completed':datetime.datetime.now(datetime.timezone.utc).isoformat(),
   'hostname':os.uname().nodename,'target':str(root),'target_absent':True,
   'available_before_bytes':before,'available_after_bytes':after,'observed_available_delta_bytes':after-before,
   'manifest_entries':len(actual),'executable_source_files_scanned':scanned,'external_scan_errors':errors,
   'external_references':refs,'login_process_references':processes,'queue_empty':True,
   'conda_segmamba_exists':(home/'.conda/envs/segmamba').is_dir(),
   'g1_exists':(home/'projects/ECNU_EYU_data/work_space/G1').is_dir(),
   'shared_data_exists':(home/'data').is_dir(),
   'boundary':'local light records only; no weights/dataset backup, cannot fully restore; process check login only; no other directory deleted'}
print(json.dumps(v,sort_keys=True))
'''.replace('PAYLOAD', repr(payload)).replace('INODE', str(receipt['target_inode'])).replace('DEV', str(receipt['target_dev']))
    compile(code, '<delete_retired_segmamba>', 'exec')
    try:
        access.start_vpn()
        result = access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -c ' + shlex.quote(code), timeout=150)
        data = json.loads(result.strip().splitlines()[-1])
        data.update(m.META)
        data['local_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit / 'DELETION_RECEIPT.json', data)
        m.exclusive(ARCHIVE / 'DELETION_RECEIPT.json', data)
        print(json.dumps(data,ensure_ascii=False))
    except Exception as error:
        m.exclusive(access.audit / 'FAILED.json', {**m.META,'status':'operation_failed_or_unconfirmed',
                                                  'exception_type':type(error).__name__,
                                                  'do_not_repeat_deletion':True})
        raise
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
