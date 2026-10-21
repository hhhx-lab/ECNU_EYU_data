"""One read-only probe for allocated storage paths, not a quota bypass."""
import datetime
import hashlib
import json
import shlex

from common import m, transport


CODE = r'''
import datetime,grp,json,os,pathlib,pwd,shutil,subprocess
def cmd(args):
    try:
        r=subprocess.run(args,capture_output=True,text=True,timeout=10)
        return dict(exit_code=r.returncode,stdout=r.stdout[-18000:],stderr=r.stderr[-1500:])
    except subprocess.TimeoutExpired:return dict(exit_code=124,status='timeout')
def inspect(path):
    p=pathlib.Path(path)
    v=dict(path=str(p),exists=p.exists(),symlink=p.is_symlink())
    if not p.exists():return v
    try:
        s=p.stat();f=os.statvfs(p)
        v.update(resolved=str(p.resolve()),uid=s.st_uid,gid=s.st_gid,
                 owner=pwd.getpwuid(s.st_uid).pw_name,group=grp.getgrgid(s.st_gid).gr_name,
                 mode=oct(s.st_mode & 0o7777),write_access=os.access(p,os.W_OK),
                 search_access=os.access(p,os.X_OK),device=s.st_dev,
                 total_bytes=f.f_blocks*f.f_frsize,available_bytes=f.f_bavail*f.f_frsize)
        if shutil.which('getfacl'):v['acl']=cmd(['getfacl','-p',str(p)])
    except (OSError,KeyError) as e:v['error']=str(e)
    return v
home='/public/home/zqchen'
roots=['/public','/hpc_stor','/scratch','/work','/storage','/mnt']
v=dict(remote_start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=os.uname().nodename,
       uid=os.getuid(),groups=os.getgroups(),commands={'id':cmd(['id']),
       'mounts':cmd(['findmnt','-J','-o','TARGET,SOURCE,FSTYPE,OPTIONS']),
       'quota':cmd(['quota','-s'])},roots=[],candidates=[])
candidates={home,home+'/projects/ECNU_EYU_data', '/public/zqchen',
            '/hpc_stor/zqchen','/hpc_stor/home/zqchen','/hpc_stor/users/zqchen'}
for root in roots:
    item=inspect(root);entries=[]
    if item['exists']:
        try:
            direct=sorted(pathlib.Path(root).iterdir())
            item['direct_entry_count']=len(direct)
            for p in direct[:60]:
                if not p.is_dir():continue
                s=p.stat();entries.append(dict(name=p.name,uid=s.st_uid,gid=s.st_gid,mode=oct(s.st_mode & 0o7777),symlink=p.is_symlink()))
                if s.st_uid==os.getuid():candidates.add(str(p))
            item['listing_truncated']=len(direct)>60
        except OSError as e:item['listing_error']=str(e)
    item['direct_directories']=entries;v['roots'].append(item)
for path in sorted(candidates):v['candidates'].append(inspect(path))
v['remote_end_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
v['method']='one read-only mount/ACL/statvfs snapshot and shallow root listing; no write test or recursive scan'
v['write_permission_is_storage_authorization']=False
print(json.dumps(v,sort_keys=True))
'''


def main():
    compile(CODE, 'readonly_storage_paths', 'exec')
    access = transport('storage_paths_readonly')
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        access.start_vpn()
        output = access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -c ' + shlex.quote(CODE), timeout=110)
        value = json.loads(output.strip().splitlines()[-1])
        value.update(m.META)
        value['local_start_utc'] = start
        value['local_received_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit / 'STORAGE_PATHS.json', value)
    finally:
        access.close_vpn()
    path = access.audit / 'STORAGE_PATHS.json'
    m.exclusive(access.audit / 'LOCAL_OPERATION_RECEIPT.json', dict(
        **m.META, status='pass_readonly_snapshot_received',
        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        local_end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        operation_vpn_closed=True, remote_writes=False, write_test_performed=False,
        remote_submissions=False, large_data_downloaded=False))
    print(json.dumps(dict(audit=str(access.audit),candidates=value['candidates']),ensure_ascii=False))


if __name__ == '__main__':
    main()
