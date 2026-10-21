"""Exact storage probe with a gated, exclusive 1 KiB owner-directory test."""
import datetime
import hashlib
import json
import shlex

from common import m, transport


CODE = r'''
import datetime,hashlib,json,os,pathlib,shutil,stat,subprocess,uuid
def cmd(args):
    try:
        r=subprocess.run(args,capture_output=True,text=True,timeout=10)
        return dict(exit_code=r.returncode,stdout=r.stdout[-12000:],stderr=r.stderr[-2000:])
    except subprocess.TimeoutExpired:return dict(exit_code=124,status='timeout')
def inspect(path):
    p=pathlib.Path(path);v=dict(path=path,exists=p.exists(),symlink=p.is_symlink())
    if not p.exists():return v
    try:
        s=p.stat();f=os.statvfs(p)
        v.update(resolved=str(p.resolve()),uid=s.st_uid,gid=s.st_gid,mode=oct(stat.S_IMODE(s.st_mode)),
                 is_directory=p.is_dir(),write_access=os.access(p,os.W_OK),search_access=os.access(p,os.X_OK),
                 device=s.st_dev,total_bytes=f.f_blocks*f.f_frsize,available_bytes=f.f_bavail*f.f_frsize)
        if shutil.which('getfacl'):v['acl']=cmd(['getfacl','-p',path])
        if shutil.which('findmnt'):v['mount']=cmd(['findmnt','-n','-T',path,'-o','TARGET,SOURCE,FSTYPE,OPTIONS'])
    except OSError as e:v['error']=str(e)
    return v
paths=['/public/home/zqchen','/hpc_stor/data','/hpc_stor/data/zqchen','/public/share','/public/share/zqchen']
v=dict(remote_start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=os.uname().nodename,
       paths=[inspect(p) for p in paths],policy_files=[],test_results=[],quota=cmd(['quota','-s']))
for parent in ('/hpc_stor/data','/public/share'):
    p=pathlib.Path(parent)
    if not p.is_dir():continue
    try:
        names=[x for x in p.iterdir() if x.name.lower().startswith(('readme','notice','policy','usage','rules'))]
        for f in sorted(names)[:8]:
            if f.is_symlink() or not f.is_file():continue
            s=f.stat()
            if s.st_size>32768 or s.st_uid!=0 or s.st_mode&0o022:continue
            v['policy_files'].append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                                          content=f.read_text(errors='replace')))
    except OSError as e:v['policy_files'].append(dict(parent=parent,error=str(e)))
for path in ('/hpc_stor/data/zqchen','/public/share/zqchen'):
    item=next(x for x in v['paths'] if x['path']==path)
    allowed=(item.get('exists') and item.get('is_directory') and not item.get('symlink')
             and item.get('resolved')==path and item.get('uid')==os.getuid()
             and item.get('write_access') and item.get('search_access')
             and not int(item.get('mode','0'),8)&0o022)
    if not allowed:
        v['test_results'].append(dict(directory=path,status='not_tested_no_confirmed_private_owner_directory'))
        continue
    dest=pathlib.Path(path)/('.ecnu_storage_probe_'+uuid.uuid4().hex+'.bin')
    data=(b'ECNU_STORAGE_PROBE_NO_RESEARCH_DATA\n'*32)[:1024]
    assert len(data)==1024
    result=dict(directory=path,path=str(dest),bytes=len(data),purpose='user_authorized_small_storage_test',
                retained=True,capacity_or_long_term_policy_validated=False)
    try:
        fd=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as h:h.write(data);h.flush();os.fsync(h.fileno())
        actual=dest.read_bytes()
        result.update(expected_sha256=hashlib.sha256(data).hexdigest(),actual_sha256=hashlib.sha256(actual).hexdigest(),
                      actual_bytes=len(actual),mode=oct(stat.S_IMODE(dest.stat().st_mode)),
                      status='pass_1KiB_write_read_SHA' if actual==data else 'failed_byte_mismatch')
    except OSError as e:result.update(status='write_or_read_failed',error=str(e))
    v['test_results'].append(result)
v['remote_end_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
v['no_old_file_deleted_or_overwritten']=True
v['no_large_data_moved']=True
v['shared_root_write_test']=False
print(json.dumps(v,sort_keys=True))
'''


def main():
    compile(CODE, 'exact_storage_probe', 'exec')
    access = transport('exact_storage_probe')
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        access.start_vpn()
        output = access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -c ' + shlex.quote(CODE), timeout=115)
        value = json.loads(output.strip().splitlines()[-1])
        value.update(m.META)
        value['local_start_utc'] = start
        value['local_received_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit / 'EXACT_STORAGE_PROBE.json', value)
    finally:
        access.close_vpn()
    path = access.audit / 'EXACT_STORAGE_PROBE.json'
    performed = any(x['status'] != 'not_tested_no_confirmed_private_owner_directory' for x in value['test_results'])
    m.exclusive(access.audit / 'LOCAL_OPERATION_RECEIPT.json', dict(
        **m.META, status='snapshot_received', file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        local_end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        operation_vpn_closed=True, small_write_test_performed=performed,
        remote_submissions=False, large_data_downloaded=False))
    print(json.dumps(dict(audit=str(access.audit),paths=value['paths'],test_results=value['test_results'],
                         policy_files=value['policy_files']),ensure_ascii=False))


if __name__ == '__main__':
    main()
