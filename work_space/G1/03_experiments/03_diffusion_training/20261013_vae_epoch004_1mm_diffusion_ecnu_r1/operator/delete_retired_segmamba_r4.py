"""Retire the exact tree with complete target and targeted dependency checks."""

import datetime
import hashlib
import json
import shlex

from archive_segmamba_retirement import ARCHIVE
from common import m, transport


def main():
    rp = ARCHIVE / 'LIGHT_ARCHIVE_RECEIPT.json'
    receipt = json.loads(rp.read_text())
    for row in receipt['preserved_files']:
        p = ARCHIVE / 'preserved' / row['path']
        assert p.stat().st_size == row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256']
    expected = {x['path']:{k:x[k] for k in ('directory','size_bytes','mtime_ns','nlink')} for x in receipt['manifest']}
    digest = hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
    access = transport('delete_retired_segmamba_r4')
    m.exclusive(access.audit / 'DELETION_AUTHORIZATION.json', {
        **m.META,'local_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'/public/home/zqchen/SegMamba-main','user_authorized_route_retirement':True,
        'light_archive_receipt_sha256':hashlib.sha256(rp.read_bytes()).hexdigest(),
        'target_manifest_digest':digest,'full_restore_possible':False,
        'dependency_scope':'known project entrypoints and installed modules, queue, login processes; full-project scans timed out before deletion',
        'preserved_previous_failures':True})
    code = r'''
import datetime,hashlib,json,os,pathlib,shutil,subprocess
home=pathlib.Path('/public/home/zqchen');root=home/'SegMamba-main'
s=root.lstat()
assert str(root)=='/public/home/zqchen/SegMamba-main' and root.parent==home
assert root.is_dir() and not root.is_symlink() and s.st_uid==os.getuid() and s.st_ino==INODE and s.st_dev==DEV
actual={}
for directory,dirs,names in os.walk(root,followlinks=False):
    for name in dirs+names:
        p=pathlib.Path(directory,name);st=p.lstat()
        assert not p.is_symlink() and st.st_dev==s.st_dev
        actual[str(p.relative_to(root))]={'directory':p.is_dir(),'size_bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'nlink':st.st_nlink}
assert hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest()==DIGEST,'manifest drift; no deletion'
q=subprocess.run(['squeue','-h','-u','zqchen'],capture_output=True,text=True,timeout=10)
assert q.returncode==0 and not q.stdout.strip(),'queue not empty; no deletion'
workspace=home/'projects/ECNU_EYU_data/work_space'
refs=[];checked=[]
bases=[home/'data',workspace]
for module in ('G1','G2','S1','S2','S3','S4','S5'):
    base=workspace/module
    bases.extend([base,base/'code',base/'data',base/'results'])
old=workspace/'G1/code/brats2025-latent-ensemble-synthesis-main-v3'
bases.extend([old/'data',old/'data/input',old/'data/input_inference',old/'training'])
for base in bases:
    checked.append(str(base))
    if base.is_symlink():
        resolved=str(base.resolve())
        if resolved==str(root) or resolved.startswith(str(root)+'/'):refs.append(str(base))
    elif base.is_dir():
        for p in base.iterdir():
            if p.is_symlink():
                resolved=str(p.resolve())
                if resolved==str(root) or resolved.startswith(str(root)+'/'):refs.append(str(p))
for envname in ('segmamba','diffusion','brats_g1_v2','brats_swin'):
    env=home/'.conda/envs'/envname
    for site in env.glob('lib/python*/site-packages'):
        for p in list(site.glob('*.pth'))+[site/'mamba_ssm',site/'causal_conv1d']:
            if p.is_symlink() and str(p.resolve()).startswith(str(root)):
                refs.append(str(p))
            if p.suffix=='.pth' and p.is_file() and p.stat().st_size<=500000 and str(root) in p.read_text(errors='replace'):
                refs.append(str(p))
assert not refs,json.dumps(refs)
processes=[]
for p in pathlib.Path('/proc').iterdir():
    if not p.name.isdigit() or p.name==str(os.getpid()):continue
    try:
        if p.stat().st_uid!=os.getuid():continue
        cmd=(p/'cmdline').read_bytes().decode(errors='replace')
        if 'manifest drift; no deletion' in cmd:continue
        cwd=os.readlink(p/'cwd');maps=(p/'maps').read_text(errors='replace')
        if cwd==str(root) or cwd.startswith(str(root)+'/') or str(root) in cmd or str(root) in maps:processes.append(p.name)
    except OSError:continue
assert not processes,'login process dependency; no deletion'
before=os.statvfs(home).f_bavail*os.statvfs(home).f_frsize
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
shutil.rmtree(root)
assert not os.path.lexists(root)
after=os.statvfs(home).f_bavail*os.statvfs(home).f_frsize
print(json.dumps({'status':'deleted_exact_authorized_tree','target':str(root),'target_absent':True,
   'hostname':os.uname().nodename,'remote_utc_started':started,
   'remote_utc_completed':datetime.datetime.now(datetime.timezone.utc).isoformat(),
   'available_before_bytes':before,'available_after_bytes':after,'observed_available_delta_bytes':after-before,
   'manifest_entries':len(actual),'dependency_entrypoints_checked':checked,'external_references':refs,
   'login_process_references':processes,'queue_empty':True,
   'conda_segmamba_exists':(home/'.conda/envs/segmamba').is_dir(),
   'g1_exists':(workspace/'G1').is_dir(),'shared_data_exists':(home/'data').is_dir(),
   'boundary':'explicitly retired exact tree; light records not full backup; known entrypoint and selected env checks only, not full account historical references; login process check only; no other directory removed'},sort_keys=True))
'''.replace('DIGEST', repr(digest)).replace('INODE', str(receipt['target_inode'])).replace('DEV', str(receipt['target_dev']))
    compile(code, '<delete_retired_segmamba_r4>', 'exec')
    try:
        access.start_vpn()
        data = json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -c ' + shlex.quote(code), timeout=110).strip().splitlines()[-1])
        data.update(m.META)
        data['local_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit / 'DELETION_RECEIPT.json', data)
        m.exclusive(ARCHIVE / 'DELETION_RECEIPT.json', data)
        print(json.dumps(data,ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
