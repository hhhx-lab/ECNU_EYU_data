"""Bounded read-only identity and storage classification; no remote writes."""

import datetime
import json
import shlex

from common import m, transport


def main():
    access = transport('readonly_two_project_inventory')
    code = r'''
import collections,datetime,json,os,pathlib,re,subprocess,time
home=pathlib.Path('/public/home/zqchen')
g1=home/'projects/ECNU_EYU_data/work_space/G1'
old=g1/'code/brats2025-latent-ensemble-synthesis-main-v3'
other=home/'zdn_deepcpg'
start=time.monotonic()
def metadata(p):
    s=p.lstat()
    return {'path':str(p),'size_bytes':s.st_size,'allocated_bytes':s.st_blocks*512,
            'mtime_utc':datetime.datetime.fromtimestamp(s.st_mtime,datetime.timezone.utc).isoformat(),
            'uid':s.st_uid,'nlink':s.st_nlink,'symlink':p.is_symlink(),
            'target':os.readlink(p) if p.is_symlink() else None,'directory':p.is_dir()}
def children(p):
    try:return [metadata(x) for x in sorted(p.iterdir())][:200]
    except OSError as e:return {'error':type(e).__name__}
def du(p,depth,timeout):
    try:
        r=subprocess.run(['du','-x','-B1','--max-depth='+str(depth),str(p)],capture_output=True,text=True,timeout=timeout)
        out=r.stdout;status=r.returncode
    except subprocess.TimeoutExpired as e:
        out=e.stdout or b'';out=out.decode(errors='replace') if isinstance(out,bytes) else out;status=124
        r=None
    rows=[]
    for line in out.splitlines():
        fields=line.split('\t',1)
        if len(fields)==2 and fields[0].isdigit():rows.append({'allocated_bytes':int(fields[0]),'path':fields[1]})
    return {'path':str(p),'exit_code':status,'complete':status==0,'records':rows,
            'stderr':r.stderr[-1500:] if r else 'bounded_timeout'}
def safe_text(p,limit=6000):
    try:
        if p.is_symlink() or p.stat().st_size>100000:return None
        text=p.read_text(errors='replace')[:limit]
        lines=[line for line in text.splitlines() if not re.search(r'password|token|secret|api.?key|private.?key|authorization|credential',line,re.I)]
        return {'path':str(p),'text':'\n'.join(lines),'redaction':'credential-like lines omitted; truncated'}
    except OSError:return None
docs=[]
for root in (other,old):
    candidates=list(root.glob('*README*'))+list(root.glob('*readme*'))+list(root.glob('*/README*'))
    for p in sorted(set(candidates))[:10]:
        if p.is_file():
            d=safe_text(p)
            if d:docs.append(d)
def filesummary(root,seconds=14):
    end=time.monotonic()+seconds;groups=collections.defaultdict(lambda:{'files':0,'logical_bytes':0,'allocated_bytes':0})
    large=[];links=[];count=0;partial=False;errors=[];cache_stems=collections.defaultdict(set)
    def err(e):errors.append(str(e)[:250])
    for directory,dirs,files in os.walk(root,followlinks=False,onerror=err):
        dirs[:]=[d for d in dirs if d not in ('.git',) and not pathlib.Path(directory,d).is_symlink()]
        for name in files:
            if time.monotonic()>end or count>=120000:partial=True;break
            p=pathlib.Path(directory,name)
            try:
                info=metadata(p)
                if info['symlink']:
                    if len(links)<80:links.append(info)
                    continue
                ext='.nii.gz' if name.endswith('.nii.gz') else p.suffix.lower() or '(no_suffix)'
                g=groups[ext];g['files']+=1;g['logical_bytes']+=info['size_bytes'];g['allocated_bytes']+=info['allocated_bytes'];count+=1
                if info['size_bytes']>=100*1024**2:large.append(info)
                if name.endswith(('.npy','.npz')):
                    cache_stems[str(p.with_suffix(''))].add(ext)
            except OSError as e:err(e)
        if partial:break
    return {'path':str(root),'complete':not partial and not errors,'files_observed':count,'by_extension':dict(groups),
            'largest_files':sorted(large,key=lambda x:x['allocated_bytes'],reverse=True)[:50],
            'symlinks':links,'errors':errors[:10],
            'npy_npz_matching_stems':sum(v=={'.npy','.npz'} for v in cache_stems.values())}
v={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,
   'root_children':{str(p):children(p) for p in (other,g1,old,old/'data',old/'training')},
   'public_readmes':docs,
   'directory_sizes':[du(other,2,25),du(old/'data',2,20),du(old/'training',2,20),
                      du(g1/'results/experimental_unvalidated/g1_vae_threeway_fixed103_validation_20260914_attempt03',2,8),
                      du(g1/'results/experimental_unvalidated/g1_vae_extended_finetune_20260901_r3_attempt03',2,8)],
   'file_summaries':[filesummary(other,10),filesummary(old/'data',12),filesummary(old/'training',12)],
   'queue':None,'boundary':'read-only metadata and selected public README; no environment/config secrets, tensor loading, dataset download, deletion or job operation'}
r=subprocess.run(['squeue','-h','-u','zqchen','-o','%i|%j|%T|%R'],capture_output=True,text=True,timeout=10)
v['queue']={'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
v['available_bytes']=os.statvfs(home).f_bavail*os.statvfs(home).f_frsize
v['elapsed_seconds']=round(time.monotonic()-start,3)
print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(v,sort_keys=True),flags=re.I))
'''
    compile(code, '<readonly_two_project_inventory>', 'exec')
    try:
        access.start_vpn()
        result = access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -c ' + shlex.quote(code), timeout=160)
        data = json.loads(result.strip().splitlines()[-1])
        data.update(m.META)
        data['local_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit / 'TWO_PROJECT_INVENTORY.json', data)
        print(json.dumps({'audit':str(access.audit),'local_utc':data['local_utc'],
                          'remote_utc':data['remote_utc'],'elapsed_seconds':data['elapsed_seconds'],
                          'public_readmes':data['public_readmes'],'directory_sizes':data['directory_sizes'],
                          'root_children':data['root_children']},ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
