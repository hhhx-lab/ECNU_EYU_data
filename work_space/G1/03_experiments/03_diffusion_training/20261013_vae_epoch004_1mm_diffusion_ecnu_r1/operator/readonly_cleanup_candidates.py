"""Bounded cleanup candidate inventory. No deletion, clean commands or remote writes."""

import datetime
import json
import shlex

from common import m, transport


def main():
    access=transport('readonly_cleanup_candidates')
    code=r'''
import collections,datetime,json,os,pathlib,re,subprocess,time
home=pathlib.Path('/public/home/zqchen')
seg=home/'SegMamba-main'
def command(args,timeout=15):
    start=time.monotonic()
    try:
        r=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
        return {'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr[-2000:],'elapsed_seconds':round(time.monotonic()-start,3)}
    except subprocess.TimeoutExpired as e:
        decode=lambda x:x.decode(errors='replace') if isinstance(x,bytes) else (x or '')
        return {'exit_code':124,'stdout':decode(e.stdout),'stderr':decode(e.stderr)[-2000:],'timed_out':True}
targets=[('.cache/pip',12),('.cache/puppeteer',8),('.cache/torch_extensions',8),('.triton',8),('.nv',8),('.npm',8),
         ('.conda/pkgs',25),('.conda/envs',15),('SegMamba-main/mamba/build',8),('SegMamba-main/causal-conv1d/build',8),
         ('.vscode-server',12)]
sizes=[]
for name,timeout in targets:
    p=home/name
    if not p.is_dir():sizes.append({'path':str(p),'exists':False});continue
    r=command(['du','-x','-B1','--max-depth=1',str(p)],timeout)
    rows=[]
    for line in r['stdout'].splitlines():
        f=line.split('\t',1)
        if len(f)==2 and f[0].isdigit():rows.append({'path':f[1],'allocated_bytes':int(f[0])})
    sizes.append({'path':str(p),'exists':True,'complete':r['exit_code']==0,'exit_code':r['exit_code'],
                  'records':sorted(rows,key=lambda x:x['allocated_bytes'],reverse=True),'stderr':r['stderr']})
archives=[]
pkgs=home/'.conda/pkgs'
if pkgs.is_dir():
    for p in pkgs.iterdir():
        if p.is_file() and not p.is_symlink() and (p.name.endswith('.conda') or p.name.endswith('.tar.bz2')):
            s=p.stat();archives.append({'path':str(p),'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'hardlinks':s.st_nlink})
data=[]
for name in ('data/fullres/train','data/validation_preprocessed'):
    p=seg/name
    if not p.is_dir():continue
    count=collections.Counter();allocated=collections.Counter();logical=collections.Counter();stem=collections.defaultdict(set)
    for f in p.iterdir():
        if not f.is_file() or f.is_symlink():continue
        s=f.stat();kind='seg.npy' if f.name.endswith('_seg.npy') else f.suffix
        count[kind]+=1;allocated[kind]+=s.st_blocks*512;logical[kind]+=s.st_size
        if f.suffix in ('.npz','.pkl'):stem[f.stem].add(f.suffix)
        elif f.name.endswith('_seg.npy'):stem[f.name[:-8]].add('seg.npy')
        elif f.suffix=='.npy':stem[f.stem].add('.npy')
    paired=[k for k,v in stem.items() if {'.npz','.npy','seg.npy','.pkl'}.issubset(v)]
    data.append({'path':str(p),'suffix_counts':dict(count),'allocated_by_suffix':dict(allocated),'logical_by_suffix':dict(logical),
                 'complete_format_sets':len(paired),'stem_count':len(stem),'sample_stems':sorted(paired)[:5],
                 'array_equality_verified':False})
references=[]
for base in (seg/'light_training',seg/'slurm',seg/'configs',seg/'model_segmamba'):
    if not base.is_dir():continue
    for p in base.rglob('*'):
        if not p.is_file() or p.is_symlink() or p.suffix not in ('.py','.sh','.slurm','.yaml') or p.stat().st_size>200000:continue
        for i,line in enumerate(p.read_text(errors='replace').splitlines()):
            if re.search(r'np\.load|mmap|unpack|\.npz|_seg\.npy|\.npy|mamba/build|causal.conv1d/build|puppeteer|torch_extensions',line):
                references.append({'path':str(p),'line':i+1,'text':line[:500]})
                if len(references)>=100:break
        if len(references)>=100:break
queue=command(['squeue','-h','-u','zqchen','-o','%i|%T|%j|%Z'],10)
# Process names only: no command-line arguments, credentials or environment.
processes=command(['ps','-u','zqchen','-o','pid=,etimes=,comm='],10)
capacity=os.statvfs(home)
p={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,
   'sizes':sizes,'conda_archives':archives,'conda_archive_allocated_bytes':sum(r['allocated_bytes'] for r in archives),
   'segmamba_format_inventory':data,'selected_loader_references':references,'user_queue':queue,'login_process_names':processes,
   'available_bytes':capacity.f_bavail*capacity.f_frsize,
   'boundary':'read-only metadata/public loader reference scan; no exact duplicate comparison, no clean/delete commands, no active job change; empty queue does not prove no future dependencies'}
print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(p,sort_keys=True),flags=re.I))
'''
    compile(code,'<readonly_cleanup_candidates>','exec')
    try:
        access.start_vpn()
        data=json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+m.PYTHON+' -B -c '+shlex.quote(code),timeout=220).strip().splitlines()[-1])
        data.update(m.META)
        data['local_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit/'CLEANUP_CANDIDATES.json',data)
        print(json.dumps({'audit':str(access.audit),'sizes':data['sizes'],'conda_archive_bytes':data['conda_archive_allocated_bytes'],
                          'formats':data['segmamba_format_inventory'],'queue':data['user_queue'],'references':data['selected_loader_references'][:35]},ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__=='__main__':
    main()
