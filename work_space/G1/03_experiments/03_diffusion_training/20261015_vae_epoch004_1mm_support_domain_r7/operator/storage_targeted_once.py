"""Read-only targeted follow-up; retain the earlier partial inventory."""
import datetime
import hashlib
import json
import shlex

from common import REMOTE, ROOT, m, transport


def remote_code(storage):
    code = r'''
import csv,datetime,json,os,pathlib,subprocess,time
root=pathlib.Path(__ROOT__)
home=pathlib.Path('/public/home/zqchen')
ws=home/'projects/ECNU_EYU_data/work_space'
budget=__BUDGET__
started=time.monotonic()
def cmd(args,seconds=15):
    t=time.monotonic()
    try:
        r=subprocess.run(args,capture_output=True,text=True,timeout=seconds)
        return dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr[-2000:],seconds=round(time.monotonic()-t,3))
    except subprocess.TimeoutExpired as e:
        dec=lambda v:v.decode(errors='replace') if isinstance(v,bytes) else (v or '')
        return dict(exit_code=124,stdout=dec(e.stdout),stderr=dec(e.stderr)[-2000:],seconds=round(time.monotonic()-t,3))
def capacity(path):
    s=os.statvfs(path)
    return dict(total_bytes=s.f_blocks*s.f_frsize,available_bytes=s.f_bavail*s.f_frsize,
                used_bytes=(s.f_blocks-s.f_bfree)*s.f_frsize)
def du(path,depth,seconds):
    if not path.exists():return dict(path=str(path),exists=False,complete=False)
    r=cmd(['du','-x','-B1','--max-depth='+str(depth),str(path)],seconds)
    rows=[]
    for line in r.pop('stdout').splitlines():
        fields=line.split('\t',1)
        if len(fields)==2 and fields[0].isdigit():rows.append(dict(path=fields[1],allocated_bytes=int(fields[0])))
    total=next((x['allocated_bytes'] for x in rows if x['path']==str(path)),None)
    return dict(path=str(path),exists=True,complete=r['exit_code']==0 and total is not None,
                allocated_bytes=total,rows=rows,**r)
observed_start=datetime.datetime.now(datetime.timezone.utc).isoformat()
v=dict(remote_start_utc=observed_start,hostname=os.uname().nodename,capacity_start=capacity(home),
       commands={n:cmd(a,10) for n,a in {
          'df_home':['df','-B1',str(home)],'df_public':['df','-B1','/public'],
          'quota':['quota','-s'],
          'jobs':['squeue','-h','-j','4096309,4096310,4096311,4096312,4096313,4100381','-o','%i|%T|%j|%P|%r|%M'],
          'accounting':['sacct','-n','-P','-j','4096309,4096310,4096311,4096312,4096313,4100381',
                        '--format=JobID,State,ExitCode,Elapsed,AllocTRES']}.items()})
targets=[(home/'zdn_deepcpg',1,15),(home/'data',1,20),(home/'.cache',1,10),
         (home/'.conda/pkgs',0,12),(ws/'G1/code/brats2025-latent-ensemble-synthesis-main-v3/data',1,10),
         (ws/'G1/code/brats2025-latent-ensemble-synthesis-main-v3/training',1,10),
         (root.parent/'g1_vae_epoch004_full_reconstruction_20261008_r2',1,10),
         (ws/'S2/experimental_unvalidated',1,20),(ws/'S3',1,10)]
v['targeted_usage']=[]
for p,depth,limit in targets:
    if time.monotonic()-started>135:
        v['targeted_usage'].append(dict(path=str(p),complete=False,not_scanned='overall_time_bound'))
    else:v['targeted_usage'].append(du(p,depth,limit))
v['top_level_directories']={str(p):[dict(name=x.name,symlink=x.is_symlink()) for x in sorted(p.iterdir()) if x.is_dir()]
                            for p in (home,home/'projects',ws)}
v['retired_segmamba_exists']=(home/'SegMamba-main').exists()
members=list(csv.DictReader((root/'plan/splits_final_train_val_test_membership.csv').open()))
active={r['source_case_id'] for r in members if r['split'] in ('train','val')}
assert len(active)==926
cases=root/'dataset/cases'
completed=[];incomplete=[];errors=[];payload_bound=0;actual_alloc=0;actual_logical=0
def compressed_bound(n):
    total=0
    for b in (4*n,4*n,4*n,4*n,n,n):
        b+=4096
        total+=b+(b>>12)+(b>>14)+(b>>25)+13+18
    return total
for p in sorted(cases.iterdir()) if cases.exists() else []:
    if not p.is_dir() or p.is_symlink():continue
    marker=p/'preprocess_complete.json'
    if not marker.is_file():incomplete.append(p.name);continue
    try:
        data=json.loads(marker.read_text())
        assert p.name in active and data['case_id']==p.name and data['status']=='complete'
        shape=data['contract']['shape']; assert len(shape)==3 and all(int(x)>0 for x in shape)
        outputs=data['outputs']; assert len(outputs)==6
        files=[p/x for x in outputs]
        assert all(x.is_file() and not x.is_symlink() and x.stat().st_size>0 for x in files)
        n=int(shape[0])*int(shape[1])*int(shape[2]);bound=compressed_bound(n)
        alloc=sum(x.stat().st_blocks*512 for x in files);logical=sum(x.stat().st_size for x in files)
        payload_bound+=bound;actual_alloc+=alloc;actual_logical+=logical
        completed.append(dict(case_id=p.name,shape=shape,compressed_bound_bytes=bound,
                              allocated_bytes=alloc,logical_bytes=logical))
    except (OSError,ValueError,KeyError,AssertionError) as e:errors.append(dict(case=p.name,error=str(e)))
remaining=max(0,budget['compressed_worst_bound_bytes']-payload_bound)
future=remaining+budget['temporary_upper_bytes']+budget['checkpoint_reserve_bytes']+budget['metadata_reserve_bytes']+4*1024**3
v['g1_remaining_budget']=dict(original_budget=budget,completed_metadata_checked=len(completed),cases=926,
    completed=completed,incomplete_case_directories=incomplete,errors=errors,
    completed_payload_allocated_bytes=actual_alloc,completed_payload_logical_bytes=actual_logical,
    completed_payload_bound_bytes=payload_bound,remaining_payload_upper_bytes=remaining,
    additional_conservative_reserve_bytes=future,
    method='subtract completed-case full-FOV zlib bounds, not compressed bytes; keep all temp/checkpoint/metadata/safety reserves',
    verification='completion JSON identity, shape and six nonempty files only; no file SHA/content/geometry revalidation')
v['dataset_usage']=du(root/'dataset',1,8)
v['training_exists']=(root/'training').exists()
v['capacity_end']=capacity(home)
v['remote_end_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
v['seconds']=round(time.monotonic()-started,3)
print(json.dumps(v,sort_keys=True))
'''
    return code.replace('__ROOT__', repr(str(REMOTE))).replace('__BUDGET__', repr(storage))


def main():
    storage = json.loads((ROOT / 'operator/submit_20261015T133410Z/SUBMISSION_RECEIPT.json').read_text())['storage']
    code = remote_code(storage)
    compile(code, 'readonly_storage_targeted', 'exec')
    access = transport('storage_targeted')
    value = None
    local_start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        access.start_vpn()
        output = access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -u -c ' + shlex.quote(code), timeout=185)
        value = json.loads(output.strip().splitlines()[-1])
        value.update(m.META)
        value['local_start_utc'] = local_start
        value['local_received_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit / 'STORAGE_TARGETED.json', value)
    finally:
        access.close_vpn()
    evidence = access.audit / 'STORAGE_TARGETED.json'
    m.exclusive(access.audit / 'LOCAL_OPERATION_RECEIPT.json', dict(
        **m.META, status='read_only_snapshot_received', local_end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        file_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        vpn_started_by_operation=access.owns_vpn, operation_vpn_closed=True,
        remote_writes=False, remote_submissions=False, large_data_downloaded=False))
    print(json.dumps(dict(audit=str(access.audit),capacity=value['capacity_end'],
                         g1_additional_reserve_bytes=value['g1_remaining_budget']['additional_conservative_reserve_bytes'])))


if __name__ == '__main__':
    main()
