"""One reservation per stage; audit all held jobs before releasing any."""
import argparse
import csv
import math
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from common import META, MODS, now, settings, sha, write
PYTHON='/public/home/zqchen/.conda/envs/segmamba/bin/python'


def call(argv,timeout=45):
    r=subprocess.run(argv,capture_output=True,text=True,timeout=timeout,cwd=ROOT)
    value=dict(argv=argv,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,utc=now())
    assert r.returncode==0,json.dumps(value)
    return value


def fields(s):
    return dict(re.findall(r'([A-Za-z][A-Za-z0-9_/:]*)=([^\s]+)',s))


def complete_jobs(jobs):
    a=call(['sacct','-n','-P','-j',','.join(jobs),'--format=JobID,State,ExitCode'])
    rows={r.split('|')[0]:r.split('|')[1:3] for r in a['stdout'].splitlines() if len(r.split('|'))>=3}
    assert all(rows.get(j)==['COMPLETED','0:0'] and rows.get(j+'.batch')==['COMPLETED','0:0'] for j in jobs)
    return a


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('smoke','full'),required=True);a=p.parse_args()
    evidence=ROOT/'evidence'/a.stage;evidence.mkdir(parents=True,exist_ok=False)
    write(evidence/'RESERVATION.json',dict(META,utc=now(),stage=a.stage,no_retry=True))
    jobs={};stage='preflight'
    try:
        cfg,r9=settings();preflight={}
        if a.stage=='smoke':
            preflight['r9_accounting']=complete_jobs(['4128262','4128266','4128267','4128268','4128269'])
            dataset=json.loads((r9/'dataset/COMPLETE.ok').read_text())
            assert dataset['cases']==926 and dataset['split_counts']==dict(train=823,val=103)
            assert dataset['status']=='complete' and not (r9/'dataset/FAILED.json').exists()
            assert sha(r9/'dataset/PREPROCESS_REPORT.json')==dataset['report_sha256']
            with (r9/'plan/splits_final_train_val_test_membership.csv').open() as f:members=list(csv.DictReader(f))
            names=[m['source_case_id'] for m in members if m['split']=='val'];assert len(set(names))==103
            shapes=[]
            for name in names:
                p=Path(dataset['case_locations'][name])/'preprocess_complete.json'
                assert sha(p)==dataset['case_manifests'][name]
                shapes.append(json.loads(p.read_text())['contract']['shape'])
            data_upper=sum(math.prod(s)*33 for s in shapes)
            required=int(data_upper*1.01+max(math.prod(s)*200 for s in shapes)+16*1024**3)
            preflight['storage_estimate']=dict(val_cases=103,data_upper_bytes=data_upper,required_bytes=required,
                method='val103_marker_shapes_33bytes_per_voxel_1pct_zip_bound_largest200bytes_temp_16GiB_reserve')
            inventory={}
            for mod in MODS:
                for step in cfg['candidate_steps']:
                    f=r9/'training'/mod/'weights'/f'diffusion_{step}.pt'
                    assert f.is_file() and not f.is_symlink() and f.resolve()==f
                    inventory[mod+'_'+str(step)]=dict(bytes=f.stat().st_size,sha256=sha(f))
            preflight['candidate_inventory']=inventory
            preflight['tests']=call([PYTHON,'-B','-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],180)
            assert 'Ran 0 tests' not in preflight['tests']['stderr']
            phases=['prepare',*['smoke_'+m for m in MODS]]
        else:
            previous=json.loads((ROOT/'evidence/smoke/SUBMISSION_RECEIPT.json').read_text())
            assert previous['status']=='released' and previous['package_sha256']==sha(ROOT/'PACKAGE_MANIFEST.json')
            preflight['smoke_accounting']=complete_jobs(list(previous['jobs'].values()))
            ref=json.loads((ROOT/'reference/COMPLETE.ok').read_text())
            assert ref['status']=='pass' and ref['report_sha256']==sha(ROOT/'reference/REPORT.json')
            assert not (ROOT/'reference/FAILED.json').exists()
            reports={}
            for mod in MODS:
                d=ROOT/'smoke'/mod;v=json.loads((d/'COMPLETE.ok').read_text())
                assert v['status']=='smoke_pass' and v['modality']==mod and v['sampled_cubes']>0
                assert not (d/'FAILED.json').exists() and v['estimated_full_seconds']<48*3600*.85
                assert v['reference_report_sha256']==ref['report_sha256'] and v['evaluation_package_sha256']==sha(ROOT/'PACKAGE_MANIFEST.json')
                assert v['candidate_summary_sha256']['200000']==sha(d/'200000/CANDIDATE_COMPLETE.json')
                reports[mod]=v
            preflight['smoke_reports']=reports
            phases=[*['eval_'+m for m in MODS],'select']
        stat=os.statvfs(ROOT);available=stat.f_bavail*stat.f_frsize
        assert available>(required if a.stage=='smoke' else 16*1024**3)
        preflight['disk']=call(['df','-B1',str(ROOT)]);preflight['available_bytes']=available
        for partition in ('cpu_192G','a100'):
            v=call(['scontrol','show','partition',partition,'-o']);f=fields(v['stdout'])
            assert f['State']=='UP' and f['ReqResv']=='NO' and f['AllowGroups']=='ALL'
            assert f['AllowAccounts']=='ALL' or 'zqchen' in f['AllowAccounts'].split(',')
            assert f['AllowQos']=='ALL' or 'user_zqchen' in f['AllowQos'].split(',')
            hosts=call(['scontrol','show','hostnames',f['Nodes']])['stdout'].splitlines()
            nodes=call(['scontrol','show','node',','.join(hosts),'-o']);valid=[]
            for line in nodes['stdout'].splitlines():
                n=fields(line);cap=int(n['RealMemory'])-int(n.get('MemSpecLimit','0'))
                if cap>=(92160 if partition=='cpu_192G' else 65536) and int(n['CPUTot'])>=8 and not any(s in n['State'] for s in ('DOWN','DRAIN','FAIL','NO_RESPOND','FUTURE')):
                    if partition!='a100' or 'gpu' in n.get('Gres','').lower():valid.append(n['NodeName'])
            assert valid,'No compatible node: '+partition
            preflight[partition]=dict(partition=v,suitable_nodes=valid)
        preflight['qos']=call(['sacctmgr','-n','-P','show','qos','where','name=user_zqchen,partition_a100','format=Name,MaxWall,MaxJobsPU,MaxSubmitPU'])
        preflight['queue']=call(['squeue','-h','-n',','.join('g1r9v_'+v for v in phases),'-o','%i|%T|%j'])
        assert not preflight['queue']['stdout'].strip(),'Duplicate evaluation stage'
        for phase in phases:call(['bash','-n',str(ROOT/'slurm'/(phase+'.slurm'))])
        write(evidence/'PREFLIGHT.json',dict(META,utc=now(),status='pass',**preflight))
        for phase in phases:
            stage='submit_'+phase
            dependency='afterok:'+jobs['prepare'] if phase.startswith('smoke_') else None
            if phase=='select':dependency='afterok:'+':'.join(jobs.values())
            write(evidence/(phase+'_RESERVATION.json'),dict(META,utc=now(),dependency=dependency,script_sha256=sha(ROOT/'slurm'/(phase+'.slurm'))))
            argv=['sbatch','--parsable','--hold','--export=NONE']
            if dependency:argv+=['--dependency='+dependency,'--kill-on-invalid-dep=yes']
            response=call(argv+[str(ROOT/'slurm'/(phase+'.slurm'))]);write(evidence/(phase+'_SUBMIT.json'),response)
            job=response['stdout'].strip().split(';')[0];assert re.fullmatch('[0-9]+',job);jobs[phase]=job
            held=call(['scontrol','show','job',job,'-o']);f=fields(held['stdout'])
            assert f['JobState']=='PENDING' and f['Reason']=='JobHeldUser'
            assert f['JobName']=='g1r9v_'+phase and f['NumCPUs']=='8' and f['NumNodes'] in ('1','1-1') and f['NumTasks']=='1'
            assert f['Command']==str(ROOT/'slurm'/(phase+'.slurm'))
            assert f['StdOut']==str(ROOT/'logs'/(phase+'_'+job+'.out')) and f['StdErr']==str(ROOT/'logs'/(phase+'_'+job+'.err'))
            gpu=phase.startswith(('smoke_','eval_'))
            assert f['Partition']==('a100' if gpu else 'cpu_192G')
            memory='64G' if gpu else ('90G' if phase=='prepare' else '32G')
            assert f.get('MinMemoryNode')==memory
            assert f['WorkDir']==str(ROOT) and f['QOS']=='user_zqchen' and f['Account']=='zqchen'
            assert f['TimeLimit']==('02:00:00' if phase.startswith('smoke_') else '2-00:00:00' if gpu else '1-00:00:00' if phase=='prepare' else '04:00:00')
            if gpu:assert 'gpu:1' in f.get('TRES','') or 'gres/gpu=1' in f.get('TRES','')
            if dependency:assert dependency in f['Dependency']
            else:assert f['Dependency']=='(null)'
            write(evidence/(phase+'_HELD_AUDIT.json'),dict(META,status='held_audited',utc=now(),job=held,dependency=dependency))
        for phase in reversed(phases):
            stage='release_'+phase
            response=call(['scontrol','release',jobs[phase]])
            write(evidence/(phase+'_RELEASE.json'),response)
        receipt=dict(META,status='released',jobs=jobs,remote_utc=now(),package_sha256=sha(ROOT/'PACKAGE_MANIFEST.json'),
            queue=call(['squeue','-h','-j',','.join(jobs.values()),'-o','%i|%T|%j|%P|%r|%M']),available_bytes=available)
        write(evidence/'SUBMISSION_RECEIPT.json',receipt)
    except BaseException as error:
        receipt=dict(META,status='stopped',stage=stage,jobs_already_created=jobs,remote_utc=now(),message=str(error),traceback=traceback.format_exc())
        write(evidence/'FAILED.json',receipt)
    receipt['evidence']={p.name:json.loads(p.read_text()) for p in evidence.glob('*.json')}
    print(json.dumps(receipt))


if __name__=='__main__':main()
