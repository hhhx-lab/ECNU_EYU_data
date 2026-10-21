"""Authorized continuation of two held jobs; frozen runtime remains unchanged."""
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from common import META,MODS,cfg,load_model,now,own_package,parent,sha,write
PYTHON='/public/home/zqchen/.conda/envs/segmamba/bin/python'
KNOWN={'audit':'4165750','smoke_t1c':'4165751'}
RECOVERY=ROOT/'evidence/continue_held_20261021'


def call(argv,timeout=45):
    p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout,cwd=ROOT)
    v=dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,utc=now())
    assert p.returncode==0,json.dumps(v)
    return v


def fields(text):
    return dict(re.findall(r'([A-Za-z][A-Za-z0-9_/:]*)=([^\s]+)',text))


def dependency_ids(text):
    if text=='(null)':return set()
    tokens=text.split(',')
    result=set()
    for token in tokens:
        assert token.startswith('afterok:'),'Unexpected dependency type: '+text
        for item in token[len('afterok:'):].split(':'):
            match=re.fullmatch(r'([0-9]+)(?:\((?:unfulfilled|fulfilled|unsatisfied|satisfied)\))?',item)
            assert match,'Unexpected dependency syntax: '+text
            assert match[1] not in result,'Duplicate dependency: '+text
            result.add(match[1])
    return result


def main():
    jobs=dict(KNOWN);stage='preflight';release=[]
    assert sha(__file__)==os.environ['G1_PAIR_CONTINUATION_SHA256'],'Continuation operator drift'
    write(RECOVERY/'RESERVATION.json',dict(META,utc=now(),no_retry=True,authorization='User approved audit original held then complete eight and release'))
    try:
        c=cfg();pkg=own_package();ref=parent();e={}
        old=json.loads((ROOT/'evidence/FAILED.json').read_text())
        assert old['status']=='stopped' and old['stage']=='submit_smoke_t1c' and old['jobs']==KNOWN and old['released']==[]
        assert not (ROOT/'evidence/JOBS.json').exists() and not (ROOT/'evidence/SUBMISSION_RECEIPT.json').exists()
        assert not list((ROOT/'evidence').glob('*_RELEASE.json'))
        assert not (ROOT/'audit/STARTED.json').exists() and not (ROOT/'smoke/t1c/STARTED.json').exists()
        for phase,job in KNOWN.items():
            sub=json.loads((ROOT/'evidence'/(phase+'_SUBMIT.json')).read_text())
            assert sub['returncode']==0 and sub['stdout'].strip()==job
            held=call(['scontrol','show','job',job,'-o']);f=fields(held['stdout'])
            assert f['JobState']=='PENDING' and f['Reason']=='JobHeldUser' and f['RunTime']=='00:00:00'
            assert f['JobName']=='g1pair_'+phase and f['Command']==str(ROOT/'slurm'/(phase+'.slurm'))
            e[phase+'_existing_held']=held
        e['tests']=call([PYTHON,'-B','-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],180)
        expected_tests=json.loads((ROOT/'LOCAL_TESTS.json').read_text())['test_count']
        assert 'Ran '+str(expected_tests)+' tests' in e['tests']['stderr']
        e['models']={}
        for version in ('july150k','r9'):
            for mod in MODS:
                info=c['models'][version][mod]
                assert sha(info['path'])==info['sha256']
                e['models'][version+'_'+mod]=dict(sha256=sha(info['path']),bytes=Path(info['path']).stat().st_size)
        shapes=[]
        for r in ref['records']:
            assert Path(r['reference']).is_file()
            shapes.append(len(r['rows']))
        stat=os.statvfs(ROOT);available=stat.f_bavail*stat.f_frsize;assert available>16*1024**3
        e['storage']=dict(available_bytes=available,required_results_reserve_bytes=16*1024**3,no_new_image_cache=True)
        e['disk']=call(['df','-B1',str(ROOT)])
        for partition in ('a100','cpu_192G'):
            v=call(['scontrol','show','partition',partition,'-o']);f=fields(v['stdout'])
            assert f['State']=='UP' and f['ReqResv']=='NO' and f['AllowGroups']=='ALL'
            assert f['AllowAccounts']=='ALL' or 'zqchen' in f['AllowAccounts'].split(',')
            assert f['AllowQos']=='ALL' or 'user_zqchen' in f['AllowQos'].split(',')
            hosts=call(['scontrol','show','hostnames',f['Nodes']])['stdout'].splitlines()
            nodes=call(['scontrol','show','node',','.join(hosts),'-o']);suitable=[]
            for line in nodes['stdout'].splitlines():
                n=fields(line)
                if int(n['CPUTot'])>=8 and int(n['RealMemory'])-int(n.get('MemSpecLimit','0'))>=(65536 if partition=='a100' else 92160) and not any(s in n['State'] for s in ('DOWN','DRAIN','FAIL','NO_RESPOND','FUTURE')):
                    if partition!='a100' or 'gpu' in n.get('Gres','').lower():suitable.append(n['NodeName'])
            assert suitable,'No eligible node'
            e[partition]=dict(partition=v,node_metadata=nodes,suitable=suitable)
        e['qos']=call(['sacctmgr','-n','-P','show','qos','where','name=user_zqchen,partition_a100','format=Name,MaxWall,MaxJobsPU,MaxSubmitPU'])
        e['queue']=call(['squeue','-h','-u','zqchen','-o','%i|%T|%j|%P'])
        existing={s.split('|')[0] for s in e['queue']['stdout'].splitlines() if '|g1pair_' in s}
        assert existing==set(KNOWN.values()),'Unexpected comparison jobs'
        queued=sum('|a100' in s for s in e['queue']['stdout'].splitlines())
        rows=[s.split('|') for s in e['qos']['stdout'].splitlines() if s]
        caps=[int(r[3]) for r in rows if len(r)>3 and r[3]]
        assert not caps or queued+7<=min(caps),'A100 submission capacity insufficient'
        phases=['audit',*['smoke_'+m for m in MODS],*['eval_'+m for m in MODS],'summarize']
        for phase in phases:
            script=ROOT/'slurm'/(phase+'.slurm');call(['bash','-n',str(script)])
            dry=subprocess.run(['bash','-u',str(script)],env={**os.environ,'G1_STATIC_DRY_RUN':'1'},capture_output=True,text=True,timeout=10)
            assert dry.returncode==0 and 'G1_PAIR_DRY_RUN_PASS' in dry.stdout
        write(RECOVERY/'PREFLIGHT.json',dict(META,status='pass',utc=now(),package_sha256=pkg,original_failure_sha256=sha(ROOT/'evidence/FAILED.json'),**e))
        for phase in phases:
            stage='submit_'+phase
            deps=[] if phase=='audit' else [jobs['audit']] if phase.startswith('smoke_') else [jobs['smoke_'+m] for m in MODS] if phase.startswith('eval_') else [jobs['eval_'+m] for m in MODS]
            dependency='afterok:'+':'.join(deps) if deps else None
            script=ROOT/'slurm'/(phase+'.slurm')
            if phase not in KNOWN:
                write(RECOVERY/(phase+'_RESERVATION.json'),dict(META,utc=now(),dependency=dependency,script_sha256=sha(script)))
                argv=['sbatch','--parsable','--hold','--export=G1_PAIR_PACKAGE_SHA256='+pkg]
                if dependency:argv+=['--dependency='+dependency,'--kill-on-invalid-dep=yes']
                sub=call(argv+[str(script)]);write(RECOVERY/(phase+'_SUBMIT.json'),sub)
                job=sub['stdout'].strip().split(';')[0];assert re.fullmatch('[0-9]+',job);jobs[phase]=job
            else:job=jobs[phase]
            held=call(['scontrol','show','job',job,'-o']);f=fields(held['stdout'])
            assert f['JobState']=='PENDING' and f['Reason']=='JobHeldUser'
            assert f['RunTime']=='00:00:00' and f['Restarts']=='0' and f['Requeue']=='0'
            assert f['JobName']=='g1pair_'+phase and f['NumCPUs']=='8' and f['NumNodes'] in ('1','1-1') and f['NumTasks']=='1'
            assert f['Command']==str(script) and f['WorkDir']==str(ROOT)
            assert f['StdOut']==str(ROOT/'logs'/(phase+'_'+job+'.out')) and f['StdErr']==str(ROOT/'logs'/(phase+'_'+job+'.err'))
            gpu=phase.startswith(('smoke_','eval_'))
            assert f['Partition']==('a100' if gpu else 'cpu_192G')
            assert f['Account']=='zqchen' and f['QOS']=='user_zqchen'
            assert f['MinMemoryNode']==('64G' if gpu else '90G' if phase=='audit' else '32G')
            assert f['TimeLimit']==('02:00:00' if phase.startswith('smoke_') else '2-00:00:00' if gpu else '04:00:00')
            if gpu:assert 'gpu:1' in f.get('TRES','') or 'gres/gpu=1' in f.get('TRES','')
            assert dependency_ids(f['Dependency'])==set(deps),'Dependency mismatch'
            write(RECOVERY/(phase+'_HELD_AUDIT.json'),dict(META,status='held_audited',utc=now(),job=held,dependency_ids=deps,reused=phase in KNOWN))
        write(ROOT/'evidence/JOBS.json',dict(META,utc=now(),package_sha256=pkg,jobs=jobs))
        for phase in reversed(phases):
            stage='release_'+phase;v=call(['scontrol','release',jobs[phase]])
            release.append(phase);write(RECOVERY/(phase+'_RELEASE.json'),v)
        result=dict(META,status='pass',submission_status='released',utc=now(),remote_utc=now(),package_sha256=pkg,jobs=jobs,released=release,
                    queue=call(['squeue','-h','-j',','.join(jobs.values()),'-o','%i|%T|%j|%P|%r|%M']),
                    available_bytes=available,full_runtime_gate='all four smoke completed0:0+identity+measured<48h85pct',no_automatic_retry=True,
                    original_failure_sha256=sha(ROOT/'evidence/FAILED.json'),continuation_sha256=sha(__file__),existing_reused=KNOWN,new_submit_count=8)
        write(ROOT/'evidence/SUBMISSION_RECEIPT.json',result)
    except BaseException as error:
        result=dict(META,status='stopped',stage=stage,remote_utc=now(),jobs=jobs,released=release,message=str(error),traceback=traceback.format_exc())
        write(RECOVERY/'FAILED.json',result)
    result['evidence']={p.name:json.loads(p.read_text()) for p in RECOVERY.glob('*.json')}
    print(json.dumps(result))


if __name__=='__main__':main()
