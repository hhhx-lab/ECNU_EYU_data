"""User-authorized, one-shot release of the exact five already-held jobs."""
import argparse
import json
import re
import shlex
from pathlib import Path

from common import ROOT, REMOTE, m, transport

JOBS = dict(eval_t1c='4159189', eval_t1n='4159190', eval_t2w='4159191',
            eval_t2f='4159192', select='4159193')
PACKAGE_SHA = '3486d52cfa995aff8cd439932384d8393224b90630cb878793aba01f7b41d2d4'


def dependency_ids(value):
    if value == '(null)':
        return set()
    terms = value.split(',')
    ids = []
    for term in terms:
        match = re.fullmatch(r'afterok:([0-9]+)(?:\(unfulfilled\))?', term)
        assert match, 'Unexpected dependency term: ' + term
        ids.append(match.group(1))
    assert len(ids) == len(set(ids)), 'Duplicate dependency'
    return set(ids)


def self_test():
    assert dependency_ids('(null)') == set()
    assert dependency_ids('afterok:1(unfulfilled),afterok:2(unfulfilled)') == {'1', '2'}
    assert dependency_ids('afterok:2,afterok:1') == {'1', '2'}
    for value in ('afterany:1', 'afterok:1?afterok:2', 'afterok:1:2',
                  'afterok:1(fulfilled)', 'afterok:1,afterok:1', '',
                  'afterok:1,afterok:2junk'):
        try:
            dependency_ids(value)
        except AssertionError:
            pass
        else:
            raise AssertionError('Accepted unsafe dependency: ' + value)
    print('10 dependency parser checks passed')


REMOTE_CODE = r'''
import json, os, re, sys, traceback
from pathlib import Path
root = Path(__ROOT__)
sys.path.insert(0, str(root / 'runtime'))
from common import META, MODS, now, settings, sha, write
import importlib.util
spec = importlib.util.spec_from_file_location('frozen_launch_audit', root/'operator/remote_launch.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)
__PARSER__
jobs = __JOBS__
evidence = root/'evidence/full_release_existing_20261020'
evidence.mkdir(exist_ok=False)
write(evidence/'RESERVATION.json', dict(META, utc=now(), jobs=jobs,
    authorization='2026-10-20 user: audit and release original five held jobs; no new sbatch',
    no_retry=True, recovery_operator_sha256=__OPERATOR_SHA__))
stage = 'audit'
released = []
try:
    cfg, r9 = settings()
    assert sha(root/'PACKAGE_MANIFEST.json') == __PACKAGE_SHA__
    original = json.loads((root/'evidence/full/FAILED.json').read_text())
    assert original == __ORIGINAL_FAILURE__
    assert original['status']=='stopped' and original['stage']=='submit_select'
    assert original['jobs_already_created']==jobs
    assert "if dependency:assert dependency in f['Dependency']" in original['traceback']
    assert not (root/'evidence/full/SUBMISSION_RECEIPT.json').exists()
    assert not list((root/'evidence/full').glob('*_RELEASE.json'))
    pre = json.loads((root/'evidence/full/PREFLIGHT.json').read_text())
    assert pre['status']=='pass'
    accounting = launch.complete_jobs(['4145277','4145278','4145279','4145280','4145281'])
    ref = json.loads((root/'reference/COMPLETE.ok').read_text())
    assert ref['status']=='pass' and ref['report_sha256']==sha(root/'reference/REPORT.json')
    assert not (root/'reference/FAILED.json').exists()
    report = json.loads((root/'reference/REPORT.json').read_text())
    assert report['cases']==103 and report['positive_cases']==94 and report['negative_cases']==9
    assert report['errors']==[]
    for mod in MODS:
        d=root/'smoke'/mod
        v=json.loads((d/'COMPLETE.ok').read_text())
        assert v==pre['smoke_reports'][mod]
        assert v['modality']==mod and v['status']=='smoke_pass' and v['sampled_cubes']>0
        assert v['estimated_full_seconds']<48*3600*.85
        assert v['reference_report_sha256']==ref['report_sha256']
        assert v['evaluation_package_sha256']==sha(root/'PACKAGE_MANIFEST.json')
        assert v['candidate_summary_sha256']['200000']==sha(d/'200000/CANDIDATE_COMPLETE.json')
        assert not (d/'FAILED.json').exists()
        assert not (root/'evaluation'/mod/'STARTED.json').exists()
    assert not (root/'selection/STARTED.json').exists()
    current_queue=launch.call(['squeue','-h','-n',','.join('g1r9v_'+p for p in jobs),'-o','%i|%T|%j'])
    expected={(j,'PENDING','g1r9v_'+p) for p,j in jobs.items()}
    assert {tuple(r.split('|')) for r in current_queue['stdout'].splitlines()}==expected
    audits={}
    for phase,job in jobs.items():
        raw=launch.call(['scontrol','show','job',job,'-o'])
        f=launch.fields(raw['stdout'])
        gpu=phase!='select'
        assert f['JobId']==job and f['JobName']=='g1r9v_'+phase
        assert f['UserId']=='zqchen(2159)' and f['Account']=='zqchen' and f['QOS']=='user_zqchen'
        assert f['JobState']=='PENDING' and f['Reason']=='JobHeldUser' and f['Priority']=='0'
        assert f['RunTime']=='00:00:00' and f['Restarts']=='0' and f['Requeue']=='0'
        assert f['NumCPUs']=='8' and f['NumTasks']=='1' and f['NumNodes'] in ('1','1-1')
        assert f['Partition']==('a100' if gpu else 'cpu_192G')
        assert f['MinMemoryNode']==('64G' if gpu else '32G')
        assert f['TimeLimit']==('2-00:00:00' if gpu else '04:00:00')
        assert f['Command']==str(root/'slurm'/(phase+'.slurm')) and f['WorkDir']==str(root)
        assert f['StdOut']==str(root/'logs'/(phase+'_'+job+'.out'))
        assert f['StdErr']==str(root/'logs'/(phase+'_'+job+'.err'))
        tres=dict(term.split('=',1) for term in f['TRES'].split(','))
        assert tres['cpu']=='8' and tres['node']=='1' and tres['mem']==('64G' if gpu else '32G')
        assert (tres.get('gres/gpu')=='1') if gpu else 'gres/gpu' not in tres
        assert dependency_ids(f['Dependency'])==(set(jobs.values())-{job} if not gpu else set())
        if not gpu:
            assert f['KillOInInvalidDependent']=='Yes'
        audits[phase]=dict(job=raw,parsed_dependency_ids=sorted(dependency_ids(f['Dependency'])),
                           script_sha256=sha(root/'slurm'/(phase+'.slurm')))
    available=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize
    assert available>16*1024**3
    write(evidence/'ALL_HELD_AUDITS.json',dict(META,status='pass',utc=now(),jobs=jobs,
          audits=audits,prerequisite_accounting=accounting,queue=current_queue,
          available_bytes=available,package_sha256=sha(root/'PACKAGE_MANIFEST.json'),
          original_failure_sha256=sha(root/'evidence/full/FAILED.json')))
    for phase in reversed(list(jobs)):
        stage='release_'+phase
        response=launch.call(['scontrol','release',jobs[phase]])
        released.append(jobs[phase])
        write(evidence/(phase+'_RELEASE.json'),response)
    receipt=dict(META,status='released',jobs=jobs,remote_utc=now(),
        package_sha256=sha(root/'PACKAGE_MANIFEST.json'),available_bytes=available,
        queue=launch.call(['squeue','-h','-j',','.join(jobs.values()),'-o','%i|%T|%j|%P|%r|%M']),
        recovery_evidence=str(evidence.relative_to(root)),
        original_failure_sha256=sha(root/'evidence/full/FAILED.json'),
        note='Original stopped launch preserved; exact existing held jobs audited and released; no sbatch')
    write(evidence/'SUBMISSION_RECEIPT.json',receipt)
    write(root/'evidence/full/SUBMISSION_RECEIPT.json',receipt)
except BaseException as error:
    receipt=dict(META,status='stopped',stage=stage,jobs=jobs,released_jobs=released,
        remote_utc=now(),message=str(error),traceback=traceback.format_exc())
    write(evidence/'FAILED.json',receipt)
receipt['evidence']={p.name:json.loads(p.read_text()) for p in evidence.glob('*.json')}
print(json.dumps(receipt))
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test()
        return
    reservation=ROOT/'EXISTING_FULL_RELEASE_RESERVATION_20261020.json'
    source=ROOT/'operator/full_20261020T030545Z/SUBMISSION_RECEIPT.json'
    original=json.loads(source.read_text())['evidence']['FAILED.json']
    assert original['jobs_already_created']==JOBS
    import inspect
    code=REMOTE_CODE.replace('__PARSER__',inspect.getsource(dependency_ids))
    for key,value in dict(__ROOT__=str(REMOTE),__JOBS__=JOBS,__PACKAGE_SHA__=PACKAGE_SHA,
                          __ORIGINAL_FAILURE__=original,__OPERATOR_SHA__=m.sha(Path(__file__))).items():
        code=code.replace(key,repr(value))
    compile(code,'<exact-held-release>','exec')
    m.exclusive(reservation,dict(m.META,jobs=JOBS,utc=m.datetime.now(m.timezone.utc).isoformat(),
        authorization='2026-10-20 explicit user approval for release of exact existing jobs',no_retry=True))
    access=transport('release_existing_full')
    started=m.datetime.now(m.timezone.utc).isoformat()
    result=None
    try:
        access.start_vpn()
        out=access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+m.PYTHON+' -B -c '+shlex.quote(code),timeout=120)
        result=json.loads(out.strip().splitlines()[-1])
        result['local_utc']=m.datetime.now(m.timezone.utc).isoformat()
        m.exclusive(access.audit/'SUBMISSION_RECEIPT.json',result)
        print(json.dumps(dict(audit=str(access.audit),**{k:v for k,v in result.items() if k!='evidence'})))
        assert result['status']=='released','Exact release stopped; preserve evidence, no retry'
    finally:
        access.close_vpn()
        m.exclusive(access.audit/'LOCAL_OPERATION_RECEIPT.json',dict(m.META,started_utc=started,
            local_completed_utc=m.datetime.now(m.timezone.utc).isoformat(),vpn_owned=access.owns_vpn,
            own_vpn_stop_completed=access.owns_vpn,preexisting_vpn_left_unchanged=not access.owns_vpn,
            result_status=result['status'] if result else 'unconfirmed'))


if __name__=='__main__':
    main()
