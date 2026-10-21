"""One authorized held continuation, without redeployment or repeated original sbatch."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import shlex

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pair_retry_access',ROOT/'operator/retry_submit_20261021.py')
retry=importlib.util.module_from_spec(spec);spec.loader.exec_module(retry)
m=retry.m;access=retry.original.access


def main():
    tests=retry.validate()
    prior=ROOT/'operator/retry_submit_20261021T061828Z/RESULT.json'
    assert m.sha(prior)=='3c47edd5a889ea58283e4f6721f706d204986087d49c950e1b03ae55ac07bb4d'
    v=json.loads(prior.read_text())
    assert v['jobs']=={'audit':'4165750','smoke_t1c':'4165751'} and v['released']==[]
    source=ROOT/'operator/remote_continue_held_20261021.py';ast.parse(source.read_text())
    spec=importlib.util.spec_from_file_location('pair_continuation_parser',source)
    parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
    for state in ('unfulfilled','fulfilled','satisfied','unsatisfied'):
        assert parser.dependency_ids('afterok:123('+state+'),afterok:124('+state+')')=={'123','124'}
    assert parser.dependency_ids('afterok:123:124')=={'123','124'}
    for bad in ('afterany:123','afterok:123,afterok:123','afterok:123(bad)','afterok:123?afterok:124'):
        try:parser.dependency_ids(bad)
        except AssertionError:pass
        else:raise AssertionError('Invalid dependency accepted')
    remote=json.loads((ROOT/'RUN.json').read_text())['remote_root']
    remote_source=remote+'/operator/remote_continue_held_20261021.py';h=m.sha(source)
    def setup(a):
        m.exclusive(a.audit/'LOCAL_AUTHORIZATION_AND_TESTS.json',dict(m.META,user_request='核验后继续本次提交',
            package_sha256=retry.EXPECTED,continuation_sha256=h,prior_result_sha256=m.sha(prior),
            dependency_cases=9,test_count=13,stdout=tests.stdout,stderr=tests.stderr,no_automatic_retry=True))
        check='import pathlib;r=pathlib.Path('+repr(remote)+');assert r.is_dir();assert not (r/"operator/remote_continue_held_20261021.py").exists();assert not (r/"evidence/continue_held_20261021").exists();print("CONTINUATION_UNUSED")'
        a.ssh(m.PYTHON+' -B -c '+shlex.quote(check),timeout=45)
        a.scp(source,remote_source)
        a.ssh(m.PYTHON+' -B -c '+shlex.quote('import pathlib,hashlib;p=pathlib.Path('+repr(remote_source)+');assert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(h)+';p.chmod(0o440);print("OPERATOR_SHA_PASS")'),timeout=45)
    code='import os,runpy;os.environ["G1_PAIR_PACKAGE_SHA256"]='+repr(retry.EXPECTED)+';os.environ["G1_PAIR_CONTINUATION_SHA256"]='+repr(h)+';runpy.run_path('+repr(remote_source)+',run_name="__main__")'
    result,audit=access.run_remote('continue_held',code,timeout=480,setup=setup)
    assert result['submission_status']=='released' and result['package_sha256']==retry.EXPECTED
    assert len(result['jobs'])==len(result['released'])==10 and result['existing_reused']==v['jobs'] and result['new_submit_count']==8
    assert result['jobs']['audit']=='4165750' and result['jobs']['smoke_t1c']=='4165751'
    for phase in result['jobs']:
        assert result['evidence'][phase+'_HELD_AUDIT.json']['status']=='held_audited'
        assert result['evidence'][phase+'_RELEASE.json']['returncode']==0
    m.exclusive(audit/'LOCAL_SUBMISSION_ACCEPTANCE.json',dict(m.META,status='pass',utc=m.datetime.now(m.timezone.utc).isoformat(),
        package_sha256=retry.EXPECTED,continuation_sha256=h,jobs=result['jobs'],release_count=10,
        reused_count=2,new_submit_count=8,remote_utc=result['remote_utc'],local_received_utc=result['local_received_utc']))
    print(json.dumps(dict(status='released',jobs=result['jobs'],queue=result['queue'],audit=str(audit))))


if __name__=='__main__':main()
