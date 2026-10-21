"""Deploy K1 MET to a new root, preflight, and release one audited GPU job."""

import hashlib
import importlib.util
import json
import shlex
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCESS_FILE = ROOT.parent / 's2_met_aug_k1245_20260923_r3_fallback/operator/access.py'
REMOTE = '/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/' + ROOT.name
META = dict(artifact_status='experimental_unvalidated', operator_approved=False,
            formal_gate_status='not_run_not_passed')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1048576), b''):
            h.update(part)
    return h.hexdigest()


def main():
    spec = importlib.util.spec_from_file_location('k1245_access', ACCESS_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    access = mod.Access('deploy_k1_met103_20261008')
    started = datetime.now(timezone.utc).isoformat()
    succeeded = False
    try:
        plan = json.loads((ROOT / 'EXECUTION_PLAN.json').read_text())
        assert plan['remote_root'] == REMOTE and plan['routes'].keys() == {'K1'}
        expected = json.loads((ROOT / 'STATIC_SHA256.json').read_text())['files']
        for relative, digest in expected.items():
            assert sha(ROOT / relative) == digest, relative
        files = [p for p in ROOT.rglob('*') if p.is_file() and 'operator' not in p.relative_to(ROOT).parts]
        stage = access.audit / 'transfer_bundle'
        stage.mkdir()
        for source in files:
            target = stage / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        (stage / 'logs').mkdir()
        access.start_vpn()
        probe = access.ssh('if test -e ' + shlex.quote(REMOTE) +
                           '; then printf EXISTS; else printf ABSENT; fi', timeout=60)
        if not probe.rstrip().endswith('ABSENT'):
            raise RuntimeError('remote attempt root already exists; no redeploy')
        access.call(['scp', '-r', '-P', access.env['HPC_LOGIN_PORT'],
                     '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15', str(stage),
                     f"{access.env['HPC_SSH_USER']}@{access.env['HPC_LOGIN_HOST']}:{REMOTE}"], timeout=180)
        check = ('import hashlib,pathlib,json; root=pathlib.Path(' + repr(REMOTE) +
                 '); expected=' + repr({str(p.relative_to(ROOT)): sha(p) for p in files}) +
                 '; assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==v for p,v in expected.items());'
                 'print("K1_MET_STATIC_SHA_PASS")')
        result = access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + mod.PYTHON +
                            ' -B -c ' + shlex.quote(check), timeout=120)
        if 'K1_MET_STATIC_SHA_PASS' not in result:
            raise RuntimeError('remote static SHA confirmation absent')
        command = ('env PYTHONDONTWRITEBYTECODE=1 MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=2 PYTHONPATH=' +
                   shlex.quote(REMOTE + '/runtime') + ' ' + mod.PYTHON + ' -B ' +
                   shlex.quote(REMOTE + '/runtime/workflow.py') + ' preflight')
        preflight = access.ssh(command, timeout=900)
        if '"status": "pass"' not in preflight:
            raise RuntimeError('preflight pass output absent')
        mod.exclusive(access.audit / 'PREFLIGHT_RECEIPT.json',
                      dict(META, status='pass', plan_sha256=sha(ROOT / 'EXECUTION_PLAN.json'),
                           remote_output_tail=preflight[-2200:]))
        launch = r'''
import datetime as dt,hashlib,json,os,pathlib,subprocess
root=pathlib.Path(REMOTE_ROOT)
plan=json.loads((root/'EXECUTION_PLAN.json').read_text())
meta=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed')
def emit(name,payload):
 payload={**meta,'remote_utc':dt.datetime.now(dt.timezone.utc).isoformat(),**payload}
 fd=os.open(root/'evidence'/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o440)
 with os.fdopen(fd,'w') as f:json.dump(payload,f,indent=2,sort_keys=True);f.write('\n')
pre=json.loads((root/'evidence/REMOTE_STATIC_PREFLIGHT.json').read_text())
assert pre['status']=='pass' and pre['plan_sha256']==hashlib.sha256((root/'EXECUTION_PLAN.json').read_bytes()).hexdigest()
assert not (root/'evidence/REMOTE_STATIC_PREFLIGHT_FAILED.json').exists()
assert not (root/'evidence/K1_RESERVATION.json').exists()
name=plan['job_names']['K1']
q=subprocess.run(['squeue','-h','-n',name,'-o','%i'],capture_output=True,text=True,check=True)
assert not q.stdout.strip(),'same job name already queued'
emit('K1_RESERVATION.json',dict(status='reserved',job_name=name,plan_sha256=pre['plan_sha256']))
r=subprocess.run(['sbatch','--hold','--parsable','--job-name='+name,str(root/'slurm/evaluate.slurm'),'K1'],capture_output=True,text=True)
if r.returncode:
 emit('K1_SUBMISSION_FAILED.json',dict(status='failed',return_code=r.returncode,stderr=r.stderr))
 print('K1_MET_LAUNCH='+json.dumps(dict(status='submission_failed',stderr=r.stderr)));raise SystemExit(0)
jid=r.stdout.strip().split(';')[0];assert jid.isdigit()
emit('K1_LAUNCH.json',dict(status='submitted_held',route='K1',job_id=jid,job_name=name))
rel=subprocess.run(['scontrol','release',jid],capture_output=True,text=True)
if rel.returncode:
 emit('K1_RELEASE_FAILED.json',dict(status='failed',job_id=jid,stderr=rel.stderr))
 print('K1_MET_LAUNCH='+json.dumps(dict(status='release_failed',job_id=jid)));raise SystemExit(0)
emit('K1_RELEASED.json',dict(status='released',route='K1',job_id=jid))
q=subprocess.run(['squeue','-j',jid,'-h','-o','%i|%j|%T|%R'],capture_output=True,text=True)
print('K1_MET_LAUNCH='+json.dumps(dict(status='released',job_id=jid,squeue_rc=q.returncode,squeue=q.stdout.strip(),squeue_stderr=q.stderr.strip())))
'''.replace('REMOTE_ROOT', repr(REMOTE))
        raw = access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + mod.PYTHON + ' -B -c ' +
                         shlex.quote(launch), timeout=120)
        line = next((item for item in raw.splitlines() if item.startswith('K1_MET_LAUNCH=')), None)
        if line is None:
            raise RuntimeError('structured launch audit result absent')
        result = json.loads(line.partition('=')[2])
        mod.exclusive(access.audit / 'LAUNCH_RECEIPT.json', dict(META, remote_root=REMOTE, **result))
        if result['status'] != 'released':
            raise RuntimeError('unique K1 MET submission did not release')
        succeeded = True
        print(json.dumps({'audit': str(access.audit), 'plan_sha256': sha(ROOT / 'EXECUTION_PLAN.json'),
                          **result}, ensure_ascii=False))
    finally:
        owned = access.owns_vpn
        access.close_vpn()
        mod.exclusive(access.audit / 'LOCAL_OPERATION_RECEIPT.json',
                      dict(META, local_started_utc=started,
                           local_completed_utc=datetime.now(timezone.utc).isoformat(),
                           operation_succeeded=succeeded, vpn_started_by_this_operation=owned,
                           vpn_closed=owned, preexisting_vpn_left_unchanged=not owned))


if __name__ == '__main__':
    main()
