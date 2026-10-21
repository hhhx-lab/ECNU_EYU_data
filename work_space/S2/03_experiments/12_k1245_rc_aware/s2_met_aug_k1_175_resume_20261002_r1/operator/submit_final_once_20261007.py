"""Submit the frozen resume package once, accepting an expired terminal squeue ID."""

import hashlib
import importlib.util
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCESS = ROOT.parent / 's2_met_aug_k1245_20260923_r3_fallback/operator/access.py'
REMOTE = '/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/' + ROOT.name

REMOTE_CODE = r'''
import sys,json,pathlib,hashlib,datetime
root=pathlib.Path(sys.argv[1])
sys.path.insert(0,str(root/'runtime'))
import control_resume as control
from resume_contract import verify_files,require,write_once
plan=verify_files(root)
original_command=control.command
raw={}
def checked_command(argv):
    result=original_command(argv)
    if argv[0] in ('squeue','sacct'):
        raw[argv[0]]=dict(result)
    if argv[0]=='squeue' and result['rc']!=0:
        require(not result['out'] and result['err']=='slurm_load_jobs error: Invalid job id specified',
                'unexpected squeue error; refuse submission')
        return dict(rc=0,out='',err=result['err'],raw_rc=result['rc'],expired_id=True)
    return result
control.command=checked_command
observed=control.snapshot(plan)
if observed['squeue'].get('expired_id'):
    require(observed['source_state']=='TIMEOUT' and observed['sacct']['rc']==0,
            'expired squeue ID requires authoritative source TIMEOUT')
write_once(root/'evidence/SCHEDULER_TERMINAL_READ_ACCEPTANCE_20261007.json',
           dict(status='pass',snapshot=observed,raw_scheduler_results=raw,
                original_package_unchanged=True,normalization_scope='empty expired ID only with exact sacct TIMEOUT'))
report=control.submit(plan,observed)
if report.get('status')=='submitted_once':
    job=report['canonical_job_id']
    report['submitted_job_squeue']=original_command(['squeue','-h','-j',job,'-o','%i|%j|%T|%M|%l|%N|%R'])
    report['submitted_job_sacct']=original_command(['sacct','-n','-P','-X','-j',job,'-o','JobIDRaw,JobName,State,ExitCode,Elapsed,Timelimit'])
write_once(root/'evidence/SUBMISSION_ACCEPTANCE_20261007.json',report)
print('K1_RESUME_OPERATION_JSON='+json.dumps(report,allow_nan=False),flush=True)
'''


def main():
    spec = importlib.util.spec_from_file_location('k1245_existing_access', ACCESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    access = mod.Access('k1_resume_final_submit_20261007')
    started = datetime.now(timezone.utc).isoformat()
    success = False
    try:
        access.start_vpn()
        command = 'set -e; export RESUME_ROOT=' + shlex.quote(REMOTE)
        command += '; source "$RESUME_ROOT/runtime/research_environment.sh"; '
        command += shlex.quote(mod.PYTHON) + ' -B -c ' + shlex.quote(REMOTE_CODE) + ' ' + shlex.quote(REMOTE)
        raw = access.ssh(command, timeout=480)
        lines = [line for line in raw.splitlines() if line.startswith('K1_RESUME_OPERATION_JSON=')]
        if len(lines) != 1:
            raise RuntimeError('structured submission response missing; inspect audit, do not resubmit')
        report = json.loads(lines[0].split('=', 1)[1])
        report['local_utc'] = datetime.now(timezone.utc).isoformat()
        mod.exclusive(access.audit / 'OPERATION_RESULT.json', report)
        collect = r'''
import json,pathlib,hashlib
root=pathlib.Path(__import__('sys').argv[1])
paths=list((root/'evidence').glob('*.json'))+list((root/'evidence').glob('*.ok'))
print('EVIDENCE_JSON='+json.dumps({str(p.relative_to(root)):{'text':p.read_text(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}))
'''
        pulled = access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + shlex.quote(mod.PYTHON) +
                            ' -B -c ' + shlex.quote(collect) + ' ' + shlex.quote(REMOTE), timeout=60)
        files = json.loads(next(line for line in pulled.splitlines() if line.startswith('EVIDENCE_JSON=')).split('=', 1)[1])
        for relative, entry in files.items():
            payload = entry['text'].encode()
            if hashlib.sha256(payload).hexdigest() != entry['sha256']:
                raise RuntimeError('pulled evidence SHA mismatch')
            if relative.endswith('.json'):
                json.loads(entry['text'])
            path = access.audit / 'pulled' / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as handle:
                handle.write(payload)
        mod.exclusive(access.audit / 'PULLED_SHA256.json', dict(mod.META,
                      files={relative: value['sha256'] for relative, value in files.items()}))
        success = True
        print(json.dumps(dict(audit=str(access.audit), result=report, pulled_files=len(files)), ensure_ascii=False))
    finally:
        owned = access.owns_vpn
        access.close_vpn()
        mod.exclusive(access.audit / 'LOCAL_OPERATION_RECEIPT.json', dict(mod.META,
                      local_started_utc=started, local_completed_utc=datetime.now(timezone.utc).isoformat(),
                      operation_succeeded=success, vpn_started_by_this_operation=owned,
                      vpn_closed=owned, preexisting_vpn_left_unchanged=not owned))


if __name__ == '__main__':
    main()
