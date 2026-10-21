"""Read only the new attempt after an ambiguous connection timeout."""
import base64
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/' + ROOT.name
VPN = '/Users/hwaigc/比赛+课题/服务器登陆/ecnu-vpn-split.sh'
env = dict(os.environ)
assert env.get('ECNU_SERVER_ENV_FILE') == '/Users/hwaigc/比赛+课题/服务器登陆/服务器登陆的env'
assert env.get('HPC_SSH_PASSWORD')
for key, default in (('HPC_LOGIN_HOST', '59.78.189.132'), ('HPC_LOGIN_PORT', '2323'), ('HPC_SSH_USER', 'zqchen')):
    env.setdefault(key, default)
code = '''import os,json,subprocess,socket,csv,hashlib,signal,math
from pathlib import Path
from datetime import datetime,timezone
root=Path(REMOTE_ROOT)
evidence={p.name:json.loads(p.read_text()) for p in (root/'evidence').glob('*.json')}
processes=[]
for proc in Path('/proc').glob('[0-9]*'):
    try:
        if proc.stat().st_uid!=os.getuid() or (proc/'cwd').resolve()!=root: continue
        command=(proc/'cmdline').read_bytes().split(b'\\0')
        if any(b'workflow.py' in arg for arg in command):
            processes.append({'pid':int(proc.name),'executable':command[0].decode(errors='replace'),'phase':'preflight' if any(b'preflight' in arg for arg in command) else 'other_workflow','argument_count':len(command)})
    except (OSError,PermissionError): pass
result={'remote_utc':datetime.now(timezone.utc).isoformat(),'hostname':socket.gethostname(),'evidence':evidence,'processes':processes,'input_manifest_exists':(root/'manifests/INPUT_REFERENCE_SHA256.json').exists(),'model_snapshots':[str(p.relative_to(root)) for p in (root/'model_snapshots').glob('*/fold_0/checkpoint_final.pth')]}
jobs=evidence.get('LAUNCH_SUMMARY.json',{}).get('jobs',{})
if jobs:
    result['sacct']=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(jobs.values()),'--format=JobIDRaw,JobName,State,ExitCode'],text=True)
    result['squeue']=subprocess.check_output(['squeue','-h','-j',','.join(jobs.values()),'-o','%i|%j|%T|%R'],text=True)
terminated=[]
for proc in Path('/proc').glob('[0-9]*'):
    try:
        if proc.stat().st_uid!=os.getuid(): continue
        args=(proc/'cmdline').read_bytes().split(b'\\0')
        if int(proc.name)!=os.getpid() and len(args)>2 and args[0]==b'/public/home/zqchen/.conda/envs/segmamba/bin/python' and args[2]==EXPECTED_AUDIT_COMMAND.encode():
            os.kill(int(proc.name),signal.SIGTERM)
            terminated.append(int(proc.name))
    except (OSError,PermissionError): pass
result['terminated_timed_out_readonly_audits']=terminated
plan=json.loads((root/'EPOCH175_OFFICIAL_MET_EXECUTION_PLAN_20260910_R1.json').read_text())
result['metrics']={}
for method in ('B','E_SNFH','F'):
    out=root/'cohort/fixed103/models'/method
    result['metrics'][method]={}
    for epoch,path in ((125,Path(plan['routes'][method]['baseline_metrics_csv'])),(175,out/'evaluation/leaderboard_metrics.csv')):
        with path.open() as f:
            reader=csv.DictReader(f); rows=list(reader); fields=reader.fieldnames
        assert len(rows)==106 and len(fields)==73
        means={}; counts={}
        for k in fields[1:]:
            v=float(rows[-3][k]) if rows[-3][k] else float('nan')
            means[k]=v if math.isfinite(v) else None
            counts[k]=sum(bool(row[k]) and math.isfinite(float(row[k])) for row in rows[:103])
        result['metrics'][method][str(epoch)]={'means':means,'finite_counts':counts,'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'case_count':103}
    inf=json.loads((out/'inference_audit.json').read_text())
    ev=json.loads((out/'evaluation/EVALUATION_VALIDATION.json').read_text())
    assert ev['status']==inf['status']=='pass' and ev['case_count']==inf['prediction_count']==103
    assert ev['metrics_csv_sha256']==result['metrics'][method]['175']['csv_sha256']
    assert inf['checkpoint_sha256']==plan['routes'][method]['checkpoint_sha256']
    result['metrics'][method]['official_evaluation_complete']=(out/'evaluation/EVALUATION_COMPLETE.ok').exists()
    result['metrics'][method]['inference_audit']=inf
    result['metrics'][method]['evaluation_validation']=ev
    result['metrics'][method]['prediction_count']=len(list((out/'prediction').glob('*.nii.gz')))
result['interpretation']='Existing evaluator outputs only; original jobs remain FAILED; corrected full-volume acceptance and paired bootstrap are not completed.'
print('INSPECTION_BEGIN')
print(json.dumps(result))
print('INSPECTION_END')
'''.replace('REMOTE_ROOT', repr(REMOTE))
audit_blob=base64.b64encode((ROOT/'operator/verify_existing_outputs.py').read_bytes()).decode('ascii')
audit_command='import base64;exec(compile(base64.b64decode(' + repr(audit_blob) + '),"<read-only-output-audit>","exec"))'
code=code.replace('EXPECTED_AUDIT_COMMAND',repr(audit_command))
env['REMOTE_COMMAND'] = 'env PYTHONDONTWRITEBYTECODE=1 /public/home/zqchen/.conda/envs/segmamba/bin/python -c ' + shlex.quote(code)
script = r'''
log_user 1
set timeout 60
set auth 0
spawn -noecho ssh -p $env(HPC_LOGIN_PORT) -o StrictHostKeyChecking=yes -o ConnectTimeout=15 "$env(HPC_SSH_USER)@$env(HPC_LOGIN_HOST)" $env(REMOTE_COMMAND)
expect {
  -re "(P|p)assword:" { incr auth; if {$auth>1} {close; exit 125}; log_user 0; send -- "$env(HPC_SSH_PASSWORD)\r"; log_user 1; exp_continue }
  eof {exit [lindex [wait] 3]}
  timeout {close; exit 124}
}
'''
try:
    started = subprocess.run([VPN,'start'], env=env, capture_output=True, timeout=100)
    if started.returncode: raise RuntimeError('VPN failed; private output suppressed')
    response = subprocess.run(['expect'], input=script, text=True, capture_output=True, env=env, timeout=90)
finally:
    stopped = subprocess.run([VPN,'stop'], env=env, capture_output=True, timeout=40)
    if stopped.returncode: raise RuntimeError('VPN cleanup unconfirmed')
if response.returncode:
    raise RuntimeError('Inspection connection failed: ' + str(response.returncode))
data = json.loads(response.stdout.split('INSPECTION_BEGIN',1)[1].split('INSPECTION_END',1)[0].strip())
data.update(local_utc=datetime.now(timezone.utc).isoformat(), artifact_status='experimental_unvalidated', operator_approved=False, formal_gate_status='not_run_not_passed', vpn_closed=True)
path=ROOT/'operator'/('EXISTING_METRICS_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
with path.open('x') as f: json.dump(data,f,indent=2)
print(json.dumps({'path':str(path),'remote_utc':data['remote_utc'],'terminated_timed_out_readonly_audits':data['terminated_timed_out_readonly_audits'],'methods':{m:{'prediction_count':v['prediction_count'],'official_evaluation_complete':v['official_evaluation_complete'],'dice175':{k:n for k,n in v['175']['means'].items() if k.startswith('lesionwise_dsc_mean_')},'dice125':{k:n for k,n in v['125']['means'].items() if k.startswith('lesionwise_dsc_mean_')}} for m,v in data['metrics'].items()}}))
