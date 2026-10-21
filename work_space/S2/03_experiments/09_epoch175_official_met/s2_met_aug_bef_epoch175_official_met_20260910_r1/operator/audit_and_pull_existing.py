"""Retrieve completed prediction/evaluator evidence without repairing or resubmitting r1."""
import base64
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VPN = '/Users/hwaigc/比赛+课题/服务器登陆/ecnu-vpn-split.sh'
env = dict(os.environ)
assert env.get('ECNU_SERVER_ENV_FILE') == '/Users/hwaigc/比赛+课题/服务器登陆/服务器登陆的env'
assert env.get('HPC_SSH_PASSWORD')
for key, default in (('HPC_LOGIN_HOST','59.78.189.132'),('HPC_LOGIN_PORT','2323'),('HPC_SSH_USER','zqchen')):
    env.setdefault(key, default)
started = datetime.now(timezone.utc).isoformat()
dest = ROOT / 'operator' / ('outputs_audit_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
dest.mkdir(exist_ok=False)
code = base64.b64encode((ROOT / 'operator/verify_existing_outputs.py').read_bytes()).decode('ascii')
remote = 'import base64;exec(compile(base64.b64decode(' + repr(code) + '),"<read-only-output-audit>","exec"))'
env['REMOTE_COMMAND'] = 'env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /public/home/zqchen/.conda/envs/segmamba/bin/python -c ' + shlex.quote(remote)
script = r'''
log_user 1
set timeout 300
set auth 0
spawn -noecho ssh -p $env(HPC_LOGIN_PORT) -o StrictHostKeyChecking=yes -o ConnectTimeout=15 "$env(HPC_SSH_USER)@$env(HPC_LOGIN_HOST)" $env(REMOTE_COMMAND)
expect {
  -re "(P|p)assword:" { incr auth; if {$auth>1} {close; exit 125}; log_user 0; send -- "$env(HPC_SSH_PASSWORD)\r"; log_user 1; exp_continue }
  eof {exit [lindex [wait] 3]}
  timeout {close; exit 124}
}
'''
try:
    start = subprocess.run([VPN,'start'], env=env, capture_output=True, timeout=100)
    if start.returncode: raise RuntimeError('VPN startup failed; private output suppressed')
    response = subprocess.run(['expect'], input=script, text=True, env=env, capture_output=True, timeout=330)
finally:
    stop = subprocess.run([VPN,'stop'], env=env, capture_output=True, timeout=40)
    if stop.returncode: raise RuntimeError('VPN cleanup unconfirmed')
if response.returncode: raise RuntimeError('Bounded read-only audit failed: ' + str(response.returncode))
payload = json.loads(response.stdout.split('OUTPUT_AUDIT_BEGIN',1)[1].split('OUTPUT_AUDIT_END',1)[0].strip())
with (dest/'READ_ONLY_OUTPUT_VERIFICATION.json').open('x') as f:
    json.dump(payload['report'], f, indent=2, sort_keys=True, allow_nan=False)
if payload['report']['status'] != 'verified_existing_outputs_only':
    print(json.dumps(payload['report']))
    raise SystemExit(2)
inventory = []
for item in payload['files']:
    rel = Path(item['relative_path'])
    assert not rel.is_absolute() and '..' not in rel.parts
    target = ROOT / rel
    data = base64.b64decode(item['content_base64'], validate=True)
    assert len(data) == item['size'] and hashlib.sha256(data).hexdigest() == item['sha256']
    if target.exists():
        assert hashlib.sha256(target.read_bytes()).hexdigest() == item['sha256'], str(rel)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f: f.write(data)
    if target.suffix == '.json': json.loads(target.read_text())
    inventory.append({key:value for key,value in item.items() if key!='content_base64'})
receipt = dict(artifact_status='experimental_unvalidated', operator_approved=False, formal_gate_status='not_run_not_passed',
    status='retrieved_existing_outputs_only', original_jobs_remain_failed=True, new_jobs_submitted=False,
    local_started_utc=started, local_completed_utc=datetime.now(timezone.utc).isoformat(),
    remote_completed_utc=payload['report']['remote_completed_utc'], vpn_closed=True, files=inventory)
with (dest/'RETRIEVAL_RECEIPT.json').open('x') as f: json.dump(receipt,f,indent=2,sort_keys=True)
print(json.dumps({key:value for key,value in receipt.items() if key!='files'}))
print(json.dumps(dict(evidence_directory=str(dest), file_count=len(inventory), routes=payload['report']['routes'])))
