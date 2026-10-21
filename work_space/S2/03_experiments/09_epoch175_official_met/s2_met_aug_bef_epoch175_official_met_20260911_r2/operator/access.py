"""Bounded VPN operations and one scheduler snapshot; credentials never persisted."""
import argparse
import base64
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/' + ROOT.name
PYTHON = '/public/home/zqchen/.conda/envs/segmamba/bin/python'
PLAN = 'CPU_RECOVERY_EXECUTION_PLAN_20260911_R2.json'
ENVFILE = '/Users/hwaigc/比赛+课题/服务器登陆/服务器登陆的env'
VPN = '/Users/hwaigc/比赛+课题/服务器登陆/ecnu-vpn-split.sh'
META = dict(artifact_status='experimental_unvalidated', operator_approved=False, formal_gate_status='not_run_not_passed')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('deploy', 'snapshot'))
    args = parser.parse_args()
    env = dict(os.environ)
    assert env.get('ECNU_SERVER_ENV_FILE') == ENVFILE and env.get('HPC_SSH_PASSWORD'), 'authorized env required'
    for k, default in (('HPC_LOGIN_HOST','59.78.189.132'),('HPC_LOGIN_PORT','2323'),('HPC_SSH_USER','zqchen')):
        env.setdefault(k,default)
    started = datetime.now(timezone.utc).isoformat()
    work = ROOT / 'operator' / (args.action + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    work.mkdir(exist_ok=False)
    def call(kind, command=None, target=None):
        if kind == 'ssh':
            env['REMOTE_COMMAND'] = command
            spawn = 'spawn -noecho ssh -p $env(HPC_LOGIN_PORT) -o StrictHostKeyChecking=yes -o ConnectTimeout=15 "$env(HPC_SSH_USER)@$env(HPC_LOGIN_HOST)" $env(REMOTE_COMMAND)'
        elif kind == 'upload':
            env['LOCAL_ROOT'],env['REMOTE_ROOT'],env['PLAN_NAME'] = str(ROOT),REMOTE,PLAN
            spawn = 'spawn -noecho scp -P $env(HPC_LOGIN_PORT) -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -r "$env(LOCAL_ROOT)/runtime" "$env(LOCAL_ROOT)/slurm" "$env(LOCAL_ROOT)/$env(PLAN_NAME)" "$env(LOCAL_ROOT)/STATIC_SHA256SUMS.txt" "$env(LOCAL_ROOT)/README.md" "$env(HPC_SSH_USER)@$env(HPC_LOGIN_HOST):$env(REMOTE_ROOT)/"'
        else:
            env['REMOTE_ROOT'],env['LOCAL_TARGET'] = REMOTE,str(target)
            spawn = 'spawn -noecho scp -P $env(HPC_LOGIN_PORT) -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -r "$env(HPC_SSH_USER)@$env(HPC_LOGIN_HOST):$env(REMOTE_ROOT)" $env(LOCAL_TARGET)'
        timeout = 90 if kind == 'ssh' else 180
        script = 'log_user 1\nset auth 0\nset timeout ' + str(timeout) + '\n' + spawn + r'''
expect {
  -re "(P|p)assword:" { incr auth; if {$auth>1} {close; exit 125}; log_user 0; send -- "$env(HPC_SSH_PASSWORD)\r"; log_user 1; exp_continue }
  eof {exit [lindex [wait] 3]}
  timeout {close; exit 124}
}
'''
        reply = subprocess.run(['expect'], input=script, text=True, capture_output=True, env=env, timeout=timeout+20)
        if reply.returncode:
            clean = (reply.stdout + reply.stderr).replace(env['HPC_SSH_PASSWORD'],'[REDACTED]')
            write(work / ('FAILED_' + kind + '.json'), dict(META, exit_code=reply.returncode, diagnostic_tail=clean[-5000:], no_automatic_retry=True))
            raise RuntimeError('Bounded operation failed: ' + kind + ' exit ' + str(reply.returncode) + '; inspect existing audit, never repeat deployment')
        return reply.stdout
    def remote(code):
        return call('ssh', 'env PYTHONDONTWRITEBYTECODE=1 ' + PYTHON + ' -c ' + shlex.quote(code))
    snapshot = None
    owns_vpn = True
    try:
        result = subprocess.run([VPN,'start'], env=env, capture_output=True, timeout=100)
        owns_vpn = b'OpenConnect is already running' not in result.stdout
        if result.returncode:
            raise RuntimeError('VPN startup failed; private output suppressed')
        if args.action == 'deploy':
            remote('from pathlib import Path; p=Path(' + repr(REMOTE) + '); p.mkdir(exist_ok=False); (p/"evidence").mkdir(); (p/"logs").mkdir(); print("FRESH_ROOT_CREATED")')
            call('upload')
            response = remote('import subprocess; r=' + repr(REMOTE) + '; p=' + repr(PYTHON) + '; s=' + repr(sha(ROOT / PLAN)) + '; subprocess.run([p,r+"/runtime/recovery.py","preflight","--plan-sha",s],check=True); subprocess.run([p,r+"/runtime/recovery.py","launch","--plan-sha",s],check=True)')
            write(work / 'DEPLOY_RECEIPT.json', dict(META, output=response.replace(env['HPC_SSH_PASSWORD'],'[REDACTED]')))
        snapshot_code = base64.b64encode((ROOT / 'runtime/snapshot.py').read_bytes()).decode('ascii')
        response = call('ssh', 'cd ' + shlex.quote(REMOTE) + ' && env PYTHONDONTWRITEBYTECODE=1 ' + PYTHON + ' -c ' + shlex.quote("import base64,sys;sys.path.insert(0, 'runtime');exec(compile(base64.b64decode(" + repr(snapshot_code) + "), '<read-only-fixed-snapshot>', 'exec'))"))
        snapshot = json.loads(response.split('CPU_SNAPSHOT_BEGIN',1)[1].split('CPU_SNAPSHOT_END',1)[0].strip())
        if snapshot['pull_allowed']:
            call('download', target=work / 'remote_exact')
    finally:
        if owns_vpn:
            result = subprocess.run([VPN,'stop'], env=env, capture_output=True, timeout=40)
            if result.returncode:
                raise RuntimeError('VPN cleanup unconfirmed')
    snapshot.update(local_started_utc=started, local_completed_utc=datetime.now(timezone.utc).isoformat(), vpn_closed=owns_vpn,
                    preexisting_vpn_left_unchanged=not owns_vpn)
    write(work / 'SNAPSHOT.json', snapshot)
    verified = []
    if snapshot['pull_allowed']:
        target = work / 'remote_exact'
        inv = json.loads((target / 'LIGHTWEIGHT_SHA256.json').read_text())
        for relative, item in inv['files'].items():
            rel = Path(relative)
            assert not rel.is_absolute() and '..' not in rel.parts
            path = target / rel
            assert path.stat().st_size == item['size'] and sha(path) == item['sha256'], relative
            if path.suffix == '.json' or path.suffix == '.ok':
                # The unchanged comparator's .ok markers are plain text, not JSON.
                if path.suffix == '.json':
                    json.loads(path.read_text())
            verified.append(relative)
        for name, item in snapshot['logs'].items():
            assert sha(target / 'logs' / name) == item['sha256'], 'log changed after completed snapshot'
        assert json.loads((target / 'results/REPORT_VALIDATION.json').read_text())['paired_records'] == 432
        with (work / 'FINAL_ACCEPTANCE.md').open('x') as f:
            f.write('# CPU Recovery Final Acceptance\n\nartifact_status=experimental_unvalidated\noperator_approved=false\nformal_gate_status=not_run_not_passed\n\n'
                    + 'Local UTC: ' + snapshot['local_completed_utc'] + '\n\nRemote UTC: ' + snapshot['remote_utc']
                    + '\n\nScheduler COMPLETED0:0; independent CPU recovery passed. Original GPU failures are unchanged. '
                    + str(len(verified)) + ' immutable lightweight files verified by SHA256; log SHA verified separately. '
                    + 'Same103 development cases: exploratory only. See remote_exact/results/COMPARISON_REPORT.md for both125/175 values and paired statistics.\n')
    write(work / 'LOCAL_OPERATION_RECEIPT.json', dict(META, status=snapshot['status'], job_id=snapshot['job_id'],
          local_utc=snapshot['local_completed_utc'], remote_utc=snapshot['remote_utc'], plan_sha256=sha(ROOT / PLAN),
          vpn_closed=owns_vpn, verified_files=verified))
    print(json.dumps(dict(status=snapshot['status'], job_id=snapshot['job_id'], sacct=snapshot['sacct'], squeue=snapshot['squeue'],
                         local_utc=snapshot['local_completed_utc'], remote_utc=snapshot['remote_utc'],
                         evidence_path=str(work), vpn_closed=owns_vpn, verified_files=len(verified)), ensure_ascii=False))


if __name__ == '__main__':
    main()
