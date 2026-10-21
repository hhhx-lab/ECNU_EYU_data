"""Bounded VPN/SSH operations; credentials are inherited only from the authorized env."""
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
VPN = '/Users/hwaigc/比赛+课题/服务器登陆/ecnu-vpn-split.sh'
ENVFILE = '/Users/hwaigc/比赛+课题/服务器登陆/服务器登陆的env'
PLAN_NAME = 'EPOCH175_OFFICIAL_MET_EXECUTION_PLAN_20260910_R1.json'
META = dict(artifact_status='experimental_unvalidated', operator_approved=False, formal_gate_status='not_run_not_passed')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('deploy', 'snapshot', 'pull'))
    args = parser.parse_args()
    env = dict(os.environ)
    if env.get('ECNU_SERVER_ENV_FILE') != ENVFILE or not env.get('HPC_SSH_PASSWORD'):
        raise RuntimeError('Source the authorized env and set ECNU_SERVER_ENV_FILE first')
    for key, default in (('HPC_LOGIN_HOST','59.78.189.132'),('HPC_LOGIN_PORT','2323'),('HPC_SSH_USER','zqchen')):
        env.setdefault(key, default)
    env['LOCAL_BUNDLE'], env['REMOTE_ROOT'], env['PLAN_NAME'] = str(ROOT), REMOTE, PLAN_NAME
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    started = datetime.now(timezone.utc).isoformat()
    work = ROOT / 'operator' / (args.action + '_' + stamp)
    work.mkdir(exist_ok=False)
    def expect(kind, command=None):
        if command:
            env['REMOTE_COMMAND'] = command
        spawn = 'spawn -noecho ssh -p $env(HPC_LOGIN_PORT) -o StrictHostKeyChecking=yes -o ConnectTimeout=20 "$env(HPC_SSH_USER)@$env(HPC_LOGIN_HOST)" $env(REMOTE_COMMAND)' if kind == 'ssh' else 'spawn -noecho scp -P $env(HPC_LOGIN_PORT) -o StrictHostKeyChecking=yes -o ConnectTimeout=20 -r "$env(LOCAL_BUNDLE)/runtime" "$env(LOCAL_BUNDLE)/slurm" "$env(LOCAL_BUNDLE)/manifests" "$env(LOCAL_BUNDLE)/$env(PLAN_NAME)" "$env(LOCAL_BUNDLE)/STATIC_SHA256SUMS.txt" "$env(LOCAL_BUNDLE)/README.md" "$env(HPC_SSH_USER)@$env(HPC_LOGIN_HOST):$env(REMOTE_ROOT)/"'
        script = 'log_user 0\nset timeout 600\nset auth 0\n' + spawn + r'''
expect {
  -re "(P|p)assword:" { incr auth; if {$auth > 1} {close; exit 125}; send -- "$env(HPC_SSH_PASSWORD)\r"; log_user 1; exp_continue }
  eof { exit [lindex [wait] 3] }
  timeout { close; exit 124 }
}
'''
        response = subprocess.run(['expect'], input=script, text=True, env=env, capture_output=True, timeout=630)
        if response.returncode:
            with (work / ('FAILED_' + kind + '.log')).open('x') as f:
                f.write(response.stdout + response.stderr)
            raise RuntimeError('Bounded ' + kind + ' failed with exit ' + str(response.returncode) + '; see local audit, do not retry deployment')
        return response.stdout
    try:
        start = subprocess.run([VPN, 'start'], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=100)
        if start.returncode:
            raise RuntimeError('VPN startup failed; private output suppressed')
        if args.action == 'deploy':
            create = 'from pathlib import Path; p=Path(' + repr(REMOTE) + '); p.mkdir(exist_ok=False); (p/"logs").mkdir(); (p/"evidence").mkdir(); print("FRESH_ROOT_CREATED")'
            expect('ssh', 'env PYTHONDONTWRITEBYTECODE=1 ' + PYTHON + ' -c ' + shlex.quote(create))
            expect('scp')
            code = 'import os,subprocess; os.chdir(' + repr(REMOTE) + '); p=' + repr(PYTHON) + '; s=' + repr(sha(ROOT / PLAN_NAME)) + '; subprocess.run([p,"runtime/workflow.py","preflight","--plan-sha",s],check=True); subprocess.run([p,"runtime/workflow.py","launch","--plan-sha",s],check=True)'
            response = expect('ssh', 'env PYTHONDONTWRITEBYTECODE=1 nnUNet_compile=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ' + PYTHON + ' -c ' + shlex.quote(code))
            with (work / 'PREFLIGHT_AND_LAUNCH.log').open('x') as f:
                f.write(response)
        response = expect('ssh', 'env PYTHONDONTWRITEBYTECODE=1 ' + PYTHON + ' ' + shlex.quote(REMOTE + '/runtime/snapshot.py') + (' --pull' if args.action == 'pull' else ''))
    finally:
        stop = subprocess.run([VPN, 'stop'], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=40)
        if stop.returncode:
            raise RuntimeError('VPN cleanup did not confirm success')
    payload = json.loads(response.split('ECNU_SNAPSHOT_JSON_BEGIN', 1)[1].split('ECNU_SNAPSHOT_JSON_END', 1)[0].strip())
    report = dict(payload['report'], local_started_utc=started, local_completed_utc=datetime.now(timezone.utc).isoformat(), vpn_closed=True)
    write_json(work / 'SNAPSHOT.json', report)
    for name, value in report['evidence'].items():
        if not name.endswith('.json'):
            continue
        target = ROOT / 'evidence' / name
        if not target.exists():
            write_json(target, value)
        elif json.loads(target.read_text()) != value:
            raise RuntimeError('Local evidence differs: ' + name)
    inventory = []
    for item in payload['files']:
        rel = Path(item['relative_path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise RuntimeError('Unsafe retrieval path')
        data = base64.b64decode(item['content_base64'], validate=True)
        if len(data) != item['size'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise RuntimeError('Transport integrity failure')
        target = ROOT / rel
        if target.exists() and sha(target) != item['sha256']:
            # Snapshot-transcribed JSON remains intact; archive exact remote bytes separately.
            if target.suffix != '.json' or json.loads(target.read_text()) != json.loads(data):
                raise RuntimeError('Refusing overwrite of differing local artifact: ' + str(rel))
            target = work / 'remote_exact_bytes' / rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as f:
                f.write(data)
        if target.suffix == '.json':
            json.loads(target.read_text())
        inventory.append(dict(relative_path=str(rel), local_path=str(target), size=item['size'], sha256=item['sha256']))
    write_json(work / 'LOCAL_OPERATION_RECEIPT.json', dict(META, status='pass', action=args.action, local_started_utc=started,
        local_completed_utc=datetime.now(timezone.utc).isoformat(), remote_utc=report['remote_utc'], plan_sha256=sha(ROOT / PLAN_NAME),
        snapshot_status=report['status'], files=inventory, vpn_closed=True))
    print(json.dumps(dict(status=report['status'], jobs=report['jobs'], prediction_counts=report['prediction_counts'], health=report['health'],
        evidence_directory=str(work), remote_utc=report['remote_utc'], local_completed_utc=report['local_completed_utc'], vpn_closed=True), ensure_ascii=False))


if __name__ == '__main__':
    main()
