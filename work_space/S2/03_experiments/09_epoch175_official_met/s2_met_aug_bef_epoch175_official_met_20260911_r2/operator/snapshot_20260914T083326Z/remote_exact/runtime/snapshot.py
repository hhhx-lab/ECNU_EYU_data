"""One immediate snapshot; never sleeps, polls, submits or mutates evidence."""
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from recovery import ROOT, META, read, sha


def main():
    launch_path = ROOT / 'evidence/CPU_LAUNCH.json'
    job = read(launch_path)['canonical_job_id'] if launch_path.exists() else None
    states, queue, detail = '', '', ''
    if job:
        states = subprocess.check_output(['sacct', '-n', '-P', '-j', job, '--format=JobIDRaw,JobName,State,ExitCode,AllocTRES,NodeList'], text=True)
        queue = subprocess.check_output(['squeue', '-h', '-j', job, '-o', '%i|%j|%T|%R'], text=True)
        command = subprocess.run(['scontrol', 'show', 'job', '-o', job], text=True, capture_output=True)
        detail = command.stdout if command.returncode == 0 else 'completed job not in scontrol cache'
    evidence = {}
    for path in sorted((ROOT / 'evidence').glob('*')):
        value = read(path)
        if 'cases' in value:
            value = {k:v for k,v in value.items() if k != 'cases'}
        evidence[path.name] = value
    logs = {}
    pattern = re.compile(r'Traceback|OutOfMemory|out of memory|oom-kill|CUDA (?:error|failure)|nonfinite prediction|ImportError|RuntimeError|Segmentation fault|Illegal instruction', re.I)
    for path in sorted((ROOT / 'logs').glob('*')):
        lines = path.read_text(errors='replace').splitlines()
        logs[path.name] = dict(size=path.stat().st_size, sha256=sha(path), tail=lines[-12:],
                               errors=[line[:500] for line in lines if pattern.search(line)][:10])
    failed = [name for name in evidence if 'FAILED' in name]
    main_rows = [line.split('|') for line in states.splitlines() if line.split('|')[0] == job]
    state = main_rows[0][2] if main_rows else None
    exit_code = main_rows[0][3] if main_rows else None
    good = state == 'COMPLETED' and exit_code == '0:0' and 'CPU_RECOVERY_COMPLETE.ok' in evidence and evidence.get('CPU_RECOVERY_VALIDATION.json',{}).get('status') == 'pass'
    bad_steps = [line for line in states.splitlines() if len(line.split('|'))>3 and (line.split('|')[2].startswith(('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL')) or line.split('|')[3] not in ('0:0',''))]
    errors = any(log['errors'] for log in logs.values())
    missing = job is not None and not queue.strip() and not main_rows
    status = 'failure' if failed or bad_steps or errors or missing else ('complete' if good else 'pending_or_running')
    if state == 'COMPLETED' and not good:
        status = 'failure'
    usage = shutil.disk_usage('/public')
    result = dict(META, status=status, remote_utc=datetime.now(timezone.utc).isoformat(),
                  hostname=subprocess.check_output(['hostname'], text=True).strip(), job_id=job,
                  sacct=states, squeue=queue, scontrol=detail,
                  disk_public=dict(total=usage.total, used=usage.used, free=usage.free),
                  evidence=evidence, logs=logs, failed_steps=bad_steps,
                  pull_allowed=good and status == 'complete')
    print('CPU_SNAPSHOT_BEGIN')
    print(json.dumps(result, allow_nan=False))
    print('CPU_SNAPSHOT_END')


if __name__ == '__main__':
    main()
