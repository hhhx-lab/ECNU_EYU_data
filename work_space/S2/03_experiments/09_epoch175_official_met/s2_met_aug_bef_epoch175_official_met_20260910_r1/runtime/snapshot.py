"""One bounded scheduler snapshot; optionally transport final lightweight evidence."""
import argparse
import base64
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from workflow import ROOT, META, read, sha, require_preflight, require


def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=45)
    return dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pull', action='store_true')
    args = parser.parse_args()
    p = require_preflight()
    launch = read(ROOT / 'evidence/LAUNCH_SUMMARY.json')
    jobs = ','.join(launch['jobs'].values())
    report = dict(META, remote_utc=datetime.now(timezone.utc).isoformat(), plan_sha256=sha(ROOT / 'EPOCH175_OFFICIAL_MET_EXECUTION_PLAN_20260910_R1.json'),
        jobs=launch['jobs'], hostname=command(['hostname']), disk=command(['df', '-h', '/public']),
        squeue=command(['squeue', '-h', '-j', jobs, '-o', '%i|%j|%T|%M|%D|%R']),
        sacct=command(['sacct', '-X', '-n', '-P', '-j', jobs, '--format=JobIDRaw,JobName,State,ExitCode,Elapsed,NodeList,AllocTRES']),
        evidence={}, logs={}, prediction_counts={})
    errors = re.compile(r'Traceback \(most recent call last\)|OutOfMemoryError|out of memory|oom-kill|CUDA error|CUDA out of memory|ImportError|ModuleNotFoundError|nonfinite|No space left on device|PermissionError|AssertionError|RuntimeError:', re.I)
    for file in sorted((ROOT / 'evidence').glob('*')):
        if file.suffix == '.json':
            report['evidence'][file.name] = read(file)
        elif file.suffix == '.ok':
            report['evidence'][file.name] = file.read_text()
    for file in sorted((ROOT / 'logs').glob('*')):
        if file.is_file():
            text = file.read_text(errors='replace')
            report['logs'][file.name] = dict(size=file.stat().st_size, tail=text.splitlines()[-6:], errors=[line for line in text.splitlines() if errors.search(line)][:12])
    for method in ('B', 'E_SNFH', 'F'):
        report['prediction_counts'][method] = len(list((ROOT / f'cohort/fixed103/models/{method}/prediction').glob('*.nii.gz')))
    require(report['squeue']['returncode'] == report['sacct']['returncode'] == 0, 'scheduler snapshot unavailable')
    states = {r[0]:r for line in report['sacct']['stdout'].splitlines() if (r := line.split('|')) and r[0] in launch['jobs'].values()}
    failure_markers = [name for name in report['evidence'] if 'FAILED' in name]
    log_errors = [name for name, log in report['logs'].items() if log['errors']]
    bad_states = [job for job, row in states.items() if row[2] not in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'COMPLETED') or row[3] != '0:0']
    queued_ids = {line.split('|')[0] for line in report['squeue']['stdout'].splitlines()}
    absent = set(launch['jobs'].values()) - set(states) - queued_ids
    report['health'] = dict(failure_markers=failure_markers, critical_log_files=log_errors, bad_job_states=bad_states, absent_sacct_jobs=sorted(absent))
    all_complete = len(states) == 4 and all(row[2:4] == ['COMPLETED', '0:0'] for row in states.values())
    healthy = not (failure_markers or log_errors or bad_states or absent)
    report['status'] = 'completed' if healthy and all_complete else 'active' if healthy else 'attention_required'
    files = []
    if args.pull:
        require(report['status'] == 'completed', 'not all jobs successfully completed')
        require((ROOT / 'evidence/EPOCH175_OFFICIAL_MET_PIPELINE_COMPLETE.ok').exists(), 'pipeline completion marker missing')
        for name in ('B_EVALUATION_VALIDATION.json', 'E_SNFH_EVALUATION_VALIDATION.json', 'F_EVALUATION_VALIDATION.json', 'PAIRED_COMPARISON_VALIDATION.json'):
            require(report['evidence'][name]['status'] == 'pass', 'validation failed')
        require(set(report['prediction_counts'].values()) == {103}, 'prediction count mismatch')
        total = 0
        for file in sorted(ROOT.rglob('*')):
            rel = file.relative_to(ROOT)
            if not file.is_file() or file.is_symlink() or any(x in rel.parts for x in ('model_snapshots', 'prediction', 'input', 'reference', '__pycache__', 'operator')):
                continue
            if file.suffix.lower() not in ('.json', '.csv', '.md', '.txt', '.ok', '.log', '.err', '.out'):
                continue
            data = file.read_bytes()
            total += len(data)
            require(len(data) < 16 * 1024**2 and total < 100 * 1024**2, 'unexpected large artifact')
            files.append(dict(relative_path=str(rel), sha256=sha(file), size=len(data), content_base64=base64.b64encode(data).decode('ascii')))
    print('ECNU_SNAPSHOT_JSON_BEGIN')
    print(json.dumps(dict(report=report, files=files), allow_nan=False))
    print('ECNU_SNAPSHOT_JSON_END')


if __name__ == '__main__':
    main()
