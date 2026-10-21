"""One-shot staging or TIMEOUT-only final acceptance and held submission."""

import argparse
import datetime as dt
import importlib
import inspect
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from resume_contract import (FOLD_REL, checkpoint, equal, load_json, require,
                             restore_test, sha, verify_files, write_once)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence'


def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    return dict(rc=result.returncode, out=result.stdout.strip(), err=result.stderr.strip())


def snapshot(plan):
    queue = command(['squeue', '-h', '-j', plan['source_job_id'], '-o', '%i|%j|%T|%M|%l|%N|%R'])
    accounting = command(['sacct', '-n', '-P', '-X', '-j', plan['source_job_id'],
                          '-o', 'JobIDRaw,JobName,State,ExitCode,Elapsed,Timelimit,End'])
    require(queue['rc'] == 0 and accounting['rc'] == 0, 'scheduler snapshot command failed')
    rows = [line.split('|') for line in accounting['out'].splitlines() if line.strip()]
    rows = [row for row in rows if row[0] == plan['source_job_id']]
    require(len(rows) == 1 and rows[0][1] == plan['source_job_name'], 'source job missing or wrong identity')
    state = rows[0][2].split()[0].rstrip('+')
    source = Path(plan['source_root'])
    logs = {}
    errors = []
    for path in sorted((source / 'logs').glob('*' + plan['source_job_id'] + '*')):
        if path.is_file():
            text = path.read_text(errors='replace')
            logs[str(path.relative_to(source))] = text[-5000:]
            errors += [line[-500:] for line in text.splitlines() if re.search(
                r'Traceback|CUDA error|out.of.memory|ImportError|nonfinite|FloatingPointError', line, re.I)]
    training_logs = sorted((source / FOLD_REL).glob('training_log_*.txt'), key=lambda p: p.stat().st_mtime_ns)
    tail = training_logs[-1].read_text(errors='replace')[-6000:] if training_logs else ''
    epochs = re.findall(r'(?m)^.*Epoch\s+(\d+)\s*$', tail)
    return dict(remote_utc=dt.datetime.now(dt.timezone.utc).isoformat(), hostname=os.uname().nodename,
                disk=command(['df', '-h', '/public']), squeue=queue, sacct=accounting,
                source_state=state, exit_code=rows[0][3], logs=logs, critical_errors=errors,
                latest_log_epoch=int(epochs[-1]) if epochs else None, training_log_tail=tail,
                source_failed_marker=(source / 'evidence/K1_175_TRAINING_FAILED.json').exists(),
                source_complete_marker=(source / 'evidence/K1_175_TRAINING_COMPLETE.ok').exists())


def dependencies(plan):
    from custom_nnunet.met_aug_k1245 import K1245Config, K1245Manifest
    source = Path(plan['source_root'])
    manifest = K1245Manifest.load(source / 'pools/rc27/component_manifest.json')
    config = K1245Config.load(source / 'config/K1245_ROUTE_K1.json', manifest=manifest,
                             plan_path=source / 'K1245_K1_175_EXECUTION_PLAN_20260929_R3.json')
    require(config.route_id == 'K1', 'wrong augmentation route')
    paths = [source / 'pools/rc27' / load_json(source / 'pools/rc27/component_manifest.json')['records_file'],
             manifest.target_groups_path]
    for key in ('S2_MET_AUG_VALID_MASK_MANIFEST', 'S2_MET_AUG_FIX_V3_CALIBRATION',
                'S2_MET_AUG_ORIGINAL_E_CHECKPOINT', 'S2_MET_AUG_G1_CHECKPOINT_SELECTION',
                'S2_MET_AUG_G2_QC_GATE', 'S2_K1245_LEGACY_COMPONENT_MANIFEST'):
        paths.append(Path(os.environ[key]))
    require(sha(os.environ['S2_MET_AUG_ORIGINAL_E_CHECKPOINT']) ==
            '4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267', 'original E drift')
    require(sha(os.environ['S2_MET_AUG_FIX_V3_CALIBRATION']) ==
            '68a5d8a6c501804b7a47d626e43a58ff71b781154c0cc955a9c5a5d5b786d37f', 'calibration drift')
    directory = Path(os.environ['nnUNet_preprocessed']) / 'Dataset264_BraTS2026_MET_Completion'
    paths += [directory / 'nnUNetPlans.json', directory / 'dataset.json']
    import nnunetv2.training.logging.nnunet_logger as logger
    import nnunetv2.training.lr_scheduler.polylr as scheduler
    paths += [Path(inspect.getsourcefile(logger)), Path(inspect.getsourcefile(scheduler))]
    splits = []
    for prefix, expected in (('train', 1035), ('val', 103)):
        path = Path(os.environ['BRATS_SPLIT_DIR']) / (prefix + '_fixed.txt')
        if not path.is_file():
            path = path.with_name(prefix + '_full.txt')
        values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        require(len(values) == len(set(values)) == expected, prefix + ' split count differs')
        paths.append(path)
        splits.append(values)
    require(not set(splits[0]) & set(splits[1]), 'train/val overlap')
    return dict(dependency_sha256={str(path): sha(path) for path in paths},
                manifest_identity_sha256=manifest.identity_sha256, train_cases=1035, validation_cases=103)


def cpu_test(path, prefix='cpu_restore_test'):
    test_root = ROOT / prefix / 'results/K1'
    require(not test_root.exists(), 'CPU restore test has already run')
    env = dict(os.environ, nnUNet_results=str(test_root))
    import sys
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'runtime/test_restore.py'),
                             str(path), str(ROOT / prefix / FOLD_REL)],
                            env=env, capture_output=True, text=True, timeout=180)
    require(result.returncode == 0, 'CPU restore failed: ' + (result.stdout + result.stderr)[-3000:])
    lines = [line for line in result.stdout.splitlines() if line.startswith('CPU_RESTORE_JSON=')]
    require(len(lines) == 1, 'CPU restore structured result absent')
    return json.loads(lines[0].split('=', 1)[1])


def stage(plan, observed):
    require(not (EVIDENCE / 'REMOTE_STAGE_VALIDATION.json').exists(), 'stage already recorded')
    require(observed['source_state'] == 'RUNNING' and not observed['critical_errors'] and
            not observed['source_failed_marker'], 'source is not a clean live run')
    deps = dependencies(plan)
    report = cpu_test(Path(plan['source_root']) / FOLD_REL / 'checkpoint_latest.pth')
    require(report['current_epoch'] < 175, 'source has no remaining training')
    write_once(EVIDENCE / 'REMOTE_STAGE_VALIDATION.json', dict(status='pass',
               snapshot=observed, cpu_restore=report, **deps, submission_performed=False,
               final_checkpoint_frozen=False, waiting_for_source_timeout=True))
    return dict(status='staged_waiting_for_TIMEOUT', cpu_restore=report,
                source_state=observed['source_state'], latest_log_epoch=observed['latest_log_epoch'],
                submission_performed=False, remote_utc=observed['remote_utc'])


def submit(plan, observed):
    if observed['source_state'] in ('RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING'):
        return dict(status='waiting_no_submission', snapshot=observed)
    if observed['source_state'] == 'COMPLETED' and observed['exit_code'] == '0:0':
        return dict(status='source_completed_do_not_resume', snapshot=observed)
    require(observed['source_state'] == 'TIMEOUT' and not observed['squeue']['out'],
            'source not an inactive TIMEOUT job; stop')
    require(not observed['source_complete_marker'] and not observed['source_failed_marker'] and
            not observed['critical_errors'], 'source has unexpected failure/completion evidence')
    require((EVIDENCE / 'REMOTE_STAGE_VALIDATION.json').is_file(), 'staging preflight absent')
    staged = load_json(EVIDENCE / 'REMOTE_STAGE_VALIDATION.json')
    current_deps = dependencies(plan)
    require(current_deps['dependency_sha256'] == staged['dependency_sha256'], 'dependency SHA drift')
    require(not any((EVIDENCE / name).exists() for name in
                    ('LAUNCH_RESERVATION.json', 'LAUNCH_AUDIT.json', 'SUBMISSION_FAILED.json',
                     'FINAL_SOURCE_ACCEPTANCE.json', 'RESUME_FAILED.json')), 'attempt cannot be reused')
    source = Path(plan['source_root'])
    source_fold = source / FOLD_REL
    require(not (source_fold / 'checkpoint_final.pth').exists(), 'unexpected source final checkpoint')
    state, latest = checkpoint(source_fold / 'checkpoint_latest.pth')
    epoch = latest['current_epoch']
    require(epoch < 175, 'source already trained 175')
    paths = sorted((source / 'evidence').glob('*')) + sorted((source / 'logs').glob('*' + plan['source_job_id'] + '*'))
    paths += sorted(source_fold.glob('training_log_*.txt')) + sorted(source_fold.glob('*.json'))
    source_hashes = {str(path.relative_to(source)): sha(path) for path in paths if path.is_file()}
    candidates = []
    for value in range(25, epoch + 1, 25):
        candidate, report = checkpoint(source_fold / f'checkpoint_epoch_{value:04d}.pth', value)
        require(equal(candidate['init_args'], state['init_args']), 'candidate model contract differs')
        candidates.append(report)
    destination = ROOT / FOLD_REL
    destination.mkdir(parents=True, exist_ok=False)
    copies = []
    for name in ['checkpoint_latest.pth'] + [f'checkpoint_epoch_{v:04d}.pth' for v in range(25, epoch + 1, 25)]:
        src, dst = source_fold / name, destination / name
        require(not dst.exists(), 'refuse overwrite')
        digest = sha(src)
        shutil.copyfile(src, dst)
        dst.chmod(0o600 if name == 'checkpoint_latest.pth' else 0o400)
        require(sha(dst) == digest == sha(src), 'checkpoint copy SHA differs')
        copies.append(dict(path=str(dst.relative_to(ROOT)), sha256=digest))
    for relative, digest in source_hashes.items():
        src = source / relative
        dst = ROOT / 'source_timeout_evidence' / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        dst.chmod(0o400)
        require(sha(dst) == digest == sha(src), 'source evidence copy SHA differs')
    write_once(EVIDENCE / 'FINAL_SOURCE_ACCEPTANCE.json', dict(status='pass', source_state='TIMEOUT',
               source_job_id=plan['source_job_id'], snapshot=observed, checkpoint=latest,
               candidate_checkpoints=candidates, source_evidence_sha256=source_hashes, checkpoint_copies=copies))
    # Fresh CPU restore against the immutable copied input; test output cannot touch training output.
    restored = cpu_test(destination / 'checkpoint_latest.pth', 'final_cpu_restore_test')
    write_once(EVIDENCE / 'REMOTE_FINAL_PREFLIGHT.json', dict(status='pass', cpu_restore=restored,
               snapshot=observed, source_state='TIMEOUT', **current_deps))
    script = ROOT / 'slurm/01_resume_k1_175.slurm'
    syntax = command(['bash', '-n', str(script)])
    require(syntax['rc'] == 0, 'Slurm syntax failed')
    write_once(EVIDENCE / 'LAUNCH_RESERVATION.json', dict(status='reserved', source_job_id=plan['source_job_id'],
               plan_sha256=sha(ROOT / 'EXECUTION_PLAN.json'), script_sha256=sha(script)))
    result = command(['sbatch', '--hold', '--parsable', str(script)])
    if result['rc'] != 0:
        write_once(EVIDENCE / 'SUBMISSION_FAILED.json', dict(status='failed', response=result))
        raise RuntimeError('unique held submission rejected; no retry')
    job = result['out'].split(';')[0].strip()
    require(job.isdigit(), 'sbatch response missing job ID; do not submit again')
    write_once(EVIDENCE / 'LAUNCH_AUDIT.json', dict(status='held', canonical_job_id=job,
               source_job_id=plan['source_job_id'], checkpoint=latest, sbatch=result,
               plan_sha256=sha(ROOT / 'EXECUTION_PLAN.json'), script_sha256=sha(script)))
    release = command(['scontrol', 'release', job])
    write_once(EVIDENCE / 'RELEASE_AUDIT.json', dict(status='released' if release['rc'] == 0 else 'failed',
               canonical_job_id=job, response=release))
    require(release['rc'] == 0, 'release failed; preserve held job, do not resubmit')
    return dict(status='submitted_once', canonical_job_id=job, checkpoint=latest,
                source_state='TIMEOUT', remote_utc=observed['remote_utc'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('stage', 'submit-after-timeout'))
    args = parser.parse_args()
    plan = verify_files(ROOT)
    observed = snapshot(plan)
    report = stage(plan, observed) if args.action == 'stage' else submit(plan, observed)
    print('K1_RESUME_OPERATION_JSON=' + json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
