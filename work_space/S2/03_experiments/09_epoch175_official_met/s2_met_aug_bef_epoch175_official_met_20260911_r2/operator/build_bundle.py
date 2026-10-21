"""Freeze a new independent CPU recovery bundle from already retrieved provenance."""
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 's2_met_aug_bef_epoch175_official_met_20260910_r1'
REMOTE_BASE = '/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/'
META = dict(artifact_status='experimental_unvalidated', operator_approved=False, formal_gate_status='not_run_not_passed')
PLAN = 'CPU_RECOVERY_EXECUTION_PLAN_20260911_R2.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old = json.loads((OLD / 'EPOCH175_OFFICIAL_MET_EXECUTION_PLAN_20260910_R1.json').read_text())
    snapshot = json.loads((OLD / 'operator/EXISTING_METRICS_20260911T075919Z.json').read_text())
    target = ROOT / 'runtime/compare_paired_met.py'
    with target.open('xb') as f:
        f.write((OLD / 'runtime/compare_paired_met.py').read_bytes())
    assert sha(target) == '9ff245240c438beb998652ee59f130fc8c012be29ac3b689c862d04135df6e12'
    for p in (ROOT / 'runtime').glob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    routes = {}
    for m, route in old['routes'].items():
        failure = snapshot['evidence'][m + '_EVALUATION_FAILED.json']
        routes[m] = dict(checkpoint_sha256=route['checkpoint_sha256'], failed_job_id=str(failure['job_id']),
                         source_failure_json=failure, metrics175_sha256=snapshot['metrics'][m]['175']['csv_sha256'],
                         summary175_sha256=snapshot['metrics'][m]['evaluation_validation']['summary_json_sha256'])
    p = dict(META, experiment_id=ROOT.name, created_local_utc=datetime.now(timezone.utc).isoformat(),
             remote_root=REMOTE_BASE+ROOT.name, source_root=REMOTE_BASE+OLD.name,
             source_plan_name='EPOCH175_OFFICIAL_MET_EXECUTION_PLAN_20260910_R1.json',
             source_plan_sha256=sha(OLD / 'EPOCH175_OFFICIAL_MET_EXECUTION_PLAN_20260910_R1.json'), routes=routes,
             failed_jobs={'B':'3671905','E_SNFH':'3671906','F':'3671908','PAIRED':'3671909'},
             user_authorization='New CPU-only recovery; report original125 and175 values side by side, not just differences.',
             failure_correction='Derive legal labels0-4 from frozen dataset.json; fix omitted legal label4 in new recovery only.',
             source_outputs_read_only=True, original_failures_preserved=True, train=False, inference=False, rerun_official_evaluator=False,
             cpu_job={'name':'s2x175cr2','partition':'cpu_192G','cpus':4,'memory_gb':32,'time_limit':'02:00:00','gpu_count':0},
             bootstrap=old['bootstrap'], comparisons=old['comparisons'], comparator_sha256=sha(target),
             acceptance=['source plan/runtime/failure hashes', 'six frozen checkpoint SHA175 and125', '515 shared input/reference SHA',
                         '309 prediction hashes, finite voxels, legal dataset labels and reference geometry',
                         'six CSV103 cases/72metrics, existing official MET pass/complete, baseline epoch175 source audits',
                         'six pairs20000 bootstrap seed20260831, original means, finite n, paired means, CI, W/T/L, p and FP/FN'],
             launch_policy='one exclusive reservation; submit held, write canonical job id, release own job; never retry or overwrite',
             interpretation=old['interpretation'], light_pull_excludes=['checkpoint','NIfTI','events JSONL'])
    with (ROOT / PLAN).open('x') as f:
        json.dump(p, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    with (ROOT / 'README.md').open('x') as f:
        f.write('# Epoch175 CPU-only official MET recovery\n\nartifact_status=experimental_unvalidated\noperator_approved=false\nformal_gate_status=not_run_not_passed\n\nIndependent recovery; original r1 GPU job failures remain unchanged. No training, GPU inference or official evaluator rerun. Original125/175 values appear side by side with paired statistics. Same103 development cases: exploratory only.\n')
    files = [ROOT / PLAN, ROOT / 'README.md', *sorted((ROOT / 'runtime').glob('*.py')), *sorted((ROOT / 'slurm').glob('*'))]
    with (ROOT / 'STATIC_SHA256SUMS.txt').open('x') as f:
        f.writelines(sha(path) + '  ' + str(path.relative_to(ROOT)) + '\n' for path in files)
    print(json.dumps(dict(plan_sha256=sha(ROOT / PLAN), runtime_files=len(files), comparator_sha256=sha(target))))


if __name__ == '__main__':
    main()
