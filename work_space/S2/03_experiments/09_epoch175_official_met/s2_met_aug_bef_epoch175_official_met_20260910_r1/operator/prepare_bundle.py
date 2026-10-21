"""Freeze a new evaluation plan from the locally accepted175 and125 evidence."""
import ast
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
REMOTE_BASE = Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2')
EXT = BASE / 's2_met_aug_bef_extend175_20260907_r4'
OLD = BASE / 's2_met_aug_btof_selected103_official_met_20260907_r2'
R4 = BASE / 's2_met_aug_btof_selected103_official_met_20260907_r4'
R_EXT = REMOTE_BASE / 'experimental_unvalidated' / EXT.name
R_OLD = REMOTE_BASE / 'results/experimental_unvalidated' / OLD.name
META = dict(artifact_status='experimental_unvalidated', operator_approved=False, formal_gate_status='not_run_not_passed')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, payload):
    with path.open('x') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


for file in ROOT.rglob('*.py'):
    ast.parse(file.read_text(), filename=str(file))
for file in [*ROOT.glob('slurm/*.slurm'), ROOT / 'runtime/run_official_met_eval.sh']:
    subprocess.run(['bash', '-n', str(file)], check=True)
assert sha(ROOT / 'runtime/run_official_met_eval.sh') == sha(R4 / 'runtime/run_official_met_eval.sh')
before = (R4 / 'runtime/frozen_s2_inference.py').read_text()
after = (ROOT / 'runtime/frozen_s2_inference.py').read_text()
assert after.replace('    "nnUNetTrainerBraTS2026RCMetAugScreen175ExtensionFocalCompletionFineTune",\n', '') == before
routes = {}
source_jobs = []
for method in ('B', 'E_SNFH', 'F'):
    val_file = EXT / f'evidence/{method}_EXTENSION_TRAINING_VALIDATION.json'
    val = read(val_file)
    old = read(OLD / f'evidence/{method}_EVALUATION_VALIDATION.json')
    best = val['best_checkpoint']
    assert val['status'] == old['status'] == 'pass' and best['total_epoch'] == 175
    source_jobs += [str(val['job_id']), str(old['job_id'])]
    routes[method] = dict(checkpoint=best['path'], checkpoint_sha256=best['sha256'], checkpoint_epoch=175, ema_fg_dice=best['ema_fg_dice'],
        training_validation=str(R_EXT / 'evidence' / val_file.name), training_validation_sha256=sha(val_file),
        training_complete=str(R_EXT / f'evidence/{method}_EXTENSION_TRAINING_COMPLETE.ok'),
        baseline_model_root=str(R_OLD / 'model_snapshots' / method), baseline_checkpoint_sha256=old['checkpoint_sha256'],
        baseline_metrics_csv=str(R_OLD / f'cohort/fixed103/models/{method}/evaluation/leaderboard_metrics.csv'),
        baseline_metrics_sha256=old['metrics_csv_sha256'], baseline_validation=str(R_OLD / f'evidence/{method}_EVALUATION_VALIDATION.json'),
        baseline_complete=str(R_OLD / f'evidence/{method}_EVALUATION_COMPLETE.ok'))
plan = dict(META, schema_version=1, experiment_id=ROOT.name, created_at_utc=datetime.now(timezone.utc).isoformat(),
    purpose='Evaluate frozen B/E_SNFH/F epoch175 on fixed103 authentic T2W and compare each against its frozen epoch125 official MET result.',
    training_allowed=False, epoch225_allowed=False, source_jobs=source_jobs, routes=routes,
    upstream_plans=[dict(path=str(R_EXT / 'EXTEND175_EXECUTION_PLAN_20260907_R4.json'), sha256='cdc19fe9b3d4d3373822c0aa0a80f594f8718d284dc12424be68114314f963ae'),
                    dict(path=str(R_OLD / 'SELECTED103_OFFICIAL_MET_EXECUTION_PLAN_20260907_R2.json'), sha256='1767372636d82e5d02bbd0e9d8ce4b982b1777a60c36c27b655c3b3b5b429c3b')],
    case_contract=dict(count=103, source_root=str(R_OLD / 'cohort/fixed103'), input_route='authentic_real_T2W',
        manifest_sha256='86ba853dadc64ffc45357895ba0bd543bf5b6b14c0c83eb789f1e875d4aa2e7e',
        audit_sha256={file:sha(OLD / 'cohort/fixed103' / file) for file in ('input_audit.json', 'input_audit.csv')},
        same_inputs_labels_and_split_as125=True, locked_test_used=False),
    inference=dict(single_gpu=True, gpu_type='A100', exclusive_node=False, execution_mode='sequential_in_process', preprocess_workers=0,
        export_workers=0, multiprocessing_queue_used=False, tile_step_size=0.5, mirroring=True, gaussian=True, fold=0),
    launch=dict(parallel_routes=True, gpu_jobs=3, dependent_cpu_jobs=1, no_resubmission=True, submit_held_then_audit_then_release=True),
    job_names=dict(B='s2b175m1', E_SNFH='s2e175m1', F='s2f175m1', PAIRED='s2x175p1'),
    official_met=dict(config='mets', BraTS_evaluation='0.0.8', panoptica='2.1.0', numpy='1.26.4', volume_threshold=27, overlap_threshold=0.2),
    bootstrap=dict(replicates=20000, seed=20260831),
    comparisons=dict(epoch175_cross_route_pairs=3, same_route_175_vs125_pairs=3, total_pairs=6, metric_count=72,
        per_metric_complete_case=True, report=['mean', 'wins/ties/losses', '95%CI', 'unadjusted_bootstrap_p', 'FP', 'FN', 'small_and_large_lesion_F1']),
    reused_runtime_sha256={f:sha(R4 / 'runtime' / f) for f in ('frozen_s2_inference.py', 'compare_paired_met.py', 'run_official_met_eval.sh')},
    runtime_deltas=['inference allowlist adds exact Screen175 trainer only; network and prediction settings unchanged',
                    'paired method count allows explicit2 or3; statistic, seed, metric definitions unchanged'],
    interpretation=dict(exploratory=True, same103_used_for_checkpoint_selection_and_reporting=True, formal_superiority_allowed=False,
        original_five_route_equal_budget_comparison_unchanged=True, extension_not_equal_budget_to125=True, input_is_not_G1_synthetic=True),
    evidence_policy=dict(preserve_old_attempts=True, fresh_output_root=True, pull_checkpoint=False, pull_nifti=False, pull_events_jsonl=False))
plan_path = ROOT / 'EPOCH175_OFFICIAL_MET_EXECUTION_PLAN_20260910_R1.json'
write(plan_path, plan)
files = [plan_path, ROOT / 'README.md', *ROOT.glob('runtime/*.py'), *ROOT.glob('runtime/*.sh'),
         *ROOT.glob('runtime/custom_nnunet/*.py'), *ROOT.glob('slurm/*.slurm'), *ROOT.glob('manifests/*.csv')]
with (ROOT / 'STATIC_SHA256SUMS.txt').open('x') as f:
    for path in sorted(files):
        f.write(sha(path) + '  ' + str(path.relative_to(ROOT)) + '\n')
write(ROOT / 'evidence/LOCAL_BUNDLE_VALIDATION.json', dict(META, status='pass', checked_at_utc=datetime.now(timezone.utc).isoformat(),
    plan_sha256=sha(plan_path), static_file_count=len(files), python_ast='pass', shell_syntax='pass', unchanged_official_met_wrapper=True,
    inference_only_allowlist_change=True, frozen175={m:r['checkpoint_sha256'] for m,r in routes.items()}))
print(json.dumps(dict(status='pass', plan_sha256=sha(plan_path), file_count=len(files))))
