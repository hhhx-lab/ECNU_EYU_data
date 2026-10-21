"""Freeze an isolated K1 selected-175 official MET package from accepted sources."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
K24 = PARENT / 's2_met_aug_k24_selected103_official_met_20261002_r1'
K1 = PARENT / 's2_met_aug_k1_175_resume_20261002_r1'
REMOTE = '/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/' + ROOT.name
META = dict(artifact_status='experimental_unvalidated', operator_approved=False,
            formal_gate_status='not_run_not_passed')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1048576), b''):
            h.update(part)
    return h.hexdigest()


def write_once(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def main():
    source_plan = K24 / 'EXECUTION_PLAN.json'
    source = json.loads(source_plan.read_text())
    resume_plan = json.loads((K1 / 'EXECUTION_PLAN.json').read_text())
    assert source['case_contract']['count'] == 103 and source['bootstrap'] == {'replicates': 20000, 'seed': 20260831}
    for relative in ('runtime/inference.py', 'runtime/frozen_s2_inference.py',
                     'runtime/run_official_met_eval.sh',
                     'runtime/custom_nnunet/nnUNetTrainerBraTS2026RC_inference.py',
                     'runtime/custom_nnunet/__init__.py',
                     'manifests/FIXED103_REAL_T2W_CASES.csv'):
        assert sha(ROOT / relative) == sha(K24 / relative), relative
    report_file = K1 / 'evidence/TRAINING_VALIDATION.json'
    report = json.loads(report_file.read_text())
    assert report['status'] == 'pass' and report['completed_epochs'] == 175
    assert report['nnunet_validation_cases'] == 103 and report['F_STRICT_PREREG_GATE'] is False
    epochs = [row['current_epoch'] for row in report['candidate_checkpoints']]
    assert epochs == [25, 50, 75, 100, 125, 150, 175]
    best = sorted(report['candidate_checkpoints'], key=lambda row: (-row['ema_fg_dice'], row['current_epoch']))[0]
    assert best == report['best_checkpoint'] and best['current_epoch'] == 175
    assert best['sha256'] == '28e81950b675cd0f5ca26cc1509f9271b2fa827b524f379024b417c0a5402204'
    route = dict(training_job_id='3919543', training_root=resume_plan['remote_root'],
                 training_validation_sha256=sha(report_file), checkpoint=best['path'],
                 checkpoint_sha256=best['sha256'], selected_epoch=175,
                 selection_ema_fg_dice=best['ema_fg_dice'])
    assert route['training_root'] == '/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/' + K1.name
    plan = dict(META, experiment_id=ROOT.name, remote_root=REMOTE, routes={'K1': route},
                case_contract=source['case_contract'], training_split_root=source['training_split_root'],
                bootstrap=source['bootstrap'], exploratory_only=True,
                fixed103_for_selection_and_report=True, F_STRICT_PREREG_GATE=False,
                training_budget=175, preprocess_workers=0, export_workers=0,
                multiprocessing_queue_used=False, exclusive_node=False,
                job_names={'K1': 's2k1met103r1'}, source_k24_plan_sha256=sha(source_plan),
                inference_scope='selected checkpoint official MET; no online augmentation during inference',
                paired_status='await K1 official MET, then paired K1/K2/K4 and scientific gate audit')
    write_once(ROOT / 'EXECUTION_PLAN.json', plan)
    write_once(ROOT / 'evidence/TRAINING_ACCEPTANCE_SOURCE.json', report)
    for path in (ROOT / 'runtime').rglob('*.py'):
        ast.parse(path.read_text(), str(path))
    subprocess.run(['bash', '-n', str(ROOT / 'slurm/evaluate.slurm')], check=True)
    files = {str(path.relative_to(ROOT)): sha(path) for path in ROOT.rglob('*')
             if path.is_file() and 'operator' not in path.relative_to(ROOT).parts}
    write_once(ROOT / 'STATIC_SHA256.json', dict(META, files=files))
    print(json.dumps({'status': 'prepared', 'plan_sha256': sha(ROOT / 'EXECUTION_PLAN.json'),
                      'static_files': len(files), 'selected_epoch': best['current_epoch'],
                      'checkpoint_sha256': best['sha256']}))


if __name__ == '__main__':
    main()
