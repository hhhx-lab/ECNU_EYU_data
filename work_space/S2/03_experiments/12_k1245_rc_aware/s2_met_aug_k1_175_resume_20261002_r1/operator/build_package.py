"""Generate derived environment and SHA contracts from the accepted source snapshot."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSPECTION = ROOT.parent / ('s2_met_aug_k1245_20260923_r3_fallback/operator/'
    'k1_resume_source_inspection_20261002_20261002T034139Z')
SOURCE = '/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1245_20260929_k1_175_r3'
REMOTE = '/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/' + ROOT.name


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path, text):
    with path.open('x') as handle:
        handle.write(text)


def main():
    snapshot = json.loads((INSPECTION / 'SOURCE_CHECKPOINT_INSPECTION.json').read_text())
    source_script = INSPECTION / 'pulled/slurm/01_train_k1_175_r3.slurm'
    script = source_script.read_text()
    start = script.index('export PATH=')
    end = script.index('\n\n"$PYTHON" - "$STARTED"', start)
    env = '#!/bin/bash\n# Derived verbatim exports from the frozen source Slurm; outputs alone are redirected.\n'
    env += 'ROOT=' + SOURCE + '\n'
    env += 'PARENT=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1\n'
    env += 'REPO="$ROOT/runtime/repository"\n'
    env += 'RESULTS="$RESUME_ROOT/results/K1"\n'
    env += 'FOLD="$RESULTS/Dataset264_BraTS2026_MET_Completion/nnUNetTrainerBraTS2026RCMetAugK1245FocalCompletionFineTune175__nnUNetPlans__3d_fullres/fold_0"\n'
    env += script[start:end] + '\nexport S2_CONTINUE=1\n'
    env += 'export S2_EXPERIMENT_MODE=met_aug_k1_175_resume_20261002_r1_K1\n'
    write_once(ROOT / 'runtime/research_environment.sh', env)
    plan = {key: snapshot[key] for key in ('artifact_status', 'operator_approved', 'formal_gate_status')}
    plan.update(experiment_id=ROOT.name, source_root=SOURCE, remote_root=REMOTE, source_job_id='3827543',
                source_job_name='s2k1train175r3', source_timeout_required=True,
                source_contract_sha256=snapshot['source_contract_sha256'],
                installed_cli_sha256=sha(INSPECTION / 'pulled/installed/run_training.py'),
                source_slurm_sha256=sha(source_script), total_epochs=175, initial_lr=0.001,
                poly_power=0.9, save_every=25, seed=20260724, train_cases=1035, validation_cases=103,
                checkpoint_epochs=[25, 50, 75, 100, 125, 150, 175],
                F_STRICT_PREREG_GATE=False, k5_authorized=False, exploratory_only=True,
                submit_policy='one held sbatch after TIMEOUT, final checkpoint validation and atomic audit',
                source_checkpoint_policy='do not freeze live latest; copy full final latest after TIMEOUT',
                frozen_source_runtime='read-only reuse; all source custom_nnunet file SHAs bound',
                known_inspection=dict(local_utc=snapshot['local_utc'], remote_utc=snapshot['remote_utc'],
                                      current_epoch=50, log_epoch=69, latest_sha256=snapshot['checkpoint']['sha256']))
    write_once(ROOT / 'EXECUTION_PLAN.json', json.dumps(plan, indent=2, sort_keys=True) + '\n')
    paths = [ROOT / 'EXECUTION_PLAN.json', ROOT / 'README.md',
             *sorted((ROOT / 'runtime').glob('*')), *sorted((ROOT / 'slurm').glob('*')),
             *sorted((ROOT / 'operator').glob('*.py'))]
    files = {str(p.relative_to(ROOT)): sha(p) for p in paths if p.is_file()}
    write_once(ROOT / 'PACKAGE_SHA256.json', json.dumps(dict(files=files, **{
        key: snapshot[key] for key in ('artifact_status', 'operator_approved', 'formal_gate_status')}),
        indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(plan_sha256=sha(ROOT / 'EXECUTION_PLAN.json'), files=len(files))))


if __name__ == '__main__':
    main()
