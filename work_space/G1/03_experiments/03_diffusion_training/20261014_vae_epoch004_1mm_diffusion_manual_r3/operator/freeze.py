"""Freeze the new waiver run without modifying its parent or active user sources."""
import json
import shutil
from common import ROOT, PROJECT, REMOTE, RECON, SOURCE_CODE, m


def main():
    parent = ROOT.parent / '20261013_vae_epoch004_1mm_diffusion_ecnu_r2'
    for relative in ('plan/G1_EPOCH004_RECON_1MM_DIFFUSION_ECNU_PLAN_20261013_R1.md',
                     'plan/splits_final_train_val_test_membership.csv', 'plan/corrected_label_overlay.csv',
                     'plan/UPSTREAM_SOURCE_MANIFEST.json', 'config/P0_SOURCE_MAPPING.json'):
        dest = ROOT / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        assert not dest.exists()
        shutil.copy2(parent / relative, dest)
    shutil.copytree(parent / 'runtime/upstream_source', ROOT / 'runtime/upstream_source')
    assert m.sha(ROOT / 'plan/G1_EPOCH004_RECON_1MM_DIFFUSION_ECNU_PLAN_20261013_R1.md') == 'fe538efa6c4d1c2fae4aed2003270a5dd785366061e8ddede0d71d17c87c5353'
    upstream = json.loads((ROOT / 'plan/UPSTREAM_SOURCE_MANIFEST.json').read_text())
    assert len(upstream['files']) == 37
    assert all(m.sha(ROOT / 'runtime/upstream_source' / p) == expected for p, expected in upstream['files'].items())
    config = {**m.META, 'attempt': 'manual_r3', 'remote_root': str(REMOTE), 'recon': str(RECON),
              'source_code': str(SOURCE_CODE), 'seed': 20261013, 'preprocess_workers': 4,
              'source_csv_sha256': '0d1765e12cdad7d8a5df57a5d2c46c79414ed99a2015a4d8fed57cd4c4fc836b',
              'training': dict(batch_size=16, num_workers=4, loader_workers=0, in_channels=5, out_channels=1,
                 feature_size=48, use_checkpoint=False, use_compile=False, optim_lr=2e-4, reg_weight=1e-5,
                 adam_beta1=.9, num_steps=200000, n_steps=1000, noise_type='gaussian_tumour',
                 generator_type='Unet_NnU', time_ch_count=8, dataset='BRATS_2024', normalization='nnunet',
                 cache_rate=1., use_all_rows=False, no_roi_cache=False, planner_gpu_gb=48., crop_size=64,
                 small_lesion_weight=3., small_lesion_threshold=27., small_lesion_clamp=1.,
                 patient_balance_mode='none', noise_schedule='edm', noise_embedding_mode='continuous',
                 sigma_data=1., sigma_min=.002, sigma_max=50., rho=7., snr_shift=0., p_uncond=0.)}
    m.exclusive(ROOT / 'config/RUN.json', config)
    m.exclusive(ROOT / 'plan/EXECUTION_AUTHORIZATION.json', {**m.META, 'status': 'execution_authorized_with_user_waiver',
                 'scope': 'one independent preprocess and four afterok single-A100 training submissions',
                 'user_request': '算了，不验收了吧，直接人工授权通过了；接下来直接提交训练新扩散模型',
                 'waiver_sha256': m.sha(ROOT / 'plan/USER_WAIVER_20261014.md'),
                 'original_plan_sha256': m.sha(ROOT / 'plan/G1_EPOCH004_RECON_1MM_DIFFUSION_ECNU_PLAN_20261013_R1.md'),
                 'membership_sha256': m.sha(ROOT / 'plan/splits_final_train_val_test_membership.csv'),
                 'overlay_sha256': m.sha(ROOT / 'plan/corrected_label_overlay.csv'),
                 'parent_timeout_review_sha256': m.sha(parent / 'P0_TIMEOUT_REVIEW_20261014.md'),
                 'P0_STRICT_ACCEPTANCE': False, 'P1_MANUAL_REVIEW': 'waived_by_user',
                 'P2_SEPARATE_SMOKE': 'waived_by_user', 'independent_manual_review_performed': False,
                 'scientific_gate_status': 'not_established', 'training_config_sha256': m.sha(ROOT / 'config/RUN.json')})
    slurm = ROOT / 'slurm'
    slurm.mkdir(exist_ok=False)
    for phase in ('prepare', 't1c', 't1n', 't2w', 't2f'):
        prepare = phase == 'prepare'
        name = 'g1d1m3p1' if prepare else 'g1d1m3' + phase
        command = 'runtime/prepare.py' if prepare else 'runtime/train.py --modality ' + phase
        script = ('#!/bin/bash\n#SBATCH --job-name=' + name + '\n#SBATCH --partition=' + ('cpu_96G' if prepare else 'a100') +
                  '\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=8\n#SBATCH --mem=' + ('90G' if prepare else '64G') +
                  '\n#SBATCH --time=' + ('24:00:00' if prepare else '48:00:00') +
                  ('\n' if prepare else '\n#SBATCH --gres=gpu:1\n#SBATCH --signal=USR1@300\n') +
                  '#SBATCH --output=' + str(REMOTE / 'logs') + '/' + name + '_%j.out\n#SBATCH --error=' + str(REMOTE / 'logs') + '/' + name + '_%j.err\n'
                  'set -euo pipefail\nif [[ "${G1_STATIC_DRY_RUN:-0}" == "1" ]]; then\n  printf "G1_R3_BASH_U_DRY_RUN_PASS\\n"\n  exit 0\nfi\n'
                  'export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=20261013\n'
                  'unset PYTHONPATH\ncd ' + str(REMOTE) + '\nprintf "START %s UTC " "${SLURM_JOB_ID}"; date -u +%FT%TZ\nhostname\n'
                  'exec ' + m.PYTHON + ' -B -u ' + command + '\n')
        path = slurm / (phase + '.slurm')
        with path.open('x') as h:
            h.write(script)
    print(json.dumps({'root': str(ROOT), 'status': 'waiver_config_and_slurm_frozen'}))


if __name__ == '__main__':
    main()
