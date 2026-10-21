"""Mechanical isolated copy; never edits or includes R7 historical runtime evidence."""
import json
import shutil
from pathlib import Path


def main():
    parent = Path(__file__).resolve().parents[1]
    root = parent.parent / '20261016_vae_epoch004_1mm_resume_localfix_r8'
    remote = '/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_diffusion_1mm_vae_epoch004_20261016_resume_localfix_r8'
    root.mkdir(exist_ok=False)
    for folder in ('runtime', 'tests', 'config', 'slurm'):
        shutil.copytree(parent / folder, root / folder)
    (root / 'plan').mkdir()
    (root / 'operator').mkdir()
    names = ('G1_EPOCH004_RECON_1MM_DIFFUSION_ECNU_PLAN_20261013_R1.md',
             'R6_DIAGNOSTIC_COMPLETE.json', 'splits_final_train_val_test_membership.csv',
             'UPSTREAM_SOURCE_MANIFEST.json', 'corrected_label_overlay.csv')
    for name in names:
        shutil.copy2(parent / 'plan' / name, root / 'plan' / name)
    for name in ('submit.py', 'common.py', 'snapshot.py'):
        shutil.copy2(parent / 'operator' / name, root / 'operator' / name)
    for p in root.rglob('*'):
        if p.is_file():
            p.chmod(0o640)
    cfg = json.loads((root / 'config/RUN.json').read_text())
    cfg.update(attempt='resume_localfix_r8', remote_root=remote,
               reuse_parent=cfg['remote_root'],
               reuse_parent_package_sha256='41c47c0516e9818c82319da2a370507ebc3fd773ef121f4baed786159e5a4158')
    (root / 'config/RUN.json').write_text(json.dumps(cfg, indent=2, sort_keys=True) + '\n')
    diagnostic = json.loads((parent / 'operator/diagnose_class_loss_20261016T075545Z/DIAGNOSTIC.json').read_text())
    reuse = {'schema': 'r8_readonly_parent_candidates_v1', 'parent': cfg['reuse_parent'],
             'parent_package_sha256': cfg['reuse_parent_package_sha256'],
             'diagnostic_sha256': '2d42f6bda6a7da4c6c2e751ecaa53fd39441c77ccb4e07111a8e771867c997fd',
             'candidates': diagnostic['reuse_inventory']['candidates']}
    (root / 'plan/REUSE_CANDIDATES.json').write_text(json.dumps(reuse, indent=2, sort_keys=True) + '\n')
    case = diagnostic['single_case']
    exception = {'schema': 'r8_one_case_nearest_background_restore_v1', 'case_id': diagnostic['case_id'],
                 'split': diagnostic['split'], 'source_seg_sha256': case['source_seg_sha256'],
                 'native_shape': case['native_shape'], 'native_spacing_mm': case['native_spacing_mm'],
                 'native_class4_voxels': case['native_label_counts']['4'],
                 'target_shape': case['target_shape'], 'target_ijk': case['nearest_class4_target_ijk_first64'],
                 'expected_default_label_counts': case['exact_label_counts'],
                 'allowed_previous_label': 0, 'restore_label': 4,
                 'native_components_6connectivity_voxels': [16, 3],
                 'all_native_components_preserved': False,
                 'scope': 'this case only; four original-grid nearest supported background targets, no global algorithm change'}
    (root / 'plan/LOCAL_LABEL_EXCEPTION.json').write_text(json.dumps(exception, indent=2, sort_keys=True) + '\n')
    old_remote = cfg['reuse_parent']
    for p in (root / 'slurm').glob('*.slurm'):
        p.write_text(p.read_text().replace(old_remote, remote).replace('g1d1m7', 'g1d1m8').replace('G1_R7_', 'G1_R8_'))
    print(root)


if __name__ == '__main__':
    main()
