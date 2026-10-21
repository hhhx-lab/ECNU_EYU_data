"""Mechanical isolated R9 copy; old frozen attempts remain untouched."""
import hashlib
import json
import shutil
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parent = root.parent / '20261016_vae_epoch004_1mm_resume_localfix_r8'
    remote = '/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_diffusion_1mm_vae_epoch004_20261016_resume_uniform_r9'
    for folder in ('runtime', 'tests', 'config', 'slurm'):
        shutil.copytree(parent / folder, root / folder, ignore=shutil.ignore_patterns('__pycache__'))
    (root / 'plan').mkdir()
    for name in ('G1_EPOCH004_RECON_1MM_DIFFUSION_ECNU_PLAN_20261013_R1.md', 'R6_DIAGNOSTIC_COMPLETE.json',
                 'splits_final_train_val_test_membership.csv', 'UPSTREAM_SOURCE_MANIFEST.json', 'corrected_label_overlay.csv'):
        shutil.copy2(parent / 'plan' / name, root / 'plan' / name)
    for name in ('common.py', 'snapshot.py', 'submit.py'):
        shutil.copy2(parent / 'operator' / name, root / 'operator' / name)
    for folder in ('runtime', 'tests', 'config', 'slurm', 'plan'):
        for p in (root / folder).rglob('*'):
            if p.is_file():
                p.chmod(0o640)
    cfg = json.loads((root / 'config/RUN.json').read_text())
    old_remote = cfg['remote_root']
    cfg.update(attempt='resume_uniform_r9', remote_root=remote, reuse_parent=old_remote,
               reuse_parent_package_sha256='a73f614ae36e2b237f8c65929faa5547aa191a9701bcc00e689e23bbb806a745',
               loss_policy='uniform_original_quantization_warning_v1',
               label_resampling_algorithm='original_physical_spacing_v2_labelwise_threshold_no_case_exception',
               nnunet_equivalence='not_independently_verified')
    (root / 'config/RUN.json').write_text(json.dumps(cfg, indent=2, sort_keys=True) + '\n')
    path = parent / 'operator/diagnose_class2_20261016T110325Z/DIAGNOSTIC.json'
    diagnostic = json.loads(path.read_text())
    reuse = dict(schema='r9_per_file_readonly_reference_v1',
        diagnostic_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        parents={old_remote: cfg['reuse_parent_package_sha256'],
                 diagnostic['existing_outputs']['candidates'][0]['resolved_directory'].split('/dataset/cases/')[0]:
                 '41c47c0516e9818c82319da2a370507ebc3fd773ef121f4baed786159e5a4158'},
        candidates=[dict(case_id=c['case_id'], split=c['split'], directory=c['resolved_directory'],
                         marker_sha256=c['marker_sha256'], mode='label_only_rebuild' if c['case_id']=='BraTS-MET-01110-002' else 'all_outputs_reference')
                    for c in diagnostic['existing_outputs']['candidates']])
    assert len(reuse['candidates']) == 514 and len(reuse['parents']) == 2
    (root / 'plan/REUSE_CANDIDATES.json').write_text(json.dumps(reuse, indent=2, sort_keys=True) + '\n')
    for p in (root / 'slurm').glob('*.slurm'):
        p.write_text(p.read_text().replace(old_remote, remote).replace('g1d1m8', 'g1d1m9').replace('G1_R8_', 'G1_R9_'))
    common = root / 'operator/common.py'
    common.write_text(common.read_text().replace(old_remote, remote))
    print(root)


if __name__ == '__main__':
    main()
