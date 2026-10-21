"""Create immutable authorization and exact source snapshots before remote work."""

import datetime
import shutil

from common import EXPECTED_MEMBERSHIP, EXPECTED_PLAN, MEMBERSHIP, OVERLAY, PLAN_DIR, PLAN_NAME, PROJECT, ROOT, m


def main():
    assert m.sha(PLAN_DIR / PLAN_NAME) == EXPECTED_PLAN, 'approved plan drift'
    assert m.sha(MEMBERSHIP) == EXPECTED_MEMBERSHIP, 'membership drift'
    destination = ROOT / 'plan'
    destination.mkdir(exist_ok=False)
    for source, name in ((PLAN_DIR / PLAN_NAME, PLAN_NAME), (MEMBERSHIP, MEMBERSHIP.name), (OVERLAY, OVERLAY.name)):
        shutil.copy2(source, destination / name)
        assert m.sha(source) == m.sha(destination / name)
    source_root = PROJECT / 'work_space/G1/01_code/tumour_diffusion_1mm'
    expected = {}
    for source in sorted(source_root.rglob('*')):
        if source.is_file() and '__pycache__' not in source.parts and source.suffix != '.pyc' and source.name != '.DS_Store':
            relative = source.relative_to(source_root)
            target = ROOT / 'runtime/upstream_source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            shutil.copy2(source, target)
            expected[str(relative)] = m.sha(source)
            assert expected[str(relative)] == m.sha(target)
    assert len(expected) == 37, f'unexpected source count: {len(expected)}'
    m.exclusive(destination / 'UPSTREAM_SOURCE_MANIFEST.json', {**m.META, 'files': expected, 'files_count': len(expected), 'source_root': str(source_root)})
    m.exclusive(destination / 'EXECUTION_AUTHORIZATION.json', {
        **m.META, 'status': 'execution_authorized', 'authorized_by': 'human_user',
        'user_statement': '就这样执行！！',
        'recorded_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'approved_plan_sha256': EXPECTED_PLAN, 'membership_sha256': EXPECTED_MEMBERSHIP,
        'overlay_sha256': m.sha(OVERLAY), 'per_modality_successful_updates': 200000,
        'modalities': ['t1c', 't1n', 't2w', 't2f'],
        'authorized_scope': ['P0_integrity', 'P1_train_val_1mm', 'independent_runtime_engineering', 'P2_smoke_and_review', 'P3_only_after_all_gates', 'P4_P5_development_selection_and_reports', 'controlled_timeout_resume_with_full_unchanged_contract'],
        'excluded_scope': ['large_dataset_download', 'drive_upload', 'locked104_quality_evaluation', 'S2_deployment', 'dependency_install', 'unexpected_failure_retry'],
        'upstream_source_manifest_sha256': m.sha(destination / 'UPSTREAM_SOURCE_MANIFEST.json'),
    })
    print('AUTHORIZATION_AND_37_SOURCE_FILES_FROZEN')


if __name__ == '__main__':
    main()
