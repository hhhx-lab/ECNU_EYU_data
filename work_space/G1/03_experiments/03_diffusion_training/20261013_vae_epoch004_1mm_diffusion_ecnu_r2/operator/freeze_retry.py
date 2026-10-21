"""Freeze the authorized independent retry without modifying the failed parent."""

import datetime
import json

from common import EXPECTED_PLAN, ROOT, m


def main():
    parent = ROOT.with_name('20261013_vae_epoch004_1mm_diffusion_ecnu_r1')
    auth = json.loads((parent / 'plan/EXECUTION_AUTHORIZATION.json').read_text())
    assert auth['approved_plan_sha256'] == EXPECTED_PLAN
    assert m.sha(ROOT / 'plan/G1_EPOCH004_RECON_1MM_DIFFUSION_ECNU_PLAN_20261013_R1.md') == EXPECTED_PLAN
    manifest = json.loads((ROOT / 'plan/UPSTREAM_SOURCE_MANIFEST.json').read_text())
    assert len(manifest['files']) == 37
    for relative, expected in manifest['files'].items():
        assert m.sha(ROOT / 'runtime/upstream_source' / relative) == expected
    parent_failure = parent / 'operator/submit_p0_20261013T130155Z/FAILED.json'
    auth.update({
        'user_statement': '现在空间充裕，进行重提！开始进行扩散模型训练',
        'recorded_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'attempt': 'r2', 'parent_failure_path': str(parent_failure),
        'parent_failure_sha256': m.sha(parent_failure),
        'retry_scope': 'one independent P0 retry; later stages retain all original gates',
    })
    m.exclusive(ROOT / 'plan/EXECUTION_AUTHORIZATION.json', auth)
    m.exclusive(ROOT / 'config/P0_CAPACITY_CONTRACT.json', {
        **m.META, 'stage': 'P0_native_integrity_only', 'minimum_free_bytes': 1024**3,
        'output_kind': 'lightweight JSON/JSONL and logs only',
        'writes_image_arrays': False, 'writes_checkpoints': False,
        'not_authorization_for': ['P1_1mm_dataset_capacity', 'P2_GPU_memory', 'P3_checkpoint_capacity'],
        'original_plan_sha256': EXPECTED_PLAN,
        'parent_failure_sha256': m.sha(parent_failure),
    })
    print('R2_RETRY_AUTHORIZATION_AND_STAGE_SPECIFIC_CAPACITY_FROZEN')


if __name__ == '__main__':
    main()
