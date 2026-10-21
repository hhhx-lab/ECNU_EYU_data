"""New attempt with measured format bounds rather than arbitrary 10-percent inflation."""
import json
import shutil
from common import ROOT, REMOTE, PARENT, m


def main():
    package = json.loads((PARENT / 'plan/PACKAGE_MANIFEST.json').read_text())
    assert all(m.sha(PARENT / p) == v for p, v in package['files'].items())
    for relative in package['files']:
        if relative == 'README.md' or relative in ('config/RUN.json', 'plan/EXECUTION_AUTHORIZATION.json') or relative.startswith('slurm/'):
            continue
        dest = ROOT / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PARENT / relative, dest)
    cfg = json.loads((PARENT / 'config/RUN.json').read_text())
    cfg.update(attempt='manual_r5', remote_root=str(REMOTE))
    m.exclusive(ROOT / 'config/RUN.json', cfg)
    auth = json.loads((PARENT / 'plan/EXECUTION_AUTHORIZATION.json').read_text())
    auth.update(attempt='manual_r5', retry_scope='one independent R5 format-bound capacity correction and unique five-job launch',
                immediate_parent_package_sha256=m.sha(PARENT / 'plan/PACKAGE_MANIFEST.json'),
                parent_capacity_failure_local_sha256=m.sha(next((PARENT/'operator').glob('complete_unexecuted_preflight_*/FAILED.json'))),
                training_config_sha256=m.sha(ROOT / 'config/RUN.json'))
    m.exclusive(ROOT / 'plan/EXECUTION_AUTHORIZATION.json', auth)
    slurm = ROOT / 'slurm'
    slurm.mkdir(exist_ok=False)
    old_remote = json.loads((PARENT / 'config/RUN.json').read_text())['remote_root']
    for path in (PARENT / 'slurm').glob('*.slurm'):
        value = path.read_text().replace(old_remote, str(REMOTE)).replace('g1d1m4', 'g1d1m5').replace('G1_R4_', 'G1_R5_')
        with (slurm / path.name).open('x') as h:
            h.write(value)
    m.exclusive(ROOT / 'plan/CAPACITY_METHOD_CORRECTION.json', {**m.META,
        'previous_required_bytes':231590329172,'previous_available_bytes':224164577280,
        'previous_data_full_FOV_bytes':192322829808,
        'new_method':'full_FOV uncompressed payload plus zlib default compressBound, 1GiB header/metadata reserve',
        'retained_reserves':'4workers two-largest temporary outputs; 12GiB four-route checkpoints; additional4GiB',
        'not_removed':'no cases, checkpoints, dtype precision, batch or updates removed; no old files deleted'})
    print('R5_INDEPENDENT_BASE_FROZEN')


if __name__ == '__main__':
    main()
