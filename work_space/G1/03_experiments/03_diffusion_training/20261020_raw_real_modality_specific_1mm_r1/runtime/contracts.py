"""Authoritative patient memberships and original-only source mapping."""
import collections
import csv
import json
from pathlib import Path

from common import ROOT, sha

MODS = ('t1c', 't1n', 't2w', 't2f')
LOSS_POLICY = 'uniform_original_quantization_warning_v1'
LABEL_ALGORITHM = 'original_physical_spacing_v2_labelwise_threshold_no_case_exception'
SUPPORT_CONTRACT = 'signal_statistics_vs_full_native_FOV_geometry_v1'
SOURCE_KIND = 'original_authentic_native'
COUNTS = {m: {'train': 823, 'val': 103} if m == 't2w' else {'train': 1035, 'val': 130} for m in MODS}


def all_members():
    with (ROOT / 'plan/master_membership.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len({r['source_case_id'] for r in rows}) == 1295
    assert collections.Counter(r['split'] for r in rows) == {'train': 1035, 'val': 130, 'test': 130}
    groups = collections.defaultdict(set)
    for row in rows:
        assert row['source_case_id'] == Path(row['source_case_id']).name
        assert row['t2w_status'] in ('authentic', 'fake_or_broken')
        assert (row['eligible_for_realonly'].lower() == 'true') == (row['t2w_status'] == 'authentic')
        groups[row['patient_group']].add(row['split'])
    assert all(len(s) == 1 for s in groups.values()), 'Patient group split leakage'
    with (ROOT / 'plan/realonly_membership.csv').open(newline='') as stream:
        real = list(csv.DictReader(stream))
    assert len(real) == len({r['source_case_id'] for r in real}) == 1030
    expected = {r['source_case_id']: r for r in rows if r['t2w_status'] == 'authentic'}
    assert expected == {r['source_case_id']: r for r in real}, 'Authentic/master identity mismatch'
    return rows


def available_mods(member):
    return MODS if member['t2w_status'] == 'authentic' else ('t1c', 't1n', 't2f')


def members(modality=None):
    assert modality is None or modality in MODS
    rows = [r for r in all_members() if r['split'] in ('train', 'val')
            and (modality != 't2w' or r['t2w_status'] == 'authentic')]
    expected = {'train': 1035, 'val': 130} if modality is None else COUNTS[modality]
    assert collections.Counter(r['split'] for r in rows) == expected
    return rows


def record_sha(record):
    import hashlib
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_records():
    path = ROOT / 'config/SOURCE_RECORDS.json'
    payload = json.loads(path.read_text())
    assert payload['source_kind'] == SOURCE_KIND and payload['status'] == 'pass'
    cases = payload['cases']
    selected = {r['source_case_id']: r for r in members()}
    assert set(cases) == set(selected), 'Source membership drift'
    roots = [Path(r) for r in payload['allowed_roots']]
    for case, entry in cases.items():
        member = selected[case]
        for key in ('split', 'patient_group', 't2w_status'):
            assert entry[key] == member[key], 'Source record membership drift'
        assert set(entry['files']) == set((*available_mods(member), 'seg')), 'Forbidden or missing source modality'
        for mod, file in entry['files'].items():
            p, target = Path(file['path']), Path(file['resolved_path'])
            assert p.is_absolute() and target.is_absolute()
            assert any(target.is_relative_to(r) for r in roots), 'Source outside original/corrected roots'
            assert p.name in (case + '-' + mod + '.nii.gz', mod + '.nii.gz')
            assert ((target.parent.name == case and target.name == mod + '.nii.gz')
                    or (mod == 'seg' and target.parent.name == 'MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels'
                        and target.name == case + '-seg.nii.gz')), 'Source case/modality redirect'
            assert len(file['sha256']) == 64 and int(file['size']) > 0
    cfg = json.loads((ROOT / 'config/RUN.json').read_text())
    assert sha(path) == cfg['source_records_sha256'], 'Frozen source records drift'
    return cases
