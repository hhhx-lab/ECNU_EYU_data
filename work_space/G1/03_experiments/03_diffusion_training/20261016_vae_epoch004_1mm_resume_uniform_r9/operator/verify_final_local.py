"""Verify pulled bytes and contracts without reading images or unpickling models."""
import collections
import csv
import json
from pathlib import Path

from pull_final import MODALITIES, PACKAGE_SHA, ROOT, gate, m, now


def main():
    audit = ROOT / 'operator/pull_final_20261018T065725Z'
    acceptance = json.loads((audit / 'PULL_ACCEPTANCE.json').read_text())
    light = audit / 'lightweight'
    gate(json.loads(Path(acceptance['snapshot']).read_text()))
    for name, entry in acceptance['lightweight_files'].items():
        file = light / name
        assert file.stat().st_size == entry['bytes'] and m.sha(file) == entry['sha256']
    for modality, entry in acceptance['models'].items():
        file = Path(entry['local_path'])
        assert file.stat().st_size == entry['bytes'] and m.sha(file) == entry['sha256']
        complete = json.loads((light / 'training' / modality / 'COMPLETE.ok').read_text())
        assert complete['checkpoint_sha256'] == entry['sha256'] and complete['global_step'] == 200000
    complete = json.loads((light / 'dataset/COMPLETE.ok').read_text())
    report = json.loads((light / 'dataset/PREPROCESS_REPORT.json').read_text())
    cfg = json.loads((light / 'config/RUN.json').read_text())
    with (light / 'dataset/lesions.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    with (light / 'plan/splits_final_train_val_test_membership.csv').open(newline='') as stream:
        membership = list(csv.DictReader(stream))
    assert collections.Counter(r['split'] for r in membership) == {'train': 823, 'val': 103, 'test': 104}
    assert len(complete['case_locations']) == len(complete['case_manifests']) == len(complete['cases_detail']) == 926
    assert len(rows) == complete['rows'] == 5570
    assert collections.Counter(r['split'] for r in rows) == {'train': 4968, 'val': 602}
    assert complete['resolution_warnings'] == report['warnings'] and not report['errors']
    assert m.sha(light / 'dataset/lesions.csv') == complete['csv_sha256']
    contracts = {mod: json.loads((light / 'training' / mod / 'RUN_CONTRACT.json').read_text()) for mod in MODALITIES}
    common = {key: value for key, value in contracts['t1c']['identity'].items() if key not in ('modality', 'consumed')}
    for modality, contract in contracts.items():
        identity = contract['identity']
        assert {key: value for key, value in identity.items() if key not in ('modality', 'consumed')} == common
        assert identity['modality'] == contract['config']['modality'] == modality
        assert identity['selection'] == 'train823_only' and identity['package_sha256'] == PACKAGE_SHA
        for key, value in cfg['training'].items():
            assert contract['config'][key] == value, key
        assert identity['csv_sha256'] == complete['csv_sha256'] and identity['report_sha256'] == complete['report_sha256']
        assert identity['case_manifests'] == complete['case_manifests']
        assert identity['case_locations'] == complete['case_locations']
        assert identity['reference_parents'] == complete['reference_parents']
        assert len(identity['index_sha256']) == len(identity['native_case_manifests']) == 926
        assert len(identity['consumed']) == 823 * 2
        assert all(name.endswith(('-seg.nii.gz', '-' + modality + '.nii.gz')) for name in identity['consumed'])
    assert len(acceptance['models']) == 4 and len(acceptance['lightweight_files']) == 50
    result = dict(m.META, status='pass', local_utc=now(), lightweight_files=50, final_models=4,
                  all_sizes_and_sha256_match=True, common_data_identity_match=True,
                  modality_specific_consumed_identity_expected=True, csv_rows=5570, train_rows=4968, val_rows=602,
                  independent_full_metadata_audit='not_performed', model_deserialization='not_performed',
                  quality_evaluation='not_run', snapshot_sha256=acceptance['snapshot_sha256'],
                  acceptance_sha256=m.sha(audit / 'PULL_ACCEPTANCE.json'))
    m.exclusive(audit / 'LOCAL_VERIFICATION.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
