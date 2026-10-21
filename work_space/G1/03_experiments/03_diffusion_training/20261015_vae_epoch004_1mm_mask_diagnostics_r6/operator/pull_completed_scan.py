"""Bounded, read-only pull of completed diagnostic evidence; never query jobs."""
import base64
import collections
import csv
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shlex

from common import ROOT, REMOTE, m, transport

SNAPSHOT = ROOT / 'operator/snapshot_scan_20261015T124723Z/SNAPSHOT.json'
LIMIT = 32 * 1024 * 1024


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    snapshot = json.loads(SNAPSHOT.read_text())
    assert '4088612|COMPLETED|0:0|' in snapshot['sacct']['stdout']
    assert not snapshot['drift'] and not snapshot['reports']['FAILED.json']['exists']
    assert snapshot['case_reports'] == 926
    access = transport('pull_completed_scan')
    code = r'''
import base64,datetime,hashlib,json,pathlib
root=pathlib.Path(__ROOT__)
assert not (root/'scan/FAILED.json').exists()
validation=json.loads((root/'scan/VALIDATION.json').read_text())
complete=json.loads((root/'scan/COMPLETE.ok').read_text())
assert validation==complete and validation['status']=='complete'
assert validation['cases']==926 and validation['split_counts']=={'train':823,'val':103}
assert validation['test_images_read']==0
assert len(validation['case_report_sha256'])==926
assert hashlib.sha256((root/'plan/PACKAGE_MANIFEST.json').read_bytes()).hexdigest()==validation['package_sha256']
paths=[root/'scan'/n for n in ('STARTED.json','VALIDATION.json','COMPLETE.ok','coverage.csv')]
paths+=sorted((root/'scan/cases').glob('*.json'))
paths+=sorted((root/'logs').glob('g1dmaskr6_4088612.*'))
assert len(paths)==932
assert sum(p.stat().st_size for p in paths)<=32*1024*1024
files={}
for p in paths:
 assert p.is_file() and not p.is_symlink()
 b=p.read_bytes();h=hashlib.sha256(b).hexdigest()
 if p.parent.name=='cases':assert h==validation['case_report_sha256'][p.stem]
 elif p.name=='coverage.csv':assert h==validation['csv_sha256']
 files[str(p.relative_to(root))]={'sha256':h,'bytes':len(b),'base64':base64.b64encode(b).decode()}
print(json.dumps({'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files}))
'''.replace('__ROOT__', repr(str(REMOTE)))
    compile(code, 'read_only_completed_pull', 'exec')
    try:
        access.start_vpn()
        raw = access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -c ' + shlex.quote(code), timeout=200)
        payload = json.loads(raw.strip().splitlines()[-1])
        assert len(payload['files']) == 932
        blobs = {}
        for name, value in payload['files'].items():
            path = Path(name)
            assert not path.is_absolute() and '..' not in path.parts
            assert path.parts[0] in ('scan', 'logs')
            data = base64.b64decode(value['base64'], validate=True)
            assert len(data) == value['bytes'] and digest(data) == value['sha256']
            blobs[name] = data
        assert sum(map(len, blobs.values())) <= LIMIT
        validation = json.loads(blobs['scan/VALIDATION.json'])
        assert validation == json.loads(blobs['scan/COMPLETE.ok'])
        assert digest(blobs['scan/VALIDATION.json']) == snapshot['reports']['VALIDATION.json']['sha256']
        assert digest(blobs['scan/COMPLETE.ok']) == snapshot['reports']['COMPLETE.ok']['sha256']
        assert digest(blobs['scan/coverage.csv']) == validation['csv_sha256']
        with (ROOT / 'plan/membership.csv').open(newline='') as f:
            members = {r['source_case_id']: r for r in csv.DictReader(f) if r['split'] != 'test'}
        reports = [json.loads(blobs['scan/cases/' + case + '.json']) for case in members]
        assert len(reports) == 926 and set(members) == set(validation['case_report_sha256'])
        for report in reports:
            case = report['case']
            assert report['split'] == members[case]['split']
            assert report['patient_group'] == members[case]['patient_group']
            assert digest(blobs['scan/cases/' + case + '.json']) == validation['case_report_sha256'][case]
        errors = [r['case'] for r in reports if r['status'] == 'identity_or_data_error']
        anomalies = [r for r in reports if r['status'] == 'coverage_anomaly']
        assert errors == validation['identity_error_cases']
        assert {r['case'] for r in anomalies} == set(validation['strict_anomaly_cases'])
        assert {r['case'] for r in reports if r.get('coverage', {}).get('nonzero_missing_voxels', 0)} == set(validation['nonzero_anomaly_cases'])
        rows = list(csv.DictReader(io.StringIO(blobs['scan/coverage.csv'].decode())))
        assert len(rows) == 926 and {r['case'] for r in rows} == set(members)
        for report in reports:
            if report['case'] in errors:
                continue
            assert report['sources']['seg']['sha256'] == report['label_sha256']
            assert set(report['sources']) == {'t1n', 't1c', 't2f', 'seg'}
        for name, data in blobs.items():
            path = access.audit / 'pulled' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440), 'wb') as f:
                f.write(data)
            assert m.sha(path) == digest(data)
        numeric = ('positive_missing_voxels', 'nonzero_missing_voxels', 'old_missing_with_negative_source',
                   'old_missing_all_three_zero', 'nonzero_missing_all_three_zero', 'label_all_three_zero')
        summary = {**m.META, 'status': 'diagnostic_complete_coverage_blocked' if anomalies or errors else 'diagnostic_complete',
                   'scope': validation['scope'], 'job_id': snapshot['job_id'],
                   'snapshot_local_utc': snapshot['local_utc'], 'snapshot_remote_utc': snapshot['remote_utc'],
                   'pull_remote_utc': payload['remote_utc'], 'local_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'scan_finished_remote_utc': validation['finished_utc'], 'files': len(blobs),
                   'bytes': sum(map(len, blobs.values())), 'all_pulled_sha_verified': True,
                   'cases': len(reports), 'split_counts': dict(collections.Counter(r['split'] for r in reports)),
                   'identity_error_cases': errors, 'anomaly_cases': len(anomalies),
                   'anomaly_split_counts': dict(collections.Counter(r['split'] for r in anomalies)),
                   'STRICT_ORIGINAL_GATE': validation['STRICT_ORIGINAL_GATE'],
                   'NONZERO_COMPARISON_GATE': validation['NONZERO_COMPARISON_GATE'],
                   'totals': {k: sum(r.get('coverage', {}).get(k, 0) for r in reports) for k in numeric},
                   'positive_missing_by_label': {k: sum(r.get('coverage', {}).get('positive_missing_by_label', {}).get(k, 0) for r in reports) for k in ('1', '2', '3', '4')},
                   'sources_with_negative_voxels': sum(r['sources'][k]['negative_voxels'] > 0 for r in reports if r['case'] not in errors for k in ('t1n', 't1c', 't2f')),
                   'test_images_read': validation['test_images_read'], 'training_submitted': False,
                   'file_sha256': {name: digest(data) for name, data in blobs.items()}}
        m.exclusive(access.audit / 'PULL_ACCEPTANCE.json', summary)
        print(json.dumps({'audit': str(access.audit), **{k: v for k, v in summary.items() if k != 'file_sha256'}}, ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
