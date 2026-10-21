"""Original-source identity checks; importing this module performs no data I/O."""
import collections
import csv
import hashlib
import json
from pathlib import Path

from common import META, ROOT, sha, verify_package
from contracts import (COUNTS, LABEL_ALGORITHM, LOSS_POLICY, MODS, SOURCE_KIND,
                       SUPPORT_CONTRACT, available_mods, members, record_sha,
                       source_records)

SCHEMA = 'raw_real_modality_specific_1mm_v1'
NORMALIZATION = 'whole_case_common_mask_zscore_before_resampling_zero_outside_support'
MASK_ALGORITHM = 'authentic_T1N_T1C_T2F_robust_0_100_nonzero_union_holefill'


def check(condition, message):
    if not condition:
        raise ValueError(message)


def validate_record(member, entry):
    check(member['split'] in ('train', 'val'), 'Test image access forbidden')
    for key in ('split', 'patient_group', 't2w_status'):
        check(entry[key] == member[key], 'Source membership drift: ' + key)
    check(set(entry['files']) == set((*available_mods(member), 'seg')),
          'Forbidden or missing source modality')


def verify_recorded_affines(entry):
    """Recompute audited metadata, preserving the legacy signed-zero hash contract."""
    import numpy as np
    for file in entry['files'].values():
        affine = np.asarray(file['affine'], dtype=np.float64)
        check(affine.shape == (4, 4) and np.isfinite(affine).all(), 'Invalid recorded affine')
        rounded = np.round(affine, 6)
        legacy = hashlib.sha256(np.ascontiguousarray(rounded).tobytes()).hexdigest()[:16]
        canonical = rounded.copy()
        canonical[np.isclose(canonical, 0., atol=5e-7, rtol=0.)] = 0.
        observed = hashlib.sha256(np.ascontiguousarray(canonical).tobytes()).hexdigest()[:16]
        check(legacy == file['archived_affine_hash'] == file['observed_legacy_signed_zero_affine_hash'],
              'Archived legacy affine hash drift')
        check(observed == file['observed_affine_hash'], 'Recorded canonical affine hash drift')


def verify_archived_headers(records):
    """CSV/JSON metadata only; this does not establish missing cases' historic byte SHA."""
    raw = {}
    for name in ('raw_manifest.csv', 'ucsd_manifest.csv'):
        with (ROOT / 'plan' / name).open(newline='') as stream:
            for row in csv.DictReader(stream):
                raw.setdefault(row['case_id'], row)
    for case, entry in records.items():
        verify_recorded_affines(entry)
        check(case in raw, 'Archived source case missing')
        for mod, file in entry['files'].items():
            row = raw[case]
            check(file['archived_affine_hash'] == row['affine_hash_' + mod], 'Archived affine baseline drift')
            check(file['shape'] == [int(v) for v in row['shape_' + mod].split('x')], 'Archived shape drift')
            check(all(abs(float(a) - float(b)) <= 1e-4 for a, b in
                      zip(file['spacing'], row['spacing_' + mod].split(','))), 'Archived spacing drift')


def verify_source_file(file):
    path, target = Path(file['path']), Path(file['resolved_path'])
    check(path.is_absolute() and target.is_absolute(), 'Nonabsolute source path')
    check(path.resolve(strict=True) == target and target.resolve(strict=True) == target,
          'Source realpath drift: ' + str(path))
    check(target.is_file() and target.stat().st_size == int(file['size']),
          'Source size drift: ' + str(path))
    check(sha(target) == file['sha256'], 'Source SHA drift: ' + str(path))
    return target


def verify_historical_source(member, entry):
    historical = entry.get('historical_authentic_source_manifest')
    if member['t2w_status'] != 'authentic':
        check(historical is None, 'Missing-T2W case cannot inherit authentic provenance')
        return None
    check(historical is not None, 'Authentic source provenance missing')
    path = Path(historical['path'])
    check(path.is_absolute() and path.name == 'case_manifest.json'
          and path.parent.name == member['source_case_id']
          and path.resolve(strict=True) == path, 'Historical source manifest redirect')
    check(sha(path) == historical['sha256'], 'Historical source manifest SHA drift')
    native = json.loads(path.read_text())
    check(native['case_id'] == member['source_case_id'] and native['split'] == member['split']
          and native['geometry'] == 'native', 'Historical source manifest identity drift')
    check(all(native['source_sha256'][mod] == file['sha256']
              for mod, file in entry['files'].items()), 'Historical original source SHA mismatch')
    return historical


def row_key(row):
    return tuple(sorted((key, str(value)) for key, value in row.items()
                        if key != 'patient_n_crops'))


def modality_rows(rows, modality):
    check(modality in MODS, 'Unknown modality')
    selected = [dict(row) for row in rows if modality != 't2w' or int(row['t2w_valid']) == 1]
    counts = collections.Counter(row['patient_id'] for row in selected)
    for row in selected:
        row['patient_n_crops'] = counts[row['patient_id']]
    return selected


def data_identity(modality):
    check(modality in MODS, 'Unknown modality')
    package_sha = verify_package()
    cfg = json.loads((ROOT / 'config/RUN.json').read_text())
    check(cfg['source_kind'] == SOURCE_KIND, 'Not an original-source experiment')
    check(cfg['training']['use_all_rows'] is False, 'Training validation rows forbidden')
    check(cfg['training']['normalization'] == 'nnunet' and cfg['training']['crop_size'] == 64,
          'Preprocessed intensity/crop contract mismatch')
    complete_path = ROOT / 'dataset/COMPLETE.ok'
    complete = json.loads(complete_path.read_text())
    check(not (ROOT / 'dataset/FAILED.json').exists(), 'Failed dataset')
    for key, value in {**META, 'schema': SCHEMA, 'status': 'complete', 'source_kind': SOURCE_KIND,
                       'loss_policy': LOSS_POLICY, 'label_resampling_algorithm': LABEL_ALGORITHM,
                       'support_contract': SUPPORT_CONTRACT, 'quality_review': 'not_run',
                       'manual_review': 'not_run', 'package_sha256': package_sha}.items():
        check(complete.get(key) == value, 'Dataset contract drift: ' + key)
    selected = {r['source_case_id']: r for r in members()}
    eligible = {r['source_case_id']: r for r in members(modality)}
    records = source_records()
    verify_archived_headers(records)
    source_config_sha = sha(ROOT / 'config/SOURCE_RECORDS.json')
    check(source_config_sha == cfg['source_records_sha256'] == complete['source_records_sha256'],
          'Source records config SHA drift')
    check(complete['cases'] == len(selected) == sum(COUNTS['t1c'].values()), 'Case count drift')
    check(complete['split_counts'] == COUNTS['t1c'] and complete['modality_counts'] == COUNTS,
          'Modality membership counts drift')
    check(set(records) == set(selected) and set(eligible).issubset(selected), 'Source membership drift')
    for key in ('case_manifests', 'index_sha256', 'case_records_sha256', 'case_locations'):
        check(set(complete[key]) == set(selected), 'Incomplete dataset metadata: ' + key)
    report_path = ROOT / 'dataset/PREPROCESS_REPORT.json'
    check(sha(report_path) == complete['report_sha256'], 'Preprocessing report SHA drift')
    report = json.loads(report_path.read_text())
    check(report['status'] == 'complete' and not report['errors']
          and report['cases_ok'] == report['cases_processed'] == len(selected), 'Preprocessing errors/count drift')
    check(report['loss_policy'] == LOSS_POLICY and report['label_resampling_algorithm'] == LABEL_ALGORITHM,
          'Report policy drift')
    all_rows, warnings, consumed, source_hashes, historical_hashes = [], [], {}, {}, {}
    for case in sorted(selected):
        member, entry = selected[case], records[case]
        validate_record(member, entry)
        entry_sha = record_sha(entry)
        check(complete['case_records_sha256'][case] == entry_sha, 'Canonical case source record drift')
        directory = ROOT / 'dataset/cases' / case
        check(directory.is_dir() and not directory.is_symlink()
              and directory.resolve() == directory, 'Dataset case directory redirect')
        check(complete['case_locations'][case] == str(directory), 'Case location drift')
        marker_path = directory / 'preprocess_complete.json'
        check(not marker_path.is_symlink() and sha(marker_path) == complete['case_manifests'][case],
              'Case marker SHA drift')
        marker = json.loads(marker_path.read_text())
        for key, value in {**META, 'schema': SCHEMA, 'status': 'complete', 'case_id': case,
                           'source_kind': SOURCE_KIND, 'source_records_sha256': source_config_sha,
                           'source_record_sha256': entry_sha, 'split': member['split'],
                           'patient_group': member['patient_group'], 't2w_status': member['t2w_status'],
                           'source_files': entry['files'], 'quality_review': 'not_run',
                           'manual_review': 'not_run'}.items():
            check(marker.get(key) == value, 'Case marker identity drift: ' + key)
        historical = verify_historical_source(member, entry)
        check(marker.get('historical_authentic_source_manifest') == historical, 'Historical provenance drift')
        if historical:
            historical_hashes[case] = historical
        contract = marker['contract']
        for key, value in {'grid_version': 'physical_spacing_v2', 'target_spacing': [1., 1., 1.],
                           'loss_policy': LOSS_POLICY, 'label_resampling_algorithm': LABEL_ALGORITHM,
                           'support_contract': SUPPORT_CONTRACT, 'normalization': NORMALIZATION,
                           'mask_algorithm': MASK_ALGORITHM, 'local_label_exception': None}.items():
            check(contract.get(key) == value, 'Spatial policy drift: ' + key)
        check(contract['bbox'] == [[0, n] for n in contract['original_shape']], 'Full-FOV contract drift')
        import numpy as np
        from src.preprocess.nnunet_like import physical_grid
        from scripts.preprocess_dataset import new_affine
        native_seg = entry['files']['seg']
        check(contract['original_shape'] == native_seg['shape']
              and np.allclose(contract['original_affine'], native_seg['affine'], rtol=0, atol=1e-5),
              'Recorded native geometry differs from frozen original')
        expected_shape, _, _ = physical_grid(tuple(native_seg['shape']), native_seg['spacing'], (1., 1., 1.))
        expected_affine, _ = new_affine(np.asarray(native_seg['affine']), contract['bbox'],
                                       tuple(native_seg['shape']), expected_shape, native_seg['spacing'], (1., 1., 1.))
        check(contract['shape'] == list(expected_shape)
              and np.allclose(contract['affine'], expected_affine, rtol=0, atol=1e-5),
              'Recorded target geometry differs from physical grid')
        lost = [str(k) for k in range(1, 5) if contract['label_native'][str(k)]['voxels']
                and not contract['label_target_1mm'][str(k)]['voxels']]
        check(contract['lost_classes'] == lost and contract['resolution_warning'] is bool(lost),
              'Quantization warning drift')
        if lost:
            warnings.append(dict(case=case, lost_classes=lost))
        spatial_path = directory / 'spatial_mapping.json'
        check(not spatial_path.is_symlink() and sha(spatial_path) == marker['spatial_mapping_sha256'],
              'Spatial mapping SHA drift')
        spatial = json.loads(spatial_path.read_text())
        check(all(spatial.get(k) == v for k, v in contract.items()), 'Spatial mapping content drift')
        expected_outputs = {case + '-' + mod + '.nii.gz' for mod in (*available_mods(member), 'seg', 'brainmask')}
        check(set(marker['outputs']) == expected_outputs, 'Forbidden or missing output modality')
        brainmask = directory / (case + '-brainmask.nii.gz')
        check(not brainmask.is_symlink() and brainmask.resolve(strict=True) == brainmask
              and sha(brainmask) == marker['outputs'][brainmask.name], 'Brainmask SHA/target drift')
        index_path = ROOT / 'dataset/indices' / (case + '.json')
        check(not index_path.is_symlink() and sha(index_path) == complete['index_sha256'][case], 'Index SHA drift')
        index = json.loads(index_path.read_text())
        check(index['case_id'] == case and index['marker_sha256'] == complete['case_manifests'][case],
              'Index marker identity drift')
        for row in index['rows']:
            check(row['source_case_id'] == case and row['split'] == member['split']
                  and row['patient_group'] == member['patient_group']
                  and row['patient_id'] == case[-9:], 'Index row membership drift')
            check(int(row['t2w_valid']) == int('t2w' in available_mods(member)), 'Row T2W validity drift')
            for mod in MODS:
                expected = str(directory / (case + '-' + mod + '.nii.gz')) if mod in available_mods(member) else ''
                check(row['scan_' + mod] == expected, 'Index image mapping drift')
            check(row['label'] == str(directory / (case + '-seg.nii.gz')), 'Index label mapping drift')
        all_rows.extend(index['rows'])
        # All cases bind metadata; only the eligible modality and segmentation are opened.
        if case in eligible:
            for mod in (modality, 'seg'):
                file = entry['files'][mod]
                verify_source_file(file)
                source_hashes[file['path']] = dict(file)
                output = directory / (case + '-' + mod + '.nii.gz')
                check(not output.is_symlink() and output.resolve(strict=True) == output,
                      'Undeclared output reference')
                check(sha(output) == marker['outputs'][output.name], 'Output SHA drift: ' + str(output))
                import nibabel as nib
                image = nib.load(str(output))
                check(list(image.shape) == list(expected_shape)
                      and np.allclose(image.affine, expected_affine, rtol=0, atol=1e-5)
                      and np.allclose(image.header.get_zooms()[:3], (1., 1., 1.), rtol=0, atol=1e-5),
                      'Actual output header geometry drift')
                if member['split'] == 'train' and index['rows']:
                    consumed[str(output)] = marker['outputs'][output.name]
    check(report['warnings'] == complete['resolution_warnings'] == warnings, 'Warning report identity drift')
    check(complete['cases_without_lesion_rows'] == sorted(set(selected) - {r['source_case_id'] for r in all_rows}),
          'No-lesion case report drift')
    check(set(complete['csv_sha256']) == set(complete['csv_rows']) == set(MODS), 'Incomplete modality CSV metadata')
    csv_path = ROOT / 'dataset' / ('lesions_' + modality + '.csv')
    check(not csv_path.is_symlink() and sha(csv_path) == complete['csv_sha256'][modality], 'Modality CSV SHA drift')
    with csv_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    expected_rows = modality_rows(all_rows, modality)
    check(collections.Counter(map(row_key, rows)) == collections.Counter(map(row_key, expected_rows)),
          'Modality CSV/index identity drift')
    check(complete['csv_rows'][modality] == dict(total=len(rows), train=sum(r['split'] == 'train' for r in rows),
                                               val=sum(r['split'] == 'val' for r in rows)), 'CSV row counts drift')
    counts = collections.Counter(r['patient_id'] for r in rows)
    check(all(int(r['patient_n_crops']) == counts[r['patient_id']] for r in rows), 'Modality crop count drift')
    check(all(r['source_case_id'] in eligible and r['split'] in ('train', 'val') for r in rows), 'Forbidden CSV case')
    check(any(r['split'] == 'train' for r in rows), 'No training lesion rows')
    return dict(schema=SCHEMA, source_kind=SOURCE_KIND, modality=modality, selection='train_only',
                package_sha256=package_sha, dataset_complete_sha256=sha(complete_path),
                source_records_sha256=source_config_sha, case_records_sha256=complete['case_records_sha256'],
                membership_sha256={name: sha(ROOT / 'plan' / name) for name in
                                   ('master_membership.csv', 'realonly_membership.csv')},
                case_manifests=complete['case_manifests'], index_sha256=complete['index_sha256'],
                csv_sha256=complete['csv_sha256'][modality], csv_rows=complete['csv_rows'][modality],
                modality_counts=COUNTS[modality], report_sha256=complete['report_sha256'],
                resolution_warnings=warnings, source_files=source_hashes,
                historical_authentic_source_manifests=historical_hashes, consumed=consumed,
                missing_t2w_historical_byte_baseline='unavailable_new_source_record_sha_frozen',
                loss_policy=LOSS_POLICY, label_resampling_algorithm=LABEL_ALGORITHM,
                support_contract=SUPPORT_CONTRACT, quality_review='not_run', manual_review='not_run', **META)
