"""Original real images, full-FOV 1mm geometry and modality-specific lesion CSVs."""
import collections
import concurrent.futures
import csv
import json
import os
import sys
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes

from common import META, ROOT, now, sha, verify_package, write
from contracts import (COUNTS, LABEL_ALGORITHM, LOSS_POLICY, MODS, SOURCE_KIND,
                       SUPPORT_CONTRACT, available_mods, members, record_sha,
                       source_records)
from identity import (MASK_ALGORITHM, NORMALIZATION, SCHEMA, check, modality_rows,
                      validate_record, verify_archived_headers, verify_historical_source,
                      verify_recorded_affines, verify_source_file)

sys.path.insert(0, str(ROOT / 'runtime/upstream_source'))
from src.preprocess.nnunet_like import GRID_VERSION, physical_grid, resample_physical
from scripts.preprocess_dataset import corner_shift_mm, new_affine, save_atomic
from src.train.csv_creator import (check_row_in_loader_axes, connected_component_analysis,
                                   merge_nearby_lesions, tile_oversized_lesion)


def load_header(file, reference=None):
    path, target = Path(file['path']), Path(file['resolved_path'])
    check(path.resolve(strict=True) == target and target.resolve(strict=True) == target,
          'Source realpath drift: ' + str(path))
    check(target.is_file() and target.stat().st_size == int(file['size']), 'Source size drift')
    image = nib.load(str(target))
    check(len(image.shape) == 3 and np.isfinite(image.affine).all(), 'Invalid source geometry')
    spacing = np.asarray(image.header.get_zooms()[:3], float)
    check(np.isfinite(spacing).all() and (spacing > 0).all(), 'Invalid source spacing')
    check(list(image.shape) == file['shape']
          and np.allclose(spacing, file['spacing'], rtol=0, atol=1e-5)
          and np.allclose(image.affine, file['affine'], rtol=0, atol=1e-5), 'Source header drift')
    check(np.allclose(np.linalg.norm(image.affine[:3, :3], axis=0), spacing, rtol=0, atol=1e-4),
          'Header spacing/affine mismatch')
    if reference is not None:
        check(image.shape == reference.shape
              and corner_shift_mm(image.affine, reference.affine, reference.shape) <= .1,
              'Native geometry mismatch: ' + str(path))
    return image


def load_volume(file, reference, label=False):
    image = load_header(file, reference)
    data = image.get_fdata(dtype=np.float32)
    check(np.isfinite(data).all(), 'Nonfinite source volume: ' + file['path'])
    if label:
        check(np.equal(data, np.rint(data)).all() and np.isin(data, [0, 1, 2, 3, 4]).all(),
              'Illegal source labels: ' + file['path'])
    return data


def common_mask(arrays):
    mask = np.zeros(arrays[0].shape, dtype=bool)
    for image in arrays:
        low, high = np.percentile(image, [0, 100])
        low = max(0., float(low))
        if high > low:
            mask |= np.clip(image, low, high) > low
    mask = binary_fill_holes(mask)
    check(mask.any(), 'Empty authentic common mask')
    return mask


def normalize(image, mask):
    out = np.zeros(image.shape, dtype=np.float32)
    values = image[mask]
    out[mask] = (values - values.mean()) / max(float(values.std()), 1e-8)
    return out


def label_summary(label, spacing):
    counts = np.bincount(label.ravel().astype(np.int64), minlength=5)
    return {str(k): dict(voxels=int(counts[k]), volume_mm3=float(counts[k] * np.prod(spacing)))
            for k in range(5)}


def loss_record(native_counts, target_counts):
    lost = [str(k) for k in range(1, 5) if native_counts[str(k)]['voxels']
            and not target_counts[str(k)]['voxels']]
    return dict(loss_policy=LOSS_POLICY, label_resampling_algorithm=LABEL_ALGORITHM,
                lost_classes=lost, resolution_warning=bool(lost), local_label_exception=None,
                native_detail_preservation='not_guaranteed_at_target_resolution')


def build_rows(case, destination, out_label, member):
    binary = out_label > 0
    components = merge_nearby_lesions(connected_component_analysis(binary), merge_dist=16, crop_size=64)
    rows = []
    for idx, cc in enumerate(components):
        b = cc['bbox']
        oversized = any(b[i + 1] - b[i] > 64 for i in (0, 2, 4))
        tiles = list(tile_oversized_lesion(cc, binary, 64)) if oversized else [cc]
        for ti, tile in enumerate(tiles):
            b = tile['bbox']
            check_row_in_loader_axes(binary, b, tile['n_voxels'], case)
            row = dict(patient_id=case[-9:], source_case_id=case, patient_group=member['patient_group'],
                       lesion_id=f'{case}_cc{idx}_t{ti}', label=str(destination / (case + '-seg.nii.gz')),
                       n_voxels=int(tile['n_voxels']), split=member['split'],
                       t2w_valid=int('t2w' in available_mods(member)))
            row.update({'scan_' + mod: str(destination / (case + '-' + mod + '.nii.gz'))
                        if mod in available_mods(member) else '' for mod in MODS})
            for ax, axis in enumerate('xyz'):
                row['center_' + axis] = int(tile['centroid'][ax])
                row[axis + '_extreme_min'], row[axis + '_extreme_max'] = int(b[ax * 2]), int(b[ax * 2 + 1])
                row[axis + '_size'] = int(b[ax * 2 + 1] - b[ax * 2])
            rows.append(row)
    return rows


def estimate(cfg):
    """Only source headers/stat: no voxel arrays, source SHA sweep or missing T2W probe."""
    check(cfg['source_kind'] == SOURCE_KIND, 'Not an original-source experiment')
    check(int(cfg['preprocess_workers']) > 0, 'Invalid preprocessing worker count')
    selected, records = members(), source_records()
    compressed_bound, data_bound, temporary, errors = 0, 0, 0, []
    for member in selected:
        case = member['source_case_id']
        try:
            entry = records[case]
            validate_record(member, entry)
            ref = load_header(entry['files']['seg'])
            for mod in available_mods(member):
                load_header(entry['files'][mod], ref)
            shape, _, _ = physical_grid(ref.shape, ref.header.get_zooms()[:3], (1., 1., 1.))
            n, native_n = int(np.prod(shape)), int(np.prod(ref.shape))
            bytes_per_voxel = 4 * len(available_mods(member)) + 2
            data_bound += n * bytes_per_voxel
            # Include native three-modality mask construction, interpolation and CC temporaries.
            temporary = max(temporary, (n + native_n) * 128)
            for width in (*([4] * len(available_mods(member))), 1, 1):
                payload = n * width + 4096
                compressed_bound += payload + (payload >> 12) + (payload >> 14) + (payload >> 25) + 31
        except Exception as error:
            errors.append(dict(case=case, error=str(error), traceback=traceback.format_exc()))
    temporary *= int(cfg['preprocess_workers'])
    checkpoint_reserve, metadata_reserve = 12 * 1024**3, 1024**3
    required = compressed_bound + temporary + checkpoint_reserve + metadata_reserve + 4 * 1024**3
    disk = os.statvfs(ROOT)
    return dict(cases=len(selected), modality_counts=COUNTS, source_kind=SOURCE_KIND,
                data_upper_bytes=data_bound, compressed_worst_bound_bytes=compressed_bound,
                temporary_upper_bytes=temporary, checkpoint_reserve_bytes=checkpoint_reserve,
                metadata_reserve_bytes=metadata_reserve, required_bytes=required,
                available_bytes=disk.f_bavail * disk.f_frsize, errors=errors,
                capacity_pass=not errors and required <= disk.f_bavail * disk.f_frsize,
                method='available-source-headers_full_FOV_zlib_bound_native_target_workspaces')


def process(job):
    cfg, member, entry, source_config_sha = job
    case = member['source_case_id']
    check(cfg['source_kind'] == SOURCE_KIND, 'Not an original-source experiment')
    validate_record(member, entry)
    verify_recorded_affines(entry)
    historical = verify_historical_source(member, entry)
    for file in entry['files'].values():
        verify_source_file(file)
    ref = load_header(entry['files']['seg'])
    real = [load_volume(entry['files'][mod], ref) for mod in ('t1n', 't1c', 't2f')]
    mask = common_mask(real)
    del real
    label = load_volume(entry['files']['seg'], ref, label=True).astype(np.uint8)
    spacing = np.asarray(ref.header.get_zooms()[:3], float)
    bbox = [[0, int(n)] for n in label.shape]
    shape, factor, offset = physical_grid(label.shape, spacing, (1., 1., 1.))
    affine, _ = new_affine(ref.affine, bbox, label.shape, shape, spacing, (1., 1., 1.))
    out_mask = resample_physical(mask[None].astype(np.uint8), shape, factor, offset,
                                 spacing, (1., 1., 1.), is_seg=True)[0] > 0
    out_label = resample_physical(label[None], shape, factor, offset, spacing, (1., 1., 1.), is_seg=True)[0]
    check(np.isin(out_label, [0, 1, 2, 3, 4]).all(), 'Illegal resampled label')
    check(np.allclose(np.linalg.norm(affine[:3, :3], axis=0), 1, atol=1e-5), 'Not a 1mm output grid')
    native_counts, target_counts = label_summary(label, spacing), label_summary(out_label, (1., 1., 1.))
    destination = ROOT / 'dataset/cases' / case
    destination.mkdir(parents=True, exist_ok=False)
    outputs, statistics = {}, {}
    for mod in available_mods(member):
        image = load_volume(entry['files'][mod], ref)
        values = image[mask]
        statistics[mod] = dict(mean=float(values.mean()), std=float(values.std()),
                               effective_std=max(float(values.std()), 1e-8), mask_voxels=int(values.size))
        normalized = normalize(image, mask)
        del values, image
        output = resample_physical(normalized[None], shape, factor, offset, spacing, (1., 1., 1.))[0]
        output[~out_mask] = 0
        check(np.isfinite(output).all(), 'Nonfinite output')
        path = destination / (case + '-' + mod + '.nii.gz')
        save_atomic(nib.Nifti1Image(output.astype(np.float32), affine), str(path))
        outputs[path.name] = sha(path)
    for name, array in (('seg', out_label.astype(np.uint8)), ('brainmask', out_mask.astype(np.uint8))):
        path = destination / (case + '-' + name + '.nii.gz')
        save_atomic(nib.Nifti1Image(array, affine), str(path))
        outputs[path.name] = sha(path)
    info = dict(grid_version=GRID_VERSION, original_shape=list(ref.shape), original_affine=ref.affine.tolist(),
                bbox=bbox, shape=[int(v) for v in shape], affine=affine.tolist(), target_spacing=[1., 1., 1.],
                mask_algorithm=MASK_ALGORITHM, normalization=NORMALIZATION, support_contract=SUPPORT_CONTRACT,
                geometric_domain='full_native_FOV_not_claimed_as_tissue_support',
                native_label_outside_signal_voxels=int(np.count_nonzero((label > 0) & ~mask)),
                target_label_outside_signal_voxels=int(np.count_nonzero((out_label > 0) & ~out_mask)),
                label_native=native_counts, label_target_1mm=target_counts,
                normalization_statistics=statistics, **loss_record(native_counts, target_counts))
    # Recheck source bytes after reading so in-place drift cannot acquire a complete marker.
    for file in entry['files'].values():
        verify_source_file(file)
    verify_historical_source(member, entry)
    write(destination / 'spatial_mapping.json', {**META, **info})
    marker = {**META, 'schema': SCHEMA, 'status': 'complete', 'case_id': case, 'source_kind': SOURCE_KIND,
              'source_records_sha256': source_config_sha, 'source_record_sha256': record_sha(entry),
              'split': member['split'], 'patient_group': member['patient_group'], 't2w_status': member['t2w_status'],
              'source_files': entry['files'], 'historical_authentic_source_manifest': historical,
              'contract': info, 'outputs': outputs, 'spatial_mapping_sha256': sha(destination / 'spatial_mapping.json'),
              'quality_review': 'not_run', 'manual_review': 'not_run'}
    write(destination / 'preprocess_complete.json', marker)
    marker_sha = sha(destination / 'preprocess_complete.json')
    rows = build_rows(case, destination, out_label, member)
    index_path = ROOT / 'dataset/indices' / (case + '.json')
    write(index_path, dict(case_id=case, marker_sha256=marker_sha, rows=rows))
    return dict(case=case, split=member['split'], rows=rows, marker_sha256=marker_sha,
                index_sha256=sha(index_path), source_record_sha256=record_sha(entry),
                directory=str(destination), lost_classes=info['lost_classes'])


def process_record(job):
    try:
        return dict(ok=True, result=process(job))
    except Exception as error:
        return dict(ok=False, case=job[1]['source_case_id'], error=str(error), traceback=traceback.format_exc())


def finish_dataset(results, errors, source_config_sha, package_sha):
    dataset = ROOT / 'dataset'
    results = sorted(results, key=lambda result: result['case'])
    warnings = [dict(case=r['case'], lost_classes=r['lost_classes']) for r in results if r['lost_classes']]
    report = {**META, 'status': 'errors' if errors else 'complete', 'cases_processed': len(results) + len(errors),
              'cases_ok': len(results), 'errors': errors, 'warnings': warnings,
              'loss_policy': LOSS_POLICY, 'label_resampling_algorithm': LABEL_ALGORITHM,
              'quality_review': 'not_run', 'manual_review': 'not_run'}
    write(dataset / 'PREPROCESS_REPORT.json', report)
    check(not errors, f'{len(errors)} source/preprocessing errors; no dataset COMPLETE')
    expected = {r['source_case_id'] for r in members()}
    check(len(results) == len(expected) and {r['case'] for r in results} == expected, 'Incomplete preprocessing membership')
    rows = [row for result in results for row in result['rows']]
    check(rows, 'No lesion rows')
    csv_hashes, csv_counts = {}, {}
    for mod in MODS:
        selected = modality_rows(rows, mod)
        check(any(r['split'] == 'train' for r in selected), 'No training rows for ' + mod)
        csv_path = dataset / ('lesions_' + mod + '.csv')
        with csv_path.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) + ['patient_n_crops'])
            writer.writeheader()
            writer.writerows(selected)
            stream.flush()
            os.fsync(stream.fileno())
        csv_hashes[mod] = sha(csv_path)
        csv_counts[mod] = dict(total=len(selected), train=sum(r['split'] == 'train' for r in selected),
                               val=sum(r['split'] == 'val' for r in selected))
    check(sha(ROOT / 'config/SOURCE_RECORDS.json') == source_config_sha, 'Source config drift during preprocessing')
    complete = {**META, 'schema': SCHEMA, 'status': 'complete', 'source_kind': SOURCE_KIND,
                'cases': len(results), 'split_counts': dict(collections.Counter(r['split'] for r in results)),
                'modality_counts': COUNTS, 'csv_sha256': csv_hashes, 'csv_rows': csv_counts,
                'case_manifests': {r['case']: r['marker_sha256'] for r in results},
                'index_sha256': {r['case']: r['index_sha256'] for r in results},
                'case_records_sha256': {r['case']: r['source_record_sha256'] for r in results},
                'case_locations': {r['case']: r['directory'] for r in results},
                'source_records_sha256': source_config_sha, 'package_sha256': package_sha,
                'report_sha256': sha(dataset / 'PREPROCESS_REPORT.json'), 'resolution_warnings': warnings,
                'cases_without_lesion_rows': sorted(r['case'] for r in results if not r['rows']),
                'support_contract': SUPPORT_CONTRACT, 'loss_policy': LOSS_POLICY,
                'label_resampling_algorithm': LABEL_ALGORITHM, 'quality_review': 'not_run',
                'manual_review': 'not_run', 'finished_utc': now()}
    check(complete['split_counts'] == COUNTS['t1c'], 'Preprocessed split counts drift')
    write(dataset / 'COMPLETE.ok', complete)
    return complete


def main():
    cfg = json.loads((ROOT / 'config/RUN.json').read_text())
    package_sha = verify_package()
    dataset = ROOT / 'dataset'
    dataset.mkdir(exist_ok=False)
    write(dataset / 'STARTED.json', {**META, 'utc': now(), 'package_sha256': package_sha,
                                    'job_id': os.environ.get('SLURM_JOB_ID')})
    try:
        storage = estimate(cfg)
        write(dataset / 'STORAGE_ESTIMATE.json', {**META, **storage})
        if storage['errors']:
            finish_dataset([], storage['errors'], '', package_sha)
        check(storage['capacity_pass'], 'Insufficient measured input/checkpoint capacity')
        records, source_config_sha = source_records(), sha(ROOT / 'config/SOURCE_RECORDS.json')
        verify_archived_headers(records)
        jobs = [(cfg, member, records[member['source_case_id']], source_config_sha) for member in members()]
        results, errors = [], []
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(cfg['preprocess_workers'])) as pool:
            for record in pool.map(process_record, jobs, chunksize=1):
                if record['ok']:
                    result = record['result']
                    results.append(result)
                    print(json.dumps(dict(case=result['case'], cases_completed=len(results), rows=len(result['rows']),
                                          lost_classes=result['lost_classes'])), flush=True)
                else:
                    errors.append(record)
                    write(dataset / 'errors' / (record['case'] + '.json'), {**META, **record})
                    print(json.dumps(record), flush=True)
        complete = finish_dataset(results, errors, source_config_sha, package_sha)
        print(json.dumps(dict(status=complete['status'], cases=complete['cases'], csv_rows=complete['csv_rows'])), flush=True)
    except BaseException as error:
        write(dataset / 'FAILED.json', {**META, 'utc': now(), 'message': str(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
