"""Full-FOV strict 1mm inputs; signal statistics and geometric domain are distinct."""
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
from scipy.ndimage import affine_transform, binary_fill_holes

from common import META, ROOT, now, sha, verify_package, write
sys.path.insert(0, str(ROOT / 'runtime/upstream_source'))
from src.preprocess.nnunet_like import GRID_VERSION, physical_grid, resample_physical
from scripts.preprocess_dataset import corner_shift_mm, new_affine, save_atomic
from src.train.csv_creator import (connected_component_analysis, merge_nearby_lesions,
                                   tile_oversized_lesion, check_row_in_loader_axes)

MODS = ('t1c', 't1n', 't2w', 't2f')
CHECKPOINT = '5d86a0b3cbc5df34d64d5a0ff376eae453634a609bd24dec1a1191fde800b13b'


def reuse_candidates(cfg):
    plan = json.loads((ROOT / 'plan/REUSE_CANDIDATES.json').read_text())
    assert plan['parent'] == cfg['reuse_parent']
    assert plan['parent_package_sha256'] == cfg['reuse_parent_package_sha256']
    parent = Path(plan['parent'])
    assert sha(parent / 'plan/PACKAGE_MANIFEST.json') == plan['parent_package_sha256']
    package = json.loads((parent / 'plan/PACKAGE_MANIFEST.json').read_text())
    assert all(sha(parent / name) == digest for name, digest in package['files'].items())
    assert (parent / 'dataset/FAILED.json').is_file() and not (parent / 'dataset/COMPLETE.ok').exists()
    candidates = {r['case_id']: r for r in plan['candidates']}
    assert len(candidates) == len(plan['candidates']) == 471
    return candidates


def apply_local_exception(case, input_sha, label, out_label, shape, factor, offset, spacing, exception):
    if case != exception['case_id']:
        return out_label, None
    assert input_sha == exception['source_seg_sha256'], 'Exception source label drift'
    assert list(label.shape) == exception['native_shape']
    assert np.allclose(spacing, exception['native_spacing_mm'], rtol=0, atol=1e-7)
    assert list(shape) == exception['target_shape']
    assert int(np.count_nonzero(label == 4)) == exception['native_class4_voxels']
    assert {str(k): int(np.count_nonzero(out_label == k)) for k in range(5)} == exception['expected_default_label_counts']
    nearest = affine_transform(label, np.diag(factor), offset, output_shape=tuple(shape), order=0, mode='nearest')
    observed = np.argwhere(nearest == 4).tolist()
    assert observed == exception['target_ijk'], 'Local nearest-supported targets drift'
    assert len(observed) == 4 and exception['allowed_previous_label'] == 0 and exception['restore_label'] == 4
    fixed = out_label.copy()
    for point in observed:
        assert out_label[tuple(point)] == 0, 'Local repair cannot overwrite another foreground class'
        fixed[tuple(point)] = 4
    assert int(np.count_nonzero(fixed != out_label)) == 4
    return fixed, dict(schema=exception['schema'], case_id=case, restored_target_ijk=observed,
                       original_class4_voxels=19, target_class4_voxels=4,
                       all_native_components_preserved=False, global_algorithm_unchanged=True)


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
                       n_voxels=int(tile['n_voxels']), split=member['split'], t2w_valid=1)
            row.update({'scan_' + m: str(destination / (case + '-' + m + '.nii.gz')) for m in MODS})
            for ax, a in enumerate('xyz'):
                row['center_' + a] = int(tile['centroid'][ax])
                row[a + '_extreme_min'], row[a + '_extreme_max'] = int(b[ax * 2]), int(b[ax * 2 + 1])
                row[a + '_size'] = int(b[ax * 2 + 1] - b[ax * 2])
            rows.append(row)
    return rows


def reuse_case(cfg, member, candidate, native_manifest_sha):
    case = member['source_case_id']
    directory = Path(cfg['reuse_parent']) / 'dataset/cases' / case
    marker = directory / 'preprocess_complete.json'
    assert sha(marker) == candidate['marker_sha256'], 'Parent completion marker drift'
    info = json.loads(marker.read_text())
    assert info['case_id'] == case and info['status'] == 'complete' and info['split'] == member['split']
    assert info['patient_group'] == member['patient_group'] and info['native_case_manifest_sha256'] == native_manifest_sha
    contract = info['contract']
    assert contract['support_contract'] == 'signal_statistics_vs_full_native_FOV_geometry_v1'
    assert contract['grid_version'] == GRID_VERSION and contract['target_spacing'] == [1., 1., 1.]
    assert contract['normalization'] == 'whole_case_common_mask_zscore_before_resampling_zero_outside_support'
    assert contract['mask_algorithm'] == 'authentic_T1N_T1C_T2F_robust_0_100_nonzero_union_holefill'
    spatial = json.loads((directory / 'spatial_mapping.json').read_text())
    assert all(spatial[k] == v for k, v in contract.items())
    assert contract['bbox'] == [[0, int(n)] for n in contract['original_shape']]
    assert all(not contract['label_native'][str(k)]['voxels'] or contract['label_target_1mm'][str(k)]['voxels'] for k in range(1, 5))
    native_path = Path(cfg['recon']) / 'dataset/cases' / case / 'case_manifest.json'
    assert sha(native_path) == native_manifest_sha
    native = json.loads(native_path.read_text())
    assert all(h == native['files'][m]['sha256'] for m, h in info['input_sha256'].items())
    assert all(h == native['source_sha256'][m] for m, h in info['authentic_mask_source_sha256'].items())
    expected = {case + '-' + m + '.nii.gz' for m in (*MODS, 'seg', 'brainmask')}
    assert set(info['outputs']) == expected
    for name, digest in info['outputs'].items():
        assert (directory / name).is_file() and not (directory / name).is_symlink()
        assert sha(directory / name) == digest, 'Reused output SHA drift: ' + case + '/' + name
    label_image = nib.load(str(directory / (case + '-seg.nii.gz')))
    assert np.allclose(np.linalg.norm(label_image.affine[:3, :3], axis=0), 1, atol=1e-5)
    data = label_image.get_fdata(dtype=np.float32)
    assert np.isfinite(data).all() and np.isin(data, [0, 1, 2, 3, 4]).all()
    assert list(data.shape) == contract['shape']
    target = ROOT / 'dataset/cases' / case
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(directory, target_is_directory=True)
    rows = build_rows(case, target, data.astype(np.uint8), member)
    write(ROOT / 'dataset/indices' / (case + '.json'), dict(case_id=case, rows=rows, marker_sha256=sha(marker)))
    return dict(case=case, split=member['split'], rows=rows, outputs=info['outputs'],
                marker_sha256=sha(marker), labels=np.unique(data).tolist(), origin='readonly_parent_reference',
                resolved_directory=str(directory), local_exception=None)


def members():
    with (ROOT / 'plan/splits_final_train_val_test_membership.csv').open(newline='') as h:
        rows = list(csv.DictReader(h))
    assert collections.Counter(r['split'] for r in rows) == {'train': 823, 'val': 103, 'test': 104}
    assert len({r['source_case_id'] for r in rows}) == 1030
    groups = collections.defaultdict(set)
    for row in rows:
        groups[row['patient_group']].add(row['split'])
        assert row['t2w_status'] == 'authentic' and row['eligible_for_realonly'].lower() == 'true'
    assert all(len(s) == 1 for s in groups.values())
    return [r for r in rows if r['split'] != 'test']


def load_volume(path, reference, label=False):
    image = nib.load(str(path))
    if (not np.isfinite(image.affine).all() or not np.isfinite(reference.affine).all()
            or image.shape != reference.shape
            or corner_shift_mm(image.affine, reference.affine, reference.shape) > .1):
        raise ValueError('Native geometry mismatch: ' + str(path))
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError('Nonfinite volume: ' + str(path))
    if label and (not np.equal(data, np.rint(data)).all() or not np.isin(data, [0, 1, 2, 3, 4]).all()):
        raise ValueError('Illegal labels: ' + str(path))
    return data


def common_mask(arrays):
    mask = np.zeros(arrays[0].shape, dtype=bool)
    for image in arrays:
        low, high = np.percentile(image, [0, 100])
        low = max(0., float(low))
        if high > low:
            mask |= np.clip(image, low, high) > low
    mask = binary_fill_holes(mask)
    if not mask.any():
        raise ValueError('Empty authentic common mask')
    return mask


def normalize(image, mask):
    out = np.zeros(image.shape, dtype=np.float32)
    values = image[mask]
    out[mask] = (values - values.mean()) / max(float(values.std()), 1e-8)
    return out


def label_summary(label, spacing):
    counts = np.bincount(label.ravel().astype(np.int64), minlength=5)
    volume = float(np.prod(spacing))
    return {str(k): {'voxels': int(counts[k]), 'volume_mm3': float(counts[k] * volume)}
            for k in range(5)}


def estimate(cfg):
    # Uncompressed full-FOV arrays bound compressed cropped outputs without guessing ratios.
    voxels, largest, compressed_bound = 0, 0, 0
    reuse = reuse_candidates(cfg)
    for row in members():
        case = row['source_case_id']
        if case in reuse:
            marker = Path(cfg['reuse_parent']) / 'dataset/cases' / case / 'preprocess_complete.json'
            assert sha(marker) == reuse[case]['marker_sha256']
            continue
        image = nib.load(str(Path(cfg['recon']) / 'dataset/cases' / case / (case + '-seg.nii.gz')))
        shape, _, _ = physical_grid(image.shape, image.header.get_zooms()[:3], (1., 1., 1.))
        n = int(np.prod(shape))
        voxels += n
        largest = max(largest, n)
        for item_bytes in (4 * n, 4 * n, 4 * n, 4 * n, n, n):
            payload = item_bytes + 4096
            compressed_bound += payload + (payload >> 12) + (payload >> 14) + (payload >> 25) + 13 + 18
    data_bound = voxels * 18  # 4 float32 images, uint8 segmentation and common mask
    temporary = largest * 18 * cfg['preprocess_workers'] * 2
    checkpoint_reserve = 12 * 1024**3
    metadata_reserve = 1024**3
    required = compressed_bound + metadata_reserve + temporary + checkpoint_reserve + 4 * 1024**3
    disk = os.statvfs(ROOT)
    result = dict(data_upper_bytes=data_bound, temporary_upper_bytes=temporary,
                  checkpoint_reserve_bytes=checkpoint_reserve, required_bytes=required,
                  compressed_worst_bound_bytes=compressed_bound, metadata_reserve_bytes=metadata_reserve,
                  available_bytes=disk.f_bavail * disk.f_frsize, cases=926, reused_candidates=471, new_cases=455,
                  method='remaining455_full_FOV+zlib_compressBound+metadata+temporary+all_checkpoints+4GiB')
    if required > result['available_bytes']:
        raise RuntimeError('Insufficient measured input/checkpoint capacity: ' + json.dumps(result))
    return result


def process(job):
    cfg, member, source_row, manifest_sha = job
    case = member['source_case_id']
    if case in cfg.get('_reuse_candidates', {}):
        return reuse_case(cfg, member, cfg['_reuse_candidates'][case], manifest_sha)
    native = Path(cfg['recon']) / 'dataset/cases' / case
    cm_path = native / 'case_manifest.json'
    assert sha(cm_path) == manifest_sha
    cm = json.loads(cm_path.read_text())
    assert cm['case_id'] == case and cm['split'] == member['split'] == source_row['split']
    assert cm['checkpoint_sha256'] == CHECKPOINT and cm['geometry'] == 'native'
    ref = nib.load(str(native / (case + '-seg.nii.gz')))
    assert list(ref.shape) == cm['native_shape']
    inputs = {m: native / (case + '-' + m + '.nii.gz') for m in (*MODS, 'seg')}
    sources = {m: Path(cfg['source_code']) / 'data/input' / case / Path(source_row[m]).name for m in (*MODS, 'seg')}
    input_sha = {m: sha(p) for m, p in inputs.items()}
    assert all(input_sha[m] == cm['files'][m]['sha256'] for m in inputs)
    assert input_sha['seg'] == sha(sources['seg']) == cm['source_sha256']['seg']
    overlay = json.loads((ROOT / 'config/P0_SOURCE_MAPPING.json').read_text())['corrected_label_paths']
    if case in overlay:
        assert sha(overlay[case]) == input_sha['seg']
    real_sha = {}
    real = []
    for m in ('t1n', 't1c', 't2f'):
        real_sha[m] = sha(sources[m])
        assert real_sha[m] == cm['source_sha256'][m]
        real.append(load_volume(sources[m], ref))
    mask = common_mask(real)
    del real
    label = load_volume(inputs['seg'], ref, label=True).astype(np.uint8)
    native_signal_missing = int(np.count_nonzero((label > 0) & ~mask))
    # Full native FOV retains zero-valued label locations without inventing tissue support.
    bbox = [[0, int(n)] for n in label.shape]
    sl = tuple(slice(a, b) for a, b in bbox)
    cropped_mask = mask[sl]
    spacing = np.asarray(ref.header.get_zooms()[:3], float)
    cropped_shape = cropped_mask.shape
    shape, factor, offset = physical_grid(cropped_shape, spacing, (1., 1., 1.))
    affine, _ = new_affine(ref.affine, bbox, cropped_shape, shape, spacing, (1., 1., 1.))
    out_mask = resample_physical(cropped_mask[None].astype(np.uint8), shape, factor, offset, spacing, (1., 1., 1.), is_seg=True)[0] > 0
    out_label = resample_physical(label[sl][None], shape, factor, offset, spacing, (1., 1., 1.), is_seg=True)[0]
    exception = json.loads((ROOT / 'plan/LOCAL_LABEL_EXCEPTION.json').read_text()) if (ROOT / 'plan/LOCAL_LABEL_EXCEPTION.json').exists() else None
    local_fix = None
    if exception:
        out_label, local_fix = apply_local_exception(case, input_sha['seg'], label, out_label, shape, factor, offset, spacing, exception)
    assert np.isin(out_label, [0, 1, 2, 3, 4]).all()
    assert np.allclose(np.linalg.norm(affine[:3, :3], axis=0), 1, atol=1e-5)
    native_counts = label_summary(label, spacing)
    target_counts = label_summary(out_label, (1., 1., 1.))
    lost_classes = [k for k in ('1', '2', '3', '4')
                    if native_counts[k]['voxels'] and not target_counts[k]['voxels']]
    if lost_classes:
        raise ValueError(case + ': whole foreground classes lost during 1mm resampling: ' + str(lost_classes))
    resampled_signal_missing = int(np.count_nonzero((out_label > 0) & ~out_mask))
    destination = ROOT / 'dataset/cases' / case
    destination.mkdir(parents=True, exist_ok=False)
    outputs = {}
    statistics = {}
    for m in MODS:
        image = load_volume(inputs[m], ref)
        values = image[sl][cropped_mask]
        statistics[m] = {'mean': float(values.mean()), 'std': float(values.std()),
                         'effective_std': max(float(values.std()), 1e-8), 'mask_voxels': int(values.size)}
        normalized = normalize(image[sl], cropped_mask)
        del values
        del image
        output = resample_physical(normalized[None], shape, factor, offset, spacing, (1., 1., 1.))[0]
        output[~out_mask] = 0
        assert np.isfinite(output).all()
        path = destination / (case + '-' + m + '.nii.gz')
        save_atomic(nib.Nifti1Image(output, affine), str(path))
        outputs[path.name] = sha(path)
    for name, array in (('seg', out_label), ('brainmask', out_mask.astype(np.uint8))):
        path = destination / (case + '-' + name + '.nii.gz')
        save_atomic(nib.Nifti1Image(array, affine), str(path))
        outputs[path.name] = sha(path)
    info = dict(grid_version=GRID_VERSION, original_shape=list(ref.shape), original_affine=ref.affine.tolist(),
                bbox=bbox, shape=[int(v) for v in shape], affine=affine.tolist(), target_spacing=[1., 1., 1.],
                mask_algorithm='authentic_T1N_T1C_T2F_robust_0_100_nonzero_union_holefill',
                normalization='whole_case_common_mask_zscore_before_resampling_zero_outside_support',
                support_contract='signal_statistics_vs_full_native_FOV_geometry_v1',
                geometric_domain='full_native_FOV_not_claimed_as_tissue_support',
                original_signal_gate_pass=native_signal_missing == 0,
                native_label_outside_signal_voxels=native_signal_missing,
                target_label_outside_signal_voxels=resampled_signal_missing,
                label_native=native_counts, label_target_1mm=target_counts,
                local_label_exception=local_fix,
                normalization_statistics=statistics,
                independent_full_revalidation='waived_by_user_not_pass')
    write(destination / 'spatial_mapping.json', {**META, **info})
    write(destination / 'preprocess_complete.json', {**META, 'status': 'complete', 'case_id': case,
          'split': member['split'], 'patient_group': member['patient_group'], 'contract': info,
          'input_sha256': input_sha, 'authentic_mask_source_sha256': real_sha, 'outputs': outputs,
          'native_case_manifest_sha256': manifest_sha})
    rows = build_rows(case, destination, out_label, member)
    write(ROOT / 'dataset/indices' / (case + '.json'), dict(case_id=case, rows=rows,
          marker_sha256=sha(destination / 'preprocess_complete.json')))
    return dict(case=case, split=member['split'], rows=rows, outputs=outputs,
                marker_sha256=sha(destination / 'preprocess_complete.json'), labels=np.unique(out_label).tolist(),
                origin='new_preprocessing', resolved_directory=str(destination), local_exception=local_fix)


def main():
    cfg = json.loads((ROOT / 'config/RUN.json').read_text())
    package_sha = verify_package()
    dataset = ROOT / 'dataset'
    dataset.mkdir(exist_ok=False)
    write(dataset / 'STARTED.json', {**META, 'utc': now(), 'job_id': os.environ.get('SLURM_JOB_ID'), 'package_sha256': package_sha})
    try:
        storage = estimate(cfg)
        write(dataset / 'STORAGE_ESTIMATE.json', {**META, **storage})
        complete = json.loads((Path(cfg['recon']) / 'dataset/COMPLETE.ok').read_text())
        assert complete['status'] == 'pass' and complete['checkpoint_sha256'] == CHECKPOINT
        source_path = Path(cfg['source_code']) / 'data/data_csv.csv'
        assert sha(source_path) == cfg['source_csv_sha256']
        with source_path.open(newline='') as h:
            source = {r['id']: r for r in csv.DictReader(h)}
        cfg['_reuse_candidates'] = reuse_candidates(cfg)
        jobs = [(cfg, r, source[r['source_case_id']], complete['manifest_sha256'][r['source_case_id']]) for r in members()]
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg['preprocess_workers']) as pool:
            for result in pool.map(process, jobs, chunksize=1):
                results.append(result)
                print(json.dumps(dict(case=result['case'], cases_completed=len(results), rows=len(result['rows']))), flush=True)
        rows = [row for result in results for row in result.pop('rows')]
        counts = collections.Counter(r['patient_id'] for r in rows)
        for row in rows:
            row['patient_n_crops'] = counts[row['patient_id']]
        assert rows and {r['split'] for r in rows} == {'train', 'val'}
        csv_path = dataset / 'lesions.csv'
        with csv_path.open('x', newline='') as h:
            writer = csv.DictWriter(h, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            h.flush()
            os.fsync(h.fileno())
        result = {**META, 'status': 'complete', 'cases': len(results), 'split_counts': {'train': 823, 'val': 103},
                  'rows': len(rows), 'train_rows': sum(r['split'] == 'train' for r in rows),
                  'cases_without_lesion_rows': sorted(r['case'] for r in results if counts[r['case'][-9:]] == 0),
                  'csv_sha256': sha(csv_path), 'case_manifests': {r['case']: r['marker_sha256'] for r in results},
                  'cases_detail': results, 'package_sha256': package_sha, 'finished_utc': now(),
                  'independent_P0_acceptance': 'waived_by_user', 'manual_review': 'waived_by_user'}
        result.update(support_contract='signal_statistics_vs_full_native_FOV_geometry_v1',
                      STRICT_ORIGINAL_GATE=False, NONZERO_COMPARISON_GATE=False,
                      independent_full_revalidation='waived_by_user_not_pass',
                      historical_diagnostic_sha256=sha(ROOT / 'plan/R6_DIAGNOSTIC_COMPLETE.json'))
        assert len(results) == 926
        result.update(case_locations={r['case']: r['resolved_directory'] for r in results},
                      reused_cases=[r['case'] for r in results if r['origin'] == 'readonly_parent_reference'],
                      reuse_parent_package_sha256=cfg['reuse_parent_package_sha256'],
                      local_exception_sha256=sha(ROOT / 'plan/LOCAL_LABEL_EXCEPTION.json'))
        assert len(result['reused_cases']) == 471
        fixes = [r['local_exception'] for r in results if r['local_exception']]
        assert len(fixes) == 1 and fixes[0]['case_id'] == 'BraTS-MET-01110-002'
        write(dataset / 'COMPLETE.ok', result)
        print(json.dumps({k: v for k, v in result.items() if k not in ('cases_detail', 'case_manifests')}), flush=True)
    except BaseException as error:
        write(dataset / 'FAILED.json', {**META, 'utc': now(), 'message': str(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
