"""Independent P0 validation of native-space reconstruction; no quality selection."""

import argparse
import collections
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

import nibabel as nib
import numpy as np

META = {'artifact_status': 'experimental_unvalidated', 'operator_approved': False,
        'formal_gate_status': 'not_run_not_passed'}
MODALITIES = ('t1n', 't1c', 't2w', 't2f', 'seg')
CHECKPOINT = '5d86a0b3cbc5df34d64d5a0ff376eae453634a609bd24dec1a1191fde800b13b'
SOURCE_CSV = '0d1765e12cdad7d8a5df57a5d2c46c79414ed99a2015a4d8fed57cd4c4fc836b'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def exclusive(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def read_csv(path):
    with Path(path).open(newline='') as h:
        return list(csv.DictReader(h))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_volume(path, reference, label=False):
    image = nib.load(str(path))
    require(len(image.shape) == 3, f'not 3D: {path}')
    require(image.shape == reference.shape, f'shape mismatch: {path}')
    require(np.isfinite(image.affine).all() and abs(np.linalg.det(image.affine[:3, :3])) > 1e-12, f'invalid affine: {path}')
    require(np.allclose(image.affine, reference.affine, atol=1e-4, rtol=0), f'affine mismatch: {path}')
    require(np.allclose(nib.affines.voxel_sizes(image.affine), nib.affines.voxel_sizes(reference.affine), atol=1e-5, rtol=0), f'spacing mismatch: {path}')
    data = np.asarray(image.dataobj, dtype=np.float32)
    require(np.isfinite(data).all(), f'nonfinite: {path}')
    result = {'shape_xyz': list(image.shape), 'affine': image.affine.tolist(),
              'spacing_xyz': nib.affines.voxel_sizes(image.affine).tolist(), 'finite': True}
    if label:
        require(np.allclose(data, np.rint(data), atol=1e-6, rtol=0), f'noninteger label: {path}')
        values = [int(x) for x in np.unique(data)]
        require(set(values).issubset({0, 1, 2, 3, 4}), f'illegal label: {path}: {values}')
        result['label_values'] = values
    else:
        require(np.any(data), f'empty image: {path}')
    return result


def validate(args):
    root = args.experiment_root
    recon = args.reconstruction_root
    dataset = recon / 'dataset'
    auth = json.loads((root / 'plan/EXECUTION_AUTHORIZATION.json').read_text())
    require(auth['status'] == 'execution_authorized' and auth['operator_approved'] is False, 'authorization metadata')
    require(sha(root / 'plan/G1_EPOCH004_RECON_1MM_DIFFUSION_ECNU_PLAN_20261013_R1.md') == auth['approved_plan_sha256'], 'plan drift')
    membership_path = root / 'plan/splits_final_train_val_test_membership.csv'
    overlay_path = root / 'plan/corrected_label_overlay.csv'
    require(sha(membership_path) == auth['membership_sha256'], 'membership drift')
    require(sha(overlay_path) == auth['overlay_sha256'], 'overlay drift')
    members = read_csv(membership_path)
    require(len(members) == 1030, 'membership count')
    by_id = {r['source_case_id']: r for r in members}
    require(len(by_id) == len(members), 'duplicate cases')
    require(collections.Counter(r['split'] for r in members) == {'train': 823, 'val': 103, 'test': 104}, 'split count')
    group_splits = collections.defaultdict(set)
    for row in members:
        require(row['t2w_status'] == 'authentic' and row['eligible_for_realonly'].lower() == 'true', 'non-authentic member')
        group_splits[row['patient_group']].add(row['split'])
    require(all(len(v) == 1 for v in group_splits.values()), 'patient leakage')
    require(not (dataset / 'FAILED.json').exists(), 'reconstruction failed marker')
    complete = json.loads((dataset / 'COMPLETE.ok').read_text())
    validation = json.loads((dataset / 'DATASET_VALIDATION.json').read_text())
    require(complete == validation and complete['status'] == 'pass', 'reconstruction incomplete')
    require(complete['cases'] == 1030 and complete['images'] == 4120 and complete['labels'] == 1030, 'reconstruction counts')
    require(complete['checkpoint_sha256'] == CHECKPOINT, 'checkpoint identity')
    sacct = subprocess.run(['sacct', '-n', '-P', '-j', '3941042', '--format=JobID,State,ExitCode'], capture_output=True, text=True, check=True, timeout=30)
    require('3941042|COMPLETED|0:0' in sacct.stdout, 'upstream job not COMPLETED 0:0')
    source_csv_path = args.source_code / 'data/data_csv.csv'
    require(sha(source_csv_path) == SOURCE_CSV, 'source CSV drift')
    source_rows = read_csv(source_csv_path)
    sources = {r['id']: r for r in source_rows}
    require(len(sources) == len(source_rows) == 1030 and set(sources) == set(by_id), 'source cohort mismatch')
    require(set(complete['manifest_sha256']) == set(by_id), 'manifest cohort mismatch')
    case_dirs = {p.name for p in (dataset / 'cases').iterdir() if p.is_dir()}
    require(case_dirs == set(by_id), 'extra/missing/partial case directories')
    overlays = {r['case_id']: r for r in read_csv(overlay_path) if r['applied'].lower() == 'true'}
    source_mapping = json.loads((root / 'config/P0_SOURCE_MAPPING.json').read_text())
    corrected_paths = source_mapping['corrected_label_paths']
    require(set(corrected_paths) == set(overlays) & set(by_id), 'corrected label mapping incomplete')
    manifest_path = args.output / 'NATIVE_FILE_VALIDATION.jsonl'
    totals = {'cases': 0, 'images': 0, 'labels': 0, 'source_files': 0, 'corrected_labels': 0}
    with manifest_path.open('x') as stream:
        for case in sorted(by_id):
            member = by_id[case]
            source_row = sources[case]
            case_dir = dataset / 'cases' / case
            cm_path = case_dir / 'case_manifest.json'
            require(sha(cm_path) == complete['manifest_sha256'][case], f'case manifest drift: {case}')
            cm = json.loads(cm_path.read_text())
            require(cm['case_id'] == case and cm['split'] == member['split'] == source_row['split'], f'case split drift: {case}')
            require(cm['checkpoint_sha256'] == CHECKPOINT and cm['geometry'] == 'native', f'case contract: {case}')
            require(cm['normalization'] == 'G1_V3_robust_normalize_then_deployed_modality_postprocess' and cm['latent_sampling'] == 'posterior_mean', f'normalization drift: {case}')
            require(cm['artifact_status'] == 'experimental_unvalidated' and cm['operator_approved'] is False and cm['formal_gate_status'] == 'not_run_not_passed', f'case metadata: {case}')
            require(set(cm['files']) == set(MODALITIES) == set(cm['source_sha256']), f'case files: {case}')
            require({p.name for p in case_dir.glob('*.nii.gz')} == {f'{case}-{m}.nii.gz' for m in MODALITIES}, f'extra/missing NIfTI: {case}')
            original_paths = {m: args.source_code / 'data/input' / case / Path(source_row[m]).name for m in MODALITIES}
            require(all(p.is_file() for p in original_paths.values()), f'missing authentic source: {case}')
            reference = nib.load(str(original_paths['t1n']))
            require(list(reference.shape) == cm['native_shape'], f'native shape drift: {case}')
            label_sha = None
            for modality in MODALITIES:
                original = original_paths[modality]
                expected = cm['files'][modality]
                require(expected['name'] == f'{case}-{modality}.nii.gz', f'filename drift: {case}/{modality}')
                target = case_dir / expected['name']
                observed_sha = sha(target)
                source_sha = sha(original)
                require(observed_sha == expected['sha256'] and target.stat().st_size == expected['bytes'], f'output byte drift: {case}/{modality}')
                require(source_sha == cm['source_sha256'][modality], f'source byte drift: {case}/{modality}')
                source_geometry = validate_volume(original, reference, label=modality == 'seg')
                geometry = validate_volume(target, reference, label=modality == 'seg')
                if modality == 'seg':
                    require(observed_sha == source_sha, f'copied label differs: {case}')
                    require(geometry['label_values'] == cm['label_values'], f'label manifest differs: {case}')
                    label_sha = observed_sha
                record = {**META, 'case_id': case, 'patient_group': member['patient_group'], 'split': member['split'], 'modality': modality,
                          'path': str(target), 'sha256': observed_sha, 'bytes': target.stat().st_size,
                          'source_path': str(original), 'source_resolved_path': str(original.resolve()),
                          'source_sha256': source_sha, 'source_geometry': source_geometry, 'geometry': geometry,
                          'case_manifest_sha256': sha(cm_path), 'status': 'pass'}
                stream.write(json.dumps(record, sort_keys=True) + '\n')
                totals['labels' if modality == 'seg' else 'images'] += 1
                totals['source_files'] += 1
            if case in overlays:
                corrected = Path(corrected_paths[case])
                require(corrected.is_file(), f'authoritative corrected label unavailable: {case}: {corrected}')
                require(sha(corrected) == label_sha, f'G2 corrected label differs: {case}')
                totals['corrected_labels'] += 1
            totals['cases'] += 1
            stream.flush()
            print(json.dumps({'case': case, 'status': 'pass', 'cases_completed': totals['cases']}), flush=True)
        os.fsync(stream.fileno())
    require(totals['cases'] == 1030 and totals['images'] == 4120 and totals['labels'] == 1030 and totals['source_files'] == 5150, 'validated counts')
    return {**META, 'status': 'pass', 'stage': 'P0_integrity_only', **totals,
            'split_counts': {'train': 823, 'val': 103, 'test': 104},
            'patient_group_counts': dict(collections.Counter(next(iter(v)) for v in group_splits.values())),
            'file_manifest_sha256': sha(manifest_path), 'membership_sha256': sha(membership_path),
            'source_csv_sha256': SOURCE_CSV, 'checkpoint_sha256': CHECKPOINT,
            'quality_scoring_performed': False, 'locked104_quality_evaluation_performed': False,
            'finished_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'hostname': os.uname().nodename}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment-root', type=Path, required=True)
    p.add_argument('--reconstruction-root', type=Path, required=True)
    p.add_argument('--source-code', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False)
    exclusive(args.output / 'STARTED.json', {**META, 'status': 'started', 'job_id': os.environ.get('SLURM_JOB_ID'), 'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    try:
        result = validate(args)
        exclusive(args.output / 'VALIDATION.json', result)
        exclusive(args.output / 'COMPLETE.ok', result)
        print(json.dumps(result, sort_keys=True))
    except Exception as error:
        exclusive(args.output / 'FAILED.json', {**META, 'status': 'failed', 'exception': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(), 'failed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        raise


if __name__ == '__main__':
    main()
