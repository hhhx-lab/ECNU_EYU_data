"""Fixed successful-update budget using upstream EDM, with complete boundary checkpoints."""
import argparse
import collections
import copy
import csv
import json
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from common import META, ROOT, now, sha, verify_package, write
sys.path.insert(0, str(ROOT / 'runtime/upstream_source'))
from src.train import tumour_main_diffusion as upstream
from src.utils.data_utils import get_loader
from src.utils.checkpoints import atomic_save

STOP = False
MODALITIES = ('t1c', 't1n', 't2w', 't2f')
LOSS_POLICY = 'uniform_original_quantization_warning_v1'
LABEL_ALGORITHM = 'original_physical_spacing_v2_labelwise_threshold_no_case_exception'
SUPPORT_CONTRACT = 'signal_statistics_vs_full_native_FOV_geometry_v1'
CHECKPOINT_SHA256 = '5d86a0b3cbc5df34d64d5a0ff376eae453634a609bd24dec1a1191fde800b13b'


def request_stop(signum, frame):
    global STOP
    STOP = True


def random_nodes(transform):
    nodes = {}
    seen = set()
    def visit(node, path):
        if id(node) in seen:
            return
        seen.add(id(node))
        if hasattr(node, 'R') and isinstance(node.R, np.random.RandomState):
            nodes[path] = node
        for index, child in enumerate(getattr(node, 'transforms', ())):
            visit(child, path + '/' + str(index))
    visit(transform, 'transform')
    return nodes


def capture_random(transform):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                monai={p: n.R.get_state() for p, n in random_nodes(transform).items()})


def restore_random(state, transform):
    nodes = random_nodes(transform)
    assert set(nodes) == set(state['monai']), 'MONAI random-state graph differs'
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        assert torch.cuda.device_count() == len(state['cuda'])
        torch.cuda.set_rng_state_all(state['cuda'])
    for path, node in nodes.items():
        node.R.set_state(state['monai'][path])


def new_cursor(n, seed):
    assert n > 0, 'Empty training dataset'
    generator = np.random.RandomState(seed)
    return dict(epoch=1, position=0, permutation=generator.permutation(n).tolist(),
                generator=generator.get_state(), n_rows=n)


def next_indices(cursor, batch_size):
    assert batch_size > 0 and cursor['n_rows'] > 0, 'Invalid batch or dataset size'
    assert 0 <= cursor['position'] <= cursor['n_rows'], 'Invalid sampler position'
    if cursor['position'] == cursor['n_rows']:
        generator = np.random.RandomState()
        generator.set_state(cursor['generator'])
        cursor['permutation'] = generator.permutation(cursor['n_rows']).tolist()
        cursor['generator'] = generator.get_state()
        cursor['position'] = 0
        cursor['epoch'] += 1
    end = min(cursor['position'] + batch_size, cursor['n_rows'])
    return cursor['permutation'][cursor['position']:end], end


def save_checkpoint(path, model, optimizer, scaler, step, cursor, transform, identity, args, schedule):
    state = {**META, 'checkpoint_schema': 'g1_complete_boundary_v1', 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'global_step': step,
             'epoch': cursor['epoch'], 'sampler': copy.deepcopy(cursor), 'rng': capture_random(transform),
             'data_identity': identity, 'training_config': vars(args), 'step_semantics': 'completed_optimizer_updates_v2',
             'schedule_config': {k: v for k, v in vars(schedule).items() if not isinstance(v, torch.Tensor)},
             'safe_boundary': True, 'utc': now()}
    for key in ('modality', 'normalization', 'crop_size', 'generator_type', 'time_ch_count', 'noise_schedule',
                'noise_embedding_mode', 'planner_gpu_gb', 'n_steps', 'p_uncond', 'in_channels', 'out_channels', 'feature_size'):
        state[key] = getattr(args, key)
    atomic_save(state, str(path))
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    fd = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return sha(path)


def restore_checkpoint(path, model, optimizer, scaler, transform, identity, args):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['checkpoint_schema'] == 'g1_complete_boundary_v1' and saved['safe_boundary'] is True
    assert saved['data_identity'] == identity and saved['training_config'] == vars(args), 'Resume identity drift'
    assert saved['step_semantics'] == 'completed_optimizer_updates_v2'
    cursor = saved['sampler']
    assert cursor['n_rows'] > 0 and 0 <= cursor['position'] <= cursor['n_rows'], 'Invalid saved sampler'
    assert sorted(cursor['permutation']) == list(range(cursor['n_rows'])), 'Invalid saved permutation'
    model.load_state_dict(saved['state_dict'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    scaler.load_state_dict(saved['scaler'])
    assert optimizer.param_groups[0]['lr'] == args.optim_lr
    assert optimizer.param_groups[0]['betas'] == (args.adam_beta1, .999)
    restore_random(saved['rng'], transform)
    return saved['global_step'], saved['sampler']


def data_identity(modality):
    assert modality in MODALITIES, 'Unknown training modality'
    complete_path = ROOT / 'dataset/COMPLETE.ok'
    complete = json.loads(complete_path.read_text())
    assert complete['status'] == 'complete' and complete['cases'] == 926
    assert complete['split_counts'] == {'train': 823, 'val': 103}
    assert not (ROOT / 'dataset/FAILED.json').exists()
    assert complete['support_contract'] == SUPPORT_CONTRACT
    assert complete['STRICT_ORIGINAL_GATE'] is False
    cfg = json.loads((ROOT / 'config/RUN.json').read_text())
    for key, expected in (('loss_policy', LOSS_POLICY), ('label_resampling_algorithm', LABEL_ALGORITHM)):
        assert cfg[key] == complete[key] == expected, 'R9 policy mismatch: ' + key
    assert not complete.get('local_exception_sha256'), 'R9 cannot consume a local label exception'
    assert not (ROOT / 'plan/LOCAL_LABEL_EXCEPTION.json').exists()

    plan = json.loads((ROOT / 'plan/REUSE_CANDIDATES.json').read_text())
    candidates = {r['case_id']: r for r in plan['candidates']}
    assert len(candidates) == len(plan['candidates']), 'Duplicate reference candidate'
    parents = complete['reference_parents']
    assert parents == plan['parents'] and parents, 'Frozen reference parents drift'
    assert parents[cfg['reuse_parent']] == cfg['reuse_parent_package_sha256']
    for parent_name, digest in parents.items():
        parent = Path(parent_name)
        assert parent.is_absolute() and parent.resolve() == parent, 'Parent path redirect'
        manifest_path = parent / 'plan/PACKAGE_MANIFEST.json'
        assert sha(manifest_path) == digest, 'Frozen parent package drift'
        package = json.loads(manifest_path.read_text())
        assert package['files'], 'Empty frozen parent package'
        for name, expected_sha in package['files'].items():
            path = parent / name
            assert not Path(name).is_absolute() and path.resolve().is_relative_to(parent)
            assert sha(path) == expected_sha, 'Frozen parent file drift: ' + str(path)
        source_manifest = parent / 'plan/UPSTREAM_SOURCE_MANIFEST.json'
        assert 'plan/UPSTREAM_SOURCE_MANIFEST.json' in package['files']
        source = json.loads(source_manifest.read_text())
        assert source['files'], 'Empty frozen parent source manifest'
        source_root = parent / 'runtime/upstream_source'
        for name, expected_sha in source['files'].items():
            path = source_root / name
            assert not Path(name).is_absolute() and path.resolve().is_relative_to(source_root)
            assert sha(path) == expected_sha, 'Frozen parent source drift: ' + str(path)

    membership_path = ROOT / 'plan/splits_final_train_val_test_membership.csv'
    with membership_path.open(newline='') as stream:
        membership_rows = list(csv.DictReader(stream))
    membership = {r['source_case_id']: r for r in membership_rows}
    assert len(membership) == len(membership_rows) == 1030, 'Membership duplicate or count drift'
    assert collections.Counter(r['split'] for r in membership_rows) == {'train': 823, 'val': 103, 'test': 104}
    groups = collections.defaultdict(set)
    for row in membership_rows:
        assert row['t2w_status'] == 'authentic' and row['eligible_for_realonly'].lower() == 'true'
        assert Path(row['source_case_id']).name == row['source_case_id'], 'Invalid case identifier'
        groups[row['patient_group']].add(row['split'])
    assert all(len(splits) == 1 for splits in groups.values()), 'Patient split leakage'
    members = {case: row for case, row in membership.items() if row['split'] != 'test'}
    details = {row['case']: row for row in complete['cases_detail']}
    assert len(details) == len(complete['cases_detail']) == 926, 'Case detail duplicate or count drift'
    assert set(complete['case_manifests']) == set(complete['case_locations']) == set(details) == set(members)

    source_path = Path(cfg['source_code']) / 'data/data_csv.csv'
    assert sha(source_path) == cfg['source_csv_sha256'], 'Source CSV drift'
    with source_path.open(newline='') as stream:
        source_rows = list(csv.DictReader(stream))
    source = {r['id']: r for r in source_rows}
    assert len(source) == len(source_rows), 'Duplicate source case'
    recon = Path(cfg['recon']) / 'dataset'
    native_complete_path = recon / 'COMPLETE.ok'
    native_complete = json.loads(native_complete_path.read_text())
    assert native_complete['status'] == 'pass' and native_complete['checkpoint_sha256'] == CHECKPOINT_SHA256

    csv_path = ROOT / 'dataset/lesions.csv'
    assert sha(csv_path) == complete['csv_sha256'], 'Lesion CSV drift'
    with csv_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert rows, 'Empty lesion CSV'
    csv_by_case = collections.defaultdict(list)
    patient_counts = collections.Counter(r['patient_id'] for r in rows)
    for row in rows:
        case = row['source_case_id']
        assert case in members, 'Lesion row is outside train/val membership'
        member = members[case]
        assert row['split'] == member['split'] and row['patient_group'] == member['patient_group'], 'Lesion membership drift'
        assert row['patient_id'] == case[-9:] and int(row['t2w_valid']) == 1
        assert int(row['patient_n_crops']) == patient_counts[row['patient_id']], 'Patient crop count drift'
        directory = ROOT / 'dataset/cases' / case
        for key, suffix in [('label', 'seg'), *(('scan_' + m, m) for m in MODALITIES)]:
            assert row[key] == str(directory / (case + '-' + suffix + '.nii.gz')), 'Lesion file mapping drift'
        csv_by_case[case].append(row)
    if 'rows' in complete:
        assert complete['rows'] == len(rows)
    if 'train_rows' in complete:
        assert complete['train_rows'] == sum(r['split'] == 'train' for r in rows)
    report_path = ROOT / 'dataset/PREPROCESS_REPORT.json'
    assert sha(report_path) == complete['report_sha256'], 'Preprocessing report drift'
    report = json.loads(report_path.read_text())
    assert report['status'] == 'complete' and not report['errors']
    assert report['cases_ok'] == report['cases_processed'] == 926
    assert report['loss_policy'] == LOSS_POLICY and report['label_resampling_algorithm'] == LABEL_ALGORITHM
    assert report['warnings'] == complete['resolution_warnings'], 'Resolution warning report drift'

    consumed, references, index_hashes, native_hashes = {}, {}, {}, {}
    warnings = {}
    for case, expected_marker_sha in complete['case_manifests'].items():
        member, detail = members[case], details[case]
        assert source[case]['split'] == member['split'], 'Source case split drift'
        directory = ROOT / 'dataset/cases' / case
        assert directory.is_dir() and not directory.is_symlink(), 'R9 case directory must be real'
        assert directory.resolve() == ROOT.resolve() / 'dataset/cases' / case, 'R9 case path redirect'
        assert str(directory.resolve()) == complete['case_locations'][case], 'Dataset case location drift'
        assert detail['split'] == member['split'], 'Case detail split drift'
        assert detail['resolved_directory'] == str(directory.resolve()), 'Case detail location drift'
        marker = directory / 'preprocess_complete.json'
        assert not marker.is_symlink() and sha(marker) == expected_marker_sha, 'Preprocessing marker drift'
        p = json.loads(marker.read_text())
        assert p['status'] == 'complete' and p['case_id'] == case
        assert p['split'] == member['split'] and p['patient_group'] == member['patient_group'], 'Marker membership drift'
        contract = p['contract']
        assert contract['loss_policy'] == LOSS_POLICY and contract['label_resampling_algorithm'] == LABEL_ALGORITHM, 'Marker policy mismatch'
        assert contract['support_contract'] == SUPPORT_CONTRACT and contract['grid_version'] == 'physical_spacing_v2'
        assert contract['target_spacing'] == [1., 1., 1.]
        assert contract['normalization'] == 'whole_case_common_mask_zscore_before_resampling_zero_outside_support'
        assert contract['mask_algorithm'] == 'authentic_T1N_T1C_T2F_robust_0_100_nonzero_union_holefill'
        assert contract['bbox'] == [[0, int(n)] for n in contract['original_shape']]
        for record in (contract, p, detail):
            assert not record.get('local_label_exception') and not record.get('local_exception'), 'R9 local label fix forbidden'
        lost = [str(k) for k in range(1, 5) if contract['label_native'][str(k)]['voxels'] and not contract['label_target_1mm'][str(k)]['voxels']]
        assert contract['lost_classes'] == detail['lost_classes'] == lost
        assert contract['resolution_warning'] is bool(lost), 'Quantization warning mismatch'
        if lost:
            warnings[case] = lost
        spatial = json.loads((directory / 'spatial_mapping.json').read_text())
        assert all(spatial.get(k) == v for k, v in contract.items()), 'Spatial mapping contract drift'
        native_path = recon / 'cases' / case / 'case_manifest.json'
        native_sha = native_complete['manifest_sha256'][case]
        assert sha(native_path) == p['native_case_manifest_sha256'] == native_sha, 'Native manifest drift'
        native = json.loads(native_path.read_text())
        assert native['case_id'] == case and native['split'] == member['split'] and native['geometry'] == 'native'
        assert native['checkpoint_sha256'] == CHECKPOINT_SHA256 and native['native_shape'] == contract['original_shape']
        assert set(p['input_sha256']) == set((*MODALITIES, 'seg'))
        assert all(h == native['files'][m]['sha256'] for m, h in p['input_sha256'].items()), 'Native input identity drift'
        assert p['input_sha256']['seg'] == native['source_sha256']['seg'], 'Native label source drift'
        assert set(p['authentic_mask_source_sha256']) == {'t1n', 't1c', 't2f'}
        assert all(h == native['source_sha256'][m] for m, h in p['authentic_mask_source_sha256'].items())
        native_hashes[case] = native_sha

        expected_outputs = {case + '-' + m + '.nii.gz' for m in (*MODALITIES, 'seg', 'brainmask')}
        assert set(p['outputs']) == expected_outputs, 'Incomplete output hashes'
        assert detail['outputs'] == p['outputs'] and detail['marker_sha256'] == expected_marker_sha
        file_references = detail['file_references']
        assert set(file_references).issubset(expected_outputs), 'Unknown output reference'
        if 'file_references' in p:
            assert p['file_references'] == file_references, 'Marker references drift'
        if file_references:
            candidate = candidates[case]
            parent_directory = Path(candidate['directory'])
            assert p['parent_directory'] == str(parent_directory), 'Parent marker directory drift'
            assert parent_directory.resolve() in {(Path(parent) / 'dataset/cases' / case).resolve() for parent in parents}
            parent_marker = parent_directory / 'preprocess_complete.json'
            assert sha(parent_marker) == p['parent_marker_sha256'] == candidate['marker_sha256'], 'Parent marker drift'
            parent_info = json.loads(parent_marker.read_text())
            assert parent_info['case_id'] == case and parent_info['split'] == member['split']
            assert parent_info['native_case_manifest_sha256'] == native_sha
            expected_references = expected_outputs - ({case + '-seg.nii.gz'} if candidate['mode'] == 'label_only_rebuild' else set())
            assert set(file_references) == expected_references, 'Reference/rebuilt label mapping drift'
            assert all(parent_info['outputs'][name] == ref['sha256'] for name, ref in file_references.items()), 'Parent output identity drift'
        for name, expected_sha in p['outputs'].items():
            path = directory / name
            assert path.is_file(), 'Missing output: ' + str(path)
            if name in file_references:
                reference = file_references[name]
                target = Path(reference['path'])
                assert path.is_symlink() and target.is_absolute() and target.resolve() == target
                assert path.resolve() == target, 'Output reference target drift'
                allowed_targets = {(Path(parent) / 'dataset/cases' / case / name).resolve() for parent in parents}
                assert target in allowed_targets, 'Output reference outside frozen parents/case'
                assert reference['sha256'] == expected_sha, 'Output reference SHA drift'
            else:
                assert not path.is_symlink() and path.resolve() == directory.resolve() / name, 'Undeclared output reference'
            assert sha(path) == expected_sha, 'Consumed data SHA drift: ' + str(path)
            if member['split'] == 'train' and name in (case + '-' + modality + '.nii.gz', case + '-seg.nii.gz'):
                consumed[str(directory / name)] = p['outputs'][name]
        references[case] = file_references

        index_path = ROOT / 'dataset/indices' / (case + '.json')
        index = json.loads(index_path.read_text())
        assert index['case_id'] == case and index['marker_sha256'] == expected_marker_sha, 'Persisted index marker drift'
        # Preprocessing persists rows before adding the final CSV's patient_n_crops column.
        def row_key(row):
            return tuple(sorted((k, str(v)) for k, v in row.items() if k != 'patient_n_crops'))
        assert collections.Counter(map(row_key, index['rows'])) == collections.Counter(map(row_key, csv_by_case[case])), 'Persisted index rows drift'
        index_hashes[case] = sha(index_path)
    reported_warnings = {row['case']: row['lost_classes'] for row in report['warnings']}
    assert len(reported_warnings) == len(report['warnings']) and reported_warnings == warnings, 'Case warning identity drift'
    return dict(package_sha256=sha(ROOT / 'plan/PACKAGE_MANIFEST.json'), csv_sha256=complete['csv_sha256'],
                dataset_complete_sha256=sha(complete_path), case_manifests=complete['case_manifests'],
                reference_parents=parents, case_locations=complete['case_locations'], file_references=references,
                index_sha256=index_hashes, native_case_manifests=native_hashes,
                membership_sha256=sha(membership_path), source_csv_sha256=cfg['source_csv_sha256'],
                native_complete_sha256=sha(native_complete_path), loss_policy=LOSS_POLICY,
                label_resampling_algorithm=LABEL_ALGORITHM,
                report_sha256=complete['report_sha256'], resolution_warnings=complete['resolution_warnings'],
                consumed=consumed, selection='train823_only', modality=modality)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', required=True, choices=['t1c', 't1n', 't2w', 't2f'])
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--output', type=Path)
    options = parser.parse_args()
    cfg = json.loads((ROOT / 'config/RUN.json').read_text())
    output = options.output or ROOT / 'training' / options.modality
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'STARTED.json', {**META, 'utc': now(), 'job_id': os.environ.get('SLURM_JOB_ID'),
                                  'modality': options.modality, 'resume': str(options.resume) if options.resume else None})
    try:
        package = verify_package()
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Single allocated CUDA GPU required'
        assert 'A100' in torch.cuda.get_device_name(0), 'A100 allocation required'
        assert int(os.environ.get('WORLD_SIZE', '1')) == 1, 'Non-DDP only'
        upstream._diffusion_utils.seed_everything(cfg['seed'])
        args = SimpleNamespace(**cfg['training'])
        args.modality = options.modality
        args.csv_path = str(ROOT / 'dataset/lesions.csv')
        args.logdir = 'epoch004_1mm_resume_uniform_r9'
        identity = data_identity(options.modality)
        schedule = upstream._diffusion_utils.make_noise_schedule(schedule='edm', n_steps=args.n_steps,
                   sigma_data=args.sigma_data, sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                   rho=args.rho, snr_shift=args.snr_shift)
        model, optimizer = upstream.get_nets(args)
        assert sum(p.numel() for p in model.parameters()) == 4475999
        scaler = torch.amp.GradScaler('cuda')
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGUSR1, request_stop)
        # Use the upstream ROI cache/collator, with an explicitly checkpointed sequential cursor.
        loader = get_loader(args)
        transform = loader.dataset.transform
        transform.set_random_state(seed=cfg['seed'])
        cursor = new_cursor(len(loader.dataset), cfg['seed'])
        step = 0
        if options.resume:
            step, cursor = restore_checkpoint(options.resume, model, optimizer, scaler, transform, identity, args)
        assert cursor['n_rows'] == len(loader.dataset)
        assert 0 <= step < args.num_steps
        write(output / 'RUN_CONTRACT.json', {**META, 'utc': now(), 'package_sha256': package, 'config': vars(args),
               'identity': identity, 'gpu': torch.cuda.get_device_name(0), 'parameters': 4475999,
               'independent_acceptance': 'waived_by_user', 'resume_checkpoint_sha256': sha(options.resume) if options.resume else None})
        weights = output / 'weights'
        weights.mkdir()
        initial_step = step
        start = time.monotonic()
        with (output / 'progress.jsonl').open('x') as progress:
            while step < args.num_steps and not STOP:
                indices, end = next_indices(cursor, args.batch_size)
                batch = loader.collate_fn([loader.dataset[i] for i in indices])
                for key in ('scan_' + options.modality + '_crop_pad', 'label_crop_pad'):
                    assert tuple(batch[key].shape[-3:]) == (64, 64, 64)
                    if not torch.isfinite(batch[key]).all():
                        raise FloatingPointError('Nonfinite actual training batch')
                previous_step = step
                step, model, optimizer, losses, x, y = upstream.train(args, step, [batch], model, optimizer, scaler, schedule, str(output))
                assert step == previous_step + 1 and len(losses) == 1, 'Expected one successful optimizer update'
                cursor['position'] = end  # Commit only after the successful optimizer update.
                del batch, x, y
                progress.write(json.dumps(dict(utc=now(), global_step=step, loss=losses[0], epoch=cursor['epoch'],
                               batch_position=end, peak_cuda_bytes=torch.cuda.max_memory_allocated())) + '\n')
                progress.flush()
                if step % 5000 == 0 or step == args.num_steps:
                    path = weights / f'diffusion_{step}.pt'
                    digest = save_checkpoint(path, model, optimizer, scaler, step, cursor, transform, identity, args, schedule)
                    write(output / 'LATEST_CHECKPOINT.json', {**META, 'path': str(path), 'sha256': digest,
                          'global_step': step, 'safe_boundary': True, 'utc': now()}, replace=True)
                if step == 1 or step % 100 == 0:
                    print(json.dumps(dict(global_step=step, updates_per_second=(step - initial_step) / (time.monotonic() - start),
                                     peak_cuda_bytes=torch.cuda.max_memory_allocated())), flush=True)
        if STOP and step < args.num_steps:
            path = weights / f'safe_term_{step}.pt'
            digest = save_checkpoint(path, model, optimizer, scaler, step, cursor, transform, identity, args, schedule)
            write(output / 'SAFE_TERM.json', {**META, 'utc': now(), 'global_step': step, 'checkpoint': str(path), 'sha256': digest,
                                            'safe_boundary': True, 'training_complete': False})
            raise SystemExit(99)
        write(output / 'COMPLETE.ok', {**META, 'utc': now(), 'status': 'training_complete', 'global_step': step,
                                     'checkpoint': str(path), 'checkpoint_sha256': digest, 'selection_status': 'not_run'})
    except Exception as error:
        write(output / 'FAILED.json', {**META, 'utc': now(), 'message': str(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
