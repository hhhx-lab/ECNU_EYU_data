"""Fixed successful-update budget using upstream EDM, with complete boundary checkpoints."""
import argparse
import copy
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
    generator = np.random.RandomState(seed)
    return dict(epoch=1, position=0, permutation=generator.permutation(n).tolist(),
                generator=generator.get_state(), n_rows=n)


def next_indices(cursor, batch_size):
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
    model.load_state_dict(saved['state_dict'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    scaler.load_state_dict(saved['scaler'])
    assert optimizer.param_groups[0]['lr'] == args.optim_lr
    assert optimizer.param_groups[0]['betas'] == (args.adam_beta1, .999)
    restore_random(saved['rng'], transform)
    return saved['global_step'], saved['sampler']


def data_identity(modality):
    complete_path = ROOT / 'dataset/COMPLETE.ok'
    complete = json.loads(complete_path.read_text())
    assert complete['status'] == 'complete' and complete['cases'] == 926
    assert complete['split_counts'] == {'train': 823, 'val': 103}
    assert not (ROOT / 'dataset/FAILED.json').exists()
    assert complete['support_contract'] == 'signal_statistics_vs_full_native_FOV_geometry_v1'
    assert complete['STRICT_ORIGINAL_GATE'] is False
    cfg = json.loads((ROOT / 'config/RUN.json').read_text())
    assert complete['reuse_parent_package_sha256'] == cfg['reuse_parent_package_sha256']
    assert sha(Path(cfg['reuse_parent']) / 'plan/PACKAGE_MANIFEST.json') == cfg['reuse_parent_package_sha256']
    assert complete['local_exception_sha256'] == sha(ROOT / 'plan/LOCAL_LABEL_EXCEPTION.json')
    assert len(complete['reused_cases']) == 471
    assert len(set(complete['reused_cases'])) == 471
    assert len(complete['case_manifests']) == len(complete['case_locations']) == 926
    assert set(complete['case_manifests']) == set(complete['case_locations'])
    assert set(complete['reused_cases']).issubset(complete['case_manifests'])
    assert sha(ROOT / 'dataset/lesions.csv') == complete['csv_sha256']
    consumed = {}
    for case, expected in complete['case_manifests'].items():
        directory = ROOT / 'dataset/cases' / case
        resolved = str(directory.resolve())
        assert resolved == complete['case_locations'][case], 'Dataset reference target drift'
        expected = Path(cfg['reuse_parent']) / 'dataset/cases' / case if case in complete['reused_cases'] else directory
        assert resolved == str(expected.resolve()), 'Unexpected parent reference'
        marker = directory / 'preprocess_complete.json'
        assert sha(marker) == expected, 'Preprocessing marker drift'
        p = json.loads(marker.read_text())
        assert p['status'] == 'complete' and p['split'] in ('train', 'val')
        if p['split'] == 'train':
            for name in (case + '-' + modality + '.nii.gz', case + '-seg.nii.gz'):
                assert sha(directory / name) == p['outputs'][name], 'Consumed data SHA drift'
                consumed[str(directory / name)] = p['outputs'][name]
    return dict(package_sha256=sha(ROOT / 'plan/PACKAGE_MANIFEST.json'), csv_sha256=complete['csv_sha256'],
                dataset_complete_sha256=sha(complete_path), case_manifests=complete['case_manifests'],
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
        args.logdir = 'epoch004_1mm_resume_localfix_r8'
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
                step, model, optimizer, losses, x, y = upstream.train(args, step, [batch], model, optimizer, scaler, schedule, str(output))
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
