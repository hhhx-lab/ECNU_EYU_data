"""Fail-closed checkpoint continuation checks; never modifies the source run."""

import datetime as dt
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path

META = dict(artifact_status='experimental_unvalidated', operator_approved=False,
            formal_gate_status='not_run_not_passed')
TRAINER = 'nnUNetTrainerBraTS2026RCMetAugK1245FocalCompletionFineTune175'
FOLD_REL = Path('results/K1/Dataset264_BraTS2026_MET_Completion') / (
    TRAINER + '__nnUNetPlans__3d_fullres/fold_0')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def write_once(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump({**META, 'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                   **payload}, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    path.chmod(0o400)


def finite(value):
    import numpy as np
    import torch
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(value)
    return True


def equal(left, right):
    import numpy as np
    import torch
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return isinstance(right, (tuple, list)) and len(left) == len(right) and all(
            equal(a, b) for a, b in zip(left, right))
    return left == right


def lr(epoch):
    return 0.001 * (1 - epoch / 175) ** 0.9


def checkpoint(path, expected_epoch=None):
    import torch
    path = Path(path)
    before = path.stat()
    state = torch.load(path, map_location='cpu', weights_only=False)
    digest = sha(path)
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), 'checkpoint changed during read')
    epoch = state.get('current_epoch')
    require(isinstance(epoch, int) and 0 < epoch <= 175 and epoch % 25 == 0,
            'invalid cumulative checkpoint epoch')
    require(expected_epoch is None or epoch == expected_epoch, 'checkpoint epoch mismatch')
    require(state['trainer_name'] == TRAINER, 'checkpoint trainer mismatch')
    require(bool(state['network_weights']), 'missing network weights')
    require(bool(state['optimizer_state']['state']), 'missing optimizer state')
    require(bool(state.get('grad_scaler_state')), 'missing AMP state')
    for key in ('network_weights', 'optimizer_state', 'grad_scaler_state', 'logging'):
        require(finite(state[key]), 'nonfinite ' + key)
    log = state['logging']
    for key in ('ema_fg_dice', 'train_losses', 'val_losses', 'lrs'):
        require(len(log[key]) == epoch, key + ' length differs from current_epoch')
    for index, value in enumerate(log['lrs']):
        require(math.isclose(float(value), lr(index), rel_tol=1e-8, abs_tol=1e-12),
                'historical 175-epoch LR schedule drift')
    for group in state['optimizer_state']['param_groups']:
        require(math.isclose(group['lr'], lr(epoch - 1), rel_tol=1e-8), 'saved LR drift')
        require(group['momentum'] == 0.99 and group['nesterov'] is True and
                group['weight_decay'] == 3e-5, 'SGD contract drift')
    return state, dict(path=str(path), sha256=digest, bytes=after.st_size,
                      current_epoch=epoch, ema_fg_dice=float(log['ema_fg_dice'][-1]),
                      network_finite=True, optimizer_finite=True, amp_finite=True,
                      logging_finite=True, next_epoch_lr=lr(epoch),
                      network_tensor_count=len(state['network_weights']),
                      optimizer_state_count=len(state['optimizer_state']['state']))


def verify_files(root):
    root = Path(root)
    manifest = load_json(root / 'PACKAGE_SHA256.json')
    for relative, digest in manifest['files'].items():
        require(sha(root / relative) == digest, 'package SHA drift: ' + relative)
        if relative.endswith('.py'):
            compile((root / relative).read_text(), str(root / relative), 'exec')
    plan = load_json(root / 'EXECUTION_PLAN.json')
    source = Path(plan['source_root'])
    for relative, digest in plan['source_contract_sha256'].items():
        require(sha(source / relative) == digest, 'source SHA drift: ' + relative)
    runmod = importlib.import_module('nnunetv2.run.run_training')
    require(sha(inspect.getsourcefile(runmod)) == plan['installed_cli_sha256'],
            'installed nnUNet CLI drift')
    for key in ('S2_K1245_EPOCHS', 'S2_COMPLETION_EPOCHS', 'S2_SCREEN_EPOCHS'):
        require(os.environ[key] == '175', key + ' drift')
    require(os.environ['S2_CONTINUE'] == '1' and os.environ['nnUNet_compile'] == '0',
            'resume/compile environment drift')
    require(Path(os.environ['nnUNet_results']).resolve() == (root / 'results/K1').resolve(),
            'output is not independent')
    require(Path(os.environ['S2_MET_AUG_AUDIT_PATH']).resolve().is_relative_to(root.resolve()),
            'events would write outside resume root')
    for name in ('REMOTE_STAGE_VALIDATION.json', 'REMOTE_FINAL_PREFLIGHT.json'):
        path = root / 'evidence' / name
        if path.exists():
            for dependency, digest in load_json(path)['dependency_sha256'].items():
                require(sha(dependency) == digest, 'dependency SHA drift: ' + dependency)
    return plan


def restore_test(path, output, gpu=False):
    import torch
    from nnunetv2.run.run_training import get_trainer_from_args
    state, report = checkpoint(path)
    trainer = get_trainer_from_args('264', '3d_fullres', 0, TRAINER,
                                    continue_training=True,
                                    device=torch.device('cuda' if gpu else 'cpu'))
    require(Path(trainer.output_folder).resolve() == Path(output).resolve(),
            'trainer output directory mismatch')
    compare_model_contract(state, trainer)
    # The frozen custom trainer accepts a filename, not a checkpoint dict.
    trainer.load_checkpoint(str(path))
    require(trainer.current_epoch == state['current_epoch'], 'current_epoch not restored')
    require(equal(state['network_weights'], trainer.network.state_dict()), 'network restore mismatch')
    require(equal(state['optimizer_state'], trainer.optimizer.state_dict()), 'optimizer restore mismatch')
    require(equal(state['logging'], trainer.logger.get_checkpoint()), 'logger restore mismatch')
    require(equal(state['_best_ema'], trainer._best_ema), 'best EMA state changed')
    require(equal(state.get('inference_allowed_mirroring_axes'),
                  trainer.inference_allowed_mirroring_axes), 'mirroring state changed')
    require((trainer.num_epochs, trainer.initial_lr, trainer.save_every) == (175, 0.001, 25),
            'training horizon reset')
    require(trainer.requires_single_threaded_augmentation and not trainer.is_ddp and
            not trainer._do_i_compile(), 'worker/DDP/compile contract drift')
    if gpu:
        require(equal(state['grad_scaler_state'], trainer.grad_scaler.state_dict()),
                'AMP restore mismatch')
    trainer.lr_scheduler.step(trainer.current_epoch)
    require(math.isclose(trainer.optimizer.param_groups[0]['lr'], lr(trainer.current_epoch),
                         rel_tol=1e-8), 'next epoch LR mismatch')
    return {**report, 'status': 'pass', 'real_trainer_network_optimizer_logger_restore': True,
            'amp_restore_verified_on_gpu': gpu,
            'amp_cpu_scope': 'saved state inspected; CUDA restore required at job startup',
            'horizon': 175, 'save_every': 25}


def compare_model_contract(state, trainer):
    expected = dict(state['init_args']['plans'])
    actual = dict(trainer.my_init_kwargs['plans'])
    expected.pop('continue_training', None)
    actual.pop('continue_training', None)
    require(equal(expected, actual), 'nnUNet model/data plans differ from checkpoint')
    require(equal(state['init_args']['dataset_json'], trainer.dataset_json),
            'dataset_json differs from checkpoint')
    require(state['init_args']['configuration'] == '3d_fullres' and
            int(state['init_args']['fold']) == 0, 'checkpoint fold/configuration mismatch')
