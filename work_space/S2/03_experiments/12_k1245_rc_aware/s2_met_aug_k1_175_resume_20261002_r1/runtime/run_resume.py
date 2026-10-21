"""Use the installed nnUNet CLI with --c, guarding its checkpoint fallback."""

import os
import sys
from pathlib import Path

from resume_contract import (FOLD_REL, TRAINER, checkpoint, compare_model_contract, equal, load_json, lr,
                             require, sha, verify_files, write_once)


def main():
    import math
    import torch
    import nnunetv2.run.run_training as cli

    root = Path(__file__).resolve().parents[1]
    plan = verify_files(root)
    evidence = root / 'evidence'
    require(not any((evidence / name).exists() for name in
                    ('RESUME_STARTED.json', 'RESUME_FAILED.json', 'RESUME_COMPLETE.ok')),
            'this attempt has already started or ended')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
            'A100' in torch.cuda.get_device_name(0), 'requires exactly one A100')
    launch = load_json(evidence / 'LAUNCH_AUDIT.json')
    require(launch['canonical_job_id'] == os.environ['SLURM_JOB_ID'], 'job identity mismatch')
    frozen = load_json(evidence / 'FINAL_SOURCE_ACCEPTANCE.json')
    require(frozen['source_state'] == 'TIMEOUT' and frozen['status'] == 'pass',
            'source TIMEOUT not accepted')
    for relative, digest in frozen['source_evidence_sha256'].items():
        require(sha(Path(plan['source_root']) / relative) == digest, 'old evidence changed')
    fold = root / FOLD_REL
    path = fold / 'checkpoint_latest.pth'
    require(not (fold / 'checkpoint_final.pth').exists(), 'unexpected final checkpoint')
    state, report = checkpoint(path, frozen['checkpoint']['current_epoch'])
    require(report['sha256'] == frozen['checkpoint']['sha256'], 'frozen resume input SHA mismatch')
    require(report['current_epoch'] < 175, 'resume would have no remaining training')
    original_load = cli.maybe_load_checkpoint

    def guarded_load(trainer, continue_training, validation_only, pretrained_weights_file=None):
        require(continue_training and not validation_only and pretrained_weights_file is None,
                'must use --c without pretrained weights or --val')
        require(Path(trainer.output_folder).resolve() == fold.resolve(), 'unsafe output path')
        require(not (fold / 'checkpoint_final.pth').exists() and sha(path) == report['sha256'],
                'checkpoint changed before restore')
        compare_model_contract(state, trainer)
        original_load(trainer, continue_training, validation_only, pretrained_weights_file)
        require(trainer.current_epoch == report['current_epoch'], 'silent scratch fallback')
        require(equal(state['network_weights'], trainer.network.state_dict()), 'network restore failed')
        require(equal(state['optimizer_state'], trainer.optimizer.state_dict()), 'optimizer restore failed')
        require(equal(state['logging'], trainer.logger.get_checkpoint()), 'logging restore failed')
        require(equal(state['grad_scaler_state'], trainer.grad_scaler.state_dict()), 'AMP restore failed')
        require(equal(state['_best_ema'], trainer._best_ema), 'best EMA not preserved')
        require(equal(state.get('inference_allowed_mirroring_axes'),
                      trainer.inference_allowed_mirroring_axes), 'mirroring state drift')
        require((trainer.num_epochs, trainer.initial_lr, trainer.save_every) == (175, 0.001, 25)
                and trainer.requires_single_threaded_augmentation and not trainer.is_ddp
                and not trainer._do_i_compile(), 'restored training contract mismatch')
        trainer.lr_scheduler.step(trainer.current_epoch)
        require(math.isclose(trainer.optimizer.param_groups[0]['lr'], lr(trainer.current_epoch),
                             rel_tol=1e-8), 'LR horizon reset')
        write_once(evidence / 'RESUME_STARTED.json', dict(status='pass',
                   canonical_job_id=os.environ['SLURM_JOB_ID'], source_job_id=plan['source_job_id'],
                   checkpoint=report, amp_restore_verified_on_gpu=True,
                   cli='nnUNetv2_train 264 3d_fullres 0 -tr ' + TRAINER + ' -num_gpus 1 --c',
                   F_STRICT_PREREG_GATE=False, total_epochs=175))
        print('K1_175_RESUME_STATE_RESTORE_PASS current_epoch=' + str(trainer.current_epoch), flush=True)
        state.clear()

    cli.maybe_load_checkpoint = guarded_load
    sys.argv = ['nnUNetv2_train', '264', '3d_fullres', '0', '-tr', TRAINER,
                '-num_gpus', '1', '--c']
    cli.run_training_entry()
    rows = []
    for epoch in (25, 50, 75, 100, 125, 150, 175):
        _, row = checkpoint(fold / f'checkpoint_epoch_{epoch:04d}.pth', epoch)
        rows.append(row)
    checkpoint(fold / 'checkpoint_final.pth', 175)
    summary = load_json(fold / 'validation/summary.json')
    require(len(summary['metric_per_case']) == 103, 'nnUNet validation case count differs')
    from resume_contract import finite
    require(finite(summary['foreground_mean']), 'nonfinite validation foreground mean')
    for relative, digest in frozen['source_evidence_sha256'].items():
        require(sha(Path(plan['source_root']) / relative) == digest, 'source evidence changed')
    best = sorted(rows, key=lambda row: (-row['ema_fg_dice'], row['current_epoch']))[0]
    write_once(evidence / 'TRAINING_VALIDATION.json', dict(status='pass', completed_epochs=175,
               candidate_checkpoints=rows, best_checkpoint=best, nnunet_validation_cases=103,
               official_met_completed=False, F_STRICT_PREREG_GATE=False))
    with (evidence / 'RESUME_COMPLETE.ok').open('x') as handle:
        handle.write('complete\n')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        root = Path(__file__).resolve().parents[1]
        failed = root / 'evidence/RESUME_FAILED.json'
        if not failed.exists():
            write_once(failed, dict(status='failed', error_type=type(error).__name__,
                       message=str(error), job_id=os.environ.get('SLURM_JOB_ID')))
        raise
