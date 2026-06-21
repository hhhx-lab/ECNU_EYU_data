"""
Training script for the 3D Wavelet Diffusion Model — T2W‑only version.

Patches `random.choice` so the diffusion module always selects "t2w" as the
missing modality during training.  All other logic is identical to the
original `generation_train.py`.
"""

import argparse
import numpy as np
import random
import sys
import torch as th
import datetime

# ── monkey-patch random.choice ────────────────────────────────────────────
# gaussian_diffusion.py 在 training_losses 中调用 random.choice(modalities_L)
# 来随机选择一个模态加噪。我们拦截它，让被选中的永远是 "t2w"。
_orig_choice = random.choice

def _patched_choice(seq):
    if isinstance(seq, list) and seq == ["t1n", "t1c", "t2f", "t2w"]:
        return "t2w"
    return _orig_choice(seq)

random.choice = _patched_choice
# ──────────────────────────────────────────────────────────────────────────

sys.path.append(".")
sys.path.append("..")

from guided_diffusion import (dist_util,
                              logger)
from guided_diffusion.c_bratsloader import c_BraTSVolumes
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (model_and_diffusion_defaults,
                                          create_model_and_diffusion,
                                          args_to_dict,
                                          add_dict_to_argparser)
from guided_diffusion.train_util import TrainLoop
from torch.utils.tensorboard import SummaryWriter


def main():
    args = create_argparser().parse_args()
    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    summary_writer = None
    if args.use_tensorboard:
        if args.tensorboard_path:
            logdir = args.tensorboard_path
        else:
            now = datetime.datetime.now()
            logdir = (f"./runs/{args.train_mode}_{now.day}_{now.month}_{now.year}_"
                      f"{str(now.hour).zfill(2)}:{str(now.minute).zfill(2)}:{str(now.second).zfill(2)}")
        summary_writer = SummaryWriter(log_dir=logdir)
        summary_writer.add_text(
            'config',
            '\n'.join([f'--{k}={repr(v)} <br/>' for k, v in vars(args).items()])
        )
        logger.configure(dir=summary_writer.get_logdir())
    else:
        logger.configure()

    args.devices = [th.cuda.current_device()]
    dist_util.setup_dist(devices=args.devices)
    print(f"Devices: {args.devices}")

    logger.log("Creating model and diffusion...")
    arguments = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**arguments)

    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion, maxt=1000)

    if args.dataset == 'brats':
        assert args.image_size in [128, 256], "We currently just support image sizes: 128, 256"
        datal, ds = c_BraTSVolumes(directory=args.data_dir,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   mode='train',
                                   img_size=args.image_size,
                                   data_split_json=args.data_split_json).get_dl_ds()
    else:
        print("We currently just support the datasets: c_brats")

    logger.log("Start training...  (T2W‑only mode)")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=datal,
        batch_size=args.batch_size,
        in_channels=args.in_channels,
        image_size=args.image_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        resume_step=args.resume_step,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dataset=args.dataset,
        summary_writer=summary_writer,
        mode=args.train_mode,
        out_channels=args.out_channels,
        tumour_loss_weight=args.tumour_loss_weight,
    ).run_loop()


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',
        resume_step=0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset='brats',
        use_tensorboard=True,
        tensorboard_path='',
        devices=[0],
        dims=3,
        learn_sigma=False,
        num_groups=32,
        channel_mult="1,2,2,4,4",
        in_channels=8,
        out_channels=8,
        bottleneck_attention=False,
        num_workers=0,
        mode='default',
        renormalize=True,
        additive_skips=False,
        use_freq=False,
        data_split_json=None,
        train_mode=None,
        tumour_loss_weight=None,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
