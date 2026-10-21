"""
Sampling script for the 3D Wavelet Diffusion Model — T2W‑only version.

Always generates T2W (no loop over all four modalities).
All other logic is identical to the original `generation_sample.py`.
"""

import argparse
import nibabel as nib
import numpy as np
import os
import pathlib
import random
import sys
import torch as th

sys.path.append(".")
sys.path.append("..")
from guided_diffusion.c_bratsloader import c_BraTSVolumes
from guided_diffusion import (dist_util,
                              logger)
from guided_diffusion.script_util import (model_and_diffusion_defaults,
                                          create_model_and_diffusion,
                                          add_dict_to_argparser,
                                          args_to_dict,
                                          )
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

# ---- fixed target modality ------------------------------------------------
TARGET_MODALITY = "t2w"


def main():
    args = create_argparser().parse_args()
    seed = args.seed
    logger.configure()

    args.devices = [th.cuda.current_device()]
    dist_util.setup_dist(devices=args.devices)
    print(f"Devices: {args.devices}")

    logger.log("Creating model and diffusion...")
    args.diffusion_steps = int(args.sampling_steps)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    logger.log("Load model from: {}".format(args.model_path))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())

    if args.use_fp16:
        raise ValueError("fp16 currently not implemented")

    model.eval()
    idwt = IDWT_3D("haar")
    dwt = DWT_3D('haar')

    datal, ds = c_BraTSVolumes(directory=args.data_dir,
                               batch_size=args.batch_size,
                               num_workers=int(args.num_workers),
                               mode=args.mode,
                               img_size=args.image_size,
                               data_split_json=args.data_split_json).get_dl_ds()
    iterator_data = iter(datal)

    for ind in range(args.num_samples // args.batch_size):
        th.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        seed += 1

        args.output_dir = f"./results/{args.train_mode}_{args.sampling_steps}"
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        modal_to_generate = TARGET_MODALITY  # fixed: always T2W

        batch = next(iterator_data)
        t1n_modal = batch["t1n"].to(dist_util.dev())
        t1c_modal = batch["t1c"].to(dist_util.dev())
        t2f_modal = batch["t2f"].to(dist_util.dev())

        # T2W from batch is used *only* as ground-truth for saving; it is
        # replaced with noise before feeding the model.
        seg = batch["seg"].to(dist_util.dev())

        # save ground-truth T2W for reference
        nifti_image = nib.Nifti1Image(np.squeeze(batch[modal_to_generate].numpy()), affine=np.eye(4))
        nib.save(nifti_image, f'{args.output_dir}/{modal_to_generate}_modal_{ind}.nii.gz')

        noise_as_input = th.randn(args.batch_size,
                                  1,
                                  args.image_size,
                                  args.image_size,
                                  args.image_size).to(dist_util.dev())
        # T2W is the missing modality → fill with noise
        t2w_modal = noise_as_input

        if args.train_mode in ("default", "known_all_time", "known_3_to_gen_1"):
            print(f"Doing mode {args.train_mode}  (T2W only)")
            combined_mri = th.cat((t1n_modal, t1c_modal, t2f_modal, t2w_modal), dim=1)
            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(combined_mri)
            x_start_dwt = th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
            print(f"x_start_dwt: {x_start_dwt.shape}")
            img = x_start_dwt
        else:
            noise_base = th.randn(args.batch_size,
                                  32,
                                  args.image_size // 2,
                                  args.image_size // 2,
                                  args.image_size // 2).to(dist_util.dev())
            modal_to_generate = None

        model_kwargs = {}
        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(model=model,
                           time=int(args.sampling_steps),
                           shape=img.shape,
                           input_volume=img,
                           clip_denoised=args.clip_denoised,
                           model_kwargs=model_kwargs,
                           modal_to_generate=modal_to_generate,
                           mode=args.train_mode,
                           )

        B, C, D, H, W = sample.size()
        print(f"sample.shape: {sample.shape}")

        if args.train_mode in ("default", "known_all_time"):
            modal_list_name = ['t1n', 't1c', 't2f', 't2w']
            for modal_idx, modal in enumerate(modal_list_name):
                new_sample = idwt(sample[:, 0 + modal_idx, :, :, :].view(B, 1, D, H, W) * 3.,
                                  sample[:, 4 + modal_idx, :, :, :].view(B, 1, D, H, W),
                                  sample[:, 8 + modal_idx, :, :, :].view(B, 1, D, H, W),
                                  sample[:, 12 + modal_idx, :, :, :].view(B, 1, D, H, W),
                                  sample[:, 16 + modal_idx, :, :, :].view(B, 1, D, H, W),
                                  sample[:, 20 + modal_idx, :, :, :].view(B, 1, D, H, W),
                                  sample[:, 24 + modal_idx, :, :, :].view(B, 1, D, H, W),
                                  sample[:, 28 + modal_idx, :, :, :].view(B, 1, D, H, W))
                if len(new_sample.shape) == 5:
                    new_sample = new_sample.squeeze(dim=1)
                for i in range(new_sample.shape[0]):
                    output_name = os.path.join(args.output_dir, f'sample_{ind}_{i}_{modal}.nii.gz')
                    nifti_out = nib.Nifti1Image(new_sample.detach().cpu().numpy()[i, :, :, :], np.eye(4))
                    nib.save(img=nifti_out, filename=output_name)
                    print(f'Saved to {output_name}')
        else:
            # known_3_to_gen_1 branch — output is 8 channels (one modality)
            new_sample = idwt(sample[:, 0, :, :, :].view(B, 1, D, H, W) * 3.,
                              sample[:, 1, :, :, :].view(B, 1, D, H, W),
                              sample[:, 2, :, :, :].view(B, 1, D, H, W),
                              sample[:, 3, :, :, :].view(B, 1, D, H, W),
                              sample[:, 4, :, :, :].view(B, 1, D, H, W),
                              sample[:, 5, :, :, :].view(B, 1, D, H, W),
                              sample[:, 6, :, :, :].view(B, 1, D, H, W),
                              sample[:, 7, :, :, :].view(B, 1, D, H, W))
            if len(new_sample.shape) == 5:
                new_sample = new_sample.squeeze(dim=1)
            for i in range(new_sample.shape[0]):
                output_name = os.path.join(args.output_dir, f'sample_{ind}_{i}_{modal_to_generate}.nii.gz')
                nifti_out = nib.Nifti1Image(new_sample.detach().cpu().numpy()[i, :, :, :], np.eye(4))
                nib.save(img=nifti_out, filename=output_name)
                print(f'Saved to {output_name}')


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        data_mode='validation',
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        class_cond=False,
        sampling_steps=0,
        model_path="",
        devices=[0],
        output_dir='./results',
        mode=None,
        renormalize=False,
        image_size=256,
        half_res_crop=False,
        concat_coords=False,
        data_split_json=None,
        num_workers=None,
        train_mode=None,
    )
    defaults.update({k: v for k, v in model_and_diffusion_defaults().items() if k not in defaults})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
