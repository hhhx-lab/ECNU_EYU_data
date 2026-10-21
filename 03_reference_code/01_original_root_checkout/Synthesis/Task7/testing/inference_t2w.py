"""MLCube handler file — T2W‑fixed version.
Always generates the missing T2W modality from T1n, T1c, T2f.
"""
import os
import torch
from os import listdir
from os.path import join
import sys

import argparse
import nibabel as nib
import numpy as np
import os
import shutil
import zipfile
import pathlib
import random
import sys
import torch as th
from scipy.stats import linregress
import json

sys.path.append(str(pathlib.Path(__file__).parent.parent / "wdm-3d"))
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


def create_data_split(data_path):
    training_list = []
    cases_folder_L = listdir(data_path)
    dataset_name = data_path.split("/")[-1]
    cases_folder_L.sort()
    already_done = f"./predictions/{dataset_name}"
    os.makedirs(already_done, exist_ok=True)
    already_done = listdir(already_done)

    for idx, case_folder in enumerate(cases_folder_L):
        dict_entry = {}
        t1c_path = join(case_folder, f"{case_folder}-t1c.nii.gz")
        if os.path.isfile(join(data_path, t1c_path)):
            dict_entry['t1c'] = t1c_path
        t1n_path = join(case_folder, f"{case_folder}-t1n.nii.gz")
        if os.path.isfile(join(data_path, t1n_path)):
            dict_entry['t1n'] = t1n_path
        t2f_path = join(case_folder, f"{case_folder}-t2f.nii.gz")
        if os.path.isfile(join(data_path, t2f_path)):
            dict_entry['t2f'] = t2f_path
        t2w_path = join(case_folder, f"{case_folder}-t2w.nii.gz")
        if os.path.isfile(join(data_path, t2w_path)):
            dict_entry['t2w'] = t2w_path

        if (f"{case_folder}-t1c.nii.gz" in already_done
            or f"{case_folder}-t1n.nii.gz" in already_done
            or f"{case_folder}-t2f.nii.gz" in already_done
            or f"{case_folder}-t2w.nii.gz" in already_done):
            print(f"Case {case_folder} already done.")
        else:
            training_list.append(dict_entry)
    final_json = {"testing": training_list}
    with open(f'./DataSet/{dataset_name}.json', 'w') as json_file:
        json.dump(final_json, json_file, indent=4)


def crop_to_240_240_155(tensor):
    start_x = (256 - 240) // 2
    start_y = (256 - 240) // 2
    start_z = (256 - 155) // 2
    cropped_tensor = tensor[start_x:start_x + 240,
                            start_y:start_y + 240,
                            start_z:start_z + 155]
    return cropped_tensor


def get_affine(scan_path):
    scan = nib.load(scan_path)
    header = scan.header
    affine = scan.affine
    return affine, header


def main(data_path, output_path, model_path):

    sampling_steps = 3000
    diffusion_steps = int(sampling_steps)
    in_channels = 36
    out_channels = 8
    noise_schedule = "linear"
    mode = "sample"
    train_mode = "known_3_to_gen_1"
    batch_size = 1
    num_workers = 2
    image_size = 256
    dataset_name = data_path.split("/")[-1]
    data_split_json = f'./DataSet/{dataset_name}.json'
    num_channels = 64

    # Fixed args
    class_cond = False
    num_res_blocks = 2
    num_heads = 1
    learn_sigma = False
    use_scale_shift_norm = False
    channel_mult = '1,2,2,4,4,4'
    rescale_learned_sigmas = False
    rescale_timesteps = False
    dims = 3
    num_groups = 32
    bottleneck_attention = False
    resample_2d = False
    renormalize = True
    additive_skips = True
    use_freq = False
    predict_xstart = True
    clip_denoised = True
    attention_resolutions = ''
    dataset = 'brats'

    data_dir = data_path
    output_dir = f"{output_path}/{dataset_name}"
    use_fp16 = False
    use_ddim = False

    seed = 42
    devices = [th.cuda.current_device()]
    dist_util.setup_dist(devices=devices)
    print(f"Devices: {devices}")

    print("Creating data splits json")
    create_data_split(data_path=data_dir)
    print("Creating model and diffusion...")

    args_in_a_dict = {
        'image_size': image_size,
        'num_channels': num_channels,
        'num_res_blocks': num_res_blocks,
        'num_heads': num_heads,
        'num_heads_upsample': -1,
        'num_head_channels': -1,
        'attention_resolutions': attention_resolutions,
        'channel_mult': channel_mult,
        'dropout': 0.0,
        'class_cond': class_cond,
        'use_checkpoint': False,
        'use_scale_shift_norm': use_scale_shift_norm,
        'resblock_updown': True,
        'use_fp16': use_fp16,
        'use_new_attention_order': False,
        'dims': dims,
        'num_groups': num_groups,
        'in_channels': in_channels,
        'out_channels': out_channels,
        'bottleneck_attention': bottleneck_attention,
        'resample_2d': resample_2d,
        'additive_skips': additive_skips,
        'mode': mode,
        'use_freq': use_freq,
        'predict_xstart': predict_xstart,
        'learn_sigma': learn_sigma,
        'diffusion_steps': diffusion_steps,
        'noise_schedule': noise_schedule,
        'timestep_respacing': '',
        'use_kl': False,
        'rescale_timesteps': rescale_timesteps,
        'rescale_learned_sigmas': rescale_learned_sigmas,
        'dataset': dataset
    }

    model, diffusion = create_model_and_diffusion(**args_in_a_dict)
    print("Load model from: {}".format(model_path))
    model.load_state_dict(dist_util.load_state_dict(model_path, map_location="cpu"))
    model.to(dist_util.dev([0, 1]) if len(devices) > 1 else dist_util.dev())
    model.eval()
    idwt = IDWT_3D("haar")
    dwt = DWT_3D('haar')

    # Creating data loader
    datal, ds = c_BraTSVolumes(directory=data_dir,
                               batch_size=batch_size,
                               num_workers=int(num_workers),
                               mode=mode,
                               img_size=image_size,
                               data_split_json=data_split_json).get_dl_ds()
    iterator_data = iter(datal)

    for ind, batch in enumerate(iterator_data):
        th.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        seed += 1

        modal_to_generate = TARGET_MODALITY  # fixed: always T2W

        print(f"######## Target modality: {modal_to_generate} ########")

        noise_as_input = th.randn(batch_size,
                                  1,
                                  image_size,
                                  image_size,
                                  image_size).to(dist_util.dev())

        # T1n, T1c, T2f are always present — load from batch
        t1n_modal = batch["t1n"].to(dist_util.dev())
        t1c_modal = batch["t1c"].to(dist_util.dev())
        t2f_modal = batch["t2f"].to(dist_util.dev())

        # T2W is missing — fill with noise
        t2w_modal = noise_as_input
        # try to get affine/header from one of the existing scans
        file_name = batch["t1n_meta_dict"]["filename_or_obj"][0].split("/")[-2]
        affine, header = get_affine(scan_path=batch["t1n_meta_dict"]["filename_or_obj"][0])

        combined_mri = th.cat((t1n_modal, t1c_modal, t2f_modal, t2w_modal), dim=1)
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(combined_mri)
        x_start_dwt = th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)

        model_kwargs = {}

        print("Generating missing modal")
        sample_fn = diffusion.p_sample_loop
        sample = sample_fn(model=model,
                           time=int(sampling_steps),
                           shape=x_start_dwt.shape,
                           input_volume=x_start_dwt,
                           clip_denoised=clip_denoised,
                           model_kwargs=model_kwargs,
                           modal_to_generate=modal_to_generate,
                           mode=train_mode,
                           )

        B, C, D, H, W = sample.size()

        # known_3_to_gen_1 branch
        new_sample = idwt(sample[:, 0, :, :, :].view(B, 1, D, H, W) * 3.,
                          sample[:, 1, :, :, :].view(B, 1, D, H, W),
                          sample[:, 2, :, :, :].view(B, 1, D, H, W),
                          sample[:, 3, :, :, :].view(B, 1, D, H, W),
                          sample[:, 4, :, :, :].view(B, 1, D, H, W),
                          sample[:, 5, :, :, :].view(B, 1, D, H, W),
                          sample[:, 6, :, :, :].view(B, 1, D, H, W),
                          sample[:, 7, :, :, :].view(B, 1, D, H, W))
        new_sample = (new_sample + 1) / 2.

        if len(new_sample.shape) == 5:
            new_sample = new_sample.squeeze(dim=1)

        for i in range(new_sample.shape[0]):
            output_name = os.path.join(output_dir, f"{file_name}-{modal_to_generate}.nii.gz")
            new_sample_arr = crop_to_240_240_155(tensor=new_sample.detach().cpu().numpy()[i, :, :, :])
            new_sample_arr = np.flip(np.flip(new_sample_arr, axis=0), axis=1)
            nifti_out = nib.Nifti1Image(new_sample_arr, affine=affine, header=header)
            nib.save(img=nifti_out, filename=output_name)
            print(f'Saved to {output_name}')

    print("INFERENCE FINISHED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation inference — T2W only")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path with the raw cases")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the predictions")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model checkpoint (.pt)")
    args = parser.parse_args()
    main(data_path=args.data_path, output_path=args.output_path, model_path=args.model_path)
