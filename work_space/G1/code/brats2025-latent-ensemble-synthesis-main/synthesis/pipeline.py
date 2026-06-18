

import os
import numpy as np
import torch
from monai.bundle import ConfigParser
from tqdm import tqdm

import configs
import synthesis.utils as utils

import models.encdec.unet as encdec
import models.bbdm.unet as bbdm
import models.bbdm.condition_tokens as bbdm_condition_tokens
import models.bbdm.bb_scheduler as bbdm_bb_scheduler

import synthesis.segment_brain_mask as segment_brain_mask


def instantiate_vae_model(device):
    parser = ConfigParser(configs.NETWORKS_CONFIG)
    parser.parse(True)

    autoencoder = parser.get_parsed_content("autoencoder_def").to(device)
    checkpoint_autoencoder = torch.load(configs.PATH_NAME_WEIGHTS_VAE, weights_only=True, map_location=device)
    autoencoder.load_state_dict(checkpoint_autoencoder)
    autoencoder.eval()
    return autoencoder




def instantiate_encdec_model(device):
    nconfig = utils.dict_to_args(configs.NETWORKS_CONFIG, deep_conversion=True)

    unet = encdec.DiffusionModelUNetMaisi(spatial_dims = nconfig.encdec.unet.spatial_dims,
                                                in_channels = nconfig.encdec.unet.in_channels,
                                                out_channels = nconfig.encdec.unet.out_channels,
                                                num_res_blocks = nconfig.encdec.unet.num_res_blocks,
                                                num_channels = nconfig.encdec.unet.num_channels,
                                                attention_levels = nconfig.encdec.unet.attention_levels,
                                                num_head_channels = nconfig.encdec.unet.num_head_channels,
                                                with_conditioning = nconfig.encdec.unet.with_conditioning,
                                                use_flash_attention = nconfig.encdec.unet.use_flash_attention,
                                                )
    unet.to(device)
    unet.eval()

    chk_path = utils.get_chkpoint_path(configs.PATH_WEIGHTS_ENCDEC, configs.MISSING_MODALITY)
    chk = torch.load(chk_path, weights_only=False, map_location=device)
    unet.load_state_dict(chk["unet_state_dict"], strict=True)
    return unet


def instantiate_bbdm_model(device):
    nconfig = utils.dict_to_args(configs.NETWORKS_CONFIG, deep_conversion=True)

    unet = bbdm.DiffusionModelUNetMaisi(spatial_dims = nconfig.bbdm.unet.spatial_dims,
                                                in_channels = nconfig.bbdm.unet.in_channels,
                                                out_channels = nconfig.bbdm.unet.out_channels,
                                                num_res_blocks = nconfig.bbdm.unet.num_res_blocks,
                                                num_channels = nconfig.bbdm.unet.num_channels,
                                                attention_levels = nconfig.bbdm.unet.attention_levels,
                                                num_head_channels = nconfig.bbdm.unet.num_head_channels,
                                                with_conditioning = nconfig.bbdm.unet.with_conditioning,
                                                use_flash_attention = nconfig.bbdm.unet.use_flash_attention,
                                                include_from_modality = nconfig.bbdm.unet.include_from_modality,
                                                include_to_modality = nconfig.bbdm.unet.include_to_modality,
                                                cross_attention_dim = nconfig.bbdm.unet.cross_attention_dim,
                                                transformer_num_layers = nconfig.bbdm.unet.transformer_num_layers,
                                                upcast_attention = nconfig.bbdm.unet.upcast_attention,
                                                )

    # ConditionTokens
    conditions_model = bbdm_condition_tokens.ConditionTokens(num_conditions=nconfig.bbdm.conditions_model.num_conditions,
                                    embed_dim=nconfig.bbdm.conditions_model.embed_dim,
                                    hidden_dim=nconfig.bbdm.conditions_model.hidden_dim,
                                    use_self_attention=nconfig.bbdm.conditions_model.use_self_attention,
                                    n_heads=nconfig.bbdm.conditions_model.n_heads,
                                    n_layers=nconfig.bbdm.conditions_model.n_layers)


    noise_scheduler = bbdm_bb_scheduler.Scheduler(
        num_timesteps=nconfig.bbdm.bb_scheduler.num_timesteps,
        sample_step=nconfig.bbdm.bb_scheduler.sample_step,
        s=nconfig.bbdm.bb_scheduler.s,
        sample_type=nconfig.bbdm.bb_scheduler.sample_type,
        objective_type=nconfig.bbdm.bb_scheduler.objective_type
    )

    chk_path = utils.get_chkpoint_path(configs.PATH_WEIGHTS_BBDM, configs.MISSING_MODALITY)
    chk = torch.load(chk_path, weights_only=False, map_location=device)

    unet.to(device)
    unet.eval()
    conditions_model.to(device)
    conditions_model.eval()
    unet.load_state_dict(chk["unet_state_dict"], strict=True)
    conditions_model.load_state_dict(chk["conditions_model_state_dict"], strict=True)

    return unet, conditions_model, noise_scheduler





@torch.no_grad()
def encode_image(img, vae):
    # with torch.no_grad(), torch.amp.autocast("cuda"):
    utils.set_seed(42)
    latent = vae.encode_stage_2_inputs(utils.prepare_image(img, vae))
    latent = latent.cpu().numpy().squeeze()
    return latent


@torch.no_grad()
def decode_latents(latents, vae):
    with torch.amp.autocast("cuda"):
        utils.set_seed(42)
        device = next(vae.parameters()).device  # Obtiene el dispositivo del modelo
        latents = torch.tensor(latents).to(device)
        latents = latents.unsqueeze(0)  # Añadir dimensión de batch

        img_decoded = vae.decode_stage_2_outputs(latents)

        img_decoded= img_decoded.detach().cpu().squeeze().numpy()
    return img_decoded



def run_encdec_synthesis(s_data, device):
    latens_list = s_data["latens_list"]

    unet = instantiate_encdec_model(device=device)
    to_modality_one_hot = torch.tensor(utils.create_modality_one_hot(configs.MISSING_MODALITY)).float().to(device)

    from_modality_latents = np.concatenate(latens_list, axis=0)
    from_modality_latents = torch.tensor(from_modality_latents).half().to(device).unsqueeze(0)


    with torch.no_grad(), torch.amp.autocast("cuda"):
        syn_latens = unet(
                            x=from_modality_latents,
                            modality_tensor=to_modality_one_hot,
                            )
        syn_latens = syn_latens.detach().cpu().squeeze(0).numpy()  # Detach and move to CPU
    torch.cuda.empty_cache()
    return syn_latens


def run_bbdm_synthesis(s_data, device):
    latens_list = s_data["latens_list"]

    unet, conditions_model, noise_scheduler = instantiate_bbdm_model(device=device)

    conditioning = torch.tensor([[[1.], [1.]]], device=device)
    conditioning_emb = conditions_model(conditioning)

    to_modality_index = configs.MODALITY_LIST.index(configs.MISSING_MODALITY)
    n_latent_channels = 4

    latens_list = utils.preprare_bbdm_latens(latens_list, to_modality_index)
    from_modality_latents = np.concatenate(latens_list, axis=0)
    from_modality_latents = torch.tensor(from_modality_latents).half().to(device).unsqueeze(0)

    utils.set_seed(42)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        latents_denoised = from_modality_latents

        for i in tqdm(range(noise_scheduler.sample_step)):
            t = noise_scheduler.steps[i]
            t_tensor = torch.zeros((from_modality_latents.shape[0],), device=device)
            t_tensor.fill_(t).to(device)

            noise_pred = unet(
                            x=latents_denoised,
                            timesteps=t_tensor,
                            context=conditioning_emb,
                            )
            latents_denoised, _ = noise_scheduler.backward_diffusion(i, latents_denoised, from_modality_latents, noise_pred)
        del noise_pred

        latents_denoised = latents_denoised.detach().cpu().squeeze(0).numpy()  # Detach and move to CPU
        syn_latens =  latents_denoised[n_latent_channels*to_modality_index:n_latent_channels*(to_modality_index+1)]
    torch.cuda.empty_cache()
    return syn_latens



# def run_synthesis(s_data, device):
def run_synthesis(s_data, synthesis_type, output_path, output_name, gpu_id=None, verbose=False, compute_bmask=False):
    # create output directory if it does not exist
    os.makedirs(output_path, exist_ok=True)

    # create intermediate output directory
    path_out_intermediate = os.path.join(output_path, f"intermediate_{s_data['s_id']}")
    os.makedirs(path_out_intermediate, exist_ok=True)

    # set output path name for the synthesized image
    path_name_syn_img = os.path.join(output_path, output_name)

    # set device
    device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None else "cpu")


    aff = None
    aff_preprocessed = None
    org_shape = None

    max_steps = 6 if compute_bmask else 5

    # 1.0: preprocessing
    if verbose:
        print(f"1/{max_steps} Preprocessing images for subject {s_data['s_id']}...")


    vae = instantiate_vae_model(device)

    latens_list = []
    for i, path_name_img in enumerate(s_data["path_name_img_list"]):
        img, aff = utils.load_nifti(path_name_img)
        org_shape = img.shape
        img, aff_preprocessed = utils.preprocessing(img, affine=aff)
        latents = encode_image(img, vae)

        # save latents for later use
        # np.save(os.path.join(path_out_intermediate, f"{s_data['s_id']}_{s_data['available_modalitites_names'][i]}_latents.npy"), latents)

        latens_list.append(latents)
    s_data["latens_list"] = latens_list

    if verbose:
        print(f"2/{max_steps} Running {synthesis_type} synthesis for subject {s_data['s_id']}...")

    # n.0: synthesis
    if synthesis_type in ("encdec", "ensamble"):
        if verbose and synthesis_type == "ensamble":
            print(f"2.1/{max_steps} Running encdec synthesis for subject {s_data['s_id']}...")
        syn_latens_encdec = run_encdec_synthesis(s_data, device)
    if synthesis_type in ("bbdm", "ensamble"):
        if verbose and synthesis_type == "ensamble":
            print(f"2.2/{max_steps} Running bbdm synthesis for subject {s_data['s_id']}...")
        syn_latens_bbdm = run_bbdm_synthesis(s_data, device)

    # m: postprocessing (ensamble)
    if verbose:
        print(f"3/{max_steps} Decoding generated latents for subject {s_data['s_id']}...")

    # m.1: decode latents (ensamble)
    if synthesis_type == "encdec":
        syn_img = decode_latents(syn_latens_encdec, vae)
    elif synthesis_type == "bbdm":
        syn_img = decode_latents(syn_latens_bbdm, vae)
    else:
        syn_img_encdec = decode_latents(syn_latens_encdec, vae)
        syn_img_bbdm = decode_latents(syn_latens_bbdm, vae)

        if verbose:
            print(f"3.1/{max_steps} Combining images from encdec and bbdm for subject {s_data['s_id']}...")

        syn_img = utils.combine_images([syn_img_encdec, syn_img_bbdm], combination_type='mean')

        path_name_raw_encdec_syn_img = os.path.join(path_out_intermediate, "raw_encdec_syn_img.nii.gz")
        path_name_raw_bbdm_syn_img = os.path.join(path_out_intermediate, "raw_bbdm_syn_img.nii.gz")
        utils.save_nifti(utils.postprocessing_raw(syn_img_encdec, org_shape), aff, path_name_raw_encdec_syn_img)
        utils.save_nifti(utils.postprocessing_raw(syn_img_bbdm, org_shape), aff, path_name_raw_bbdm_syn_img)

        # save_nifti(np.clip(syn_img_encdec, 0, 1), aff_preprocessed, path_name_raw_encdec_syn_img)
        # save_nifti(np.clip(syn_img_bbdm, 0, 1), aff_preprocessed, path_name_raw_bbdm_syn_img)

    # m.1: postprocessing
    bmask = None
    if compute_bmask:
        if verbose:
            print(f"4/{max_steps} Segmenting brain mask for subject {s_data['s_id']}...")
        bmask = segment_brain_mask.segment_brain_mask(s_data["path_name_img_list"], path_out_intermediate, s_data["available_modalitites_names"], gpu_id=gpu_id)


    step_count = 5 if bmask is not None else 4
    if verbose:
        print(f"{step_count}/{max_steps} Postprocessing synthesized image for subject {s_data['s_id']}...")

    syn_img = utils.postprocessing(syn_img, configs.MISSING_MODALITY, org_shape, bmask=bmask)

    # m.2: save image
    utils.save_nifti(syn_img, aff, path_name_syn_img)

    if verbose:
        step_count += 1
        print(f"{step_count}/{max_steps} Synthesis completed for subject {s_data['s_id']}\nOutput saved to: {path_name_syn_img}")