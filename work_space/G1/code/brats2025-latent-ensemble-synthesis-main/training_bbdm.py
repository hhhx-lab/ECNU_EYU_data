import json
import os
import numpy as np
from tqdm import tqdm
import datetime



# pytorch
import torch
try:
    from torch.amp import GradScaler, autocast
    _TORCH_AMP_NEW = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _TORCH_AMP_NEW = False

def _autocast(enabled=True):
    """Compatibility wrapper for autocast across torch versions."""
    if _TORCH_AMP_NEW:
        return autocast("cuda", enabled=enabled)
    else:
        return autocast(enabled=enabled)
from torch.utils.tensorboard import SummaryWriter
from torch.utils.checkpoint import checkpoint

# mine
import configs
import synthesis.utils as utils
import synthesis.dataloader as dataloader

# monai
from monai.bundle import ConfigParser
import models.bbdm.unet as diffusion_model_unet_maisi
import models.bbdm.bb_scheduler as bb_scheduler
import models.bbdm.condition_tokens as condition_tokens

# images
from PIL import Image

# device_name = f"cuda:{gpu_selector.get_least_used_gpu()}"
gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
device_name = f"cuda:{gpu_id}"
device = torch.device(device_name)

print("Flash SDP enabled:", torch.backends.cuda.flash_sdp_enabled())

def set_seed(seed: int):
    # random.seed(seed)  # Semilla para Python
    np.random.seed(seed)  # Semilla para NumPy
    torch.manual_seed(seed)  # Semilla para PyTorch en CPU
    torch.cuda.manual_seed(seed)  # Semilla para PyTorch en GPU
    torch.cuda.manual_seed_all(seed)  # Semilla para todas las GPUs
    torch.backends.cudnn.deterministic = True  # Garantizar reproducibilidad en CNNs
    torch.backends.cudnn.benchmark = False  # Desactivar optimización no determinista








def instantiate_unconditioned_models(device, use_conditions_self_attention, reserved_space=0):

    networks_config =  {
        "autoencoder_def": {
            "_target_": "monai.apps.generation.maisi.networks.autoencoderkl_maisi.AutoencoderKlMaisi",
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "latent_channels": 4,
            "num_channels": [
                64,
                128,
                256
            ],
            "num_res_blocks": [2,2,2],
            "norm_num_groups": 32,
            "norm_eps": 1e-06,
            "attention_levels": [
                False,
                False,
                False
            ],
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
            "use_checkpointing": False,
            "use_convtranspose": False,
            "norm_float16": True,
            "num_splits": 8,
            "dim_split": 1
        },

        "diffusion_unet_def": {
            "_target_": "monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi",
            "spatial_dims": 3,
            "in_channels": 4*4,
            "out_channels": 4*4,
            "num_res_blocks": 2,
            "num_channels": [
                64,
                128,
                256,
                512
            ],
            "attention_levels": [
                False,
                False,
                True,
                True
            ],
            "num_head_channels": [
                0,
                0,
                32,
                32
            ],
            "use_flash_attention": True, # keep it true for self attention, in cross attenttion you will not implement it
            "with_conditioning": True,
            "cross_attention_dim": 256,
            "transformer_num_layers": 1, # number of transformer blocks
            "upcast_attention": True,


            # for the first test i will not use the conditioning, i will just convert from t2w to t2f
            "include_from_modality": False,
            "include_to_modality": False,
        },

        # "noise_scheduler": {
        #     "_target_": "monai.networks.schedulers.DDIMScheduler", # faster scheduler
        #     "beta_start": 0.0015,
        #     "beta_end": 0.0205,
        #     "num_train_timesteps": 1000,
        #     "schedule": "scaled_linear_beta",
        #     "clip_sample": False
        # }

        "bb_scheduler": {
            "num_timesteps": 1000,
            "sample_step": 50,
            "s": 0.01, # default 1.0
            "sample_type": "linear",
            "objective_type": "grad"
        },

        "conditions_model": {
            "num_conditions": 2,  # number of conditions
            "embed_dim": 256,  # same as the cross_attention_dim in the unet
            "hidden_dim": 128,  # half of the embedding dimension
            "use_self_attention": use_conditions_self_attention,  # whether to use self-attention
            "n_heads": 8,  # number of attention heads
            "n_layers": 1,  # number of layers in the MLP
        }
    }

    # instantiate model
    parser = ConfigParser(networks_config)
    parser.parse(True)

    args = utils.dict_to_args(networks_config, deep_conversion=True)

    # unet
    unet = diffusion_model_unet_maisi.DiffusionModelUNetMaisi(spatial_dims = args.diffusion_unet_def.spatial_dims,
                                                in_channels = args.diffusion_unet_def.in_channels,
                                                out_channels = args.diffusion_unet_def.out_channels,
                                                num_res_blocks = args.diffusion_unet_def.num_res_blocks,
                                                num_channels = args.diffusion_unet_def.num_channels,
                                                attention_levels = args.diffusion_unet_def.attention_levels,
                                                #    norm_num_groups = args.diffusion_unet_def.norm_num_groups,
                                                #    norm_eps = args.diffusion_unet_def.norm_eps,
                                                #    resblock_updown = args.diffusion_unet_def.resblock_updown,
                                                num_head_channels = args.diffusion_unet_def.num_head_channels,
                                                with_conditioning = args.diffusion_unet_def.with_conditioning,
                                                use_flash_attention = args.diffusion_unet_def.use_flash_attention,
                                                include_from_modality = args.diffusion_unet_def.include_from_modality,
                                                include_to_modality = args.diffusion_unet_def.include_to_modality,
                                                cross_attention_dim = args.diffusion_unet_def.cross_attention_dim,
                                                transformer_num_layers = args.diffusion_unet_def.transformer_num_layers,
                                                upcast_attention = args.diffusion_unet_def.upcast_attention,
                                                )

    # ConditionTokens
    conditions_model = condition_tokens.ConditionTokens(num_conditions=args.conditions_model.num_conditions,
                                    embed_dim=args.conditions_model.embed_dim,
                                    hidden_dim=args.conditions_model.hidden_dim,
                                    use_self_attention=args.conditions_model.use_self_attention,
                                    n_heads=args.conditions_model.n_heads,
                                    n_layers=args.conditions_model.n_layers)



    # noise scheduler
    # noise_scheduler = parser.get_parsed_content("noise_scheduler", instantiate=True)
    # noise_scheduler.set_timesteps(num_inference_steps=50)

    noise_scheduler = bb_scheduler.Scheduler(
        num_timesteps=args.bb_scheduler.num_timesteps,
        sample_step=args.bb_scheduler.sample_step,
        s=args.bb_scheduler.s,
        sample_type=args.bb_scheduler.sample_type,
        objective_type=args.bb_scheduler.objective_type
    )

    # autoencoder (just for validation)
    autoencoder = parser.get_parsed_content("autoencoder_def").to(device)
    checkpoint_autoencoder = torch.load(configs.PATH_NAME_WEIGHTS_VAE, weights_only=True, map_location=device)
    autoencoder.load_state_dict(checkpoint_autoencoder)
    autoencoder.eval()

    return {"unet": unet,
            "conditions_model": conditions_model,
            "autoencoder": autoencoder,
            "noise_scheduler": noise_scheduler,
            "networks_config": args,}

# instantiate_unconditioned_models(device, reserved_space=0)







def instantiate_dataset(df_path, data_path, batch_size, gen_dataloader, split="train", to_modality_name="t2w", mode="3-to-1", from_modality_name=None, seed=42, max_images=None, new_shape=None, load_only_latents=True, attmasks_path=None, attmasks_shapes_list=None):
    # ---- Data set creation
    train_dataset = dataloader.PrepareDataset(
        df_path = df_path,
        data_path = data_path,
        split=split,
        to_modality_name = to_modality_name,
        mode=mode,
        from_modality_name=from_modality_name,
        seed=seed,
        max_images=max_images,
        new_shape=new_shape, #(64, 64, 48) # 64x64x48
        load_only_latents=load_only_latents,
        attmasks_path=attmasks_path,
        attmasks_shapes_list=attmasks_shapes_list,

    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True if split == "train" else False,
        collate_fn=lambda examples: dataloader.collate_fn(examples),
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=True,
        generator=gen_dataloader,
    )

    return train_dataloader

def find_bigger_mask_layers(mask, return_layer_offset=False):
    _bigger_dim_x = np.argmax(np.sum(mask, axis=(1,2)))
    _bigger_dim_y = np.argmax(np.sum(mask, axis=(0,2)))
    _bigger_dim_z = np.argmax(np.sum(mask, axis=(0,1)))

    if return_layer_offset:
        layer_offset = np.array((_bigger_dim_x, _bigger_dim_y, _bigger_dim_z)) - np.array(mask.shape) // 2
        return (_bigger_dim_x, _bigger_dim_y, _bigger_dim_z), layer_offset
    else:
        return (_bigger_dim_x, _bigger_dim_y, _bigger_dim_z)






def cat_n_views_different_layers(imgs3D_list, view_layersoffset_list, axis=0, img_cropping=0, to_uint8=False, normalize=False):
    # set all images isotropic
    bigger_dim = max(imgs3D_list[0].shape) - img_cropping
    imgs3D_iso_list = []
    for __img in imgs3D_list:
        # verify if img is not rgb
        if len(__img.shape) == 3:
            __img = utils.gray_to_rgb(__img, to_uint8=to_uint8, normalize=normalize)
        imgs3D_iso_list.append(np.stack([utils.resize_center_crop_pad(__img[...,rgb_i], [bigger_dim]*3)[0] for rgb_i in range(3)], axis=-1))

    layer = [bigger_dim // 2]*len(view_layersoffset_list)
    imgs2D_list = []
    for _img in imgs3D_iso_list:
        __c_image = []
        for i, (view, layeroffset) in enumerate(view_layersoffset_list):
            slice_tuple = [slice(None)] * _img.ndim
            slice_tuple[view] = layer[i] + layeroffset
            __c_image.append(np.flipud(_img[tuple(slice_tuple)].transpose(1,0,2)))
        imgs2D_list.append(np.concatenate(__c_image, axis=axis))

    return imgs2D_list


def cat_3_views(imgs3D_list, axis=0, layer_offset=None, img_cropping=0, normalize=False):
    if layer_offset is None:
        layer_offset = [0, 0, 0]
    imgs_2D_list = cat_n_views_different_layers(imgs3D_list,
                                            view_layersoffset_list=[(2, layer_offset[2]), (1, layer_offset[1]), (0, layer_offset[0])],
                                            axis=axis,
                                            img_cropping=img_cropping,
                                            normalize=normalize,)
    return imgs_2D_list


@torch.no_grad()
def validation(
    unet,
    conditions_model,
    autoencoder,
    noise_scheduler,
    step,
    args,
    save_lattents=False,
):

    print(f"Validation step {step}")
    unet.eval()
    conditions_model.eval()

    attmasks_shapes_list=[(256,256,160)]

    gen_dataloader = torch.Generator().manual_seed(args.val_seeds[0])
    val_dataloader = instantiate_dataset(
        df_path = args.df_path,
        data_path = args.latents_path,
        batch_size = min(4,args.nb_val_images),
        gen_dataloader = gen_dataloader,
        split="val",
        to_modality_name = args.to_modality_name,
        mode=args.dataloader_mode,
        from_modality_name = args.from_modality_name,
        seed=args.val_seeds[0],
        max_images=args.nb_val_images,
        new_shape=None,
        load_only_latents=False,
        attmasks_path=args.val_attmasks_path,
        attmasks_shapes_list=attmasks_shapes_list,
        )

    syn_imgs_list = []
    org_img_list = []
    tumor_mask_list = []
    lattents_list = []
    layer_offset_list = []

    n_latent_channels = 4
    # modality_names = ["t1n", "t1c", "t2w", "t2f"]

    for i, batch in enumerate(val_dataloader):
        for cond_i, conditions in enumerate(args.val_conditions_list):
            from_modality_latents = batch["from_modality_latents"].half().to(device)
            # preparing conditioning
            conditioning = torch.tensor([[
                                            conditions["healthy"],
                                            conditions["tumor"]]]).to(device).float().unsqueeze(1).permute(0,2,1)

            conditioning_emb = conditions_model(conditioning)

            with torch.no_grad(), _autocast():
                latents_denoised = from_modality_latents

                for i in tqdm(range(noise_scheduler.sample_step)):
                    t = noise_scheduler.steps[i]
                    t_tensor = torch.zeros((from_modality_latents.shape[0],), device=device)
                    t_tensor.fill_(t).to(device)

                    noise_pred = unet(
                                    x=latents_denoised,
                                    timesteps=t_tensor,
                                    # to_modality_tensor=to_modality_one_hot,
                                    context=conditioning_emb,
                                    )
                    latents_denoised, _ = noise_scheduler.backward_diffusion(i, latents_denoised, from_modality_latents, noise_pred)

                del noise_pred
                torch.cuda.empty_cache()

                for bt, __lt_denoised in enumerate(latents_denoised): # loop over the subjetcs

                    # obtain the important modality
                    to_modality_one_hot = batch["to_modality_one_hot"][bt]
                    to_modality_index = torch.argmax(to_modality_one_hot).item()

                    __lt_mod_denoised = __lt_denoised[to_modality_index*n_latent_channels:n_latent_channels*(to_modality_index+1)]
                    img_syn = autoencoder.decode_stage_2_outputs(__lt_mod_denoised.unsqueeze(0))
                    img_syn = torch.clip(img_syn, 0.0, 1.0).cpu().squeeze().numpy()

                    if cond_i == 0:
                        img_org_show = torch.clip(batch["to_modality_images"][bt][to_modality_index], 0.0, 1.0).squeeze().numpy()
                        tumor_mask = batch[f"attmasks_{'_'.join(str(x) for x in attmasks_shapes_list[0])}"][bt].squeeze().numpy().astype(np.float32)
                        tumor_mask = np.clip(tumor_mask, 0.0, 1.0)
                        bmask = np.where(img_org_show > 0.1, 1, 0)

                        _, layer_offset = find_bigger_mask_layers(tumor_mask, return_layer_offset=True)

                        org_img_list.append(cat_3_views([img_org_show], axis=0, layer_offset=layer_offset, img_cropping=0)[0])
                        tumor_mask_list.append(cat_3_views([tumor_mask+bmask], axis=0, layer_offset=layer_offset, img_cropping=0, normalize=True)[0])
                        layer_offset_list.append(layer_offset)
                        # layer_offset_list.append([0,0,0])

                    img_syn_show = cat_3_views([img_syn], axis=0, layer_offset=layer_offset_list[bt], img_cropping=0)[0]
                    syn_imgs_list.append(img_syn_show)


     # organize syntetic images to match with the original images
    imgs_list = []
    nb_subjects = len(org_img_list)
    nb_conditions = len(syn_imgs_list) // nb_subjects
    print(f"nb_subjects: {nb_subjects}, nb_conditions: {nb_conditions}")
    for i in range(nb_subjects):
        s_imgs = [org_img_list[i], tumor_mask_list[i]] + syn_imgs_list[i::nb_subjects]
        imgs_list.append(np.concatenate(s_imgs, axis=1)) # concatenate all images in the same row


    complete_img = np.concatenate(imgs_list, axis=0)
    print(f"complete_img shape: {complete_img.shape}")
    complete_img = Image.fromarray((complete_img*255).astype(np.uint8))
    os.makedirs(f"{args.output_path}/{args.val_imgs_dir_name}", exist_ok=True)
    complete_img.save(f"{args.output_path}/{args.val_imgs_dir_name}/imgs_step_{step}.png" )


    unet.train()
    conditions_model.train()




def test_validation():
    args = {
            "output_path": os.path.join(configs.PATH_TRAINING, "bbdm/test_val"),
            "val_imgs_dir_name": "val_imgs2",
            "val_seeds": [42],

            "df_path" : "/data_csv.csv",
            "latents_path" : "latents",
            "attmasks_path": "attention_masks",
            "val_attmasks_path": "attention_masks_val",

            "to_modality_name": "t2w",
            "from_modality_name": None,
            "dataloader_mode": "4b-to-4",
            "val_conditions_list": [
                {"healthy": 1.0, "tumor": 0.0},
                {"healthy": 0.0, "tumor": 1.0},
                # {"healthy": 0.5, "tumor": 0.5}
            ],
                }

    args = utils.dict_to_args(args, deep_conversion=True)

    models_dict = instantiate_unconditioned_models(device, False, 0)
    unet = models_dict["unet"]
    autoencoder = models_dict["autoencoder"]
    noise_scheduler = models_dict["noise_scheduler"]
    conditions_model = models_dict["conditions_model"]

    unet.to(device)
    conditions_model.to(device)
    validation(
        unet,
        conditions_model,
        autoencoder,
        noise_scheduler,
        step=0,
        args=args,
        save_lattents=True,
    )

# test_validation()








def save_model(unet, conditions_model, optimizer, lr_scheduler, global_step, out_model_path, ema=None):  # MOD: se añade parámetro ema
    # Guardar el modelo
    unet_state_dict = unet.module.state_dict() if torch.distributed.is_initialized() else unet.state_dict()
    checkpoint = {
        "unet_state_dict": unet_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "num_train_timesteps": global_step,
        "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "conditions_model_state_dict": conditions_model.state_dict(),
    }
    # MOD: Agregar los pesos EMA en el checkpoint
    if ema is not None:
        checkpoint["ema_state_dict"] = ema.shadow
    torch.save(checkpoint, f"{out_model_path}/model_{global_step}.pt")
    print(f"Modelo guardado en {out_model_path}/model_{global_step}.pt")



def save_configurations(args_train, networks_config, config_path, config_name="model_config.json"):
    argparse_dict = {
        "args_train": utils.args_to_dict(args_train, deep_conversion=True),
        "networks_config": utils.args_to_dict(networks_config, deep_conversion=True)
    }
    argparse_json = json.dumps(argparse_dict, indent=4)
    with open(os.path.join(config_path, config_name), "w") as outfile:
        outfile.write(argparse_json)
    print(f"Model configurations saved in: {os.path.join(config_path, config_name)}")







def train(
    args_train,
    device,
    reserved_space=0,
):

    # ---- reproducibility
    set_seed(args_train.seed)
    gen_t = torch.Generator().manual_seed(args_train.seed)
    gen_noise = torch.Generator().manual_seed(args_train.seed)
    gen_dataloader = torch.Generator().manual_seed(args_train.seed)
    gen_cond_guidance = torch.Generator().manual_seed(args_train.seed)

    # gen_missing_modality = torch.Generator().manual_seed(args_train.seed)

    # ---- instantiate models
    models_dict = instantiate_unconditioned_models(device, False, reserved_space)
    unet = models_dict["unet"]
    conditions_model = models_dict["conditions_model"]
    noise_scheduler = models_dict["noise_scheduler"]
    autoencoder = models_dict["autoencoder"]
    networks_config = models_dict["networks_config"]

    # ---- instantiate dataset
    train_dataloader = instantiate_dataset(
        df_path = args_train.df_path,
        data_path = args_train.latents_path,
        batch_size = args_train.batch_size,
        gen_dataloader = gen_dataloader,
        split="train",
        to_modality_name = args_train.to_modality_name,
        mode=args_train.dataloader_mode,
        from_modality_name = args_train.from_modality_name,
        seed=args_train.seed,
        load_only_latents=True,
        attmasks_path=args_train.attmasks_path,
        attmasks_shapes_list=args_train.attmasks_shapes_list,
        )

    # ---- create folders
    os.makedirs(args_train.output_path, exist_ok=True)
    _checkpoint_dir_name =  os.path.join(args_train.output_path, args_train.checkpoints_dir_name)
    _logs_dir_name = os.path.join(args_train.output_path, args_train.logs_dir_name)
    _val_imgs_dir_name = os.path.join(args_train.output_path, args_train.val_imgs_dir_name)
    os.makedirs(_checkpoint_dir_name, exist_ok=True)
    os.makedirs(_logs_dir_name, exist_ok=True)
    os.makedirs(_val_imgs_dir_name, exist_ok=True)

    # ---- save configurations
    save_configurations(args_train, networks_config, args_train.output_path)

    # ---- create tensorboard writer and save configurations
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")  # Formato: Año-Mes-Día_Hora-Minuto
    _sum_writter_dir = os.path.join(_logs_dir_name, f"logs_{timestamp}")
    os.makedirs(_sum_writter_dir, exist_ok=True)
    writer = SummaryWriter(_sum_writter_dir)

    # ---- optimizer and lr_scheduler
    # optimizer = torch.optim.Adam(params=unet.parameters(), lr=args_train.lr) # for maisi  1e-4 # for blsmd 2.5e-5
    optimizer = torch.optim.Adam(
        list(unet.parameters()) + list(conditions_model.parameters()),
        lr=args_train.lr,
        weight_decay=args_train.weight_decay,
    )
    if args_train.lr_scheduler is not None:
        if args_train.lr_scheduler.name == "PolynomialLR":
            lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args_train.max_train_steps, power=args_train.lr_scheduler.power)
    else:
        lr_scheduler = None

    # ---- loss function
    loss_pt = torch.nn.MSELoss()

    def loss_pt_per_channel(x, y, to_mod_index, channel_importance_weights=None, target_modality_weight=1, extra_modalites_weight=1):
        # x, y: B x (4 * n_modalities) x W x H x D
        total_loss = 0
        n_latent_channels = 4
        n_modalities = x.shape[1] // n_latent_channels

        for i in range(n_modalities):
            init_modality = i * n_latent_channels
            modality_loss = 0
            for j in range(n_latent_channels):
                x_ch = x[:, init_modality + j:init_modality + j + 1, ...]
                y_ch = y[:, init_modality + j:init_modality + j + 1, ...]
                ch_loss = loss_pt(x_ch, y_ch)

                if channel_importance_weights is not None:
                    ch_loss *= channel_importance_weights[j]

                modality_loss += ch_loss

            if to_mod_index == i:
                modality_loss *= target_modality_weight
            else:
                modality_loss *= extra_modalites_weight

            total_loss += modality_loss

        return total_loss / n_modalities


    # ---- training loop
    first_epoch = 0
    global_step = 0
    max_epochs = (args_train.max_train_steps * args_train.gradient_accumulation_steps * args_train.batch_size) // len(train_dataloader) + 1
    print(f"Max epochs: {max_epochs}")

    # ---- resume from checkpoint
    unet.to(device)
    conditions_model.to(device)

    # Initilize ema
    if args_train.use_ema:
        ema = utils.EMA(unet, decay=args_train.ema_params.decay, warm_up_steps=args_train.ema_params.warm_up_steps, warm_up_decay=args_train.ema_params.warm_up_decay)
    else:
        ema = None

    # priority is to resume from check point
    if args_train.resume_from_checkpoint_path_name is not None:
        checkpoint = torch.load(args_train.resume_from_checkpoint_path_name, map_location=device_name)
        unet.load_state_dict(checkpoint["unet_state_dict"], strict=False)
        conditions_model.load_state_dict(checkpoint["conditions_model_state_dict"], strict=False)
        if args_train.use_ema and "ema_state_dict" in checkpoint:
            ema.shadow = checkpoint["ema_state_dict"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        global_step = checkpoint["num_train_timesteps"]
        first_epoch = (global_step * args_train.gradient_accumulation_steps * args_train.batch_size) // len(train_dataloader) + 1
        print(f"Model loaded from {args_train.resume_from_checkpoint_path_name}")
        print(f"Resuming training from epoch {first_epoch} and global step {global_step}")

    elif args_train.load_pretrained_model_from is not None:
        checkpoint = torch.load(args_train.load_pretrained_model_from, weights_only=False, map_location=device_name)
        unet.load_state_dict(checkpoint["unet_state_dict"], strict=False)
        print(f"Pretrained model loaded from {args_train.load_pretrained_model_from}")

    unet.train()
    conditions_model.train()#$$

    # ---- memory reduction
    # -------- automatic mixed precision
    if args_train.amp:
        scaler = GradScaler()
    else:
        scaler = None
    gradient_accumulation_count = 0

    # ---- training loop
    progress_bar = tqdm(
        range(0, args_train.max_train_steps),
        desc="Steps",
        initial=global_step
    )

    for epoch in range(first_epoch, max_epochs):
        for batch in train_dataloader:

            # prepare inputs
            from_modality = batch["from_modality_latents"].to(device)
            to_modality = batch["to_modality_latents"].to(device)
            # from_modality_one_hot = batch["from_modality_one_hot"].to(device)
            # to_modality_one_hot = batch["to_modality_one_hot"].to(device)

            # integer [0,1]
            # tumor_cond = torch.randint(0,2,(from_modality.shape[0],), device="cpu", generator=gen_cond_guidance).float().to(device)
            # float [0.0,1.0]
            # tumor_cond = torch.rand(from_modality.shape[0], device="cpu", generator=gen_cond_guidance).to(device)
            # healthy_cond = 1- tumor_cond

            tumor_cond = torch.tensor([1]).to(device).float()
            healthy_cond = torch.tensor([1]).to(device).float()
            conditioning = torch.stack([healthy_cond, tumor_cond], dim=1)
            conditioning = conditioning.unsqueeze(-1)

            attmask_64_64_40 = batch.get("attmasks_64_64_40", torch.ones_like(from_modality[:, :1, ...])).to(device)


            # Forward pass
            with _autocast(enabled=args_train.amp):
                # generate noise and timesteps with dedicate generatos and in the cpu for reproducibility
                noise = torch.randn(to_modality.shape, device="cpu", generator=gen_noise).to(device)
                timesteps = torch.randint(0, noise_scheduler.num_timesteps, (from_modality.shape[0],), device="cpu", generator=gen_t).long().to(device)

                noisy_latent, objective = noise_scheduler.forward_diffusion(t=timesteps, x_0=to_modality, x_T=from_modality, noise=noise)
                conditioning_emb = conditions_model(conditioning)

                noise_pred = unet(x=noisy_latent,
                                  timesteps=timesteps,
                                    # from_modality_tensor=from_modality_one_hot,
                                    # to_modality_tensor=to_modality_one_hot,
                                    context=conditioning_emb,
                                    )

                __to_mod_index = torch.argmax(batch["to_modality_one_hot"][0].cpu(), dim=0)
                # loss = loss_pt_per_channel(noise_pred.float(), objective.float(), __to_mod_index, channel_importance_weights=args_train.channel_importance_weights, target_modality_weight=args_train.target_modality_weight) / args_train.gradient_accumulation_steps  # Dividir para escalar la pérdida
                loss_healthy = loss_pt_per_channel(noise_pred.float() * (1-attmask_64_64_40) * healthy_cond[:, None, None, None, None],
                                                   objective.float() * (1-attmask_64_64_40) * healthy_cond[:, None, None, None, None],
                                                   __to_mod_index,
                                                   channel_importance_weights=args_train.channel_importance_weights,
                                                   target_modality_weight=args_train.target_modality_weight, extra_modalites_weight=args_train.extra_modalites_weight)
                loss_tumor = loss_pt_per_channel(noise_pred.float() * attmask_64_64_40 * tumor_cond[:, None, None, None, None],
                                                    objective.float() * attmask_64_64_40 * tumor_cond[:, None, None, None, None],
                                                    __to_mod_index,
                                                    channel_importance_weights=args_train.channel_importance_weights,
                                                    target_modality_weight=args_train.target_modality_weight, extra_modalites_weight=args_train.extra_modalites_weight)
                loss = (loss_healthy + loss_tumor) / args_train.gradient_accumulation_steps
            # Acumulación de gradientes
            if args_train.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            gradient_accumulation_count += 1  # Contador de pasos acumulados

            # Solo se actualizan los pesos cada `gradient_accumulation_steps` pasos
            if gradient_accumulation_count % args_train.gradient_accumulation_steps == 0:
                if args_train.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if args_train.use_ema:
                    ema.update(step=global_step)

                if lr_scheduler is not None:
                    lr_scheduler.step()

                optimizer.zero_grad()  # Solo hacer zero_grad() después de actualizar los pesos
                gradient_accumulation_count = 0  # Reiniciar el contador


                # update writter
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], global_step)
                # __from_mod_index = torch.argmax(batch["from_modality_one_hot"][0].cpu(), dim=0)
                # writer.add_scalar("from modality index", __from_mod_index, global_step)
                writer.add_scalar("to modality index", __to_mod_index, global_step)
                writer.add_scalar("healthy_cond", healthy_cond[0], global_step)
                writer.add_scalar("tumor_cond", tumor_cond[0], global_step)

                writer.add_scalar("to_modality_latents_min", batch["to_modality_latents"].min(), global_step)
                writer.add_scalar("to_modality_latents_max", batch["to_modality_latents"].max(), global_step)
                writer.add_scalar("noisy_latent_min", noisy_latent.min(), global_step)
                writer.add_scalar("noisy_latent_max", noisy_latent.max(), global_step)
                writer.add_scalar("objective_min", objective.min(), global_step)
                writer.add_scalar("objective_max", objective.max(), global_step)


                # update progress bar
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(),
                        # "fm_index": __from_mod_index,
                        "tm_index": __to_mod_index.item(),
                        "healthy_cond": healthy_cond[0].item(),
                        "tumor_cond": tumor_cond[0].item(),
                        }
                progress_bar.set_postfix(**logs)

                # update global step
                global_step += 1

                # # Guardar modelo cada cierto intervalo
                if global_step % args_train.save_checkpoint_interval == 0:
                    save_model(unet, conditions_model, optimizer, lr_scheduler, global_step, _checkpoint_dir_name, ema=ema, )

                # Generar imágenes en intervalos
                if args_train.initial_val or global_step % args_train.val_interval == 0:
                    try:
                        if args_train.use_ema:
                            ema.apply_shadow()
                        validation(unet, conditions_model, autoencoder, noise_scheduler, global_step, args_train, save_lattents=True)
                    except Exception as e:
                        print(f"ERROR DURING VALIDATION STEP {global_step}: {e}")
                    finally:
                        if args_train.use_ema:
                            ema.restore()
                    args_train.initial_val = False

                if global_step >= args_train.max_train_steps:
                    break

        if global_step >= args_train.max_train_steps:
            break

    # make sure the progress bar closes
    progress_bar.close()

    # make  out_model_path dir if it does not exist
    save_model(unet, conditions_model, optimizer, lr_scheduler, global_step, _checkpoint_dir_name, ema=ema, )






args_train = {
    # directories
    "output_path": os.path.join(configs.PATH_TRAINING, "bbdm"),

    "checkpoints_dir_name": "check_points",
    "logs_dir_name": "logs",
    "val_imgs_dir_name": "val_imgs",

    # data
    "df_path": os.path.join(configs.PATH_DATA, "data_csv.csv"),
    "latents_path": os.path.join(configs.PATH_DATA, "latents"),
    "attmasks_path": os.path.join(configs.PATH_DATA, "attention_masks"),
    "val_attmasks_path": os.path.join(configs.PATH_DATA, "attention_masks"),
    "to_modality_name": "t2w",
    "from_modality_name": None,
    "dataloader_mode": "4b-to-4",

    # training configuration (1080 epochs × 1489 subjects / batch_size=4 ≈ 402000)
    "max_train_steps": 402000,
    "save_checkpoint_interval": 5000,

    # ---- memory reduction
    "amp": True,

    # ---- Training stability
    "batch_size": 4, # possible 6 (no flash att), 8 (flash att)
    "gradient_accumulation_steps": 1,
    "use_ema": True,
    "ema_params": {
        "decay": 0.8, # weght of the memory
        "warm_up_steps": 2000,
        "warm_up_decay": 0.2, # weght of the memory
    },

    # ---- optimizer
    "lr": 1e-4, # for maisi  1e-4 # for blsmd 2.5e-5
    "weight_decay": 0.0, # set to 1e-5 ~ 1e-4 if overfitting is observed

    # ---- lr_scheduler
    "lr_scheduler": {"name": "PolynomialLR", "power": 2.0},
    # "lr_scheduler": None,

    # ---- pretrained_model
    "load_pretrained_model_from": None, # not working


    # ---- resume from checkpoint
    "resume_from_checkpoint_path_name": None,

    # reproducibility
    "seed": 42,
    "val_seeds": [42],


    # validation
    "val_interval": 500,
    "initial_val": True,

    # weighting loss per channels
    "channel_importance_weights": [0.50995545, 0.12407539, 0.20278341, 0.16318575], # obtained from compute_channels_importance using rms
    # "channel_importance_weights": None,

    "target_modality_weight": 1., # for the extra modalities (t1n, t1c, t2w) in the 4b-to-4 mode
    "extra_modalites_weight":0.0,

    "attmasks_shapes_list":[(64, 64, 40)],
    "val_conditions_list": [
        {"healthy": 1.0, "tumor": 1.0},
        {"healthy": 0.0, "tumor": 0.0},
        # {"healthy": 1.0, "tumor": 0.0},
        # {"healthy": 0.0, "tumor": 1.0},
        # {"healthy": 0.5, "tumor": 0.5},
        ],
    "nb_val_images": 4, # number of images to show in the validation step
}


args_train = utils.dict_to_args(args_train, deep_conversion=True)
train(
    args_train,
    device,
    reserved_space=0
)
