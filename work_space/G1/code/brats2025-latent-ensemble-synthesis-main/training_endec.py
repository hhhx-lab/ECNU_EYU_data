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

# mine
import configs
import synthesis.utils as utils
import synthesis.dataloader as dataloader


# monai
from monai.bundle import ConfigParser
import models.encdec.unet as diffusion_model_unet_maisi

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





def instantiate_unconditioned_models(device, reserved_space=0):

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
            "_target_": "models.encdec.unet.DiffusionModelUNetMaisi",
            "spatial_dims": 3,
            "in_channels": 4*3,
            "out_channels": 4,
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
            "use_flash_attention": False,
            "with_conditioning": False,
        },
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
                                                )


    # autoencoder (just for validation)
    autoencoder = parser.get_parsed_content("autoencoder_def").to(device)
    checkpoint_autoencoder = torch.load(configs.PATH_NAME_WEIGHTS_VAE, weights_only=True, map_location=device)
    autoencoder.load_state_dict(checkpoint_autoencoder)
    autoencoder.eval()



    return {"unet": unet,
              "autoencoder": autoencoder,
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
    autoencoder,
    step,
    args,
):

    print(f"Validation step {step}")
    unet.eval()
    imgs_list = []

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
    layer_offset_list = []



    for i, batch in enumerate(val_dataloader):

        from_modality_latents = batch["from_modality_latents"].half().to(device)
        to_modality_one_hot = batch["to_modality_one_hot"].float().to(device)

        with torch.no_grad(), _autocast():
            missing_lattent_pred = unet(
                            x=from_modality_latents,
                            modality_tensor=to_modality_one_hot,
                            )

            torch.cuda.empty_cache()

            for bt, __lt_denoised in enumerate(missing_lattent_pred): # loop over the subjetcs
                to_modality_one_hot = batch["to_modality_one_hot"][bt]

                img_syn = autoencoder.decode_stage_2_outputs(__lt_denoised.unsqueeze(0))
                img_syn = torch.clip(img_syn, 0.0, 1.0).cpu().squeeze().numpy()


                img_org_show = torch.clip(batch["to_modality_images"][bt][0], 0.0, 1.0).squeeze().numpy()
                if f"attmasks_{'_'.join(str(x) for x in attmasks_shapes_list[0])}" in batch:
                    tumor_mask = batch[f"attmasks_{'_'.join(str(x) for x in attmasks_shapes_list[0])}"][bt].squeeze().numpy().astype(np.float32)
                else:
                    tumor_mask = np.zeros_like(img_org_show)
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






def save_model(unet, optimizer, lr_scheduler, global_step, out_model_path, ema=None):  # MOD: se añade parámetro ema
    # Guardar el modelo
    unet_state_dict = unet.module.state_dict() if torch.distributed.is_initialized() else unet.state_dict()
    checkpoint = {
        "unet_state_dict": unet_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "num_train_timesteps": global_step,
        "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
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
    gen_dataloader = torch.Generator().manual_seed(args_train.seed)
    gen_missing_modality = torch.Generator().manual_seed(args_train.seed)

    # ---- instantiate models
    models_dict = instantiate_unconditioned_models(device, reserved_space)
    unet = models_dict["unet"]
    autoencoder = models_dict["autoencoder"]
    networks_config = models_dict["networks_config"]

    # ---- instantiate dataset
    train_dataloader = instantiate_dataset(
        df_path=args_train.df_path,
        data_path=args_train.latents_path,
        batch_size=args_train.batch_size,
        gen_dataloader=gen_dataloader,
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
        list(unet.parameters()),
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

    # ---- training loop
    first_epoch = 0
    global_step = 0
    max_epochs = (args_train.max_train_steps * args_train.gradient_accumulation_steps * args_train.batch_size) // len(train_dataloader) + 1
    print(f"Max epochs: {max_epochs}")

    # ---- resume from checkpoint
    unet.to(device)

    # Initilize ema
    if args_train.use_ema:
        ema = utils.EMA(unet, decay=args_train.ema_params.decay, warm_up_steps=args_train.ema_params.warm_up_steps, warm_up_decay=args_train.ema_params.warm_up_decay)
    else:
        ema = None

    # priority is to resume from check point
    if args_train.resume_from_checkpoint_path_name is not None:
        checkpoint = torch.load(args_train.resume_from_checkpoint_path_name, map_location=device_name)
        unet.load_state_dict(checkpoint["unet_state_dict"], strict=False)
        if args_train.use_ema and "ema_state_dict" in checkpoint:
            ema.shadow = checkpoint["ema_state_dict"]
        if not args_train.extra_steps:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if lr_scheduler is not None:
            if args_train.extra_steps:
                lr_scheduler.last_epoch = checkpoint["num_train_timesteps"]
            else:
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

            # concatenate the rest of the modalities
            from_modality = batch["from_modality_latents"].to(device)
            to_modality = batch["to_modality_latents"].to(device)
            to_modality_one_hot = batch["to_modality_one_hot"].to(device)

            # Forward pass
            with _autocast(enabled=args_train.amp):
                missing_modality_pred = unet(x=from_modality,
                                            modality_tensor=to_modality_one_hot,
                                            )
                loss = loss_pt(missing_modality_pred.float(), to_modality.float()) / args_train.gradient_accumulation_steps  # Dividir para escalar la pérdida

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

                __to_mod_index = torch.argmax(batch["to_modality_one_hot"][0].cpu(), dim=0)


                # update writter
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("missing modality index", __to_mod_index, global_step)

                # update progress bar
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(),
                        "m_index": __to_mod_index
                        }
                progress_bar.set_postfix(**logs)

                # update global step
                global_step += 1

                # # Guardar modelo cada cierto intervalo
                if global_step % args_train.save_checkpoint_interval == 0:
                    save_model(unet, optimizer, lr_scheduler, global_step, _checkpoint_dir_name, ema=ema, )

                # Generar imágenes en intervalos
                if args_train.initial_val or global_step % args_train.val_interval == 0:
                    try:
                        if args_train.use_ema:
                            ema.apply_shadow()
                        validation(unet, autoencoder, global_step, args_train)
                    except Exception as e:
                        print(f"ERROR DURING VALIDATION STEP {global_step}: {e}")
                    finally:
                        if args_train.use_ema:
                            ema.restore()

                    # validation(unet, autoencoder, global_step, args_train)

                    args_train.initial_val = False

                if global_step >= args_train.max_train_steps:
                    break

        if global_step >= args_train.max_train_steps:
            break

    # make sure the progress bar closes
    progress_bar.close()

    # make  out_model_path dir if it does not exist
    save_model(unet, optimizer, lr_scheduler, global_step, _checkpoint_dir_name, ema=ema, )










args_train = {
    # directories
    "output_path": os.path.join(configs.PATH_TRAINING, "endec"),
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
    "dataloader_mode": "3-to-1",
    "attmasks_shapes_list": [(64, 64, 40)],

    # training configuration (540 epochs × 1489 subjects / batch_size=6 ≈ 134000)
    "max_train_steps": 134000,
    "save_checkpoint_interval": 5000,

    # ---- memory reduction
    "amp": True,

    # ---- Training stability
    "batch_size": 6, # possible 6 (no flash att), 8 (flash att)
    "gradient_accumulation_steps": 1,
    "use_ema": False,
    "ema_params": {
        "decay": 0.5,
        "warm_up_steps": 20000,
        "warm_up_decay": 0.1,
    },

    # ---- optimizer
    "lr": 1e-4, # for maisi  1e-4 # for blsmd 2.5e-5
    "weight_decay": 0.0, # set to 1e-5 ~ 1e-4 if overfitting is observed

    # ---- lr_scheduler
    "lr_scheduler": {"name": "PolynomialLR", "power": 2.0},
    # "lr_scheduler": None,
    "extra_steps": True,

    # ---- pretrained_model
    "load_pretrained_model_from": None, # not working


    # ---- resume from checkpoint
    "resume_from_checkpoint_path_name": None,

    # reproducibility
    "seed": 42,
    "val_seeds": [42],

    # validation
    "val_interval": 1000,
    "initial_val": True,

    # specialied synthesis
    "nb_val_images": 4, # number of images to show in the validation step

}


args_train = utils.dict_to_args(args_train, deep_conversion=True)
train(
    args_train,
    device,
    reserved_space=0
)
