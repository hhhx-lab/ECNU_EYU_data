import os
from pathlib import Path

def find_g1_workspace_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if parent.name == "G1" and (parent / "code").exists() and (parent / "docs").exists():
            return parent
    raise RuntimeError(f"Could not locate work_space/G1 from {start}")


# code root stays here for checkpoints and model weights
PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
PATH_WORKSPACE = str(find_g1_workspace_root(Path(__file__).resolve()))

PATH_MODELS = os.path.join(PATH_ROOT, "models")
PATH_WEIGHTS = os.path.join(PATH_ROOT, "weights")

PATH_NAME_WEIGHTS_VAE = os.path.join(PATH_WEIGHTS, "vae",  "autoencoder_epoch273.pt")

MODALITY_LIST = ["t1n", "t1c", "t2w", "t2f"]
MISSING_MODALITY = "t2w"
AVAILABLE_MODALITIES = ["t1n", "t1c", "t2f"]
SHAPE_PREPROCESS_IMG=(256,256,160)


PATH_DATA = os.path.join(PATH_WORKSPACE, "data")
PATH_RAW_DATA = os.path.join(PATH_DATA, "raw")
PATH_INPUT = os.path.join(PATH_DATA, "input")
PATH_INPUT_INFERENCE = os.path.join(PATH_DATA, "input_inference")
PATH_OUTPUT = os.path.join(PATH_DATA, "output")

PATH_TRAINING = os.path.join(PATH_ROOT, "training")

PATH_WEIGHTS_ENCDEC = os.path.join(PATH_TRAINING, "endec", "check_points")
PATH_WEIGHTS_BBDM = os.path.join(PATH_TRAINING, "bbdm", "check_points")


NETWORKS_CONFIG =  {
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
        "norm_float16": False,
        "num_splits": 8,
        "dim_split": 1
    },


    "encdec": {
        "unet": {
            # "_target_": "monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi",
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
            "use_flash_attention": True,
            "with_conditioning": False,
        }
    },

    "bbdm": {
        "unet": {
            # "_target_": "monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi",
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

            "include_from_modality": False,
            "include_to_modality": False,
        },

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
            "use_self_attention": False,  # whether to use self-attention
            "n_heads": 8,  # number of attention heads
            "n_layers": 1,  # number of layers in the MLP
        }
    }
}
