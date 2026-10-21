# G1 Shared Data Root

This folder is the single shared data entry for both G1 lines.

## Layout

- `raw/`: mount or symlink the raw 2026 Task1 archive here once.
- `input/`: complete cases used by the T2W completion line.
- `input_inference/`: cases missing T2W used by the T2W completion line.
- `output/`: synthesized T2W outputs from the completion line.
- `latents/`: cached VAE latents from `preprocess.py`.
- `attention_masks/`: latent-space tumor masks from `generate_attmask.py`.
- `diffusion_cache/`: optional local cache for the diffusion augmentation line.

The code directory `work_space/G1/code/brats2025-latent-ensemble-synthesis-main/data`
is a symlink to this folder so both code lines read the same shared paths.
