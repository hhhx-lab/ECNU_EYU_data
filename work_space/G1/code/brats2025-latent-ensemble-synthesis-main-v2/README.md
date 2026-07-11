# Latent-Space Ensemble Synthesis of Missing Brain Tumor MRI Modalities for BraTS Challenge

This repository contains the official code for the paper:

> **Latent-Space Ensemble Synthesis of Missing Brain Tumor MRI Modalities for BraTS Challenge**
> Cartaya Lathulerie A. et al., MICCAI 2025 BraTS Challenge, LNCS 16377, 2026

**Second place** in the BraTS 2025 Global Synthesis Challenge (Task 8).

---

## Method Overview

An ensemble of two generative models operating in a VAE-compressed latent space:

1. **MT-ED** (Modality Translation Encoder-Decoder) — deterministic mapping
2. **MT-BBDM** (Modality Translation Brownian Bridge Diffusion Model) — diffusion-based generation

Both models use the MAISI pretrained VAE to compress 3D MRI volumes from `(1, 256, 256, 160)` into a compact latent representation `(4, 64, 64, 40)`, enabling whole-volume training on a single GPU.

---

## 1. Environment Setup

```bash
# Python >= 3.10, CUDA 12.x recommended
pip install -r requirements.txt
```

Core dependencies: `torch>=2.0.0`, `monai>=1.4.0`, `nibabel`, `numpy<2.0.0`, `pandas`, `tensorboard`

**Verify VAE weights are loadable:**

```bash
python test_vae.py
# Expected output: "VAE loaded OK, state_dict keys: ..."
```

### 1.1 GPU Selection

Training scripts use the `CUDA_VISIBLE_DEVICES` environment variable to select which GPU to use. Default is GPU 0.

```bash
# Use GPU 0 (default)
python training_bbdm.py

# Use GPU 2
CUDA_VISIBLE_DEVICES=2 python training_bbdm.py

# Use multiple GPUs (if DataParallel/DDP support is added)
CUDA_VISIBLE_DEVICES=0,1 python training_bbdm.py
```

For inference, pass `--gpu_id` directly:

```bash
python main.py --synthesis_type bbdm --gpu_id 2 --verbose
```

### 1.1 Flash Attention (Important — 1.3x~2x speedup)

The U-Net uses 3D self-attention at its deepest two levels. Vanilla attention scales as O(N²) where N = 64×64×40 = 163,840 spatial positions — computationally very expensive. **Flash Attention** avoids this by tiling the attention matrix into smaller blocks computed directly in GPU SRAM, avoiding the memory-bound read/write of the full N×N matrix.

The code enables Flash Attention by default (`use_flash_attention: True`), but **whether it actually works depends on your environment**:

| Requirement | Minimum |
|-------------|---------|
| PyTorch | **≥ 2.0.0** |
| CUDA | **≥ 11.6** |
| GPU architecture | **Ampere or newer** (A100/A30/RTX 3090/RTX 4090, etc.) |

If any requirement is not met, PyTorch **silently falls back to vanilla attention** — you get no warning, but training runs 1.3x~2x slower and uses more VRAM.

**Verify at runtime** (also printed automatically when training starts):

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('Flash SDP:', torch.backends.cuda.flash_sdp_enabled())"
```

- `Flash SDP: True` → Flash Attention is active
- `Flash SDP: False` → silently degraded to vanilla attention

**If False**, the fix is usually upgrading PyTorch:

```bash
pip install 'torch>=2.0.0' --index-url https://download.pytorch.org/whl/cu121
```

Note: **V100 GPUs** (Volta architecture) do NOT support Flash Attention regardless of PyTorch version. If you are on V100 and training is too slow, consider reducing `attention_levels` to `[False, False, False, False]` in `configs.py` as a temporary measure (trades speed for some synthesis quality).

### 1.2 VAE Fine-tuning (Optional — improve small lesion preservation, default: off)

**Background.** The MAISI VAE compresses `256³ → 64³` (64× spatial reduction). A 5mm metastasis (~10 native voxels) occupies only 1-2 latent voxels after compression. The pretrained weights were trained on multi-body-region CT/MRI, not brain tumors, so the encoder may discard small lesions during compression.

**Approach.** Full-model domain fine-tuning (Plan C) with seg-guided loss. Both encoder and decoder co-adapt to the target domain with differential learning rates.

**Loss design:**

```
tumor_mask  = (seg > 0)           # BraTS-MET 2026: 1=NETC, 2=SNFH, 3=ET, 4=RC
brain_mask  = (mean(images) > 0.02)  # image-derived brain mask
healthy     = brain_mask - tumor_mask
lambda_t    = clamp(n_healthy / n_tumor, 3, 30)  # adaptive per-subject

loss = mse_whole + 0.1 * mse_healthy + lambda_t * mse_tumor + 0.01 * kl
```

The adaptive loss forces the encoder to preserve tumor-region information: each tumor voxel contributes 3-30× the gradient of a healthy voxel. KL regularization (weight 0.01) prevents latent-space drift.

**Recommended defaults:** all train subjects, synchronized `128×128×96` patches,
`80%` uniformly selected tumor-component centers, `20%` random brain centers,
four modalities per subject, `3` epochs, BF16, no activation checkpointing, and a
fixed seeded 20-subject quick-val subset after every epoch. Early stopping uses
quick-val tumor MSE with patience `2`; model selection is still closed by a
separate full-volume comparison on every val subject. Encoder LR is `2e-6` and
decoder LR is `1e-6`.

**Quick start:**

```bash
# 0. Generate metadata CSV, then create a deterministic patient-grouped split.
# Seg is required because finetune_vae.py uses it for tumor masking.
python preprocess.py --metadata-only
python filter_g1_v2_invalid_labels.py
python mark_g1_train_val_test_split.py \
    --data-csv data/data_csv.csv \
    --seed 42 \
    --val-fraction 0.10 \
    --test-fraction 0.10
python validate_g1_v2_dataset.py

# 1. Baseline evaluation (frozen VAE reconstruction quality)
python validate_vae_recon.py \
    --data_csv data/data_csv.csv \
    --data_dir data/input \
    --mode baseline \
    --device cuda:0

# 2. Fine-tune
python finetune_vae.py \
    --data_csv data/data_csv.csv \
    --data_dir data/input \
    --epochs 3 \
    --batch_size 2 \
    --patch-size 128 128 96 \
    --tumor-patch-probability 0.8 \
    --quick-val-subjects 20 \
    --early-stopping-patience 2 \
    --amp-dtype bfloat16 \
    --no-gradient-checkpointing \
    --skip_initial_validation \
    --device cuda:0

# 3. Compare reconstruction quality (frozen vs fine-tuned)
python validate_vae_recon.py \
    --data_csv data/data_csv.csv \
    --data_dir data/input \
    --finetuned_weights training/vae_finetuned/best_model.pt \
    --mode compare \
    --device cuda:0
```

The same crop is applied to `t1n/t1c/t2w/t2f`, segmentation, and all loss masks.
Tumor components use 26-connectivity and are selected uniformly by component,
so a large lesion does not dominate a small lesion. `quick_val_subjects.json`
records the fixed quick-val IDs. This quick subset controls checkpointing and
early stopping only.

**Verification.** The standalone baseline and final comparison must run on the
full val split. `delta_metrics.csv` reports the full-volume validation change
and `comparison_samples/` contains axial montages for the five worst cases.
`vae_selection.json` selects the fine-tuned VAE only when mean
`delta_tumor_SSIM >= 0.03` and mean `delta_whole_SSIM >= -0.005`.

**Downstream impact.** Fine-tuning changes the latent space, so you MUST re-encode all training latents and re-train both EncDec and BBDM (adds ~1 day on A100). If the improvement is marginal (< 0.03 SSIM), skip retraining and instead tune BBDM's tumor loss weights in `training_bbdm.py`.

**Script reference:**

| Script | Purpose |
|--------|---------|
| `finetune_vae.py` | VAE fine-tuning training loop |
| `validate_vae_recon.py` | Reconstruction quality comparison (frozen vs fine-tuned) |

Both scripts save outputs under `training/vae_finetuned/`. The original VAE at `weights/vae/autoencoder_epoch273.pt` is never modified. MONAI's VAE encoder returns `(mu, sigma)`; the training and comparison scripts use that contract directly for sampling and KL computation.

`--batch_size` is the number of subjects accumulated before one optimizer step; every subject contributes four modality reconstructions. Start with `2` on A100 and reduce to `1` if memory is insufficient. Record actual time and peak memory from the first full server run instead of relying on unverified estimates.

### 1.3 Seg-Guided Loss Weighting for BBDM/EncDec (Optional — boost small lesion supervision, default: off)

**Background.** In both EncDec and BBDM training, the standard MSE loss treats all voxels uniformly. This means a 100-voxel micro-metastasis contributes the same gradient as 100 voxels of healthy brain tissue, making it easy for the model to ignore small lesions. The seg-guided loss splits the objective into healthy and tumor components, with per-lesion V-weighting that amplifies the loss on smaller lesions.

**V-weight design (per-patient, per-lesion):**

```
V_i = clamp(max_lesion_vol_in_patient / vol_i, 1, 5)
```

For a patient with lesions [1000, 200, 50, 10] voxels:
- max = 1000, V = [1, 5, 20→5, 100→5] = [1, 5, 5, 5]
- The smallest lesions get 5× the loss weight of the largest.

**Quick start (manual flag enable required):**

```bash
# 1. Pre-compute per-lesion V-weight masks
python precompute_lesion_weights.py \
    --data_csv data/data_csv.csv \
    --data_dir data/input \
    --output_dir data/lesion_weights

# 2. Edit training_bbdm.py: set use_lesion_weights=True in args_train
# 3. Edit training_endec.py: set use_seg_loss=True in args_train
# 4. Train as usual
python training_bbdm.py
python training_endec.py
```

These flags default to `False`. Without editing the training scripts, training runs with the original uniform-loss behavior regardless of whether `data/lesion_weights/` exists.

**Effect on EncDec (new):** The EncDec training currently uses uniform MSE. With `use_seg_loss=True`, it splits the loss into healthy + tumor components with adaptive `lambda_tumor = clamp(n_healthy / n_tumor, 3, 30)`, matching BBDM's tumor-aware loss structure. When lesion weight masks are also available, tumor voxels receive per-lesion V-weights.

**Effect on BBDM (modified):** BBDM already uses healthy/tumor loss separation via attention masks. With `use_lesion_weights=True`, the tumor loss term is further weighted by per-lesion V-weights, so smaller lesions within a patient get amplified supervision.

**Script reference:**

| Script | Purpose |
|--------|---------|
| `precompute_lesion_weights.py` | Compute per-lesion V-weight masks from seg files |

**Backward compatibility:** Both `use_seg_loss` and `use_lesion_weights` default to `False`. Without pre-computed weight masks, training behaves identically to the original.

---

## 2. Quick Start: Train → Evaluate → Generate

Typical workflow for synthesizing missing T2W given two sets of subjects — one complete (all 4 modalities) for training, one incomplete (missing T2W) to fill in.

### Step 1 — Train on complete subjects

```bash
# Place complete subjects in data/input/
cp -r /path/to/complete_set/* data/input/

# Build metadata and a leakage-safe train/val/test split
python preprocess.py --metadata-only
python filter_g1_v2_invalid_labels.py
python mark_g1_train_val_test_split.py

# Optional VAE fine-tuning must happen here. After selecting a VAE,
# encode every case once with that same weight file.
G1_VAE_WEIGHTS=/absolute/path/to/selected_vae.pt \
python preprocess.py --respect-existing-csv --clean-latents --device cuda:0

# Optional: generate attention masks (requires seg files, enables tumor-aware BBDM loss)
python generate_attmask.py

# Optional: compute channel importance weights for BBDM (run once, then copy output to training_bbdm.py)
python compute_weights.py

# Train both models
python training_endec.py       # → training/endec/check_points/
python training_bbdm.py        # → training/bbdm/check_points/
```

### Step 2 — Evaluate on val subjects (sanity check)

```bash
# Default: evaluates split=val from data_csv.csv, reads from data/input/
python evaluate.py --gpu_id 0 --verbose
# Terminal: SSIM / PSNR / MSE / MAE per subject + summary
# Saved:    data/eval_synthesized/<subject_id>-t2w.nii.gz (with --save_output)
```

If metrics look good, proceed. Otherwise tune hyperparameters and retrain.

### Step 3 — Generate missing T2W for incomplete subjects

```bash
# Place incomplete subjects (no T2W required) in data/input_inference/
cp -r /path/to/incomplete_set/* data/input_inference/

# Run inference (default: ensemble, CPU — always pass --gpu_id for GPU)
python main.py --synthesis_type ensamble --gpu_id 0 --verbose

# Results
ls data/output/   # <subject_id>-t2w.nii.gz
```

### Directory overview

| Directory | Who uses it | Contents |
|-----------|-------------|----------|
| `data/input/` | Training (`training_*.py`), Evaluation (`evaluate.py`) | All 4 modalities (t1n, t1c, t2w, t2f) |
| `data/input_inference/` | `main.py` | Exactly the 3 available modalities; T2W must be absent |
| `data/output/` | `main.py` output | Synthesized `.nii.gz` files |
| `data/eval_synthesized/` | `evaluate.py --save_output` | Synthesized images for inspection |

---

## 3. Data Preparation

### 3.1 Input Format

Place raw NIfTI files under `data/input/<subject_id>/` for training, and `data/input_inference/<subject_id>/` for inference:

```
data/input/
├── BraTS-MET-00000-000/
│   ├── BraTS-MET-00000-000-t1n.nii.gz    # T1-weighted
│   ├── BraTS-MET-00000-000-t1c.nii.gz    # T1 contrast-enhanced
│   ├── BraTS-MET-00000-000-t2w.nii.gz    # T2-weighted (target to synthesize)
│   ├── BraTS-MET-00000-000-t2f.nii.gz    # T2 FLAIR
│   └── BraTS-MET-00000-000-seg.nii.gz    # tumor segmentation (optional)
├── BraTS-MET-00000-001/
│   └── ...
└── ...
```

**Requirements:**
- Files must be `.nii.gz` or `.nii`
- Training subjects must have all 4 modalities (t1n, t1c, t2w, t2f)
- Filenames must end with `-t1n`, `-t1c`, `-t2w`, `-t2f`, or `-seg` before `.nii`/`.nii.gz`
- Already skull-stripped, registered, and resampled to 1mm³ is recommended

### 3.2 Preprocessing (VAE Encoding)

```bash
# Metadata only, before VAE fine-tuning and split creation
python preprocess.py --metadata-only

# Latent encoding, after the VAE selection gate
G1_VAE_WEIGHTS=/absolute/path/to/selected_vae.pt \
python preprocess.py --respect-existing-csv --clean-latents --device cuda:0
```

This will:
1. Scan all subjects under `data/input/`
2. Normalize each volume to [0, 1]
3. Zero-pad / center-crop to `(256, 256, 160)`
4. Encode with the pretrained MAISI VAE → latent arrays of shape `(4, 64, 64, 40)`
5. Save `.npy` files to `data/latents/<subject_id>/`
6. Generate `data/data_csv.csv`

### 3.3 Train/Validation Split

Do not edit `data/data_csv.csv` manually. Create a deterministic, patient-grouped split:

```bash
python mark_g1_train_val_test_split.py \
    --seed 42 \
    --val-fraction 0.10 \
    --test-fraction 0.10
python validate_g1_v2_dataset.py
```

The default targets are approximately 80% train, 10% val, and 10% locked test. All records sharing the same `BraTS-MET-xxxxx` patient group remain in one split. VAE/model training reads only train, tuning reads val, and locked test is reserved for final internal verification.

### 3.4 (Optional) Attention Masks

For tumor-aware training in MT-BBDM, the model uses binary attention masks in latent space to separate tumor vs. healthy tissue in the loss function.

If your dataset includes tumor segmentation files (`<subject_id>-seg.nii.gz`) alongside the MRI volumes, generate the masks with:

```bash
python generate_attmask.py
```

This will:
1. Load each subject's seg file
2. Binarize (threshold > 0.5)
3. Resize to `(256, 256, 160)` then downsample to latent space `(64, 64, 40)` using nearest-neighbor
4. Save as `data/attention_masks/<subject_id>/<subject_id>_attmask_64_64_40.npy`

The output masks are binary: `1` = tumor, `0` = healthy tissue.

If no seg files are available, skip this step. The model will treat the entire volume as healthy tissue (training still works but loses tumor-specific supervision).

### 3.5 Compute Channel Importance Weights

The BBDM loss weights each of the 4 latent channels differently, based on their RMS energy in the training set. **These weights depend on your dataset and should be recomputed** whenever you switch to a different tumor type or data source.

```bash
# Run AFTER preprocess.py has generated all latents
python compute_weights.py
```

Example output:
```
Loaded t2w latents from 500 training subjects.

  ch0  RMS = 0.234567
  ch1  RMS = 0.056789
  ch2  RMS = 0.092345
  ch3  RMS = 0.074321

channel_importance_weights = [0.512345, 0.124111, 0.201678, 0.161866]
sum = 1.0000
```

The script writes `data/channel_weights.json`. `training_bbdm.py` loads `channel_importance_weights` from that file automatically, so no manual source edit is required.

---

## 4. Configuration

All key settings are in `configs.py` and the training scripts themselves.

**`configs.py` — path-level settings:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MISSING_MODALITY` | `"t2w"` | Which modality to synthesize (t1n / t1c / t2w / t2f) |
| `AVAILABLE_MODALITIES` | `["t1n", "t1c", "t2f"]` | Must be the other 3 modalities |
| `SHAPE_PREPROCESS_IMG` | `(256, 256, 160)` | Preprocessed volume size |

**Training scripts** (`training_bbdm.py` and `training_endec.py`) — edit the `args_train` dictionary at the bottom of each file:

| Parameter | Default BBDM | Default EncDec | Description |
|-----------|-------------|---------------|-------------|
| `max_train_steps` | 402000 | 134000 | Total training steps |
| `batch_size` | 4 | 6 | Reduce if OOM |
| `lr` | 1e-4 | 1e-4 | Learning rate |
| `amp` | True | True | Automatic mixed precision |
| `dataloader_mode` | `"4b-to-4"` | `"3-to-1"` | Data loading scheme |
| `nb_val_images` | 4 | 4 | Number of val images to visualize |

**Adjusting `max_train_steps` for your dataset size:**

The paper used 1,489 training subjects. Adapt to your dataset:

```python
# Desired epochs (paper: 1080 for BBDM, 540 for EncDec)
epochs_desired = 1080
max_train_steps = epochs_desired * (num_train_subjects / batch_size)
```

For example, with 500 subjects and batch_size=4:
- BBDM: `1080 * (500/4) ≈ 135,000` steps
- EncDec: `540 * (500/6) ≈ 45,000` steps

---

## 5. Training

Both models can be trained independently (recommended: train both, then use ensemble inference).

### 5.1 Train MT-EncDec

```bash
python training_endec.py
```

- GPU memory: fits in 24GB with batch_size=6. For 16GB cards, reduce to 4.
- Output: checkpoints saved to `training/endec/check_points/`, logs to `training/endec/logs/`

### 5.2 Train MT-BBDM

```bash
python training_bbdm.py
```

- GPU memory: fits in 24GB with batch_size=4. For 16GB cards, reduce to 2.
- Output: checkpoints saved to `training/bbdm/check_points/`, logs to `training/bbdm/logs/`

### 5.3 Monitoring Training

```bash
tensorboard --logdir training/bbdm/logs
# or
tensorboard --logdir training/endec/logs
```

Key metrics to watch: `Loss/train`, `Learning_rate`, `to modality index`.

Validation images are saved to `training/bbdm/val_imgs/` (or `endec/val_imgs/`) at each `val_interval`.

### 5.4 Resume from Checkpoint

To resume interrupted training, set in `args_train`:

```python
"resume_from_checkpoint_path_name": "training/bbdm/check_points/model_50000.pt",
```

---

## 6. Deploy Trained Weights for Inference

Training checkpoints are saved directly to the paths that inference reads from — no manual copy needed.

| Model | Checkpoint path |
|-------|----------------|
| EncDec | `training/endec/check_points/model_*.pt` |
| BBDM | `training/bbdm/check_points/model_*.pt` |

Inference automatically loads the latest `.pt` file from these directories.

---

## 7. Inference

Place input data under `data/input_inference/<subject_id>/` (same format, T2W not required), then run:

### Single-model inference

```bash
# EncDec only
python main.py --synthesis_type encdec --gpu_id 0 --verbose

# BBDM only
python main.py --synthesis_type bbdm --gpu_id 0 --verbose
```

### Ensemble inference (recommended)

```bash
python main.py --synthesis_type ensamble --gpu_id 0 --verbose --compute_bmask
```

### Per-lesion ROI inference (parallel pipeline for small lesion enhancement)

```bash
# Requires seg files in subject folders
python main.py --per_lesion --gpu_id 0 --verbose
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--synthesis_type` | `ensamble` | `encdec` / `bbdm` / `ensamble` |
| `--gpu_id` | `None` (CPU) | GPU device ID, e.g. `0` |
| `--verbose` | False | Print progress |
| `--compute_bmask` | False | Segment and apply brain mask (requires TotalSegmentator) |
| `--input_dir` | `data/input_inference` | Directory containing subjects to synthesize |
| `--per_lesion` | False | Enable per-lesion ROI synthesis overlay (requires seg files) |

### Output

Synthesized NIfTI volumes are saved to `data/output/` with the naming convention `<subject_id>-<missing_modality>.nii.gz`.

When using `--verbose`, intermediate results (raw encdec/bbdm outputs before postprocessing) are also saved under `data/output/intermediate_<subject_id>/`.

### 7.1 Quantitative Evaluation

If you have subjects with **all 4 modalities** (including ground truth T2W) and they are listed in `data_csv.csv`, you can compute SSIM, PSNR, MSE, MAE against the real T2W:

```bash
# Evaluate val split (default)
python evaluate.py --gpu_id 0 --verbose

# Evaluate a different split
python evaluate.py --split train --gpu_id 0 --verbose

# Single model
python evaluate.py --synthesis_type bbdm --gpu_id 0 --verbose
python evaluate.py --synthesis_type encdec --gpu_id 0 --verbose

# Save per-subject results to CSV
python evaluate.py --gpu_id 0 --save_csv results.csv
```

The script reads subjects from `data_csv.csv` filtered by `--split` (default: `val`), loads images from `data/input/`, and reports metrics for the **whole volume** and **brain region** (masked by mean intensity of input modalities).

### 7.2 Per-Lesion ROI Synthesis (Optional, `--per_lesion` flag, default: off)

**Background.** Small metastases (< 5mm) occupy only 1-3 latent voxels after VAE compression. The full-image synthesis models (EncDec + BBDM) may fail to reconstruct them accurately because the lesion signal is dominated by surrounding healthy tissue in the latent space. The per-lesion ROI pipeline addresses this by cropping each lesion region, running synthesis on the local ROI (where the lesion occupies a larger fraction of the latent space), and feather-blending the result back into the full-image ensemble output.

**Key design decisions:**

- **Pad-to-cube, no zoom:** ROIs are zero-padded to a uniform cube size (min 64³). No resolution change — the models see the original 1mm³ resolution. Zero padding matches the natural background in skull-stripped images.
- **Feather blending:** A 16-voxel 3D ramp mask ensures smooth transitions between the ROI patch and the background canvas. The blending zone is well outside the tumor region (32-voxel context margin).
- **Overlapping ROI merge:** When lesions are close together, their ROIs are merged into a single larger ROI before processing, avoiding redundant computation and double-counting at boundaries.
- **Parallel pipeline:** This is an inference-only feature. The standard full-image synthesis runs first as the base canvas, then per-lesion ROIs are overlaid as enhancement. Training is unchanged.

**Quick start:**

```bash
# Inference with per-lesion ROI overlay
python main.py --per_lesion --gpu_id 0 --verbose

# Evaluation with per-lesion ROI overlay
python evaluate.py --per_lesion --gpu_id 0 --verbose --save_csv results_roi.csv
```

**Requirements:** Subject folders must contain a seg file (e.g., `*-seg.nii.gz`) for lesion detection. Subjects without seg files skip the ROI overlay and use the standard ensemble result.

**Performance note:** Each ROI requires one EncDec forward pass + one BBDM 50-step diffusion process. With ~10 lesions per subject (typical for metastasis patients), per-lesion inference adds ~3-5× runtime compared to standard ensemble inference. This is a quality-speed tradeoff intended for offline evaluation, not real-time deployment.

**Script reference:**

| Script | Purpose |
|--------|---------|
| `synthesis/roi_synthesis.py` | Per-lesion ROI detection, merging, synthesis, and blending |

---

## 8. Key Hyperparameters for Tuning

If you need to optimize synthesis quality, the most impactful parameters are:

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `bb_scheduler.s` | `configs.py` NETWORKS_CONFIG | `0.01` | Brownian bridge variance. Higher = more diversity, lower = more deterministic |
| `channel_importance_weights` | `training_bbdm.py` | `[0.51, 0.12, 0.20, 0.16]` | Per-channel latent loss weights (computed from RMS) |
| `extra_modalites_weight` | `training_bbdm.py` | `0.0` | Loss weight for non-target modalities (0 = only supervise T2W) |
| `bb_scheduler.sample_step` | `configs.py` | `50` | Inference diffusion steps (fewer = faster, more = potentially better) |
| `lr` | training scripts | `1e-4` | Learning rate |
| `weight_decay` | training scripts | `0.0` (disabled) | L2 regularization for Adam. Set to `1e-5`~`1e-4` only if overfitting is observed |
| `batch_size` | training scripts | 4~6 | Adjust to GPU memory |

### 8.1 Tuning the BBDM Variance `s`

`s` controls the noise variance in the Brownian bridge: `variance_t = 2 * (m_t - m_t²) * s`. It affects both training dynamics and inference diversity.

| s | Behavior | Risk |
|---|----------|------|
| 0.001~0.005 | Near-deterministic, output closely mirrors input | Low diversity |
| **0.01** (default) | Conservative, stable training | May be slightly blurry |
| 0.03~0.05 | Good diversity-fidelity balance | Monitor loss |
| 0.1+ | High randomness | Loss may oscillate, artifacts may appear |

**Procedure** (run AFTER full dataset is ready, not with a handful of samples):

1. Set a quick-test step count in `training_bbdm.py`:
   ```python
   "max_train_steps": 5000,  # temporary, for fast experiments
   ```
2. In `configs.py`, change `s` in `NETWORKS_CONFIG["bbdm"]["bb_scheduler"]["s"]`.
3. Train for 5000 steps, then check results:
   ```bash
   python training_bbdm.py
   python check_loss.py                   # look at jitter + trend
   # Open training/bbdm/val_imgs/imgs_step_5000.png for visual check
   ```
4. Compare across values (e.g. `0.005`, `0.01`, `0.05`, `0.1`) — this is the **coarse search**.
5. Pick the best coarse value, then run a **fine search** around it with smaller steps. For example, if `0.05` was best: test `0.03`, `0.04`, `0.05`, `0.06`, `0.07`.
6. Pick the winner from the fine search, restore `max_train_steps: 402000`, and run full training.

**How to judge:**

| Signal | Diagnosis |
|--------|-----------|
| `jitter` low, `trend ↓` | s is good ✓ |
| `jitter` high (10x baseline), trend flat | s too large ✗ |
| Val images nearly identical across conditions | s too small ✗ (no diversity) |
| Val images have checkerboard or salt-pepper noise | s too large ✗ |
| Loss explodes (NaN) | s way too large ✗ |

---

## 9. Directory Structure After Setup

```
project/
├── configs.py                  # Global paths and network configs
├── main.py                     # Inference entry point
├── preprocess.py               # Data preprocessing (VAE encoding)
├── training_bbdm.py            # MT-BBDM training
├── training_endec.py           # MT-EncDec training
├── finetune_vae.py             # VAE fine-tuning (optional)
├── validate_vae_recon.py       # VAE reconstruction comparison (optional)
├── precompute_lesion_weights.py # Per-lesion V-weight pre-computation (optional)
├── test_vae.py                 # Verify VAE weights
├── requirements.txt
├── weights/
│   └── vae/
│       └── autoencoder_epoch273.pt    # Pretrained MAISI VAE (provided)
├── data/
│   ├── input/<subject_id>/            # ← Training data (all 4 modalities)
│   ├── input_inference/<subject_id>/  # ← Inference data (3 modalities, T2W optional)
│   ├── latents/<subject_id>/          # Generated by preprocess.py
│   ├── attention_masks/<subject_id>/  # Optional: tumor masks
│   ├── lesion_weights/<subject_id>/   # Optional: per-lesion V-weight masks
│   ├── data_csv.csv                   # Generated by preprocess.py
│   ├── output/                        # Inference results (main.py)
│   └── eval_synthesized/              # Evaluation results (evaluate.py --save_output)
├── training/
│   ├── endec/check_points/            # EncDec checkpoints (training output + inference input)
│   ├── bbdm/check_points/             # BBDM checkpoints (training output + inference input)
│   └── vae_finetuned/                 # VAE fine-tuning outputs (optional)
├── models/                            # Model architecture definitions
└── synthesis/                         # Inference pipeline + ROI synthesis
    └── roi_synthesis.py               # Per-lesion ROI synthesis module
```

---

## 10. Citation

```bibtex
@inproceedings{cartaya2026latent,
  author = {Cartaya Lathulerie, A. and others},
  title = {Latent-Space Ensemble Synthesis of Missing Brain Tumor MRI Modalities for BraTS Challenge},
  booktitle = {Segmentation, Classification, and Synthesis for Brain Tumors and Traumatic Brain Injuries},
  series = {LNCS},
  volume = {16377},
  publisher = {Springer},
  year = {2026},
  doi = {10.1007/978-3-032-16370-7_3}
}
```
