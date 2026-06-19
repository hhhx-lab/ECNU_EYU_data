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

---

## 2. Quick Start: Train → Evaluate → Generate

Typical workflow for synthesizing missing T2W given two sets of subjects — one complete (all 4 modalities) for training, one incomplete (missing T2W) to fill in.

If you are starting from the raw BraTS Task1 package, first read `docs/G1_G2_服务器训练推理QC运行手册.md`. It shows how the three raw folders are turned into the two working folders used here: `data/input/` and `data/input_inference/`.

### Step 1 — Train on complete subjects

```bash
# Place complete subjects in data/input/
cp -r /path/to/complete_set/* data/input/

# Preprocess, then apply the G2 fixed train/val/test split automatically
python preprocess.py
python mark_val_split_from_g2.py

# Optional: generate attention masks (requires seg files)
python generate_attmask.py

# Compute channel weights → copy output into training_bbdm.py args_train
python compute_weights.py

# Train both models
python training_endec.py       # → training/endec/check_points/
python training_bbdm.py        # → training/bbdm/check_points/
```

### Step 2 — Evaluate on complete subjects (sanity check)

```bash
python evaluate.py --input_dir data/input --gpu_id 0 --verbose --save_output
# Terminal: SSIM / PSNR / MSE / MAE per subject + summary
# Saved:    data/eval_synthesized/<subject_id>-t2w.nii.gz
```

If metrics look good, proceed. Otherwise tune hyperparameters and retrain.

### Step 3 — Generate missing T2W for incomplete subjects

```bash
# Place incomplete subjects in data/input_inference/; T2W is omitted automatically by prepare_g1_t2w_data.py
cp -r /path/to/incomplete_set/* data/input_inference/

# Run inference
python main.py --synthesis_type ensamble --gpu_id 0 --verbose --output_dir data/output

# Results
ls data/output/<subject_id>/   # <subject_id>-t2w.nii.gz plus mirrored source files
```

### Directory overview

| Directory | Who uses it | Contents |
|-----------|-------------|----------|
| `data/input/` | Training | All 4 modalities (t1n, t1c, t2w, t2f) |
| `data/input_inference/` | `main.py` | t1n, t1c, t2f, seg; T2W is intentionally absent or ignored |
| `data/output/` | `main.py` output | Per-subject output folders containing mirrored source files and synthesized `.nii.gz` |
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
- Training subjects in `data/input/` must have all 4 MRI modalities: t1n, t1c, t2w, t2f
- Inference subjects in `data/input_inference/` must have t1n, t1c, t2f, and seg; T2W is not required
- Already skull-stripped, registered, and resampled to 1mm³ is recommended

### 3.2 Project Data Placement From G2 Manifests

For ECNU-NYU 2026 Task1, prefer the automatic placement script instead of manually copying or deleting files:

```bash
python prepare_g1_t2w_data.py --mode symlink --clean --overwrite
```

This reads the G2 manifests under `work_space/G2/results/` and creates:

```text
data/input/            # complete real T2W cases for training
data/input_inference/  # fake/missing T2W cases for generation; T2W is omitted
data/g1_data_placement_manifest.csv
```

Use `--mode symlink` on the server when raw data and project code are on the same filesystem. Use `--mode copy` only when disk space is sufficient. Use `--mode manifest-only` to inspect the planned placement without creating links or copying data.

### 3.3 Preprocessing (VAE Encoding)

```bash
python preprocess.py
```

This will:
1. Scan all subjects under `data/input/`
2. Normalize each volume to [0, 1]
3. Zero-pad / center-crop to `(256, 256, 160)`
4. Encode with the pretrained MAISI VAE → latent arrays of shape `(4, 64, 64, 40)`
5. Save `.npy` files to `data/latents/<subject_id>/`
6. Generate `data/data_csv.csv`

### 3.4 Train/Validation/Test Split

`preprocess.py` writes all complete four-modality subjects as `split=train` first. For this project, do **not** hand-edit the CSV unless G2 explicitly changes the split. Apply the fixed G2 split with:

```bash
python mark_val_split_from_g2.py
```

This rewrites `data/data_csv.csv` in place and creates `data/data_csv.csv.bak_before_g2_split` before changing it. The script uses:

```text
work_space/G2/results/splits/splits_final_train_val_test.json
work_space/G2/results/manifests/nnunet_case_mapping_realonly.csv
```

Only subjects that survived `preprocess.py` and have complete `t1n/t1c/t2w/t2f` latents can become train, val, or test rows. Cases missing T2W are never written to `data/data_csv.csv`.

Current expected G1 projection from the G2 split:

```text
train = 660
val   = 160
test  = 210
```

The G2 full-case split is `829/207/259`; G1 sees fewer cases because fake/broken T2W cases are routed to `data/input_inference/` instead of training CSV.

### 3.5 (Optional) Attention Masks

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

### 3.6 Compute Channel Importance Weights

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

Copy the printed `channel_importance_weights` array into `training_bbdm.py`, replacing the default value in the `args_train` dictionary. If you skip this step, the default weights (computed on the original BraSyn glioma+metastasis mix) will be used, which is still reasonable but suboptimal for a different tumor population.

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

Place input data under `data/input_inference/<subject_id>/` with t1n/t1c/t2f/seg files. T2W is not required for inference and will be ignored if present. Then run:

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

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--synthesis_type` | `ensamble` | `encdec` / `bbdm` / `ensamble` |
| `--gpu_id` | `None` (CPU) | GPU device ID, e.g. `0` |
| `--verbose` | False | Print progress |
| `--compute_bmask` | False | Segment and apply brain mask (requires TotalSegmentator) |
| `--input_dir` | `data/input_inference` | Directory containing subjects to synthesize |

### Output

Each subject is written to `data/output/<subject_id>/`. The synthesized T2W is saved as `<subject_id>-t2w.nii.gz` alongside mirrored source files for downstream QC.

When using `--verbose`, intermediate results (raw encdec/bbdm outputs before postprocessing) are saved under `data/output/<subject_id>/intermediate_<subject_id>/`.

### 7.1 Quantitative Evaluation

If you have subjects with **all 4 modalities** (including ground truth T2W), you can compute SSIM, PSNR, MSE, MAE against the real T2W:

```bash
# Ensemble (recommended)
python evaluate.py --gpu_id 0 --verbose

# Single model
python evaluate.py --synthesis_type bbdm --gpu_id 0 --verbose
python evaluate.py --synthesis_type encdec --gpu_id 0 --verbose

# Save per-subject results to CSV
python evaluate.py --gpu_id 0 --save_csv results.csv
```

The script reports metrics for the **whole volume** and **brain region** (masked by mean intensity of input modalities). Place eval subjects in `data/input_inference/` — the script automatically finds those with all 4 modalities.

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
├── test_vae.py                 # Verify VAE weights
├── requirements.txt
├── weights/
│   └── vae/
│       └── autoencoder_epoch273.pt    # Pretrained MAISI VAE (provided)
├── data/
│   ├── input/<subject_id>/            # ← Training data (all 4 modalities)
│   ├── input_inference/<subject_id>/  # ← Inference data (t1n/t1c/t2f/seg; no T2W needed)
│   ├── latents/<subject_id>/          # Generated by preprocess.py
│   ├── attention_masks/<subject_id>/  # Optional: tumor masks
│   ├── data_csv.csv                   # Generated by preprocess.py
│   ├── output/                        # Inference results (main.py)
│   └── eval_synthesized/              # Evaluation results (evaluate.py --save_output)
├── training/
│   ├── endec/check_points/            # EncDec checkpoints (training output + inference input)
│   └── bbdm/check_points/             # BBDM checkpoints (training output + inference input)
├── models/                            # Model architecture definitions
└── synthesis/                         # Inference pipeline
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
