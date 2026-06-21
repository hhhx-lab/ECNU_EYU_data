"""Quantitative evaluation of synthesized T2W against ground truth.

Computes SSIM, PSNR, MSE, MAE per subject and aggregate statistics.
Requires subjects with all 4 modalities (including ground truth T2W).

Usage:
    python evaluate.py                          # ensemble (recommended)
    python evaluate.py --synthesis_type bbdm    # BBDM only
    python evaluate.py --synthesis_type encdec   # EncDec only
    python evaluate.py --gpu_id 0 --verbose
    python evaluate.py --save_csv results.csv   # save per-subject CSV
"""

import os
import argparse
import re
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

import configs
import synthesis.pipeline as pipeline
import synthesis.utils as utils


MODALITY_PATTERN = re.compile(
    r"^(?P<prefix>.+?)-(?P<mod>t1n|t1c|t2w|t2f|seg)(?P<suffix>\.nii(?:\.gz)?)$",
    re.IGNORECASE,
)


def parse_modality_file(file_name):
    match = MODALITY_PATTERN.match(file_name)
    if not match:
        return None
    return match.group("mod").lower()


def compute_metrics(pred, target, mask=None, data_range=1.0):
    """Compute SSIM, PSNR, MSE, MAE between two 3D images.

    Args:
        pred, target: 3D numpy arrays in (256, 256, 160), normalized to [0, 1]
        mask: optional 3D binary mask to restrict computation
        data_range: dynamic range (1.0 for [0,1] images)
    """
    if mask is not None and mask.sum() > 0:
        pred_m = pred[mask > 0]
        target_m = target[mask > 0]
    else:
        pred_m = pred.ravel()
        target_m = target.ravel()

    mse = float(np.mean((pred_m - target_m) ** 2))
    mae = float(np.mean(np.abs(pred_m - target_m)))
    max_val = max(float(pred_m.max()), float(target_m.max()), data_range)
    psnr = float(10 * np.log10(max_val ** 2 / mse)) if mse > 0 else float('inf')

    # 3D SSIM via MONAI
    from monai.losses import SSIMLoss
    t1 = torch.tensor(pred).unsqueeze(0).unsqueeze(0).float()
    t2 = torch.tensor(target).unsqueeze(0).unsqueeze(0).float()
    ssim_loss_fn = SSIMLoss(spatial_dims=3, data_range=data_range)
    ssim_val = float(1.0 - ssim_loss_fn(t1, t2).item())

    return {"SSIM": ssim_val, "PSNR": psnr, "MSE": mse, "MAE": mae}


def create_brain_mask(images, threshold=0.02):
    """Create brain mask from mean of available modality images."""
    mean_img = np.mean(images, axis=0)
    return (mean_img > threshold).astype(np.float32)


def load_and_preprocess_first(path):
    """Load first NIfTI, returning image + affine info."""
    img, aff = utils.load_nifti(path)
    org_shape = img.shape
    img, aff = utils.preprocessing(img, affine=aff)
    return img, aff, org_shape


def load_and_preprocess(path, aff):
    """Load NIfTI and preprocess using shared affine."""
    img, _ = utils.load_nifti(path)
    img, _ = utils.preprocessing(img, affine=aff)
    return img


def find_eval_subjects(input_dir):
    """Find subjects that have all 4 modalities (including ground truth T2W)."""
    all_dirs = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])

    subjects = []
    for d in all_dirs:
        files = os.listdir(os.path.join(input_dir, d))
        suffixes = {parse_modality_file(f) for f in files}
        suffixes.discard(None)
        if all(m in suffixes for m in configs.MODALITY_LIST):
            mod_to_file = {}
            for f in files:
                suffix = parse_modality_file(f)
                if suffix in configs.MODALITY_LIST:
                    mod_to_file[suffix] = f
            subjects.append({
                "id": d,
                "path": os.path.join(input_dir, d),
                "files": mod_to_file,
            })
    return subjects


def run_encdec_forward(unet, latens_list, device):
    """Synthesize using pre-loaded EncDec model."""
    to_modality_one_hot = torch.tensor(
        utils.create_modality_one_hot(configs.MISSING_MODALITY)
    ).float().to(device)

    from_modality_latents = np.concatenate(latens_list, axis=0)
    from_modality_latents = torch.tensor(from_modality_latents).half().to(device).unsqueeze(0)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        syn_latens = unet(x=from_modality_latents, modality_tensor=to_modality_one_hot)
    return syn_latens.detach().cpu().squeeze(0).numpy()


def run_bbdm_forward(unet, conditions_model, noise_scheduler, latens_list, device):
    """Synthesize using pre-loaded BBDM model."""
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
        for i in range(noise_scheduler.sample_step):
            t = noise_scheduler.steps[i]
            t_tensor = torch.zeros((from_modality_latents.shape[0],), device=device)
            t_tensor.fill_(t)
            noise_pred = unet(x=latents_denoised, timesteps=t_tensor, context=conditioning_emb)
            latents_denoised, _ = noise_scheduler.backward_diffusion(
                i, latents_denoised, from_modality_latents, noise_pred
            )

    latents_denoised = latents_denoised.detach().cpu().squeeze(0).numpy()
    syn_latens = latents_denoised[
        n_latent_channels * to_modality_index : n_latent_channels * (to_modality_index + 1)
    ]
    return syn_latens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthesis_type", choices=["encdec", "bbdm", "ensamble"], default="ensamble")
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save_csv", type=str, default=None, help="Save per-subject results to CSV")
    parser.add_argument(
        "--save_output",
        action="store_true",
        help="Save synthesized NIfTI files to data/eval_synthesized/"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=configs.PATH_INPUT,
        help="Directory containing complete subject folders for evaluation"
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cpu")

    subjects = find_eval_subjects(args.input_dir)
    if not subjects:
        print(f"No subjects with all 4 modalities found in {args.input_dir}")
        print("Place subjects with t1n, t1c, t2w, t2f files to evaluate.")
        return

    print(f"Found {len(subjects)} subjects with all 4 modalities.")

    # --- Instantiate models once ---
    print("Loading VAE ...")
    vae = pipeline.instantiate_vae_model(device)

    unet_encdec = None
    unet_bbdm = None
    conditions_model = None
    noise_scheduler = None

    if args.synthesis_type in ("encdec", "ensamble"):
        print("Loading EncDec model ...")
        unet_encdec = pipeline.instantiate_encdec_model(device)

    if args.synthesis_type in ("bbdm", "ensamble"):
        print("Loading BBDM model ...")
        unet_bbdm, conditions_model, noise_scheduler = pipeline.instantiate_bbdm_model(device)

    # --- Evaluate each subject ---
    results = []

    for subj in tqdm(subjects, desc="Evaluating"):
        s_id = subj["id"]
        s_path = subj["path"]
        f = subj["files"]

        try:
            # Load + preprocess: first modality sets affine reference
            first_mod = configs.AVAILABLE_MODALITIES[0]
            img_ref, aff, org_shape = load_and_preprocess_first(
                os.path.join(s_path, f[first_mod])
            )

            # Preprocess available modalities
            imgs_pp_list = [img_ref]
            for mod in configs.AVAILABLE_MODALITIES[1:]:
                imgs_pp_list.append(
                    load_and_preprocess(os.path.join(s_path, f[mod]), aff)
                )

            # Preprocess ground truth T2W
            gt = load_and_preprocess(
                os.path.join(s_path, f[configs.MISSING_MODALITY]), aff
            )

            # Brain mask from available modalities
            brain_mask = create_brain_mask(imgs_pp_list)

            # Encode to latents
            utils.set_seed(42)
            latens_list = [pipeline.encode_image(img, vae) for img in imgs_pp_list]

            # Synthesize
            if args.synthesis_type in ("encdec", "ensamble"):
                syn_lat_encdec = run_encdec_forward(unet_encdec, latens_list, device)
                syn_img_encdec = pipeline.decode_latents(syn_lat_encdec, vae)

            if args.synthesis_type in ("bbdm", "ensamble"):
                syn_lat_bbdm = run_bbdm_forward(
                    unet_bbdm, conditions_model, noise_scheduler, latens_list, device
                )
                syn_img_bbdm = pipeline.decode_latents(syn_lat_bbdm, vae)

            if args.synthesis_type == "encdec":
                syn_img = syn_img_encdec
            elif args.synthesis_type == "bbdm":
                syn_img = syn_img_bbdm
            else:
                syn_img = utils.combine_images(
                    [syn_img_encdec, syn_img_bbdm], combination_type='mean'
                )

            # Compute metrics
            met_whole = compute_metrics(syn_img, gt)
            met_brain = compute_metrics(syn_img, gt, mask=brain_mask)

            results.append({
                "subject": s_id,
                **{f"whole_{k}": v for k, v in met_whole.items()},
                **{f"brain_{k}": v for k, v in met_brain.items()},
            })

            # Save synthesized image if requested
            if args.save_output:
                EVAL_OUTPUT = os.path.join(configs.PATH_DATA, "eval_synthesized")
                os.makedirs(EVAL_OUTPUT, exist_ok=True)
                out_name = f[first_mod][:-10] + configs.MISSING_MODALITY + f[first_mod][-7:]
                syn_post = utils.postprocessing(syn_img, configs.MISSING_MODALITY, org_shape)
                utils.save_nifti(syn_post, aff, os.path.join(EVAL_OUTPUT, out_name))

            if args.verbose:
                print(f"  {s_id}: whole SSIM={met_whole['SSIM']:.4f}  PSNR={met_whole['PSNR']:.2f}  "
                      f"brain SSIM={met_brain['SSIM']:.4f}  PSNR={met_brain['PSNR']:.2f}")

        except Exception as e:
            print(f"  ERROR processing {s_id}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    torch.cuda.empty_cache()

    if not results:
        print("No subjects processed successfully.")
        return

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"SUMMARY  (n={len(results)}, type={args.synthesis_type})")
    print("=" * 60)

    metric_names = ["SSIM", "PSNR", "MSE", "MAE"]
    for region in ["whole", "brain"]:
        print(f"\n  {region.upper()} VOLUME:")
        for m in metric_names:
            key = f"{region}_{m}"
            vals = [r[key] for r in results]
            print(f"    {m:6s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    if args.save_csv:
        df = pd.DataFrame(results)
        df.to_csv(args.save_csv, index=False)
        print(f"\nPer-subject results saved to {args.save_csv}")


if __name__ == "__main__":
    main()
