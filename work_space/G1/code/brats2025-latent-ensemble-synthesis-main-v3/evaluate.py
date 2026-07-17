"""Quantitative evaluation of synthesized T2W against ground truth.

Computes SSIM, PSNR, MSE, MAE per subject and aggregate statistics.
Reads subjects from data_csv.csv filtered by --split (default: val).

Usage:
    python evaluate.py                          # val set, ensemble
    python evaluate.py --split test             # test set
    python evaluate.py --synthesis_type bbdm    # BBDM only
    python evaluate.py --synthesis_type encdec   # EncDec only
    python evaluate.py --gpu_id 0 --verbose
    python evaluate.py --save_csv results.csv   # save per-subject CSV
"""

import csv
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

import configs
import synthesis.pipeline as pipeline
import synthesis.utils as utils
from synthesis.spatial import restore_to_native


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
    if mask is not None and mask.sum() > 0:
        coords = np.argwhere(mask > 0)
        lower = np.maximum(coords.min(axis=0) - 4, 0)
        upper = np.minimum(coords.max(axis=0) + 5, np.asarray(mask.shape))
        roi = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
        pred_ssim = pred[roi] * mask[roi]
        target_ssim = target[roi] * mask[roi]
    else:
        pred_ssim = pred
        target_ssim = target
    t1 = torch.tensor(pred_ssim).unsqueeze(0).unsqueeze(0).float()
    t2 = torch.tensor(target_ssim).unsqueeze(0).unsqueeze(0).float()
    ssim_loss_fn = SSIMLoss(spatial_dims=3, data_range=data_range)
    ssim_val = float(1.0 - ssim_loss_fn(t1, t2).item())

    return {"SSIM": ssim_val, "PSNR": psnr, "MSE": mse, "MAE": mae}


def create_brain_mask(images, threshold=0.02):
    """Create brain mask from mean of available modality images."""
    mean_img = np.mean(images, axis=0)
    return (mean_img > threshold).astype(np.float32)


def prepare_eval_subject(
    subject,
    *,
    target_shape=None,
    base_spacing_mm=1.0,
    margin_mm=5.0,
):
    modality_paths = [
        os.path.join(subject["path"], subject["files"][modality])
        for modality in configs.MODALITY_LIST
    ]
    seg_path = os.path.join(subject["path"], subject["files"]["seg"])
    foreground_indices = tuple(
        configs.MODALITY_LIST.index(modality)
        for modality in configs.AVAILABLE_MODALITIES
    )
    prepared = utils.prepare_subject_space(
        modality_paths,
        seg_path=seg_path,
        target_shape=target_shape or configs.SHAPE_PREPROCESS_IMG,
        base_spacing_mm=base_spacing_mm,
        margin_mm=margin_mm,
        foreground_indices=foreground_indices,
    )
    prepared["images_by_modality"] = dict(
        zip(configs.MODALITY_LIST, prepared["images"])
    )
    return prepared


def save_synthesized_output(model_image, prepared, destination):
    native = restore_to_native(model_image, prepared["transform"], order=1)
    native = utils.postprocess_intensity(native, configs.MISSING_MODALITY)
    utils.save_nifti(native, prepared["native_affine"], str(destination))
    return native


def find_eval_subjects(data_csv, input_dir, split="val", case_ids=None):
    """Load subjects from data_csv.csv filtered by split column.

    Returns list of dicts with keys: id, path, files (modality→filename mapping).
    """
    subjects = []
    with open(data_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split", "val") != split:
                continue
            s_id = row["id"]
            if case_ids is not None and s_id not in case_ids:
                continue
            s_path = os.path.join(input_dir, s_id)

            # Build modality → filename mapping from CSV columns
            mod_to_file = {}
            for mod in configs.MODALITY_LIST:
                fname = os.path.basename(row.get(mod, ""))
                if fname:
                    mod_to_file[mod] = fname

            seg_name = os.path.basename(row.get("seg", ""))
            if seg_name:
                mod_to_file["seg"] = seg_name

            if all(key in mod_to_file for key in (*configs.MODALITY_LIST, "seg")):
                subjects.append({
                    "id": s_id,
                    "path": s_path,
                    "files": mod_to_file,
                })

    if case_ids is not None:
        found = {subject["id"] for subject in subjects}
        missing = sorted(set(case_ids) - found)
        if missing:
            raise ValueError(f"requested evaluation cases were not found in split {split}: {missing}")
    return subjects


def read_case_list(path):
    if path is None:
        return None
    case_ids = set()
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line in handle:
            value = line.strip().split(",", 1)[0].strip()
            if value and not value.lower() in {"id", "case_id", "subject"}:
                case_ids.add(value)
    if not case_ids:
        raise ValueError(f"case list is empty: {path}")
    return case_ids


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
    parser.add_argument("--split", type=str, default="val",
                        help="Which CSV split to evaluate (default: val)")
    parser.add_argument(
        "--save_output",
        action="store_true",
        help="Save synthesized NIfTI files to data/eval_synthesized/"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(configs.PATH_DATA, "eval_synthesized"),
        help="Directory for --save_output synthesized NIfTI files"
    )
    parser.add_argument(
        "--data_csv",
        type=str,
        default=os.path.join(configs.PATH_DATA, "data_csv.csv"),
        help="Path to data_csv.csv"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=configs.PATH_INPUT,
        help="Directory containing subject folders (default: data/input)"
    )
    parser.add_argument(
        "--per_lesion",
        action="store_true",
        help="Enable per-lesion ROI synthesis overlay (requires seg files, always uses ensemble)"
    )
    parser.add_argument(
        "--case-list",
        type=str,
        default=None,
        help="Optional text/CSV file whose first column lists case IDs to evaluate",
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cpu")

    case_ids = read_case_list(args.case_list)
    subjects = find_eval_subjects(
        args.data_csv,
        args.input_dir,
        split=args.split,
        case_ids=case_ids,
    )
    if not subjects:
        raise RuntimeError(
            f"No subjects with all 4 modalities found in {args.input_dir} "
            f"(split='{args.split}')"
        )

    print(f"Found {len(subjects)} subjects (split='{args.split}').")

    # --- Instantiate models once ---
    print("Loading VAE ...")
    vae = pipeline.instantiate_vae_model(device)

    unet_encdec = None
    unet_bbdm = None
    conditions_model = None
    noise_scheduler = None

    # Per-lesion ROI always needs both EncDec and BBDM
    _load_encdec = args.synthesis_type in ("encdec", "ensamble") or args.per_lesion
    _load_bbdm = args.synthesis_type in ("bbdm", "ensamble") or args.per_lesion

    if _load_encdec:
        print("Loading EncDec model ...")
        unet_encdec = pipeline.instantiate_encdec_model(device)

    if _load_bbdm:
        print("Loading BBDM model ...")
        unet_bbdm, conditions_model, noise_scheduler = pipeline.instantiate_bbdm_model(device)

    # --- Evaluate each subject ---
    results = []
    failures = []
    spatial_audits = []

    for subj in tqdm(subjects, desc="Evaluating"):
        s_id = subj["id"]
        s_path = subj["path"]
        f = subj["files"]

        try:
            prepared = prepare_eval_subject(subj)
            images_by_modality = prepared["images_by_modality"]
            imgs_pp_list = [
                images_by_modality[modality]
                for modality in configs.AVAILABLE_MODALITIES
            ]
            gt = images_by_modality[configs.MISSING_MODALITY]

            # Brain mask from available modalities
            brain_mask = create_brain_mask(imgs_pp_list)

            # Encode to latents
            utils.set_seed(42)
            latens_list = [pipeline.encode_image(img, vae) for img in imgs_pp_list]

            # Synthesize (always ensemble for per_lesion)
            _use_encdec = args.synthesis_type in ("encdec", "ensamble") or args.per_lesion
            _use_bbdm = args.synthesis_type in ("bbdm", "ensamble") or args.per_lesion

            if _use_encdec:
                syn_lat_encdec = run_encdec_forward(unet_encdec, latens_list, device)
                syn_img_encdec = pipeline.decode_latents(syn_lat_encdec, vae)

            if _use_bbdm:
                syn_lat_bbdm = run_bbdm_forward(
                    unet_bbdm, conditions_model, noise_scheduler, latens_list, device
                )
                syn_img_bbdm = pipeline.decode_latents(syn_lat_bbdm, vae)

            if args.per_lesion:
                syn_img = utils.combine_images(
                    [syn_img_encdec, syn_img_bbdm], combination_type='mean'
                )
            elif args.synthesis_type == "encdec":
                syn_img = syn_img_encdec
            elif args.synthesis_type == "bbdm":
                syn_img = syn_img_bbdm
            else:
                syn_img = utils.combine_images(
                    [syn_img_encdec, syn_img_bbdm], combination_type='mean'
                )

            # ---- Per-Lesion ROI overlay (inference-only, parallel pipeline) ----
            if args.per_lesion:
                from synthesis import roi_synthesis
                seg = prepared["segmentation"]
                if seg is not None and seg.max() > 0:
                    s_data = {
                        "s_id": s_id,
                        "imgs_pp_list": imgs_pp_list,
                    }
                    syn_img = roi_synthesis.run_per_lesion_synthesis(
                        s_data, syn_img, vae, unet_encdec, unet_bbdm,
                        conditions_model, noise_scheduler, seg, device,
                        verbose=args.verbose
                    )

            # Compute metrics
            met_whole = compute_metrics(syn_img, gt)
            met_brain = compute_metrics(syn_img, gt, mask=brain_mask)

            results.append({
                "subject": s_id,
                **{f"whole_{k}": v for k, v in met_whole.items()},
                **{f"brain_{k}": v for k, v in met_brain.items()},
            })
            transform = prepared["transform"]
            spatial_audits.append({
                "subject": s_id,
                "native_shape": "x".join(str(value) for value in transform.native_shape),
                "target_shape": "x".join(str(value) for value in transform.target_shape),
                "target_spacing_mm": transform.target_spacing_mm,
                "foreground_voxel_count": transform.foreground_voxel_count,
                "lesion_voxel_count": transform.lesion_voxel_count,
                "foreground_outside_voxel_count": prepared["foreground_support_audit"]["outside_voxel_count"],
                "lesion_outside_voxel_count": (
                    prepared["lesion_support_audit"]["outside_voxel_count"]
                    if prepared["lesion_support_audit"] is not None
                    else 0
                ),
            })

            # Save synthesized image if requested
            if args.save_output:
                EVAL_OUTPUT = args.output_dir
                os.makedirs(EVAL_OUTPUT, exist_ok=True)
                out_name = f"{s_id}-{configs.MISSING_MODALITY}.nii.gz"
                save_synthesized_output(
                    syn_img,
                    prepared,
                    os.path.join(EVAL_OUTPUT, out_name),
                )

            if args.verbose:
                print(f"  {s_id}: whole SSIM={met_whole['SSIM']:.4f}  PSNR={met_whole['PSNR']:.2f}  "
                      f"brain SSIM={met_brain['SSIM']:.4f}  PSNR={met_brain['PSNR']:.2f}")

        except Exception as e:
            print(f"  ERROR processing {s_id}: {e}")
            failures.append({"subject": s_id, "error": repr(e)})
            if args.verbose:
                import traceback
                traceback.print_exc()

    torch.cuda.empty_cache()

    failure_dir = os.path.dirname(args.save_csv) if args.save_csv else configs.PATH_DATA
    failure_dir = failure_dir or "."
    failure_csv = os.path.join(failure_dir, f"eval_failures_{args.split}.csv")
    if failures:
        os.makedirs(os.path.dirname(failure_csv), exist_ok=True)
        pd.DataFrame(failures).to_csv(failure_csv, index=False)
        if results and args.save_csv:
            pd.DataFrame(results).to_csv(args.save_csv, index=False)
        if spatial_audits and args.save_csv:
            pd.DataFrame(spatial_audits).to_csv(
                Path(args.save_csv).with_name("spatial_audit.csv"), index=False
            )
        raise RuntimeError(
            f"Evaluation failed for {len(failures)}/{len(subjects)} subjects; "
            f"see {failure_csv}."
        )
    if os.path.exists(failure_csv):
        os.remove(failure_csv)

    if not results:
        raise RuntimeError("No subjects processed successfully.")

    # --- Summary ---
    print("\n" + "=" * 60)
    _mode_str = f"{args.synthesis_type}" + ("+per_lesion" if args.per_lesion else "")
    print(f"SUMMARY  (n={len(results)}, type={_mode_str})")
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
        pd.DataFrame(spatial_audits).to_csv(
            Path(args.save_csv).with_name("spatial_audit.csv"), index=False
        )
        print(f"\nPer-subject results saved to {args.save_csv}")


if __name__ == "__main__":
    main()
