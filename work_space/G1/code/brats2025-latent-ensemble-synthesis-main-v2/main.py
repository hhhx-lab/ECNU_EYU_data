import os
import sys
import csv
import shutil
import configs
import argparse

import synthesis.pipeline as pipeline
import synthesis.utils as utils


def read_subject_folder(subject_folder):
    file_list = sorted(os.listdir(subject_folder))
    # Always: t1n, t1c, t2f present; T2W is missing
    available_modalities = configs.AVAILABLE_MODALITIES  # ["t1n", "t1c", "t2f"]
    miss_one = configs.MISSING_MODALITY  # "t2w"
    if any(
        f.endswith(f"-{miss_one}.nii.gz") or f.endswith(f"-{miss_one}.nii")
        for f in file_list
    ):
        raise ValueError(
            f"{miss_one} is present; inference accepts only genuinely missing-{miss_one} cases"
        )

    # Build mapping from modality suffix to file path
    suffix_to_file = {}
    for f in file_list:
        if not (f.endswith(".nii.gz") or f.endswith(".nii")):
            continue
        for suffix in available_modalities:
            if f.endswith(f"-{suffix}.nii.gz") or f.endswith(f"-{suffix}.nii"):
                suffix_to_file[suffix] = f
                break

    missing = [m for m in available_modalities if m not in suffix_to_file]
    if missing:
        raise ValueError(f"missing required input modalities: {missing}")

    # Order files to match available_modalities order
    ordered_files = [os.path.join(subject_folder, suffix_to_file[m]) for m in available_modalities]

    out_name = f"{os.path.basename(subject_folder)}-{miss_one}.nii.gz"
    return ordered_files, available_modalities, miss_one, out_name


def find_seg_file(subject_folder):
    """Locate the seg file in a subject folder."""
    for f in sorted(os.listdir(subject_folder)):
        if f.endswith(('.nii.gz', '.nii')) and 'seg' in f.lower():
            return os.path.join(subject_folder, f)
    return None


def prepare_s_data(subject_folder, load_seg=False):
    """
    Reads the subject folder and returns a dictionary with the subject data.
    """
    s_data = {}
    s_data["s_id"] = os.path.basename(subject_folder)
    file_list, modality_list, miss_one, out_name = read_subject_folder(subject_folder)
    s_data["path_name_img_list"] = file_list
    s_data["available_modalitites_names"] = modality_list
    s_data["modality"] = miss_one
    s_data["out_name"] = out_name

    seg_path = find_seg_file(subject_folder)
    if seg_path is None:
        raise ValueError("missing required segmentation file")
    if load_seg:
        if seg_path is not None:
            import numpy as np
            seg, _ = utils.load_nifti(seg_path)
            seg, _ = utils.resize_center_crop_pad(seg, configs.SHAPE_PREPROCESS_IMG)
            s_data["seg"] = seg.astype(np.int16)
        else:
            s_data["seg"] = None

    return s_data


def read_data_folder(input_folder):
    """
    Reads the data folder and returns a list of subject folders.
    """
    subject_folders = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]
    subject_folders.sort()
    return subject_folders


def materialize_case_inputs(s_data, subject_folder, case_output_path):
    """Copy source modalities and seg so each completion case is self-contained."""
    os.makedirs(case_output_path, exist_ok=True)
    case_id = s_data["s_id"]
    for modality, source in zip(
        s_data["available_modalitites_names"], s_data["path_name_img_list"]
    ):
        extension = ".nii.gz" if source.endswith(".nii.gz") else ".nii"
        destination = os.path.join(
            case_output_path, f"{case_id}-{modality}{extension}"
        )
        shutil.copy2(source, destination)

    seg_source = find_seg_file(subject_folder)
    if seg_source is None:
        raise ValueError("missing required segmentation file")
    seg_extension = ".nii.gz" if seg_source.endswith(".nii.gz") else ".nii"
    shutil.copy2(
        seg_source,
        os.path.join(case_output_path, f"{case_id}-seg{seg_extension}"),
    )


def process_multiple_subjects(input_subject_list, synthesis_type, output_path,
                              gpu_id=None, verbose=False, compute_bmask=True,
                              per_lesion=False):
    """
    Processes multiple subject folders and runs synthesis for each.
    """
    os.makedirs(output_path, exist_ok=True)
    failures = []
    completed = 0
    # Pre-load models for per-lesion ROI synthesis (shared across subjects)
    roi_models = None
    if per_lesion:
        import torch
        from synthesis import roi_synthesis
        device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None else "cpu")
        print("Loading models for per-lesion ROI synthesis ...")
        roi_models = roi_synthesis.prepare_roi_models(device)
        print("Models loaded.")

    for subject_folder in input_subject_list:
        try:
            s_data = prepare_s_data(subject_folder, load_seg=per_lesion)
            if verbose:
                print("\n======== Starting Processing for Subject ========")
                print(f"Subject ID            : {s_data['s_id']}")
                print(f"Available Modalities  : {', '.join(s_data['available_modalitites_names'])}")
                print(f"Modality for Synthesis: {s_data['modality']}")
                print(f"Synthesis Type Chosen : {synthesis_type}")
                print(f"Create brain mask     : {compute_bmask}")
                print(f"Per-Lesion ROI        : {per_lesion}")
                print(f"Used GPU ID           : {gpu_id if gpu_id is not None else 'CPU'}")
                print("=" * 49 + "\n")

            out_name = s_data["out_name"]
            case_output_path = os.path.join(output_path, s_data["s_id"])
            materialize_case_inputs(s_data, subject_folder, case_output_path)

            if per_lesion:
                run_synthesis_per_lesion(s_data, synthesis_type, case_output_path,
                                         out_name, roi_models, gpu_id=gpu_id,
                                         verbose=verbose, compute_bmask=compute_bmask)
            else:
                # ---- Original pipeline (unchanged) ----
                pipeline.run_synthesis(s_data, synthesis_type, case_output_path,
                                       out_name, gpu_id=gpu_id, verbose=verbose,
                                       compute_bmask=compute_bmask)
            completed += 1
        except Exception as exc:
            case_id = os.path.basename(subject_folder)
            failures.append({"subject": case_id, "error": repr(exc)})
            print(f"[ERROR] {case_id}: {exc}")

    failure_csv = os.path.join(output_path, "inference_failures.csv")
    if failures:
        with open(failure_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["subject", "error"])
            writer.writeheader()
            writer.writerows(failures)
        raise RuntimeError(
            f"Inference failed for {len(failures)}/{len(input_subject_list)} cases; "
            f"see {failure_csv}."
        )
    if os.path.exists(failure_csv):
        os.remove(failure_csv)
    print(f"Inference completed successfully for {completed} cases.")


def run_synthesis_per_lesion(s_data, synthesis_type, output_path, output_name,
                             roi_models, gpu_id=None, verbose=False,
                             compute_bmask=False):
    """Run full-image ensemble synthesis + per-lesion ROI overlay."""
    import numpy as np
    import torch
    from synthesis import roi_synthesis

    device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None else "cpu")
    vae = roi_models["vae"]
    unet_encdec = roi_models["unet_encdec"]
    unet_bbdm = roi_models["unet_bbdm"]
    conditions_model = roi_models["conditions_model"]
    noise_scheduler = roi_models["noise_scheduler"]

    os.makedirs(output_path, exist_ok=True)
    path_out_intermediate = os.path.join(output_path, f"intermediate_{s_data['s_id']}")
    os.makedirs(path_out_intermediate, exist_ok=True)

    aff = None
    aff_preprocessed = None
    org_shape = None

    # 1. Preprocess images (same as standard pipeline)
    imgs_pp_list = []
    for i, path_name_img in enumerate(s_data["path_name_img_list"]):
        img, aff = utils.load_nifti(path_name_img)
        org_shape = img.shape
        img, aff_preprocessed = utils.preprocessing(img, affine=aff)
        imgs_pp_list.append(img)

    # Store preprocessed images for ROI extraction
    s_data["imgs_pp_list"] = imgs_pp_list

    # 2. Encode to latents
    latens_list = [pipeline.encode_image(img, vae) for img in imgs_pp_list]
    s_data["latens_list"] = latens_list

    # 3. Full-image ensemble synthesis (always ensemble for per-lesion)
    if verbose:
        print(f"  Running full-image ensemble synthesis...")
    syn_lat_encdec = pipeline.run_encdec_synthesis(s_data, device)
    syn_lat_bbdm = pipeline.run_bbdm_synthesis(s_data, device)

    syn_img_encdec = pipeline.decode_latents(syn_lat_encdec, vae)
    syn_img_bbdm = pipeline.decode_latents(syn_lat_bbdm, vae)
    base_syn_img = utils.combine_images([syn_img_encdec, syn_img_bbdm],
                                        combination_type='mean')

    # 4. Per-lesion ROI overlay
    seg = s_data.get("seg", None)
    if seg is not None and seg.max() > 0:
        if verbose:
            print(f"  Running per-lesion ROI overlay...")
        final_img = roi_synthesis.run_per_lesion_synthesis(
            s_data, base_syn_img, vae, unet_encdec, unet_bbdm,
            conditions_model, noise_scheduler, seg, device, verbose=verbose
        )
    else:
        if verbose:
            print(f"  [WARN] No seg found, skipping per-lesion ROI overlay.")
        final_img = base_syn_img

    # 5. Postprocess and save
    if compute_bmask:
        import synthesis.segment_brain_mask as segment_brain_mask
        bmask = segment_brain_mask.segment_brain_mask(
            s_data["path_name_img_list"], path_out_intermediate,
            s_data["available_modalitites_names"], gpu_id=gpu_id
        )
    else:
        bmask = None

    final_img = utils.postprocessing(final_img, configs.MISSING_MODALITY,
                                     org_shape, bmask=bmask)

    path_name_syn_img = os.path.join(output_path, output_name)
    utils.save_nifti(final_img, aff, path_name_syn_img)

    # Save intermediate raw outputs
    path_encdec_raw = os.path.join(path_out_intermediate, "raw_encdec_syn_img.nii.gz")
    path_bbdm_raw = os.path.join(path_out_intermediate, "raw_bbdm_syn_img.nii.gz")
    utils.save_nifti(utils.postprocessing_raw(syn_img_encdec, org_shape), aff, path_encdec_raw)
    utils.save_nifti(utils.postprocessing_raw(syn_img_bbdm, org_shape), aff, path_bbdm_raw)

    if verbose:
        print(f"  Synthesis completed. Output saved to: {path_name_syn_img}")



def main(args):
    input_subject_list = read_data_folder(args.input_dir)
    if not input_subject_list:
        raise RuntimeError(f"No subject folders found in {args.input_dir}")
    if args.verbose:
        print(f"Found {len(input_subject_list)} subject folders to process.")
    process_multiple_subjects(input_subject_list, args.synthesis_type,
                              args.output_dir, gpu_id=args.gpu_id,
                              verbose=args.verbose,
                              compute_bmask=args.compute_bmask,
                              per_lesion=args.per_lesion)



def parse_args():
    parser = argparse.ArgumentParser(description="Run synthesis pipeline")

    parser.add_argument(
        "--synthesis_type",
        type=str,
        choices=["encdec", "bbdm", "ensamble"],
        default="ensamble",
        help='Type of synthesis ("encdec", "bbdm", or "ensamble")'
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="GPU ID to use, set to None for CPU"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Set to enable detailed output"
    )
    parser.add_argument(
        "--compute_bmask",
        action="store_true",
        help="Set to segment brain mask"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=configs.PATH_INPUT_INFERENCE,
        help="Directory containing subject folders for inference"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=configs.PATH_OUTPUT,
        help="Directory where synthesized NIfTI files will be saved"
    )
    parser.add_argument(
        "--per_lesion",
        action="store_true",
        help="Enable per-lesion ROI synthesis overlay (requires seg files in subject folders)"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # defaut settings for local testing
    # args.gpu_id = 0
    # args.verbose = not args.verbose
    # args.compute_bmask = False

    # Example of how to set the arguments
    # args = {
    #     "synthesis_type": "encdec",  # "encdec", "bbdm", "ensamble"
    #     "gpu_id": 2,  # None for CPU
    #     "verbose": True,  # Set to True for detailed output
    #     "compute_bmask": True,  # Set to True to segment brain mask
    # }
    # args = utils.dict_to_args(args, deep_conversion=True)


    main(args)


# example to launch
# python main.py --synthesis_type encdec --gpu_id 2 --no_verbose --compute_bmask
# python main.py --per_lesion --gpu_id 0 --verbose  # per-lesion ROI pipeline

# how to create the docker file:
# https://github.com/BraTS/Instructions/tree/master/docker_templates/template_2020
