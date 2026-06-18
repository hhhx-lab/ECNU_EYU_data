

import os
import re
import configs
import argparse

import synthesis.pipeline as pipeline
import synthesis.utils as utils


MODALITY_PATTERN = re.compile(
    r"^(?P<prefix>.+?)-(?P<mod>t1n|t1c|t2w|t2f|seg)(?P<suffix>\.nii(?:\.gz)?)$",
    re.IGNORECASE,
)
SYNTHESIS_INPUT_MODALITIES = tuple(configs.AVAILABLE_MODALITIES)  # t1n, t1c, t2f
REQUIRED_INFERENCE_FILES = SYNTHESIS_INPUT_MODALITIES + ("seg",)


def parse_modality_file(file_name):
    match = MODALITY_PATTERN.match(file_name)
    if not match:
        return None
    return {
        "prefix": match.group("prefix"),
        "modality": match.group("mod").lower(),
        "suffix": match.group("suffix"),
    }


def read_subject_folder(subject_folder):
    file_list = sorted(os.listdir(subject_folder))
    # Always synthesize from t1n/t1c/t2f. A seg file is required for downstream QC,
    # while T2W is intentionally absent or ignored for inference.
    available_modalities = list(SYNTHESIS_INPUT_MODALITIES)
    miss_one = configs.MISSING_MODALITY  # "t2w"

    # Build mapping from modality suffix to file path
    suffix_to_file = {}
    parsed_by_modality = {}
    for f in file_list:
        parsed = parse_modality_file(f)
        if parsed is None:
            continue
        suffix = parsed["modality"]
        if suffix in REQUIRED_INFERENCE_FILES or suffix == miss_one:
            suffix_to_file[suffix] = f
            parsed_by_modality[suffix] = parsed

    missing = [mod for mod in REQUIRED_INFERENCE_FILES if mod not in suffix_to_file]
    if missing:
        raise ValueError(
            f"missing required inference files: {', '.join(missing)}"
        )

    # Order files to match available_modalities order
    ordered_files = [os.path.join(subject_folder, suffix_to_file[m]) for m in available_modalities]
    seg_path = os.path.join(subject_folder, suffix_to_file["seg"])

    reference = parsed_by_modality[available_modalities[0]]
    out_name = f"{reference['prefix']}-{miss_one}{reference['suffix']}"
    return ordered_files, available_modalities, miss_one, out_name, seg_path



def prepare_s_data(subject_folder):
    """
    Reads the subject folder and returns a dictionary with the subject data.
    """
    s_data = {}
    s_data["s_id"] = os.path.basename(subject_folder)
    file_list, modality_list, miss_one, out_name, seg_path = read_subject_folder(subject_folder)
    s_data["path_name_img_list"] = file_list
    s_data["available_modalitites_names"] = modality_list
    s_data["modality"] = miss_one
    s_data["out_name"] = out_name
    s_data["seg_path"] = seg_path

    return s_data


def mirror_source_case(s_data, output_case_dir, overwrite=False):
    """Mirror source inference files into the output case directory."""
    os.makedirs(output_case_dir, exist_ok=True)
    source_files = list(s_data["path_name_img_list"])
    seg_path = s_data.get("seg_path")
    if seg_path:
        source_files.append(seg_path)

    for src_path in source_files:
        if not src_path or not os.path.exists(src_path):
            continue
        dst_path = os.path.join(output_case_dir, os.path.basename(src_path))
        if os.path.lexists(dst_path):
            if not overwrite:
                continue
            os.remove(dst_path)
        os.symlink(src_path, dst_path)


def read_data_folder(input_folder):
    """
    Reads the data folder and returns a list of subject folders.
    """
    subject_folders = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]
    subject_folders.sort()
    return subject_folders

def process_multiple_subjects(input_subject_list, synthesis_type, output_path, gpu_id=None, verbose=False, compute_bmask=True):
    """
    Processes multiple subject folders and runs synthesis for each.
    """
    processed = 0
    skipped = 0
    for subject_folder in input_subject_list:
        try:
            s_data = prepare_s_data(subject_folder)
        except ValueError as exc:
            skipped += 1
            if verbose:
                print(f"Skipping {os.path.basename(subject_folder)}: {exc}")
            continue

        case_output_dir = os.path.join(output_path, s_data["s_id"])
        os.makedirs(case_output_dir, exist_ok=True)
        mirror_source_case(s_data, case_output_dir, overwrite=True)

        if verbose:
            print("\n======== Starting Processing for Subject ========")
            print(f"Subject ID            : {s_data['s_id']}")
            print(f"Available Modalities  : {', '.join(s_data['available_modalitites_names'])}")
            print(f"Segmentation File     : {s_data['seg_path']}")
            print(f"Modality for Synthesis: {s_data['modality']}")
            print(f"Synthesis Type Chosen : {synthesis_type}")
            print(f"Create brain mask     : {compute_bmask}")
            print(f"Used GPU ID           : {gpu_id if gpu_id is not None else 'CPU'}")
            print("=" * 49 + "\n")

        out_name = s_data["out_name"]
        pipeline.run_synthesis(s_data, synthesis_type, case_output_dir, out_name, gpu_id=gpu_id, verbose=verbose, compute_bmask=compute_bmask)
        processed += 1

    print(f"Processed subjects: {processed}; skipped incomplete subjects: {skipped}")



def main(args):
    input_subject_list = read_data_folder(args.input_dir)
    if not input_subject_list:
        print(f"No subject folders found in {args.input_dir}")
        return
    if args.verbose:
        print(f"Found {len(input_subject_list)} subject folders to process.")
    process_multiple_subjects(input_subject_list, args.synthesis_type, args.output_dir, gpu_id=args.gpu_id, verbose=args.verbose, compute_bmask=args.compute_bmask)



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
        help="Directory where synthesized case folders will be written"
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

# how to create the docker file:
# https://github.com/BraTS/Instructions/tree/master/docker_templates/template_2020
