




import os
import subprocess

import synthesis.utils as utils

# https://github.com/wasserth/TotalSegmentator#class-details
def segment_anything(img_path_name, out_path, task='total_mr', verify=False, verbose=False, gpu=None):
    if verify:
        exist = os.path.exists(out_path)
        if exist:
            print(f"Segmentaition already done for: {img_path_name}\nfile in: {out_path}")
            return True

    command = ["TotalSegmentator",
               "-i", img_path_name,
               "-o", out_path,
               "--task", task
               ]

    if gpu is not None:
        command.extend(["--device", f"gpu:{gpu}"])

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        if verbose:
            print(f"Segmentation done saved in: {out_path}")
    else:
        print("Error:", result.stderr)




def segment_brain_mask(path_name_img_list, path_out_intermediate, modality_names_list, gpu_id=None):
    bmask_list = []
    aff = None

    path_name_combined_bmask = os.path.join(path_out_intermediate, "combined_brain_mask.nii.gz")

    if os.path.exists(path_name_combined_bmask):
        print(f"Combined brain mask already exists at {path_name_combined_bmask}, skipping segmentation.")
        return utils.load_nifti(path_name_combined_bmask)[0]

    for modality_name, img_path_name in zip(modality_names_list, path_name_img_list):
        path_out_modality_total_segmentator = os.path.join(path_out_intermediate, "total_segmentator", f"{modality_name}")
        os.makedirs(path_out_modality_total_segmentator, exist_ok=True)

        # if os.path.exists(path_out_modality_total_segmentator):
        #     print(f"Brain mask already exists for {img_path_name}, skipping...")
        #     continue

        segment_anything(
            img_path_name=img_path_name,
            out_path=path_out_modality_total_segmentator,
            task='total_mr',
            verify=False,
            gpu=gpu_id,
            verbose=False
        )

        bmask, aff = utils.load_nifti(os.path.join(path_out_modality_total_segmentator, "brain.nii.gz"))
        bmask_list.append(bmask)

    combined_brain_mask = utils.combine_masks(bmask_list, combination="or")
    utils.save_nifti(combined_brain_mask, aff, path_name_combined_bmask)
    return combined_brain_mask
