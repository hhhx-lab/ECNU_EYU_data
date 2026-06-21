import os
import csv
import numpy as np
from scipy import ndimage
import nibabel as nib
import argparse
from pathlib import Path


REQUIRED_MODALITIES = ("t1c", "t2w", "t2f", "t1n")
MET_PREFIX = "BraTS-MET-"


def find_g1_workspace_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if parent.name == "G1" and (parent / "code").exists() and (parent / "docs").exists():
            return parent
    raise RuntimeError(f"Could not locate work_space/G1 from {start}")


PROJECT_ROOT = find_g1_workspace_root(Path(__file__).resolve())
DEFAULT_RAW_DATA_ROOT = PROJECT_ROOT / "data" / "raw"

def center_of_mass(mask_data):
    """
    Compute the center of mass of a binary mask
    Returns:
        x, y and z center of mass
    """
    x, y, z = ndimage.center_of_mass(mask_data) # This gives the center of mass 
    return round(x), round(y), round(z)


def bbox_and_center(mask_data):
    coords = np.argwhere(mask_data)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    center = np.rint(coords.mean(axis=0)).astype(int)
    size = maxs - mins
    return {
        "center": [int(center[0]), int(center[1]), int(center[2])],
        "mins": [int(mins[0]), int(mins[1]), int(mins[2])],
        "maxs": [int(maxs[0]), int(maxs[1]), int(maxs[2])],
        "size": [int(size[0]), int(size[1]), int(size[2])],
    }

def x_extremes(matrix): 
    '''
    This function gives the extremes of the bounding box in the x axis.
    The min and max are the first and the last slice where the tumour label is non zero.
        Parameters:
                matrix (array): numpy array of the label
        Returns:
                min_x (int): x position of the first slice with non zero tumour voxel 
                max_x (int): x position of the last slice with non zero tumour voxel 
    '''
    min_x = 0
    max_x = 500
    for x_idx, x_slice in enumerate(matrix):
        if sum(sum(x_slice))>=1:
                min_x = x_idx+1
                break
    for x_idx, x_slice in enumerate((np.fliplr(matrix))):
       if sum(sum(x_slice))>=1:
        max_x = x_idx+1
    return min_x, max_x

def y_extremes(matrix): 
    '''
    This function gives the extremes of the bounding box in the y axis.
    The min and max are the first and the last slice where the tumour label is non zero.
        Parameters:
                matrix (array): numpy array of the label
        Returns:
                min_y (int): y position of the first slice with non zero tumour voxel 
                max_y (int): y position of the last slice with non zero tumour voxel 
    '''
    min_y = 0
    max_y = 500
    for y_idx, y_slice in enumerate(matrix.transpose(1,0,2)):
        if sum(sum(y_slice))>=1:
                min_y = y_idx+1
                break
    for y_idx, y_slice in enumerate((np.fliplr(matrix.transpose(1,0,2)))):
       if sum(sum(y_slice))>=1:
        max_y = y_idx+1
    return min_y, max_y

def z_extremes(matrix): 
    '''
    This function gives the extremes of the bounding box in the z axis.
    The min and max are the first and the last slice where the tumour label is non zero.
        Parameters:
                matrix (array): numpy array of the label
        Returns:
                min_z (int): z position of the first slice with non zero tumour voxel 
                max_z (int): z position of the last slice with non zero tumour voxel 
    '''
    min_z = 0
    max_z = 500
    for z_idx, z_slice in enumerate(matrix.transpose(2,1,0)):
        if sum(sum(z_slice))>=1:
                min_z = z_idx+1
                break
    for z_idx, z_slice in enumerate((np.fliplr(matrix.transpose(2,1,0)))):
       if sum(sum(z_slice))>=1:
        max_z = z_idx + 1
    return min_z, max_z

def case_id_is_allowed(case_id, args):
    if args.require_met and not case_id.startswith(MET_PREFIX):
        return False
    return True


def find_case_dirs(datadir):
    root = Path(datadir)
    if not root.exists():
        raise FileNotFoundError(f"datadir does not exist: {datadir}")
    if root.is_dir() and root.name.startswith(MET_PREFIX):
        return [root]
    case_dirs = []
    for path in root.rglob(f"{MET_PREFIX}*"):
        if path.is_dir():
            case_dirs.append(path)
    return sorted(case_dirs, key=lambda p: str(p))


def modality_from_name(file_name, args):
    name = file_name.lower()
    suffix_map = {
        "t1c": args.t1c_ending.lower(),
        "t2w": args.t2w_ending.lower(),
        "t2f": args.t2f_ending.lower(),
        "t1n": args.t1n_ending.lower(),
        "seg": args.seg_ending.lower(),
    }
    for modality, ending in suffix_map.items():
        if name.endswith(ending):
            return modality
    return None


def get_training_dict(args, datadir):
    '''
    Creates a dictionary with the scans and the lables
        Parameters:
                datadir (str): path to the data directory
        Returns:
                training_dict (dict): dictionary with image:path and label:path
    '''
    training = []
    skipped = []
    for case_dir in find_case_dirs(datadir):
        case_id = case_dir.name
        if not case_id_is_allowed(case_id, args):
            skipped.append({
                "id": case_id,
                "case_dir": str(case_dir),
                "reason": "non_met_case_id",
                "present_modalities": "",
                "files": "",
            })
            continue

        files = sorted([path for path in case_dir.iterdir() if path.is_file() and path.name.endswith((".nii", ".nii.gz"))])
        modality_map = {}
        for file_path in files:
            modality = modality_from_name(file_path.name, args)
            if modality:
                modality_map[modality] = str(file_path)

        missing = [modality for modality in REQUIRED_MODALITIES + ("seg",) if modality not in modality_map]
        if missing:
            skipped.append({
                "id": case_id,
                "case_dir": str(case_dir),
                "reason": f"missing:{','.join(missing)}",
                "present_modalities": ",".join([mod for mod in REQUIRED_MODALITIES + ("seg",) if mod in modality_map]),
                "files": ",".join([path.name for path in files]),
            })
            print(f"Skip {case_id}: missing {', '.join(missing)}")
            continue

        dict_entry = {
            "id": case_id,
            "image": [modality_map[mod] for mod in REQUIRED_MODALITIES],
            "label": modality_map["seg"],
        }
        training.append(dict_entry)
    training_dict = {"training" : training, "skipped": skipped}
    return training_dict

def modal_paths(args, mask_path):
                """
                Returns the path to the respective modal
                """
                scan_path_t1c = None
                scan_path_t2w = None
                scan_path_t2f = None
                scan_path_t1n = None
                for scan_paths in mask_path['image']:
                    if args.t1c_ending in scan_paths:
                        scan_path_t1c = scan_paths
                    elif args.t2w_ending in scan_paths:
                        scan_path_t2w = scan_paths
                    elif args.t2f_ending in scan_paths:
                        scan_path_t2f = scan_paths
                    elif args.t1n_ending in scan_paths:
                        scan_path_t1n = scan_paths
                label_path = mask_path['label']
                return scan_path_t1c, scan_path_t2w, scan_path_t2f, scan_path_t1n, label_path

def create_csv(args, DATASET_NAME, CSV_PATH, DATADIR):
    """ 
    Creation of the complete CSV dataset with size smaller than or equal to 96 in all directions
    """
    # Define the header for the csv file
    header = ['id', 'scan_t1c', 'scan_t2w', 'scan_t2f', 'scan_t1n', 'label', 'center_x', 'center_y', 'center_z',
            'x_extreme_min', 'x_extreme_max', 'y_extreme_min', 'y_extreme_max', 'z_extreme_min', 'z_extreme_max', 'x_size', 'y_size', 'z_size']

    # Getting all files in a folder DATADIR
    ### Create a dictionary with scans and labels paths
    training_dict = get_training_dict(args, DATADIR)
    training = training_dict['training']
    written_rows = 0
    with open(CSV_PATH, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for mask_path in training:
            if ("brats" in DATASET_NAME.lower()) and ("goat" in DATASET_NAME.lower()) and ("2024" in DATASET_NAME.lower()):
                id = mask_path['label'].split("/")[-2][-5:] # For BraTS 2024 GOAT
            elif ("brats" in DATASET_NAME.lower()) and ("2023" in DATASET_NAME.lower()) and ("goat" not in DATASET_NAME.lower()) and ("meningioma" not in DATASET_NAME.lower()):
                id = mask_path['label'].split("/")[-2][-9:] # For BraTS 2023
            elif ("brats" in DATASET_NAME.lower()) and (("2024" in DATASET_NAME.lower()) or ("2026" in DATASET_NAME.lower())) and ("goat" not in DATASET_NAME.lower()) and ("meningioma" not in DATASET_NAME.lower()):
                id = mask_path['label'].split("/")[-2] # For BraTS MET 2024/2026
            elif  ("brats" in DATASET_NAME.lower()) and ("meningioma" in DATASET_NAME.lower()):
                id = mask_path['label'].split("/")[-2][-6:] # For BraTS Meningioma Radiotherapy 2024
            else:
                raise ValueError("Datasets available: Brats2023, Brats_goat2024, Brats2024 or Brats2024_Meningioma")

            if not case_id_is_allowed(id, args):
                print(f"Case id: {id} was skipped (non MET case id)")
                continue

            print(f"Doing case ID: {id}")
            # Load mask data
            mask = nib.load(mask_path['label'])
            mask_data = np.asarray(mask.get_fdata())
            # Binary mask
            mask_data = np.where(mask_data > 0.5, 1, 0)
            mask_data = mask_data > 0 
            # Dividing by modalities
            scan_path_t1c, scan_path_t2w, scan_path_t2f, scan_path_t1n, label_path = modal_paths(args, mask_path)
            missing_modal_paths = [
                name for name, value in (
                    ("t1c", scan_path_t1c),
                    ("t2w", scan_path_t2w),
                    ("t2f", scan_path_t2f),
                    ("t1n", scan_path_t1n),
                    ("seg", label_path),
                )
                if value is None
            ]
            if missing_modal_paths:
                print(f"Case id: {id} was skipped (missing {', '.join(missing_modal_paths)})")
                continue

            bbox = bbox_and_center(mask_data)
            if bbox is None:
                print(f"Case id: {id} was skipped (empty tumour label)")
                continue
            center_x, center_y, center_z = bbox["center"]
            min_x, min_y, min_z = bbox["mins"]
            max_x, max_y, max_z = bbox["maxs"]
            x_size, y_size, z_size = bbox["size"]

            # Only cases with a size smaller than or equal to 96 in all directions are used.
            if x_size<=96 and y_size<=96 and  z_size<=96:
                # Save one line in the csv
                row = [id, scan_path_t1c, scan_path_t2w, scan_path_t2f, scan_path_t1n, label_path, center_x, center_y, center_z, min_x, max_x, min_y, max_y, min_z, max_z, x_size, y_size, z_size]
                writer.writerow(row)
                written_rows += 1
            else:
                print(f"Case id: {id} was skipped (one dimention bigger than 96)")
        print(f"Done. Saved in {CSV_PATH}")
    skipped_rows = training_dict.get("skipped", [])
    if skipped_rows:
        skipped_path = os.path.splitext(CSV_PATH)[0] + "_skipped.csv"
        with open(skipped_path, "w", newline="") as skipped_f:
            fieldnames = ["id", "case_dir", "reason", "present_modalities", "files"]
            writer = csv.DictWriter(skipped_f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(skipped_rows)
        print(f"Skipped manifest saved in {skipped_path}")
    if written_rows == 0:
        raise RuntimeError(
            "No training rows were written. Check that datadir contains new BraTS-MET-* "
            "cases with complete t1c/t1n/t2w/t2f/seg and tumour bbox <= 96."
        )
    return training

def __main__():
    parser = argparse.ArgumentParser(description="Label generator Training")
    parser.add_argument("--logdir", default="brats2026_diffusion", type=str, help="Directory to save the experiment (save CSV)")
    parser.add_argument("--dataset", default="BRATS_2026", type=str, help="Dataset name. Current 2026 MET line should use BRATS_2026.")
    parser.add_argument(
        "--datadir",
        default=os.environ.get("TASK1_ROOT", str(DEFAULT_RAW_DATA_ROOT)),
        type=str,
        help="2026 raw data root. Default is work_space/G1/data/raw; set TASK1_ROOT only if you mount it elsewhere.",
    )
    parser.add_argument("--debug", default="True", type=str, help="If want to show some output for debugging")
    parser.add_argument("--csv_path", default="", type=str, help="Path to the CSV with all cases to use when training")
    parser.add_argument("--seg_ending", default="seg.nii.gz", type=str, help="Ending to the segmentation file. Important to locate the correct segmentation file. Default: seg.nii.gz")
    parser.add_argument("--t1n_ending", default="t1n.nii.gz", type=str, help="Ending to the t1n file. Important to locate the correct t1n file. Default: t1n.nii.gz")
    parser.add_argument("--t1c_ending", default="t1c.nii.gz", type=str, help="Ending to the t1c file. Important to locate the correct t1c file. Default: t1c.nii.gz")
    parser.add_argument("--t2w_ending", default="t2w.nii.gz", type=str, help="Ending to the t2w file. Important to locate the correct t2w file. Default: t2w.nii.gz")
    parser.add_argument("--t2f_ending", default="t2f.nii.gz", type=str, help="Ending to the t2f file. Important to locate the correct t2f file. Default: t2f.nii.gz")
    parser.add_argument("--require_met", default="True", type=str, help="Only include BraTS-MET-* cases. Default: True")
    args = parser.parse_args()
    args.require_met = str(args.require_met).strip().lower() in {"1", "true", "yes", "y"}

    if not args.datadir:
        raise SystemExit("Missing --datadir. The default raw root should be work_space/G1/data/raw.")

    # Making dir to save the csv
    HOME_DIR = f'../../Checkpoint/{args.logdir}'
    if not os.path.exists(HOME_DIR):
        os.makedirs(HOME_DIR)
        print(f"Directory {HOME_DIR} created")
    else:
        print(f"Directory {HOME_DIR} already exists")
    if not os.path.exists(f"{HOME_DIR}/debug"):
        os.makedirs(f"{HOME_DIR}/debug")
        print(f"Directory {HOME_DIR}/debug created")
    else:
        print(f"Directory {HOME_DIR}/debug already exists")

    if args.csv_path == "":
        CSV_PATH = f'../../Checkpoint/{args.logdir}/{args.logdir}.csv'
    else:
        CSV_PATH = args.csv_path
    print(f"CSV_PATH: {CSV_PATH}")

    training = create_csv(args=args, DATASET_NAME=args.dataset, CSV_PATH=CSV_PATH, DATADIR=args.datadir)

    
    if args.debug=="True":
        print("####################################")
        print(f"Output for debug")
        print(f"Number of cases in the training dict: {len(training)}")
        # Check CSV file
        import pandas as pd
        df = pd.read_csv(CSV_PATH)
        print(f"Number of rows in the csv file: {len(df)}")
        print(f"If the {len(training)}!={len(df)}, it is normal, as some cases have tumours bigger than 96.\nIn case you did not see any line saying 'Case id: id was skipped (one dimention bigger than 96)' something is wrong.")
        print(f"### Some rows of the csv file ###")
        print(df)

        # Getting some data from the dataframe
        # Get the scan list
        print(f"Path to the t1c file: {df['scan_t1c'][0]}")
        print(f"Path to the t2w file: {df['scan_t2w'][0]}")
        print(f"Path to the t2f file: {df['scan_t2f'][0]}")
        print(f"Path to the t1n file: {df['scan_t1n'][0]}")
        # Get the label
        print(f"Path to the label file: {df['label'][0]}")
        # Get the center
        print(f"Center of mass -> x: {df['center_x'][0]}, y: {df['center_y'][0]}, z: {df['center_z'][0]}")

        # Create an image with a sample (several slices)
        print(f"Creating an image with a sample (several slices)")
        import nibabel as nib
        import numpy as np
        import matplotlib.pyplot as plt

        def visualize_sample(idx, slice, types=('scan_t1c','scan_t2w', 'scan_t2f', 'scan_t1n')):
            plt.figure(figsize=(16, 5))
            for i, t in enumerate(types, 1):
                data = nib.load(df[t][idx])
                data = np.asarray(data.get_fdata())
                plt.subplot(1, 4, i)
                plt.imshow(data[:,:,slice], cmap='gray')
                plt.title(f'{t}', fontsize=16)
                plt.axis('off')
            plt.suptitle(f'idx: {idx}', fontsize=16)
            plt.savefig(f"{HOME_DIR}/debug/sample_{slice}.png", format='png')
            plt.close()
        
        if ("brats" in args.dataset.lower()) and ("meningioma" in args.dataset.lower()):
            types = ['scan_t1c']
        else:
            types=('scan_t1c','scan_t2w', 'scan_t2f', 'scan_t1n')
        for slice in range (5):
            visualize_sample(idx=0, slice=100+slice*5, types=types)

if __name__ == "__main__":
    __main__()
    print("Finished!")
