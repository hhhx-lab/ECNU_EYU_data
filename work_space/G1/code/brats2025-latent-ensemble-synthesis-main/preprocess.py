import os
import re
import sys
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

import configs
import synthesis.utils as utils
import synthesis.pipeline as pipeline


MODALITY_PATTERN = re.compile(r"^(?P<prefix>.+?)-(?P<mod>t1n|t1c|t2w|t2f|seg)\.(?:nii|nii\.gz)$", re.IGNORECASE)
TRAIN_MODALITIES = tuple(configs.MODALITY_LIST)
SKIPPED_CSV_NAME = "data_csv_skipped_subjects.csv"


def extract_modality(file_name):
    match = MODALITY_PATTERN.match(file_name)
    if not match:
        return None
    return match.group("mod").lower()


def scan_subjects(input_dir):
    """扫描所有受试者文件夹，识别可用模态。

    训练集只接收 t1n/t1c/t2w/t2f 四模态齐全的病例。
    缺 T2W 或任何其他必需模态的病例会被跳过，并写入 skipped 清单。
    """
    subjects = []
    skipped = []
    for folder in sorted(os.listdir(input_dir)):
        folder_path = os.path.join(input_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        files = sorted(os.listdir(folder_path))
        files = [f for f in files if f.endswith(".nii.gz") or f.endswith(".nii")]

        # 从文件名提取模态标识
        modality_map = {}
        for f in files:
            modality = extract_modality(f)
            if modality is None:
                continue
            if modality in configs.MODALITY_LIST:
                modality_map[modality] = f
            elif modality == "seg":
                # seg 对训练入表不是必需，但允许出现在目录里
                continue
            else:
                print(f"  Warning: unrecognized modality '{modality}' in file {f}")

        missing_modalities = [mod for mod in TRAIN_MODALITIES if mod not in modality_map]
        if missing_modalities:
            skipped.append({
                "id": folder,
                "reason": f"missing:{','.join(missing_modalities)}",
                "present_modalities": ",".join([m for m in TRAIN_MODALITIES if m in modality_map]),
                "files": ",".join(files),
            })
            print(f"  Skip {folder}: missing {', '.join(missing_modalities)}")
            continue

        subjects.append({
            "id": folder,
            "path": folder_path,
            "modality_map": modality_map,
        })

    return subjects, skipped


def preprocess_subject(sub_data, vae, output_latents_dir, device):
    """对单个受试者：预处理 + VAE 编码 → 保存 .npy"""
    s_id = sub_data["id"]
    subject_out_dir = os.path.join(output_latents_dir, s_id)
    os.makedirs(subject_out_dir, exist_ok=True)

    latent_info = {}
    for modality_name in TRAIN_MODALITIES:
        file_name = sub_data["modality_map"][modality_name]
        file_path = os.path.join(sub_data["path"], file_name)

        img, aff = utils.load_nifti(file_path)
        img, _ = utils.preprocessing(img, affine=aff)

        latent = pipeline.encode_image(img, vae)
        latent = np.expand_dims(latent, 0)     # 恢复 batch 维度 (1,4,64,64,40)

        base = os.path.basename(file_name).split(".")[0]
        npy_name = f"{base}_latent.npy"
        npy_path = os.path.join(subject_out_dir, npy_name)
        np.save(npy_path, latent)

        latent_info[modality_name] = file_name     # CSV 里存原始文件名

    return latent_info


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 扫描输入目录
    print(f"Scanning subjects in: {configs.PATH_INPUT}")
    subjects, skipped_subjects = scan_subjects(configs.PATH_INPUT)
    print(f"Found {len(subjects)} subjects with complete training modalities.\n")

    if not subjects:
        print("No subjects found. Put NIfTI files in data/input/<subject_id>/")
        sys.exit(1)

    # 加载 VAE
    print("Loading VAE...")
    vae = pipeline.instantiate_vae_model(device)
    print("Done.\n")

    # 预处理 + 编码
    output_latents_dir = os.path.join(configs.PATH_DATA, "latents")
    os.makedirs(output_latents_dir, exist_ok=True)

    csv_rows = []
    for sub in tqdm(subjects, desc="Preprocessing subjects"):
        try:
            latent_info = preprocess_subject(sub, vae, output_latents_dir, device)
            row = {"id": sub["id"]}
            for m in configs.MODALITY_LIST:
                row[m] = latent_info.get(m, "")
            csv_rows.append(row)
        except Exception as e:
            print(f"Error processing {sub['id']}: {e}")

    # 生成 CSV（split 全设为 train，后续手动调整）
    df = pd.DataFrame(csv_rows, columns=["id"] + configs.MODALITY_LIST)
    df["split"] = "train"

    csv_path = os.path.join(configs.PATH_DATA, "data_csv.csv")
    df.to_csv(csv_path, index=False)

    if skipped_subjects:
        skipped_csv_path = os.path.join(configs.PATH_DATA, SKIPPED_CSV_NAME)
        pd.DataFrame(skipped_subjects, columns=["id", "reason", "present_modalities", "files"]).to_csv(skipped_csv_path, index=False)
        print(f"Skipped subject manifest saved to: {skipped_csv_path}")

    print(f"\nCSV saved to: {csv_path}")
    print(f"Latent .npy files saved to: {output_latents_dir}/")
    print(f"Total subjects processed: {len(csv_rows)}")


if __name__ == "__main__":
    main()
