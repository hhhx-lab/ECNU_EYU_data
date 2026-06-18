"""Compute per-channel importance weights from T2W latent RMS.

Usage:
    python compute_weights.py

Output:
    Per-channel RMS values and normalized weights ready for
    pasting into training_bbdm.py as channel_importance_weights.
"""
import os
import numpy as np
import pandas as pd
import configs


def main():
    csv_path = os.path.join(configs.PATH_DATA, "data_csv.csv")
    latents_dir = os.path.join(configs.PATH_DATA, "latents")
    missing_modality = configs.MISSING_MODALITY  # "t2w"

    df = pd.read_csv(csv_path)
    df_train = df[df["split"] == "train"]

    if len(df_train) == 0:
        print("ERROR: No training samples found. Check data_csv.csv split column.")
        return

    ch_vals = {0: [], 1: [], 2: [], 3: []}
    skipped = 0

    for _, row in df_train.iterrows():
        s_id = row["id"]
        raw_name = os.path.basename(row[missing_modality]).split(".")[0]
        latent_path = os.path.join(latents_dir, s_id, f"{raw_name}_latent.npy")

        if not os.path.exists(latent_path):
            print(f"  SKIP: {latent_path} not found")
            skipped += 1
            continue

        latent = np.load(latent_path).squeeze(0)  # (4, 64, 64, 40)
        for ch in range(4):
            ch_vals[ch].append(latent[ch].ravel())

    if skipped:
        print(f"\nWARNING: {skipped} subjects skipped (missing latents).")
    if len(ch_vals[0]) == 0:
        print("ERROR: No latents loaded. Run preprocess.py first.")
        return

    print(f"\nLoaded {missing_modality} latents from {len(ch_vals[0])} training subjects.\n")

    rms = []
    for ch in range(4):
        vals = np.concatenate(ch_vals[ch])
        r = np.sqrt(np.mean(vals ** 2))
        rms.append(r)
        print(f"  ch{ch}  RMS = {r:.6f}")

    weights = np.array(rms) / np.sum(rms)
    weights_list = [float(round(w, 6)) for w in weights]
    print(f"\nchannel_importance_weights = {weights_list}")
    print(f"sum = {sum(weights_list):.4f}")


if __name__ == "__main__":
    main()
