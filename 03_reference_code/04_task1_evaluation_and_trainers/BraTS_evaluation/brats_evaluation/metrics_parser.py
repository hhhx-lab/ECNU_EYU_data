import json
import pandas as pd
import numpy as np
import os
import argparse


def _calculate_mean_std_median(df):
    """Calculates mean, standard deviation, and median for numeric columns and appends them."""
    mean_row = df.mean(numeric_only=True).to_dict()
    std_row = df.std(numeric_only=True).to_dict()
    median_row = df.median(numeric_only=True).to_dict()

    mean_row['subject_id'] = 'mean'
    std_row['subject_id'] = 'std'
    median_row['subject_id'] = 'median'

    return pd.concat([df, pd.DataFrame([mean_row, std_row, median_row])], ignore_index=True)


def _handle_missing_data(data, final_data_rows):
    """Fills in default values for missing subjects."""
    missing_data = data.get("missings", [])
    if missing_data:
        keys_to_fill = []
        if final_data_rows:
            keys_to_fill = [k for k in final_data_rows[0].keys() if k != "subject_id"]

        for missing_subject in missing_data:
            missing_row = {"subject_id": missing_subject}
            for key in keys_to_fill:
                if "hd95" in key:
                    missing_row[key] = 373 # diameter of the cube in SRI space; using the maximum penalty instead of INF.
                else:
                    missing_row[key] = 0
            final_data_rows.append(missing_row)
    return final_data_rows


def parse_seg_results(json_path, output_csv_path):
    """
    Parses the panoptica JSON output to report the metrics required for the challenge.
    """
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    print("JSON file loaded successfully.")
    metrics_data = data.get("metrics", [])
    final_data_rows = []

    for subject_data in metrics_data:
        if "error" in subject_data:
            print(f"Skipping subject {subject_data.get('subject_name')} due to error: {subject_data.get('error')}")
            continue

        subject_id = subject_data.get("subject_name")
        subject_row_data = {"subject_id": subject_id}

        for key, region_data in subject_data.items():
            if key == "subject_name" or not isinstance(region_data, dict):
                continue

            # Extract overall instance metrics first
            subject_row_data[f"all_instance_tp_{key}"] = region_data.get("tp", np.nan)
            subject_row_data[f"all_instance_fp_{key}"] = region_data.get("fp", np.nan)
            subject_row_data[f"all_instance_fn_{key}"] = region_data.get("fn", np.nan)
            subject_row_data[f"all_instance_f1_{key}"] = region_data.get("rq", np.nan)

            # Global segmentation metrics
            subject_row_data[f"global_dsc_{key}"] = region_data.get("global_bin_dsc", np.nan)
            subject_row_data[f"global_nsd_{key}"] = region_data.get("global_bin_nsd", np.nan)
            global_hd95 = region_data.get("global_bin_hd95")
            if global_hd95 is not None and np.isinf(global_hd95):
                global_hd95 = 373 # diameter of the cube in SRI space; using the maximum penalty instead of INF.
            subject_row_data[f"global_hd95_{key}"] = global_hd95

        final_data_rows.append(subject_row_data)

    final_data_rows = _handle_missing_data(data, final_data_rows)

    if final_data_rows:
        df = pd.DataFrame(final_data_rows)
        df = _calculate_mean_std_median(df)
        df.to_csv(output_csv_path, index=False)
        print(f"DataFrame saved successfully to {output_csv_path}")
        print(df.tail())
    else:
        print("No data was processed to create a DataFrame.")


def parse_mets_results(json_path, vol_threshold, overlap_threshold, output_csv_path):
    """
    Parses the panoptica JSON output to calculate custom statistics per region.
    """
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    print("JSON file loaded successfully.")
    metrics_data = data.get("metrics", [])
    final_data_rows = []

    for subject_data in metrics_data:
        if "error" in subject_data:
            print(f"Skipping subject {subject_data.get('subject_name')} due to error: {subject_data.get('error')}")
            continue

        subject_id = subject_data.get("subject_name")
        subject_row_data = {"subject_id": subject_id}

        for key, region_data in subject_data.items():
            if key == "subject_name" or not isinstance(region_data, dict):
                continue

            large_lesion_dsc, large_lesion_hd95, large_lesion_nsd = [], [], []
            detection_tp, detection_fn = 0, 0
            large_lesion_detection_tp, large_lesion_detection_fn = 0, 0
            small_lesions_found, large_lesions_present = False, False

            metric_average = region_data
            lesion_instances = region_data.get("reference_instances", [])
            n_ref_instances = metric_average.get("n_ref_instances", 0)

            subject_row_data[f"all_instance_tp_{key}"] = metric_average.get("tp", np.nan)
            subject_row_data[f"all_instance_fp_{key}"] = metric_average.get("fp", np.nan)
            subject_row_data[f"all_instance_fn_{key}"] = metric_average.get("fn", np.nan)
            subject_row_data[f"all_instance_f1_{key}"] = metric_average.get("rq", np.nan)

            for lesion_data in lesion_instances:
                instance_volume = lesion_data.get("volume")
                if instance_volume is None:
                    continue

                is_large_lesion = instance_volume >= vol_threshold
                
                if is_large_lesion:
                    large_lesions_present = True
                else:
                    small_lesions_found = True

                if lesion_data.get("is_matched") == 1:
                    sq_dsc = lesion_data.get("sq_dsc")
                    if is_large_lesion:
                        large_lesion_dsc.append(sq_dsc)
                        large_lesion_hd95.append(lesion_data.get("sq_hd95"))
                        large_lesion_nsd.append(lesion_data.get("sq_nsd"))
                        if sq_dsc is not None and sq_dsc >= overlap_threshold:
                            large_lesion_detection_tp += 1
                        else:
                            large_lesion_detection_fn += 1
                    else: # Small lesion
                        if sq_dsc is not None:
                            if sq_dsc >= overlap_threshold:
                                detection_tp += 1
                            else:
                                detection_fn += 1
                else: # Unmatched
                    if is_large_lesion:
                        large_lesion_detection_fn += 1
                        large_lesion_dsc.append(0)
                        large_lesion_nsd.append(0)
                        large_lesion_hd95.append(373) # 373 diameter of the cube in SRI space; using the maximum penalty instead of INF.
                    else: # Small lesion
                        detection_fn += 1
            
            num_fp = metric_average.get("fp", 0)
            if num_fp > 0:
                large_lesion_dsc.extend([0] * num_fp)
                large_lesion_nsd.extend([0] * num_fp)
                large_lesion_hd95.extend([373] * num_fp) # 373 diameter of the cube in SRI space; using the maximum penalty instead of INF.

            large_lesion_fp = num_fp
            large_denominator = (2 * large_lesion_detection_tp) + large_lesion_fp + large_lesion_detection_fn
            large_f1 = (2 * large_lesion_detection_tp) / large_denominator if large_denominator > 0 else 0
            subject_row_data[f"large_instance_tp_{key}"] = large_lesion_detection_tp
            subject_row_data[f"large_instance_fp_{key}"] = large_lesion_fp
            subject_row_data[f"large_instance_fn_{key}"] = large_lesion_detection_fn
            subject_row_data[f"large_instance_f1_{key}"] = large_f1

            if n_ref_instances == 0 or not large_lesions_present:
                subject_row_data[f"lesionwise_dsc_mean_{key}"] = np.nan
                subject_row_data[f"lesionwise_dsc_std_{key}"] = np.nan
                subject_row_data[f"lesionwise_hd95_mean_{key}"] = np.nan
                subject_row_data[f"lesionwise_hd95_std_{key}"] = np.nan
                subject_row_data[f"lesionwise_nsd_mean_{key}"] = np.nan
                subject_row_data[f"lesionwise_nsd_std_{key}"] = np.nan
            else:
                valid_hd95 = [v for v in large_lesion_hd95 if v is not None and not np.isinf(v)]
                subject_row_data[f"lesionwise_dsc_mean_{key}"] = np.mean(large_lesion_dsc) if large_lesion_dsc else 0
                subject_row_data[f"lesionwise_dsc_std_{key}"] = np.std(large_lesion_dsc) if large_lesion_dsc else 0
                subject_row_data[f"lesionwise_hd95_mean_{key}"] = np.mean(valid_hd95) if valid_hd95 else 373 # diameter of the cube in SRI space; using the maximum penalty instead of INF.
                subject_row_data[f"lesionwise_hd95_std_{key}"] = np.std(valid_hd95) if valid_hd95 else 0
                subject_row_data[f"lesionwise_nsd_mean_{key}"] = np.mean(large_lesion_nsd) if large_lesion_nsd else 0
                subject_row_data[f"lesionwise_nsd_std_{key}"] = np.std(large_lesion_nsd) if large_lesion_nsd else 0

            if small_lesions_found:
                subject_row_data[f"small_instance_tp_{key}"] = detection_tp
                subject_row_data[f"small_instance_fn_{key}"] = detection_fn
                subject_row_data[f"small_instance_fp_{key}"] = num_fp
                denominator = (2 * detection_tp) + num_fp + detection_fn
                subject_row_data[f"small_instance_f1_{key}"] = (2 * detection_tp) / denominator if denominator > 0 else 0
            else:
                subject_row_data[f"small_instance_tp_{key}"] = np.nan
                subject_row_data[f"small_instance_fn_{key}"] = np.nan
                subject_row_data[f"small_instance_fp_{key}"] = np.nan
                subject_row_data[f"small_instance_f1_{key}"] = np.nan

        final_data_rows.append(subject_row_data)

    final_data_rows = _handle_missing_data(data, final_data_rows)

    if final_data_rows:
        df = pd.DataFrame(final_data_rows)
        df = _calculate_mean_std_median(df)
        df.to_csv(output_csv_path, index=False)
        print(f"DataFrame saved successfully to {output_csv_path}")
        print(df.tail())
    else:
        print("No data was processed to create a DataFrame.")


def main():
    parser = argparse.ArgumentParser(
        description="Parse Panoptica JSON results and calculate custom statistics."
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    parser_seg = subparsers.add_parser("seg", help="Parse basic segmentation metrics.")
    parser_seg.add_argument("--json_path", type=str, required=True, help="Path to the Panoptica JSON file.")
    parser_seg.add_argument("--output_csv_path", type=str, default="./parsed_panoptica_seg_stats.csv", help="Path to save the resulting CSV file.")

    parser_mets = subparsers.add_parser("mets", help="Parse METs-specific metrics (includes size thresholds).")
    parser_mets.add_argument("--json_path", type=str, required=True, help="Path to the Panoptica JSON file.")
    parser_mets.add_argument("--vol_threshold", type=float, default=20.0, help="Volume threshold to differentiate large/small lesions.")
    parser_mets.add_argument("--overlap_threshold", type=float, default=0.1, help="Dice score threshold to classify small lesions as TP/FN.")
    parser_mets.add_argument("--output_csv_path", type=str, default="./parsed_panoptica_mets_stats.csv", help="Path to save the resulting CSV file.")

    args = parser.parse_args()

    if args.command == "seg":
        parse_seg_results(
            json_path=args.json_path,
            output_csv_path=args.output_csv_path
        )
    elif args.command == "mets":
        parse_mets_results(
            json_path=args.json_path,
            vol_threshold=args.vol_threshold,
            overlap_threshold=args.overlap_threshold,
            output_csv_path=args.output_csv_path
        )

if __name__ == "__main__":
    main()
