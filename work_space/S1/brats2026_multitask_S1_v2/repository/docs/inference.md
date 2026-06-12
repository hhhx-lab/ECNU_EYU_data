# Inference

Sliding Window Inference

ROI Size:

96 x 96 x 96

Outputs:

tumor_pred.nii.gz

rc_pred.nii.gz

Example:

python inference/infer_multitask.py \
    --checkpoint best.pth \
    --case_dir CASE_DIR \
    --output_dir OUTPUT_DIR

