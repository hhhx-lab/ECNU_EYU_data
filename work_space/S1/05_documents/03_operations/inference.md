# Inference

Joint sliding-window inference (single backbone pass for tumor + RC heads).

ROI Size (default):

96 x 96 x 96

Preprocessing:

- Same channel order as training: `t1n, t1c, t2w, t2f`
- Per-modality nonzero-brain Z-score normalization (required for consistent metrics)

Outputs:

```text
tumor_pred.nii.gz
rc_pred.nii.gz
# optional with --save_fused
<case_id>.nii.gz   # fused labels: tumor 0/1/2/3, RC overwritten as 4
```

Example:

```bash
python inference/infer_multitask.py \
    --checkpoint /path/to/best.pth \
    --case_dir /path/to/BraTS-MET-XXXXX-000 \
    --output_dir /path/to/output \
    --config configs/multitask_v1_full.yaml \
    --save_fused
```

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--roi_size 96 96 96` | Override SWI window |
| `--overlap 0.5` | Window overlap |
| `--amp_dtype auto` | `auto` / `bf16` / `fp16` / `none` |
| `--save_fused` | Write BraTS-style multi-class seg |
| `--no_normalize` | Disable Z-score (not recommended) |

Why joint SWI:

Previously tumor and RC each ran a separate `sliding_window_inference`, repeating the backbone. The current helper concatenates both head logits in one predictor and splits them after SWI, roughly cutting inference compute in half.
