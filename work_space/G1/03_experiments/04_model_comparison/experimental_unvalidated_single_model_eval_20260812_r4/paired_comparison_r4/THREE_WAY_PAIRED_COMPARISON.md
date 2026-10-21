# G1 r4 three-way paired comparison

Status: `experimental_unvalidated`; `operator_approved=false`.

All three methods use the same 103 cases, frozen checkpoints, and spatial preprocessing contract.

| Metric | Direction | 1st | 2nd | 3rd |
|---|---|---|---|---|
| whole_SSIM | higher | bbdm_only | ensemble_r4 | encdec_only |
| whole_PSNR | higher | ensemble_r4 | encdec_only | bbdm_only |
| whole_MSE | lower | ensemble_r4 | encdec_only | bbdm_only |
| whole_MAE | lower | ensemble_r4 | bbdm_only | encdec_only |
| brain_SSIM | higher | ensemble_r4 | bbdm_only | encdec_only |
| brain_PSNR | higher | ensemble_r4 | encdec_only | bbdm_only |
| brain_MSE | lower | ensemble_r4 | encdec_only | bbdm_only |
| brain_MAE | lower | ensemble_r4 | bbdm_only | encdec_only |

Unanimous image-quality winner: `none`.

This result compares missing-T2W reconstruction quality only. It does not show which method is best for downstream nnU-Net segmentation.
