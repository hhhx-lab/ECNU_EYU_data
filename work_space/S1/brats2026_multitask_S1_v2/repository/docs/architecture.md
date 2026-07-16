# Architecture

## Overview

Shared 3D UNet backbone with two lightweight 1×1×1 heads:

1. **Tumor head** — 4 classes (BG / NETC / SNFH / ET)
2. **RC head** — 2 classes (BG / RC)

```text
MRI (t1n,t1c,t2w,t2f)
        |
 nonzero Z-score (per modality, brain mask)
        |
   MONAI 3D UNet backbone
        |
   feature volume (C=64)
     /              \
 tumor_head        rc_head
 (4-class)         (2-class)
```

## Training path

```text
full case
  -> lesion-balanced 96^3 crop (≈80% CC lesion / 20% random brain)
  -> light aug (flip / small affine / gamma / noise / bias)
  -> multitask DiceCE + uncertainty weighting (clamped)
  -> micro-batch 1 + grad accumulation 2 + AMP
```

## Validation / selection path

```text
full case (no center crop)
  -> same Z-score
  -> joint sliding-window inference (tumor+RC logits concatenated)
  -> compose labels (RC=4)
  -> BraTS-compatible region / lesion / small-lesion proxies
  -> checkpoint_score -> best.pth
```

Center-crop validation is intentionally **not** used: metastases are often off-center, so a central 96³ patch can miss all lesions and produce an untrustworthy val_loss.

## Inference path

One backbone forward per window for both heads (see `inference/sliding_window_multitask.py`), then optional fused BraTS label map.

## Memory notes

~19.4M parameters is modest; OOM is dominated by 96³ feature maps, skip connections, and reverse-mode grads. Prefer:

1. batch_size=1 + accumulation
2. BF16/FP16 AMP
3. ≥40GB GPU
4. 80³ patch only if still OOM
5. width reduction last
