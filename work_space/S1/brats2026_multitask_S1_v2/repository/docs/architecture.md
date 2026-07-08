# Architecture

Input

4-channel MRI

- T1N
- T1C
- T2W
- T2F

Backbone

3D MONAI UNet

Shared Encoder

Shared Decoder

Heads

Tumor Head

4 classes

RC Head

2 classes

Loss

DiceCE

+

Uncertainty Weighting

(V5)
