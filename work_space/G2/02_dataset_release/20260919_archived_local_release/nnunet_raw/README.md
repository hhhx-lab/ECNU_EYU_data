# nnunet_raw

这里是 nnU-Net 原始数据的轻量入口。当前仓库只放占位说明、dataset.json 和路径契约，不放正式大体积影像。`Dataset260_BraTS2026_MET_RealOnly/` 记录 real-only 基线；synthetic intake 通过 G2 产出 accepted/rejected、QC 和 batch summary 后，再决定是否另起新的 dataset id。

本目录的核心职责是保存轻量契约。正式物化统一由 `work_space/G2/code/g2_materialize_nnunet_dataset.py` 完成：train/val 写入 `imagesTr/labelsTr`，locked test 写入 `imagesTs/labelsTs`，并同时生成按 `train/val/test` 分区的 case-folder view。
