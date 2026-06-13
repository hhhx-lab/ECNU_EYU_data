# nnunet_raw

这里是 nnU-Net 原始数据的轻量入口。当前仓库只放占位说明、dataset.json 和路径契约，不放正式大体积影像。`Dataset260_BraTS2026_MET_RealOnly/` 记录 real-only 基线；synthetic intake 通过 G2 产出 accepted/rejected、QC 和 batch summary 后，再决定是否另起新的 dataset id。

本目录的核心职责是把 `manifests/nnunet_case_mapping_realonly.csv` 和未来 synthetic accepted 映射变成训练机上的物化动作说明，而不是在仓库里复制影像本体。
