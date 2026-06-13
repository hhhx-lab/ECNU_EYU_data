# Dataset260_BraTS2026_MET_RealOnly

本目录当前只保存 `dataset.json` 和映射说明，不复制或软链接全量 NIfTI。需要正式训练时，由 S1/S2 根据 `manifests/nnunet_case_mapping_realonly.csv` 在训练机器上物化数据集并运行 nnU-Net 预处理。synthetic accepted 结果会另起 dataset id，不混进这个 real-only 占位目录。
