# Dataset260_BraTS2026_MET_RealOnly

本目录当前只保存 `dataset.json` 和映射说明，不复制或软链接全量 NIfTI。正式训练时由 G2 materializer 根据 real-only mapping 与 patient-group fixed split 生成数据：823 train、103 val、104 locked test。synthetic accepted 结果另起 dataset ID，不覆盖历史 Dataset260。
