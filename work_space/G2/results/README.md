# G2 Results

本目录保存 G2 在模型训练完成前已经能在本机完成的小型结果文件。这里承接真实数据清单、G1 兼容 source 表、synthetic intake 模板、QC 策略、官方指标对齐模板和进度索引；NIfTI 大数据、nnU-Net 大量预处理缓存、正式 synthetic 影像和临时 smoke run 产物都不复制到仓库。

当前 G2 的正式执行入口在 `../code/`：

1. `g2_create_train_val_test_split.py`：生成固定 train/val/test split，默认口径为 829/207/259。
2. `g2_synthetic_raw_intake_qc.py`：接收 G1 raw output，生成 QC、accepted/rejected 和质量报告。
3. `g2_materialize_nnunet_dataset.py`：把 accepted 数据和真实数据映射成 nnU-Net raw dataset 入口。
4. `g2_official_mets_metrics_parser.py`：解析或校验 2026 Task1 官方 leaderboard 字段。
