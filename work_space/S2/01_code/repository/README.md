# S2 代码

`custom_nnunet` 保持原模块名称和结构，供nnUNet动态加载和checkpoint兼容使用；本次未修改其30个源码文件。
`train.sh`、`infer.sh`、`inference_frozen.py` 保留为真实入口。空的train.py/inference.py已归档。
固定划分与审计已移至 [数据合同](../../02_datasets/undated_fixed_split_contracts/README.md)，BRATS_SPLIT_DIR显式设置优先。

- [custom_nnunet](custom_nnunet/)
- [scripts](scripts/)
- [tests](tests/)

历史源码原件、备份及占位文件保留在项目 `99_archive/s1_s2_legacy_20260919`。
冻结实验runtime和已提交作业证据未改写；重新运行须重新核对数据路径、环境和实验计划。

整理时间：2026-09-19。本地整理标记：experimental_unvalidated；operator_approved=false；formal_gate_status=not_run_not_passed。
