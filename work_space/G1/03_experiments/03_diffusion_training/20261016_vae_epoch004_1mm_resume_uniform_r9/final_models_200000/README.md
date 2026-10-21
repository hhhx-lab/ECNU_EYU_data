# R9最终200000步完整checkpoint

2026-10-21选模与下载补充：本目录仍只代表四路最终200000训练状态，不改写原验收。固定val103评价已选出并下载[四最佳模型](../../20261018_r9_fixedval_selection_r1/best_models_fixedval103/README.md)：T1C120000、T1N40000、T2W200000、T2F200000。T2W/T2F与本目录同SHA，T1C/T1N不是本目录的200k。需要所选模型以新MODEL_MANIFEST为准；人工质量审批仍未通过。下方selection_status=not_run是原训练保存时字段，不代表后续评价未执行。

四路训练均COMPLETED0:0，COMPLETE/LATEST为200000成功更新。各模态子目录一份diffusion_200000.pt；总223224828bytes。远端/本地大小与SHA已核对，未反序列化，不标为最佳模型或生成质量pass。

四SHA、配置/数据身份及容量见[训练完成报告](../TRAINING_COMPLETION_20261018.md)、[拉回回执](../operator/pull_final_20261018T065725Z/PULL_ACCEPTANCE.json)。配套合同/配置在[轻量结果](../operator/pull_final_20261018T065725Z/lightweight)，不要把远端manifest路径自动当成本地影像路径。

文件由冻结runtime/train.py保存，含state_dict、optimizer、scaler、RNG/MONAI、sampler/permutation/cursor、data_identity/training_config/schedule。没有另导出网络权重、下载中间checkpoint或MRI。加载需项目约定的兼容环境及显式信任，不临时混用系统/Homebrew Python安装依赖。

experimental_unvalidated / operator_approved=false / formal_gate_status=not_run_not_passed；selection_status=not_run。
