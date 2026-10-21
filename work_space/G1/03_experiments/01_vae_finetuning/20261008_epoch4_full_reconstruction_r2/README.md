# G1 epoch004全量重建独立恢复r2

本批次仅修复r1全量作业`3940993`在Python启动前的Slurm shell错误：`set -u`下空`CASES[@]`展开失败。r1双病例smoke `3940942`已通过，r1失败作业、日志和审计保持原样，不在原根重投。

r2使用同一已冻结epoch004权重、1030病例CSV、四模态重建runtime及原样seg合同，只另建全量专用Slurm脚本。新脚本不使用可为空的参数数组，并提供不运行模型的`RECON_PREFLIGHT_ONLY=1`分支，以便在远端`bash -u`下实际检查启动路径。全量目标仍为train823/val103/test104，4120幅影像及1030份逐字节相同的seg。

本地来源与失败快照的SHA、远端逐文件SHA、`bash -n`、`bash -u` dry-run、Python真实import和checkpoint严格加载均须在唯一held提交前通过。提交和结果证据记录在本批次`operator/`与远端`evidence/`；任何新失败保留原件，不原位重跑。

提交状态以本批次`operator/`中的回执和Slurm记录为准，不能仅凭本文生成时的状态推定作业终态。目标Drive文件夹ID为`1JICH__ZmoCrF856Lvh65QLgSMV9_da8s`；只有全量完成、逐文件验收并拉回后才上传。`artifact_status=experimental_unvalidated`，`operator_approved=false`，`formal_gate_status=not_run_not_passed`；用于四模态Diffusion的探索输入，不代表质量优越。
