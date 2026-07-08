# Manifests

保存真实训练/验证清单、corrected overlay、G1 兼容 source CSV、synthetic intake 模板，以及正式 G1 批次到来后生成的 accepted/rejected 索引文件。旧 smoke run 演示输出已清理。

real-only baseline 训练前由 `../../code/g2_build_realonly_from_raw.py` 从 raw data 生成：

1. `nnunet_case_mapping_realonly.csv`
2. `realonly_skipped_incomplete_cases.csv`

其中 `realonly_skipped_incomplete_cases.csv` 记录缺 `t1n/t1c/t2w/t2f/seg` 的病例；这些病例不会进入 S1/S2/S3 real-only 训练。

正式 G1 批次到来后由 `../../code/g2_synthetic_raw_intake_qc.py` 自动生成：

1. `synthetic_generation_manifest_{run_id}.csv`
2. `synthetic_candidate_manifest_{run_id}.csv`
3. `synthetic_accepted_manifest_{run_id}.csv`
4. `synthetic_rejected_manifest_{run_id}.csv`
5. `synthetic_normalized_mapping_{run_id}.csv`
