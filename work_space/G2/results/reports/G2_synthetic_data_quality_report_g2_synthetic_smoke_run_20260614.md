# G2 Synthetic Data Quality Report

生成日期：2026-06-14
run_id：`g2_synthetic_smoke_run_20260614`

## 1. 本轮概况

- 候选数：3
- accepted：2
- ablation only：1
- needs regeneration：1
- rejected：1

## 2. 生成与接收

- `generation_config.json`：存在
- `generation_log.jsonl`：存在
- `synthetic_generation_manifest.csv`：缺失，已由 G2 补建

## 3. accepted / rejected 结果

### accepted

| synthetic_raw_id | synthetic_final_id | source_case_id | qc_decision |
| --- | --- | --- | --- |
| BraTS-MET-00009-000_fake_label_0 | SYN-MET-000002 | BraTS-MET-00009-000 | accepted_for_training |
| BraTS-MET-00013-000_real_label_0 | SYN-MET-000003 | BraTS-MET-00013-000 | accepted_for_ablation_only |

### rejected

| synthetic_raw_id | synthetic_final_id | source_case_id | qc_reject_reason |
| --- | --- | --- | --- |
| BraTS-MET-00007-000_fake_label_1 | SYN-MET-000001 | BraTS-MET-00007-000 | mixed_suffix_scheme |

## 4. 主要问题

| synthetic_raw_id | qc_status | qc_reject_reason | manual_review_reason |
| --- | --- | --- | --- |
| BraTS-MET-00007-000_fake_label_1 | reject | mixed_suffix_scheme |  |
| BraTS-MET-00009-000_fake_label_0 | pass |  |  |
| BraTS-MET-00013-000_real_label_0 | review | legacy_suffix_normalized | legacy_suffix_normalized |

## 5. 输出文件

- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/synthetic_generation_manifest_g2_synthetic_smoke_run_20260614.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/synthetic_candidate_manifest_g2_synthetic_smoke_run_20260614.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/synthetic_accepted_manifest_g2_synthetic_smoke_run_20260614.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/synthetic_rejected_manifest_g2_synthetic_smoke_run_20260614.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/synthetic_normalized_mapping_g2_synthetic_smoke_run_20260614.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/qc_metrics_g2_synthetic_smoke_run_20260614.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/diffusion_quality_metrics_g2_synthetic_smoke_run_20260614.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/qc_case_review_g2_synthetic_smoke_run_20260614.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/qc_batch_summary_g2_synthetic_smoke_run_20260614.json`

## 6. 结论

1. G2 已经可以从 G1 legacy raw output 里自动恢复 source、label_kind、run 信息、suffix scheme，并补建 synthetic manifest。
2. G2 会额外生成 `synthetic_normalized_mapping_{run_id}.csv`，逐模态记录 raw legacy/native 文件到 2026 标准文件名和 nnU-Net 目标文件名的映射。
3. 通过的样本会进入 accepted manifest，未通过的样本会进入 rejected manifest，人工复查项会单独落表。
4. 真实验证 fold 和官方 validation 仍然不能作为 synthetic source。
