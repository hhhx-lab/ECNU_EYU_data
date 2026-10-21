# K4 175 R3 提交层阻塞

- 本地记录时间：2026-09-25 03:11 CST；远端诊断时间：2026-09-24T19:00:09Z，`login1`。
- 旧 job `3741523` 为 `FAILED 1:0`，旧根与失败证据未改写。
- 新独立根：`/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1245_20260925_r3_175`。
- 新计划 SHA256：`6eda2dd148f18ff289a48d3fd66f283cfdbaf4f210a0c673dd81898b2424c963`；Slurm SHA256：`5e25a3cc810a8ae0b1a954e1c9abc0ce0959f630a42bceca8b5c7150b0d48cb4`。
- 校准文件存在，SHA256 为 `68a5d8a6c501804b7a47d626e43a58ff71b781154c0cc955a9c5a5d5b786d37f`；新 Slurm 显式导出 `S2_MET_AUG_FIX_V3_CALIBRATION`，STARTED 前再次核验其 SHA 与配置。修正了旧 `FOLD` 路径未包含 trainer 后缀 `175` 的后续验收风险。
- 远端 `K4_175_R3_STATIC_PREFLIGHT.json` 和 `K4_175_R3_RESERVATION.json` 已生成；`sbatch --hold` 返回 1。诊断 `sbatch --test-only` 返回 `QOSMaxSubmitJobPerUserLimit`。无 job ID、无 launch/released、无训练开始。
- 本地失败审计：`../../s2_met_aug_k1245_20260923_r3_fallback/operator/submit_k4_175_r3_20260924T190847Z/FAILED.json`；本轮只读诊断不修改远端作业。
- R3 attempt 不再重提或覆盖。待 QOS 限额释放后，应建立新独立 attempt、重新核验 SHA/合同，再唯一 held 提交；不得将 R3 静态预检等同于训练提交成功。
- `artifact_status=experimental_unvalidated`，`operator_approved=false`，`formal_gate_status=not_run_not_passed`，`F_STRICT_PREREG_GATE=false`。
