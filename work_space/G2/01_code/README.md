# G2 代码索引

本目录保留 BraTS 2026 Task 1 从数据准备、missing-T2W 验收到 Diffusion augmentation、下游发布和官方指标解析所需的完整代码。阶段已经执行完不代表代码可以删除；只要它属于完整可复现流程，就继续保留。

## 完整执行顺序

| 阶段 | 入口 | 责任 |
|---|---|---|
| 1. 主数据准备 | `g2_build_realonly_from_raw.py` | 从 raw/corrected labels 建立 master、real-only 和 Diffusion source mapping |
| 2. 固定切分 | `g2_create_train_val_test_split.py` | 生成 patient-group 隔离的 train/val/test split |
| 3. 训练前审计 | `g2_pretraining_audit.py` | 检查数据、标签、病灶、source/synthetic 契约并生成审计表 |
| 4. Missing-T2W Stage 5 QC | `g2_v3_paired_quality.py` | 对固定 103 例真实/生成 T2W 做影像、区域、病灶和空间配对 QC |
| 5. Missing-T2W teacher gate | `g2_s2_v3_teacher_eval.py` | 用冻结 S2 对同一 103 例做成对 teacher 验收 |
| 6. Missing-T2W completion intake | `g2_v3_completion_intake.py` | 接收阶段 6 的 265 例 completion，验证元数据和原病例身份 |
| 7. Completion 人工复核 | `g2_v3_completion_visual_review.py` | 为 flagged completion 生成三平面 montage 和复核索引 |
| 8. Diffusion full-eval cohort | `g2_freeze_diffusion_full_eval.py` | 从冻结 split 派生 94 个 lesion-positive 生成清单和 9 个 lesion-negative strict no-op 审计 |
| 9. Diffusion checkpoint gate | `g2_diffusion_checkpoint_qc.py` | 对固定 smoke/full split 的四模态 support 输出做几何、数值、checkpoint 哈希、边界/伪影/连续性和 montage 人工复核 |
| 10. Diffusion 人工复核队列 | `g2_prepare_diffusion_manual_review.py` | 生成 RC/tiny/large-tiled/低分/伪影/smoke-risk 并集及分批逐例模板 |
| 11. Diffusion gate 冻结 | `g2_finalize_diffusion_gate.py` | 强制核对 94+9、必审分层、风险接受、采样配置和三方 checkpoint 哈希后生成 selection/gate/SHA256 |
| 12. Diffusion augmentation composition | `g2_v2_compose_augmentation.py` | 将通过 checkpoint gate 的 G1 四模态 ROI 生成结果恢复为完整、可审计病例 |
| 13. 通用 synthetic intake/QC | `g2_synthetic_raw_intake_qc.py` | 生成 candidate、rejected、pending、accepted manifests 和质量报告 |
| 14. 下游数据发布 | `g2_materialize_nnunet_dataset.py` | 物化 nnU-Net 与 case-folder 双视图，并执行完整性检查 |
| 15. 官方指标解析 | `g2_official_mets_metrics_parser.py` | 解析或校验 BraTS MET 官方评估结果和 leaderboard 字段 |

## 两条数据线

```text
Missing-T2W:
master/split -> Stage 5 paired QC -> frozen-S2 teacher -> 265 completion intake
             -> visual review -> materialize -> S2 completion-only

Diffusion augmentation:
master/split/source audit -> freeze 94 positive + audit 9 strict no-op
                          -> four-modality generation -> checkpoint gate
                          -> composition
                          -> generic intake/QC -> materialize
                          -> S2 online augmentation -> official-compatible evaluation
```

`g2_v2_compose_augmentation.py` 中的 `V2` 是历史接口名。当前 G1 生产实现已经升级到 Diffusion V3，但输出仍遵循该 composition 契约；不要因为文件名包含 V2 而删除它。

## 保留规则

- 保留所有上表入口及其测试，即使对应阶段已经完成。
- 保留历史 gate 和 run 结果，用于复核 checkpoint、数据身份与审批过程。
- 只清理 `__pycache__`、`.pyc`、临时 NIfTI、调试输出和明确无调用关系的重复副本。
- 重构或替换入口前，必须先迁移测试、Slurm、README 和跨组引用，再删除旧文件。

A800 无 Slurm 的完整 94 例运行使用 `../cloud/watch_full94_then_qc.sh` 在生成进程
完成后串接 G2 QC。watcher 只在 `metrics.json`、`generation_manifest.csv` 完整且
日志无错误模式时启动 QC，不会覆盖已有输出。

## 验证

```bash
conda run -n g1_t2w_bbdm python -m unittest discover -s work_space/G2/tests -v
```
