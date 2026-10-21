# G2 测试索引

测试文件与 `../code/` 的完整流程入口一一对应。已完成阶段的测试继续保留，防止后续整理或适配新服务器时破坏历史契约。

| 测试 | 覆盖入口 |
|---|---|
| `test_g2_build_realonly_from_raw.py` | master/real-only/corrected-label intake |
| `test_g2_create_train_val_test_split.py` | patient-group fixed split |
| `test_g2_synthetic_intake.py` | pretraining audit 与通用 synthetic intake |
| `test_g2_v2_compose_augmentation.py` | Diffusion ROI composition |
| `test_g2_v3_completion_intake.py` | 265 例 completion intake |
| `test_g2_v3_completion_visual_review.py` | completion montage 与复核选择 |
| `test_g2_v3_paired_quality.py` | fixed 103 paired image/lesion/spatial QC |
| `test_g2_s2_v3_teacher_eval.py` | frozen-S2 teacher gate |
| `test_g2_freeze_diffusion_full_eval.py` | fixed 94+9 cohort、源哈希与 validation strict no-op |
| `test_g2_diffusion_checkpoint_qc.py` | 四模态 Diffusion checkpoint 技术 gate、support 契约与 montage |
| `test_g2_prepare_diffusion_manual_review.py` | RC/tiny/large-tiled/低分/伪影/smoke-risk 必审并集 |
| `test_g2_finalize_diffusion_gate.py` | 必审分层、风险接受和最终 selection/gate 冻结契约 |
| `test_g2_materialize_nnunet_dataset.py` | nnU-Net/case-folder 物化与 split 隔离 |

完整运行：

```bash
conda run -n g1_t2w_bbdm python -m unittest discover -s work_space/G2/tests -v
```
