# G2 Slurm 入口

本目录保存完整流程中需要集群资源的 G2 作业包装。已经执行完成的 Stage 5 作业仍保留，便于复现审批链和重新核验历史 run。

| 脚本 | 对应代码 | 用途 |
|---|---|---|
| `01_g2_v3_paired_quality.slurm` | `../code/g2_v3_paired_quality.py` | fixed 103 missing-T2W 配对影像/病灶/空间 QC |
| `02_g2_v3_s2_teacher.slurm` | `../code/g2_s2_v3_teacher_eval.py` | fixed 103 frozen-S2 teacher 成对验收 |

当前 Diffusion augmentation checkpoint QC 尚未复用这两个作业。新作业必须绑定 checkpoint selection、固定 103 例和新的 run ID，不能覆盖历史 Stage 5 输出。

提交前检查：

```bash
bash -n work_space/G2/slurm/01_g2_v3_paired_quality.slurm
bash -n work_space/G2/slurm/02_g2_v3_s2_teacher.slurm
```
