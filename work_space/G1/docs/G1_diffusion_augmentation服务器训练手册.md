# G1 Diffusion augmentation V3 服务器训练手册

更新日期：2026-07-15

## 1. 任务边界

本线输入真实病例的 `seg` 和四个完整模态，学习从分割条件生成四模态病灶影像，用于后续 synthetic augmentation。

本线不是缺失 T2W 填补算法。缺失 T2W V3 保留原病例 ID，只修复 T2W；本线会在 G2 composer 后形成新的 synthetic 病例。

正式代码唯一入口：

```text
work_space/G1/code/BraTS_2023_2024_solutions-main 3/Segmentation_Tasks/GliGAN
```

模型参数细节见该目录 `README_DIFFUSION.md`，Slurm 参数和顺序见 `slurm/README.md`。

## 2. 数据口径

唯一 source manifest：

```text
work_space/G2/results/manifests/g1_v2_source_manifest.csv
```

正式准备脚本固定得到：

| split | authentic 病例 | 行为 |
|---|---:|---|
| train | 823 | 参与训练 |
| val | 103 | 只参与验证 |
| test | 104 | locked，不建立训练视图 |

265 个 fake/broken T2W 病例不进入本线。患者组 `BraTS-MET-xxxxx` 不得跨 split。两份 corrected seg 必须按 manifest 覆盖。

ECNU 原始数据可以保持扁平布局：

```text
/public/home/${USER}/data/<case_id>/
  t1n.nii.gz
  t1c.nii.gz
  t2w.nii.gz
  t2f.nii.gz
  seg.nii.gz
```

`prepare_dataset_from_g2_manifest.py` 会创建标准命名软链接，无需复制原始数据。

## 3. 环境

当前复用已经验证的独立 Conda 环境：

```text
/public/home/${USER}/.conda/envs/segmamba/bin/python
```

版本和限制见：

```text
work_space/G1/docs/G1_Diffusion_V3环境清单.txt
```

禁止 `sudo pip`、禁止在作业中安装包、禁止混用系统 Python。ECNU GCC 4.8 不支持当前 Triton 编译，正式任务保持 `USE_COMPILE=0`。

## 4. 服务器目录

正式隔离根目录：

```bash
export PROD_ROOT=/public/home/${USER}/g1_diffusion_v3_production_20260715
```

结构：

```text
${PROD_ROOT}/
  code/                 # 已验证代码快照
  manifests/            # G2 source manifest
  corrected_labels/     # corrected seg
  DataSet/              # 926 病例软链接视图
  splits/current/       # membership 和 lesions.csv
  checkpoints/          # 四模态 checkpoint
  eval/                 # 103 val 指标
  logs/                 # Slurm 日志
```

大数据、checkpoint、eval 输出不提交 Git。

## 5. 一次性提交生产链

smoke、正式尺寸预检和 CPU 数据准备使用不同资源，可以并行；四模态训练必须等待三者全部成功：

```bash
export PROD_ROOT=/public/home/${USER}/g1_diffusion_v3_production_20260715
export SMOKE_ROOT=/public/home/${USER}/g1_diffusion_v3_smoke_20260715
mkdir -p "${PROD_ROOT}/logs" "${SMOKE_ROOT}/logs"
cd "${PROD_ROOT}/code/Segmentation_Tasks/GliGAN"

SMOKE_JOB=$(sbatch --parsable -p a100 slurm/00_smoke_v3_ecnu.slurm)
PREFLIGHT_JOB=$(sbatch --parsable slurm/00_preflight_crop64_v3_ecnu.slurm)
PREP_JOB=$(sbatch --parsable slurm/01_prepare_dataset_v3_ecnu.slurm)
TRAIN_JOB=$(sbatch --parsable \
  --dependency=afterok:${SMOKE_JOB}:${PREFLIGHT_JOB}:${PREP_JOB} \
  slurm/02_train_4modal_v3_ecnu.slurm)
EVAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} \
  slurm/03_eval_4modal_v3_ecnu.slurm)

printf 'smoke=%s preflight=%s prep=%s train=%s eval=%s\n' \
  "${SMOKE_JOB}" "${PREFLIGHT_JOB}" "${PREP_JOB}" "${TRAIN_JOB}" "${EVAL_JOB}"
```

`SLURM_SUBMIT_DIR` 会把 smoke/preflight 固定到当前 production 代码快照；不要在其他目录执行上述 `sbatch`。两项回归必须分别出现：

```text
SMOKE_TEST_PASS
CROP64_BATCH4_EAGER_PASS
```

任何一项非零退出都会通过 `afterok` 阻止正式训练。

## 6. 四模态并行方式

ECNU 每个 A100 节点只有 2 张 GPU，代码也没有 DDP。正确方案是一个 Slurm array、四个单卡 task：

```text
task 0 = t1c
task 1 = t1n
task 2 = t2w
task 3 = t2f
```

四个 task 可同时占用四个节点上的 A100，也可以根据空闲资源独立启动，不会等待同一节点四卡。

## 7. 默认训练参数

```text
64^3 crop
batch 4
Unet_NnU
channels 48,96,192,384
strides 2,2,2
EDM
zscore + sigma_data 1.0
small_lesion_weight 3.0
patient_balance_mode sqrt
100000 steps
AMP + TF32
```

这是本轮生产基线。不要在四个模态之间改归一化、网络宽度或 noise schedule。

## 8. 查看状态与日志

```bash
squeue -u "${USER}"
sacct -j "${TRAIN_JOB}" --format=JobID,State,ExitCode,Elapsed
tail -f "${PROD_ROOT}/logs/train_${TRAIN_JOB}_0.out"
```

每个模态完成时日志必须出现：

```text
TRAIN_MODALITY_PASS modality=<modality>
```

权重位置：

```text
${PROD_ROOT}/checkpoints/brats2026_diffusion_v3_edm_zscore/<modality>/weights/
```

## 9. 断点续训

脚本默认 `AUTO_RESUME=1`，按数值寻找最大 checkpoint step。单独重跑某模态：

```bash
sbatch --array=0 slurm/02_train_4modal_v3_ecnu.slurm  # t1c
sbatch --array=2 slurm/02_train_4modal_v3_ecnu.slurm  # t2w
```

不要删除其他模态 checkpoint，不要把 `diffusion_90000.pt` 当作比 `diffusion_100000.pt` 更新。

## 10. 验证输出

四个模态全部结束后，`03_eval_4modal_v3_ecnu.slurm` 在固定 103 val 上运行 whole-brain、tile 模式评估：

```text
${PROD_ROOT}/eval/brats2026_diffusion_v3_edm_zscore_whole_brain/metrics.json
```

这些 MSE/MAE/PSNR/SSIM 是 G1 生成质量诊断，不是 BraTS 官方分割榜单指标。最终是否让 synthetic 进入 S1-S5，仍由 G2 QC 和 real-only vs real+synth 消融决定。

## 11. 交给 G2

G1 生成结果是病灶区域 raw diffusion 输出。G2 必须：

1. 按 source case 回填非生成区域。
2. 恢复真实 affine/header。
3. 复制 corrected seg。
4. 写 generation config、checkpoint、seed 和 source manifest。
5. 执行 full-generation QC。

对应脚本仍使用既有接口名：

```text
work_space/G2/code/g2_v2_compose_augmentation.py
work_space/G2/code/g2_synthetic_raw_intake_qc.py
```

这里的 `v2` 是 G2 augmentation 接口名称，不代表继续使用旧的 `BraTS_2023_2024_solutions-main 2` 代码。

## 12. 禁止事项

1. 不使用旧 `main` 或 `main 2` 作为新生产训练入口。
2. 不把 val/test 病例作为训练 source。
3. 不把 fake/broken T2W 病例混入训练。
4. 不手工复制 926 份数据；使用软链接视图。
5. 不把四模态塞进一个未实现 DDP 的四卡进程。
6. 不打开未经服务器验证的 `torch.compile`。
7. 不把 checkpoint、NIfTI、缓存和日志提交 Git。
