# G1 Diffusion augmentation V3 服务器训练手册（NYU Greene）

更新日期：2026-07-15

## 1. 任务边界

本线输入真实病例的 `seg` 和四个完整模态，学习从分割条件生成四模态病灶影像，用于后续 synthetic augmentation。

本线不是缺失 T2W 填补算法。缺失 T2W V3 保留原病例 ID，只修复 T2W；本线会在 G2 composer 后形成新的 synthetic 病例。

正式代码唯一入口：

```text
work_space/G1/code/BraTS_2023_2024_solutions-main 3/Segmentation_Tasks/GliGAN
```

模型参数细节见该目录 `README_DIFFUSION.md`，Slurm 参数和顺序见 `slurm/README.md`。

**目标集群：NYU Greene**（脚本风格对齐 `BraTS_2023_2024_solutions-main 2/.../slurm/*_v2_nyu.slurm`）。  
ECNU 专用 `*_v3_ecnu.slurm` 已从本入口移除；历史部署记录仍保留在 `G1_Diffusion_V3_ECNU部署记录_2026-07-15.md`。

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

265 个 fake/broken T2W 病例不进入本线。患者组 `BraTS-MET-xxxxx` 不得跨 split。两份 corrected seg 按 manifest 相对路径覆盖。

默认按 manifest 中相对 `PROJ` 的路径建软链接，无需复制原始数据。

## 3. 环境（NYU）

```bash
export PROJ=/scratch/bf2260/ECNU_EYU_data
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate g1_diffusion_v3
```

| 项 | 默认 |
|---|---|
| 账号 | `torch_pr_522_general` |
| Conda | `/share/apps/anaconda3/2025.06` |
| 环境名 | `g1_diffusion_v3`（可用 `CONDA_ENV` 覆盖为已有 `brats` / `g1_diffusion_v2`） |
| 邮件 | `bf2260@nyu.edu` |
| 日志 | `/scratch/bf2260/ECNU_EYU_data/logs` |

版本与包要求见：

```text
work_space/G1/docs/G1_Diffusion_V3环境清单.txt
```

禁止在作业中 `pip install`、禁止混用系统 Python。  
NYU 默认 `USE_COMPILE=1`；若 Triton/编译失败再设 `USE_COMPILE=0`。

## 4. 服务器目录

```bash
export PROJ=/scratch/bf2260/ECNU_EYU_data
export RUN_ROOT=${PROJ}/runs/g1_diffusion_v3
```

结构：

```text
${RUN_ROOT}/
  DataSet/              # 926 病例软链接视图（823 train + 103 val）
  splits/current/       # membership 和 lesions.csv
  checkpoints/          # 四模态 checkpoint
  eval/                 # val 指标
  visual/               # 单例可视化
  PREPARED.ok           # 数据准备通过标记
```

大数据、checkpoint、eval 输出不提交 Git。

## 5. 一次性提交生产链

```bash
export PROJ=/scratch/bf2260/ECNU_EYU_data
export RUN_ROOT=${PROJ}/runs/g1_diffusion_v3
mkdir -p "${PROJ}/logs" "${RUN_ROOT}"
cd "${PROJ}/work_space/G1/code/BraTS_2023_2024_solutions-main 3/Segmentation_Tasks/GliGAN"

SMOKE_JOB=$(sbatch --parsable slurm/00_smoke_v3_nyu.slurm)
PREFLIGHT_JOB=$(sbatch --parsable slurm/00_preflight_crop64_v3_nyu.slurm)
PREP_JOB=$(sbatch --parsable slurm/01_prepare_dataset_v3_nyu.slurm)
TRAIN_JOB=$(sbatch --parsable \
  --dependency=afterok:${SMOKE_JOB}:${PREFLIGHT_JOB}:${PREP_JOB} \
  slurm/02_train_4modal_v3_nyu.slurm)
EVAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} \
  slurm/03_eval_4modal_v3_nyu.slurm)
sbatch --dependency=afterok:${TRAIN_JOB} slurm/04_generate_visual_v3_nyu.slurm

printf 'smoke=%s preflight=%s prep=%s train=%s eval=%s\n' \
  "${SMOKE_JOB}" "${PREFLIGHT_JOB}" "${PREP_JOB}" "${TRAIN_JOB}" "${EVAL_JOB}"
```

回归日志必须分别出现：

```text
SMOKE_TEST_PASS
CROP64_BATCH4_EAGER_PASS
PREPARE_DATASET_PASS
```

任何一项非零退出都会通过 `afterok` 阻止正式训练。

## 6. 四模态并行方式（NYU）

与 V2 NYU 一致：一个作业申请 **4 张 GPU**，同节点按 `CUDA_VISIBLE_DEVICES=0..3` 各跑一个模态进程。

```text
GPU 0 = t1c
GPU 1 = t1n
GPU 2 = t2w
GPU 3 = t2f
```

不使用 ECNU 式 4 个 array 单卡任务。若节点没有 4 卡可用，需要改脚本或分模态提交。

## 7. 默认训练参数

```text
64^3 crop
batch 8（OOM 时 BATCH_SIZE=4）
Unet_NnU
channels 48,96,192,384
strides 2,2,2
EDM
zscore + sigma_data 1.0
small_lesion_weight 3.0
patient_balance_mode sqrt
100000 steps
USE_COMPILE=1
AMP + TF32
```

不要在四个模态之间改归一化、网络宽度或 noise schedule。

## 8. 查看状态与日志

```bash
squeue -u "${USER}"
sacct -j "${TRAIN_JOB}" --format=JobID,State,ExitCode,Elapsed
tail -f /scratch/bf2260/ECNU_EYU_data/logs/g1_diffv3_train4_${TRAIN_JOB}.out
tail -f "${PROJ}/work_space/G1/code/BraTS_2023_2024_solutions-main 3/Segmentation_Tasks/GliGAN/slurm/logs/train_${LOGDIR}_t1c_${TRAIN_JOB}.log"
```

每个模态完成时对应日志必须出现：

```text
TRAIN_MODALITY_PASS modality=<modality>
```

权重位置：

```text
${RUN_ROOT}/checkpoints/brats2026_diffusion_v3_edm_zscore/<modality>/weights/
```

## 9. 断点续训

脚本默认 `AUTO_RESUME=1`，按数值寻找最大 checkpoint step。整作业重提即可；已完成模态会跳过。

不要删除其他模态 checkpoint，不要把 `diffusion_90000.pt` 当作比 `diffusion_100000.pt` 更新。

## 10. 验证输出

四个模态全部结束后，`03_eval_4modal_v3_nyu.slurm` 在固定 103 val 上运行 whole-brain、tile 模式评估：

```text
${RUN_ROOT}/eval/brats2026_diffusion_v3_edm_zscore_whole_brain/metrics.json
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
5. 不把四模态塞进未实现 DDP 的单进程。
6. 不把 checkpoint、NIfTI、缓存和日志提交 Git。
7. 不在本入口提交 ECNU 分区/路径脚本。
