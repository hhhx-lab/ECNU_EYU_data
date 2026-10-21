# S1 工作区

## 整体任务

详见 [task_assignment.md](task_assignment.md)。

## 快速入口

| 文档 / 脚本 | 路径 |
| --- | --- |
| **服务器运行手册（权威）** | [docs/S1_服务器运行手册.md](docs/S1_服务器运行手册.md) |
| **推荐 Slurm** | [slurm/01_s1_realonly.slurm](slurm/01_s1_realonly.slurm) |
| 执行逻辑 | [slurm/run_s1_realonly.sh](slurm/run_s1_realonly.sh) |
| 代码仓库 | [brats2026_multitask_S1_v2/repository](brats2026_multitask_S1_v2/repository) |
| 环境创建 | [brats2026_multitask_S1_v2/repository/docs/S1_experiment.txt](brats2026_multitask_S1_v2/repository/docs/S1_experiment.txt) |

正式训练（在项目根目录）：

```bash
export PROJECT_ROOT=/scratch/bf2260/ECNU_EYU_data
cd "${PROJECT_ROOT}" && mkdir -p logs
sbatch --gres=gpu:a100:1 \
  --export=ALL,PROJ="${PROJECT_ROOT}" \
  work_space/S1/slurm/01_s1_realonly.slurm
```

## 当前进度

- 多任务训练代码已按 96³ 显存安全默认 + 全量 SWI 验证优化
- Slurm / 运行手册已 fail-fast 重写

## 本周计划

待填写。

## 提交记录

| 日期 | 内容 | 备注 |
| --- | --- | --- |
| 2026-07-15 | 重写 S1 Slurm 与服务器运行手册 | fail-fast 预检、≥40GB、G2 view 审计 |
