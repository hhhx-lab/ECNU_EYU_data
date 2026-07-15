# G1 Diffusion augmentation V3 ECNU 部署记录

## 1. 部署结论

第三版代码已在 ECNU 真实 Slurm/GPU 环境完成端到端 smoke 与正式尺寸预检。正式训练采用四个独立 A100 array task，不使用单节点四卡或未实现的 DDP。

## 2. 真实测试证据

| 作业 | 资源 | 状态 | 验证内容 |
|---|---|---|---|
| `3086919` | V100 x 1 | COMPLETED 0:0 | 第一轮训练、checkpoint、评估、生成、在线增强 |
| `3088533` | V100 x 1 | COMPLETED 0:0 | 修正坐标后的端到端回归 |
| `3088957` | V100 x 1 | COMPLETED 0:0 | 固化网络结构元数据后的最终回归 |
| `3088958` | V100 32GB x 1 | COMPLETED 0:0 | `64^3 + batch=4 + channels 48/96/192/384` 一步训练 |

最终日志标记：

```text
axis_contract=PASS
gpu_pipeline_contract=PASS
SMOKE_TEST_PASS
CROP64_BATCH4_EAGER_PASS
```

## 3. 正式数据口径

来源：`work_space/G2/results/manifests/g1_v2_source_manifest.csv`。

| 角色 | 病例数 | 用途 |
|---|---:|---|
| authentic train | 823 | 四模态训练 |
| authentic val | 103 | 固定验证 |
| locked test | 104 | 不读取 |
| fake/broken T2W | 265 | 不进入 diffusion augmentation 训练 |

使用 corrected seg 的病例按 manifest 覆盖。数据视图只创建软链接，不复制 40GB 原始影像。

## 4. 正式作业

| 作业 | 说明 | 依赖 | 2026-07-15 02:58 状态 |
|---|---|---|---|
| `3089280` | `64^3/batch=4` 正式配置预检 | 无 | `COMPLETED 0:0` |
| `3089332` | 最终部署代码 A100 端到端 smoke | 无 | 运行中；已通过 CSV 契约、一步训练并写出 checkpoint |
| `3089742` | CPU 数据视图、几何/标签扫描、lesion CSV | 无，与 smoke 并行 | 运行中，节点 `node32` |
| `3089743_[0-3]` | `t1c/t1n/t2w/t2f` 四个单卡 A100 训练 task | `afterok:3089332:3089742` | 等待双重依赖 |
| `3089744` | 四模态固定 val whole-brain 评估 | `afterok:3089743` | 等待训练数组 |

数据准备与 smoke 使用不同资源并行执行，以缩短启动等待；正式训练必须同时等待两者成功，任何上游非零退出都不会放行训练。四个模态是四个独立 array task，每个 task 只申请一张 A100，可独立排队、续训和失败重跑。

第一轮生产链 `3089016/3089057/3089058` 在训练启动前主动取消，原因是复核发现 patient balance 应按 patient group 而不是 case timepoint 统计；它们没有产生或覆盖任何正式 checkpoint。

`3089272` 因预检脚本缺少 `CODE_DIR` 默认值在 1 秒内失败，依赖保护自动取消 `3089273/3089274/3089275`。修复后 `3089280` 已完成正式尺寸预检；当时提交的其余依赖任务后来由当前生产链替换，均未进入正式训练。

`3089271` 的 A100、Python、Torch 和 MONAI 检查均通过，但 smoke 默认指向服务器上不存在的 Git 工作区路径，最终以 `No such file or directory` 退出；其依赖链 `3089281-3089283` 未进入数据准备或训练。`3089297` 使用 Slurm 临时 spool 目录推导代码路径，同样在训练前被替换。最终脚本优先从 `SLURM_SUBMIT_DIR` 使用操作者实际提交的 production 代码快照，仅在该变量不存在时回退到 `${SMOKE_ROOT}/code/Segmentation_Tasks/GliGAN`；真实作业 `3089332` 已越过路径、数据契约和 GPU 训练阶段。

正式服务器目录：

```text
/public/home/${USER}/g1_diffusion_v3_production_20260715/
  code/
  manifests/
  corrected_labels/
  DataSet/
  splits/current/
  checkpoints/
  eval/
  logs/
```

## 5. 本轮修复

1. 修复 NIfTI `(x,y,z)` 与旧 `(z,y,x)` 混用导致的训练 crop 错位。
2. 修复 oversized lesion tile 的轴顺序和病灶体素统计。
3. 强制 train/val/test 和 authentic T2W 策略，禁止 locked test 进入训练。
4. 去掉会冻结随机 crop/noise 的全量缓存。
5. 增加非法 label、缺模态、shape/affine、空 crop、非有限值硬失败。
6. checkpoint 记录归一化、数据集、crop、噪声策略和网络结构。
7. 修复 checkpoint 数字排序。
8. 补齐 EDM 在线增强 known-region projection。
9. 适配服务器扁平数据命名，无需复制或手工改 926 个病例。
10. 关闭 ECNU 不兼容的 `torch.compile`，保留 AMP、TF32 和四卡任务并行。
11. `patient_balance_mode=sqrt` 按统一 `BraTS-MET-xxxxx` patient group 统计，不按时间点 case ID 重复加权。

## 6. 运行维护

训练脚本默认自动从当前模态最大 step 续跑。某一模态失败时只重提对应 array task，例如 `--array=2` 只重跑 T2W。任何上游非零退出都会阻止 `afterok` 下游任务启动。
