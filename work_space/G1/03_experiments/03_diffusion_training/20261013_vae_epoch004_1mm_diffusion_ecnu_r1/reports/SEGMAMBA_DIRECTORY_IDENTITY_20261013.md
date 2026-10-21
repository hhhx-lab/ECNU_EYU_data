# 远端SegMamba-main目录身份与占用

本地整理UTC2026-10-13T13:55:59Z。远端根`/public/home/zqchen/SegMamba-main`；研究登记`experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。用户要求解释该目录，仅只读检查，不授权清理或实验。

## 是什么

这是账号主目录中独立存放的BraTS脑转移瘤SegMamba分割代码、预处理数据和实验结果。`3_train_brats2026.py`164–166行导入并实例化`model_segmamba.segmamba.SegMamba`，默认4模态输入，支持NETC/SNFH/ET/RC的指标和CE/Dice/Focal组合损失。这里不是Diffusion，也不是Conda `segmamba`环境目录（后者位于`.conda/envs/segmamba`）。

已见模型mtime为2026-06-28至07-06（文件时间不是完整训练审计），保留`exp_ce_dice`、`exp_ce_focal`、`exp_dice_focal`、`exp1_larger_patch`、`exp3_deeper_model`的best/final/中间`.pt`。configs还包含adamw_cosine/no_augmentation/high_overlap设计，但仅配置存在不代表训练完成。文件名中的0.6793等数值不是本次已验收的官方MET，不作科学比较。

它不等于现行比赛原E或K1/K2/K4：那些本地合同绑定nnU-Net Dataset264/原E SHA及各自独立运行包。本次未找到本地总纲和相关活动S1/S2目录将这个SegMamba旧实验直接绑定到现行S1或原E，故登记为旧独立分割分支、具体项目阶段归属待核实；不能按名称猜成当前S1正式结果，也不能视为垃圾。

## 135.90 GiB组成

| 内容 | GiB | 说明 |
|---|---:|---|
| data整体 | 129.91 | 下三项属于这里，不再重复相加 |
| data/fullres/train | 111.96 | 5184平铺文件，示例同病例.npy/.npz/.pkl/_seg.npy；不能仅除以4就认定有效病例/合法split |
| data/validation_preprocessed | 12.46 | 716平铺文件，预处理缓存 |
| data/validation | 5.06 | 179病例目录；仅路径数，不冒充官方集身份验收 |
| results | 5.45 | 五类已见模型实验及其他结果 |
| mamba | 0.34 | 包含build目录；不是135 GiB的主要部分 |
| logs | 0.14 | 包含Slurm日志 |

`data/raw_data`另约0.43 GiB，包含在data整体。主要空间来源是预处理数据，而不是源码或权重。`.npy`与`.npz`可能是缓存/压缩与解压版本，但本次未做内容/SHA/loader依赖核验，不能宣布重复可删除。

results细分：CE+Dice1.81 GiB、Dice+Focal1.29、CE+Focal1.04、大patch0.76、更深模型0.55；从文件名和metadata知道曾有训练产物，不代表其完成状态/最佳选点已经重新验收。没有加载checkpoint或推理。

## 方法与证据

一次`du -x -B1 --max-depth=3`完整返回145924839424分配bytes；不解引用symlink。后续一次只读训练代码选择行、配置文件名、模型stat和病例目录数量，不读取影像/权重内容、不运行项目脚本、不连接Git、不提交/取消作业。VPN每次显式启动并仅关闭本次VPN或保留既有VPN。

- `operator/readonly_segmamba_inventory_20261013T135421Z/SEGMAMBA_INVENTORY.json`，远端UTC2026-10-13T13:43:36.012394+00:00。
- `operator/readonly_segmamba_identity_20261013T135528Z/SEGMAMBA_IDENTITY.json`，含实际SegMamba实例化行、模型文件metadata、配置和数据目录数；本机与远端时钟约11分钟差，未同步。

未清理/迁移任何文件。若后续考虑释放这块空间，先确认分支归属、当前任务依赖、与现有源数据逐SHA等价性及完整可恢复归档；仅问“是什么”不授权删除。
