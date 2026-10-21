# ECNU_EYU

本仓库用于整理 ECNU_EYU 参赛资料、任务分工、成员工作区、阶段进度和参考代码。当前结构围绕 2026 BraTS Task 1 建立，公共资料放在 `data_space`，个人工作内容放在 `work_space`，阶段计划放在 `time_pipeline`。

## 文件结构

```text
.
├── README.md
├── PROJECT_RULES.md
├── data_space/
│   ├── README.md
│   ├── task1_2026/
│   │   ├── BraTS2026_Task1_中文.md
│   │   ├── BraTS2026_Task1_English.md
│   │   ├── datasets/
│   │   │   ├── README.md
│   │   │   ├── 数据集现状.md
│   │   │   ├── 数据契约.md
│   │   │   └── Sage Bionetworks 登录.pdf
│   │   └── reference_code/
│   │       └── nnunet_custom/
│   ├── past_articles/
│   │   ├── 2023/
│   │   └── 2025/
│   ├── past_code/
│   │   ├── 2023/
│   │   ├── 2025/
│   │   ├── 代码解析.md
│   │   └── 往年冠军代码_逐文件作用详解.md
│   ├── bios_course_pdfs/
│   │   ├── C1.pdf ... C11.pdf
│   │   └── c6_diffusion_outputs/
│   └── task_assignments/
│       └── ECNU-NYU2026分工架构(2).docx
├── work_space/
│   ├── README.md
│   ├── G1/
│   ├── G2/
│   ├── S1/
│   ├── S2/
│   ├── S3/
│   ├── S4/
│   └── S5/
└── time_pipeline/
    └── README.md
```

## 资料区

`data_space` 存放公共参考材料和项目资料：

- `task1_2026/`：BraTS 2026 Task 1 中英文说明、数据说明、数据契约、Sage 登录资料和 nnU-Net 自定义参考代码。
- `task1_2026/datasets/`：只保存数据说明和访问资料，不保存大型医学影像原始数据。
- `past_articles/`：往年冠军论文和参考文章。
- `past_code/`：往年代码、示例工程、依赖文件和代码解析资料。
- `bios_course_pdfs/`：BIOS AI 课程 PDF 及 C6 扩散模型整理材料。
- `task_assignments/`：成员分工、任务安排和会议/职责资料。

大型训练集、验证集、`.nii/.nii.gz` 医学影像和数据压缩包不进入仓库，具体保留位置见 `data_space/task1_2026/datasets/README.md`。

## 工作区

`work_space` 按成员或小组划分：

- 生成模型小组：`G1`、`G2`
- 分割模型小组：`S1`、`S2`、`S3`、`S4`、`S5`

每个工作区包含：

- `README.md`：记录整体任务、当前进度、本周计划和提交记录。
- `task_assignment.md`：从原始分工文档拆出的个人任务。
- `code/`：个人代码、脚本、notebook 或实验工程。
- `data/`：个人处理后的数据说明或小规模派生数据。
- `docs/`：个人任务文档、分析记录和方法说明。
- `results/`：图表、实验结果和阶段产出。

每个人仅在自己的工作区内修改和上传内容；需要共享的资料应放入 `data_space`。

## 时间进度线

`time_pipeline/README.md` 用于记录比赛时间节点、阶段性任务、成员进度和下一步计划。每次阶段性提交后，应同步更新自己的任务状态。

## 提交规则

1. Git commit message 使用简体中文，简要描述改动内容。
2. 代码需要具有可读性，必要时补充注释或文档说明。
3. 提交文档必须可读，不出现乱码；正式文档中的公式不要保留为 LaTeX 源码。
4. 不要改动其他同学工作区的内容。
5. 不要提交缓存、临时测试文件、调试产物、本地代理配置或大型原始数据。
6. 提交频率建议一周两次左右，不低于两次。
