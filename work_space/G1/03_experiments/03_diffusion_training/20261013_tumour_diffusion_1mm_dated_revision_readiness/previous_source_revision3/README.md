# Tumour Diffusion（1 mm 版）— 在 nnU-Net 1 mm 空间训练的肿瘤条件扩散模型

训练条件扩散模型，输入分割标签（seg），输出单模态脑 MRI（t1c / t1n / t2w / t2f 各一个模型），用于 S2 nnU-Net 的在线肿瘤增强。

本目录复制自 `Segmentation_Tasks/GliGAN/`（扩散部分）和仓库根目录的 `model.py`。**网络结构、扩散框架、损失和训练超参与原版一致**（唯一例外是默认训练步数从 100k 改为 200k），改动集中在数据和工程层面：

- 训练数据先按 nnU-Net v2 的方式预处理，即重采样到 1×1×1 mm、做整例带掩膜的 Z-score，和 S2 在线增强推理时扩散模型实际看到的输入完全相同；
- 修复了原版数据管线的几个问题：**病灶坐标错轴**（原版约 80% 的训练 crop 不含目标病灶）、内存占用过大、随机偏移被缓存冻结；
- 训练大幅提速：原版 10 万步要两三天，本版本 20 万步在 4×A100 上约 1.5–2 小时，全流程约 3–4.5 小时（原因与估算见第 3 节）。

全部改动按类别列在 [4.2 节](#42-修改内容及原因)；原版代码嫁接这些改进后与本方法还有什么不同，见 [4.5 节](#45-原版嫁接这些改进后与本方法还有什么不同)。

验证集的指定方式与原版相同，由使用者自己选择；区别是指定后的验证集确实不参与训练（原版训练代码实际上也拿它来训练），其余病例全部进入训练集。

原目录保持不变。

> ⚠️ **不要沿用原版 GliGAN 或本目录早期版本生成的 CSV 和 checkpoint。** 它们受病灶坐标错轴影响（见 [第 8 节](#8-修复记录2026-10-12审查反馈)）：请按第 2 节重新建 CSV、从头训练。这类 CSV 在训练建缓存时会直接报错，用旧 checkpoint 配新 CSV 续训也会被拒绝。

> 噪声调度、EDM / logsnr 调参、CFG、小病灶与患者级加权的原理和参数说明，见本目录的 [`README_DIFFUSION.md`](README_DIFFUSION.md)（第 3、4、7、8、9 节）。这些内容对本版本完全适用，本文不再重复。

---

## 运行前须知

💻 建议用 conda 单独建一个环境，并使用带 GPU 的 Linux 服务器（S2 用的 4×A40 即可）。

**创建 conda 环境：**

1. `conda create -n tumour_diff python=3.11 -y && conda activate tumour_diff`
2. `pip install -r requirements.txt`
3. 安装 nnU-Net v2（Step 1 预处理要用）：
   1. 如果在 S2 的环境里运行，它已经装好了，直接用即可（**推荐**，这样版本与 S2 完全一致）；
   2. 否则 `cd ../Segmentation_Tasks/nnUNet_install && pip install -e .`（nnU-Net v2.1.1）。

⚠️ **内存（RAM）**：和原版一样，训练默认用 MONAI `CacheDataset` 把训练数据全部缓存进内存（`cache_rate=1`）。本版本每行 CSV 只缓存病灶周围最大 80³ 的区域（实测平均 2.0 MB/行，原版是 111 MB/行），每个模态的进程约需 15–20 GB，4 个模态同时训练合计约 60–70 GB。内存不够时用 `--cache_rate` 调低（例如 `0.5`），代价是未缓存的行每次都要从磁盘读取。

⚠️ **显存（VRAM）**：batch 16 时实测约 1.2 GB，任何卡都能跑。但网络宽度是由"GPU 显存规划器"决定的，训练时必须保留 `--planner_gpu_gb 48`，才能保证得到和 A40 / A100 上相同的参考架构（channels 48–384），详见 [7. 常见问题](#7-常见问题与注意事项)。

⚠️ **CPU**：随机偏移和加噪每次迭代都要在 CPU 上重新计算，每个样本约 10–14 ms。建议每个训练进程开 `--loader_workers 8`，4 个进程共需约 32 个 CPU 核；核数不够时可以设成 4。

⚠️ **磁盘**：预处理后，每例平均约 60 MB；全部 1,296 例约 75–80 GB。

🤞 本目录的最终代码已在本地用 `scripts/smoke_test.sh` 跑通全流程（2026-10-12），环境为 Windows、Python 3.9.21、torch 2.1.0+cu121、MONAI 1.5.2、RTX 4070 Laptop 8 GB。正式训练在 Linux 服务器上进行，上机后请先跑一遍冒烟测试（第 2 节开头）。如果遇到版本不兼容，优先对齐 S2 环境的版本。

---

## 0. 环境

```bash
conda create -n tumour_diff python=3.11 -y && conda activate tumour_diff
pip install -r requirements.txt
# 预处理用 nnU-Net v2（S2 环境中已有则跳过）
cd ../Segmentation_Tasks/nnUNet_install && pip install -e . && cd -
```

**所有命令都在 `TumourDiffusion_1mm/` 目录下执行。**`Checkpoint/`、`DataSet/` 都在本目录下（原版在仓库根目录的 `../../Checkpoint`）。

---

## 1. 数据

### 1.1 原始数据

使用 `E:\medical` 下 `1`–`9`、`u1`–`u9` 共 18 个文件夹里的全部 1,296 例。每个病例一个文件夹：

```
1/
├── BraTS-MET-00001-000/
│   ├── BraTS-MET-00001-000-t1c.nii.gz
│   ├── BraTS-MET-00001-000-t1n.nii.gz
│   ├── BraTS-MET-00001-000-t2w.nii.gz
│   ├── BraTS-MET-00001-000-t2f.nii.gz
│   └── BraTS-MET-00001-000-seg.nii.gz
├── BraTS-MET-00002-000/
└── ...
u1/ ... u9/
```

> 这 1,296 例中只有 375 例是 1 mm 各向同性；其余 921 例保留原始采集分辨率，体素体积在 0.065–2.42 mm³ 之间。这正是需要 Step 1 预处理的原因，详见 [4.2 节](#42-修改内容及原因)。
>
> 文件夹里的 `__MACOSX/`、误放在 `1/BraTS-MET-00035-000/` 里的 zip、`u1/BraTS-MET-01057-000/` 下解压出来的 `*.nii/` 目录等杂项，预处理时会自动忽略（只读取上面 5 个标准文件名）。

**放在哪里**：这 18 个文件夹可以放在服务器上任何位置，**不需要复制进本目录**，Step 1 用 `--raw_dirs` 指向它们即可，例如放在 `/data/medical/1`、…、`/data/medical/u9`。要求只有一条：传给 `--raw_dirs` 的每个文件夹下面**直接**就是 `BraTS-MET-*` 病例文件夹。如果上传的是 zip，解压后先 `ls /data/medical/1 | head` 确认没有多套一层（例如 `1/1/BraTS-MET-...`）；如果多了一层，`--raw_dirs` 就指向里面那一层。

> ⚠️ **不要把原始病例直接放进 `DataSet/` 去训练。** 训练读取的是 Step 1 的输出：1 mm、已按整例 Z-score 归一化。`--normalization nnunet` 不会再做任何归一化，原始数据放进去会以原始强度值直接进入网络，而且回到了体素间距不一致的问题。

### 1.2 划分训练集与验证集（全部可选）

和原版一样，**由你自己决定**哪些病例做验证集：

- **哪些病例参与**：由 `DataSet/` 里有哪些病例决定。Step 1 默认处理全部 1,296 例；只想用其中一部分时，用 `--case_list` / `--exclude_list` 筛选。
- **哪些做验证集**：Step 2 用 `--val_patients`（原版参数，逗号分隔）或 `--val_list`（txt 文件，每行一个病例）指定。其余病例全部是训练集，不会被跳过。
- **不指定验证集**：所有病例都进入训练集。此时 Step 5 评估会报错，需显式传 `--eval_split all`，评估的就是训练数据，指标会偏乐观。

名单文件每行一个病例编号，写完整名称（`BraTS-MET-00001-000`）或 patient ID（`00001-000`）都可以，`#` 开头的行是注释。另有两个可选的辅助工具：

| 工具 | 内容 | 用法 |
|---|---|---|
| `scripts/make_case_lists.py` 的输出 | 从 S2 的 split 导出的 `train_cases.txt` / `val_cases.txt` / `test_cases.txt` | 想与 S2 划分保持一致时使用，见 Step 0 |
| `lists/t2w_invalid_cases.txt` | 266 例 T2W 不含有效信息的病例 | 想让 t2w 模型排除这些病例时，传给 Step 2 的 `--t2w_invalid_list` |

> **提醒（不强制）**：验证集的病例不参与训练；但如果把 S2 的验证集或测试集病例放进训练集，扩散模型会记住它们的病灶外观，再通过在线增强"带进" nnU-Net 的训练，可能使 nnU-Net 的验证、测试分数偏高。如何划分请根据实验目的自行决定。
>
> **关于 `t2w_invalid_cases.txt`**：S2 中有 212 例训练病例的 T2W 文件存在，但不含与其他三个模态对应的真实信息，这些病例在 S2 里用填补后的 T2W 替换。本名单是根据 `Segmentation_Tasks/GliGAN/brats2026_diffusion_v2_complete_only.csv` 反推出来的：标签非空、但不在该"四模态完整"名单中的 266 例，集中在 4–8 号文件夹，u2 有 1 例。使用前**请用 S2 的权威补全名单核对**，212 例应当是它的子集。抽查中，名单里的 `BraTS-MET-00570-000` 的 T2W 看起来有正常的病灶信号，需要确认。不传这个名单时，t2w 模型和其他模态一样使用全部训练病例。

### 1.3 预处理后的数据（`DataSet/`，Step 1 生成）

目录结构与原版 `DataSet/` 相同，后续步骤无需改动：

```
DataSet/
├── BraTS-MET-00001-000/
│   ├── BraTS-MET-00001-000-t1c.nii.gz   # float32，1×1×1 mm（文件头间距精确为 1），nnU-Net 整例带掩膜 Z-score
│   ├── ...-t1n / -t2w / -t2f.nii.gz
│   └── BraTS-MET-00001-000-seg.nii.gz   # uint8，标签 0–4（nnU-Net 的 -1 "脑外" 记为 0）
├── ...
├── preprocess_report.csv                # 每例的原始尺寸、间距、新尺寸、重采样前后病灶数
└── preprocess_config.json               # 使用的后端与参数
```

---

## 2. 运行

### 上机后先跑冒烟测试（约 5 分钟，单卡）

正式训练前，先用 4 个真实病例把整条流程跑一遍：预处理 → 建 CSV → 训练 t1c 和 t2w（各 120 步）→ 续训 → 自检（每个 crop 都包含它的病灶、checkpoint 含 AMP 状态和数据指纹、无残留临时文件）→ 按步数评估 → 全脑评估 → 从标签生成。任何一步失败都会停下并打印日志，全部通过时最后打印 `SMOKE TEST PASSED`。

```bash
RAW_DIRS="/data/medical/1 /data/medical/5" bash scripts/smoke_test.sh
RAW_DIRS="/data/medical/1 /data/medical/5" COMPILE=--use_compile bash scripts/smoke_test.sh   # 同时测试 torch.compile
```

默认病例为 `BraTS-MET-00001/00002/00004-000`（1 号文件夹）和 `BraTS-MET-00572-000`（5 号文件夹），可用 `CASES`、`VAL` 改。输出写在 `smoke_tmp/` 和 `Checkpoint/zz_smoke/`，结束后自动删除；加 `KEEP=1` 保留。

### Step 0（可选）: 从 S2 的划分导出病例名单

只在想与 S2 的划分保持一致时需要；自己指定验证集时跳过本步。从 S2 的 split JSON（格式 `{"train": [...], "val": [...], "test": [...]}`，里面是 nnU-Net 病例编号）和映射表（列名 `nnunet_case_id, source_case_id`）导出原始病例编号：

```bash
python scripts/make_case_lists.py \
    --split_json /path/to/S2/split.json \
    --mapping_csv /path/to/S2/mapping.csv \
    --out_dir lists
```

之后可以把 `lists/val_cases.txt` 传给 Step 2 的 `--val_list`；不想让测试集参与时，把 `lists/test_cases.txt` 传给 Step 1 的 `--exclude_list`。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--split_json` | (必填) | S2 `scripts/04_build_fixed_split.py` 使用的 split 文件 |
| `--mapping_csv` | "" | nnU-Net 编号 → 原始编号的映射；split 里已经是原始编号时可省略 |
| `--out_dir` | `lists` | 输出目录 |

### Step 1: 预处理到 nnU-Net 1 mm 空间

```bash
python scripts/preprocess_dataset.py \
    --raw_dirs /data/medical/{1..9} /data/medical/u{1..9} \
    --out_dir DataSet \
    --num_workers 8
```

把 `/data/medical` 换成原始数据的实际位置；`{1..9}`、`u{1..9}` 是 bash 写法，会展开成 18 个文件夹。默认处理这些文件夹里的全部病例，输出写到本目录的 `DataSet/`。只想处理一部分时，加 `--case_list <名单>` 或 `--exclude_list <名单>`。

处理顺序与 nnU-Net v2 `DefaultPreprocessor`（3d_fullres）一致：

1. 读入 4 个模态和标签；
2. 裁剪到 4 个模态非零区域并集（补洞后）的包围盒，区域外的标签记为 −1；
3. 在**原始分辨率**上逐模态做带掩膜的 Z-score；
4. 重采样到 1×1×1 mm：影像用三阶样条插值，标签逐类别一阶插值后取 ≥ 0.5；各向异性比大于 3 的病例，z 轴单独用最近邻处理。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--raw_dirs` | (必填) | 包含 `BraTS-MET-*` 病例文件夹的目录，可以列多个 |
| `--out_dir` | `DataSet` | 输出目录。可以放到别的磁盘，此时 Step 2 的 `--datadir` 要写同一路径（建议用绝对路径）；建好 CSV 后不要再移动它，否则要重建 CSV |
| `--case_list` | "" | 只处理名单中的病例；为空则处理全部 |
| `--exclude_list` | "" | 跳过名单中的病例 |
| `--target_spacing` | `1 1 1` | 目标间距（mm），须与 S2 的 nnU-Net plans 一致 |
| `--num_workers` | 4 | 并行病例数；最大的病例（640×640×392）每个进程约需 10 GB 内存 |
| `--overwrite` | 关闭 | 默认跳过已存在的输出，所以中断后可以直接重新运行续上（每个文件先写临时文件再改名，中断不会留下半个文件） |
| `--allowed_labels` | `0 1 2 3 4` | **[新增]** 标签只允许这些值；含其他值、非整数或 NaN 的病例被拒绝 |
| `--affine_tol_mm` | 0.1 | **[新增]** 4 个模态与标签的体素网格在空间中最多可偏差多少 mm，超出即拒绝 |

**数据检查**（[新增]，不通过的病例不会写入 `DataSet/`，在报告里标为 `rejected: 原因`，结束时汇总打印）：
- 4 个模态与标签的尺寸相同，且空间坐标（affine）一致：8 个角点的偏差不超过 `--affine_tol_mm`；
- 影像和标签里没有 NaN / Inf；
- 标签是整数，且只含 `--allowed_labels` 中的值。

在 1,296 例原始数据上，只有标签这一项会拒绝病例：`BraTS-MET-01094-002`（含标签 6）和 `BraTS-MET-01184-002`（含标签 8）。这两例很可能是标注错误，请向数据方或 S2 确认正确的标签后修正，再重新运行；不修正则这两例不参与训练。模态间的最大偏差只有 0.000005 mm，没有 NaN。

> **后端**：启动时会打印 `Backend: ...`。
> - `nnunetv2`：直接调用 nnU-Net 自己的裁剪和重采样函数，与 S2 的数组逐位一致，**推荐**；
> - `skimage`：使用内置的 nnU-Net v2.1 重采样逐行复刻；
> - `scipy`：使用 scikit-image `resize` 的等价实现。
>
> 后两种的数值差异极小，但正式训练请在装有 nnunetv2 的环境里运行，且版本与 S2 一致。
>
> **跑完后检查** `DataSet/preprocess_report.csv`：
> - `status` 列应为 `ok`，`rejected` 的病例见上方"数据检查"；
> - `true_spacing` 列是取整后的真实间距（与 1 mm 相差千分之几）。文件头里写的是精确的 1 mm，与 nnU-Net 的约定相同：nnU-Net 处理出的是同一个数组，并把它当作 1 mm；
> - `lesions_lost` 列是重采样后消失的连通域数（主要来自 u 组 0.5 mm 病例里小于 1 mm³ 的极小病灶）。nnU-Net 在 S2 中做的是同样的重采样，分割端本来就看不到这些病灶。

### Step 2: 创建 CSV 索引

```bash
python src/train/csv_creator.py \
    --dataset BRATS_2024 \
    --datadir DataSet \
    --logdir brats_1mm \
    --crop_size 64 \
    --merge_dist 16 \
    --val_list my_val_cases.txt          # 可选；或 --val_patients "00002-000,00005-000"
```

输出 `Checkpoint/brats_1mm/brats_1mm.csv`。`DataSet/` 里的每个病例都会写入；每行代表一个病灶（或超大病灶的一个瓦片），连通域分析、邻近病灶合并、预切片的规则与原版相同，只修正了坐标轴顺序（见下方）。`split` 列与原版一样（val / train），另外多了一列 `t2w_valid`（0/1，不传 `--t2w_invalid_list` 时全为 1）。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--crop_size` | 64 | crop/pad 目标尺寸，须与训练一致 |
| `--merge_dist` | 16 | 合并质心距离小于 N 体素的病灶（1 mm 下即 N mm） |
| `--val_patients` | "" | 原版参数：逗号分隔的验证集 patient ID（如 `00002-000`），这些病例 `split=val`，其余为 `train` |
| `--val_list` | "" | **[新增]** 与 `--val_patients` 作用相同，从 txt 文件读取，适合病例较多时；两者可同时使用，取并集 |
| `--t2w_invalid_list` | "" | **[新增，可选]** 列出的病例 `t2w_valid=0`，只有 t2w 模型训练时会排除它们 |

> 终端会打印 `Train lesions / Val lesions`。约 1,000 例训练病例对应约 8,000 行，以实际输出为准。
>
> **[修复] 坐标轴**：CSV 里的 `x/y/z` 列现在依次对应数组的第 0/1/2 轴，与训练时裁剪的方式一致（原版把第 0 轴和第 2 轴弄反了，见第 8 节）。每写一行都会自检"按训练时的取法，这一行的范围内包含它的全部病灶体素"，不满足会直接报错。
>
> **[新增] 标签检查**：含有数据集不认识的标签值（BRATS_2024 允许 0–4）的病例会被跳过，结束时打印 `[WARNING] skipped N case(s) with invalid labels`。

### Step 3: 训练（每个模态单独训练，4 卡并行）

**一键启动**，每张卡训练一个模态，日志写到 `logs/train_<模态>.log`：

```bash
bash scripts/train_4gpu.sh                                  # 默认 200k 步
LOGDIR=brats_1mm NUM_STEPS=100000 bash scripts/train_4gpu.sh  # 改步数
GPUS="4 5 6 7" LOADER_WORKERS=4 bash scripts/train_4gpu.sh   # 指定 GPU、CPU 核较少时
```

**等价的单模态命令**。网络和扩散相关参数与 README_DIFFUSION 的 Step 2 完全相同，只多了标 [新增] 的参数：

```bash
CUDA_VISIBLE_DEVICES=0 python src/train/tumour_main_diffusion.py \
    --dataset BRATS_2024 --modality t1c --logdir brats_1mm \
    --batch_size 16 --generator_type Unet_NnU \
    --crop_size 64 --small_lesion_weight 3.0 \
    --num_steps 200000 --noise_schedule edm --use_compile \
    --normalization nnunet --sigma_data 1.0 \
    --planner_gpu_gb 48 --num_workers 8 --loader_workers 8
```

> 把 `--modality t1c` 依次换成 `t1n`、`t2w`、`t2f`，`--logdir` 保持一致。**必须用 `CUDA_VISIBLE_DEVICES` 为每个进程指定不同的 GPU**，否则 4 个进程会全部挤在 0 号卡上（代码里是 `model.cuda()`）。
>
> 训练时只使用 `split == train` 的行，验证集的行不参与训练。如果 Step 2 传了 `--t2w_invalid_list`，训练 t2w 时还会去掉 `t2w_valid == 0` 的行，日志里会打印 `[t2w] dropped N rows`。
>
> `--use_compile` 开启后，前几十步要编译，速度较慢，属于正常现象。

**本版本新增或改变含义的训练参数：**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--normalization` | `minmax` | **新增取值 `nnunet`**：数据已经在 Step 1 里按整例 Z-score 归一化，crop 不再单独归一化，背景和填充值为 0。本版本**必须用 `nnunet`** |
| `--planner_gpu_gb` | 实际显存 | 网络宽度规划器假定的显存。**须设为 48**，以获得 A40/A100 参考架构（channels 48/96/192/384，约 4.5M 参数）。该值会写进 checkpoint，推理和续训时自动沿用 |
| `--loader_workers` | 0 | DataLoader 的工作进程数。0 等同原版行为；Linux 上建议 8 |
| `--num_workers` | 2 | 原有参数：构建缓存时使用的线程数，建议 8 |
| `--cache_rate` | 1.0 | 缓存进内存的行的比例 |
| `--use_all_rows` | 关闭 | 使用 CSV 全部行，验证集的行也参与训练（原版代码的实际行为）。默认只用 `split == train` 的行 |
| `--no_roi_cache` | 关闭 | 退回原版的全脑缓存流程（每行约 111 MB），**不要开** |
| `--num_steps` | 200000 | **默认值改变**：原版为 100000，见下方说明 |
| `--allow_data_change` | 关闭 | **[新增]** 允许用与 checkpoint 不同的 CSV 续训（默认拒绝，见下方"断点续训"） |

**其余参数全部保持默认**，与原版相同：EDM，σ_max=50，σ_min=0.002，ρ=7；`--small_lesion_weight 3.0`，`--small_lesion_threshold 27`，`--small_lesion_clamp 1.0`；AdamW，lr 2e-4，weight decay 1e-5，betas (0.5, 0.999)；AMP。含义和调参方法见 README_DIFFUSION 第 4、7、9 节。

> **关于 `--num_steps`**：默认 200,000 步。原版的 100,000 是参照同任务 GAN 定的经验值，README_DIFFUSION 自己也建议首次全量训练设 150,000–200,000 步。本版本的随机偏移每次迭代都重新抽取，样本比原版多变，收敛通常需要更多步。约 8,000 行训练数据时，200k 步约 400 个 epoch。
>
> 训练代码的学习率是常数（没有随 `num_steps` 衰减的调度），也没有使用 EMA。所以 200k 那次训练中途存下的 100k checkpoint，基本就等于单独训练 100k 步的结果，多训几步只多花时间。训练完用 Step 5 对各个 checkpoint 跑指标、画曲线，**挑最好的那个，不要直接用最后一个**。

**checkpoint**：`Checkpoint/brats_1mm/<模态>/weights/diffusion_<step>.pt`。与原版一样，每 10 个 epoch 保存一次，训练结束时再保存一次。每个 epoch 的步数是"训练行数 ÷ 16"，例如 8,000 行时每 500 步一个 epoch，也就是每 5,000 步保存一次；200k 步约 40 个 checkpoint，每个约 55 MB，每个模态约 2 GB。

**监控**：

```bash
# 文本模式（SSH 无图形界面可用），每 10 秒刷新
python scripts/watch_loss.py Checkpoint/brats_1mm t1c --live
# 或查看原始日志
tail -f logs/train_t1c.log
```

> 日志里 `Time:` 只统计 GPU 前向和反向的时间，不包含取数时间。实际速度请用单位时间内推进的步数来估算。

**断点续训**：在原命令后加 `--resume_iter <步数>`。网络架构会自动从 checkpoint 中读取。

- **[新增] 数据绑定**：每个 checkpoint 记录了训练用 CSV 的路径、SHA-256、行数和实际训练行数（`data_identity`）。续训时如果 CSV 内容不同，会打印两边的信息并拒绝启动，需要换一个新的 `--logdir` 从头训练；确实要继续时加 `--allow_data_change`。
- **[新增] AMP 状态**：checkpoint 里保存了混合精度的缩放因子，续训时恢复（日志打印 `AMP scaler restored`）。原版每次续训都从初始值 65536 重新开始。
- **[新增] 原子写入**：checkpoint 先写成 `diffusion_<步数>.pt.tmp`，写完再改名。作业在保存时被杀，也不会留下损坏的 `.pt` 文件。
- **[新增] 建缓存时的自检**：如果 CSV 与标签对不上（例如旧坐标约定的 CSV，或换过 `DataSet/`），建缓存时会报错 `tumour voxels inside its bbox ... rebuild it`，不会带着错位的数据静默训练。

### Step 4: 生成（从 label 生成 4 模态 MRI）

```bash
python src/infer/generate_from_label.py \
    --label_path DataSet/BraTS-MET-00001-000/BraTS-MET-00001-000-seg.nii.gz \
    --diffusion_ckpt_dir Checkpoint/brats_1mm \
    --dataset BRATS_2024 \
    --output_dir ./output \
    --generator_type Unet_NnU \
    --crop_size 64 --merge_dist 16 \
    --noise_schedule edm \
    --sampling_method edm_heun --sampling_steps 18 \
    --modality all --use_compile
```

> 输入标签应来自 `DataSet/`（1 mm）；生成结果也在 nnU-Net 的 1 mm Z-score 空间里，并沿用输入标签的空间坐标（原版写死为单位矩阵）。默认加载步数最大的 checkpoint，用 `--ckpt_step 150000` 可指定某一步。其他参数与 README_DIFFUSION 的 Step 3 相同。

### Step 5: 评估

```bash
python src/infer/evaluate_generation.py \
    --diffusion_ckpt_dir Checkpoint/brats_1mm \
    --csv_path Checkpoint/brats_1mm/brats_1mm.csv \
    --dataset BRATS_2024 \
    --output_dir ./eval_results \
    --generator_type Unet_NnU \
    --crop_size 64 --normalization nnunet --sigma_data 1.0 \
    --noise_schedule edm \
    --sampling_method edm_heun --sampling_steps 18 --use_compile
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--normalization` | `minmax` | 本版本用 **`nnunet`**（不再重新归一化，指标的 data_range 与 zscore 相同，都是 6.0） |
| `--eval_split` | `val` | **[新增]** 只评估 `split == val` 的行（Step 2 指定的验证集）。**CSV 里没有验证集时直接报错**，不会悄悄改用训练行；确实要评估全部行时传 `all`（原版行为） |
| `--evaluation_mode` | `patch` | `patch` / `whole_brain`，含义同 README_DIFFUSION 的 Step 4 |
| `--ckpt_step` | `latest` | **[新增]** 评估哪个 checkpoint：`latest` 为步数最大的一个，或写具体步数，如 `150000` |
| `--seed` | 0 | **[新增]** 采样噪声的随机种子，保证不同 checkpoint 用同一组噪声比较；`-1` 为不固定（原版行为） |

输出 `./eval_results/metrics.json`，包含 MSE、MAE、PSNR、SSIM，以及实际使用的 checkpoint 路径。

> **[新增]** 如果 `--csv_path` 与 checkpoint 训练时用的 CSV 不同，会打印 `[WARNING] ... its val rows may overlap the training data`，因为此时验证集可能和训练数据重叠。
>
> **[修复]** patch 模式按 CSV 坐标裁剪，原版受坐标错轴影响，大部分评估窗口不含肿瘤；用修复后的 CSV 评估才有意义。

**挑选最优 checkpoint**（推荐）：对每个模态按步数评估一串 checkpoint，汇总成表格和曲线，每个模态占一张卡：

```bash
mkdir -p logs
for i in 0 1 2 3; do
  mod=$(echo t1c t1n t2w t2f | cut -d' ' -f$((i+1)))
  CUDA_VISIBLE_DEVICES=$i nohup python scripts/eval_checkpoints.py --modality $mod --every 20000       > logs/eval_$mod.log 2>&1 &
done
wait
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--every` | 20000 | 大约每隔多少步评估一个 checkpoint（取每个整数倍之后的第一个），最后一个总会评估 |
| `--steps` | — | 直接列出要评估的步数，覆盖 `--every` |
| `--max_cases` | 0 | 每个 checkpoint 最多评估多少行验证集（0 = 全部）；想更快时设为 300 左右 |
| `--logdir` / `--out_dir` | `brats_1mm` / `eval_ckpts` | CSV 与权重所在的实验名；结果目录 |

每个 checkpoint 的结果在 `eval_ckpts/<模态>/step_<步数>/metrics.json`，汇总在 `eval_ckpts/<模态>/summary.csv` 和 `summary.png`，终端会打印 PSNR、SSIM 最高的步数。已评估过的步数重跑时自动跳过。`--` 之后的参数会原样传给 `evaluate_generation.py`（例如 `-- --evaluation_mode whole_brain`）。

### Step 6: 交付给 S2 在线增强

需要交付的是 4 个模态各自选定的 checkpoint：`Checkpoint/brats_1mm/{t1c,t1n,t2w,t2f}/weights/diffusion_<step>.pt`。

- **S2 现有的旧扩散模型也是在错轴的 crop 上训练的**（见第 8 节）。新旧模型对比要用修复后的 CSV 和评估方式重新做。
- **本目录的 `on_the_fly_augmentation.py` 不能直接用这批模型做插入增强。** 它和原版 GliGAN 一样，补全采样（`sample_tumour_diffusion_inpaint`）只实现了 DDPM / DDIM，传入 `edm_heun` 会报错；而本版本训练的是 EDM 模型。S2 计划中按 `edm_heun` 18 步调用，说明 S2 服务器上用的是支持 EDM 的修改版，但它不在本仓库中，所以本地无法测试这条路径。交付时只需要 checkpoint，由 S2 端用自己的版本加载；需要在本目录里也能跑插入增强时，要先把 S2 的 EDM 补全实现同步过来。
- **每个模态的 `weights/` 里只放选定的那一个文件。** 原版加载器用 `sorted(glob)[-1]` 取"最后一个"，这是按文件名的字符串排序，例如 `diffusion_98800.pt` 会排在 `diffusion_200000.pt` 之后。本目录的加载器已改为按步数数值选取，并支持 `--ckpt_step`；但 S2 端如果用的是原版加载器，目录里有多个文件时会选错。
- **格式与原版兼容**：checkpoint 的键与原版相同，只额外多了 `planner_gpu_gb`、`normalization`、`modality`、`crop_size`、`generator_type` 几个说明字段，不认识这些字段的加载器会直接忽略。
- **S2 调用方式不变**：S2 按 `zscore` 模式调用，直接把 nnU-Net 预处理后的 64³ 窗口送进模型、不再归一化，这与本版本的训练输入一致；采样设置保持 S2 增强器里的 EDM-Heun 18 步。（本目录的 `on_the_fly_augmentation.py` 也接受 `nnunet` 作为 `zscore` 的别名，但如上所述不能做 EDM 补全。）
- **架构**：S2 端如果在 ≥ 32 GB 的 GPU（A40 / A100）上加载，会得到同样的 48–384 网络，可以直接加载；在更小的卡上加载，需要沿用本目录 `load_diffusion_model` 里读取 `planner_gpu_gb` 的那一处改动。
- **空间轴顺序**：本版本的数组按 nibabel 的体素顺序 (x, y, z) 存储，和原版训练时一样（MONAI `LoadImaged`）；nnU-Net 用 SimpleITK 读出的数组是 (z, y, x)，两者互为转置。S2 接口里那一步"交换模态顺序和空间轴"做的正是这个转换，**请与 S2 接口代码核对一次**。

---

## 3. 资源与耗时估算

以约 1,000 例训练病例、每个进程 `--loader_workers 8`（共约 32 个 CPU 核）为例。4 个模态各占一张卡同时训练，总耗时等于单个模态的耗时。以下除冒烟测试外都是推算值：GPU 部分由本地 RTX 4070 Laptop 实测折算，取数部分用本地 CPU 实测，尚未在服务器上实测。

### 3.1 全流程耗时

| 阶段 | 4×A100 | 4×A40 | 依据 |
|---|---|---|---|
| 冒烟测试（第 2 节开头） | 约 5 分钟 | 同左 | 本地实测 |
| Step 1 预处理（全部 1,296 例，8 进程） | 约 0.5–1 小时；磁盘约 75–80 GB | 同左（只用 CPU） | 本地实测每例 2–16 s、平均约 60 MB |
| Step 2 建 CSV | 约 10–30 分钟 | 同左 | 单进程，逐例做连通域分析和自检 |
| Step 3 建缓存、编译预热（一次性） | 约 10–20 分钟 | 同左 | 每行读取一个 1 mm 体积 |
| Step 3 内存 | 每进程 15–20 GB，合计 60–70 GB | 同左 | 实测平均 2.0 MB/行 × 约 8,000 行 |
| Step 3 每步耗时（batch 16） | 约 0.025–0.04 s | 约 0.035–0.05 s | GPU 部分：本地实测 0.067–0.083 s，A100 按快 3–4 倍、A40 按快 2–3 倍折算。取数单线程约 0.2–0.26 s/步，8 个 worker 并行后约 0.025–0.04 s；A100 上瓶颈在 CPU 取数 |
| **训练 200k 步（默认）** | **约 1.5–2 小时** | **约 2–3 小时** | — |
| 训练 100k 步 | 约 0.7–1 小时 | 约 1–1.5 小时 | — |
| Step 5 挑最优步数（每 2 万步评估一个，约 10 个 checkpoint） | 约 0.5–1 小时 | 约 1–1.5 小时 | 每个 checkpoint 评估全部验证行约 4–7 分钟（每行 EDM-Heun 18 步 = 35 次前向，加读取）；加 `--max_cases 300` 约减半 |
| Step 4 生成 | 每例几秒 | 同左 | — |
| **合计** | **约 3–4.5 小时** | **约 4–6 小时** | 原版训练 10 万步就要两三天，原因见 3.2 |
| 不开 `--loader_workers` | 每步约 0.2–0.3 s，200k 步需 11–17 小时 | 同左 | 取数成为瓶颈 |

> 上机后先跑约 2,000 步，用"步数 ÷ 实际经过的时间"核对真实速度（日志里的 `Time:` 不含取数时间，不能用来估算）。如果明显慢于上表，多半是 CPU 核数不够或 `LOADER_WORKERS` 没生效，可以用 `top` 看每个训练进程是否带起了 8 个子进程；CPU 核多时可以设 `LOADER_WORKERS=12` 或 16。

### 3.2 原版为什么要两三天，嫁接改进后会怎样

原版和本版本的网络、每步的 GPU 计算量完全相同（同样的 64³ crop，约 0.03–0.07 s/步），慢在数据管线。这是一条因果链：

1. **根因：全脑体积一路带进了 batch。** 原版的裁剪变换把整个全脑影像写回了样本（`d[scan_key] = scan_data`），`ToTensord` 的键里也包含全脑影像和全脑标签，于是每一步 `pad_list_data_collate` 都要把 16 个全脑体积 pad 到这一批里最大的尺寸再拼接，而训练只用其中的 64³ crop。原始数据尺寸差异很大（最大 640×640×392），pad 后的 batch 中位约 14 GB。本地实测：16 个尺寸相同的 1 mm 全脑拼一次 0.1 s；尺寸只是略有不同时 1.1 s；原始数据的实际情况估计每次好几秒。
2. **强制 `num_workers=0`。** 上面的拼接只能在主进程里串行做，GPU 一直在等。原版代码注释写的理由是：开子进程后变换重新执行，出现"空 crop 崩溃"。这个症状与坐标错轴（第 8 节）吻合：x、z 坐标套到相反的轴上，窗口很容易越界或落空。当时没找到真正的原因，只好关掉子进程、靠缓存冻结来规避（未复现确认）。
3. **缓存太大。** 每行至少缓存 111 MB，原始分辨率的大病例单行接近 2 GB，全部缓存需要 TB 级内存；内存不够时会用到 swap，或只能调低缓存比例改从磁盘读。
4. **日志掩盖了问题。** `Time:` 在拿到 batch 之后才开始计时，只统计 GPU 时间，显示每步零点几秒，所以一直没人发现大头在取数据。

按每步约 2 s 算，10 万步约 55 小时，正好对应两三天。

**把本版本的改进嫁接回原版代码**（仍用原始分辨率数据；A100、batch 16 推算）：

| 嫁接组合 | 每步耗时 | 200k 步 | 说明 |
|---|---|---|---|
| 原版，不改 | 约 2 s 以上 | 约 4–5 天 | 由 10 万步两三天推算 |
| 只加 ROI 缓存 | 约 0.02–0.03 s | 约 1–1.5 小时 | 很快，但随机偏移仍被冻结，增强等于没做 |
| ROI 缓存 + 每次重新抽偏移，不开取数进程 | 约 0.2–0.26 s | 约 11–15 小时 | 每个样本 10–14 ms 的 CPU 计算成为瓶颈 |
| **全部嫁接**（ROI 缓存 + 重新抽偏移 + 8 个取数进程） | **约 0.025–0.04 s** | **约 1.5–2 小时** | 与本版本相同 |

- **全部嫁接后为什么一样快**：缓存的 ROI（约 80³）和裁出的 crop（64³）都按体素数计算，与体素间距无关，所以每行缓存（约 2 MB）、每个样本的 CPU 开销和 GPU 计算量都相同。
- **原始数据的差别只在一次性开销**：建缓存时每行都要读并解压该病例的整个原始体积（平均约为 1 mm 裁剪后的 10 倍），估计 0.5–1 小时，而本版本是 10–20 分钟，且每个读取线程临时占用约 1 GB 内存；u 组 0.5 mm 病例会多切出一些瓦片，每个 epoch 稍长；但省掉了 Step 1 的 0.5–1 小时和 75–80 GB 磁盘。总耗时与本版本大致相当，速度不是两者之间做选择的理由，区别见 4.5。
- **要点**：让原版变快的关键只有 **ROI 缓存**和**取数进程并行**两项；前者消除了每步拼接全脑体积，也让缓存小到可以开子进程。"每次重新抽偏移"本身会让每步变慢，它修的是"增强没生效"，必须配合取数进程才能抵消。坐标错轴与速度没有直接关系，但很可能间接导致了 `num_workers=0`。

---

## 4. 与原版的对照

### 4.1 保持不变

| 项目 | 内容 |
|---|---|
| 网络 | `TimeConditionedUNet_NnU`（MONAI UNet），≥ 32 GB 规划下 channels (48, 96, 192, 384)、strides (2, 2, 2, 1)；输入 13 通道 = 1 加噪影像 + 4 标签 + 8 时间嵌入 |
| 扩散框架 | `model.py` 原样复制：EDM 预条件、λ(σ) 加权 MSE、log-normal σ 采样、Heun 采样 |
| 损失加权 | 小病灶加权 `1 + w·clamp(T/N, 0, c)` 和患者级均衡，公式与默认值都不变 |
| 训练超参 | batch 16、AdamW（lr 2e-4、weight decay 1e-5、betas (0.5, 0.999)）、AMP、常数学习率，与原版训练代码一致（原版代码中没有梯度裁剪，见第 7 节）；只有默认步数改为 200k，见 4.2 |
| 样本构造 | 连通域分析、邻近病灶合并、64³ 预切片、窗口与填充公式、自适应偏移幅度、肿瘤区加噪、`effective_n_voxels` 的定义（只修正了 CSV 的坐标轴顺序，见第 8 节） |
| 输出 | checkpoint 的原有键名（只新增字段，见 Step 6）、保存间隔、loss 日志格式；生成和评估脚本的采样流程 |

### 4.2 修改内容及原因

共约 30 处，按类别列出。标 ★ 的在 [4.3 节](#43-主要改动详解) 展开；标 §8 的是审查反馈引出的修复，原因和在真实数据上的影响统计见 [第 8 节](#8-修复记录2026-10-12审查反馈)。

**数据预处理（新增）**

| 修改 | 文件 | 原因 | 效果 |
|---|---|---|---|
| ★① 重采样到 1 mm 并做整例带掩膜 Z-score | `scripts/preprocess_dataset.py`、`src/preprocess/nnunet_like.py` | S2 推理时扩散模型看到的是 nnU-Net 的 1 mm 数组。原版直接在原始网格上训练，71% 的病例不是 1 mm：病灶在网络输入中被拉伸（最大 11.6 倍）；64³ 窗口的物理范围在 23–75 mm × 32–320 mm 之间变化；按体素数算的小病灶权重有 11.7% 的病灶判错；236 个病灶被不必要地切片 | 训练输入与推理输入一致，"27 体素"严格等于 27 mm³ |
| §8 数据检查 | 同上 | 原版不检查模态错位、NaN 和非法标签 | 不合格病例被拒绝并写进报告；真实数据中只有含标签 6、8 的 2 例被拒 |
| §8 文件头写精确的目标间距 | 同上 | 取整后的真实间距与 1 mm 差千分之几 | 与 nnU-Net 的约定一致，真实间距记在报告的 `true_spacing` 列 |
| 输出原子写入、可断点续跑 | 同上 | — | 中断不会留下半个文件；重跑时跳过已完成的病例 |

**样本构造与数据管线（修改）**

| 修改 | 文件 | 原因 | 效果 |
|---|---|---|---|
| §8 **修正病灶坐标错轴**，建 CSV 和建缓存时各自检一次 | `csv_creator.py`、`gaussian_noise_tumour.py` | 原版 CSV 的 x/z 列被套到了相反的轴上，约 80% 的训练 crop 不含目标病灶 | 每个 crop 都包含它的病灶；旧 CSV 在建缓存时直接报错 |
| ★② `--normalization nnunet` | `gaussian_noise_tumour.py`、训练、评估、在线增强脚本 | 原版对每个 crop 单独做 Z-score，推理时却是整例 Z-score，病灶强度中位相差约 0.2 个标准差 | 训练强度与推理强度一致 |
| ★③ 缓存只存病灶 ROI（`CropLesionROId`） | `gaussian_noise_tumour.py`、`data_utils.py` | 原版每行缓存整张全脑体积（111 MB/行；全部病例需数 TB 内存），而且每步要拼接 16 个全脑体积（3.2 节） | 内存降到约 2 MB/行；裁剪公式与原版逐体素相同 |
| ★④ 随机偏移每次迭代重新抽取（`Randomizable`） | `gaussian_noise_tumour.py` | 原版的变换没有被 MONAI 识别为随机变换，偏移随缓存一起被冻结，与 README_DIFFUSION 写的"每次迭代独立抽签"不符 | 恢复了原本设计的随机偏移（同时输出的肿瘤区加噪是 GAN 遗留字段，扩散 loss 不用） |
| ★⑤ 验证集的行不参与训练 | `csv_creator.py`、`data_utils.py` | 原版 `csv_creator` 会写 `split` 列，但训练代码忽略它 | 验证集仍由使用者指定，指定后确实不参与训练；不指定时与原版相同 |
| 可选：t2w 模型排除指定病例 | 同上 | 部分病例的 T2W 不含与其他模态对应的信息 | 默认关闭，传 `--t2w_invalid_list` 才生效 |
| §8 建 CSV 时检查标签 | `csv_creator.py` | 原版把标签 6、8 当成肿瘤写进 CSV，但 4 通道条件里是空的 | 含非法标签的病例被跳过并警告 |
| §8 训练 CSV 按固定规则查找 | `data_utils.py`、`tumour_main_diffusion.py` | 原版取文件夹里 `os.listdir` 列出的第一个 `.csv` | 优先 `<logdir>.csv`；有歧义时报错 |
| 取数并行与等价提速 | `gaussian_noise_tumour.py`、`data_utils.py` | 原版强制 `num_workers=0`；`effective_n_voxels` 对整个体积做连通域标记 | 新增 `--loader_workers`；`effective_n_voxels` 只在肿瘤包围盒内计算（300 个 crop 上结果完全一致，快 2.3 倍）；batch 只传必要字段 |
| 跳过数据目录中的非病例文件 | `csv_creator.py` | `DataSet/` 下多了预处理报告文件 | — |

**训练（修改）**

| 修改 | 文件 | 原因 | 效果 |
|---|---|---|---|
| ★⑥ 固定网络宽度（`--planner_gpu_gb`），写进 checkpoint | `tumour_main_diffusion.py`、`DiffusionNetwork.py`、`diffusion_inference_utils.py`、`evaluate_generation.py` | 原版按当前 GPU 显存决定网络宽度，换卡就是另一个网络 | 任何 GPU 上都得到相同的参考架构 |
| 默认训练步数 100k → 200k | `tumour_main_diffusion.py`、`train_4gpu.sh` | 原版的 100k 未经调参，README_DIFFUSION 建议首次训练 150k–200k；随机偏移恢复后样本更多变 | 常数学习率下，中途的 100k checkpoint 与单独训练 100k 基本相同，最终用 Step 5 挑最优 |
| §8 checkpoint 原子保存、保存 AMP 状态、绑定训练数据 | `tumour_main_diffusion.py`、`data_utils.py` | 原版中断会留下坏文件、续训时缩放因子重置、换了数据也不会察觉 | 续训恢复 AMP 状态；换 CSV 续训被拒绝（`--allow_data_change` 可强制） |
| loss 曲线图保存失败时只警告 | `tumour_main_diffusion.py` | 文件被其他程序占用时（Windows 上实际遇到）原版会直接中断训练 | — |

**推理与评估（修改）**

| 修改 | 文件 | 原因 | 效果 |
|---|---|---|---|
| checkpoint 按步数数值选取，新增 `--ckpt_step` | `diffusion_inference_utils.py`、`generate_from_label.py`、`evaluate_generation.py`、`on_the_fly_augmentation.py` | 原版 `sorted(glob)[-1]` 是字符串排序：本地 50–230 步的 checkpoint 选中的是 `diffusion_50.pt`；200k 步时会选中 98,800 步。原版也无法指定某一步 | 默认取真正的最新一步，也可以指定任意一步 |
| 评估默认只用验证集 | `evaluate_generation.py` | 原版评估 CSV 中的所有行，包括训练行 | §8 没有验证集时报错（传 `--eval_split all` 才评估全部行）；CSV 与训练时不同则警告 |
| 评估固定随机种子 `--seed` | `evaluate_generation.py` | 原版每次评估的采样噪声不同，比较 checkpoint 时混入随机波动 | 同一 checkpoint 重复评估结果完全相同 |
| whole_brain 评估去掉 `_orig_mod.` 前缀 | `evaluate_generation.py` | 开 `--use_compile` 训练的权重在 whole_brain 模式下加载失败 | 可以正常评估 |
| 生成结果沿用标签的空间坐标 | `generate_from_label.py` | 原版保存时坐标写死为单位矩阵 | 生成结果与标签在查看器里对齐 |
| 新增 `scripts/eval_checkpoints.py` | — | 原版没有按指标曲线挑最优步数的工具 | 自动评估一串 checkpoint，输出表格和曲线 |

**脚本与文档（新增）**：`train_4gpu.sh`（4 卡并行训练）、`smoke_test.sh`（冒烟测试）、`make_case_lists.py`（从 S2 划分导出名单）、`requirements.txt`、本 README；修复了 `watch_loss.py` 文本曲线报错。

**已知未处理**：本目录的 `on_the_fly_augmentation.py` 不支持 EDM 补全（见 Step 6）；没有 Slurm 脚本；`label_main_diffusion.py` 未做 1 mm 适配；未复制 Flow Matching 分支。

### 4.3 主要改动详解

#### ① 预处理到 nnU-Net 1 mm 空间

- **问题**：S2 在线增强时，扩散模型拿到的是 nnU-Net 预处理后的 64³ 窗口（1 mm 网格、整例 Z-score），原版却直接在原始数据上训练：68 种体素间距，71% 不是 1 mm。由此产生 4.2 表中列出的病灶拉伸、视野不一、小病灶权重误判和多余切片等偏差。
- **改法**：`preprocess_dataset.py` 按 nnU-Net v2 的顺序处理：裁剪 → 在原始分辨率上做 Z-score（nnU-Net 源码规定归一化须在重采样之前）→ 重采样到 1 mm（细节见 Step 1）。输出目录结构与原版 `DataSet/` 相同，下游读取逻辑不用改。有三种后端：装了 nnunetv2 时直接调用其函数，与 S2 逐位一致；否则依次退回 scikit-image、scipy 的复刻实现，数值差异极小。正式运行请用 nnunetv2。
- **验证**：5 类间距（含走 z 轴单独插值分支的 0.43×0.43×5.0 mm）的输出均为 1 mm，脑内均值 0、标准差约 0.9–1.0，标签只有 0–4，影像与病灶轮廓对齐。

#### ② `--normalization nnunet`

- **问题**：原版 `zscore` 模式对每个 64³ crop 单独做 Z-score，统计范围是整个 crop，连背景 0 也算在内；S2 推理时窗口里的值是整例带掩膜 Z-score，不会再归一化。同一块组织在两种口径下数值不同：换算到整例口径后，crop 均值的 5–95% 区间为 −0.35 到 +0.51，标准差在 0.68 到 1.20 之间，病灶强度中位相差约 0.2 个标准差。推理是 inpainting，周围已知区域用的是全局口径，模型学的却是局部口径，生成病灶的明暗可能和周围对不上。
- **改法**：数据在 Step 1 已按整例归一化，crop 原样使用，padding 和背景为 0。S2 按 `zscore` 调用时直接送入窗口，与此一致；评估和在线增强脚本也支持 `nnunet`。`sigma_data` 保持 1.0，因为 1 mm 数据中病灶 crop 的实际标准差约 0.9，与 1.0 接近。
- **验证**：测试中 crop 均值 −0.05、标准差 0.70。如果是逐 crop 归一化，这两个值会恰好是 0 和 1。

#### ③ 缓存只存病灶周围的 ROI

- **问题**：MONAI `CacheDataset` 会把变换的输出整个存进内存。原版的输出里有全脑影像（float32）、全脑 4 通道标签，还有一份从全脑副本上切出来的视图，会拖住整块内存不释放。实测每行 111 MB，全部病例需数 TB。每个 batch 还要把 16 个全脑体积 pad 到同一尺寸再拼接，中位约 14 GB。
- **改法**：新增确定性变换 `CropLesionROId`，放在缓存边界之前：
  1. 用原版公式算出不加偏移时的窗口；
  2. 每边再扩 8 体素（随机偏移的最大幅度），并限制在体积范围内；
  3. 只缓存这一块：影像存 float32，标签存单通道 uint8，同时记下这一块的起点。

  后续裁剪仍按原公式在全脑坐标里算窗口，再减去起点，到这一块里取数，所以 padding、超大病灶的缩放都与原版的公式逐体素一致。注意：原版公式本身是对的，错的是原版 CSV 的坐标轴顺序，已在第 8 节修正。多通道标签的转换挪到裁剪之后；它是逐体素运算，先裁后转和先转后裁结果相同。
- **效果**：实测平均 2.0 MB/行（最大 2.6 MB），每个进程 15–20 GB；batch 里只有 64³ crop 和少量标量。

```
原版    读取全脑 → 多通道标签 → 偏移+裁剪+逐 crop Z-score  ┃ 组 batch（含全脑体积）
1mm 版  读取 1 mm 数据 → 病灶 ROI（≤ 80³）                 ┃ 偏移+裁剪（不再归一化）→ 组 batch（仅 64³）
        ┃ = 缓存边界：左侧只在建缓存时算一次并存进内存，右侧每次迭代重新计算
```

#### ④ 随机偏移每次迭代重新抽取

- **问题**：`CacheDataset` 一直缓存到第一个 `Randomizable` 变换为止。原版 `GaussianNoiseTumour` 只继承了 `MapTransform`，内部直接用 `np.random`，被 MONAI 当成确定性变换，连同结果一起缓存。于是每个病灶的偏移只在建缓存时抽一次，此后每个 epoch 都是同一个位置。README_DIFFUSION 所说的"每次迭代随机偏移 crop 中心"实际上没有生效。
- **改法**：改为继承 `Randomizable`，随机数改用 MONAI 管理的 `self.R`，放到缓存边界之后（见上图）。`effective_n_voxels` 按窗口中心所在的连通域计算，会随窗口一起重算。多个取数进程时，MONAI 为每个进程设置不同的随机种子。
- **说明**：这个变换同时输出一个肿瘤区加噪的 `scan_noisy`，它是 GAN 版本遗留的字段，扩散 loss 不使用（EDM 的噪声在 loss 内部采样）。所以原版的噪声被冻结对训练没有影响，实际起作用的修复是**偏移**。
- **代价**：随机部分每个样本在 CPU 上需 10–14 ms，须开 `--loader_workers`（见第 3 节）。其中最慢的 `effective_n_voxels` 已换成等价算法：只在肿瘤包围盒里标记连通域，质心用 `bincount` 一次算完。在 300 个随机 crop 上结果与原算法完全一致，速度快 2.3 倍。
- **验证**：同一行连续取 20 次，得到 20 个不同的窗口中心。

#### ⑤ 训练集、验证集由使用者指定

- **原版**：`csv_creator` 用 `--val_patients` 把指定病例标为 `split=val`，但训练代码读 CSV 时不看 `split` 列，验证集的行同样参与训练。
- **本版本**：划分方式不变，仍由使用者指定验证集（`--val_patients`，或从 txt 读取的 `--val_list`），`DataSet/` 中的其余病例全部是训练集，不按任何名单跳过病例。唯一的区别是训练时真正排除了 `split == val` 的行；不指定验证集时，与原版完全相同。需要让验证集也参与训练时，加 `--use_all_rows`。
- **可选**：S2 中有 212 例训练病例的 T2W 不含有效信息（抽样中病灶区信号中位数只有 +0.18，完整病例为 +0.51）。想让 t2w 模型排除它们时，在 Step 2 传 `--t2w_invalid_list`；默认不排除。

#### ⑥ 固定网络宽度（`--planner_gpu_gb`）

- **问题**：`TimeConditionedUNet_NnU` 构建时按**当前 GPU 显存**选择 channels：≥ 32 GB 为 48–384，16–32 GB 为 32–256，< 16 GB 为 24–192。训练和加载 checkpoint 时各定一次，网络结构随硬件变化：在小卡上训练会得到更窄的网络；在 A40 上训练的权重拿到小卡上加载，会报 state_dict 不匹配（本地评估时实际遇到过）。
- **改法**：`--planner_gpu_gb`（`train_4gpu.sh` 固定为 48）写进 checkpoint；`load_diffusion_model`、评估脚本和断点续训都先读出这个值再建网络；规划值小于 32 时打印警告。网络类本身没有改，只是把它原有的 `gpu_memory_gb` 参数从外部传入。
- **验证**：在本地 8 GB 卡上训练得到 48–384 的网络（约 448 万参数），并能在同一张卡上加载和续训。S2 在 A40 / A100 上加载时得到的本来就是这个网络，不受影响。

### 4.4 兼容性

使用原版的参数组合（`--normalization zscore/minmax`、`--use_all_rows`、`--no_roi_cache`）时，训练流程回到原版，但坐标错轴已修正、随机偏移每次迭代重新抽取，且 CSV 必须用本目录的 `csv_creator.py` 重建。

### 4.5 原版嫁接这些改进后，与本方法还有什么不同

上面的改进大多与训练数据所在的空间无关，可以原样搬回原版代码：坐标错轴修正、ROI 缓存、随机偏移、验证集处理、数据检查、固定网络宽度、checkpoint 相关修复、评估修复、200k 步。嫁接之后，与本方法**只剩一个区别：训练数据所在的空间**。

| | 原版 + 全部可嫁接的改进 | 本方法 |
|---|---|---|
| 训练用的体素网格 | 原始采集分辨率（68 种间距，71% 不是 1 mm） | 统一 1×1×1 mm |
| 强度归一化 | 每个 crop 单独做 Z-score（连背景也算） | 整例带掩膜 Z-score，与 S2 推理时的输入完全相同 |
| 网络、损失、采样、超参 | 相同 | 相同 |
| 训练速度 | 相同；建缓存更慢，但省掉 Step 1（见 3.2） | — |

S2 推理时给扩散模型的是 1 mm、整例 Z-score 的窗口，所以前者仍然存在 4.2 表中 ①② 两行列出的训练与推理不一致：病灶拉伸、窗口物理范围不一、小病灶权重误判、多余切片和强度口径偏差。

- **归一化不能单独嫁接。** `nnunet` 模式要求数据事先已按整例归一化，所以至少要加一步整例 Z-score 预处理，没法只改一个参数。
- **这正好构成一组干净的对照实验。** 目前还没有证据表明 1 mm 版本生成得更好；"原版 + 可嫁接改进"与本方法只差数据空间这一个变量，用同一套修正后的评估比较，就能单独看出 1 mm 统一带来多少收益。旧模型同时受坐标错轴等问题影响，直接拿它对比说明不了问题。

---

## 5. 文件结构

```
TumourDiffusion_1mm/
├── README.md                          # 本文件（1 mm 版主文档）
├── README_DIFFUSION.md                # 原版文档（参数与调参说明，仍然适用）
├── requirements.txt                   # [新增]
├── model.py                           # 扩散数学（loss / 采样 / schedule），原样复制
├── lists/
│   └── t2w_invalid_cases.txt          # [新增，可选] T2W 无效病例（266 例，需核对）
├── scripts/
│   ├── make_case_lists.py             # [新增，可选] Step 0：从 S2 split 导出名单
│   ├── preprocess_dataset.py          # [新增] Step 1：nnU-Net 1 mm 预处理、数据检查
│   ├── train_4gpu.sh                  # [新增] Step 3：4 卡并行训练
│   ├── eval_checkpoints.py            # [新增] Step 5：评估一串 checkpoint，挑最优步数
│   ├── smoke_test.sh                  # [新增] 冒烟测试：几分钟跑通整条流程并自检
│   └── watch_loss.py                  # [修改] loss 监控（修复文本曲线报错）
├── src/
│   ├── preprocess/nnunet_like.py      # [新增] nnU-Net v2 裁剪 / Z-score / 重采样
│   ├── train/
│   │   ├── csv_creator.py             # [修改] 坐标轴修正与自检、标签检查、--val_list、可选 t2w_valid 列
│   │   ├── tumour_main_diffusion.py   # [修改] 新参数、规划值写入 checkpoint、原子保存、AMP 状态、数据绑定
│   │   └── label_main_diffusion.py    # 标签扩散模型，原样复制，未做 1 mm 适配
│   ├── utils/
│   │   ├── gaussian_noise_tumour.py   # [修改] ROI 缓存、Randomizable、nnunet 归一化、CSV 坐标自检
│   │   ├── data_utils.py              # [修改] 行筛选、ROI 流程、缓存与 worker 参数
│   │   └── ...                        # 其余原样复制
│   ├── networks/DiffusionNetwork.py   # [修改] 仅支持固定规划值，网络结构不变
│   └── infer/
│       ├── diffusion_inference_utils.py  # [修改] 加载时读取规划值、find_checkpoint
│       ├── evaluate_generation.py        # [修改] nnunet 归一化、eval_split、--ckpt_step、--seed
│       ├── generate_from_label.py        # [修改] --ckpt_step、保留标签坐标
│       └── on_the_fly_augmentation.py    # [修改] nnunet 作为 zscore 的别名、按步数选 checkpoint
├── docs/                              # 原版的方法说明文档
├── DataSet/                           # Step 1 输出（初始为空）
└── Checkpoint/                        # CSV、权重、loss 日志（初始为空）
```

---

## 6. 已验证内容（本地）

本目录的**最终代码**已用 `scripts/smoke_test.sh` 在本地跑通全流程（见第一条）；其余各条是针对单项改动做的专项检查。

- **冒烟测试与 4 卡脚本**：`scripts/smoke_test.sh` 在本地完整跑通（8 步全部通过，27 行 CSV × 3 次随机偏移，每个 crop 都包含 100% 的病灶）；`train_4gpu.sh` 在本地单卡上同时启动 4 个模态各训练 40 步，均正常结束并保存 checkpoint（本地为 Windows，`torch.compile` 不可用，测试时去掉了该参数）。
- **预处理**：覆盖 5 类间距——1 mm 各向同性、0.86×0.86×1.5 mm、0.49×0.49×1.0 mm、0.43×0.43×5.0 mm（走 z 轴单独插值分支），以及 `t2w_invalid_cases.txt` 名单中的一例（00570）。输出均为 1 mm，脑内近似 N(0,1)，影像与标签对齐，无病灶丢失。
- **CSV**：`--val_list`（完整名称）和 `--val_patients`（patient ID）都能正确标出 `split=val`，其余病例全部为 `train`，没有病例被跳过；`--t2w_invalid_list` 只影响 `t2w_valid` 列。
- **数据流程**：训练时排除 `split == val` 的行；传了 `--t2w_invalid_list` 时 t2w 额外去掉对应病例；缓存平均 2.0 MB/行；同一行连续取 20 次得到 20 个不同的窗口中心；crop 保留 nnU-Net 归一化后的原值。
- **`effective_n_voxels`**：新旧算法在 300 个随机 crop 上结果完全一致。
- **训练**：t1c（单进程取数）和 t2w（2 个取数进程）各训练若干步；在 8 GB 卡上用 `--planner_gpu_gb 48` 得到参考架构；checkpoint 能被 `load_diffusion_model` 和评估脚本按记录的架构正确加载；断点续训正常。
- **评估**：`--normalization nnunet --eval_split val` 全流程跑通。
- **第 8 节的修复**：
  - 坐标轴：真实病例的全部 21 行、每行 3 次随机偏移，经训练用的真实 transform 后，crop 都包含 100% 的病灶体素（含超大病灶的 5 个瓦片）；把 x/z 列换回旧约定后，建缓存时报错；
  - 预处理：标签 6（真实病例 01094-002）、人为构造的标签 9、t1c 含 NaN、t2f 坐标偏移 10 mm 四例全部被拒绝；正常病例通过，文件头间距为精确的 1.0 mm，无临时文件残留；
  - 训练：checkpoint 含 AMP 状态和数据指纹；同一 CSV 续训时恢复缩放因子；换 CSV 续训被拒绝，加 `--allow_data_change` 后可继续；
  - 评估：没有验证集时报错；加 `--eval_split all` 可评估全部行；CSV 与训练时不同时打印警告。
- **生成与 checkpoint 选择**（3 个真实病例，训练到 230 步，保存了 50/100/150/200/230 五个 checkpoint）：
  - 原版规则选中 `diffusion_50.pt`，新规则选中 `diffusion_230.pt`；`--ckpt_step 100` 能取到指定一步；
  - Step 4 生成跑通，输出与标签尺寸、坐标一致；
  - patch 评估在固定种子下重复两次结果完全相同；whole_brain 评估能加载带 `_orig_mod.` 前缀的权重；
  - `eval_checkpoints.py` 正确选步、汇总出 `summary.csv` 和曲线；本目录的 `OnTheFlyTumourAugmenter` 能按步数选取并加载 checkpoint（其 EDM 补全的限制见 Step 6）；
  - `watch_loss.py` 能正常画出文本曲线。

---

## 7. 常见问题与注意事项

**nnunetv2 的版本**：重采样的细节（插值阶数、z 轴单独处理的阈值、标签插值方式）以 S2 实际使用的 nnU-Net 为准。请在 S2 环境中运行 Step 1，并确认启动时打印的是 `Backend: nnunetv2`。目标间距须与 S2 的 Dataset264 plans 一致，即 `[1, 1, 1]`。

**极小病灶**：u 组 0.5 mm 病例中，体积小于 1 mm³ 的病灶在重采样到 1 mm 后可能消失（见 `preprocess_report.csv` 的 `lesions_lost` 列）。S2 的 nnU-Net 和在线增强都在同一个 1 mm 空间里工作，分割端本来就看不到这些病灶；在线增强的供体池也排除了小于 27 mm³ 的病灶。

**显存小于 32 GB 的卡**：原版的规划器会给出更窄的网络（16–32 GB 为 channels 32–256，小于 16 GB 为 24–192）。本版本必须加 `--planner_gpu_gb 48`；脚本 `train_4gpu.sh` 已经默认带上。如果没加，训练开始时会打印 `[WARNING] planner_gpu_gb=...`。

**`--loader_workers` 与平台**：Linux 下 worker 以 fork 方式共享缓存，内存一般不会成倍增加。Windows 下以 spawn 方式启动，每个 worker 都要复制一份缓存，所以 Windows 上建议设为 0 或 2。

**`--patient_balance_mode`**：README_DIFFUSION 的 Step 2 默认命令没有这个参数（默认值为 `none`），但其 7.5 节"推荐配置"和 `docs/diffusion_model_for_brain_tumor_generation.md` 的表 3 写的是 `sqrt`。`train_4gpu.sh` 跟随 Step 2 的默认命令。请先确认 S2 现有 checkpoint 用的是哪一种，需要 `sqrt` 时运行 `EXTRA_ARGS="--patient_balance_mode sqrt" bash scripts/train_4gpu.sh`。

**文档与代码的两处出入**：`docs/diffusion_model_for_brain_tumor_generation.md` 写的网络是"5 个阶段、约 2,700 万参数、梯度裁剪最大范数 12"；而原版代码在 ≥ 32 GB 的卡上实际是 4 个阶段（48/96/192/384）、约 450 万参数，训练循环里也没有梯度裁剪。本版本以原版**代码**为准，没有改动这两处。

**`sigma_data`**：沿用 README_DIFFUSION 规定的 `1.0`（Z-score 数据）。1 mm 数据中病灶 crop 的实际标准差约为 0.9，与 1.0 接近，因此不做改动，以保持与原版对齐。

**标签扩散模型**：`label_main_diffusion.py` 原样复制，没有做 1 mm 适配，也没有加 split 过滤。如果要用 1 mm 数据训练它，需要自己处理训练集和验证集的划分。

**Flow Matching**：README_DIFFUSION 第 10 节的 flow 分支没有复制到本目录。

---

## 8. 修复记录（2026-10-12，审查反馈）

第三方审查指出了以下问题，均已核实并修复（Slurm 运行脚本暂未提供）。"影响"一栏是在 1,296 例真实数据上统计的。

| # | 问题 | 原因 | 影响 | 修改 | 文件 |
|---|---|---|---|---|---|
| 1 | **病灶坐标错轴**（原版遗留） | nibabel 和 MONAI 的数组都是 (x, y, z)，训练时按 `[:, x, y, z]` 裁剪；原版 `csv_creator` 却把第 0 轴当 z、第 2 轴当 x 写进 CSV | 随机抽 120 例、604 个窗口：80% 不含目标病灶，76% 一点肿瘤都没有，只有 12% 完整包含病灶。模型主要在学生成正常组织，小病灶加权也算错了对象；patch 评估同样错位。**S2 现有的旧模型也是这样训练的** | 连通域分析和超大病灶切片改为 (x, y, z) 顺序；建 CSV 时逐行自检；建缓存时再检查一次，旧 CSV 会直接报错 | `csv_creator.py`、`gaussian_noise_tumour.py` |
| 2a | 输出间距不是精确的 1 mm | 新尺寸按 nnU-Net 的规则取整，文件头里记录的是取整后的真实间距 | 数组与 nnU-Net 完全相同，训练不读文件头，实际无影响 | 文件头写精确的目标间距（nnU-Net 的约定），真实间距记在报告的 `true_spacing` 列 | `preprocess_dataset.py` |
| 2b | 模态之间错位不检查 | 预处理只比较了尺寸 | 真实数据中没有发生（最大偏差 0.000005 mm） | 比较 4 个模态与标签的 affine，超过 `--affine_tol_mm`（默认 0.1 mm）即拒绝 | `preprocess_dataset.py` |
| 3a | 非法标签、NaN 不拒绝 | 预处理和建 CSV 都没有检查 | 2 例含标签 6、8；这些区域会成为"条件标签为空"的肿瘤样本。NaN 在真实数据中未发现，但一旦出现，整个模态会变成 NaN，训练会悄悄停止学习 | 预处理拒绝非整数、NaN/Inf 和 `--allowed_labels` 之外的标签；建 CSV 时跳过含非法标签的病例 | `preprocess_dataset.py`、`csv_creator.py` |
| 3b | 没有验证集时评估回退到训练集 | 本目录早期版本在 CSV 无验证集时改为评估全部行（原版则总是评估全部行） | 指标虚高，`eval_checkpoints.py` 会挑出过拟合的步数 | 改为报错，必须显式传 `--eval_split all`；CSV 与训练时不同则警告 | `evaluate_generation.py` |
| 4a | checkpoint 不是原子写入 | `torch.save` 直接写最终文件名 | 保存时被杀会留下损坏文件，而"取最新"恰好选中它 | 先写 `.tmp` 再 `os.replace`；预处理输出同样处理 | `tumour_main_diffusion.py`、`preprocess_dataset.py` |
| 4b | 续训不恢复 AMP 状态 | checkpoint 没有保存 GradScaler | 续训后头几步可能溢出被跳过 | 保存并恢复 `scaler` | `tumour_main_diffusion.py` |
| 4c | checkpoint 没有与数据绑定 | checkpoint 没有记录训练数据 | 换了 CSV 续训或评估都不会察觉，例如重建 CSV 后接着旧 checkpoint 训练 | 记录 `data_identity`（CSV 路径、SHA-256、行数、训练行数）；续训不一致时拒绝（`--allow_data_change` 可强制继续）；评估时不一致则警告 | `tumour_main_diffusion.py`、`data_utils.py`、`evaluate_generation.py` |
| 4d | 没有 Slurm 运行脚本 | — | — | **暂未处理**，需要集群的分区、账号、GPU 写法、时限等信息 | — |
| 5 | 训练可能读到错误的 CSV（自查发现） | 原版取 `Checkpoint/<logdir>/` 下 `os.listdir` 列出的第一个 `.csv`，顺序不固定 | 文件夹里留有备份 CSV 时，可能悄悄用错数据 | 优先使用 `<logdir>.csv`；没有它而又有多个 CSV 时报错 | `data_utils.py`、`tumour_main_diffusion.py` |
| 6 | 包内的在线增强器不支持 EDM 补全（自查发现，未修改） | 补全采样只实现了 DDPM / DDIM，原版也是如此 | 不能用本目录的 `on_the_fly_augmentation.py` 对这批 EDM 模型做插入增强；S2 用的是服务器上的修改版 | 见 Step 6 的说明；需要 S2 的 EDM 补全代码才能在本地测试这条路径 | — |

**用过原版 GliGAN 或本目录早期版本的，需要重新做的步骤**（建议先跑一遍 `scripts/smoke_test.sh`，见第 2 节开头）：

1. **Step 1**：建议加 `--overwrite` 重新运行（新增了数据检查，文件头间距也改为精确的 1 mm）；不重跑也能训练，含非法标签的 2 例会在 Step 2 被跳过。
2. **Step 2**：重建 CSV。旧 CSV 的坐标是错的，训练建缓存时会报错。
3. **Step 3**：用新的 `--logdir` 从头训练，不要接着旧 checkpoint 训练。
4. 评估和新旧对比都用修复后的 CSV 重新做。
