# G1 Diffusion V2 计划交付书

更新日期：2026-07-06

适用对象：负责在服务器上运行 G1 diffusion augmentation V2 线的操作者。

适用代码：

```text
work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN
```

配套参考：

```text
work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN/README_DIFFUSION.md
work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN/slurm/README.md
work_space/G1/docs/G1_G2_服务器训练推理QC运行手册.md
work_space/G1/docs/G1_diffusion_augmentation服务器训练手册.md
```

## 0. 先确认这条线的任务边界

G1 当前有两条线，不能混着跑：

| 线 | 代码目录 | 任务 | 输入 | 输出 | 当前状态 |
|---|---|---|---|---|---|
| T2W completion | `work_space/G1/code/brats2025-latent-ensemble-synthesis-main` | 缺失模态填补 | `t1n/t1c/t2f/seg`，不带 `t2w` | 补出的 `t2w` | 先用于修复 fake/broken T2W 病例 |
| Diffusion V2 augmentation | `work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN` | 从 `seg` 学习肿瘤内部结构并生成完整四模态 MRI | 完整 `t1n/t1c/t2w/t2f/seg` | synthetic `t1c/t1n/t2w/t2f/seg` | 本计划书对应这一条 |

本计划书只管 **Diffusion V2 augmentation**。第一轮只训练没有缺失、没有 fake/broken T2W 的完整病例。等 T2W completion 结果通过 G2 QC 后，第二轮再把补齐病例并入全集，重新生成 train/val，再训练 V2。

## 1. 本轮数据口径

### 1.1 当前固定 complete-only 数据

第一轮 V2 只使用完整样本：

| 项目 | 数量 / 规则 |
|---|---|
| 来源清单 | `work_space/G1/data/g1_data_placement_manifest.csv` |
| 纳入条件 | `is_fake_t2w=False` 且 `final_qc_pass=True` |
| 排除条件 | `is_fake_t2w=True` 的 fake/broken T2W 病例 |
| complete-only 总数 | 1030 例 |
| train | 824 例 |
| val | 206 例 |
| split 规则 | 对 9 位 patient suffix 做 deterministic SHA256 hash，最低 20% 进 val |

固定划分文件已经放在：

```text
work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN/splits/
```

关键文件：

```text
brats2026_v2_complete_only_train_val_split.csv
brats2026_v2_complete_only_train_patients.txt
brats2026_v2_complete_only_val_patients.txt
brats2026_v2_complete_only_val_patients_one_line.txt
brats2026_v2_complete_only_split_summary.md
```

`csv_creator.py --val_patients` 必须使用：

```text
splits/brats2026_v2_complete_only_val_patients_one_line.txt
```

注意：`--val_patients` 传的是患者编号后 9 位，例如：

```text
00001-000
00005-000
```

不要传完整 `BraTS-MET-00001-000`。

### 1.2 代码参数为什么仍用 BRATS_2024

虽然数据是 2026 Task1 MET，本 V2 代码内部仍沿用 BraTS 2024 转移瘤标签转换逻辑。正式命令里必须使用：

```text
--dataset BRATS_2024
```

不要改成 `BRATS_2026`。`BRATS_2024` 在这里表示代码内部的 MET label transform，不表示我们在训练旧数据。

## 2. 服务器目录约定

推荐项目根目录：

```text
/scratch/<user>/ECNU_EYU_data
```

以下命令用变量表示，操作者按服务器实际路径替换：

```bash
export PROJ=/scratch/<user>/ECNU_EYU_data
export CODE_DIR="${PROJ}/work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN"
export RAW_ROOT="${PROJ}/work_space/G1/data/raw"
export LOG_ROOT="${PROJ}/logs"
```

原始数据只挂一份：

```text
work_space/G1/data/raw/
```

V2 代码实际读取：

```text
work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN/DataSet/
```

`DataSet/` 是 V2 运行入口，可以是 symlink，也可以是由 raw 数据软链接出来的轻量目录。不要把 NIfTI 复制进 Git，也不要把 `DataSet/` 里的大文件提交。

## 3. DataSet 摆放标准

每个病例一个目录，目录名和文件名前缀必须一致：

```text
DataSet/BraTS-MET-00001-000/
  BraTS-MET-00001-000-t1c.nii.gz
  BraTS-MET-00001-000-t1n.nii.gz
  BraTS-MET-00001-000-t2w.nii.gz
  BraTS-MET-00001-000-t2f.nii.gz
  BraTS-MET-00001-000-seg.nii.gz
```

第一轮必须只放 complete-only 的 1030 例。`DataSet/` 中不要放：

1. 缺 `t2w` 的病例。
2. fake/broken T2W 病例。
3. Validation、test 或官方隐藏测试数据。
4. 旧 GliGAN/GLI 病例。
5. smoke test 病例。

如果 raw 数据目录内部有多层子目录，可以用下面这段命令按 split 表自动建立 symlink。它会递归找 `BraTS-MET-*` 病例，并只链接 split 表里列出的 1030 例：

```bash
cd "${CODE_DIR}"
rm -rf DataSet
mkdir -p DataSet

python - <<'PY'
from pathlib import Path
import csv
import os

code_dir = Path(os.environ["CODE_DIR"])
raw_root = Path(os.environ["RAW_ROOT"])
split_csv = code_dir / "splits" / "brats2026_v2_complete_only_train_val_split.csv"
dataset_dir = code_dir / "DataSet"
required = ["t1c", "t1n", "t2w", "t2f", "seg"]

if not raw_root.exists():
    raise SystemExit(f"RAW_ROOT not found: {raw_root}")

with split_csv.open(newline="") as f:
    cases = [row["case_id"] for row in csv.DictReader(f)]

case_dirs = {}
for p in raw_root.rglob("BraTS-MET-*"):
    if p.is_dir():
        case_dirs.setdefault(p.name, p)

missing_cases = []
missing_files = []
for case_id in cases:
    src = case_dirs.get(case_id)
    if src is None:
        missing_cases.append(case_id)
        continue
    dst = dataset_dir / case_id
    dst.mkdir(parents=True, exist_ok=True)
    for mod in required:
        src_file = src / f"{case_id}-{mod}.nii.gz"
        if not src_file.exists():
            missing_files.append(str(src_file))
            continue
        dst_file = dst / src_file.name
        if dst_file.exists() or dst_file.is_symlink():
            dst_file.unlink()
        dst_file.symlink_to(src_file)

if missing_cases or missing_files:
    print("Missing cases:", len(missing_cases))
    print("Missing files:", len(missing_files))
    for item in missing_cases[:20]:
        print("  missing case:", item)
    for item in missing_files[:20]:
        print("  missing file:", item)
    raise SystemExit("DataSet symlink build failed; fix raw data before training.")

print(f"Linked complete-only cases: {len(cases)}")
PY
```

建完后必须检查：

```bash
find DataSet -mindepth 1 -maxdepth 1 -type d | wc -l
find DataSet -type f -name '*.nii.gz' -o -type l -name '*.nii.gz' | wc -l
find DataSet -mindepth 1 -maxdepth 1 -type d | head
```

理想结果：

```text
case dir count = 1030
nii.gz count = 5150
```

## 4. 环境要求

建议使用单独 Conda 环境，不要混用系统 Python、Homebrew Python 或 `sudo pip`。

基础环境：

```bash
conda create -n brats python=3.11 -y
conda activate brats
pip install torch monai nibabel numpy scipy pandas matplotlib tqdm
```

在服务器上正式跑前检查：

```bash
cd "${CODE_DIR}"
python - <<'PY'
import torch, monai, nibabel, numpy, scipy, pandas
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("gpu count", torch.cuda.device_count())
print("monai", monai.__version__)
print("ok")
PY
```

如果 PyTorch/CUDA 不匹配，先修环境，不要直接提交 SLURM。

## 5. 推荐运行方式：直接用 SLURM

先确认 SLURM 脚本顶部这些变量符合服务器：

```text
#SBATCH --account=...
#SBATCH --gres=gpu:...
#SBATCH --output=...
#SBATCH --error=...
CONDA_SH=...
CONDA_ENV=...
PROJ=...
```

如果日志目录不存在：

```bash
mkdir -p "${LOG_ROOT}"
```

提交顺序：

```bash
cd "${CODE_DIR}"

CSV_JOB=$(sbatch --parsable slurm/01_create_csv_v2_nyu.slurm)
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${CSV_JOB} slurm/02_train_4modal_v2_nyu.slurm)
EVAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} slurm/03_eval_v2_nyu.slurm)
VIS_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} slurm/04_generate_visual_v2_nyu.slurm)

echo "CSV_JOB=${CSV_JOB}"
echo "TRAIN_JOB=${TRAIN_JOB}"
echo "EVAL_JOB=${EVAL_JOB}"
echo "VIS_JOB=${VIS_JOB}"
```

常用监控：

```bash
squeue -u "${USER}"
tail -f "${LOG_ROOT}/g1_diffv2_csv_<JOB_ID>.out"
tail -f "${LOG_ROOT}/g1_diffv2_train4_<JOB_ID>.out"
tail -f "${LOG_ROOT}/g1_diffv2_eval_<JOB_ID>.out"
```

## 6. 手动运行命令

如果不用 SLURM，可以按本节手动跑。

### 6.1 Step 1：创建 lesion-level CSV

```bash
cd "${CODE_DIR}"
VAL_PATIENTS=$(tr -d '\n\r ' < splits/brats2026_v2_complete_only_val_patients_one_line.txt)

python src/train/csv_creator.py \
  --dataset BRATS_2024 \
  --datadir DataSet \
  --logdir brats2026_diffusion_v2_complete_only \
  --crop_size 64 \
  --merge_dist 16 \
  --val_patients "${VAL_PATIENTS}"
```

输出：

```text
../../Checkpoint/brats2026_diffusion_v2_complete_only/brats2026_diffusion_v2_complete_only.csv
```

检查：

```bash
python - <<'PY'
from pathlib import Path
import csv
from collections import Counter, defaultdict

p = Path("../../Checkpoint/brats2026_diffusion_v2_complete_only/brats2026_diffusion_v2_complete_only.csv")
rows = list(csv.DictReader(p.open()))
print("rows", len(rows))
print("patients", len({r["patient_id"] for r in rows}))
print("split rows", Counter(r["split"] for r in rows))
d = defaultdict(set)
for r in rows:
    d[r["split"]].add(r["patient_id"])
print("split patients", {k: len(v) for k, v in d.items()})
print("missing path rows", sum(
    not Path(r[c]).exists()
    for r in rows
    for c in ["scan_t1c", "scan_t1n", "scan_t2w", "scan_t2f", "label"]
))
PY
```

验收标准：

1. patients 应为 1030。
2. split patients 应为 train 824、val 206。
3. missing path rows 必须为 0。
4. CSV 里必须有 `split` 列。

### 6.2 Step 2：训练四个模态

四个模态都要训练：

```text
t1c
t1n
t2w
t2f
```

单模态命令示例：

```bash
python src/train/tumour_main_diffusion.py \
  --dataset BRATS_2024 \
  --modality t1c \
  --logdir brats2026_diffusion_v2_complete_only \
  --split train \
  --batch_size 16 \
  --generator_type Unet_NnU \
  --crop_size 64 \
  --small_lesion_weight 3.0 \
  --num_steps 100000 \
  --noise_schedule edm \
  --use_compile
```

四卡并行时，每张卡跑一个模态；单卡时按模态逐个跑。输出：

```text
../../Checkpoint/brats2026_diffusion_v2_complete_only/<modality>/weights/
../../Checkpoint/brats2026_diffusion_v2_complete_only/<modality>/loss_lists/loss_diffusion.log
```

训练过程中不要改 `logdir`，除非是重新开一轮实验。不要用 `--split all` 做正式训练。

### 6.3 Step 3：监控 loss

```bash
python scripts/watch_loss.py ../../Checkpoint/brats2026_diffusion_v2_complete_only t1c --live
python scripts/watch_loss.py ../../Checkpoint/brats2026_diffusion_v2_complete_only t1n --live
python scripts/watch_loss.py ../../Checkpoint/brats2026_diffusion_v2_complete_only t2w --live
python scripts/watch_loss.py ../../Checkpoint/brats2026_diffusion_v2_complete_only t2f --live
```

或直接看：

```bash
tail -f ../../Checkpoint/brats2026_diffusion_v2_complete_only/t1c/loss_lists/loss_diffusion.log
```

如果某个模态 loss 长时间不下降，先保留日志和 checkpoint，不要覆盖同名 `logdir` 盲目重跑。

### 6.4 Step 4：评估

本轮默认使用 whole-brain 评估，因为转移瘤常见多病灶，需要看融合回全脑后的质量：

```bash
python src/infer/evaluate_generation.py \
  --diffusion_ckpt_dir ../../Checkpoint/brats2026_diffusion_v2_complete_only \
  --csv_path ../../Checkpoint/brats2026_diffusion_v2_complete_only/brats2026_diffusion_v2_complete_only.csv \
  --dataset BRATS_2024 \
  --output_dir ./eval_results/brats2026_diffusion_v2_complete_only_whole_brain \
  --generator_type Unet_NnU \
  --crop_size 64 \
  --evaluation_mode whole_brain \
  --split val \
  --noise_schedule edm \
  --sampling_method edm_heun \
  --sampling_steps 18 \
  --use_compile
```

输出：

```text
eval_results/brats2026_diffusion_v2_complete_only_whole_brain/metrics.json
```

评估只能说明生成模型在 validation 上的 MRI 质量趋势，不等同于官方 segmentation 排行榜指标。官方 lesionwise DSC/NSD/F1 需要后续 S1/S2 segmentation pipeline 和 BraTS evaluation 才能得到。

### 6.5 Step 5：生成肉眼检查样例

```bash
CASE_ID=BraTS-MET-00001-000

python src/infer/generate_from_label.py \
  --label_path DataSet/${CASE_ID}/${CASE_ID}-seg.nii.gz \
  --diffusion_ckpt_dir ../../Checkpoint/brats2026_diffusion_v2_complete_only \
  --dataset BRATS_2024 \
  --output_dir ./visual_output/brats2026_diffusion_v2_complete_only_${CASE_ID} \
  --generator_type Unet_NnU \
  --crop_size 64 \
  --merge_dist 16 \
  --noise_schedule edm \
  --sampling_method edm_heun \
  --sampling_steps 18 \
  --modality all \
  --use_compile
```

输出应包含：

```text
<case_id>-t1c.nii.gz
<case_id>-t1n.nii.gz
<case_id>-t2w.nii.gz
<case_id>-t2f.nii.gz
```

肉眼检查重点：

1. 是否全黑、全白、全噪声。
2. 肿瘤区域是否和 `seg` 大致对齐。
3. 四个模态是否出现明显错位。
4. 是否存在裁剪、翻转、方向异常。
5. 多病灶是否都被生成并融合回全脑。

## 7. 超参数策略

第一轮以跑通和建立质量基线为主：

| 参数 | 当前建议 | 说明 |
|---|---|---|
| `crop_size` | 64 | 与 CSV、训练、推理、评估保持一致 |
| `merge_dist` | 16 | 邻近病灶合并阈值 |
| `generator_type` | `Unet_NnU` | 当前 V2 脚本默认设置 |
| `noise_schedule` | `edm` | 训练和推理必须一致 |
| `sampling_method` | `edm_heun` | 与 `edm` 配套 |
| `sampling_steps` | 18 | 第一轮验证速度和质量折中 |
| `small_lesion_weight` | 3.0 | 强化小病灶生成 |
| `batch_size` | 16 | A100 参考值；显存不足先降到 8 或 4 |
| `num_steps` | 100000 | 第一轮可先跑；若资源允许，后续比较 150000/200000 |

调参顺序：

1. 先保证 DataSet、CSV、四模态训练、评估、G2 intake 全链路跑通。
2. 再比较 `num_steps` 的 checkpoint 曲线。
3. 再比较 `sampling_steps`。
4. 再调 `small_lesion_weight` 和 `merge_dist`。
5. 最后才改 backbone、学习率、weight decay、CFG 等更大变量。

## 8. G2 接收和 QC

V2 diffusion augmentation 属于 full generation，不是 T2W completion。交给 G2 时必须使用：

```text
--generation-mode full_generation
```

单病例 `visual_output` 只适合人工检查，不建议作为正式 synthetic run 直接纳入训练。正式 synthetic run 应该放在一个独立输出目录，例如：

```text
/scratch/<user>/ECNU_EYU_data/runs/g1_diffusion_v2_full_generation_v1/
  BraTS-MET-SYN-000001/
    BraTS-MET-SYN-000001-t1c.nii.gz
    BraTS-MET-SYN-000001-t1n.nii.gz
    BraTS-MET-SYN-000001-t2w.nii.gz
    BraTS-MET-SYN-000001-t2f.nii.gz
    BraTS-MET-SYN-000001-seg.nii.gz
```

G2 intake 命令：

```bash
cd "${PROJ}"

python work_space/G2/code/g2_synthetic_raw_intake_qc.py \
  --synthetic-run-root /scratch/<user>/ECNU_EYU_data/runs/g1_diffusion_v2_full_generation_v1 \
  --synthetic-run-id g1_diffusion_v2_full_generation_v1 \
  --generation-mode full_generation \
  --refresh-templates
```

G2 会生成：

```text
work_space/G2/results/manifests/synthetic_generation_manifest_g1_diffusion_v2_full_generation_v1.csv
work_space/G2/results/manifests/synthetic_candidate_manifest_g1_diffusion_v2_full_generation_v1.csv
work_space/G2/results/manifests/synthetic_accepted_manifest_g1_diffusion_v2_full_generation_v1.csv
work_space/G2/results/manifests/synthetic_rejected_manifest_g1_diffusion_v2_full_generation_v1.csv
work_space/G2/results/manifests/synthetic_normalized_mapping_g1_diffusion_v2_full_generation_v1.csv
work_space/G2/results/qc/qc_metrics_g1_diffusion_v2_full_generation_v1.csv
work_space/G2/results/qc/diffusion_quality_metrics_g1_diffusion_v2_full_generation_v1.csv
work_space/G2/results/qc/qc_case_review_g1_diffusion_v2_full_generation_v1.csv
work_space/G2/results/qc/qc_batch_summary_g1_diffusion_v2_full_generation_v1.json
work_space/G2/results/reports/G2_synthetic_data_quality_report_g1_diffusion_v2_full_generation_v1.md
```

G2 决策口径：

1. `accepted_for_training=True` 才能进入主训练集。
2. `accepted_for_ablation_only=True` 只能进入消融实验，不进主训练。
3. rejected 不允许进入 S1/S2 训练。
4. 如果出现方向、shape、affine、强度、肿瘤 mask 或 validation leakage 问题，必须先修 G1 输出或重生成。

## 9. 完成标准

服务器操作者至少需要回传以下轻量结果：

```bash
cd "${CODE_DIR}"

find DataSet -mindepth 1 -maxdepth 1 -type d | wc -l
find DataSet -type f -name '*.nii.gz' -o -type l -name '*.nii.gz' | wc -l

CSV=../../Checkpoint/brats2026_diffusion_v2_complete_only/brats2026_diffusion_v2_complete_only.csv
test -f "${CSV}" && echo "CSV exists: ${CSV}"

for m in t1c t1n t2w t2f; do
  echo "== ${m} =="
  find ../../Checkpoint/brats2026_diffusion_v2_complete_only/${m}/weights -type f -name '*.pt' | sort | tail -5
  test -f ../../Checkpoint/brats2026_diffusion_v2_complete_only/${m}/loss_lists/loss_diffusion.log && tail -5 ../../Checkpoint/brats2026_diffusion_v2_complete_only/${m}/loss_lists/loss_diffusion.log
done

find ./eval_results/brats2026_diffusion_v2_complete_only_whole_brain -type f | sort
find ./visual_output -type f -name '*.nii.gz' | head -50
```

理想状态：

1. `DataSet/` 有 1030 个病例目录、5150 个 NIfTI 链接或文件。
2. CSV 创建成功，且 train/val 两类 split 都存在。
3. CSV 中 patient 数为 1030，train 824，val 206。
4. 四个模态都有权重文件。
5. 四个模态都有 loss 日志。
6. `metrics.json` 生成。
7. 视觉检查样例不是空图、错位、全噪声或明显裁剪失败。
8. 正式 synthetic run 能被 G2 intake 生成 accepted/rejected/QC/report。

## 10. 常见失败和处理

| 现象 | 最可能原因 | 处理 |
|---|---|---|
| `DataSet directory not found` | 没在 V2 GliGAN 目录下建 `DataSet/` | 按第 3 节重新建立 symlink |
| CSV patients 不是 1030 | raw 数据不全、split 表不匹配、混入旧数据 | 停止训练，先修 DataSet |
| CSV 只有 train 没有 val | `--val_patients` 传错，传了完整 case_id 或文件没读到 | 使用 one-line 文件内容，只传 9 位 suffix |
| `missing t2w` 大量出现 | 把缺失/fake T2W 病例放进了 V2 DataSet | 重新按 complete-only split 建 DataSet |
| `Unknown dataset` | 错用 `BRATS_2026` | 改回 `BRATS_2024` |
| CUDA 不可用 | 环境或 SLURM GPU 申请错误 | 先修 PyTorch/CUDA 或 `#SBATCH --gres` |
| 显存不足 | batch 太大或 cache 太重 | 先把 `BATCH_SIZE` 降到 8，再降到 4 |
| `metrics.json` 不生成 | checkpoint 不完整或评估 split 空 | 先检查四模态 weights 和 CSV val 行 |
| G2 intake 全 rejected | 输出结构、命名、shape、affine、强度或 leakage 不合格 | 看 G2 `qc_case_review` 和 `qc_batch_summary` 后重生成 |

## 11. 第二轮计划

第一轮完成后不要立刻把所有 generated cases 混入训练。下一步顺序是：

1. T2W completion 线先补 fake/broken T2W 病例。
2. G2 对 completion 输出跑 QC。
3. 只把 G2 accepted 的 completion T2W 替换进真实数据口径。
4. G2 重新生成完整 real+completion manifest 和 train/val/test。
5. V2 diffusion augmentation 重新按新 manifest 建 `DataSet/`。
6. V2 重新创建 CSV、训练、评估。
7. 生成 synthetic full-generation run。
8. G2 对 synthetic run 做 QC 和 accepted/rejected。
9. S1/S2 只使用 G2 accepted 数据做训练或消融。

## 12. 不要做的事

1. 不要把缺 T2W 病例手动塞进 V2 训练。
2. 不要把 fake/broken T2W 当真实 T2W 训练。
3. 不要把 `Validation/` 或官方隐藏测试数据混进训练。
4. 不要把 `--dataset` 改成 `BRATS_2026`。
5. 不要把 `--split all` 用作正式训练。
6. 不要把大体积 NIfTI、训练 checkpoint、评估图片或临时输出提交进 Git。
7. 不要覆盖同名 `logdir` 的失败实验，除非已经备份日志和 checkpoint。
8. 不要跳过 G2 QC 直接把 synthetic 数据交给 S1/S2。
