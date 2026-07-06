# G1 Diffusion V2 计划交付书

适用代码：

```text
work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN
```

目标：请服务器操作者按 `README_DIFFUSION.md` 跑 V2 diffusion augmentation 线。当前第一轮只使用 **没有缺失 / 非 fake T2W** 的完整样本训练；缺失模态填补完成后，再把补齐后的样本并入全集，重新划分 train/val，再跑第二轮。

## 1. 本轮输入数据口径

数据必须逐病例放在：

```text
work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN/DataSet/
```

每个病例一个目录，目录内必须有 5 个文件：

```text
DataSet/BraTS-MET-00001-000/
  BraTS-MET-00001-000-t1c.nii.gz
  BraTS-MET-00001-000-t1n.nii.gz
  BraTS-MET-00001-000-t2w.nii.gz
  BraTS-MET-00001-000-t2f.nii.gz
  BraTS-MET-00001-000-seg.nii.gz
```

本轮只使用完整样本：

1. 来源：`work_space/G1/data/g1_data_placement_manifest.csv`
2. 纳入：`is_fake_t2w=False` 且 `final_qc_pass=True`
3. 排除：`is_fake_t2w=True` 的 265 个 fake/broken T2W 病例
4. 当前固定全集：1030 例
5. 当前固定划分：824 train / 206 val

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

`csv_creator.py --val_patients` 必须使用 `brats2026_v2_complete_only_val_patients_one_line.txt` 里的内容。注意这里的 ID 是患者编号后九位，例如：

```text
00001-000
00005-000
```

不要传完整 `BraTS-MET-00001-000`。

## 2. 推荐运行顺序

所有命令都在 V2 GliGAN 目录执行：

```bash
cd "/scratch/bf2260/ECNU_EYU_data/work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN"
```

推荐直接用本次交付的 SLURM：

```bash
CSV_JOB=$(sbatch --parsable slurm/01_create_csv_v2_nyu.slurm)
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${CSV_JOB} slurm/02_train_4modal_v2_nyu.slurm)
EVAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} slurm/03_eval_v2_nyu.slurm)
sbatch --dependency=afterok:${TRAIN_JOB} slurm/04_generate_visual_v2_nyu.slurm
```

## 3. Step 1：创建 CSV

README 第 2 节 Step 1 的核心命令是：

```bash
VAL_PATIENTS=$(tr -d '\n\r ' < splits/brats2026_v2_complete_only_val_patients_one_line.txt)
python src/train/csv_creator.py \
  --dataset BRATS_2024 \
  --datadir DataSet \
  --logdir brats2026_diffusion_v2_complete_only \
  --crop_size 64 \
  --merge_dist 16 \
  --val_patients "${VAL_PATIENTS}"
```

输出位置：

```text
../../Checkpoint/brats2026_diffusion_v2_complete_only/brats2026_diffusion_v2_complete_only.csv
```

CSV 的 `split` 列由 `--val_patients` 决定。`val_patients` 中列出的患者，其所有病灶行都是 `val`；其余完整病例都是 `train`。

## 4. Step 2：四模态训练

README 第 2 节 Step 2 的默认训练命令如下，本次脚本保持这个设置：

```bash
python src/train/tumour_main_diffusion.py \
  --dataset BRATS_2024 \
  --modality t1c \
  --logdir brats2026_diffusion_v2_complete_only \
  --batch_size 16 \
  --generator_type Unet_NnU \
  --crop_size 64 \
  --small_lesion_weight 3.0 \
  --num_steps 100000 \
  --noise_schedule edm \
  --use_compile
```

四个模态必须都训练：

```text
t1c
t1n
t2w
t2f
```

建议申请 4 张 A100，一张卡跑一个模态。本次脚本 `02_train_4modal_v2_nyu.slurm` 会同时启动四个进程：

```text
GPU 0 -> t1c
GPU 1 -> t1n
GPU 2 -> t2w
GPU 3 -> t2f
```

训练权重输出：

```text
../../Checkpoint/brats2026_diffusion_v2_complete_only/<modality>/weights/
```

## 5. Loss 监控

训练过程中可以另开终端看 loss：

```bash
cd "/scratch/bf2260/ECNU_EYU_data/work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN"
python scripts/watch_loss.py ../../Checkpoint/brats2026_diffusion_v2_complete_only t1c --live
python scripts/watch_loss.py ../../Checkpoint/brats2026_diffusion_v2_complete_only t1n --live
python scripts/watch_loss.py ../../Checkpoint/brats2026_diffusion_v2_complete_only t2w --live
python scripts/watch_loss.py ../../Checkpoint/brats2026_diffusion_v2_complete_only t2f --live
```

或直接看原始日志：

```bash
tail -f ../../Checkpoint/brats2026_diffusion_v2_complete_only/t1c/loss_lists/loss_diffusion.log
```

如果某个模态 loss 长时间不下降，先保留日志，不要覆盖同一 `logdir` 重跑。

## 6. Step 3：评估

训练完成后直接跑 README 第 2 节 Step 4 的评估命令。评估本身会调用模型生成，再计算指标。

本次默认使用转移瘤更合适的 whole-brain 评估：

```bash
python src/infer/evaluate_generation.py \
  --diffusion_ckpt_dir ../../Checkpoint/brats2026_diffusion_v2_complete_only \
  --csv_path ../../Checkpoint/brats2026_diffusion_v2_complete_only/brats2026_diffusion_v2_complete_only.csv \
  --dataset BRATS_2024 \
  --output_dir ./eval_results/brats2026_diffusion_v2_complete_only_whole_brain \
  --generator_type Unet_NnU \
  --crop_size 64 \
  --evaluation_mode whole_brain \
  --noise_schedule edm \
  --sampling_method edm_heun \
  --sampling_steps 18 \
  --use_compile
```

输出重点：

```text
eval_results/brats2026_diffusion_v2_complete_only_whole_brain/metrics.json
```

## 7. Step 4：推理给人工肉眼检查

README 第 2 节 Step 3 的推理命令主要用于生成 MRI 给人看，不是替代正式评估。

示例：

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

## 8. 本轮交付物

已经新增：

```text
work_space/G1/docs/G1_diffusion_V2_计划交付书.md
work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN/splits/brats2026_v2_complete_only_train_val_split.csv
work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN/splits/brats2026_v2_complete_only_train_patients.txt
work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN/splits/brats2026_v2_complete_only_val_patients.txt
work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN/splits/brats2026_v2_complete_only_val_patients_one_line.txt
work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN/splits/brats2026_v2_complete_only_split_summary.md
work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN/slurm/01_create_csv_v2_nyu.slurm
work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN/slurm/02_train_4modal_v2_nyu.slurm
work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN/slurm/03_eval_v2_nyu.slurm
work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN/slurm/04_generate_visual_v2_nyu.slurm
work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN/slurm/README.md
```

## 9. 完成标准

服务器操作者跑完后，至少回传这些轻量结果：

```bash
find ../../Checkpoint/brats2026_diffusion_v2_complete_only -type f | head -100
find ../../Checkpoint/brats2026_diffusion_v2_complete_only -path '*weights*' -name '*.pt' | wc -l
find ./eval_results/brats2026_diffusion_v2_complete_only_whole_brain -type f | sort
find ./visual_output -type f -name '*.nii.gz' | head -50
```

理想状态：

1. CSV 创建成功，且存在 train/val 两类 split。
2. 四个模态都有权重文件。
3. 四个模态都有 loss 日志。
4. `metrics.json` 生成。
5. 视觉检查输出能打开，不是空图、错位或全噪声。
6. 后续 G2 再接 `visual_output` 或正式 synthetic 输出做 QC。
