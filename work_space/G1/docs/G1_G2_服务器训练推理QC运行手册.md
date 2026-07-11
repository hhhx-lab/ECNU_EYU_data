# G1 V2 缺失 T2W 填补服务器运行手册

更新日期：2026-07-10

适用对象：在华东师范大学超算八期运行 G1 V2 的操作者。

## 0. 流程结论

G1 V2 学习 `t1n + t1c + t2f -> t2w`，完整流程是：

```text
完整四模态 raw data
  -> patient-grouped train/val/locked test
  -> VAE baseline + train 微调 + val 选择
  -> 使用被选中的同一个 VAE 重编码全部 latent
  -> 并行训练 EncDec 与 BBDM
  -> val 验证 ensemble
  -> 真正缺 T2W 病例重建
  -> G2 QC accepted/rejected
```

VAE 微调不是独立终点。若采用微调 VAE，旧 VAE 生成的 latent 全部失效，必须重编码并重训 EncDec/BBDM。G2 不参与 VAE 训练；只有最终 fake T2W 生成后才进入 G2 QC。

本地原始数据：

```text
/Users/hwaigc/比赛+课题/ECNU-NYU2026/2026的task1以及数据
```

已核对的服务器准备结果：raw root 共识别 `1296` 个病例，其中 `265` 个被识别为缺失或伪 T2W，不进入 VAE 训练；完整候选 `1031` 个。`BraTS-MET-01094-002` 含非法标签 `6` 且无可用修正版，自动排除；`BraTS-MET-01184-002` 使用 corrected label。最终 `data_csv.csv` 共 `1030` 例：`823 train / 103 val / 104 locked test`。缺 T2W 病例只在模型定型后进入重建阶段。

## 1. 路径约定

本地代码：

```text
/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v2
```

服务器路径：

```bash
PROJECT_ROOT=/public/home/${USER}/projects/ECNU_EYU_data
RAW_DATA_ROOT=${HOME}/data
CODE_DIR=${PROJECT_ROOT}/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v2
```

服务器目录名使用无空格的 `brats2025-latent-ensemble-synthesis-main-v2`。Slurm 从脚本位置自动定位代码，不硬编码具体用户名。

## 2. 连接服务器

登录凭据只保存在本机私密目录：

```text
/Users/hwaigc/比赛+课题/服务器登陆
```

禁止把 `服务器登陆的env` 上传到 Git、网盘或服务器项目目录。

本机先检查并启动分流 VPN：

```bash
cd "/Users/hwaigc/比赛+课题/服务器登陆"
./ecnu-vpn-split.sh doctor
./ecnu-vpn-split.sh start
./ecnu-vpn-split.sh check
./ecnu-vpn-split.sh ssh-test
```

只有 `check` 显示默认外网路由未改变、HPC IP 单独走 VPN，且 `ssh-test` 成功，才继续。进入交互登录：

```bash
./ecnu-vpn-split.sh ssh-hpc
```

使用完毕后：

```bash
./ecnu-vpn-split.sh stop
```

## 3. 上传代码与数据

在本机终端加载私密变量：

```bash
set -a
source "/Users/hwaigc/比赛+课题/服务器登陆/服务器登陆的env"
set +a
```

上传原始数据，可断点续传：

```bash
rsync -avP -e "ssh -p ${HPC_LOGIN_PORT}" \
  "/Users/hwaigc/比赛+课题/ECNU-NYU2026/2026的task1以及数据/" \
  "${HPC_SSH_USER}@${HPC_BACKUP_HOST}:/public/home/${HPC_SSH_USER}/projects/ECNU_EYU_data/raw_task1_2026/"
```

上传 V2 代码，不上传本地运行产物：

```bash
rsync -avP -e "ssh -p ${HPC_LOGIN_PORT}" \
  --exclude ".git/" \
  --exclude ".idea/" \
  --exclude "__pycache__/" \
  --exclude "data/input/" \
  --exclude "data/input_inference/" \
  --exclude "data/latents/" \
  --exclude "data/output/" \
  --exclude "data/eval_synthesized*/" \
  --exclude "training/endec/" \
  --exclude "training/bbdm/" \
  --exclude "training/vae_finetuned/" \
  "/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v2/" \
  "${HPC_SSH_USER}@${HPC_BACKUP_HOST}:/public/home/${HPC_SSH_USER}/projects/ECNU_EYU_data/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v2/"
```

## 4. 环境配置

只使用独立 Conda 环境，不使用 `sudo pip`，不混用系统 Python：

```bash
module purge
module load apps/envs/miniconda3/25.5.1
module load compiler/cuda/12.1

conda create -n brats_g1_v2 python=3.10 -y
conda activate brats_g1_v2
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

在计算节点作业外完成依赖安装。验证环境：

```bash
python - <<'PY'
import torch, monai, nibabel, numpy
print("torch", torch.__version__)
print("torch CUDA", torch.version.cuda)
print("MONAI", monai.__version__)
print("nibabel", nibabel.__version__)
print("numpy", numpy.__version__)
PY

python test_vae.py
```

预训练权重必须存在：

```text
weights/vae/autoencoder_epoch273.pt
```

## 5. 明日先跑 VAE 微调

登录服务器后：

```bash
cd /public/home/${USER}/projects/ECNU_EYU_data/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v2
mkdir -p logs
```

先投递数据准备：

```bash
PREP_JOB=$(sbatch --parsable slurm/01_prepare_data.slurm)
echo "PREP_JOB=${PREP_JOB}"
```

`01` 做以下事情：

1. 从 raw root 建立 `data/input/` 和 `data/input_inference/` 软链接。
2. 只收四模态和 seg 完整的训练病例，优先 corrected labels。
3. 先生成 metadata-only `data/data_csv.csv`，此时不浪费 GPU 编码旧 latent。
4. 完整扫描 seg 值域，写 `g1_v2_label_filter_report.csv`；非法且无 corrected label 的病例自动排除，不静默改值。
5. 按 patient group 固定切分，默认 seed `42`，目标比例约 `80/10/10`。
6. 同一 `BraTS-MET-xxxxx` 下所有记录保持在同一 split，防止患者泄漏。
7. 检查路径、NIfTI、shape、affine 和 split 非空。

当前服务器用完整 raw root、seed 42 的准备结果应为：`1030` 个有效完整病例，`823 train / 103 val / 104 locked test`。如果数量不同，先核对 raw root、缺失/伪 T2W 识别和 corrected labels，再决定是否继续。

数据准备成功后再投递 VAE job：

```bash
VAE_JOB=$(sbatch --parsable --dependency=afterok:${PREP_JOB} slurm/02_finetune_vae.slurm)
echo "VAE_JOB=${VAE_JOB}"
```

`02` 完整执行 README 1.2：

1. 用原始 VAE 在全部 `103` 个 val 病例上做完整体积 baseline。
2. 读取全部 `823` 个 `split=train` 病例，每例四模态共用一个 `128×128×96` patch：`80%` 均匀选择一个 26 连通肿瘤域并以其中心裁剪，`20%` 从脑区随机取中心。
3. 最多训练 `3 epochs`，使用 BF16、关闭梯度检查点；每个 epoch 使用固定、可复现的 `20` 个 val 病例快速验证，按 tumor MSE 保存 `best_model.pt`，连续 `2` 次无提升则 early stop。
4. 训练结束后在全部 `103` 个 val 病例上做完整体积比较，生成 delta 指标和 5 张最差病例可视化。
5. locked test 不参与训练、调参或 VAE 选择。

默认参数：

```text
VAE_EPOCHS=3
VAE_BATCH_SIZE=2
VAE_VAL_INTERVAL=1
VAE_SAVE_INTERVAL=1
VAE_PATCH_SIZE="128 128 96"
VAE_TUMOR_PATCH_PROBABILITY=0.8
VAE_QUICK_VAL_SUBJECTS=20
VAE_EARLY_STOPPING_PATIENCE=2
VAE_AMP_DTYPE=bfloat16
```

正式训练前必须先做一次 20 病例 benchmark：

```bash
BENCH_DIR="training/vae_finetuned/benchmark_${USER}_$(date +%Y%m%d_%H%M%S)"
VAE_EPOCHS=1 \
VAE_MAX_TRAIN_SUBJECTS=20 \
VAE_MAX_VAL_SUBJECTS=2 \
VAE_QUICK_VAL_SUBJECTS=2 \
PUBLISH_VAE_SELECTION=0 \
VAE_OUTPUT_DIR="${BENCH_DIR}" \
sbatch slurm/02_finetune_vae.slurm
```

benchmark 必须满足：无 OOM/NaN、日志显示 `optimizer_steps > 0`、输出存在 `best_model.pt` 与 `vae_selection.json`。用日志中纯训练段耗时按 `823×3/20` 外推；确认连同两次完整 val 总时长可落在 `18` 小时限制内，再提交默认正式任务。若 OOM，再将 `VAE_BATCH_SIZE=1` 重投，不先改学习率、patch 或损失。

## 6. VAE 输出与验收

监控：

```bash
squeue -u ${USER}
tail -f logs/g1v2_prep_${PREP_JOB}.out
tail -f logs/g1v2_vae_${VAE_JOB}.out
```

VAE 输出：

```text
training/vae_finetuned/run_<jobid>/baseline_metrics.csv
training/vae_finetuned/run_<jobid>/finetuned_metrics.csv
training/vae_finetuned/run_<jobid>/delta_metrics.csv
training/vae_finetuned/run_<jobid>/training_history.csv
training/vae_finetuned/run_<jobid>/quick_val_subjects.json
training/vae_finetuned/run_<jobid>/finetune_config.json
training/vae_finetuned/run_<jobid>/comparison_samples/*.png
training/vae_finetuned/run_<jobid>/best_model.pt
training/vae_finetuned/run_<jobid>/vae_selection.json
```

读取结论：

```bash
VAE_RUN_DIR=$(head -n 1 training/vae_finetuned/latest_run.txt)
python -m json.tool "${VAE_RUN_DIR}/vae_selection.json"
```

自动采用微调 VAE 的最低门槛：

- mean `delta_tumor_SSIM >= 0.03`
- mean `delta_whole_SSIM >= -0.005`

还必须人工看 `comparison_samples/`，排除空白、错位、肿瘤结构消失和明显伪影。`vae_selection.json` 中 `selected_weights` 是后续唯一权重口径；未达标时自动保留原始 VAE。

## 7. VAE 验收后的阶段

确认 VAE 结果后，统一重编码：

```bash
ENCODE_JOB=$(sbatch --parsable slurm/03_encode_latents.slurm)
echo "ENCODE_JOB=${ENCODE_JOB}"
```

`03` 会清空旧 latent，使用 `selected_weights` 重新编码全部病例，并生成 attention masks 和 channel weights。不要跳过这一步。

EncDec 与 BBDM 可并行：

```bash
ENDEC_JOB=$(TRAIN_TARGET=endec sbatch --parsable --dependency=afterok:${ENCODE_JOB} slurm/04_train_models.slurm)
BBDM_JOB=$(TRAIN_TARGET=bbdm sbatch --parsable --dependency=afterok:${ENCODE_JOB} slurm/04_train_models.slurm)
echo "ENDEC_JOB=${ENDEC_JOB} BBDM_JOB=${BBDM_JOB}"
```

二者都成功后验证 val ensemble：

```bash
EVAL_JOB=$(sbatch --parsable \
  --dependency=afterok:${ENDEC_JOB}:${BBDM_JOB} \
  slurm/05_evaluate_val.slurm)
echo "EVAL_JOB=${EVAL_JOB}"
```

先检查：

```text
data/eval_metrics_val.csv
data/eval_synthesized_val/
training/endec/val_imgs/
training/bbdm/val_imgs/
```

确认模型通过，且 `data/input_inference/` 已有真正缺 T2W 病例后，才投递：

```bash
sbatch slurm/06_infer_missing_t2w.slurm
```

当前 raw data 没有缺 T2W 病例，`06` 会返回错误码 `2`，这是数据状态提示，不是模型故障。

## 8. 调参顺序

1. 先决定 VAE 是否采用微调权重，不要同时改 VAE 和生成模型超参数。
2. 固定 VAE 后训练 EncDec/BBDM baseline。
3. BBDM 第一优先调 `configs.py` 中 `bb_scheduler.s`。
4. 第二优先调 BBDM/EncDec 的 `weight_decay`。
5. 再考虑 batch size、训练步数、seg loss 与 lesion weights。
6. 每组参数必须复用同一 split seed 和同一 VAE，才能公平比较。

## 9. 拉回结果

本机连接 VPN并加载私密变量后：

```bash
LOCAL_BACK="/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G1/data/g1_v2_server_return"
REMOTE_CODE="/public/home/${HPC_SSH_USER}/projects/ECNU_EYU_data/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v2"
mkdir -p "${LOCAL_BACK}"
```

先拉 VAE 报告和最佳权重：

```bash
rsync -avP -e "ssh -p ${HPC_LOGIN_PORT}" \
  "${HPC_SSH_USER}@${HPC_BACKUP_HOST}:${REMOTE_CODE}/training/vae_finetuned/latest_run.txt" \
  "${LOCAL_BACK}/"

rsync -avP -e "ssh -p ${HPC_LOGIN_PORT}" \
  --include "*/" --include "*.csv" --include "*.json" --include "*.png" \
  --include "best_model.pt" --exclude "*" \
  "${HPC_SSH_USER}@${HPC_BACKUP_HOST}:${REMOTE_CODE}/training/vae_finetuned/" \
  "${LOCAL_BACK}/vae_finetuned/"
```

拉最终 val 与推理输出：

```bash
rsync -avP -e "ssh -p ${HPC_LOGIN_PORT}" \
  "${HPC_SSH_USER}@${HPC_BACKUP_HOST}:${REMOTE_CODE}/data/eval_metrics_val.csv" \
  "${LOCAL_BACK}/"

rsync -avP -e "ssh -p ${HPC_LOGIN_PORT}" \
  "${HPC_SSH_USER}@${HPC_BACKUP_HOST}:${REMOTE_CODE}/data/output/" \
  "${LOCAL_BACK}/output/"
```

## 10. 快速排错

1. `RAW_DATA_ROOT not found`：服务器原始数据路径不对，提交 `01` 时显式设置 `RAW_DATA_ROOT`。
2. `pretrained VAE is missing`：80 MB 权重未上传，检查 `weights/vae/autoencoder_epoch273.pt`。
3. `empty_split:val`：切分未运行或 CSV 被人工覆盖，重跑 `01`，不要手改。
4. `affine_mismatch` / `shape_mismatch`：病例模态没有正确配准，不能直接进入训练。
5. `illegal_seg_labels`：查看 `g1_v2_label_filter_report.csv`；已知 01094-002 会被自动排除，其他新增非法病例需要核查 corrected-labels 目录。
6. `CUDA was requested`：环境是 CPU torch，或作业未申请 GPU。
7. `best_model.pt was not produced`：检查 VAE job 的首个 val interval 是否成功完成。
8. `data/selected_vae.json not found`：未运行阶段 3，禁止直接训练 EncDec/BBDM。
9. `input_inference contains no true missing-T2W cases`：当前数据确实没有推理目标，不要复制完整病例冒充缺失病例。
10. `ensamble` 是代码现有合法选项，运行命令保持该拼写。
