# G1-G2 V2/V3 输出契约

更新日期：2026-07-12

## 1. 不可混用的两种模式

### V2 augmentation

V2 raw output 不能直接交给下游模型。它必须先进入：

```text
g2_v2_compose_augmentation.py
```

V2 run 必需元数据：

```json
{
  "generation_run_id": "v2_run_xxx",
  "generator_name": "g1_diffusion_v2",
  "generation_mode": "full_generation",
  "seed": 42,
  "source_csv": "work_space/G2/results/manifests/g1_v2_source_manifest.csv",
  "diffusion_checkpoint_dir": "/server/path/to/checkpoints",
  "sampling_method": "edm_heun",
  "sampling_steps": 18,
  "eta": 0.0,
  "crop_size": 64,
  "label_channels": 4
}
```

V2 flat raw 文件名：

```text
<source_case_id>-t1n.nii.gz
<source_case_id>-t1c.nii.gz
<source_case_id>-t2w.nii.gz
<source_case_id>-t2f.nii.gz
```

Composer 输出：

```text
composed_run/
  generation_config.json
  generation_log.jsonl
  synthetic_generation_manifest.csv
  <source_case_id>_v2aug_label_0/
    <raw_id>-t1n.nii.gz
    <raw_id>-t1c.nii.gz
    <raw_id>-t2w.nii.gz
    <raw_id>-t2f.nii.gz
    <raw_id>-seg.nii.gz
    <raw_id>-generation_support.nii.gz
```

Composer 强制检查：source 属于 master train、T2W 真实、四模态齐全、shape 一致、seg 合法。输出恢复 source affine/header，非 generation support 区域保持 source 值。输出目录非空时默认拒绝；`--overwrite` 表示清空整个 composed run 后重建。

### V3 completion

V3 run 直接进入：

```text
g2_v3_completion_intake.py
```

目录：

```text
run_<id>/
  generation_config.json
  inference_run.json
  generation_log.jsonl
  synthetic_generation_manifest.csv
  <source_case_id>/
    <source_case_id>-t1n.nii.gz
    <source_case_id>-t1c.nii.gz
    <source_case_id>-t2w.nii.gz
    <source_case_id>-t2f.nii.gz
    <source_case_id>-seg.nii.gz
```

V3 元数据至少包括：

```text
generation_run_id
generator_name
generation_mode=completion
source_csv
seed
vae_weights
encdec_checkpoint
bbdm_checkpoint
bbdm_s
validation_run
```

V3 completion 的 `synthetic_final_id` 等于 `source_case_id`，nnU-Net ID 复用 master mapping。只有 T2W 可以替换，其他模态和 seg 最终从真实 source 读取。

## 2. Split 约束

1. V2 source 只能是 `master split=train` 且 `eligible_for_realonly=True`。
2. V2 augmentation 只加入输出 split 的 train。
3. V3 train completion 经批准后可用于训练。
4. V3 val/test completion 经批准后只用于原 val/test。
5. official validation 没有公开标签，不得作为 V2 source，也不进入内部训练 QC。

## 3. Metadata 硬门

缺少以下任一关键项即 rejected：

```text
generation config/inference run
generation log
generation manifest
generation_run_id
generator_name
seed
source_csv
required checkpoint(s)
V3 bbdm_s
V3 validation_run
```

缺少上述运行证据时 G2 直接拒绝，不会虚构 checkpoint、seed、日志或 manifest。每个病例还必须在 generation manifest 和 JSONL log 中各有唯一 `status=success` 记录，且 source、run ID、seed 一致。

## 4. QC 审批文件

技术 QC 通过后，操作者在 run root 放置：

```text
g2_approval_manifest.csv
```

表头：

```csv
synthetic_raw_id,approved_for_training,approved_for_evaluation,reviewer,reason
```

没有该审批时，病例保持 `pending_review`。

## 5. 下游输出

G2 materializer 同时输出：

1. `DatasetXXX_*/imagesTr + labelsTr` 保存 train/val，供 S2。
2. `DatasetXXX_*/imagesTs + labelsTs` 物理隔离 locked test。
3. `<dataset>_case_folders/{train,val,test}/<case_id>/`，供 S1/S3/S4/S5。
4. `g2_fixed_split.json`、`splits_final.json`。
5. `g2_materialization_manifest.csv`。
6. `g2_integrity_report.json`。

固定通道顺序只有：`t1n,t1c,t2w,t2f`。
