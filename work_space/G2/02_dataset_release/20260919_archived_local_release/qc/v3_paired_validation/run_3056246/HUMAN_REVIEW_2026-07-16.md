# G1 V3 run_3056246 人工分层复核与阶段 6 结论

## 最终结论

`reject_and_retune`

当前 G1 V3 输出不得进入阶段 6，不得用于替换缺失 T2W，也不得作为 S1-S5 的正式训练或评估输入。
本次阻断首先是可逆预处理和原生空间恢复问题，不应先通过调整 BBDM `s`、weight decay 或继续训练来掩盖。

## 1. 已核验产物

- G1 Stage 5 run：`run_3056246`。
- paired QC：103 个病例、515 条区域记录、749 条连通病灶记录、103 张 montage。
- 冻结 S2 teacher：103 个 baseline/generated 成对病例。
- geometry audit：103 个病例；原输出 103/103 affine 不一致；修复 103/103；未做重采样；最大体素差为 0。
- 本地与服务器结果已用 `rsync -n -c --delete` 逐文件校验，无差异。

病例 ID 在 paired QC、teacher、review index 和 geometry audit 四套结果中完全一致，均为 103 个唯一病例。

## 2. 人工分层复核范围

按 `review_index.csv` 顺序复核全部 103 张 montage：

| 层级 | 病例数 | 复核状态 |
|---|---:|---|
| high | 86 | 已复核 |
| medium | 14 | 已复核 |
| routine | 3 | 已复核 |
| 合计 | 103 | 已复核 |

每例均同时查看真实 T2W、生成 T2W、绝对误差和生成 T2W + segmentation overlay 的轴位、冠状位、矢状位。

## 3. 自动指标

### 3.1 Paired QC

- whole-volume SSIM：均值 0.924893。
- brain SSIM：均值 0.781331。
- tumor SSIM：94 个非空肿瘤病例均值 0.943131。
- 49/103 病例触发 brain void。
- 61/103 病例触发 external signal。
- 29/103 病例触发 blur。
- 86/103 病例至少触发一个人工筛查原因。
- 17/103 病例的 `brain_void_excess > 0.20`。
- 10/103 病例的 `lesion_void_fraction > 0.50`；其中 7 例超过 0.80。

whole/tumor SSIM 会被大面积背景和很小的 segmentation ROI 抬高，因此不能覆盖下面的空白视野问题。

### 3.2 冻结 S2 teacher

- teacher 技术执行成功，103/103 预测完整。
- ET/RC/TC/WT macro Dice：0.577287 降至 0.558031，变化 -0.019256。
- 该值仅比既定最大下降 0.02 好 0.000744，处在门槛边缘。
- 24/103 病例的逐例 macro Dice 下降超过 0.02。
- 11/103 病例下降超过 0.05。
- 10/103 病例至少一个区域下降超过 0.10。
- 大病灶漏检为 0/172，但新增 4 个大病灶。
- RC Dice 变化 -0.025421；TC Dice 变化 -0.025859。

teacher 只是冻结模型敏感性证据，不是官方 lesionwise DSC/NSD 评价；其批次均值也不能推翻病例级严重失败。

## 4. 阻断性人工发现

下列 montage 出现肿瘤所在平面整片为空、明显中心裁切、视野不完整或原生空间恢复后大面积补零：

- `BraTS-MET-00625-000`
- `BraTS-MET-01058-001`
- `BraTS-MET-01105-001`
- `BraTS-MET-01105-002`
- `BraTS-MET-01119-000`
- `BraTS-MET-01119-001`
- `BraTS-MET-01119-003`
- `BraTS-MET-01187-000`
- `BraTS-MET-01284-000`
- `BraTS-MET-01284-002`
- `BraTS-MET-01331-002`

其中：

- `BraTS-MET-01105-001/002` 的 lesion void 分别为 0.950492/0.961109，逐例 teacher macro Dice 分别下降 0.518196/0.410428。
- `BraTS-MET-01331-002` 的 lesion void 为 1.0，TC 和 WT Dice 均下降 0.314994，并新增一个大病灶。
- `BraTS-MET-01058-001` 的 tumor SSIM 仅 0.575719，brain void excess 为 0.364108。
- `BraTS-MET-01119-000/001/003` 的 brain void excess 均约为 0.37，lesion void 均超过 0.82。

此外，`BraTS-MET-01117-001`、`BraTS-MET-00202-000` 和 `BraTS-MET-01351-003` 虽未全部表现为整片空白，但逐例 teacher macro Dice 分别下降 0.244814、0.160752 和 0.154633，仍属于必须回归检查的严重异常。

## 5. 根因定位

V3 当前预处理在 `synthesis/utils.py` 中对所有原始体积做固定中心裁切/补零：

```text
SHAPE_PREPROCESS_IMG = (256, 256, 160)
preprocessing -> resize_center_crop_pad(..., SHAPE_PREPROCESS_IMG)
postprocessing -> resize_center_crop_pad(..., org_shape)
```

当原始体积大于 256 x 256 x 160 时，中心窗口外的真实脑区和病灶在前向预处理时已被丢弃；后处理恢复原始 shape 时只能补零，不能恢复被裁掉的内容。

本批数据中的证据：

- 54/103 病例至少一个原始维度超过 256 x 256 x 160。
- 49/54 超窗病例触发 brain void。
- 0/49 未超窗病例触发 brain void。
- 10 个 `lesion_void_fraction > 0.50` 病例全部属于超窗组。

该 49/54 对 0/49 的分层结果说明问题与固定中心裁切高度一致。geometry repair 只修正 NIfTI header/affine，并保持体素不变，因此无法修复已经丢失的视野。

## 6. 重新进入阶段 6 的必要条件

1. 暂停阶段 6，不把本 run 的 generated T2W 交给下游。
2. 将固定中心裁切替换为可逆、affine-aware 的标准化流程：统一 canonical orientation/spacing，并使用可记录和可逆的 brain/foreground crop；图像使用连续插值，seg 使用 nearest-neighbor。
3. 新增预处理保真硬门：前向再逆变换后 brain/seg 支持域不得被裁掉，尤其要求 lesion support 100% 保留。
4. 保存结果时直接使用正确原生 qform/sform/header；新的 geometry audit 必须 103/103 原生一致，不再依赖事后 header repair。
5. 使用相同 fixed validation 重新运行 Stage 5、paired QC、冻结 S2 teacher 和全部 mandatory-strata montage 复核。
6. 只有无空白/截断视野、无裁切导致的病灶丢失，并同时通过既定 teacher 批次门槛后，才可重新考虑 `approve_stage6`。

在以上问题修复前，继续调 BBDM `s` 或 weight decay 不会恢复已经在预处理阶段被裁掉的体素。
