# G2 任务分工

## 所属小组

生成模型小组。

## 角色定位

数据生成与质量控制。

## 具体任务

1. 深入理解往年冠军方案中的 GAN 代码。
2. 利用 G1 训练好的扩散模型，大规模生成合成病例。
3. 开发自动化脚本，将生成的合成数据直接转换为 nnU-Net 输入格式，即 `imagesTr` 与 `labelsTr`。
4. 评估合成数据质量，可使用 FID、MS-SSIM 等指标，或通过预训练分割模型的 Dice 改进程度进行间接衡量。

## 2026-07-12 当前执行口径

1. G1 Diffusion V2 属于新增 synthetic augmentation，必须先由 G2 恢复 source 强度、affine/header、seg 和非 ROI 内容，再经 QC 后只追加到 train。
2. G1 缺失模态 V3 属于原病例 T2W completion，保留原病例身份，只替换 T2W，不计入新增 synthetic 病例数。
3. G2 输出两种视图：nnU-Net view 供 S2，按 split 分区的 case-folder view 供 S1/S3/S4/S5；S5 再运行自己的固定 split 预处理。
4. 全队使用 patient-group master split，同一患者不跨 train/val/test；locked test 与 train/val 物理隔离。
5. FID、MS-SSIM 和 teacher 推理目前是待完成的批次级扩展检查；未运行时不得填占位值或据此自动放行。
6. 最终训练价值以固定真实 val/test 上的官方 lesionwise DSC/NSD 和 small-instance F1 成对消融为准。

## 交付物

- 高质量合成数据集。
- 合成数据质量评估报告。
- 合成数据到 nnU-Net 与病例目录双视图的自动转换脚本。

## 协作接口

- 接收 G1 提供的扩散模型与生成样例。
- 向 S1、S2、S3、S4、S5 提供合成数据，并协助验证合成数据增强前后的性能变化。
