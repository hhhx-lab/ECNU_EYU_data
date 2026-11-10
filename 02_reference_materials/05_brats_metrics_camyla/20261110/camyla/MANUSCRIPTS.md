# Camyla 公开分割手稿索引

复核日期：2026-11-10。**公开原稿10篇，报告生成总量40篇；未取得余下30篇。** 以下是匿名生成手稿，不是已确认期刊录用稿或可部署模型。数字为主对比表报告值，未独立训练/推理复现。完整评议见 [调研报告](../REPORT.md)。

系统论文：[本地PDF](Camyla_system_paper.pdf)，[arXiv](https://arxiv.org/abs/2604.10696)。最高优先级为10，其次02/03/01；全部分割相关原稿已保留，未下载医学影像或权重。

## 10 HRFuse-Net，脑转移瘤，最高优先级

**HRFuse-Net: Hierarchical Resolution-Retentive Feature Encoding for Brain Metastasis Segmentation**

[本地PDF](manuscripts/10.pdf) · [冻结原始PDF](https://raw.githubusercontent.com/yifangao112/Camyla/df4434f9d4aef5b7394ed03a4e877a8130c1b6cf/assets/paper_pdf/10.pdf) · [检索文本](searchable_text/10.txt)

LMD-BM，手稿声称44例、35/9划分、三前景无RC。PHRS高分辨率残差流+ASCA轴向上下文+门控。主表Dice0.5069/HD95 25.4869，nnU-Net0.4619/32.5388；绝对+4.50pp。消融full为0.5101/20.2752，不能混用。另一基线nnFormer HD95 24.5692更低。

公开独立HRFuse-Net仓库API返回404；数据原论文PROTEAS为40名患者，与44未对齐；16实际页/模板15页、引用和图表问题另见报告。**只可列为方法候选，不能直接称已复现或下载模型。** [PROTEAS原始论文PDF](PROTEAS_dataset_descriptor.pdf)。

## 02 SFA-Net，多发性硬化小病灶，高优先级

**Scale-Frequency Adaptive Fusion for Multiple Sclerosis Lesion Segmentation**

[本地PDF](manuscripts/02.pdf) · [冻结原始PDF](https://raw.githubusercontent.com/yifangao112/Camyla/df4434f9d4aef5b7394ed03a4e877a8130c1b6cf/assets/paper_pdf/02.pdf) · [检索文本](searchable_text/02.txt)

MSLesSeg，115扫描；多尺度空洞卷积+频域融合+双注意力。主表0.7063/15.2980，U-Mamba0.6957/16.3300；消融另报0.7147/13.6329。可借鉴多尺度病灶保留，不能把MS模型直接当METS四类模型。

## 03 HCF-Net，新生儿脑病灶，中高优先级

**Hierarchical Context Gating for Neonatal Brain Lesion Segmentation**

[本地PDF](manuscripts/03.pdf) · [冻结原始PDF](https://raw.githubusercontent.com/yifangao112/Camyla/df4434f9d4aef5b7394ed03a4e877a8130c1b6cf/assets/paper_pdf/03.pdf) · [检索文本](searchable_text/03.txt)

BONBID2023，HIE；上下文低频门控+高频支路+可分离卷积。主表0.5823/11.9295，MedNeXt0.5585/15.8881。轻量小结构思路，病种/模态不同。

## 01 CDFL，颅内生殖细胞瘤，中优先级

**Cross-Directional Feature Lattice for Brain Tumor Segmentation**

[本地PDF](manuscripts/01.pdf) · [冻结原始PDF](https://raw.githubusercontent.com/yifangao112/Camyla/df4434f9d4aef5b7394ed03a4e877a8130c1b6cf/assets/paper_pdf/01.pdf) · [检索文本](searchable_text/01.txt)

GDMRI-CT，50例、多模态MRI，目标为germinoma而非metastases/glioma。三向特征分解、方向格交互。主表0.7387/5.5790，STU-Net0.7233/8.4485。Dice+1.54pp，不是+2.12pp；2.12%为相对值。

## 07 ScaleGate-Net，肝病灶，中优先级

**Adaptive Scale-Aware Feature Integration for Liver Lesion Segmentation**

[本地PDF](manuscripts/07.pdf) · [冻结原始PDF](https://raw.githubusercontent.com/yifangao112/Camyla/df4434f9d4aef5b7394ed03a4e877a8130c1b6cf/assets/paper_pdf/07.pdf) · [检索文本](searchable_text/07.txt)

PLC-CECT，278CT；差异引导可变形跳连+各向异性上下文。主表0.5544/85.3516，U-Net++0.5322/107.4067。消融0.5660/84.9302不是主比较值。

## 08 BHD-Net，胎脑组织，中优先级

**Boundary-Hierarchical Decomposition for Fetal Brain Tissue Segmentation**

[本地PDF](manuscripts/08.pdf) · [冻结原始PDF](https://raw.githubusercontent.com/yifangao112/Camyla/df4434f9d4aef5b7394ed03a4e877a8130c1b6cf/assets/paper_pdf/08.pdf) · [检索文本](searchable_text/08.txt)

SMRI-FB，70扫描；结构/边界分解、边界引导聚合。主表0.8574/1.1851，MedNeXt0.8530/1.2305。Dice+0.44pp，系统胜者规则在该处主要依赖HD95。

## 06 SRD-Net，腹腔镜少数类，中低优先级

**Specular-Residual Decoupled Encoding for Surgical Scene Segmentation**

[本地PDF](manuscripts/06.pdf) · [冻结原始PDF](https://raw.githubusercontent.com/yifangao112/Camyla/df4434f9d4aef5b7394ed03a4e877a8130c1b6cf/assets/paper_pdf/06.pdf) · [检索文本](searchable_text/06.txt)

Endoscapes2023；结构与反光残差拆分+少数类融合。主表0.4813/45.1510，nnU-Net0.4753/39.0041。Dice稍高但HD95更差，MRI无同样的光学反光机制。

## 04 CSM-Net，肺泡灌洗细胞，低优先级

**Cross-Scale Mutual Refinement for Bronchoalveolar Lavage Fluid Cell Segmentation**

[本地PDF](manuscripts/04.pdf) · [冻结原始PDF](https://raw.githubusercontent.com/yifangao112/Camyla/df4434f9d4aef5b7394ed03a4e877a8130c1b6cf/assets/paper_pdf/04.pdf) · [检索文本](searchable_text/04.txt)

PW-BALFC，2105显微图；互尺度细化和稀疏采样。主表0.7663/47.8435，nnU-Net0.7666/52.4164。不是Dice更高；二维细胞不可直接替代MRI四类推理。

## 05 SAC-Net，全景牙齿，低优先级

**Symmetry-Aware Cascaded Attention for Panoramic Tooth Segmentation**

[本地PDF](manuscripts/05.pdf) · [冻结原始PDF](https://raw.githubusercontent.com/yifangao112/Camyla/df4434f9d4aef5b7394ed03a4e877a8130c1b6cf/assets/paper_pdf/05.pdf) · [检索文本](searchable_text/05.txt)

STS-Tooth，850图；对称注意力+级联细化。主表0.9416/0.7943，U-Mamba0.9367/2.6072；基线更好的HD95来自UKAN0.9394。不对称转移灶不宜强制对称。

## 09 VBR-Net，OCTA血管，低优先级

**Vessel-Guided Boundary Residual Networks for Dermatological Vessel Segmentation**

[本地PDF](manuscripts/09.pdf) · [冻结原始PDF](https://raw.githubusercontent.com/yifangao112/Camyla/df4434f9d4aef5b7394ed03a4e877a8130c1b6cf/assets/paper_pdf/09.pdf) · [检索文本](searchable_text/09.txt)

DERMA-OCTA；血管引导边界残差。主表0.8381/3.0712，U-Mamba0.8391/3.9159。Dice略低、HD95更好。二维边界距离单位不与MRI的毫米直接混用。

## 获取与核验

原稿绑定同一冻结Git提交。逐项bytes/SHA256/Git blob SHA1见 [PUBLIC_ASSET_MANIFEST.json](PUBLIC_ASSET_MANIFEST.json)，页数/搜索文本SHA见 [PDF_TEXT_INVENTORY.json](PDF_TEXT_INVENTORY.json)。不改原PDF，抽取连字/公式有局限，关键数字已回到原PDF表格复核。

除HRFuse-Net的链接外，本次没有逐一验证其余9篇摘要宣称的独立代码或checkpoint仓库；不能据这些声明认定模型现成可用。CamylaNet框架可公开定位，也不等于这些生成架构的实际训练权重已经发布。
