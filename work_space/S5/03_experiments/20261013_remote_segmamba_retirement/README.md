# S5旧SegMamba路线停用与轻量留存

用户于2026-10-13明确决定不再做这条路线，授权只留本地模型参数/结果记录并删除远端`/public/home/zqchen/SegMamba-main`。该路线是S5 SegMamba，不是S3 SwinUNETR、S4 SAM2-UNet或S2 nnU-Net原E/K路线。当前状态：**远端指定目录已删除，独立复核确认不存在**；本地原S5源码仍在。

`artifact_status=experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。删除授权是工程保留决策，不等于科学质量批准。

## 留存范围

- `preserved/`：134份远端原代码、8配置、Slurm、评价实现、CSV/JSON指标表及曲线PNG，合计7641986 bytes（7.29 MiB），逐文件SHA通过。
- [LIGHT_ARCHIVE_RECEIPT.json](LIGHT_ARCHIVE_RECEIPT.json)：8749项目录/文件metadata清单、21个checkpoint名称和大小、48份日志头尾节选、20个events文件metadata，以及留存文件SHA。
- [DELETION_RECEIPT.json](DELETION_RECEIPT.json)：删除授权范围、目录身份/清单一致性、定向依赖检查、删除前后容量、远端/本地时间。
- [POST_DELETION_VERIFICATION.json](POST_DELETION_VERIFICATION.json)：独立只读确认目标不存在、保护目录仍在和容量。

**不保存checkpoint权重、训练/验证影像、预测NIfTI、完整events和完整长日志。轻量记录不构成完整备份，不能完整恢复旧训练、模型推理或逐体素复核。** 本地已有S5源码与历史文档不删除；旧失败记录和前序只读审计继续保留在原本地目录。

## 模型参数记录

以留存的`preserved/configs/*.yaml`为准，这是删除前配置快照，可能已在历史运行后被改写，不能推定就是每份旧权重的原始合同。

共同架构：4模态输入、5标签输出，feature `[48,96,192,384]`；普通depth `[2,2,2,2]`，更深模型 `[3,3,6,6]`。普通crop128³，大patch192³；batch2、seed123、每2 epoch验证。普通优化器SGD，lr0.01、weight_decay0.00003、momentum0.99、Nesterov，poly调度。AdamW配置lr0.0001、weight_decay0.01、cosine。默认推理overlap0.5，高重叠配置0.75；小病灶配置阈值27 mm³。未加载权重或实例化模型计算参数总数。

| 配置 | 当前max_epoch | 核心差别 | 保留checkpoint文件名best/final标量 |
|---|---:|---|---|
| CE+Dice | 170 | CE1+Dice1 | 0.6230 / 0.6096 |
| Dice+Focal | 300 | Dice1+Focal0.5 | 0.6483 / 0.6483 |
| CE+Focal | 1 | CE1+Focal1 | 0.6544 / 0.6067 |
| 大patch | 160 | 192³、CE+Dice | 0.6793 / 0.6426 |
| 更深模型 | 300 | depths `[3,3,6,6]` | 0.3830 / 0.2947 |
| AdamW+Cosine | 300 | 优化器/调度改变 | 未见pt，不记完成 |
| 无增强 | 5 | augmentation=false | 未见独立pt，不记完成 |
| 高重叠推理 | 5 | overlap0.75 | 未见独立pt，不记完成 |

表中best/final是**原文件名标量**，未绑定官方MET、准确split或原训练终态；当前yaml CE+Focal仅1轮却有ep199权重等差异说明不能反推实际训练轮数。

## 历史结果原始值

三份`segmentation_metrics.csv`各1299行，包含1296病例及`mean/std/median`三汇总行。下面直接引用原`subject_id=mean`，不把汇总行当病例再次平均，不解释为fixed103官方成绩。原CSV中部分病例指标为空，汇总的NaN/缺失处理及评价权重SHA没有独立重验；保留原件供追溯。

| 实验 | ET Global DSC | RC Global DSC | TC Global DSC | WT Global DSC |
|---|---:|---:|---:|---:|
| CE+Dice | 0.626517 | 0.091841 | 0.642285 | 0.670128 |
| Dice+Focal | 0.654418 | 0.093350 | 0.672373 | 0.701766 |
| CE+Focal | 0.572576 | 0.141803 | 0.598007 | 0.635947 |

| 实验 | ET Instance F1 | RC Instance F1 | TC Instance F1 | WT Instance F1 |
|---|---:|---:|---:|---:|
| CE+Dice | 0.572535 | 0.113637 | 0.582834 | 0.561581 |
| Dice+Focal | 0.606697 | 0.109858 | 0.615932 | 0.595003 |
| CE+Focal | 0.503145 | 0.187906 | 0.510553 | 0.540632 |

对应原CSV/JSON在`preserved/results/<experiment>/metrics/`。原summary标题虽含Official，不自动证明评价版本、病例集或无训练泄漏；不混入S2 fixed103、原E/K paired，不宣称正式优越性。1296病例指标集与旧train1296同数，成员身份/是否训练集评价未逐例核实，不能当独立验证。

## 删除和空间验收

删除前目标不是symlink、uid2159、inode507637312259、dev42；8749项metadata与轻量保存时一致，无内部链接或跨设备条目。已核队列空、login本账号进程无目标引用、已知G1/G2/S1/S2/S3/S4/S5数据入口和选定环境模块/`.pth`无直接目标引用。全项目扫描曾超时，最终定向检查**不是账号全历史依赖/全部节点证明**。

三次操作在删除前停止：参数长度限制、Python全项目遍历超时、系统find全项目链接遍历超时；全部失败回执保留。最终r4仅执行一次指定tree删除，未取消或提交Slurm、未安装依赖、未删其他路径。每次显式分流VPN并仅关闭本次拥有的VPN。

实际删除远端UTC2026-10-13T14:53:33.456266+00:00至14:53:45.049855+00:00，本地接收UTC15:04:47.624240+00:00。独立复核远端UTC14:54:54.583054+00:00、本地UTC15:05:57.112075+00:00；时钟约差11分钟，不当耗时。

删除前可用78263615488 bytes（72.89 GiB）；立即回执222775214080 bytes（207.48 GiB），稍后独立复核224188694528 bytes（**208.79 GiB**）。最终观测增加145925079040 bytes（**135.90 GiB**），与原du135.90 GiB基本相符，记账可能延迟，不宣称逐文件释放精确对账。

独立复核确认`.conda/envs/segmamba`、共享`/public/home/zqchen/data`、项目G1/S2/S3/S4均存在。没有删除本地S5、G1 epoch004、原E/K、zdn_deepcpg或账号缓存；没有重启新Diffusion训练，容量改善不自动通过其余数据/工程/质量门。
