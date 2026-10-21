# G1服务器存储与重复大数据只读核查

- 用户请求：查询服务器容量与近期多attempt是否反复保存大数据；不授权删除、迁移或重投。
- 本地快照完成UTC：2026-10-14T08:13:24.324404+00:00（北京时间16:13:24）。
- 远端采样开始UTC：2026-10-14T08:00:19.102042+00:00；两端时钟独立记录，不推算耗时。
- 原始证据：[STORAGE_INVENTORY.json](operator/storage_readonly_20261014T081114Z/STORAGE_INVENTORY.json)。
- artifact_status=experimental_unvalidated；operator_approved=false；formal_gate_status=not_run_not_passed。

## 方法与边界

本次显式启动约定分流VPN，经一次有界SSH读取本地/远端时间、hostname、df/statvfs、五个R5作业squeue/sacct、G1与账号home的du及近期attempt文件metadata，结束只关闭本次VPN。远端没有写入、删除、移动、提交作业或下载大影像/权重；仅本地新增操作器、轻量JSON及报告。G1 du与各attempt du均exit0完成；home一级du在90秒超时，未取得全账号各目录完整分解，不重复扫描。

du采用`-x -B1`且不跟随symlink；表内父子目录不能相加。文件计数/大小/allocated blocks可以判定是否存在新全量输出，不能证明不同目录逐字节相同；本次不读取全量影像SHA或做跨目录内容去重。当前扫描是有界时间窗口，不是文件系统原子快照。

## 账号容量

| 口径 | 总容量 | 已用 | 可用 |
|---|---:|---:|---:|
| 账号home路径df -B1/statvfs | 2048 GiB | 1839.23 GiB | 208.77 GiB |
| /public共享盘df -h | 248 TiB | 223 TiB | 约26 TiB |

账号路径可用224162480128 bytes。共享盘余量不能替代账号路径限制；具体服务端quota/fileset机制仍未核实。相对原72.89 GiB，先前授权删除旧SegMamba-main释放的空间仍反映在账号可用值中，本次未清理。

## G1主要占用

G1实际总占用608206942720 bytes，即566.44 GiB，以下为其中主要项。

| 目录/内容 | 占用GiB | 解释 |
|---|---:|---|
| epoch004原生空间全量重建r2 | 355.26 | 1030例、5150个NIfTI；4120影像+1030标签，COMPLETE报告pass。现训练源，只有这一份近期全量重建输出 |
| 旧V3/data | 109.81 | 历史派生mask、weights、latent、缺失T2W补全及评价；不是近期R1-R5全量复制 |
| 旧V3/training | 83.67 | 历史模型/训练产物，不是新训练数据 |
| 历史threeway validation attempt03 | 8.43 | stage1_reconstruction影像；与新全量数据是否相同未逐文件核验 |
| epoch004微调训练attempt03 | 4.69 | 主要training产物 |
| 旧单模型r4评价 | 2.65 | 三路评价影像/证据，与全量新训练表示不同 |
| epoch004重建r1 smoke | 0.94 | 两病例smoke历史输出；r1没有全量dataset |

V3/data子项：attention_masks40.87 GiB、lesion_weights40.87 GiB、旧补全output16.38 GiB、latents10.06 GiB、evaluation1.63 GiB。这些是不同用途的表示，不能仅凭相同病例或大小宣称字节重复或可删；此前高分辨率mask/weight约80.47 GiB的可复现空间合同仍有缺口，本次未解除保护。

V3/input与input_inference是指向V2入口的symlink，不能当作又一份全量实体数据；本次du没有解引用再次计费。原始MRI位于共享data来源，与VAE重建/latent/1mm不同表示，不能混为重复拷贝。

## 近期1mm Diffusion各attempt

| attempt | 实际占用MiB | dataset NIfTI | training checkpoint |
|---|---:|---:|---:|
| R1 | 0.54 | 0 | 0 |
| R2 | 7.36 | 0 | 0 |
| R3 | 2.00 | 0 | 0 |
| R4 | 3.15 | 0 | 0 |
| R5 | 2.04 | 0 | 0 |
| 合计 | 15.09 | 0 | 0 |

五个attempt的dataset/training目录均未生成；主要是独立冻结源码、配置、预检/失败/提交审计及轻量完整性manifest。R2 P0虽然检查过3730文件，但读取已有重建源，并未复制这3730影像。R3/R4在实际预处理与sbatch前停止，保留失败证据不造成大数据复制。

**结论：近期反复修复R1-R5没有重复存几版数百GiB大数据。** 确有原始影像、旧latent/派生数组、新epoch004原生空间重建这些历史/不同表示并存；字节级冗余尚未全量核验，不能直接删除。

## 同次作业与后续空间

同次squeue/sacct显示CPU4069915 PENDING Priority；四A100训练4069916-4069919均PENDING Dependency，依赖4069915。R5 logs为空、dataset/STARTED/FAILED/COMPLETE和training不存在，符合未启动，不是消失/失败；PENDING的0:0不代表成功。

四个训练消费同一个`R5/dataset`（runtime/train.py），不会各自生成一套1mm全量数据。未来新增的是一套926例1mm派生影像/共同mask及四路checkpoint；它与原生空间数据表示不同，不能在训练前删原生重建或真实源mask输入。

R5冻结最坏空间预算213513415862 bytes（198.85 GiB，含metadata/临时输出/四路完整checkpoint/额外余量）；本次可用比预算多9.92 GiB。现在尚未占用这部分，但开始预处理后账号可用空间会下降；不能把当前208.77 GiB当作训练后仍剩208.77 GiB，也不能据此授权其他作业占用余量。

本次只整理存储事实，不修改原冻结runtime/SHA/训练合同、不创建新attempt、不接触无关作业或删除任何历史失败。
