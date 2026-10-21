# ECNU账号存储占用只读核查

本地汇总UTC：2026-10-13T13:20:21Z（北京时间21:20:21）。研究登记保持`experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。本次仅存储查询，不授权清理、迁移、修P0或重提训练。

## 容量结论

账号`zqchen`（uid2159）在login1通过NFSv3挂载`12.12.15.3:/public`。同一次查询：`/public`返回总容量272678883688448 bytes（248 TiB）、可用约25.67 TiB；`/public/home/zqchen`及其data/projects/ECNU项目/实验路径均返回总容量2199023255552 bytes（2 TiB）、可用78263615488 bytes（72.89 GiB）、已用1975.11 GiB（97%）。后续定向读取中可用量相同。

**72.89 GiB不是仅新实验目录的剩余量，而是我们使用的整个账号目录树当前返回的可用量。** 共享盘尚有空闲，不代表账号可调用全部空闲。路径相关2 TiB容量表现与服务端目录/账号限制一致，但`quota -vs -u zqchen`虽exit0却无stdout，登录节点没有lfs/mmlsquota/xfs_quota；尚未读到正式quota条目，不能断言限制是user quota、fileset还是其他机制，应请管理员确认2 TiB额度与扩容方法。

inode余量约22.36亿，不是本次主要限制。没有尝试写入探针、管理员权限或安装工具。

## 主要目录已完成统计

使用`du -x -B1`，统计实际分配字节，包含隐藏内容、不解引用symlink、不跨文件系统。GiB为bytes/2^30。下表目录是独立范围，但各次独立du遇到共享硬链接时可能重复计数，不能拿其和作为配额精确对账；没有完整home总量du结果。

| 远端目录（相对/public/home/zqchen） | GiB | 状态 |
|---|---:|---|
| zdn_deepcpg | 874.70 | 完整 |
| projects/ECNU_EYU_data/work_space/G1 | 566.42 | 完整 |
| SegMamba-main | 135.90 | 完整 |
| projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated | 82.32 | 完整 |
| projects/ECNU_EYU_data/work_space/S2/data/ecnu_completion_emergency | 68.11 | 该子目录完整 |
| data | 31.17 | 完整 |
| projects/ECNU_EYU_data/work_space/S3 | 21.87 | 完整 |
| yy_grapecpg | 17.36 | 完整 |
| .cache | 9.81 | 该子目录完整，其中pip9.20 |

## G1与S2细分

以下是父目录中的组成，**不能再和上表父目录相加**。

| 范围 | GiB |
|---|---:|
| G1/results整体 | 371.96 |
| G1 epoch004全量重建r2 | 355.26 |
| G1/code整体 | 194.46 |
| G1 V3/data | 109.81 |
| G1 V3/training | 83.67 |
| G1 threeway fixed103 attempt03 | 8.43 |
| G1 VAE延长微调attempt03 | 4.69 |
| S2 FixV3 full200 r1 | 57.79 |
| S2 B-F screen125 r1 | 9.01 |
| S2 B/E/F extend175 r4 | 3.67 |
| S2 K4-175 r6 | 2.28 |
| S2 K2-175 r2 | 2.21 |
| S2 K1-175 resume r1 | 2.18 |

epoch004重建355.26 GiB是输出当前分配大小，不等于真实源data31.17 GiB；重建浮点图像压缩效果可能不同，本次没有读取NIfTI体素或将两者等同。V3/code内有data/training，所以“code占194 GiB”不表示Python源码本身194 GiB。

## 未完整范围与边界

- 首次home全树du150秒超时，未得到home总量，保留部分输出。
- S2整体50秒扫描超时；随后S2独立实验区24.24秒完整，但不据此称S2整体完整。
- zjd_projects、hzj_project、yy_mambacpg、dogs-vs-cats及projects/financial_ts_overlap_20260819_ecnu各12秒超时，未给其0值。
- projects/zjd_projects有权限拒绝；已报告5.52 GiB为部分值，不作为完整占用，不改变权限。
- 远端work_space/S1这个路径不存在，不等于整个账号没有S1数据。work_space/S3/S4确实存在，只做大小统计，不改其研究分类。
- 没有清理pip/cache/checkpoint、原始数据、重建数据、失败目录或其他研究目录；没有提交/取消Slurm任务、安装依赖或下载大型文件。每次显式启动分流VPN，结束只关闭本次VPN或保留既有VPN。

## 后续建议

针对zdn_deepcpg和G1进一步核身份，见[两项目组成与G1候选分级](TWO_PROJECT_IDENTITY_AND_G1_CLEANUP_20261013.md)。zdn是独立单细胞CpG甲基化预测，其人/鼠H5占851.76 GiB，非MRI缓存；G1高分辨率mask/weight80.47 GiB有原空间transform缺失阻塞，checkpoint及补全中间图仅待归档审查，当前不直接可删。

用户进一步要求先只读筛查清理候选，见[分级候选与保护清单](CLEANUP_CANDIDATES_READONLY_20261013.md)：较低风险下载包/索引缓存20.56 GiB；SegMamba解压数组94.61 GiB须全量一致性与依赖核验，不把3例抽查通过当作可删授权。实际未清理，训练容量门未放行。

用户随后追问SegMamba-main身份，见[独立只读说明](SEGMAMBA_DIRECTORY_IDENTITY_20261013.md)：实际是旧SegMamba BraTS分割分支，不是Conda环境或Diffusion；135.90 GiB中data129.91（train111.96、validation预处理12.46、validation5.06），results5.45 GiB。真实训练代码实例化SegMamba，已有6月底至7月初mtime的CE/Dice/Focal、大patch及更深模型产物，不能绑定现行原E/K路线或直接归为S1。未清理。

先向管理员确认账号路径2 TiB容量合同及可扩容量，不把总盘25.67 TiB直接当可用额度。若另安排归档，先确认目录归属、完整备份与恢复证据，并获明确清理授权；`zdn_deepcpg`等账号内其他项目不是本次可删除范围。缓存约9.8 GiB也不足以解决大规模1mm新数据与训练存储预算。

P0仅轻量证据的100 GiB断言仍不合理，但此次查询不修改冻结attempt；P1的926例1mm数据、完整optimizer/checkpoint另估容量并门控，不能因为P0可执行就推定训练空间足够。

## 原始证据

- `operator/readonly_storage_inventory_20261013T131040Z/STORAGE_INVENTORY.json`：容量、mount、quota及部分home du。
- `operator/readonly_storage_targeted_20261013T131416Z/STORAGE_TARGETED.json`：G1/data完整、S2部分及目录清单。
- `operator/readonly_storage_experiments_20261013T131646Z/STORAGE_EXPERIMENTS.json`：G1批次/V3及S2独立实验区完整。
- `operator/readonly_storage_remaining_20261013T131833Z/STORAGE_REMAINING.json`：其他主要目录、超时和权限情况。

远端时间分别保留原JSON，约比本机慢11分钟，未进行时钟同步。初次STORAGE_INVENTORY的`remote_utc_started`字段在du之后构造，并非du真实开始时间；时长以`elapsed_seconds`150.011记录为准，不以该字段倒推扫描起始。
