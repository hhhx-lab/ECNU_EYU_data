# ECNU服务器清理候选：只读筛查

汇总UTC：2026-10-13T14:25:03Z。`artifact_status=experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。用户仅授权只读查看；**没有删除、移动或修改服务器文件，没有清环境、安装依赖、提交/取消作业或恢复P0**。只新增本地查询脚本、轻量审计和总纲说明。

## 范围与时间

远端根为`/public/home/zqchen`，主机`login1`。候选盘点的远端UTC为2026-10-13T14:09:14.124820+00:00，本地接收UTC为14:20:16.783036+00:00；补充验证远端UTC为14:10:32.438947+00:00，本地接收UTC为14:21:34.983312+00:00。两端时钟约差11分钟，未同步时钟，不能把此差值当操作耗时。

盘点时路径可用78263615488 bytes（72.89 GiB）。前序容量报告确认账号路径返回2 TiB、97%已用；共享`/public`的25.67 TiB余量不是账号可支配余量，具体quota/fileset机制待管理员确认。

大小采用`du -x -B1`或文件`st_blocks*512`，GiB=bytes/2^30。父子目录不重复相加；释放量只是候选估计，最终还受硬链接、打开文件和服务端记账影响。

## 较低风险的缓存候选

以下路径相对远端根。它们不是已安装环境或实验结果；仍须明确授权并确认无下载/环境维护并发操作后才能清理。清除后可能无法离线重装，本次未承诺自动重新下载。

| 精确范围 | GiB | 核查和限制 |
|---|---:|---|
| `.cache/pip` | 9.20 | 完整du；主要为HTTP下载缓存，不是site-packages。保留离线恢复需求需另确认 |
| `.conda/pkgs`顶层618个`.conda`/`.tar.bz2`归档文件 | 10.19 | 已逐文件记录路径、大小和链接数；均nlink=1。只指压缩包，不包括解压包目录、envs或整个pkgs |
| `.conda/pkgs/cache` | 1.17 | du已输出该子树大小；属于索引/repodata缓存，整体pkgs扫描超时，不宣称pkgs总量完成 |

上述合计22078486528 bytes（**20.56 GiB**）。假设全部实际释放且无其他变化，可用量约93.45 GiB；**不足以证明926例1mm预处理、四路200k训练和完整恢复checkpoint的存储预算已满足**。

## 较大但尚未可删的解压缓存

| 远端根下范围 | 病例数 | `.npy`与`_seg.npy`合计GiB |
|---|---:|---:|
| `SegMamba-main/data/fullres/train` | 1296 | 85.62 |
| `SegMamba-main/data/validation_preprocessed` | 179 | 8.99 |
| 合计 | 1475 | **94.61** |

1475个文件stem均同时存在`.npz`、`.pkl`、`.npy`和`_seg.npy`。读取源码确认`MedicalDataset`初始化调用`unpack_dataset(overwrite_existing=False)`，缺失数组会从`.npz`的`data`/`seg`重新解压；读取数据使用数组mmap `r+`，因此不能仅凭文件名断言所有现存数组从未修改。

只读抽查train的`BraTS-MET-00001-000`、`BraTS-MET-00002-000`及validation的`BraTS-MET-00833-000`：全部data/seg的shape、dtype相同，finite与逐元素一致通过。**仅3例抽查，不是1475例全量验收**；未运行项目loader或自动解压函数，没有重新生成数组。

清理这94.61 GiB的前提：独立全量验证压缩包可读且数组内容一致，保留每例`.npz`、`.pkl`及split/来源清单，确认没有训练/读取/维护进程依赖这些数组，获得明确删除授权。再次运行旧SegMamba时会重新占用约同量空间；不属于永久节省。不得删整个data、validation或SegMamba-main目录。

## 其他候选与不推荐项

- `SegMamba-main/mamba/build`0.343 GiB及`SegMamba-main/causal-conv1d/build`0.057 GiB：合计约0.40 GiB编译中间物候选。已查选定环境模块路径与`.pth`，未发现指向这些build的引用，login本账号进程maps未见映射；这不是所有环境/节点/未来工作流无依赖的证明。完整确认后才可授权，保留源码和安装后的环境。
- `.npm/_cacache`约0.062 GiB：下载缓存候选，收益很小。
- `.cache/puppeteer`约0.61 GiB：包含Chrome和headless-shell可执行资产，**不默认清理**；删除可能破坏浏览器工作流，本次也不授权下载替代品。
- `.vscode-server`约0.061 GiB：远程开发运行组件，不默认清理；`.triton`/`.nv`各不到1 MiB，无实质收益。
- `.conda/pkgs`整体25秒扫描超时，`.conda/envs`15秒扫描超时；不把未完整范围算0，也不以“缓存”名义删除解压库或整个环境。没有确认的大型临时文件/孤儿产物，不编造可删额度。

## 必须保护

- G1 epoch004全量重建r2约355.26 GiB，用户要用于新Diffusion；不删、不拉回或上传Drive。
- 原始真实MRI、corrected标签、G2成员表、split、原E与所选Diffusion模型、K1/K2/K4及优化器/续训checkpoint。
- 官方MET、paired、选择证据、预注册方案、SHA、冻结runtime、所有失败和历史审计。旧V3的数据/训练占用不等于垃圾。
- `zdn_deepcpg`约874.70 GiB等其他研究目录，归属和备份未核，不能纳入本次可删范围。

盘点`squeue -h -u zqchen`exit0且为空，只证明该次队列查询无条目。`ps`格式返回不完整，不能据此认定本账号无后台进程；补充maps只涵盖login节点。本报告没有给任何目录出具无运行依赖的全局证明。

## 原始证据与后续入口

- [候选盘点JSON](../operator/readonly_cleanup_candidates_20261013T141909Z/CLEANUP_CANDIDATES.json)：618压缩包逐文件清单、1475例格式统计、缓存du、容量和队列。
- [补充验证JSON](../operator/readonly_cleanup_verification_20261013T142116Z/CLEANUP_VERIFICATION.json)：3例逐元素比较、loader源码片段、选定环境路径和login进程build映射。
- [账号存储占用](STORAGE_OCCUPANCY_20261013.md)、[旧SegMamba身份](SEGMAMBA_DIRECTORY_IDENTITY_20261013.md)。

两次操作均显式调用分流VPN入口，并在finally仅关闭本次拥有的VPN；执行包装器成功结束。远端仅读取。后续若授权清理，须另建精确白名单、清前依赖/备份验收和独立清后容量回执；本报告不是清理授权，也不放行被冻结的P0/训练。
