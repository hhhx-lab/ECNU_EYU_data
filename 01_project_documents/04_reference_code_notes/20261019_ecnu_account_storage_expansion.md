# ECNU账号存储扩容只读核验

2026-10-19用户说明已申请账号存储扩容，并要求查看本账号服务器存储。此次仅显式aTrust start和一次有界SSH，查询df/statvfs/inode/findmnt/quota；未递归du、打开影像/模型、查询或修改作业、安装、删除/移动/写入远端数据。

## 当前账号容量

账号zqchen、UID2159、login1。本地收到UTC2026-10-19T14:21:29.759316，远端UTC14:10:20.685571分别记录，不混作同一时钟；本地北京时间22:21。

| 指标 | 精确bytes | TiB/GiB |
|---|---:|---:|
| 总容量 | 3298534883328 | 3.000 TiB / 3072 GiB |
| 已用 | 2010192543744 | 1.828259 TiB / 1872.138 GiB |
| 可用 | 1288342339584 | 1.171741 TiB / 1199.862 GiB |
| 相比历史2TiB容量增加 | 1099511627776 | 1.000 TiB / 1024 GiB |

df占用61%，原始bytes比例60.94198%。本账号home `/public/home/zqchen`、项目 `/public/home/zqchen/projects/ECNU_EYU_data`、R9评价目录的df和statvfs总容量/可用量完全一致，目录owner UID2159、解析路径不变。

## 账号视图与共享盘区分

同一次检查、同设备42和NFS挂载 `12.12.15.3:/public`：

- `/public`及`/public/home`共享根视图：总272678883688448bytes、可用28143650865152bytes（25.5965TiB）、占用90%。
- `/public/home/zqchen`及其项目子目录账号视图：总3TiB、可用1.171741TiB、占用61%。

由同挂载不同路径返回的账号限定容量视图，结合用户扩容申请和历史账号2TiB记录，可以确认扩容已在本账号路径的容量报告中生效为3TiB；不是拿共享盘25.60TiB当账号额度。未做写入边界试验，不承诺可突破quota，也不自动授权其他项目物化数据。

`quota -v -u zqchen`退出0、stdout/stderr均空，无软配额/硬配额/宽限期明细，不能把空输出当无限额度。inode报告同共享挂载：总2721923868、空闲2232175947、18%占用；未返回账号专属文件数限额。

上午R9快照只有单一项目路径df变成3TiB，故当时保留“账号quota未独立核验”；本次home/共享根对照补齐容量解释，不改写原始快照。历史179.822GiB是扩容前观测，不再作为当前账号剩余量；未来新增任务仍须当次精确容量与自身预算核查。

## 审计与边界

独立本地工具 `operator/account_storage.py` 不在原R9评价24文件冻结manifest内，原runtime/包SHA不改。本次VPN owned=true且finally stop成功，本地完成UTC14:21:31.199163；用户现有其他会话未处理。

原始证据：[存储快照](../../work_space/G1/03_experiments/03_diffusion_training/20261018_r9_fixedval_selection_r1/operator/account_storage_20261019T142105Z/ACCOUNT_STORAGE.json)、[会话回执](../../work_space/G1/03_experiments/03_diffusion_training/20261018_r9_fixedval_selection_r1/operator/account_storage_20261019T142105Z/LOCAL_OPERATION_RECEIPT.json)。

扩容不是训练/生成评价完成证据。CPU准备与四A100排队是上午的历史状态，本次没有查询新进度或GPU资源，未放行全量评价/重提任务/恢复PAUSED monitor。test104/旧失败/人工质量待验与experimental_unvalidated、operator_approved=false、formal_gate_status=not_run_not_passed保持。
