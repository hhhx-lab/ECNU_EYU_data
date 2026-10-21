# 共享存储路径只读探测

北京时间2026-10-16 00:22:43接收；远端UTC2026-10-15 16:11:28至16:11:38，本地UTC16:22:25至16:22:43。主机login1，两端时钟分别保存。

用户要求尝试共享盘存储，本次仅一次有界SSH/VPN查询路径存在性、ACL、statvfs及根目录浅层列表。没有创建目录或写测试文件、迁移、绕过配额、提交任务或拉回大数据；VPN关闭。

`artifact_status=experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。

## 实测结果

| 路径 | 观测 | 当前结论 |
|---|---|---|
| `/public` | root:root、0755、write_access=false；共享空闲约25.48 TiB | 不能直接在共享根目录建实验目录 |
| `/public/share` | 根目录浅层stat显示root:root、0777 | 有共享入口，但未对该路径独立查ACL/statvfs/使用规则；权限不证明存储授权或独立额度，不写入 |
| `/public/home/zqchen` | zqchen:users、0700、可写；2 TiB总容量、216084250624 bytes可用（约201.24 GiB） | 仍是原账号空间，没有新增额度 |
| `/public/home/zqchen/projects/ECNU_EYU_data` | zqchen拥有、可写；同2 TiB容量 | 换原项目子目录不能证明扩容 |
| `/public/zqchen` | 不存在 | 没有核到该专属共享目录，不自行建立 |
| `/hpc_stor` | 存在、root:root、0755、不可写；device43不同于/public的42；总2.50 PiB，共享空闲约2.20 PiB | 是可进一步申请的其他存储，不是本账号已获配容量 |
| `/hpc_stor/home/zqchen`、`/hpc_stor/users/zqchen`、`/hpc_stor/zqchen` | 均不存在 | 本次这三个候选未核到专属路径 |
| `/scratch`、`/work`、`/storage` | 不存在 | 本次登录节点未找到这些路径 |
| `/mnt` | 是登录节点本地盘、root拥有不可写 | 不是已分配的持久实验空间 |

本次groups只有users(100)，未发现额外课题组授权证据。quota命令返回0但空明细。findmnt的JSON选项不被当前版本支持，返回1并保留诊断；没有安装或再次执行。statvfs与device只能证明观测到不同存储，不能替代挂载参数/配额合同。

## 缺口与下一步

旧[S2运行手册](../../../../S2/05_documents/03_operations/S2_服务器运行手册.md)明确推荐的是`/hpc_stor/data/zqchen`，由管理员创建并授权。本次根目录确认有`/hpc_stor/data`，**没有进一步检查`/hpc_stor/data/zqchen`，不能说它不存在或不可用**。旧手册不是当前已授权证明。

最有价值的后续为：一次独立只读核查精确`/hpc_stor/data/zqchen`和`/public/share`的ACL/路径配额，并向管理员确认能否分配本项目专属目录、容量、计算节点可见性、保留/清理期限及费用。不要扫描其他用户数据。

只有确认专属目录或明确共享使用授权后，再独立做小文件写入/读回/SHA验证；无需现在用145 GiB概率试写。权限0777尤其不能当长期数据可靠性保证。本次没有发出管理员申请、增加配额或改变现有G1/S2合同；K1概率继续服务器容量hold。

## 原始证据

- [路径/ACL/容量快照](operator/storage_paths_readonly_20261015T162225Z/STORAGE_PATHS.json)
- [只读与VPN关闭回执](operator/storage_paths_readonly_20261015T162225Z/LOCAL_OPERATION_RECEIPT.json)
- [此前容量与预算报告](STORAGE_ALLOCATION_READONLY_20261016.md)
