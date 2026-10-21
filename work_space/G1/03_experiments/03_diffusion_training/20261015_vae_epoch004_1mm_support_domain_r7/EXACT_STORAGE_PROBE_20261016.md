# 精确共享存储核查与小文件测试门

用户授权“试试”。本次一次有界SSH/VPN操作；本地UTC2026-10-15 16:29:13至16:29:34（北京时间2026-10-16 00:29），远端UTC16:18:26至16:18:30，login1。时钟分别记录，不混作耗时。

`artifact_status=experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。

## 结果

| 路径 | 权限/容量实测 | 判断 |
|---|---|---|
| `/hpc_stor/data/zqchen` | 不存在 | 旧手册指定专属路径尚未分配，不能测试 |
| `/hpc_stor/data` | root:root 0755，本账号不可写，NFS `12.12.15.4:/hpc_stor`，共享空闲约2.20 PiB | 不自行创建用户目录；需管理员授权分配 |
| `/public/share` | root:root 0777，ACL other/group rwx，本账号write_access=true；NFS `12.12.15.3:/public`，返回共享余约25.48 TiB | 技术权限可写，但不是独立配额/使用授权/长期保留证明 |
| `/public/share/zqchen` | 不存在 | 无已分配本账号专属目录；不直接占用共享区 |
| `/public/home/zqchen` | 0700、可写、总2 TiB、余200.93 GiB | 原账号空间未扩容 |

本次以现有findmnt文本格式成功确认两个NFS挂载；原JSON选项不兼容失败保留，不安装工具。quota返回0但空明细，具体账号/路径/group/fileset额度机制仍未知。

只查看`/hpc_stor/data`及`/public/share`顶层名称以README/NOTICE/POLICY/USAGE/RULES开头且root拥有、非可写、32KiB以下的说明文件；符合条件文件为0。这不证明没有管理员规则或自动清理策略。没有进入其他用户数据目录。

## 为什么没有小文件写入

预设测试门要求：专属路径已经存在、非symlink、路径不重定向、由本账号拥有、可写可进入、无group/other写权限；通过后才用唯一名字/O_EXCL/O_NOFOLLOW创建0600的1KiB探针、fsync、读回和SHA。两个精确候选均不存在，测试门未满足，`small_write_test_performed=false`。

`/public/share`的0777还没有sticky bit，其他用户具有父目录写/执行权限，存在改名或移除目录项的风险；内层文件设0600也不能证明父目录安全。没有直接向共享根创建目录/试写文件，不凭宽松权限绕配额或放入145GiB概率。

此次没有远端新产物、清理、迁移、配额更改或作业提交。未声称写入成功，更没有证明独立配额或计算节点可见性。旧失败、源数据、冻结runtime和SHA不改，概率仍仅服务器且容量hold。

## 可实际推进的选择

1. 优先请管理员创建并授权`/hpc_stor/data/zqchen`，确认至少500GiB项目额度、计算节点可访问、保留期限和费用。
2. 若管理员确认`/public/share`允许科研持久存储，应由管理员提供受保护的项目专属路径、访问/删除隔离及额度规则，而不是直接使用世界可写根目录。
3. 获得实际专属路径后，再单独执行已准备的1KiB写读SHA探针及计算节点可见性验证；当前不把共享盘free代替账户容量门。

尚未发送管理员申请、联系负责人或进行平台操作。

## 证据

- [精确路径/ACL/NFS/容量与测试门结果](operator/exact_storage_probe_20261015T162913Z/EXACT_STORAGE_PROBE.json)
- [SHA与VPN关闭操作回执](operator/exact_storage_probe_20261015T162913Z/LOCAL_OPERATION_RECEIPT.json)
- [此前浅层探测](SHARED_STORAGE_PATH_PROBE_20261016.md)
- [存储分配与保守预算](STORAGE_ALLOCATION_READONLY_20261016.md)

本地JSON/回执SHA一致、未测试与无写入标记一致、VPN已关闭；核查完成不等于存储可用验收通过。
