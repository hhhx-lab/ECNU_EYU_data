# 2026-10-18 aTrust 校园接入适配与验证

本次输入是用户新安装的aTrust、原校园/超算env和原分流脚本；目标为更新登录与分流方案，不继续旧训练查询或下载。

**当前入口补充（优先于下文双文件历史）：**用户随后要求单sh和本机短README，已将Python原实现内嵌到 `服务器登陆/ecnu-vpn-split.sh`，不再需要独立py；本机只留脚本、原env、README。详细三平台指南/模板及原版本移入[接入文档区](atrust_access/README.md)，历史原字节与SHA保留。[单文件验证](atrust_access/SINGLE_FILE_VERIFICATION_20261018.md)记录20项离线测试及PTY只读状态通过，不冒充重新实机登录或Windows/Linux验收。下面的双文件/外部长指南路径是此前历史。

## 实际动作与结果

外部工作目录 `/Users/hwaigc/比赛+课题/服务器登陆/` 的原 `ecnu-vpn-split.sh` 已适配aTrust 2.5.16.31；新增 `ecnu_atrust.py` 和16项离线安全测试，更新同目录指南。原脚本/指南保留于 `history/20261018_openconnect/`，没有把凭据复制入项目。

当前学校入口为 `https://vpn.ecnu.edu.cn`，SSO为 `https://sso.ecnu.edu.cn`。现有Conda与Playwright MCP驱动隔离Chrome读取原env完成认证，两次完整自动start通过。默认hpc范围保持59.78.189.132/133/195三个 `/32`，实机只移除Sangfor宽范围IPv4路由217条，公网三个探针和默认出口恢复原en0，DNS配置与系统代理不变。只读SSH exit0与交互登录/exit通过，原实验操作器按旧字符串判归属的接口兼容保留，不改冻结实验源码。

最新成功分流本地UTC `2026-10-18T06:12:52.622169Z`；正常注销本地UTC `06:13:05.365454Z`。只读SSH回执本地UTC `06:09:07.423101Z`，远端返回 `05:58:00Z`，两端时钟偏差分别保留。最后只读status仍为logout/tunnelUnavailable。16项最终离线测试、bash语法/Python AST通过；最后异常安全补强没有再次发起实机登录。

## 证据位置与边界

同日后续用户要求可分享版本并补Windows/Linux：外部新增 `ECNU校园VPN分流连接指南_通用版.md` 和不含真实账号/作者路径的 `.env.example`，说明现有sh加载配置、py实现功能且macOS当前入口两份都需保留。通用指南分别列macOS复用配置、Windows PowerShell、Linux策略路由与回退验收；Windows/Linux和其他机器未实测，优先校方允许的按资源分流，不绕过客户端强制策略。本轮只写文档/空模板并做静态及独立读者审查，不改现有脚本/env、不联网或改变训练证据。

读者审查指出的连接前快照顺序/Mac只读命令、Windows独立接口变量/持久路由门/确认与唯一匹配、Linux via回退示例已补齐。空模板bash语法、Markdown围栏/本地链接、作者身份和真实凭据字面量检查通过；本机未安装PowerShell，没有执行Windows/Linux命令或跨平台实机验收，未因文档任务安装依赖。

此前完整工程验收及当时脚本SHA原文件已移至[历史验收](atrust_access/history/20261018_two_file/ATRUST_SPLIT_ACCEPTANCE_20261018.md)；当前单文件SHA见上面的单文件验证，旧验收不改写。

私有原始回执：`/Users/hwaigc/.ecnu-vpn/atrust/20261018T061252622169Z_split_verified.json`、`20261018T061305365454Z_logged_out.json`、`20261018T060907423101Z_ssh_test.json`。回执未复制进项目，凭据仅留在原env。

本地agent使用安装包CA核验链/主机名，没有绕过学校TLS、安装全局CA、改PATH、安装依赖或修改其他程序/PF。厂商DNS的PF packet路径不完全等于旧方案，ecnu/custom仅离线验证；重连/换网/升级后需start/check，Bridge模式本次超时未稳定验收。

本次只恢复接入能力：没有新的G1 R9终态、checkpoint、squeue/sacct或正式容量证据，未下载模型、改远端作业或恢复暂停monitor。先前过期证书阻断及训练进度是保留的历史，不以新SSH连通覆盖它们。训练完成仍须另取得COMPLETED0:0、COMPLETE200000及最终模型身份。

`artifact_status=experimental_unvalidated`，`operator_approved=false`，`formal_gate_status=not_run_not_passed`。连接工程pass不是科学质量pass。
