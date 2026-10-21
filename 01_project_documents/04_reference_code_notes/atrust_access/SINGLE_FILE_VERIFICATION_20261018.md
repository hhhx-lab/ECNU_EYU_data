# 2026-10-18 单文件入口与文档整理验证

用户要求：合成一份sh，三平台指南详述新VPN的分流方法与脚本写法；本机只留短README供Agent快速启动，不留长指南。

## 实际改动

`/Users/hwaigc/比赛+课题/服务器登陆/ecnu-vpn-split.sh` 内嵌此前Python实现，经引用heredoc的fd3读取/关闭源码，使用Python `-c`，保留stdin供sudo/SSH。未写临时py，不需要相邻py；仍使用现有Conda环境和官方aTrust/Chrome/MCP。

启动器增加macOS平台门；env默认脚本同目录原文件，缺失时可用同目录.env，始终可显式 `ECNU_SERVER_ENV_FILE`。Python默认为当前用户 `$HOME/miniforge3/bin/python` 或env配置。Python核心代码与移动前源码逐字一致，仅外层包装改变；本机env原字节不改、不复制。

本机README列status/start/check/ssh-test/connect/stop、凭据精确位置、归属规则、失败停止与边界。原独立py、长指南、工程验收及超算说明移入 `history/20261018_two_file/`，原包装器复制保留；原私有网络审计不动。当前三平台指南/空模板在本目录，不再摆在本机启动目录。

三平台指南补认证与数据路径分离、最长前缀/宽路由/DNS/IPv6/代理/过滤驱动、唯一隧道来源、脚本状态机/合同/权限/逐步日志/同会话回滚、Mac fd3内嵌结构、Windows只读PS1骨架、Linux结构化ip JSON骨架。所有骨架明确非可直接登录的完整脚本，不自动安装、关闭保护或绕过校方策略。

## 验证与SHA

命令：`conda run --no-capture-output -n base python -B -m unittest -v tools/test_ecnu_vpn_single_file.py`，20项pass。

- 内嵌Python与归档原文件字节内容一致且AST通过。
- Bash语法通过，合成env下help/未知命令分别exit0/1。
- 模拟解释器确认fd3可读完整源码入口，参数保留、stdin仍收到调用者输入；测试不读真实env、不sudo/SSH/登录。
- 16项路由/状态/归属/回滚/IPv6既有安全场景覆盖；本轮套件总20项，不声称20项实机测试。
- 本地UTC `2026-10-18T06:49:02Z` 附近在PTY下只读status exit0，logout/tunnelUnavailable，managed state=logged_out；未新登录/路由变更、未SSH/查询训练。

当前sh SHA256：`c6384030aa1964cd5df9745da41ba1b6a6d79ed03cf0bebe7a20e0a3ae0944c9`。

归档原Python SHA256：`b0f176ca14779e54bb265a4efbec7854c7d9f2a6de7b129ab762e3d678947f38`。旧工程验收中成功实机登录对应此前双文件实现，不改写旧SHA，也不宣称合并后的单文件重新做了自动登录/交互SSH实测。

Windows/Linux及其他机器未实测；本机无PowerShell，本轮不安装依赖。连接工程不等于训练/质量验收，`experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`保持。

独立读者审查后的修订：本机README用&&落实失败门控并补校方允许分流前提；Windows删/恢复模板限定已核实可完整恢复的简单静态管理路由，动态/特殊属性不处理。文档围栏/链接、Python及JSON骨架静态解析、空模板bash语法通过；13个当前/归档文件凭据字面量检查无命中。总纲首次登记含VPN文件名的证据被既有隐私规则拒绝，规则未降低，改用本非敏感验证摘要登记，脚本/空env只作限制提取的路径收录。
