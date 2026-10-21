# aTrust 分流文档与历史证据

当前本机入口在外部目录 `服务器登陆/`，那里只需 `ecnu-vpn-split.sh`、本人原env和短README；不是从本历史区启动VPN。

- [三平台详细指南](ECNU校园VPN分流连接指南_通用版.md)：macOS/Windows/Linux的认证、分流、精确路由、验收、回退和脚本架构；Windows/Linux未实测。
- [空配置模板](.env.example)：仅供其他Mac准备本人的配置，没有真实凭据。
- [单文件合并验证](SINGLE_FILE_VERIFICATION_20261018.md)：源码一致性、stdin/参数/退出码测试与当前路径。
- [原双文件Python](history/20261018_two_file/ecnu_atrust.py)、[原包装脚本](history/20261018_two_file/ecnu-vpn-split.sh)：原字节保留，非当前启动入口。
- [原工程验收](history/20261018_two_file/ATRUST_SPLIT_ACCEPTANCE_20261018.md)：旧SHA/实机登录记录保留，不改写为新单文件实测。
- [历史指南](history/20261018_two_file/ECNU校园VPN分流连接指南.md)、[历史通用指南](history/20261018_two_file/ECNU校园VPN分流连接指南_通用版.md)、[历史超算说明](history/20261018_two_file/华东师范大学超算八期集群使用指南.md)：保留来源，本机操作以新的短README与真实状态为准。

本目录未复制真实env、网络回执或浏览器状态；原回执仍在本人 `~/.ecnu-vpn/atrust/`。旧文档中的外部路径是历史位置，本目录移动不改变原审计/SHA。不要打包整个项目给他人，分享只选通用指南、空模板及审阅过的无秘密代码。

本次连接整理不改变G1训练状态、作业/下载授权或研究结论。
