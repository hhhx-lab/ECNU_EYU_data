# ECNU aTrust 校园 VPN 分流指南：macOS、Windows、Linux

更新：2026-10-18。面向使用本人校园账号、本人电脑和获授权校园资源的用户。

**目标：校园目的走 aTrust，其他流量保留连接前的出口和代理；不破坏其他 VPN、DNS、防火墙或虚拟机网络。**

“通用”指方案覆盖三平台，不表示一个脚本能跨平台运行。本文没有在其他人的电脑、Windows或Linux上实测；实际策略由学校下发，必须在每台电脑独立验收。本文既给普通用户的手动路线，也给维护者编写本机脚本的结构、边界和示例；示例不是厂商正式CLI。

推荐阅读：先看第2节判断能否分流，再按第3/4/5节完成自己平台的只读诊断和最小配置，最后看第8/9节了解脚本如何组织。只需要登录服务器的人不必先实现自动浏览器认证。

## 1. 先看适用范围

| 平台 | 可用方法 | 验证范围 |
|---|---|---|
| macOS | 单文件 `ecnu-vpn-split.sh` 内嵌Python，手动认证后分流或可选自动SSO | aTrust 2.5.16.31的原实现单机通过；合并后另做离线/终端输入验证，不冒充其他机器实测 |
| Windows | 官方客户端登录，管理员PowerShell核查及经允许的精确路由配置 | 本文提供操作流程，不提供已实测的一键脚本 |
| Linux | 官方支持的客户端登录，`iproute2`核查及经允许的精确路由配置 | 本文提供操作流程，不提供已实测的一键脚本 |

WSL中的Linux命令只管理WSL自己的网络，不能替代Windows宿主机VPN分流。Windows不要在Git Bash运行这份macOS脚本；Linux不能直接运行它。

### 单文件与实际依赖

当前入口已合成**一份sh，不需要旁边另放py**：

| 文件 | 职责 | 是否必需 |
|---|---|---|
| `ecnu-vpn-split.sh` | 加载env并运行内嵌Python，含状态/认证/路由/回退/SSH逻辑 | 当前macOS入口必需 |
| `.env` | 每个人自己的配置；不要分享 | 使用包装脚本时必需 |
| [`.env.example`](.env.example) | 空配置模板与变量说明 | 配置参考，不放真实秘密 |

单文件并不等于零依赖：仍需aTrust、现有Conda Python、系统网络工具；自动SSO才额外需要兼容的Playwright MCP和Chrome。Python内嵌在sh的引用heredoc中，经独立文件描述符读取，stdin保留给sudo/交互SSH，不写临时py文件。外观是一个sh，不代表改成了纯Shell实现。

本机启动目录只留脚本、本人env和短README；本详细文档和空模板独立保存，不是运行依赖。分享时只分享文档、审阅过的不含秘密的代码与空模板，不打包整个人目录。

## 2. 所有平台的共同准备

1. 从学校官方渠道取得适合系统/CPU架构的aTrust，按官方说明安装并授予所需网络权限。不要用来源不明的安装包或卸载其他人的安全组件。
2. **先不连接VPN**，按对应平台章节保存连接前快照；记录物理/代理出口、路由、DNS、IPv6和其他VPN接口。若已连接，先按客户端正常注销，确认原网络恢复，再记录基线。不要把别人电脑上的utun7、接口编号、网关照搬。
3. 从超算管理员取得自己的服务器账号、端口和获授权地址。校园账号不等于超算账号。
4. 启动aTrust，选择浏览器登录。入口 `https://vpn.ecnu.edu.cn`，学校统一认证为 `https://sso.ecnu.edu.cn`；核对域名和证书，再输入自己的学号/密码。
5. 完成验证码/MFA/客户端回跳，确认学校资源已登录且隧道可用，再保存连接后快照；仅看到客户端进程不算成功。

### 先确认学校允许分流

优先使用学校下发的按资源分流模式，或请管理员调整资源策略。这是最稳的跨平台方案。

若学校规定必须全隧道，或客户端的WFP/PF/TUN/策略路由强制公网经过VPN，**停止自行改路由，向校方申请分流**。不关闭客户端保护、杀网络服务、清空防火墙或绕过权限。客户端可能重装路由；添加校园 `/32` 不会自动撤销已有 `/1`、`/2` 或其他宽路由。

本指南的本地路由步骤仅适用于：学校允许分流、能唯一确认VPN接口及其路由来源、普通公网出口可以正常恢复的情况。否则只做只读诊断。

### 默认最小校园范围

```text
59.78.189.132/32  超算login1
59.78.189.133/32  超算login2
59.78.189.195/32  历史超算手册资源
```

这些是目前脚本的历史地址，不保证所有用户有权限或学校永远不改地址，使用前核实。访问其他资源优先由校方给出精确IP/CIDR，不把所有私网、所有校园域名或默认路由直接塞入VPN。SSH端口目前配置2323，仍应以管理员说明为准。下文新增/删除路由都属于管理员网络操作，只能对本人电脑经允许的校园会话执行，不能把模板当无条件自动修复脚本。

## 3. macOS：复用现有脚本

### 3.1 放置文件与配置

建议目录 `~/ecnu-vpn/`。取得**仅含代码**的单文件sh，不分发作者的env、登录回执、浏览器日志、known_hosts或历史私有文件。Windows/Linux需要各自平台适配器，不能靠改文件扩展名复用macOS实现。

在自己的目录根据同目录 `.env.example` 新建 `.env`，手工填写自己的配置。里面至少设置自己的 `ECNU_VPN_PYTHON`、三个服务器地址和超算账号；自动SSO才需要校园密码，脚本自动SSH才需要超算密码。可只手动登录VPN，然后使用系统SSH，不在文件中保存密码。

当前sh默认先读脚本同目录 `服务器登陆的env`，不存在时读同目录 `.env`，Python默认当前用户 `$HOME/miniforge3/bin/python`。**推荐每次仍显式指定自己的env，在该env里设置自己Conda环境Python绝对路径**，不依赖安装路径猜测。超算用户也必须明确填写自己的 `HPC_SSH_USER`，不能沿用代码的历史默认账号。

```bash
cd "$HOME/ecnu-vpn"
chmod 700 ecnu-vpn-split.sh
chmod 600 .env
export ECNU_SERVER_ENV_FILE="$PWD/.env"
./ecnu-vpn-split.sh doctor
```

env会被shell `source`，必须是自己编写的可信文件，不运行别人发来的带命令替换/脚本语句的env。涉及密码的值不要贴进终端命令、聊天、截图或共享文档。简单的单引号值适合大多数密码；包含单引号等特殊字符时需正确shell转义，同时自动SSO的MCP解析也必须兼容，不能假设所有dotenv和shell语法完全一致。

### 3.2 依赖与版本检查

- 使用已经安装、自己管理的Conda环境，建议先按原验证环境Python 3.13复核；Python实现本身只使用标准库。不改系统PATH，不使用sudo pip。
- 系统工具为macOS的route/netstat/ifconfig/scutil/sudo/ssh/expect。若缺失，停止并由电脑所有者决定如何准备环境，不自动下载。
- 手动完成学校登录后执行start，不依赖Playwright。自动SSO额外需要已安装Google Chrome及兼容的 `playwright-mcp` 命令。
- 原验证环境Playwright MCP版本为0.0.82，且必须支持 `--secrets`、`--isolated`、`--snapshot-mode none`、`browser_run_code_unsafe`、secret alias和selector target。这不是任意版本都具备的接口，不能把普通Python Playwright包当作该命令。先检查现有命令与接口，缺失则用手动登录路线；本文不要求临时安装一堆依赖。
- 客户端须有安装包CA `/Library/sangfor/SDP/aTrust.app/Contents/Resources/bin/SangforTechnologiesInc.cer` 和端口文件 `/Library/Application Support/aTrust/var/run/httpserver`。路径/协议/证书变化需重新适配，不能下载替代CA或关闭证书验证。

```bash
command -v conda
command -v playwright-mcp
/usr/bin/expect -v
```

自动SSO默认隔离Chrome，不保存storage-state。Bridge可选，但原测试有授权超时，不视为已稳定的一键路线。

### 3.3 先手动登录，再试分流

首次登录**之前**，在普通终端查询并把输出留在自己的私有记录中；不要发到公共群。登录后、分流后、注销后各查询一次对比：

```bash
/sbin/ifconfig
/usr/sbin/netstat -rn -f inet
/usr/sbin/netstat -rn -f inet6
/usr/sbin/scutil --dns
/usr/sbin/scutil --proxy
/sbin/route -n get default
/sbin/route -n get 1.1.1.1
/sbin/route -n get 8.8.8.8
/sbin/route -n get 9.9.9.9
```

然后按第2节在aTrust完成学校登录，再在同一终端执行：

```bash
cd "$HOME/ecnu-vpn"
export ECNU_SERVER_ENV_FILE="$PWD/.env"
./ecnu-vpn-split.sh start
./ecnu-vpn-split.sh check
```

sudo只授权路由操作，不要 `sudo ./ecnu-vpn-split.sh`，不要把整套浏览器/Python作为root运行。首次终端运行会按需提示本机管理员密码；推荐不在env保存sudo密码。

默认hpc会识别Sangfor专属IPv6标识和IPv4 peer，确认唯一隧道后收窄其IPv4路由，不硬编码utun编号。无唯一隧道、IPv6全局接管、基线漂移或路由错误时停止，不flush整张路由表。首次采用已连接客户端时只能重建部分公网基线，仍需人工对照连接前快照确认其他出口/DNS。

只有check通过后才能使用脚本SSH：

```bash
./ecnu-vpn-split.sh ssh-test
./ecnu-vpn-split.sh ssh-hpc
```

SSH使用严格known_hosts，不自动信任首次主机。先从管理员独立核对指纹，再用系统SSH完成首次登记：

```bash
ssh -p 2323 YOUR_HPC_USER@59.78.189.132
```

不要把上述占位账号原样运行。`ssh-keyscan`取得的指纹本身不是可信身份依据。

### 3.4 可选自动SSO与日常命令

手动路线通过后，有兼容Playwright MCP和自己填写的校园凭据才使用未登录状态的start或connect：

```bash
cd "$HOME/ecnu-vpn"
export ECNU_SERVER_ENV_FILE="$PWD/.env"
./ecnu-vpn-split.sh connect
```

connect = 自动/复用VPN + 分流验收 + 交互SSH。`exit`只离开SSH。

- `status`：当前状态；`check`：只读分流验收。
- `stop`：仅断开脚本创建且身份一致的会话；手动登录会保留。
- `logout`：明确注销当前校园会话，包括手动登录；不会强杀其他VPN。
- `restart`：按归属规则停止后再启动，不强制注销用户会话。
- 休眠/换网/重连后重新start/check。没有后台守护；检查失败不能当成功。

## 4. Windows：官方分流优先，PowerShell精确配置

### 4.1 只读快照

在PowerShell分别于VPN连接前、连接后记录以下输出。若查询cmdlet不存在，停下来确认系统版本/NetTCPIP模块，不换成清空网络的命令。

```powershell
Get-NetAdapter | Format-Table ifIndex, Name, InterfaceDescription, Status
Get-NetIPInterface -AddressFamily IPv4 | Format-Table InterfaceIndex, InterfaceAlias, ConnectionState, InterfaceMetric
Get-NetRoute -AddressFamily IPv4 -PolicyStore ActiveStore | Sort-Object InterfaceIndex, DestinationPrefix
Get-NetRoute -AddressFamily IPv6 -PolicyStore ActiveStore
Find-NetRoute -RemoteIPAddress 1.1.1.1
Find-NetRoute -RemoteIPAddress 8.8.8.8
Find-NetRoute -RemoteIPAddress 9.9.9.9
Find-NetRoute -RemoteIPAddress 59.78.189.132
Get-DnsClientServerAddress
Get-DnsClientNrptPolicy -Effective
Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' | Select-Object ProxyEnable, ProxyServer, AutoConfigURL
netsh winhttp show proxy
```

不要只比较0.0.0.0/0：更具体的VPN宽路由会胜过默认路由；也不要只看route print就忽略WFP、DNS NRPT、PAC、IPv6或其他代理。

从客户端、接口描述、连接前后差异及管理员说明确认校园VPN的**实际接口索引和下一跳**。aTrust也可能用过滤驱动而不提供可用的普通路由接口；该情况不能套用下面命令，应使用官方资源分流。

### 4.2 管理员PowerShell：仅允许的临时配置

满足共同准备中的授权条件后才操作。先由管理员允许的策略撤销公网接管，并验证三个公网Find-NetRoute恢复连接前出口，再考虑补校园主机路由。

确需本机撤销某一条宽路由时，必须逐条从审计中取得精确前缀、接口、下一跳。以下是**模板，不是可直接粘贴的批量清理脚本**；不删默认/其他VPN/物理网卡路由，不删除整接口所有路由：

```powershell
# 用已经独立核实的值替换占位符。这里处理一条ActiveStore路由。
$vpnIf = [int](Read-Host '已核实的校园VPN接口索引')
$prefix = Read-Host '已核实且获允许移除的一条VPN宽路由CIDR'
$nextHop = Read-Host '该条路由的实际下一跳'
if ($prefix -eq '0.0.0.0/0') { throw '本模板不处理默认路由' }
$old = @(Get-NetRoute -AddressFamily IPv4 -PolicyStore ActiveStore -InterfaceIndex $vpnIf -DestinationPrefix $prefix | Where-Object { $_.NextHop -eq $nextHop })
if ($old.Count -ne 1) { throw '匹配不唯一，不修改' }
# 只适用于经独立确认的简单静态路由；此门不是来源核验的替代。
if ([string]$old[0].Protocol -ne 'NetMgmt') { throw '非简单静态管理路由，不使用此模板' }
$persistent = @(Get-NetRoute -AddressFamily IPv4 -PolicyStore PersistentStore -ErrorAction Stop | Where-Object { $_.InterfaceIndex -eq $vpnIf -and $_.DestinationPrefix -eq $prefix -and $_.NextHop -eq $nextHop })
if ($persistent.Count -gt 0) { throw '有持久路由，不修改' }
$backup = Join-Path $HOME ("ecnu-one-route-before-" + (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '.xml')
$old | Export-Clixml $backup -NoClobber
# 确认来源/授权后才单独执行以下一行；保留系统确认提示。
$old[0] | Remove-NetRoute -Confirm
```

此删/恢复模板只适用于已确认前缀、接口、下一跳、metric足以完整恢复的简单静态管理路由（无限/默认寿命、无特殊来源/其他属性）。NetMgmt本身不是校园归属证明；动态/有限寿命/特殊协议/未知属性的条目不删除，交官方客户端或管理员恢复。先检查同一路由是否在PersistentStore中：若存在或客户端自动重装，停止，向校方处理，不改持久策略。以上备份只保存这一条，不代表完整系统备份；操作另一条前使用不同备份文件。

只有公网已恢复且VPN有普通可用接口时，才在同一管理员会话中补一条校园 `/32`。**下一跳必须取自正常校园路由或官方说明**；仅当接口确为on-link时才用0.0.0.0，不从别人机器猜值。

```powershell
$vpnIf = [int](Read-Host '再次填写已核实的校园VPN接口索引，不能依赖上一段变量')
$campusNextHop = Read-Host '已核实的校园VPN下一跳，on-link接口才填0.0.0.0'
$campusPrefix = '59.78.189.132/32'
$existing = @(Get-NetRoute -AddressFamily IPv4 -PolicyStore ActiveStore -ErrorAction Stop | Where-Object { $_.DestinationPrefix -eq $campusPrefix })
if ($existing.Count -gt 0) { throw '已有此前缀，先复核是否正确；本模板不重复添加' }
New-NetRoute -DestinationPrefix $campusPrefix -InterfaceIndex $vpnIf -NextHop $campusNextHop -RouteMetric 5 -PolicyStore ActiveStore
```

分别核对其余获授权地址后重复一次一条的流程，不自动覆盖既有条目。ActiveStore是运行时配置，但客户端/系统行为仍需在断开和重连后复核，不能视为永远自动清除。

### 4.3 回退与验收

如新增路由要回退，只删除自己登记的**精确前缀+接口+下一跳**匹配项；仍保留确认提示：

```powershell
$added = @(Get-NetRoute -PolicyStore ActiveStore -InterfaceIndex $vpnIf -DestinationPrefix $campusPrefix |
    Where-Object { $_.NextHop -eq $campusNextHop })
if ($added.Count -ne 1) { throw '回退匹配不唯一，不修改' }
$added[0] | Remove-NetRoute -Confirm
```

如恢复被移除的旧路由，先确认还是同一VPN会话/接口、没有已有同条目，然后使用可信本地备份：

```powershell
$saved = Import-Clixml $backup
New-NetRoute -DestinationPrefix $saved.DestinationPrefix -InterfaceIndex $saved.InterfaceIndex -NextHop $saved.NextHop -RouteMetric $saved.RouteMetric -PolicyStore ActiveStore
```

`$backup`是前面生成的唯一文件路径；另开PowerShell时先手工设成自己那次备份的准确路径。不导入别人的备份；断开、接口编号复用或身份变化后不能盲目恢复。出现复装/强制策略时用官方注销再重新登录恢复厂商配置，仍要验收，不重置全部网络。

检查3个公网和3个校园地址的Find-NetRoute，DNS/代理/IPv6与预期一致，再验证实际服务：

```powershell
Test-NetConnection 59.78.189.132 -Port 2323
ssh -p 2323 YOUR_HPC_USER@59.78.189.132
```

超算SSH指纹需管理员独立确认；仅端口通不代表认证成功。不要在PowerShell保存或命令行传入密码。退出SSH不会断开aTrust，最后在客户端注销并确认路由恢复。

## 5. Linux：先核查策略路由，再精确配置

### 5.1 客户端与只读快照

先确认学校提供适配发行版/CPU架构的aTrust Linux客户端和官方浏览器登录流程；没有官方支持版本就向校方询问，不运行来源不明的兼容层。客户端可能依赖图形会话，无头Linux不能直接假设macOS的SSO流程可用。

分别在连接前和连接后记录：

```bash
ip -brief address
ip -4 rule show
ip -4 route show table all
ip -6 rule show
ip -6 route show table all
ip -4 route get 1.1.1.1
ip -4 route get 8.8.8.8
ip -4 route get 9.9.9.9
ip -4 route get 59.78.189.132
```

DNS使用当前系统实际的管理器查询：有systemd-resolved用 `resolvectl status`，否则检查 `/etc/resolv.conf` 及NetworkManager当前连接。桌面代理和应用代理分别核对，不要把含凭据的完整环境变量转存/分享。

如果普通route输出不足以解释出口，可由管理员只读审查当前nftables/iptables规则，确认是否存在mark/TUN重定向。**不执行flush、删除整张表、关闭防火墙或清空ip rule。** 普通 `ip route get` 不能代表全部带mark或不同用户的应用流量。

确认实际校园接口、隧道peer/下一跳、有效route table、命中rule和来源。名称可能是tun、tap或其他，不能假设utun7/tun0；若同时存在多个VPN或来源不明则停止。

### 5.2 经允许的临时主机路由

优先让官方分流策略恢复公网出口。策略路由存在时，往main表添加 `/32` 可能根本不被使用，必须先理解命中顺序，不为了“生效”随意提升rule优先级。

只有校园接口和有效路由表已核验、学校允许本地分流，才能用下面**占位模板**。把尖括号替换为真实值后逐行人工执行；命令未在本文作者的Linux机器实测。

```bash
# 已核实为点对点/on-link隧道时：
sudo ip -4 route add 59.78.189.132/32 dev <校园接口> table <有效路由表>
# 若该接口必须经过网关，改用下面形式，不与上行同时执行：
sudo ip -4 route add 59.78.189.132/32 via <已核实的VPN下一跳> dev <校园接口> table <有效路由表>
```

使用add不使用replace：已有条目报冲突时停下来检查，不覆盖其他程序的路由。其余校园地址一条一条核实。修改后用route get和真实SSH检查；系统代理出口也要测。

如果已审计确认宽路由属于当前校园会话且获允许撤销，可保存该**单条完整路由属性和来源**后精确删除。例如：

```bash
ip -4 route show table <已核实的表> exact <已核实的VPN宽路由CIDR>
# 仅匹配本会话、已允许的一条；下一跳版本需补齐实际via属性。
sudo ip -4 route del <已核实的VPN宽路由CIDR> dev <校园接口> table <已核实的表>
```

不删除0.0.0.0/0、不删物理网卡/其他VPN路由、不循环批量删除。多条相同前缀、不同metric/下一跳/协议时必须进一步指定准确属性，否则不执行。本地删除不是绕过客户端强制策略的方法；复装或公网仍被mark捕获时立即停止。

### 5.3 回退与验收

只移除本次新增的主机路由，沿用新增时的dev、table以及实际via，不清空表：

```bash
# 仅删除本次登记的唯一on-link主机路由：
sudo ip -4 route del 59.78.189.132/32 dev <原校园接口> table <原表>
# 若新增时使用via，则只执行以下匹配形式，不与上行同时执行：
sudo ip -4 route del 59.78.189.132/32 via <原VPN下一跳> dev <原校园接口> table <原表>
```

执行前用route show核对唯一候选及本次新增记录；若存在多个候选，必须补齐区分所需的实际metric等属性，否则停止。被删除的原宽路由只能在**同一会话/接口**下用记录的完整属性精确add恢复；未知规则/mark不手工修补，使用官方注销/重连并复核。不要把临时命令写进NetworkManager、netplan、开机脚本或systemd，首次验收成功也不代表适合持久化。

核对3个公网、3个校园地址，DNS/IPv6/代理及实际应用都符合预期，再使用：

```bash
ssh -p 2323 YOUR_HPC_USER@59.78.189.132
```

系统没有OpenSSH时先由电脑所有者批准准备依赖；不为查端口临时安装工具。SSH首次指纹独立核验，密码交互输入，退出后在aTrust官方客户端注销并复核网络恢复。

## 6. 三平台都必须通过的验收

| 检查 | 通过标准 |
|---|---|
| 登录 | 校园账号已认证、资源有授权、隧道可用 |
| 校园流量 | 各获授权校园地址走核实的aTrust路径，实际SSH/资源可用 |
| 公网流量 | 至少3个公网IP查询恢复原出口；实际常用应用也正常 |
| 默认与宽路由 | 默认没改且不存在更具体的公网接管；有客户端强制策略则不能宣称通过 |
| DNS | 系统配置、域名解析可用；配置不变不等于DNS数据包路径不变 |
| IPv6 | 无未经确认的全局VPN接管；没有为了分流关闭系统IPv6 |
| 其他程序 | 原代理、其他VPN、局域网、容器/虚拟机等保持功能 |
| 注销 | 仅注销校园会话，本次路由清理/恢复有证据，普通网络正常 |

HTTPS返回403/登录挑战不一定是网络故障；路由检查通过也不等于全部应用可用。记录错误原文与网络上下文，不把端口通当完整验收。

休眠、换Wi-Fi、代理模式变更、aTrust升级、会话重连后要重做检查。默认不装后台守护，不无限重试登录，不把上一会话的接口编号用于下一会话。

## 7. 分享清单与故障处理

可分享：本指南、经审阅不含秘密的单文件macOS代码、空白 `.env.example`。不可分享：任何真实env、账号密码、sudo密码、token/cookie、个人known_hosts、网络审计/登录日志。

| 现象 | 处理 |
|---|---|
| 已打开aTrust但没有资源 | 完成浏览器认证/MFA与客户端回跳，不以进程存在判成功 |
| 公网查询仍走VPN | 检查宽路由及客户端策略；不能靠增加校园 `/32` 修复 |
| DNS/IPv6/其他VPN异常 | 停止本地改动，按本次记录回退或官方注销，找管理员；不flush |
| 自动SSO报MCP缺失/接口不兼容 | 改用官方浏览器手动登录；不关TLS或反复提交密码 |
| SSH提示未知/改变的指纹 | 独立询问管理员，未确认不接受，不删除known_hosts掩盖告警 |
| 断开重连后配置失效 | 新会话重新识别并验收，不套旧接口/网关/基线 |

这份指南只处理连接与分流，不授权运行远端任务、读取他人数据或突破学校访问策略。Windows/Linux步骤是待本机验证的配置参考，不是跨平台测试通过声明。

## 8. 新VPN怎样实现分流：不是替代客户端，而是约束数据路径

### 8.1 两个互相独立的环节

**认证/建隧道**由官方aTrust负责：浏览器SSO、可能的MFA、客户端回跳、设备/资源策略、加密连接和学校赋权。脚本不重新实现私有VPN协议，也不把OpenConnect地址换个域名当成迁移。

**数据分流**要检查连接后的操作系统状态：校园目的的路由指向该隧道，普通目的仍使用原出口。校方按资源策略能做到时直接使用，无须本地删路由；只在获允许且路由可控的模式下用本地精确调整。

学校资源可能通过合成IP、内置代理、浏览器代理、专属DNS解析或过滤驱动访问，不一定表现成普通IP路由。通过portal能打开页面，不代表这个原始IP可走SSH；IP路由也不能表达“按HTTPS域名分流”。动态域名需校方资源规则或受支持的域名代理策略，不能把一次DNS解析结果永久写成 `/32`。

### 8.2 为什么只保留默认网关还不够

路由通常先比较**最长前缀**，同长度再按平台metric/策略等选路。`0.0.0.0/0` 是最宽默认；`0.0.0.0/1` 与 `128.0.0.0/1` 合起来覆盖整个IPv4，且都优先于默认。VPN也能用一组 `/2`、`/8`、主机路由覆盖公网。

因此，默认网关仍是Wi-Fi，不等于公网不进VPN。新增学校 `/32` 只改变这些目标，不能消除其他宽路由。需要对照前后完整路由，确认宽路由来源；校方允许时撤销本隧道的接管范围，再精确添加获授权校园目的。

原单机观察到217条宽范围路由只是该客户端/账号/时刻现象，不应把“删217条”写进通用脚本。其他人可能无需删任何路由，也可能策略根本不允许本地改。

### 8.3 不能忽略的另外四层

| 层次 | 为什么需要检查 | 脚本边界 |
|---|---|---|
| DNS | 校园域名可能专属解析；代理DNS可能被厂商规则转发 | 不替换全部DNS、不外泄校园解析、不以配置字符串相同判逐包一致 |
| IPv6 | IPv4分流完成，IPv6仍可能走全隧道 | 两族都核查，未适配全局IPv6时停止，不靠关闭IPv6解决 |
| 代理/PAC | 浏览器可能通过原代理服务器发请求，实际连接目的不同 | 保持现有代理，验证真实应用，不擅自修改其他程序配置 |
| 过滤/策略规则 | Windows WFP、macOS PF、Linuxmark/rule会改变普通route查询之外的路径 | 不停服务/清空规则，存在强制策略时请求校方调整 |

### 8.4 如何识别“只属于这次校园VPN”的路由

不能仅看接口名含VPN或“最后一个tun”。结合客户端真实在线状态、校园入口、当前会话身份、接口地址/peer、连接前后新增项、已知厂商标识及管理员说明。至少做到：只有一个匹配隧道；每条待修改路由的接口/下一跳/table可确认；有其他VPN时它们的路由不在计划中。

macOS原实现用厂商IPv6前缀标识与peer联合识别，属于当前版本经验，不是厂商承诺所有系统相同。Windows由接口描述/索引/来源定位，Linux还要绑定table/rule/mark。未知客户端新版本直接失败，不写猜测性回退。

## 9. 脚本大概怎么写：公共状态机，平台适配器

### 9.1 最小可维护结构

建议命令接口统一为 `doctor/status/plan/start/check/stop`，可选 `ssh-test/connect`。本机现有脚本没有plan命令；以下是**新开发脚本的设计建议**，不能把建议当作已有功能。先做手动登录+只读plan，再实现变更，最后才做自动SSO。

```text
配置读取（不输出秘密）
  -> 生命周期锁（同一时间一个变更操作）
  -> 连接前网络快照
  -> 复用已在线会话，或打开官方客户端/浏览器等待登录
  -> 确认真实在线、资源、唯一隧道及会话身份
  -> 读取IPv4/IPv6/策略/DNS/代理的当前状态
  -> 生成dry-run计划（仅获允许的本隧道精确路由）
  -> 校验计划及权限
  -> 记录原条目，执行一次精确修改，每成功一步追加日志
  -> 验收校园、公网、DNS/IPv6/代理/其他程序
  -> pass退出0，或同会话精确回滚并退出非0
```

“打开客户端”“提交表单”“命令返回0”都不是VPN/分流成功。只有认证状态+隧道+网络验收证据齐全才输出成功。

### 9.2 会话和网络合同

日志/状态文件至少包含以下结构；**不包含密码、token、cookie、证书私钥或完整浏览器响应**：

```json
{
  "schema_version": 1,
  "platform": "example",
  "portal": "https://vpn.ecnu.edu.cn",
  "observed_at_utc": "<actual UTC>",
  "session_identity": "<local session identifier, not authentication token>",
  "owned_session": false,
  "interface_identity": {"index_or_name": "<verified>", "peer": "<verified>"},
  "baseline": {"default": {}, "public_paths": [], "dns": {}, "proxy": {}, "ipv6": {}},
  "planned_changes": [],
  "applied_changes": [],
  "rollback_errors": [],
  "verification": "not_run"
}
```

真实原条目需保留前缀、下一跳、接口、table、metric、有效来源/寿命/标志中恢复所需的字段。日志由本人持有，Unix目录700/文件600，Windows使用用户专属ACL。UTC记录来自本机；如果远端返回UTC单独保存，不能混用时钟。

没有连接前基线时只做plan或人工确认，不声称已证明DNS/代理“保持不变”。现有Mac首次采用在线会话的公网重建能力有边界，后续通用实现不能依赖它完全补齐基线。

### 9.3 macOS单文件的安全内嵌方式

不使用 `python - <<EOF` 运行交互SSH：stdin已被源码占用，sudo/交互会受影响。下面是**结构示例**，不是另一份可登录VPN的完整脚本；真正的配置检查和VPN逻辑应放在相应位置：

```bash
#!/usr/bin/env bash
set -euo pipefail
set +x
umask 077
# 先校验本人可信env的路径/所有者/权限，再source并export配置。
# Python路径由env配置，不能把密码放进命令行参数。
: "${ECNU_VPN_PYTHON:?set existing Conda Python path in env}"
exec "$ECNU_VPN_PYTHON" -B -c 'import os
with os.fdopen(3) as f:
    code = f.read()
exec(compile(code, "embedded-vpn", "exec"))' "$@" 3<<'VPN_PY'
import sys
# 在这里定义状态查询、计划、执行、验收、回退和命令分派。
# 这个示例不执行任何路由或认证动作。
print("Skeleton only; implement and test platform adapter before use.")
VPN_PY
```

引用heredoc避免Shell把Python内的 `$`、反引号或秘密替换掉；独立fd读取后关闭，不写临时源码。入口需要传递参数和返回Python退出码，用exec避免多余父进程。实现可依赖现有Python环境，单文件不应趁机安装依赖或把整个客户端/浏览器放在sudo下运行。

macOS路由修改示例只适用于已确认on-link隧道：`sudo route -n add -host <校园IP> -interface <已核验utun>`；网关型路由必须按真实字段构造，不能机械移植Windows/Linux语法。实现中使用参数数组和结构化IP/CIDR解析，不把未经校验的env拼成shell命令。

### 9.4 Windows脚本的组织示例

建议使用 `.ps1` 和系统NetTCPIP对象，不调用本Mac sh。不为运行脚本全局关闭ExecutionPolicy；按单位允许的签名/本次会话策略运行。

以下函数只展示**计划**读取，不删除/添加任何路由。`VerifiedVpnIf`须由前面的客户端/会话身份核验取得，不能由用户随意输入后直接信任。返回的差异也只是候选，要和校园范围、原基线、授权规则再匹配：

```powershell
param([Parameter(Mandatory=$true)][int]$VerifiedVpnIf)
$ErrorActionPreference = 'Stop'
function Get-NetworkSnapshot {
    [pscustomobject]@{
        Utc = (Get-Date).ToUniversalTime().ToString('o')
        Interfaces = @(Get-NetIPInterface)
        Routes4 = @(Get-NetRoute -AddressFamily IPv4 -PolicyStore ActiveStore)
        Routes6 = @(Get-NetRoute -AddressFamily IPv6 -PolicyStore ActiveStore)
        Public = @(Find-NetRoute -RemoteIPAddress '1.1.1.1')
        Dns = @(Get-DnsClientServerAddress)
        Nrpt = @(Get-DnsClientNrptPolicy -Effective)
    }
}
$snapshot = Get-NetworkSnapshot
$candidates = @($snapshot.Routes4 | Where-Object {
    $_.InterfaceIndex -eq $VerifiedVpnIf
})
# 候选不是删除清单；此示例只有只读plan，不保存认证秘密。
$candidates | Select-Object DestinationPrefix, InterfaceIndex, NextHop, RouteMetric, Protocol
```

完整脚本再加入：本人配置校验、进程级Mutex、可信状态存储、会话一致性、New/Remove-NetRoute的 `SupportsShouldProcess`/WhatIf/Confirm、唯一匹配和PersistentStore门、逐步变更日志、错误回滚、实际服务检查。删除时传原匹配对象，恢复时用已记录属性，不能只用前缀删除所有条目。不硬编码接口编号，不把只读plan输出全自动接到Remove-NetRoute。

PowerShell不原生等价支持shell dotenv；若设计JSON/PSD1配置，限制字段类型并拒绝执行任意表达式。敏感值用本人既有凭据存储或安全交互输入；若单位约定env文件则使用严格解析器/ACL，不将文件 `Invoke-Expression`。Windows SSH优先密钥或交互认证，避免模拟键盘把密码送错窗口。

### 9.5 Linux脚本的组织示例

Linux推荐 `bash` 管生命周期锁、现有Conda/uv Python解析 `ip -j` JSON；不要用awk猜复杂route/rule字段。下面仅只读采集，未实现aTrust登录、改路由或日志持久化：

```python
import json
import subprocess

def ip_json(*args):
    result = subprocess.run(["ip", "-j", *args], check=True,
                            capture_output=True, text=True, timeout=10)
    return json.loads(result.stdout)

def snapshot():
    return {
        "links": ip_json("link", "show"),
        "routes4": ip_json("-4", "route", "show", "table", "all"),
        "rules4": ip_json("-4", "rule", "show"),
        "routes6": ip_json("-6", "route", "show", "table", "all"),
        "rules6": ip_json("-6", "rule", "show"),
        "public": ip_json("-4", "route", "get", "1.1.1.1"),
    }

print(json.dumps(snapshot(), indent=2))
```

数据只留本机，不发公共群。旧iproute2不支持JSON时停止并选择兼容适配器，不把失败输出当空路由表。完整实现还需DNS管理器、桌面代理与应用代理、mark/cgroup/用户选择的策略路径，以上快照不是全系统网络证明。

执行器使用数组 `['sudo','-n','ip','-4','route','add', ...]` 和严格IP/table/interface验证；sudo预授权由终端管理，不能把登录密码硬编码。添加使用add而不是replace；每步记录实际结果；未知规则或多路径路由拒绝自动处理。lock可用当前用户私有目录中的flock，仅锁本脚本生命周期，不锁其他程序。

### 9.6 自动浏览器登录是可选适配，不是核心分流依赖

维护者先跑通官方手动登录。自动化时只使用已安装、允许的Chrome/Playwright，访问学校入口并检查SSO origin、service目的和TLS；表单选择器绑定实际版本，缺失/MFA/验证码则停止，最多一次密码提交。禁止关闭证书验证、抓取/重放认证token、保存cookie到分享包。

确认aTrust已完成回跳且隧道可用之后才能关闭仅属于本脚本的临时登录浏览器。不关日常Chrome全部窗口；未取得可靠会话身份的中断不自动注销并发手动会话。macOS的已观察loopback status/logout协议、CA路径和端口文件不能假定Windows/Linux相同；没有公开受支持CLI时使用官方客户端状态和用户确认，不猜接口。

### 9.7 错误与验收测试怎么设计

| 用例 | 必须行为 |
|---|---|
| env/依赖缺失 | 明确失败，不安装、不泄露值 |
| 两个疑似校园隧道 | 不修改，要求人工核对 |
| 客户端运行但未登录 | 不修改路由、不称成功 |
| 默认没变但公网被宽路由接管 | check失败，不漏检 |
| 其他VPN路由/原主机路由 | 不在变更集，不覆盖 |
| 源码执行、参数、stdin | 单文件内嵌不损坏参数/终端输入/退出码 |
| 修改一半失败 | 同会话只撤销已成功步骤并准确记录回滚错误 |
| 会话退出/接口被复用 | 停止，不向新会话恢复旧路由 |
| 回滚删除失败 | 记恢复不完整，不输出恢复成功 |
| TLS/MFA/指纹变化 | 停下核实，不降低校验 |
| 超时/重连/睡眠 | 有界失败，新会话重验，不持续重试 |

先用合成路由/模拟平台命令做离线测试，再由电脑所有者在受控会话实机验收。原单机macOS通过不替代Windows/Linux测试；命令模板“读起来正确”也不等于已在目标系统验证。
