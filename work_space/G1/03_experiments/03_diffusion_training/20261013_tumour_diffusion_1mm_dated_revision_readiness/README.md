# TumourDiffusion 1mm 20261013改进包复核与替换

复核日期：2026-10-13，北京时间；主要CPU证据UTC为12:31:44，补充合同证据UTC为12:34:16。
状态：`experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。

## 结论

**主要旧问题确实改善，新包已原样替换活动目录；但仍有可信评价、数据身份和长任务恢复缺口，不能直接宣布全量训练就绪。** 不修改用户源码、不安装依赖、不连接服务器、不提交训练、不拉回epoch4影像或上传Drive。检查仅使用既有Conda环境和隔离的合成数据。

| 内容 | 路径 |
|---|---|
| 当前活动源码，37文件 | [tumour_diffusion_1mm](../../../01_code/tumour_diffusion_1mm) |
| 原始改进ZIP | [TumourDiffusion_1mm_20261013.zip](TumourDiffusion_1mm_20261013.zip) |
| ZIP安全导入、两版逐文件SHA及差异 | [STAGING_RECEIPT.json](STAGING_RECEIPT.json) |
| 活动发布与旧版保存验收 | [REPLACEMENT_RECEIPT.json](REPLACEMENT_RECEIPT.json) |
| 本机回归与独立探针 | [DATED_REVISION_READINESS_EVIDENCE.json](DATED_REVISION_READINESS_EVIDENCE.json) |
| 上游逐项回归日志 | [UPSTREAM_REGRESSION_TEST_LOG.txt](UPSTREAM_REGRESSION_TEST_LOG.txt) |
| 评价身份和AdamW补充探针 | [ADDITIONAL_CONTRACT_PROBES.json](ADDITIONAL_CONTRACT_PROBES.json) |
| 被替换的修订3历史源码 | [previous_source_revision3](previous_source_revision3) |
| 修订3历史报告 | [revision3审计](../20261013_tumour_diffusion_1mm_revision3_readiness/README.md) |

来源：`/Users/hwaigc/Downloads/TumourDiffusion_1mm_20261013.zip`，SHA256=`742a4b6a31d7d775d12e6deb0ba0cef47aed7b8e1e149ecccd58a6bf699f4b2b`。39个条目、613780解压bytes；对比修订3为14文件变化、6新增、17不变、0删除。校验路径越界、重复项、symlink、加密和体积，活动源码与ZIP逐文件SHA一致。旧修订3的31源码移入本批次历史目录并核对SHA，不覆盖旧审计。此前用户要求删除的20261009原30源码不恢复，旧150k权重、S2及其他路线不触碰。

新增文件为修复说明、`checkpoints.py`、`splits.py`、`volume_validation.py`和两份回归测试。`model.py`、网络和ROI变换未变；仍是四个独立单模态标签条件EDM，不是联合四模态新架构。

## 实测修好的问题

| 上次问题 | 此次实际检查 | 结论 |
|---|---|---|
| 缺失/全拒绝病例仍exit0 | 真实CLI针对missing、invalid、empty输入非0退出 | 修好 |
| 坏已有输出只看存在就skip | 完成marker、源/输出SHA、配置与NIfTI合法性验证；破损或来源变化拒绝复用 | 修好预处理复用路径 |
| CSV失败覆盖旧文件 | 不完整输入报错，旧CSV内容不变，成功后原子发布 | 修好 |
| 缺split列放行/跨访视泄漏 | loader/eval/list共用fail-closed选择；00001-000与00001-001同组 | 修好算法，真实权威名单仍待绑定 |
| RC随机crop丢失 | 真NIfTI -> CSV -> loader；24随机次均保留216 RC体素 | 通过回归 |
| 目标2步执行3步 | EDM/lognsr/cosine精确计数；独立真实checkpoint恢复2->3仅新增1更新，优化器step=3 | 修好 |
| compile/eager键不兼容 | 实际OptimizedModule包装与普通模型互载 | 修好包装兼容，不等于Inductor/Triton验收 |
| 原子保存/AMP scaler | 中断保存保留旧checkpoint；跳过AMP尝试不计步 | CPU回归通过，CUDA恢复项未跑 |
| 1mm只改header | 采样与affine共用target/source物理映射，非整数shape比例斜置/反向affine坐标图 | 修好核心映射 |
| smoke递归删除既有证据 | 已移除rm -rf，WORK存在即拒绝；哨兵文件保留 | 修好该删除风险 |
| lognsr拼写 | 真训练循环三种schedule均通过 | 修好 |
| 评价误用默认sigma_data | 从每模态checkpoint读取schedule，异路schedule/不匹配sampler拒绝 | CPU单元验证通过 |
| 四模态启动忽略子任务失败 | 1–4槽串行/并发、失败传播、全部等待 | 大部分模拟Bash回归通过 |

独立坐标图内部最大误差`0.002731895 mm`（容差0.003，含三阶插值/float32误差），输出spacing为1×1×1；不是声称边缘插值零误差或所有真实影像已验收。新实现明确不与S2 shape-based resize逐位一致。

本机Conda `g1_t2w_bbdm`，Python3.10.20、torch2.7.1、MONAI1.5.2、NiBabel5.4.2、nnunetv2 2.8.0，无CUDA。28份Python AST、2份shell语法与10模块真实import通过。参考网络4,475,999参数，64³前向、16³反向finite；不是64³/batch16完整GPU反向验收。

**上游测试42项：37通过、4因无CUDA跳过、1失败。** 四个GPU项分别是完整CLI训练恢复、AdamW beta1端到端、真实采样schedule和CUDA scaler恢复。包内其他机器“GPU已通过”仅为作者说明，不冒充本机或ECNU独立验收。

## 仍需处理

### 高优先级：评价身份不匹配仍只警告

[evaluate_generation.py:479](../../../01_code/tumour_diffusion_1mm/src/infer/evaluate_generation.py:479)在首个模态之后跳过数据身份核对；首模态CSV不符也只WARNING继续。补充探针对两个都不符的checkpoint执行实际AST元数据分支：读取2模型，只警告1次，无拒绝。患者group检测只约束**当前CSV内部**，不能证明当前val从未用于旧checkpoint训练。可信选点/报告必须逐模态锁定训练成员、CSV、数据和空间合同；声明的探索性重评也应另列，不能默认为独立验证。

### 高优先级：训练/评价缓存身份仍未绑定当前影像内容

[csv_identity:43](../../../01_code/tumour_diffusion_1mm/src/train/tumour_main_diffusion.py:43)新增`preprocessing_sha256`，能检出完成marker改动。但保持CSV和marker不变、直接改影像bytes，identity完全不变；marker缺失时仍返回None并可继续。loader不重算预处理marker中的输出SHA，[eval_checkpoints.py:104](../../../01_code/tumour_diffusion_1mm/scripts/eval_checkpoints.py:104)缓存request也只有CSV/checkpoint SHA，影像变化仍可能复用旧指标。需要训练/恢复/评价前独立逐文件manifest复核或共享严格验证入口，不把“marker自带SHA”误说成“消费端已核对当前影像”。

### 长任务可靠性：仍非精确可复现续训

checkpoint有网络/optimizer/scaler/step语义，但无Python/NumPy/torch/CUDA/MONAI RNG和sampler/batch位置状态，无SIGTERM安全保存；仍每10次loader遍历加最终保存。恢复保持若干架构/schedule参数，但small-lesion weight/threshold/clamp、patient_balance等增强/损失参数不完整锁定。精确**更新数量**恢复已通过，不等于同随机路径恢复。ECNU长任务需独立Slurm、排他审计、冻结全数据与增强合同、定步保存及超时恢复验收。

### 条件性启动器问题

[train_4gpu.sh:59](../../../01_code/tumour_diffusion_1mm/scripts/train_4gpu.sh:59)在本机GNU Bash3.2.57关闭输出管道时，状态文件混入进度文本，`test_serial_queue_survives_closed_output`连续两次失败；其余12个launcher测试通过。未验证Linux Bash5是否复现，不能把此故障泛化为所有Linux训练失败。启动器也覆盖固定`logs/train_<mod>.log`并删除旧status，不能直接用于项目历史证据保留；应在新独立attempt中运行并绑定受分配的GPU，不复用旧logdir。

[smoke_test.sh:63](../../../01_code/tumour_diffusion_1mm/scripts/smoke_test.sh:63)仍只训练t1c/t2w，而非四模态完整验收；默认空COMPILE_ARGS在Bash3.2 `set -u`下亦需目标shell验证。此轮只执行隔离的哨兵拒绝测试，没有启动其真实数据/GPU流程。

### 数据和空间门仍待项目实证

- 患者分组代码并未加载项目权威823/103/104成员合同；CSV只提供val指定，其他病例默认train，test104必须通过输入名单严格隔离，不能用整目录自动入train。跨访视分组可能改变历史分母，须核对并留记录，不能静默修改split。
- T2W排除名单仍266条，未与项目权威265逐病例对齐；四模态完整输入检查也不等于T2W信息有效性QC。
- 严格physical1mm算法已修好，但逐例原affine、新affine、bbox完整逆映射还未持久化到完整空间合同；S2原E spacing `[1,0.8593999743,0.8593999743]`接入/回写需单独验收。
- A100完整64³/目标batch/AMP/compile/内存与吞吐、四模态生成QC和原空间roundtrip未跑。禁止以本机CPU成功代替这些门。

## 新增研究因素

AdamW beta1由0.5变为0.9，默认不再严格复现旧优化器。补充CPU探针验证新默认0.9，旧0.5 checkpoint不显式指定0.5会被拒绝，显式0.5可恢复。它是新增训练因素，未证明质量更优；与1mm、整例归一化或VAE epoch4重建影像同时改变时不能归因单机制。预算仍是Diffusion更新step，不套用S2的175 epochs。VAE重建图与真实图、G1图像质量与S2分割增益继续分开。

## 证据复现与历史说明

`import_dated_revision.py stage/publish`为一次性流程，已完成后拒绝覆盖；源码仍与原ZIP逐SHA一致。`audit_dated_revision.py`和`additional_contract_probes.py`可阅读复现实验设计，但同名证据已存在即拒绝写入；新复核应新建取证attempt，不覆盖本次。

初次独立探针误用新版`resample_physical`签名，同时测试子进程生成11份pyc；原37源文件没有修改。初次JSON/日志保留为`INITIAL_HARNESS_*`，生成bytecode移到`initial_probe_generated_bytecode`，不混入活动源码；修正检查器签名并设置子进程禁写bytecode后再次完整复核，无probe_error，source_unchanged=true。真实launcher失败在两轮一致，不归咎于检查器签名问题。

下一步：先补消费端数据/评价身份和长任务合同、冻结真实成员及1mm数据，再做单A100独立四模态smoke与恢复验证；当前状态仍`not_ready_for_full_training`，未授权或提交训练。
