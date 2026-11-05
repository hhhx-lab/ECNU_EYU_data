# 两路外部模型推理接续

## 最新回执

2025全量4310575已核验/同ID首次释放0，最新PENDING/Priority；R5环境COMPLETE及独立实际容器CPU依赖/trainer验收通过，但Bio全量R1首次sbatch返回非零，无JobID。只读queue/accounting确认无Bio作业，原失败和冻结包保留；操作器遗漏stderr已仅本机修正/回归，独立提交R2已请求确认尚未执行。详见[STATUS_20261106.md](STATUS_20261106.md)。下面smoke-first及R4/R5进行中为历史，不覆盖本段。

R5独立验收入口为`operator/accept_runtime_r5.py`：只消费终态COMPLETE，核对远端与本机回执SHA、原精确依赖计划、freeze/实际安装报告、39基础包版本/build/SHA、旧失败/环境和作者身份，在同2025 SIF内只读绑定R5原绝对prefix与作者源码完成CPU依赖/trainer导入。无MRI/GT/模型权重挂载、无`--nv`、无额外GPU作业。验收结果绑定到Bio全量合同及环境身份文件，入口/释放再核SHA，不凭目录存在称环境就绪。最新[本机回归](operator/verify_transport_20261105T172052Z/VERIFICATION.json)为279发现/277pass/2skip、99生产3.10 AST。原安装SSH停滞在完成后仅关本次本地连接，255传输失败与用户VPN保留回执完整，不重装。

## 本次范围

用户最新要求“开始提交推理作业”。接续CPU4309579成功完成及2025 SIF/三个复用wheel身份验收，不重提CPU、不重编、不重下载或重新转换。保持原作者revision、完整七分支集成、fixed103真实四模态及原J01-J07门控：完整环境/静态预检通过后才提交J01-J04；两真实GPU smoke及联合容量/耗时门通过后才提交J05-J07，不提前称全量运行。

## 准备检查

新增只读入口`operator/inspect_inference_readiness.py`。实际执行一次，审计`operator/inspect_inference_readiness_20261105T160852Z`；成功CPU/SIF身份再次通过，author revision为`83c73e0cac0b5a3c77a298547040ea3ea7f99854`，源码逐SHA一致。R3只有Torch2.6.0+cu124、torchvision0.21.0+cu124、pandas2.3.3、NumPy2.2.6和Pillow10.4.0就位，作者其他依赖及三个编译wheel尚未消费；R4环境和审计均未存在。作者接口小源码只在工具输出读取，不作为公开Git正文副本保存；来源身份保留在RESULT。

新R4操作器只做已批准依赖的独立消费：先验证CPU COMPLETE/COMPLETED0:0及实际SIF/wheel SHA、原作者源码和旧失败，再在R3解释器进行pip dry-run并验收官方来源/版本/实际三个wheel路径SHA；通过才Conda离线copy-clone，按解析后的精确版本约束安装，注册作者trainer、逐import和pip check。无系统编译器或solver重跑，旧环境/失败保持。计划与实际选择/下载身份必须一致，否则FAILED并停止，无自动retry或新GPU作业。

本机检查见[R4准备记录](RUNTIME_R4_LOCAL_CHECKS_20261106.json)：base初次全量测试因缺nibabel发生15个既有影像测试错误，未安装或提交；换用已有项目`brats2023_seg`后260发现、258通过、2原有版本限定skip。新增readiness5项/runtime计划9项均通过，远端Python3.10语法检查通过；操作器和catalog仍用Conda base，不混装依赖或PATH。这些是本机准备检查，不是完整环境或GPU模型pass。

## 当前执行

已准备独立`operator/install_runtime_r4.py`首次执行。尚未取得R4远端完成结果，也未创建J01-J07；本节后续按真实回执更新。两端UTC分别记录，日期标题使用客户端2026-11-06；源码/模型/数据及历史失败不覆盖。

## 用户最新变更：直接两路全量

用户随后明确要求“不用做这些了，直接提交！！！把脚本写对环境准备好！”。以[独立授权记录](DIRECT_FULL103_AUTHORIZATION_20261106.json)覆盖原先需等待J01-J04的执行顺序，原PLAN及上述前一阶段历史保留不改写。取消额外GPU smoke和联合smoke作业，完整环境准备和本地脚本检查后直接创建两路各103例的单A100/8CPU/64GiB/48h全量作业，held后独立核验并释放原ID。用户豁免的smoke、实测耗时/峰值及独立J01-J04记为`waived_by_user_not_pass`，不冒称已实测通过或生成质量通过。

必要的固定数据/原作者全部权重SHA、禁止缺失模态fallback、native几何/finite/标签及GPU映射/网络/GT/凭据隔离检查保留在两路全量入口。任何真实错误仍FAILED并保留证据，不降级、不换模型/fold/病例、不自动retry或重新提交。完整推理环境仍需准备好，当前R4首次执行正在进行；GPU作业尚未创建。
