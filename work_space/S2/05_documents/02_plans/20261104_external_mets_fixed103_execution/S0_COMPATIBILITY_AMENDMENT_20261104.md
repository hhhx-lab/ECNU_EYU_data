# S0 兼容准备补充方案（待确认）

本文件是2026-11-04重新执行请求后的提案，不是安装、重试或作业授权；原PLAN、已验收模型及失败记录不改。当前未新增SSH、依赖下载/安装、环境、转换或Slurm作业。

## 事实与阻断

- 修正后的官方Conda-forge预检已实际结束，缺锁定connected-components-3d4.1.1；R4未创建。真实服务器glibc2.17，原R1-R3及失败保留。
- 一次本机只读PyPI元数据检查在UTC11:43:26至11:43:33确认三个原版本HTTP200。所列CPython3.10 Linux wheel中，numexpr2.14.1要求manylinux2.27/2.28，cc3d4.1.1和ndindex1.10.1要求2.24/2.28；musllinux/macOS不是替代。没有下载或安装依赖，不把版本存在当服务器兼容通过。
- 2025固定镜像资产/11层diff_id及标准archive已验收，但旧native sandbox转换超1800秒，无可验收SIF。原archive、sandbox和失败永久保留。

## 两项独立确认

1. **同版源码构建方法。**拟仅本批新增Conda管理的用户态构建环境，候选gcc_linux-64/gxx_linux-64/binutils_linux-64及sysroot_linux-64=2.17，Python构建依赖setuptools/wheel/Cython和原sdist实际要求的必要依赖。先核真实glibc与精确求解计划、源包SHA/构建要求、容量和版本，再冻结清单；构建三个原版本，不降低直接版本、替换Torch/cu124或修改模型。最终推理环境与构建环境分离，不升级系统glibc、不装系统编译器、不改全局Conda/PATH。构建兼容性尚未实测，失败保留、不自动换方法重试。
2. **一次独立镜像转换与CPU准备作业。**拟新增仅CPU的S0准备作业（资源需鲜活分区/容量核验），在新attempt复用同一已验收镜像archive，单次转换上限7200秒；隔离本次临时路径，不复用失败rootfs，不重下载、不删旧失败/资产、不改变namespace或管理员保护。保持原镜像及模型算法不变；是否能在时限内完成尚未验证。

两项均已询问，尚未获得新答复。未确认前不得安装工具、创建该准备作业或重试转换。即使批准，也须先核新增依赖的实际清单/平台/来源、静态测试、实时容量和Slurm合同，不能用候选名称当已锁定清单。

## 后续提交边界

若批准新增S0 CPU准备作业，总数从原拟7作业变为8作业，其中仍仅4个单GPU作业。新增作业只处理准备，不做推理、模型加载或读GT；S0通过后才能冻结并创建原J01-J04，held资源/脚本/SHA/依赖核验后释放；S1两smoke及联合容量/估时门通过后才创建J05-J07。

同103真实四模态、完整七fold/BrainIAC、2025固定digest、K1 selected175、原预处理/后处理、统一MET和患者组paired合同不变。不读test，不恢复监控；公开训练暴露继续披露，不称独立泛化或科学pass。

证据：执行批次的EXECUTION_CONTINUATION_REVIEW_20261104.json、ENVIRONMENT_R4_OFFICIAL_R2_FAILURE_ANALYSIS.json及operator/pypi_binary_options_20261104T114326Z/RESULT.json。

`experimental_unvalidated` / `operator_approved=false` / `formal_gate_status=not_run_not_passed`。
