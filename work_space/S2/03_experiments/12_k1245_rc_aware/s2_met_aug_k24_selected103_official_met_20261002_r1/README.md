# K2/K4 selected-checkpoint fixed103 官方MET

本批从两路175轮训练各自25/50/75/100/125/150/175候选中选EMA最高点，并列取更早epoch。核验后两路均选epoch175，SHA与选点证据绑定。

`EXECUTION_PLAN.json`为冻结执行合同，`STATIC_SHA256.json`登记冻结文件SHA，`runtime`为推理与验收程序，`slurm`为单GPU评价脚本，`evidence`保存训练来源。远端运行证据与模型快照在同名远端实验根中生成；本地operator目录保存每次操作审计。

使用真实T2W fixed103，逐例顺序推理，预处理/导出worker均为0。模型标签允许0–4，RC=4。官方MET按冻结评价器和mets参数运行，后续paired须另核验共同病例、20000 bootstrap seed20260831、原始值、CI、W/T/L、p与FP/FN。

状态：远端预检pass，515输入/标签SHA、1035/103 split映射、模型严格加载及官方评价环境通过。K2 job `3853018`（s2k2met103r1）、K4 job `3853019`（s2k4met103r1）已唯一held提交、审计后释放；2026-10-02本地UTC03:33:33/远端UTC03:22:42分别PENDING(Resources)、PENDING(Priority)，尚未推理，无FAILED。首次部署因SSH命令参数过长在远端脚本执行前退出；未产生job，失败审计保留，传输改为SCP后成功。

artifact_status=experimental_unvalidated
operator_approved=false
formal_gate_status=not_run_not_passed

同一103用于选点与报告，只能exploratory；K5仍需K1/K2各自科学门。
