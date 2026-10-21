# G1 epoch004 strict1mm局部修复接续R8

用户授权“就这样改然后接着做预处理，完成后自动开始四路训练”。独立R8，旧R7失败和输出不改。仅01110-002标签4精确4个源最近邻命中背景位置恢复；原19体素3.8mm3的3体素小组件未采到，不宣称全部细节保留。全局源码37份SHA、图像/统计mask/归一化/网格/其他病例算法不改。

471候选先逐6文件SHA校验、重建病灶rows，建立新目录符号链接与显式真实路径身份引用，不复制大影像。455未完成例重采样，若其他整类丢失/合同漂移则停止，不自动扩大例外或降门。末尾926=train823+val103完整CSV/marker/例外和实际路径身份，test104不读图。

CPU cpu_192G8CPU90G24h；四单A1008CPU64G48h各64³batch16 AMP eager非DDP seed20261013 AdamW2e-4beta1=.9各200000成功更新，完整5k/final/safeTERM checkpoint不改。新唯一held审计释放，afterok新CPU，非立即GPU运行保证。固定val103评价沿用，test104不读图/不评分。

独立全量/P1/P2等待waived_by_user_not_pass，原STRICT_ORIGINAL_GATE/NONZERO_COMPARISON_GATE仍false，工程成功不称科学优越。experimental_unvalidated/operator_approved=false/formal_gate_status=not_run_not_passed。失败保留无自动newattempt。
