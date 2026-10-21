# R9四路训练完成与最终模型拉回验收

## 输入与实际动作

用户要求检查是否结束，完成后拉回轻量结果与最终模型并检查服务器存储。2026-10-18通过aTrust，仅运行一次既有只读snapshot.py；完成门满足后新增pull_final.py读取限定范围的结果/模型身份、传输四份最终文件并后验身份/容量。没有提交、release、恢复训练、修改参数、源图重扫、影像下载、模型反序列化或服务器写入/删除。暂停监控未恢复。

## 训练终态

CPU4128262和四GPU作业及各batch均COMPLETED、ExitCode=0:0，squeue为空；不是依据空队列单独判成功。四路COMPLETE training_complete/global_step=200000，LATEST safe_boundary=true/global_step=200000，最终文件SHA与两marker一致。STARTED/RUN_CONTRACT存在，FAILED/SAFE_TERM无。60文件冻结包SHA仍80f432b454ea80cf3d4142cb87d194c7913c0eabd2da473b3821b06d10df63f6，无漂移。

| 模态 | Job | 作业用时 | 完成标记UTC（远端时钟） | 成功更新 |
|---|---|---|---|---|
| T1C | 4128266 | 18:28:25 | 2026-10-17T17:51:08.677008Z | 200000 |
| T1N | 4128267 | 16:27:27 | 2026-10-17T15:50:37.489075Z | 200000 |
| T2W | 4128268 | 16:46:15 | 2026-10-17T16:09:32.690412Z | 200000 |
| T2F | 4128269 | 17:45:46 | 2026-10-17T17:09:00.166347Z | 200000 |

完整progress逐行核对各200000条连续成功更新、loss全部finite，总800000更新；保存每1000步均值/最小/最大及末段。完整stdout/stderr逐行检查无关键错误，保存原日志SHA/大小与尾段。单步loss仅训练诊断，不是生成评价。

CPU工程完成926=train823+val103；PREPROCESS_REPORT零错误五量化警告与COMPLETE一致：01110-002/01339-001类别4，01132-002/01166-003/01245-003类别2。CSV5570行=train4968+val602；26例没有病灶行是报告字段，未exclude病例或改变823/103。四路共同数据身份一致，modality及对应consumed影像身份按模态不同，配置/父来源/CSV/报告/926 marker与索引哈希清单一致。实际训练入口消费门和本地报告复核，不冒充独立重新核验全926远端metadata或逐体素质量。

## 本地交付

- [最终模型](final_models_200000)：各模态一份diffusion_200000.pt，每份55806207bytes，总223224828bytes=212.884MiB；完整训练checkpoint含网络/optimizer/scaler/RNG等，不是仅网络导出或最佳选择点。
- [轻量结果](operator/pull_final_20261018T065725Z/lightweight)：50文件18882341bytes=18.008MiB，冻结计划/配置、submission、预处理报告/COMPLETE、lesions.csv、四路STARTED/RUN_CONTRACT/LATEST/COMPLETE、loss汇总及日志尾段。原bytes保留，派生汇总另命名。
- [拉回验收](operator/pull_final_20261018T065725Z/PULL_ACCEPTANCE.json)与[二次复核](operator/pull_final_20261018T065725Z/LOCAL_VERIFICATION.json)：54文件大小/SHA一致。首次附加比较误要求四模态全部identity相等，发现只有modality/consumed预期不同；按冻结入口修正本地检查后pass，无远端或合同更改，不归训练失败。

| 模态 | 最终模型SHA256 |
|---|---|
| T1C | cc1eb4e2ae6b0607dc9d1235ca01ac4745e858937d1e4af49a68e355d32686e0 |
| T1N | f6018169a53201a4ba28ed20ab90f88fe11480422038b6ab46d91896f515002b |
| T2W | 27ea3175e26b7aeaaba28f30a553cd71fa2eab0c7e79989355da46ef39aa28a5 |
| T2F | c642ea88fe73e479a2ddf5d663fe220c5bcada31e03cfea9d981ff24d0acc93b |

## 存储与时间

账号项目路径df/statvfs：2TiB总量，已用2005940568064bytes，余193082687488bytes=179.822GiB，92%占用；拉回前后相同。共享/public的df -h约25TiB余量，不替代账号可用额度。定向du R9 dataset9563752960bytes=8.907GiB、training9068377088bytes=8.446GiB；符号引用不重复计父R7/R8数据，不当全部926影像总物理占用。没有全盘扫描/释放空间/删除旧checkpoint。本地容量通过总模型量加4GiB保留门，下载后余190577344512bytes=177.489GiB。

一次快照本地UTC2026-10-18T06:53:57.498706Z/远端06:42:45.755062Z；拉回开始本地06:57:25.273256Z，远端清单06:47:04.767046Z、后验06:48:19.827249Z，本地验收06:59:27.678741Z、二次复核07:01:10.382340Z。两端约11分钟偏差分别记录，不混用计算时长或修改原UTC。finally关闭本次拥有的VPN，未影响先存用户会话。

## 验证边界

完成门与5项异常快照拒绝测试、远端代码语法检查、本地54文件二次SHA/配置/报告结构核对pass；模型未反序列化或逐tensor检查。fixedval103/20k候选/四seed/Heun18及SSIM/患者bootstrap评价、最佳checkpoint选择尚未执行，实现待独立核验；test104未读图评分。nnU-Net后续实际等价、独立全量/P1人工/P2未实测通过，waived_by_user_not_pass不是pass。

artifact_status=experimental_unvalidated；operator_approved=false；formal_gate_status=not_run_not_passed。训练完成和传输工程验收不等于生成质量或分割增益。历史失败、原覆盖门false及冻结包保持。
