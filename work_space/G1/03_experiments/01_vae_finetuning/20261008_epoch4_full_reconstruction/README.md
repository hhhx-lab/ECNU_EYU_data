# G1 VAE epoch004全量图像重建与Drive交付

批次状态：双病例smoke job `3940942`已`COMPLETED 0:0`，两例8幅重建影像和2份原样标签的SHA、finite、原生几何、来源一致性验收通过。全量job `3940993`经独立held审计释放后`FAILED 1:0`，零病例输出；1030例数据、拉回及Drive上传均未完成。不得在原批次重投或覆盖失败证据。`artifact_status=experimental_unvalidated`，`operator_approved=false`，`formal_gate_status=not_run_not_passed`。

用户于2026-10-08指定epoch4权重重建1030例完整病例的四模态影像，拉回本地，并上传到Google Drive文件夹 `1JICH__ZmoCrF856Lvh65QLgSMV9_da8s`。输出为原生空间NIfTI float32，保持源seg逐字节相同，保存train823/val103/test104划分。epoch004权重SHA `5d86a0b3cbc5df34d64d5a0ff376eae453634a609bd24dec1a1191fde800b13b`。

与真实影像训练集不同，这些是VAE posterior mean重建并经G1 V3模态后处理的图像。用于四模态病灶Diffusion时必须新建独立实验，先做图像与标签质量验收，不能自动认定比真实影像更好。此前fixed103 threeway的失败证据保持原样。

2026-10-08只读预检见 `operator/readonly_preflight_20261008T041140Z/REMOTE_SOURCE_PREFLIGHT.json`：1030例四模态和seg均存在，源共30395884643 bytes，远端/public余27T，checkpoint SHA一致。Drive目标文件夹已在授权浏览器和现有`ecnu_gdrive`同步配置下验证为空，容量5TB；本地可用约160GiB。上传前仍需核实输出总大小、本地空间与逐文件SHA。

执行门控：静态预检/真实import/权重加载 -> 两例smoke（含旧验证报错的BraTS-MET-01117-003） -> 1030例全量 -> 逐文件SHA/geometry/finite/标签验收 -> 分批拉回 -> Drive上传并核验清单。旧结果、旧失败与原始数据不覆盖。

静态预检 `evidence/REMOTE_STATIC_PREFLIGHT.json` 已pass，真实导入和CPU严格加载epoch004权重通过。远端运行包SHA与本地源一致；本地提交回执见 `operator/submit_smoke_20261008T042937Z/SUBMISSION_RECEIPT.json`，计划SHA `d081764c30103d9f3d9010604a4fb0d78613066a6196d7f995ce0f1e6b54441a`，运行脚本SHA `525b2dd071c956a2d24d6ad9885118a2f45bfe5f7401998e6ea553d96e33a5da`。

2026-10-08阶段证据：`operator/smoke_snapshot_20261008T051424Z/SMOKE_STATUS.json`记录`3940942 COMPLETED 0:0`、`DATASET_VALIDATION.json`和`COMPLETE.ok`均pass，两个case manifest及10个实际文件重新核验。`operator/submit_full_20261008T052341Z/SUBMISSION_RECEIPT.json`记录全量job `3940993`的远端SHA/烟测文件门控、held审计和释放。全量运行包仍使用原冻结版本，`RECON_PHASE=full`，单A100、2天时限；但作业启动即失败，未进入重建。远端预检UTC `2026-10-08T05:13:39Z`，本地提交回执UTC `2026-10-08T05:24:54Z`，两机时钟约有11分钟差，不以时间戳先后代替作业证据。

全量失败只读快照见`operator/readonly_full_20261008T054740Z/FULL_STATUS.json`：远端UTC `2026-10-08T05:36:50Z`，job/batch均`FAILED 1:0`，空`squeue`符合已离队终态。stderr明确指出冻结Slurm脚本第35行`CASES[@]: unbound variable`。在该节点的Bash中，`set -u`与全量分支的空`CASES=()`组合导致启动即退出；没有`dataset/RUN_CONTRACT.json`、`COMPLETE.ok`或病例目录。不是VAE推理、OOM、CUDA、源影像或标签验收失败。此失败记录、原脚本和审计保持不变；后续如获授权，须独立新attempt修正空数组参数合同并重新静态验证，不得把smoke通过当作全量通过。
