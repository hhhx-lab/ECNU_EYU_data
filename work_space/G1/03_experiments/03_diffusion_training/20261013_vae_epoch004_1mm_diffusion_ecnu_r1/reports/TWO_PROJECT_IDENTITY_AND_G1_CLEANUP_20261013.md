# 两个远端项目身份与G1清理候选

整理UTC：2026-10-13T14:40:30Z；末次证据接收时间另见原始JSON。仅只读核查和本地清单整理，**没有删除、改名、腾挪服务器目录，没有安装、训练、提交或取消作业**。研究登记保持`experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。

## 1. zdn_deepcpg是什么

远端路径：`/public/home/zqchen/zdn_deepcpg`，完整du为874.70 GiB。

`setup.py`明确描述为Deep learning for predicting CpG methylation、Single cells、Epigenetics、DNA methylation，包版本1.0.3。实际Slurm调用`dcpg_train.py`，训练CpG上下文RnnL2和DNA序列CnnL2h128模型，环境`deepcpg-train`。人神经元脚本使用c5作验证染色体，其他指定染色体作训练；也有小鼠神经元、造血细胞和Melissa相关文件。**这是单细胞DNA/CpG甲基化预测项目，不是MRI比赛G1，不是普通缓存。** 名称zdn不能独立证明实际负责人身份。

| 内容，相对zdn_deepcpg | GiB | 解释 |
|---|---:|---|
| `neuron_homo_h5_new` | 459.44 | 人神经元HDF5输入，实际训练脚本明确引用 |
| `neuron_mouse_h5_new` | 392.32 | 小鼠神经元HDF5输入，目录和脚本分支对应 |
| `melissa` | 9.18 | Melissa代码/细胞表/染色体特征等，含约6.27 GiB的LG_ACCPU_18chr |
| `logs` | 4.48 | 历史作业日志，未逐条确认保留规则 |
| `grch38_fasta` | 2.93 | 人参考基因组FASTA |
| `mm10_fasta` | 2.59 | 小鼠参考基因组FASTA |
| `models` | 1.99 | CpG/DNA模型结果目录 |

两个HDF5目录合计851.76 GiB，是本项目占用的绝大部分。文件清点观察313个H5，另有基因组、特征和结果；没有读取HDF5内容或模型来验证有效性、压缩率、原始来源、重复与备份。**不能因为体积大或当前队列为空就删除。** 可另向负责人核实H5是否可从完整保留的原始profiles/参考基因组精确重建或迁移归档；本次不开展此项目的清理。

## 2. 比赛G1是什么，占用在哪

远端路径：`/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1`。前序完整du566.42 GiB；本次继续细分其中的大目录，不重复做全home扫描。GiB为实际分配字节/2^30，父子不能重复相加。

```text
G1/                                      566.42 GiB（前序全量du）
  code/                                  194.46 GiB
    brats2025-latent-ensemble-...-v3/      193.60 GiB
      data/                              109.81 GiB
        attention_masks/                  40.87 GiB
        lesion_weights/                   40.87 GiB
        latents/                          10.06 GiB
        output/run_3104668/                16.38 GiB
        evaluation/val/                    1.63 GiB
        input/, input_inference/           指向V2的symlink，不计为完整复制数据
      training/                           83.67 GiB
        bbdm/                             68.10 GiB
        endec/                            14.80 GiB
        vae_finetuned/                     0.78 GiB
    ...-v2/                                0.87 GiB（不解引用链接）
  results/                               371.96 GiB
    experimental_unvalidated/
      epoch004_full_reconstruction_r2/   355.26 GiB
      threeway_fixed103_attempt03/         8.43 GiB
      extended_finetune_attempt03/          4.69 GiB
      epoch004_full_reconstruction_r1/      0.94 GiB
      其他runtime、预检和失败记录          轻量历史证据
    single_model_eval_r4/                  2.65 GiB
  data/, docs/                             主要为链接/轻量资料
```

V3不是只有源码：它同时存训练输入缓存、补全影像、checkpoint。该线是MAISI VAE压缩后用EncDec/BBDM进行缺失T2W补全。`selected_vae.json`实际仍指旧run3034060，而不是新epoch004；4120 latent不能冒充epoch004重建或新1mm四模态Diffusion输入。新重建r2和新Diffusion实验是另外的分支。

README包含上游论文/比赛名次说明，只是来源介绍，不是我们实验取得同等结果的证据。当前公开代码、目录metadata和旧运行状态要分别理解。

## 3. G1候选分级：没有当前可直接批量删除的白名单

| 精确候选 | GiB | 当前判断与必须补的条件 |
|---|---:|---|
| V3 `attention_masks/*/*_attmask_256_256_160.npy`，1030文件 | 40.23 | 理论派生mask；当前再生成所需空间合同缺失，暂不可认定安全可删 |
| V3 `lesion_weights/*/*_lesion_weights_256_256_160.npy`，1030文件 | 40.23 | 理论派生weight；同样须恢复原生成合同、逐例核验再生成一致性 |
| V3主run3056244/3056245非最终checkpoint，23文件 | 64.70 | 仅按文件名扣除已知评价权重后的盘点，不是冗余证明；须全历史评价/选点/恢复引用核查、完整备份并另获授权 |
| V3 `data/output/run_3104668/**/intermediate*`内文件，795文件 | 12.84 | 补全中间影像/元数据；保留正式五文件与空间/生成证据、完整备份/SHA通过后才考虑，不以路径名直接删 |
| `__pycache__` | 不足0.001 | 极小可重建缓存候选，收益可忽略；本次也未删除 |

### mask/weight：80.47 GiB潜力，但存在硬性缺口

每类1030份256×256×160 float32各40.23438 GiB；各自64×64×40 latent数组只约0.62866 GiB，两套低分辨率共约1.26 GiB应继续保留。

源代码会从真实seg按`spatial_transform.json`映射，写高分辨率数组再max-pool到latent。训练loss使用64×64×40，而训练中的验证/可视化代码还会读取高分辨率attention mask；不是所有高分辨率文件都从未被使用。

**实际存在性核查：1030/1030源seg在，`data/latents/<case>/spatial_transform.json`为0/1030。** 所以目前脚本不能在不补原空间合同的情况下直接重建这些数组。此缺口不自动证明旧数组/结果错误，只表明当前源码与旧生成资产合同不完整。不能运行当前preprocess随意覆盖旧latent或套用新epoch004合同补洞。

正确次序是找回原始生成版本/空间记录，冻结源标签和split/源SHA，在独立位置验证重建一致性，确认无训练/验证/下游引用，再申请清理精确高分辨率白名单。**没有进行此全量验收，80.47 GiB不是已确认可释放量。**

### checkpoint：原件不能从最终权重“重新生成”

V3 training共有35份`.pt`约83.60 GiB，其余少量日志/配置。BBDM主run3056245有21权重约62.17 GiB；EncDec主run3056244有4权重约7.39 GiB。保留已知评价的`model_111105.pt`和`model_37035.pt`后，剩余23份约64.70 GiB只是归档审查候选，可能仍是历史比较或恢复依据，不能作为可重建缓存处理。

至少保护旧VAE run3034060的best/选点/训练证据、新epoch004 SHA绑定权重、实际用于原生空间补全的EncDec/BBDM权重和所有配置/split/MET/paired。旧run3049489/3049490和其他VAE checkpoints本次不列入删除额度，不因为未跑完/旧日期就删除。

### 生成结果与失败产物

run3104668有795个intermediate范围文件共12.84 GiB；剩余1329文件约3.54 GiB含265例正式结果和元数据，但本次仅文件统计，未重新验收1325个正式五文件及其SHA/备份。中间路径内可能有空间变换/生成说明，后续必须保留轻量证据。

threeway attempt03的8.43 GiB主要为`stage1_reconstruction/native_t2w`。它是失败阶段的已有输出，属于必须保留的原失败证据，**不以失败名义列为可删**。r1 smoke/失败、其他预检attempt、logs和冻结runtime亦不清。

epoch004重建355.26 GiB是用户指定的新训练数据，不删；V3 latent10.06 GiB绑定旧VAE/空间合同，也不默认删除。评价影像、标准补全数据和其他历史数据只有逐SHA确认完整外部备份后才可能另行归档。

## 4. 建议顺序与边界

1. 优先考虑前次较低风险的账号pip/Conda压缩包/索引缓存20.56 GiB，这不属于G1模型数据；仍需明确授权。
2. G1优先补齐80.47 GiB派生数组的旧空间/重建合同，不能先删后补。
3. checkpoint和12.84 GiB中间图走“查引用、保留原件备份、验证恢复、精确白名单”的归档审查，不按缓存清理。
4. zdn_deepcpg先确认负责人/用途/原始来源和备份，不参与本次MRI清理。

本次账号可用仍约72.89 GiB，squeue单次成功空输出；这不是无后台进程/未来引用的证明。未做全目录hash或NIfTI/H5体素验收，也不把所有潜在候选相加宣称可释放。远端三次有界只读，各次显式分流VPN并只关闭本次拥有的VPN，包装器均成功结束。

## 原始证据

- [目录/格式/大小](../operator/readonly_two_project_inventory_20261013T143600Z/TWO_PROJECT_INVENTORY.json)：远端UTC14:25:06，本地接收14:36:26；含逐目录和35权重前的文件metadata。
- [实际源码与权重清单](../operator/readonly_two_project_identity_20261013T143727Z/TWO_PROJECT_IDENTITY.json)：实际DeepCpG Slurm、G1生成源码、数组header抽查、35 checkpoint路径与大小、旧VAE选择和数据链接。
- [空间合同缺口与中间图](../operator/readonly_g1_rebuild_contract_20261013T144116Z/G1_REBUILD_CONTRACT.json)：1030标签/0空间记录，代码消费引用、输出分类；双方时间戳在JSON。
- [先前账号缓存候选](CLEANUP_CANDIDATES_READONLY_20261013.md)、[总容量与占用](STORAGE_OCCUPANCY_20261013.md)。

原始大小/存在性证据不改写；本报告负责人工解释和风险分级，不授权清理、不恢复冻结P0、不启动新的训练。
