# BraTS 2026 Task 1 Validation Leaderboard 指标归档

> 状态：`experimental_unvalidated`

本目录归档了 BraTS 2026 Task 1 validation leaderboard 按 24 个指标分别排序后的最佳 75 条有限值记录，对应每个指标三页、每页 25 条。数据抓取自公开排行榜页面，未上传任何内容，也未包含访问凭据。

## 来源

- 排行榜：https://challenges.synapse.org/Challenges/DetailsPage/Task1?id=syn74274097
- Synapse 表实体：`syn74508245`
- 抓取完成时间（UTC）：`2026-08-06T13:45:02.800Z`
- 排行榜总记录数（抓取时页面显示）：1,004
- 单页记录数：25
- 每指标归档：75 条有限值记录

## 文件结构

- `csv/`：24 份完整字段 CSV 排名表。
- `json/`：24 份完整字段 JSON 快照。
- `BraTS2026_Task1_Valid_Leaderboard_Top75_By_Metric_20260806.xlsx`：索引页加 24 个指标页，共 25 个工作表。
- `字段与排序方法.md`：字段、排序方向、并列和缺失值处理规则。
- `CAPTURE_MANIFEST.json`：来源、行数、排名边界和每份 CSV/JSON 的 SHA256。
- `WORKBOOK_VERIFICATION.json`：工作簿结构、代表性检查和渲染结果。
- `SHA256SUMS.txt`：本目录所有交付文件的 SHA256（不含清单自身）。

## 指标文件

| 指标 ID | 中文说明 | 最优方向 | CSV | JSON |
|---|---|---|---|---|
| `Lesionwise_dsc_mean_et` | 病灶级 DSC 均值（ET） | 降序（越大越优） | `csv/Lesionwise_dsc_mean_et.csv` | `json/Lesionwise_dsc_mean_et.json` |
| `Lesionwise_dsc_mean_rc` | 病灶级 DSC 均值（RC） | 降序（越大越优） | `csv/Lesionwise_dsc_mean_rc.csv` | `json/Lesionwise_dsc_mean_rc.json` |
| `Lesionwise_dsc_mean_tc` | 病灶级 DSC 均值（TC） | 降序（越大越优） | `csv/Lesionwise_dsc_mean_tc.csv` | `json/Lesionwise_dsc_mean_tc.json` |
| `Lesionwise_dsc_mean_wt` | 病灶级 DSC 均值（WT） | 降序（越大越优） | `csv/Lesionwise_dsc_mean_wt.csv` | `json/Lesionwise_dsc_mean_wt.json` |
| `Lesionwise_nsd_mean_et` | 病灶级 NSD 均值（ET） | 降序（越大越优） | `csv/Lesionwise_nsd_mean_et.csv` | `json/Lesionwise_nsd_mean_et.json` |
| `Lesionwise_nsd_mean_rc` | 病灶级 NSD 均值（RC） | 降序（越大越优） | `csv/Lesionwise_nsd_mean_rc.csv` | `json/Lesionwise_nsd_mean_rc.json` |
| `Lesionwise_nsd_mean_tc` | 病灶级 NSD 均值（TC） | 降序（越大越优） | `csv/Lesionwise_nsd_mean_tc.csv` | `json/Lesionwise_nsd_mean_tc.json` |
| `Lesionwise_nsd_mean_wt` | 病灶级 NSD 均值（WT） | 降序（越大越优） | `csv/Lesionwise_nsd_mean_wt.csv` | `json/Lesionwise_nsd_mean_wt.json` |
| `Small_instance_tp_et` | 小实例 TP（ET） | 降序（越大越优） | `csv/Small_instance_tp_et.csv` | `json/Small_instance_tp_et.json` |
| `Small_instance_tp_rc` | 小实例 TP（RC） | 降序（越大越优） | `csv/Small_instance_tp_rc.csv` | `json/Small_instance_tp_rc.json` |
| `Small_instance_tp_tc` | 小实例 TP（TC） | 降序（越大越优） | `csv/Small_instance_tp_tc.csv` | `json/Small_instance_tp_tc.json` |
| `Small_instance_tp_wt` | 小实例 TP（WT） | 降序（越大越优） | `csv/Small_instance_tp_wt.csv` | `json/Small_instance_tp_wt.json` |
| `Small_instance_fn_et` | 小实例 FN（ET） | 升序（越小越优） | `csv/Small_instance_fn_et.csv` | `json/Small_instance_fn_et.json` |
| `Small_instance_fn_rc` | 小实例 FN（RC） | 升序（越小越优） | `csv/Small_instance_fn_rc.csv` | `json/Small_instance_fn_rc.json` |
| `Small_instance_fn_tc` | 小实例 FN（TC） | 升序（越小越优） | `csv/Small_instance_fn_tc.csv` | `json/Small_instance_fn_tc.json` |
| `Small_instance_fn_wt` | 小实例 FN（WT） | 升序（越小越优） | `csv/Small_instance_fn_wt.csv` | `json/Small_instance_fn_wt.json` |
| `Small_instance_fp_et` | 小实例 FP（ET） | 升序（越小越优） | `csv/Small_instance_fp_et.csv` | `json/Small_instance_fp_et.json` |
| `Small_instance_fp_rc` | 小实例 FP（RC） | 升序（越小越优） | `csv/Small_instance_fp_rc.csv` | `json/Small_instance_fp_rc.json` |
| `Small_instance_fp_tc` | 小实例 FP（TC） | 升序（越小越优） | `csv/Small_instance_fp_tc.csv` | `json/Small_instance_fp_tc.json` |
| `Small_instance_fp_wt` | 小实例 FP（WT） | 升序（越小越优） | `csv/Small_instance_fp_wt.csv` | `json/Small_instance_fp_wt.json` |
| `Small_instance_f1_et` | 小实例 F1（ET） | 降序（越大越优） | `csv/Small_instance_f1_et.csv` | `json/Small_instance_f1_et.json` |
| `Small_instance_f1_rc` | 小实例 F1（RC） | 降序（越大越优） | `csv/Small_instance_f1_rc.csv` | `json/Small_instance_f1_rc.json` |
| `Small_instance_f1_tc` | 小实例 F1（TC） | 降序（越大越优） | `csv/Small_instance_f1_tc.csv` | `json/Small_instance_f1_tc.json` |
| `Small_instance_f1_wt` | 小实例 F1（WT） | 降序（越大越优） | `csv/Small_instance_f1_wt.csv` | `json/Small_instance_f1_wt.json` |

## 使用边界

该归档只是排行榜公开页面的时间点快照，不代表官方最终榜单、获奖结论、模型复现结果或比赛方审计通过。队伍名称、提交时间和指标值均按抓取时页面保留。
