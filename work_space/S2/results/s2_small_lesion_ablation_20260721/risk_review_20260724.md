# S2 逐病例风险复核

评估口径：BraTS-evaluation `mets`，固定 103 例，`vol_threshold=27`，
`overlap_threshold=0.2`。下表仅列用于定位风险的极端病例，不替代完整
`leaderboard_metrics.csv`。

## 共同高风险病例

| 风险 | B | A-1 | E | A-1+E |
|---|---:|---:|---:|---:|
| `BraTS-MET-01351-002` WT FN | 95 | 92 | 94 | 92 |
| `BraTS-MET-00014-000` WT FN | 33 | 32 | 32 | 33 |
| `BraTS-MET-01351-003` WT FN | 21 | 20 | 21 | 20 |
| `BraTS-MET-01191-003` RC FN | 2 | 2 | 3 | 1 |
| `BraTS-MET-01134-003` RC lesionwise DSC | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 选择相关风险

- E 的 WT small-instance F1 为 `0.333083`，高于 B 的 `0.277048`；相对 B 的逐病例可比
  统计为改善 9、持平 30、下降 7。
- E 的 RC all-instance F1 为 `0.421053`，但 RC FN 为 `0.116505`，高于 B 的
  `0.067961`。E 的 RC FP 仅 `0.038835`，说明主要收益来自减少假阳性，而不是完整
  解决漏检。
- RC small-instance F1 仅有 3 个可比较病例：B/A-1/E 均为 `0`，A-1+E 为 `0.166667`。
  该分层样本量过小，不能单独支持 A-1+E。
- A-1+E 的 WT all FP 为 `1.310680`，高于 B/A-1/E 的 `0.747573/1.097087/0.970874`，
  与其 WT all-instance F1、DSC、NSD 同步退化。

## 复核范围

- 已检查 WT/RC 的 lesionwise DSC、NSD、small/large-instance F1 以及 all-instance FN/FP。
- 官方解析只提供 `small_instance` 和 `large_instance`，没有独立 tiny 列；没有将
  small 指标冒充 tiny。
- 完整逐病例数据保留在四组 `leaderboard_metrics.csv`，原始 Panoptica JSON 未改写。
