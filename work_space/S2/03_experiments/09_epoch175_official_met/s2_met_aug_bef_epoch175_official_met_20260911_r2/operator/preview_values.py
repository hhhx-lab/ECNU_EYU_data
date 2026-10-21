"""A clearly provisional raw-value table from a previously SHA-verified snapshot."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 's2_met_aug_bef_epoch175_official_met_20260910_r1/operator/EXISTING_METRICS_20260911T075919Z.json'
META = dict(artifact_status='experimental_unvalidated', operator_approved=False, formal_gate_status='not_run_not_passed')


def main():
    snapshot = json.loads(SOURCE.read_text())
    target = ROOT / 'operator/existing_official_values_20260911'
    target.mkdir(exist_ok=False)
    rows = []
    for method, route in snapshot['metrics'].items():
        for metric, v125 in route['125']['means'].items():
            v175 = route['175']['means'][metric]
            rows.append(dict(method=method, metric=metric, mean125=v125, mean175=v175,
                             finite_n125=route['125']['finite_counts'][metric], finite_n175=route['175']['finite_counts'][metric],
                             delta175_minus125=v175-v125 if v125 is not None and v175 is not None else None))
    assert len(rows) == 216
    with (target / 'OFFICIAL_MET_125_175_SIDE_BY_SIDE.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = ['---', 'artifact_status: experimental_unvalidated', 'operator_approved: false',
            'formal_gate_status: not_run_not_passed', '---', '', '# 125 与 175 epoch 官方 MET 原始值对比', '',
            '本表读取已核对 SHA 的官方 MET 现有输出。独立 CPU 恢复作业 3679404 已提交，完整体素验收和 paired bootstrap 尚在处理中；本表不是最终验收通过通知。原 GPU 失败证据保持不变。', '',
            '三路均为同一固定 103 例、真实 T2W 输入；每项均值的有效病例数按官方 CSV 保留。空值不补零。同一 103 例兼用于 checkpoint 选择和报告，只作 exploratory，不宣称正式优越性。', '',
            '数值快照本机 UTC：' + snapshot['local_utc'] + '；远端 UTC：' + snapshot['remote_utc'] + '。远端时钟落后约 10 分半。', '']
    sections = [('病灶 Dice', 'lesionwise_dsc_mean_'), ('全部病灶 F1','all_instance_f1_'),
                ('大病灶 F1','large_instance_f1_'), ('小病灶 F1','small_instance_f1_'),
                ('假阳性 FP','all_instance_fp_'), ('假阴性 FN','all_instance_fn_'),
                ('病灶 HD95','lesionwise_hd95_mean_'), ('病灶 NSD','lesionwise_nsd_mean_')]
    def fmt(value):
        return 'NA' if value is None else f'{value:.6f}'
    for title,prefix in sections:
        text += ['## ' + title, '', '| 路线 | 标签 | 125 均值 | 175 均值 | 125 有效 n | 175 有效 n | 175-125 |',
                 '|---|---|---:|---:|---:|---:|---:|']
        for row in rows:
            if row['metric'].startswith(prefix):
                text.append(f"| {row['method']} | {row['metric'][len(prefix):].upper()} | {fmt(row['mean125'])} | {fmt(row['mean175'])} | {row['finite_n125']} | {row['finite_n175']} | {fmt(row['delta175_minus125'])} |")
        text += ['']
    text += ['## 目前可读出的变化', '',
             'E-SNFH 四标签病灶 Dice 均值均小幅上升；B 的 ET/TC 下降，F 的 TC 下降、WT 基本持平。内部 EMA 最佳出现在175，不等于官方分割指标必然改善。统计显著性待 paired 结果，不能只根据均值判断。', '',
             '病灶 Dice 对应 lesionwise_dsc_mean，并非全体素 Dice。不能把图像重建质量、offline 补全加分割、online 增强加分割的结论混在一起。本批输入为真实 T2W，不是 G1 合成图像。', '']
    with (target / 'OFFICIAL_MET_125_175_SIDE_BY_SIDE.md').open('x') as f:
        f.write('\n'.join(text))
    with (target / 'PROVENANCE.json').open('x') as f:
        json.dump(dict(META, acceptance='pending_cpu_recovery', cpu_job='3679404', created_local_utc=datetime.now(timezone.utc).isoformat(),
                       source_snapshot=str(SOURCE), source_snapshot_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                       metric_rows=216, metric_count=72, methods=3), f, indent=2)
    with (target / 'OFFICIAL_MET_125_175_SIDE_BY_SIDE.csv').open(newline='') as f:
        check = list(csv.DictReader(f))
    assert len(check) == 216
    for row in check:
        assert row['mean125'] and row['mean175']
    print(json.dumps(dict(directory=str(target), raw_comparison_rows=len(check)), ensure_ascii=False))


if __name__ == '__main__':
    main()
