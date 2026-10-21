"""Publish verified metric summaries, not a successful-pipeline acceptance report."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = sorted(HERE.glob('EXISTING_METRICS_*.json'))[-1]
snapshot = json.loads(source.read_text())
out = HERE / 'metrics_summary_20260911'
out.mkdir(exist_ok=False)
meta = dict(artifact_status='experimental_unvalidated', operator_approved=False, formal_gate_status='not_run_not_passed')
rows = []
for method in ('B','E_SNFH','F'):
    data = snapshot['metrics'][method]
    assert data['prediction_count'] == 103 and data['official_evaluation_complete'] is True
    assert data['evaluation_validation']['metrics_csv_sha256'] == data['175']['csv_sha256']
    for epoch in ('125','175'):
        for metric, value in data[epoch]['means'].items():
            rows.append(dict(method=method,epoch=int(epoch),metric=metric,official_mean=value,
                finite_case_count=data[epoch]['finite_counts'][metric],source_csv_sha256=data[epoch]['csv_sha256'],
                **meta, full_corrected_acceptance='not_completed', paired_bootstrap='not_run'))
assert len(rows) == 432
with (out/'OFFICIAL_MET_125_175_MEANS.csv').open('x',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

lines=['---','artifact_status: experimental_unvalidated','operator_approved: false','formal_gate_status: not_run_not_passed','---','',
    '# B/E-SNFH/F：125与175官方MET现有指标','',
    '三路175各103预测已生成，官方MET自身的validation和complete均存在，CSV摘要读取时已核对SHA。这里只报告现有评估指标，不宣布整个流水线验收通过。',
    '原GPU作业B=3671905、E-SNFH=3671906、F=3671908均为FAILED；paired=3671909为CANCELLED。失败原因是本次新增封装验收误把合法标签集合限制为0–3，遗漏合法标签4。原失败证据和runtime均未改写。',
    '全量修正后的体素复核因有界SSH超时未完成；本次没有重新提交GPU或创建恢复attempt。后续需要独立CPU恢复验收与paired统计，不能把以下均值变化解读为统计显著改善。','',
    f"远端摘要快照UTC：{snapshot['remote_utc']}；本机接收UTC：{snapshot['local_utc']}。远端时钟约落后本机10分钟；本次VPN已关闭。",'',
    '## 病灶Dice','',
    '| 路线 | ET125 | ET175 | RC125 | RC175 | TC125 | TC175 | WT125 | WT175 |',
    '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
for method in ('B','E_SNFH','F'):
    data=snapshot['metrics'][method]
    cells=[method]
    for label in ('et','rc','tc','wt'):
        cells.extend(f"{data[epoch]['means']['lesionwise_dsc_mean_'+label]:.6f}" for epoch in ('125','175'))
    lines.append('| '+' | '.join(cells)+' |')
lines += ['', '| 路线 | ET变化 | RC变化 | TC变化 | WT变化 |','| --- | --- | --- | --- | --- |']
for method in ('B','E_SNFH','F'):
    data=snapshot['metrics'][method]
    cells=[method]+[f"{data['175']['means']['lesionwise_dsc_mean_'+label]-data['125']['means']['lesionwise_dsc_mean_'+label]:+.6f}" for label in ('et','rc','tc','wt')]
    lines.append('| '+' | '.join(cells)+' |')
lines += ['', 'E-SNFH四标签Dice均值小幅上升；B的ET/TC下降；F的TC下降、WT基本持平。内部EMA上升没有转化为三路一致的官方Dice收益。','']
for title,prefix in (('全部病灶F1','all_instance_f1'),('小病灶F1','small_instance_f1'),('FP/例','all_instance_fp')):
    lines += ['## '+title,'','| 路线/epoch | ET | RC | TC | WT |','| --- | --- | --- | --- | --- |']
    for method in ('B','E_SNFH','F'):
        for epoch in ('125','175'):
            data=snapshot['metrics'][method][epoch]
            cells=[method+'/'+epoch]+[f"{data['means'][prefix+'_'+label]:.4f} ({data['finite_counts'][prefix+'_'+label]})" for label in ('et','rc','tc','wt')]
            lines.append('| '+' | '.join(cells)+' |')
    lines.append('')
lines += ['括号是对应指标实际finite病例数。病例总数103不等于每个指标都具有103个有效观测，不能把缺失填成0，也不能直接拿不同有效子集的均值差作显著性判断。','',
    '## 未完成事项','',
    '- 以冻结dataset.json为准接受合法标签0–4，完成全部预测体素与几何的独立复核。',
    '- 保留原FAILED状态，在独立恢复记录中验收现有预测和官方MET，不重跑GPU。',
    '- 补做175三路之间3个pair及各路175对125的3个pair；20000 bootstrap、seed20260831，报告有效n、胜平负、95%CI、未校正p和FP。',
    '- 固定103参与了模型选择与报告，所有结果仍是exploratory，formal gate未运行未通过。','',
    '[全部432个均值、有效n与源CSV SHA](OFFICIAL_MET_125_175_MEANS.csv)',
    f'[远端指标和审计摘要](../{source.name})','']
report=out/'EXISTING_OFFICIAL_MET_COMPARISON_20260911.md'
with report.open('x') as f:f.write('\n'.join(lines))
with (out/'SUMMARY_PROVENANCE.json').open('x') as f:
    json.dump(dict(meta,source_snapshot=str(source),source_snapshot_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        created_local_utc=datetime.now(timezone.utc).isoformat(),status='metrics_summary_only',original_gpu_jobs_remain_failed=True,
        corrected_full_volume_acceptance='not_completed',paired_bootstrap='not_run',new_attempt_created=False,new_jobs_submitted=False,
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.name!='SUMMARY_PROVENANCE.json'}),f,indent=2)
print(json.dumps(dict(report=str(report),mean_records=len(rows)),ensure_ascii=False))
