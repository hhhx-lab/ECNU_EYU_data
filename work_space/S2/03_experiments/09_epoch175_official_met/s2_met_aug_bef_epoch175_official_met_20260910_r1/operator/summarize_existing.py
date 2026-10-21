"""Summarize verified175 MET files while retaining the original failed-job status."""
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 's2_met_aug_btof_selected103_official_met_20260907_r2'
META = dict(artifact_status='experimental_unvalidated', operator_approved=False, formal_gate_status='not_run_not_passed')
dirs = sorted(p.parent for p in (ROOT/'operator').glob('outputs_audit_*/RETRIEVAL_RECEIPT.json'))
assert dirs, 'No completed retrieval receipt'
DEST = dirs[-1]
audit = json.loads((DEST/'READ_ONLY_OUTPUT_VERIFICATION.json').read_text())
receipt = json.loads((DEST/'RETRIEVAL_RECEIPT.json').read_text())
assert audit['status'] == 'verified_existing_outputs_only'
for item in receipt['files']:
    path = ROOT/item['relative_path']
    assert path.stat().st_size == item['size'] and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']

data, all_rows = {}, []
for method in ('B','E_SNFH','F'):
    data[method] = {}
    for epoch, root in ((125,OLD),(175,ROOT)):
        path = root/f'cohort/fixed103/models/{method}/evaluation/leaderboard_metrics.csv'
        with path.open(newline='') as f:
            reader = csv.DictReader(f)
            rows, fields = list(reader), reader.fieldnames
        assert len(rows)==106 and len(fields)==73
        data[method][epoch] = {}
        for metric in fields[1:]:
            values = []
            for row in rows[:103]:
                value = float(row[metric]) if row[metric] else float('nan')
                if math.isfinite(value): values.append(value)
            aggregate = float(rows[-3][metric])
            assert math.isclose(mean(values),aggregate,rel_tol=0,abs_tol=1e-12)
            data[method][epoch][metric] = (aggregate,len(values))
            all_rows.append(dict(method=method,epoch=epoch,metric=metric,official_mean=aggregate,finite_case_count=len(values),
                **META, source_jobs175_status='FAILED_LABEL_CHECK', paired_inference_status='not_run'))
with (DEST/'OFFICIAL_MET_125_175_MEANS.csv').open('x',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)

lines=['---','artifact_status: experimental_unvalidated','operator_approved: false','formal_gate_status: not_run_not_passed','---','',
    '# 三路175官方MET现有结果：只读复核','',
    '原GPU作业3671905/3671906/3671908仍为FAILED，paired3671909为CANCELLED。失败点是新增封装验收遗漏合法标签4；不是将旧作业改判成功。',
    '本次按冻结模型dataset.json的0–4标签定义重新只读核查，三路各103预测的病例、几何、finite与标签检查通过，官方MET自身validation/complete和SHA通过。未提交任何恢复作业，也未重跑GPU。',
    '以下为现有官方CSV均值，括号为实际finite病例数。175三路横评和175对125的bootstrap/置信区间尚未完成，不作显著性结论。','',
    f"远端复核结束UTC：{audit['remote_completed_utc']}；本机拉回结束UTC：{receipt['local_completed_utc']}。两地时钟不同，远端约慢10分钟。",'',
    f"轻量文件：{len(receipt['files'])}份，逐文件SHA复核通过；checkpoint、NIfTI和events JSONL未拉回。本次VPN已关闭。",'']
labels=('et','rc','tc','wt')
for title,prefix in (('病灶Dice','lesionwise_dsc_mean'),('全部病灶F1','all_instance_f1'),('小病灶F1','small_instance_f1'),('FP/例','all_instance_fp')):
    lines += ['## '+title,'','| 路线/epoch | ET | RC | TC | WT |','| --- | --- | --- | --- | --- |']
    for method in ('B','E_SNFH','F'):
        for epoch in (125,175):
            cells=[f'{method}/{epoch}']+[f'{data[method][epoch][prefix+"_"+label][0]:.4f} ({data[method][epoch][prefix+"_"+label][1]})' for label in labels]
            lines.append('| '+' | '.join(cells)+' |')
    lines.append('')
lines += ['## 边界','',
    '固定103例同时用于checkpoint选择和报告，只能作exploratory。原五路125预算结果不变；本次175是额外训练预算的收敛实验，不是与125等预算比较。',
    '原r1计划、runtime、失败标记和launch audit均保留。恢复工作应使用独立记录，仅复用已验收预测和官方MET，补正确的标签验收和配对统计；不原位覆盖或重复GPU推理。','',
    '[全部432个均值与有效n](OFFICIAL_MET_125_175_MEANS.csv)',
    '[只读复核细节](READ_ONLY_OUTPUT_VERIFICATION.json)',
    '[轻量拉回SHA清单](RETRIEVAL_RECEIPT.json)','']
report=DEST/'EXISTING_OFFICIAL_MET_RESULTS_20260911.md'
with report.open('x') as f: f.write('\n'.join(lines))
print(json.dumps(dict(report=str(report),light_files=len(receipt['files']),means=data),ensure_ascii=False))
