"""Local marker/summary consistency review; not scientific or full-runtime acceptance."""
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import m


def main():
    audit = Path(sys.argv[1])
    result = json.loads((audit / 'RESULT.json').read_text())
    receipt = json.loads((audit / 'LOCAL_OPERATION_RECEIPT.json').read_text())
    for key in ('account_squeue', 'account_sacct', 'tracked_sacct', 'disk'):
        assert result[key]['returncode'] == 0, key
    assert receipt['own_vpn_stop_completed'] or receipt['preexisting_vpn_preserved']
    jobs = {r[0]: r for line in result['tracked_sacct']['stdout'].splitlines() if (r := line.split('|'))}
    for batch in result['batches'].values():
        assert batch['package_identity_match'] and not batch['package_drift']
        for markers in batch['markers'].values():
            assert not markers['FAILED.json']['exists']
        assert not any(log['critical_tail_lines'] for log in batch['logs'].values())
    raw = result['batches']['raw_r2']
    done = raw['markers']['dataset']['COMPLETE.ok']['summary']
    report = raw['markers']['dataset']['PREPROCESS_REPORT.json']
    assert jobs['4161285'][1:3] == ['COMPLETED', '0:0']
    assert jobs['4161285.batch'][1:3] == ['COMPLETED', '0:0']
    assert done['status'] == 'complete' and done['cases'] == 1165
    assert done['report_sha256'] == report['sha256']
    assert report['summary']['cases_ok'] == report['summary']['cases_processed'] == 1165
    assert report['summary']['errors'] == []
    assert done['resolution_warnings'] == report['summary']['warnings']
    for mod in ('t1c', 't1n', 't2w', 't2f'):
        expected = dict(train=823, val=103) if mod == 't2w' else dict(train=1035, val=130)
        assert done['modality_counts'][mod] == expected
    evaluation = result['batches']['r9_evaluation']
    selection_markers = evaluation['markers']['selection']
    selection = selection_markers['SELECTION.json']['summary']
    assert selection['status'] == 'exploratory_selected'
    assert selection_markers['COMPLETE.ok']['summary']['selection_sha256'] == selection_markers['SELECTION.json']['sha256']
    assert jobs['4159193'][1:3] == jobs['4159193.batch'][1:3] == ['COMPLETED', '0:0']
    selected = {}
    for mod, job in zip(('t1c', 't1n', 't2w', 't2f'), ('4159189', '4159190', '4159191', '4159192')):
        assert jobs[job][1:3] == jobs[job + '.batch'][1:3] == ['COMPLETED', '0:0']
        complete = evaluation['markers']['evaluation/' + mod]['COMPLETE.ok']['summary']
        assert complete['status'] == 'candidate_evaluation_complete'
        assert complete['cases']['count'] == 103 and len(complete['steps']) == 10
        rows = evaluation['candidates'][mod]
        assert len(rows) == 10
        values = selection['modalities'][mod]
        for row in rows:
            step = str(row['step'])
            assert row['cases'] == 103 and row['positive_cases'] == 94 and row['all_qc_pass']
            assert row['seeds'] == [20261013, 20261014, 20261015, 20261016]
            assert complete['candidate_summary_sha256'][step] == row['sha256']
            candidate = values['candidates'][step]
            assert candidate['candidate_summary_sha256'] == row['sha256']
            assert candidate['checkpoint_sha256'] == row['checkpoint_sha256']
            assert math.isclose(candidate['score'], row['primary_ssim'], abs_tol=1e-12)
        maximum = max(row['primary_ssim'] for row in rows)
        best = min(row['step'] for row in rows if maximum - row['primary_ssim'] <= 1e-6)
        assert values['selected_step'] == best
        assert values['selected_checkpoint_sha256'] == values['candidates'][str(best)]['checkpoint_sha256']
        selected[mod] = dict(step=best, primary_ssim=values['candidates'][str(best)]['score'],
                             authentic_secondary_ssim=values['secondary_scores'][str(best)],
                             checkpoint_sha256=values['selected_checkpoint_sha256'])
    progress = {}
    for mod, job in zip(('t1c', 't1n', 't2w', 't2f'), ('4161286', '4161287', '4161288', '4161289')):
        markers = raw['markers']['training/' + mod]
        assert jobs[job][1] == 'RUNNING' and not markers['COMPLETE.ok']['exists']
        last = markers['progress']['last_records'][-1]
        assert math.isfinite(last['loss']) and 0 < last['global_step'] < 200000
        progress[mod] = dict(job=job, step=last['global_step'], target=200000, percent=last['global_step']/2000,
                             checkpoint_step=markers['LATEST_CHECKPOINT.json']['summary']['global_step'])
    history = [line.split('|') for line in result['account_sacct']['stdout'].splitlines()]
    counts = {}
    for row in history:
        state = row[3].split()[0]
        counts[state] = counts.get(state, 0) + 1
    payload = dict(m.META, status='marker_and_summary_consistency_pass',
                   utc=datetime.now(timezone.utc).isoformat(), snapshot_sha256=m.sha(audit/'RESULT.json'),
                   remote_utc=result['remote_utc'], local_received_utc=result['local_received_utc'],
                   account_history_window='2026-10-01 onward (includes jobs overlapping start)',
                   account_history_counts=counts, selected=selected, raw_progress=progress,
                   raw_preprocess_cases=1165, raw_warning_count=len(report['summary']['warnings']),
                   limitations=['log tails only; no full-log or full-progress scan',
                                'no source images or checkpoint byte hashing/loading',
                                'case marker/index/CSV identities not individually reaudited',
                                'selection summaries verified; per-seed and bootstrap evidence not independently reaudited',
                                'no manual quality review, test images, model download, or remote writes'])
    m.exclusive(audit/'LOCAL_STATUS_REVIEW.json', payload)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == '__main__':
    main()
