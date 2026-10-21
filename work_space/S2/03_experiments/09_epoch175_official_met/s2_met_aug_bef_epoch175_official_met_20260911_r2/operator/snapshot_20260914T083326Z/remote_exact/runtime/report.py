"""Show original means alongside pairwise estimates, with explicit denominators."""
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from recovery import ROOT, META, METHODS, require, read, write


def metrics(method, epoch):
    path = ROOT / 'sources' / method / f'epoch{epoch}/leaderboard_metrics.csv'
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    return {r['subject_id']: r for r in rows[:103]}, rows[-3]


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def fmt(value):
    return 'NA' if value is None else f'{value:.6f}'


def main():
    rows = []
    for method in METHODS:
        a, mean_a = metrics(method, 175)
        b, mean_b = metrics(method, 125)
        require(list(a) == list(b), 'comparison case order')
        payload = read(ROOT / 'results' / (method + '_175_vs125') / 'PAIRED_COMPARISON.json')
        require(payload['status'] == 'pass' and payload['case_count'] == 103 and payload['metric_count'] == 72 and len(payload['pairs']) == 1, 'paired scope contract')
        for record in payload['pairs'][0]['metrics']:
            metric = record['metric']
            va, vb = [number(a[c][metric]) for c in a], [number(b[c][metric]) for c in a]
            common = [(x, y) for x, y in zip(va, vb) if x is not None and y is not None]
            require(len(common) == record['paired_complete_case_count'], 'paired denominator mismatch')
            raw_low, raw_high = record['benefit_ci95_low'], record['benefit_ci95_high']
            if raw_low is not None and record['direction'] == 'lower_is_better':
                raw_low, raw_high = -raw_high, -raw_low
            ma, mb = number(mean_a[metric]), number(mean_b[metric])
            row = dict(method=method, metric=metric, direction=record['direction'],
                       official_mean125=mb, official_mean175=ma, finite_n125=sum(x is not None for x in vb),
                       finite_n175=sum(x is not None for x in va), aggregate_delta175_minus125=None if ma is None or mb is None else ma-mb,
                       paired_n=len(common), paired_mean125=sum(y for _, y in common)/len(common) if common else None,
                       paired_mean175=sum(x for x, _ in common)/len(common) if common else None,
                       paired_delta175_minus125=record['raw_mean_a_minus_b'], raw_delta_ci95_low=raw_low,
                       raw_delta_ci95_high=raw_high, wins175=record['wins_a'], ties=record['ties'], losses175=record['wins_b'],
                       unadjusted_bootstrap_p=record['paired_two_sided_bootstrap_p'], bootstrap_replicates=20000, seed=20260831)
            if record['status'] == 'available':
                require(row['wins175'] + row['ties'] + row['losses175'] == len(common), 'W/T/L count')
                require(math.isclose(row['paired_mean175']-row['paired_mean125'], row['paired_delta175_minus125'], abs_tol=1e-10), 'paired delta direction')
            rows.append(row)
    require(len(rows) == 216, 'longitudinal metric count')
    cross = read(ROOT / 'results/epoch175_cross_route/PAIRED_COMPARISON.json')
    require(cross['status'] == 'pass' and len(cross['pairs']) == 3 and cross['metric_count'] == 72 and cross['case_count'] == 103, 'cross-route count')
    for folder in ROOT.glob('results/*'):
        if folder.is_dir():
            p = read(folder / 'PAIRED_COMPARISON.json')
            require(p['bootstrap'] == dict(replicates=20000, seed=20260831), 'bootstrap parameters')
            for pair in p['pairs']:
                require(len(pair['metrics']) == 72, 'pair metric count')
                for record in pair['metrics']:
                    if record['status'] == 'available':
                        require(record['wins_a']+record['ties']+record['wins_b'] == record['paired_complete_case_count'], 'pair W/T/L count')
    path = ROOT / 'results/OFFICIAL_MET_125_VS175_VALUES_AND_PAIRED.csv'
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(ROOT / 'results/REPORT_VALIDATION.json', dict(META, status='pass', methods=3, epochs=[125,175],
          same_route_pairs=3, cross_route_pairs=3, metric_count=72, paired_records=432,
          raw_value_rows=216, remote_utc=datetime.now(timezone.utc).isoformat()))
    text = ['---', 'artifact_status: experimental_unvalidated', 'operator_approved: false',
            'formal_gate_status: not_run_not_passed', '---', '', '# B/E-SNFH/F: 125 vs 175 Official MET', '',
            'CPU-only recovery of existing official MET outputs. Original GPU jobs remain FAILED; no training, prediction or evaluator was rerun.', '',
            'Same fixed103 authentic-T2W development cases were used for checkpoint selection and reporting. Exploratory convergence experiment, not an untouched-test result or formal superiority claim. The original five-route equal-budget comparison is unchanged.', '',
            'Official means retain their original per-metric finite denominators. Paired statistics use only cases finite at both epochs. The difference of aggregate means can differ from the mean paired difference. Missing metrics are not zero-imputed.', '',
            'Bootstrap: 20000 resamples, seed20260831. P values are unadjusted exploratory bootstrap values, not multiplicity-controlled confirmatory tests. Wins/ties/losses favor175 and respect metric direction. CI below is for raw175-minus125, so negative FP/FN/HD95 differences favor175.', '',
            'ET/RC/TC/WT denote the official MET regions. Dice is lesionwise_dsc_mean, not voxel/global Dice. Input is authentic T2W, not G1 synthesis. Image reconstruction, offline completion-plus-segmentation, and online augmentation-plus-segmentation conclusions must remain separate.', '',
            'Remote generation UTC: ' + datetime.now(timezone.utc).isoformat(), '']
    sections = [('Lesion Dice', 'lesionwise_dsc_mean_'), ('All-lesion F1', 'all_instance_f1_'),
                ('Large-lesion F1', 'large_instance_f1_'), ('Small-lesion F1', 'small_instance_f1_'),
                ('False Positives', 'all_instance_fp_'), ('False Negatives', 'all_instance_fn_'),
                ('Lesion HD95', 'lesionwise_hd95_mean_'), ('Lesion NSD', 'lesionwise_nsd_mean_')]
    for title, prefix in sections:
        text += ['## ' + title, '', '| Method | Region | 125 mean (n) | 175 mean (n) | Paired n | Paired delta | Raw delta 95% CI | 175 W/T/L | p (unadjusted) |',
                 '|---|---|---:|---:|---:|---:|---|---|---:|']
        for row in rows:
            if row['metric'].startswith(prefix):
                text.append(f"| {row['method']} | {row['metric'][len(prefix):].upper()} | {fmt(row['official_mean125'])} ({row['finite_n125']}) | {fmt(row['official_mean175'])} ({row['finite_n175']}) | {row['paired_n']} | {fmt(row['paired_delta175_minus125'])} | [{fmt(row['raw_delta_ci95_low'])}, {fmt(row['raw_delta_ci95_high'])}] | {row['wins175']}/{row['ties']}/{row['losses175']} | {fmt(row['unadjusted_bootstrap_p'])} |")
        text += ['']
    text += ['## Epoch175 Rankings', '', '| Metric | Ranking (best first) |', '|---|---|']
    for metric, order in cross['rankings'].items():
        if any(metric.startswith(prefix) for _, prefix in sections):
            text.append('| ' + metric + ' | ' + ' > '.join(order) + ' |')
    text += ['', 'All72 metrics with both raw epoch means, paired common-case means, denominators and statistics are in OFFICIAL_MET_125_VS175_VALUES_AND_PAIRED.csv. The three epoch175 cross-route paired results and their raw means are in epoch175_cross_route/PAIRWISE_BOOTSTRAP.csv.', '']
    with (ROOT / 'results/COMPARISON_REPORT.md').open('x') as f:
        f.write('\n'.join(text))


if __name__ == '__main__':
    main()
