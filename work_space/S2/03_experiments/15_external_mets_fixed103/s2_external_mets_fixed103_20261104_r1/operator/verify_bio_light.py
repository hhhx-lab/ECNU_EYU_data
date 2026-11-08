"""Independently check downloaded tables without MRI reads or bootstrap reruns."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('EXT2025', 'K1_175', 'original_E_200', 'EXT2026_BIO_INTEL_FULL')
META = dict(artifact_status='experimental_unvalidated', operator_approved=False,
            formal_gate_status='not_run_not_passed')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def value(cell):
    return float(cell) if cell else math.nan


def mean(values):
    finite = [v for v in values if math.isfinite(v)]
    return statistics.fmean(finite) if finite else None


def equal(cell, expected):
    assert (not cell and expected is None) or (expected is not None and
        math.isclose(float(cell), expected, rel_tol=1e-12, abs_tol=1e-12)), (cell, expected)


def verify(destination):
    assert destination.parent == ROOT / 'results'
    acceptance = json.loads((destination / 'DOWNLOAD_ACCEPTANCE.json').read_text())
    assert acceptance['job_id'] == '4321579'
    assert acceptance['status'] == 'completed_accounting_and_light_output_SHA_pass'
    assert sum(e['bytes'] for e in acceptance['files'].values()) == acceptance['total_bytes'] < 16 * 1024**2
    for name, entry in acceptance['files'].items():
        path = destination / name
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], name
    complete = json.loads((destination / 'reports/COMPLETE.json').read_text())
    assert complete['job_id'] == '4321579' and complete['models'] == list(MODELS)
    assert complete['cases'] == 103 and complete['metric_columns'] == 72
    assert complete['mean_rows'] == 288 and complete['paired_rows'] == 432
    assert complete['old_mean_and_paired_rows_unchanged'] and not complete['locked_test_read']
    payload = json.loads((ROOT / 'operator/submit_bio_comparison_20261107T080223Z/SUBMISSION_PAYLOAD.json').read_text())
    contract = payload['contract']
    assert hashlib.sha256(payload['files']['contract/COMPARISON_CONTRACT.json'].encode()).hexdigest() == complete['contract_sha256']
    groups = contract['patient_groups']
    assert len(groups) == 103 and len(set(groups.values())) == 69
    old = ROOT / 'results/comparison_light_20261106T121856Z'
    tables = {}
    fields = None
    for model in MODELS:
        parent = destination if model == MODELS[-1] else old
        path = parent / 'evaluation' / model / 'leaderboard_metrics.csv'
        assert sha(path) == complete['outputs']['evaluation/' + model + '/leaderboard_metrics.csv']['sha256']
        headers, data = rows(path)
        assert headers[0] == 'subject_id' and len(headers) == 73 and len(data) == 106
        assert [r['subject_id'] for r in data[-3:]] == ['mean', 'std', 'median']
        assert len({r['subject_id'] for r in data[:103]}) == 103
        assert {r['subject_id'] for r in data[:103]} == set(groups)
        if fields is None:
            fields = headers[1:]
        assert headers[1:] == fields
        tables[model] = {r['subject_id']: {f: value(r[f]) for f in fields} for r in data[:103]}
    _, means = rows(destination / 'reports/MEANS_288.csv')
    _, pairs = rows(destination / 'reports/PAIRED_ALL_432.csv')
    assert len(means) == 288 and len(pairs) == 432
    assert means[:216] == rows(old / 'reports/MEANS_216.csv')[1]
    assert pairs[:216] == rows(old / 'reports/PAIRED_ALL_216.csv')[1]
    assert {(r['model'], r['metric']) for r in means} == set(itertools.product(MODELS, fields))
    assert len({(r['model'], r['metric']) for r in means}) == 288
    assert {(r['first'], r['second'], r['metric']) for r in pairs} == {
        (a, b, f) for a, b in itertools.combinations(MODELS, 2) for f in fields}
    for row in means:
        values = [r[row['metric']] for r in tables[row['model']].values()]
        equal(row['mean'], mean(values))
        assert int(row['finite_cases']) == sum(math.isfinite(v) for v in values)
    for row in pairs:
        a, b, f = row['first'], row['second'], row['metric']
        ids = [n for n in sorted(groups) if math.isfinite(tables[a][n][f]) and math.isfinite(tables[b][n][f])]
        sign = 0 if '_std_' in f else -1 if any(x in f for x in ('_fp_', '_fn_', '_hd95_')) else 1
        assert row['direction'] == ('higher' if sign == 1 else 'lower' if sign == -1 else 'descriptive_std')
        delta = {n: (tables[a][n][f] - tables[b][n][f]) * (sign or 1) for n in ids}
        for key, expected in dict(first_mean=mean([r[f] for r in tables[a].values()]),
                second_mean=mean([r[f] for r in tables[b].values()]),
                first_common_mean=mean([tables[a][n][f] for n in ids]),
                second_common_mean=mean([tables[b][n][f] for n in ids]),
                raw_delta=mean([tables[a][n][f] - tables[b][n][f] for n in ids]),
                oriented_delta=mean(list(delta.values()))).items():
            equal(row[key], expected)
        assert int(row['common_finite_cases']) == len(ids)
        assert int(row['wins']) == sum(v > 1e-12 for v in delta.values())
        assert int(row['ties']) == sum(abs(v) <= 1e-12 for v in delta.values())
        assert int(row['losses']) == sum(v < -1e-12 for v in delta.values())
        labels = sorted({groups[n] for n in ids})
        assert int(row['patient_groups']) == len(labels)
        equal(row['patient_equal_delta'], mean([mean([delta[n] for n in ids if groups[n] == g]) for g in labels]))
        assert int(row['bootstrap_iterations']) == 20000 and int(row['seed']) == 20260831
        for low, high in (('ci_low', 'ci_high'), ('patient_equal_ci_low', 'patient_equal_ci_high')):
            assert (not ids and not row[low] and not row[high]) or (
                ids and math.isfinite(float(row[low])) and math.isfinite(float(row[high])) and float(row[low]) <= float(row[high]))
    absent = [n for n, r in tables[MODELS[0]].items() if r['all_instance_tp_rc'] + r['all_instance_fn_rc'] == 0]
    for model in MODELS:
        assert set(absent) == {n for n, r in tables[model].items() if r['all_instance_tp_rc'] + r['all_instance_fn_rc'] == 0}
    selected = ['lesionwise_dsc_mean_' + r for r in ('et', 'rc', 'tc', 'wt')] + ['small_instance_f1_wt', 'all_instance_fp_rc']
    return dict(META, status='local_light_SHA_and_288_means_432_pair_statistics_pass',
        reviewed_utc=datetime.now(timezone.utc).isoformat(), job_id='4321579',
        download=str(destination.relative_to(ROOT)), light_files=len(acceptance['files']), light_bytes=acceptance['total_bytes'],
        old_216_means_and_216_pairs_identical=True, independent_mean_rows=288, independent_pair_statistic_rows=432,
        bootstrap_CI_independently_reproduced=False, bootstrap_CI_structure_checked=True,
        MRI_GT_predictions_models_read=False, remote_operations=False,
        RC_absent_scans=len(absent), RC_absent_false_positive_instances={m:sum(tables[m][n]['all_instance_fp_rc'] for n in absent) for m in MODELS},
        RC_absent_scans_with_false_positives={m:sum(tables[m][n]['all_instance_fp_rc'] > 0 for n in absent) for m in MODELS},
        selected_means=[r for r in means if r['metric'] in selected],
        selected_WT_pairs=[r for r in pairs if r['metric'] == 'lesionwise_dsc_mean_wt'],
        evidence_SHA256={str(p.relative_to(ROOT)):sha(p) for p in [destination / 'DOWNLOAD_ACCEPTANCE.json', destination / 'reports/COMPLETE.json',
            destination / 'reports/MEANS_288.csv', destination / 'reports/PAIRED_ALL_432.csv', Path(__file__)]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.results.resolve())
    with args.output.open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
