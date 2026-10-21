"""Read all 926 train/val sources once and collect every mask anomaly, no images written."""
import collections
import concurrent.futures
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import traceback

import nibabel as nib
import numpy as np

from support_audit import coverage, VERSION

ROOT = Path(__file__).resolve().parents[1]
META = dict(artifact_status='experimental_unvalidated', operator_approved=False,
            formal_gate_status='not_run_not_passed')
MODS = ('t1n', 't1c', 't2f')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o440), 'w') as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def geometry(image, ref):
    if image.shape != ref.shape:
        raise ValueError('Shape mismatch')
    corners = np.array([[i, j, k, 1] for i in (0, ref.shape[0]-1)
                        for j in (0, ref.shape[1]-1) for k in (0, ref.shape[2]-1)])
    shift = float(np.linalg.norm((image.affine @ corners.T - ref.affine @ corners.T)[:3], axis=0).max())
    if not np.isfinite(image.affine).all() or shift > .1:
        raise ValueError('Affine corner shift exceeds 0.1mm: '+str(shift))
    return {'shape': list(image.shape), 'affine': image.affine.tolist(),
            'spacing': list(map(float, image.header.get_zooms()[:3])), 'corner_shift_mm': shift}


def process(job):
    cfg, member, row, expected_cm, overlay = job
    case = member['source_case_id']
    result = {**META, 'case': case, 'split': member['split'], 'patient_group': member['patient_group'],
              'status': 'identity_or_data_error', 'finished_utc': None}
    try:
        assert member['split'] == row['split'] and member['split'] in ('train', 'val')
        native = Path(cfg['recon']) / 'dataset/cases' / case
        cm_path = native / 'case_manifest.json'
        assert sha(cm_path) == expected_cm
        cm = json.loads(cm_path.read_text())
        assert cm['case_id'] == case and cm['split'] == member['split'] and cm['geometry'] == 'native'
        assert cm['checkpoint_sha256'] == cfg['checkpoint_sha256']
        native_seg = native / (case+'-seg.nii.gz')
        ref = nib.load(str(native_seg))
        assert list(ref.shape) == cm['native_shape']
        label = ref.get_fdata(dtype=np.float32)
        source_root = Path(cfg['source_code']) / 'data/input' / case
        paths = {k: source_root / Path(row[k]).name for k in (*MODS, 'seg')}
        details = {}
        arrays = []
        for k, path in paths.items():
            digest = sha(path)
            assert digest == cm['source_sha256'][k], 'Source SHA drift: '+k
            image = nib.load(str(path))
            g = geometry(image, ref)
            a = image.get_fdata(dtype=np.float32)
            assert np.isfinite(a).all(), 'Nonfinite source: '+k
            details[k] = {'path': str(path), 'sha256': digest, 'geometry': g,
                          'minimum': float(a.min()), 'maximum': float(a.max()),
                          'negative_voxels': int(np.count_nonzero(a < 0))}
            if k == 'seg':
                assert np.array_equal(a, label)
                assert digest == sha(native_seg) == cm['files']['seg']['sha256']
            else:
                arrays.append(a)
        if case in overlay:
            assert sha(overlay[case]) == sha(native_seg), 'Corrected overlay drift'
        result.update(case_manifest_sha256=expected_cm, label_sha256=sha(native_seg),
                      corrected_overlay=overlay.get(case), sources=details)
        counts, old_missing, missing = coverage(label, arrays)
        result['coverage'] = counts
        result['status'] = ('coverage_anomaly' if counts['positive_missing_voxels'] or
                            not counts['positive_support_nonempty'] else 'strict_coverage_pass')
        if old_missing.any() or missing.any():
            anomaly = old_missing | missing
            indices = np.argwhere(anomaly)
            chosen = indices[np.unique(np.linspace(0, len(indices)-1, min(24,len(indices)), dtype=int))]
            result['anomaly_bbox_ijk'] = [indices.min(0).tolist(), indices.max(0).tolist()]
            result['anomaly_samples'] = [{'ijk': p.tolist(), 'label': int(label[tuple(p)]),
                'xyz_mm': (ref.affine @ np.r_[p,1])[:3].tolist(),
                'values': {k:float(a[tuple(p)]) for k,a in zip(MODS,arrays)}} for p in chosen]
            result['source_values_at_anomaly'] = {k:{'min':float(a[anomaly].min()),
                'max':float(a[anomaly].max()), 'negative':int(np.count_nonzero(a[anomaly]<0)),
                'zero':int(np.count_nonzero(a[anomaly]==0))} for k,a in zip(MODS,arrays)}
        result['finished_utc'] = now()
        return result
    except Exception as error:
        result.update(message=str(error), traceback=traceback.format_exc(), finished_utc=now())
        return result


def main():
    cfg = json.loads((ROOT/'config/SCAN.json').read_text())
    evidence = ROOT/'scan'
    evidence.mkdir(exist_ok=False)
    write(evidence/'STARTED.json', {**META, 'utc': now(), 'job_id':os.environ.get('SLURM_JOB_ID')})
    try:
        package = json.loads((ROOT/'plan/PACKAGE_MANIFEST.json').read_text())
        assert all(sha(ROOT/p)==v for p,v in package['files'].items()), 'Scan package drift'
        parent = Path(cfg['parent_r5'])
        pp = json.loads((parent/'plan/PACKAGE_MANIFEST.json').read_text())
        assert sha(parent/'plan/PACKAGE_MANIFEST.json') == cfg['parent_package_sha256']
        assert all(sha(parent/p)==v for p,v in pp['files'].items()), 'Frozen R5 drift'
        source_csv = Path(cfg['source_code'])/'data/data_csv.csv'
        assert sha(source_csv) == cfg['source_csv_sha256']
        with source_csv.open(newline='') as f:
            source = {r['id']: r for r in csv.DictReader(f)}
        with (ROOT/'plan/membership.csv').open(newline='') as f:
            all_members = list(csv.DictReader(f))
        assert collections.Counter(r['split'] for r in all_members)=={'train':823,'val':103,'test':104}
        assert len({r['source_case_id'] for r in all_members})==1030
        groups = collections.defaultdict(set)
        for r in all_members:
            groups[r['patient_group']].add(r['split'])
        assert all(len(g)==1 for g in groups.values())
        members = [r for r in all_members if r['split']!='test']
        complete = json.loads((Path(cfg['recon'])/'dataset/COMPLETE.ok').read_text())
        assert complete['status']=='pass' and complete['checkpoint_sha256']==cfg['checkpoint_sha256']
        overlay = json.loads((parent/'config/P0_SOURCE_MAPPING.json').read_text())['corrected_label_paths']
        jobs = [(cfg,r,source[r['source_case_id']],complete['manifest_sha256'][r['source_case_id']],overlay) for r in members]
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
            for result in pool.map(process, jobs, chunksize=1):
                write(evidence/'cases'/(result['case']+'.json'), result)
                results.append(result)
                print(json.dumps({'case':result['case'],'status':result['status'],'cases_scanned':len(results),
                                  'coverage':result.get('coverage')}), flush=True)
        assert len(results)==926
        errors=[r['case'] for r in results if r['status']=='identity_or_data_error']
        anomalies=[r['case'] for r in results if r['status']=='coverage_anomaly']
        nonzero_anomalies=[r['case'] for r in results if r.get('coverage',{}).get('nonzero_missing_voxels',0)>0]
        fieldnames=['case','split','status','label_voxels','positive_missing_voxels','nonzero_missing_voxels',
                    'old_missing_with_negative_source','old_missing_all_three_zero','label_all_three_zero']
        with (evidence/'coverage.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fieldnames);writer.writeheader()
            for r in results:
                writer.writerow({k:({**r,**r.get('coverage',{})}).get(k) for k in fieldnames})
            f.flush();os.fsync(f.fileno())
        total={**META,'status':'complete','scope':'diagnostic_only_no_training_authorization',
               'cases':926,'split_counts':{'train':823,'val':103},'test_images_read':0,
               'identity_error_cases':errors,'strict_anomaly_cases':anomalies,'nonzero_anomaly_cases':nonzero_anomalies,
               'STRICT_ORIGINAL_GATE':not errors and not anomalies,
               'NONZERO_COMPARISON_GATE':not errors and not nonzero_anomalies and
                    all(r['coverage']['nonzero_support_nonempty'] for r in results),
               'algorithm_version':VERSION,'label_repair_performed':False,'mask_repair_performed':False,
               'case_report_sha256':{r['case']:sha(evidence/'cases'/(r['case']+'.json')) for r in results},
               'csv_sha256':sha(evidence/'coverage.csv'),'parent_package_sha256':cfg['parent_package_sha256'],
               'package_sha256':sha(ROOT/'plan/PACKAGE_MANIFEST.json'),'finished_utc':now()}
        write(evidence/'VALIDATION.json',total);write(evidence/'COMPLETE.ok',total)
        print(json.dumps({k:v for k,v in total.items() if k!='case_report_sha256'}),flush=True)
    except BaseException as error:
        write(evidence/'FAILED.json',{**META,'utc':now(),'message':str(error),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':
    main()
