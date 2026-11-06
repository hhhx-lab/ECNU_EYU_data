"""Read only local fixed103 original-E predictions and labels; no raw MRI or test."""
import csv
import json
from pathlib import Path
from session import access_module

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[4]
K1 = PROJECT / 'work_space/S2/03_experiments/12_k1245_rc_aware/s2_met_aug_k1_selected103_official_met_20261008_r1'
E = PROJECT / 'work_space/S2/07_submissions/original_e_20260727_20260728/s2_official_validation_inference_20260727_r1/selection_evidence/E'


def main():
    import numpy as np
    import nibabel as nib
    stamp = access_module.datetime.now(access_module.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'operator' / ('audit_local_comparison_' + stamp)
    out.mkdir()
    csv_path = K1 / 'manifests/FIXED103_REAL_T2W_CASES.csv'
    ref_path = K1 / 'manifests/INPUT_REFERENCE_SHA256.json'
    assert access_module.sha(csv_path) == '86ba853dadc64ffc45357895ba0bd543bf5b6b14c0c83eb789f1e875d4aa2e7e'
    assert access_module.sha(ref_path) == '5f52bfdc3a957d848063850460226a7c3d9c25f0926e827f728f21e525fbd9df'
    rows = list(csv.DictReader(csv_path.open()))
    names = {r['source_case_id'] + '.nii.gz' for r in rows}
    hashes = json.loads(ref_path.read_text())['files']
    expected = {Path(p).name: h for p, h in hashes.items() if '/reference/' in p}
    assert len(names) == len(expected) == 103 and set(expected) == names
    provenance = json.loads((E / 'preparation_summary.json').read_text())
    assert provenance['checkpoint_sha256'] == '4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267'
    assert provenance['case_count'] == provenance['prediction_count'] == 103
    assert access_module.sha(E / 'leaderboard_metrics.csv') == '294a1451a8737344a929902c8ef118c89bd9a3da971952d11b453fafccc3e167'
    local_ref = PROJECT / 'work_space/S2/02_datasets/undated_realonly_fixed103_reference_labels'
    assert {p.name for p in (E / 'prediction').glob('*.nii.gz')} == names
    assert {p.name for p in (E / 'reference').glob('*.nii.gz')} == names
    assert {p.name for p in local_ref.glob('*.nii.gz')} == names
    files = {}; issues = []
    for name in sorted(names):
        record = {}
        arrays = {}; images = {}
        for role, path in [('original_E', E / 'prediction' / name), ('E_reference', E / 'reference' / name), ('current_local_reference', local_ref / name)]:
            image = nib.load(str(path)); data = np.asanyarray(image.dataobj)
            assert data.ndim == 3 and np.isfinite(data).all() and set(np.unique(data)) <= {0,1,2,3,4}, role + ' invalid labels: ' + name
            record[role] = dict(path=str(path), bytes=path.stat().st_size, sha256=access_module.sha(path), shape=list(image.shape), affine=image.affine.tolist(), spacing=list(map(float,image.header.get_zooms())))
            arrays[role] = data; images[role] = image
        record['current_reference_exact_manifest_sha'] = record['current_local_reference']['sha256'] == expected[name]
        record['E_reference_exact_manifest_sha'] = record['E_reference']['sha256'] == expected[name]
        record['E_reference_same_voxels'] = np.array_equal(arrays['E_reference'], arrays['current_local_reference'])
        record['E_reference_same_geometry'] = images['E_reference'].shape == images['current_local_reference'].shape and np.allclose(images['E_reference'].affine, images['current_local_reference'].affine, rtol=0, atol=1e-5)
        record['E_prediction_native_geometry'] = images['original_E'].shape == images['current_local_reference'].shape and np.allclose(images['original_E'].affine, images['current_local_reference'].affine, rtol=0, atol=1e-5)
        if not all(record[k] for k in ('current_reference_exact_manifest_sha', 'E_reference_same_voxels', 'E_reference_same_geometry', 'E_prediction_native_geometry')): issues.append(name)
        files[name] = record
    result = dict(access_module.META, status='fixed103_local_identity_pass' if not issues else 'comparison_identity_hold', cases=103,
        original_E_provenance=provenance, original_E_checkpoint_not_loaded=True, files=files, issues=issues,
        local_utc=access_module.datetime.now(access_module.timezone.utc).isoformat(), scoring_run=False,
        raw_MRI_read=False, locked_test_read=False)
    access_module.exclusive(out / 'RESULT.json', result)
    print(json.dumps(dict(audit=str(out),status=result['status'],issues=len(issues))))


if __name__ == '__main__': main()
