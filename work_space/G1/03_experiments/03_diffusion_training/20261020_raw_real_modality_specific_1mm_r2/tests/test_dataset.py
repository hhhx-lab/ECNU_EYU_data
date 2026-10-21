"""Temporary synthetic data only; no project MRI, remote access or dependency installs."""
import copy
import csv
import hashlib
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

RUNTIME = Path(__file__).resolve().parents[1] / 'runtime'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import nibabel
            import numpy
            import scipy
        except ImportError as error:
            raise unittest.SkipTest('Existing scientific Conda environment required: ' + str(error))
        cls.nib, cls.np = nibabel, numpy
        with patch.dict(sys.modules):
            cls.common = load_module('common', RUNTIME / 'common.py')
            cls.contracts = load_module('contracts', RUNTIME / 'contracts.py')
            cls.identity = load_module('identity', RUNTIME / 'identity.py')
            cls.prepare = load_module('prepare', RUNTIME / 'prepare.py')

    def setUp(self):
        self.contexts = ExitStack()
        self.addCleanup(self.contexts.close)
        self.temporary = tempfile.TemporaryDirectory(prefix='raw-real-dataset-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        for name in ('config', 'plan', 'dataset'):
            (self.root / name).mkdir()
        for module in (self.common, self.contracts, self.identity, self.prepare):
            self.contexts.enter_context(patch.object(module, 'ROOT', self.root))
        self.rows = [dict(source_case_id='BraTS-MET-00001-000', patient_group='BraTS-MET-00001',
                          split='train', t2w_status='authentic', eligible_for_realonly='True'),
                     dict(source_case_id='BraTS-MET-00002-000', patient_group='BraTS-MET-00002',
                          split='train', t2w_status='fake_or_broken', eligible_for_realonly='False'),
                     dict(source_case_id='BraTS-MET-00003-000', patient_group='BraTS-MET-00003',
                          split='val', t2w_status='authentic', eligible_for_realonly='True')]
        counts = {m: dict(train=1, val=1) if m == 't2w' else dict(train=2, val=1)
                  for m in self.contracts.MODS}
        def synthetic_members(modality=None):
            return [dict(r) for r in self.rows if modality != 't2w' or r['t2w_status'] == 'authentic']
        for module in (self.contracts, self.identity, self.prepare):
            self.contexts.enter_context(patch.object(module, 'COUNTS', counts))
            self.contexts.enter_context(patch.object(module, 'members', synthetic_members))
        self.records = {}
        archived_rows = []
        for member in self.rows:
            case = member['source_case_id']
            entry = {k: member[k] for k in ('split', 'patient_group', 't2w_status')}
            entry['files'] = {}
            directory = self.root / 'data' / case
            directory.mkdir(parents=True)
            shape = (7, 9, 11)
            affine = self.np.eye(4)
            for mod in (*self.contracts.available_mods(member), 'seg'):
                path = directory / (mod + '.nii.gz')
                if mod == 'seg':
                    array = self.np.zeros(shape, dtype=self.np.uint8)
                    array[2:4, 3:5, 4:6] = 4
                else:
                    array = self.np.arange(self.np.prod(shape), dtype=self.np.float32).reshape(shape) + 1
                self.nib.save(self.nib.Nifti1Image(array, affine), path)
                entry['files'][mod] = dict(path=str(path), resolved_path=str(path),
                                           sha256=self.common.sha(path), size=path.stat().st_size,
                                           shape=list(shape), spacing=[1., 1., 1.], affine=affine.tolist())
                digest = hashlib.sha256(affine.astype(self.np.float64).tobytes()).hexdigest()[:16]
                entry['files'][mod].update(archived_affine_hash=digest, observed_affine_hash=digest,
                                          observed_legacy_signed_zero_affine_hash=digest)
            archived = dict(case_id=case)
            for mod, file in entry['files'].items():
                archived.update({'affine_hash_' + mod: file['archived_affine_hash'],
                                 'shape_' + mod: 'x'.join(map(str, file['shape'])),
                                 'spacing_' + mod: ','.join(map(str, file['spacing']))})
            archived_rows.append(archived)
            if member['t2w_status'] == 'authentic':
                history_dir = self.root / 'history' / case
                history_dir.mkdir(parents=True)
                manifest_path = history_dir / 'case_manifest.json'
                self.common.write(manifest_path, dict(case_id=case, split=member['split'], geometry='native',
                                  source_sha256={m: f['sha256'] for m, f in entry['files'].items()},
                                  files={'t2w': {'path': '/forbidden/reconstructed-image.nii.gz'}}))
                entry['historical_authentic_source_manifest'] = dict(path=str(manifest_path),
                                                                     sha256=self.common.sha(manifest_path))
            self.records[case] = entry
        for name in ('raw_manifest.csv', 'ucsd_manifest.csv'):
            with (self.root / 'plan' / name).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(archived_rows[0]))
                writer.writeheader()
                writer.writerows(archived_rows)
        self.common.write(self.root / 'config/SOURCE_RECORDS.json',
                          dict(source_kind=self.contracts.SOURCE_KIND, status='pass',
                               allowed_roots=[str(self.root / 'data')], cases=self.records))
        self.source_sha = self.common.sha(self.root / 'config/SOURCE_RECORDS.json')
        self.cfg = dict(source_kind=self.contracts.SOURCE_KIND, source_records_sha256=self.source_sha,
                        preprocess_workers=1, seed=42,
                        training=dict(use_all_rows=False, normalization='nnunet', crop_size=64))
        self.common.write(self.root / 'config/RUN.json', self.cfg)
        self.common.write(self.root / 'plan/PACKAGE_MANIFEST.json', dict(files={}))
        self.package_sha = self.common.sha(self.root / 'plan/PACKAGE_MANIFEST.json')
        for name in ('master_membership.csv', 'realonly_membership.csv'):
            (self.root / 'plan' / name).write_text('synthetic membership metadata\n')

    def jobs(self):
        return [(self.cfg, member, self.records[member['source_case_id']], self.source_sha)
                for member in self.rows]

    def build(self):
        results = [self.prepare.process(job) for job in self.jobs()]
        return self.prepare.finish_dataset(results, [], self.source_sha, self.package_sha)

    def test_all_modalities_identity_and_original_geometry(self):
        complete = self.build()
        self.assertEqual(complete['cases'], 3)
        self.assertEqual(complete['csv_rows']['t1c'], dict(total=3, train=2, val=1))
        self.assertEqual(complete['csv_rows']['t2w'], dict(total=2, train=1, val=1))
        for modality in self.contracts.MODS:
            identity = self.identity.data_identity(modality)
            self.assertEqual(identity['manual_review'], 'not_run')
            self.assertFalse(identity['operator_approved'])
            self.assertTrue(all('/forbidden/' not in path for path in identity['consumed']))
        case = self.rows[0]['source_case_id']
        image = self.nib.load(self.root / 'dataset/cases' / case / (case + '-t1c.nii.gz'))
        self.assertEqual(image.shape, (7, 9, 11))
        self.np.testing.assert_allclose(image.affine, self.np.eye(4))
        index = json.loads((self.root / 'dataset/indices' / (case + '.json')).read_text())
        self.assertEqual(index['rows'][0]['x_extreme_min'], 2)
        self.assertEqual(index['rows'][0]['z_extreme_min'], 4)

    def test_missing_t2w_is_never_opened_in_estimate_process_or_identity(self):
        missing = self.rows[1]['source_case_id']
        poison = self.root / 'data' / missing / 't2w.nii.gz'
        poison.write_bytes(b'forbidden T2W placeholder')
        actual_load = self.prepare.nib.load
        actual_sha = self.common.sha
        def guarded_load(path, *args, **kwargs):
            self.assertNotEqual(Path(path), poison)
            return actual_load(path, *args, **kwargs)
        def guarded_sha(path):
            self.assertNotEqual(Path(path), poison)
            return actual_sha(path)
        with patch.object(self.prepare.nib, 'load', guarded_load), \
                patch.object(self.identity, 'sha', guarded_sha), \
                patch.object(self.prepare, 'sha', guarded_sha), \
                patch.object(self.common, 'sha', guarded_sha):
            estimate = self.prepare.estimate(self.cfg)
            self.assertFalse(estimate['errors'])
            self.build()
            self.identity.data_identity('t1c')
            self.identity.data_identity('t2w')
        missing_dir = self.root / 'dataset/cases' / missing
        self.assertFalse((missing_dir / (missing + '-t2w.nii.gz')).exists())

    def test_estimate_is_header_only(self):
        with patch.object(self.prepare.nib.Nifti1Image, 'get_fdata', side_effect=AssertionError('voxel read')), \
                patch.object(self.prepare, 'sha', side_effect=AssertionError('source SHA sweep')):
            estimate = self.prepare.estimate(self.cfg)
        self.assertFalse(estimate['errors'])
        self.assertEqual(estimate['cases'], 3)
        self.assertGreater(estimate['required_bytes'], estimate['data_upper_bytes'])

    def test_forbidden_missing_modality_and_test_rejected_before_io(self):
        job = list(self.jobs()[1])
        job[2] = copy.deepcopy(job[2])
        job[2]['files']['t2w'] = dict(path='/forbidden/t2w.nii.gz')
        with patch.object(self.prepare, 'verify_source_file', side_effect=AssertionError('unexpected I/O')):
            self.assertFalse(self.prepare.process_record(tuple(job))['ok'])
            job[2] = self.records[self.rows[1]['source_case_id']]
            job[1] = {**job[1], 'split': 'test'}
            self.assertFalse(self.prepare.process_record(tuple(job))['ok'])

    def test_source_output_marker_csv_and_history_drift_fail_closed(self):
        self.build()
        case = self.rows[0]['source_case_id']
        paths = [Path(self.records[case]['files']['t1c']['path']),
                 self.root / 'dataset/cases' / case / (case + '-t1c.nii.gz'),
                 self.root / 'dataset/cases' / case / 'preprocess_complete.json',
                 self.root / 'dataset/lesions_t1c.csv',
                 Path(self.records[case]['historical_authentic_source_manifest']['path']),
                 self.root / 'dataset/indices' / (case + '.json'),
                 self.root / 'config/SOURCE_RECORDS.json']
        for path in paths:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                changed = bytearray(original)
                changed[-1] ^= 1
                path.write_bytes(changed)
                try:
                    with self.assertRaises((ValueError, AssertionError, UnicodeError, json.JSONDecodeError)):
                        self.identity.data_identity('t1c')
                finally:
                    path.write_bytes(original)

    def test_source_realpath_redirect_detected(self):
        self.build()
        file = self.records[self.rows[0]['source_case_id']]['files']['t1c']
        path = Path(file['path'])
        redirected = path.with_name('redirected.nii.gz')
        path.rename(redirected)
        path.symlink_to(redirected)
        with self.assertRaisesRegex(ValueError, 'realpath drift'):
            self.identity.data_identity('t1c')

    def test_quantization_warning_continues_without_label_patching(self):
        original_resample = self.prepare.resample_physical
        def lose_rc(data, *args, **kwargs):
            result = original_resample(data, *args, **kwargs)
            if kwargs.get('is_seg') and self.np.any(data == 4):
                result = result.copy()
                result[result == 4] = 0
            return result
        with patch.object(self.prepare, 'resample_physical', lose_rc):
            result = self.prepare.process(self.jobs()[1])
        self.assertEqual(result['lost_classes'], ['4'])
        self.assertEqual(result['rows'], [])
        marker = json.loads((Path(result['directory']) / 'preprocess_complete.json').read_text())
        self.assertTrue(marker['contract']['resolution_warning'])
        self.assertIsNone(marker['contract']['local_label_exception'])
        self.assertEqual(marker['status'], 'complete')

    def test_warning_report_drift_even_when_rehashed(self):
        self.build()
        path = self.root / 'dataset/PREPROCESS_REPORT.json'
        report = json.loads(path.read_text())
        report['warnings'] = [dict(case=self.rows[0]['source_case_id'], lost_classes=['4'])]
        self.common.write(path, report, replace=True)
        complete_path = self.root / 'dataset/COMPLETE.ok'
        complete = json.loads(complete_path.read_text())
        complete['report_sha256'] = self.common.sha(path)
        self.common.write(complete_path, complete, replace=True)
        with self.assertRaisesRegex(ValueError, 'Warning report'):
            self.identity.data_identity('t1c')

    def test_per_modality_crop_counts_are_recomputed(self):
        rows = [dict(patient_id='same', t2w_valid=1), dict(patient_id='same', t2w_valid=0)]
        self.assertEqual(self.identity.modality_rows(rows, 't1c')[0]['patient_n_crops'], 2)
        self.assertEqual(self.identity.modality_rows(rows, 't2w')[0]['patient_n_crops'], 1)
        self.assertNotIn('patient_n_crops', rows[0])

    def test_recorded_affine_legacy_signed_zero_and_drift(self):
        entry = copy.deepcopy(self.records[self.rows[1]['source_case_id']])
        file = entry['files']['t1c']
        file['affine'][0][3] = -0.0
        affine = self.np.asarray(file['affine'], dtype=self.np.float64)
        legacy = hashlib.sha256(affine.tobytes()).hexdigest()[:16]
        file['archived_affine_hash'] = file['observed_legacy_signed_zero_affine_hash'] = legacy
        self.identity.verify_recorded_affines(entry)
        file['affine'][0][3] = 2.
        with self.assertRaisesRegex(ValueError, 'legacy affine'):
            self.identity.verify_recorded_affines(entry)

    def test_cpu_error_collection_writes_failed_without_complete(self):
        file = self.records[self.rows[0]['source_case_id']]['files']['t1c']
        path = Path(file['path'])
        changed = bytearray(path.read_bytes())
        changed[-1] ^= 1
        path.write_bytes(changed)
        records = [self.prepare.process_record(job) for job in self.jobs()]
        self.assertEqual(sum(not r['ok'] for r in records), 1)
        # Exercise the production failure handler without spawning workers or source header sweeps.
        class Pool:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def map(self, *args, **kwargs):
                return iter(records)
        # The successful records above wrote cases; isolate only main's dataset directory.
        import shutil
        shutil.rmtree(self.root / 'dataset', ignore_errors=True)
        with patch.object(self.prepare, 'estimate', return_value=dict(errors=[], capacity_pass=True)), \
                patch.object(self.prepare.concurrent.futures, 'ProcessPoolExecutor', Pool):
            with self.assertRaises(ValueError):
                self.prepare.main()
        self.assertTrue((self.root / 'dataset/FAILED.json').exists())
        self.assertFalse((self.root / 'dataset/COMPLETE.ok').exists())
        report = json.loads((self.root / 'dataset/PREPROCESS_REPORT.json').read_text())
        self.assertEqual(report['cases_processed'], 3)
        self.assertEqual(len(report['errors']), 1)


if __name__ == '__main__':
    unittest.main()
