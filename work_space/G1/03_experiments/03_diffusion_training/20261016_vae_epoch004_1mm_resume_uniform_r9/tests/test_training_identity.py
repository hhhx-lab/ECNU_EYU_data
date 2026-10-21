"""Full-cohort identity tests use memory only; optimizer/checkpoint checks run on CPU."""
import copy
import csv
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
spec = importlib.util.spec_from_file_location('r9_identity_training', ROOT / 'runtime/train.py')
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


class MemoryFiles:
    def __init__(self):
        self.files, self.links, self.directories = {}, {}, set()

    def put(self, path, value):
        path = Path(path)
        self.files[path] = value.encode() if isinstance(value, str) else value
        self.directories.update(path.parents)

    def json(self, path, value):
        self.put(path, json.dumps(value, sort_keys=True))

    def read_json(self, path):
        return json.loads(self.files[Path(path)])

    def csv(self, path, rows):
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        self.put(path, stream.getvalue())

    def resolve(self, path):
        path = Path(path)
        for prefix in [*reversed(path.parents), path]:
            if prefix in self.links:
                return self.resolve(self.links[prefix] / path.relative_to(prefix))
        return path

    def digest(self, path):
        return hashlib.sha256(self.files[self.resolve(path)]).hexdigest()

    def open(self, path, mode='r', **kwargs):
        if mode not in ('r', 'rb'):
            raise AssertionError('Identity verification must not write files')
        data = self.files[self.resolve(path)]
        return io.BytesIO(data) if 'b' in mode else io.StringIO(data.decode(), newline=kwargs.get('newline'))

    def patched(self):
        stack = ExitStack()
        stack.enter_context(patch.object(training, 'ROOT', Path('/r9')))
        stack.enter_context(patch.object(Path, 'resolve', lambda p, **kw: self.resolve(p)))
        stack.enter_context(patch.object(Path, 'open', lambda p, mode='r', **kw: self.open(p, mode, **kw)))
        stack.enter_context(patch.object(Path, 'read_text', lambda p, **kw: self.files[self.resolve(p)].decode()))
        stack.enter_context(patch.object(Path, 'exists', lambda p: self.resolve(p) in self.files or self.resolve(p) in self.directories))
        stack.enter_context(patch.object(Path, 'is_file', lambda p: self.resolve(p) in self.files))
        stack.enter_context(patch.object(Path, 'is_dir', lambda p: self.resolve(p) in self.directories))
        stack.enter_context(patch.object(Path, 'is_symlink', lambda p: p in self.links))
        return stack


def cohort():
    fs = MemoryFiles()
    root = Path('/r9')
    parents = {}
    for name in ('r7', 'r8'):
        parent = Path('/' + name)
        fs.put(parent / 'runtime/upstream_source/model.py', 'frozen ' + name)
        fs.put(parent / 'runtime/train.py', 'frozen runtime ' + name)
        fs.json(parent / 'plan/UPSTREAM_SOURCE_MANIFEST.json', {
            'files': {'model.py': fs.digest(parent / 'runtime/upstream_source/model.py')}})
        fs.json(parent / 'plan/PACKAGE_MANIFEST.json', {'files': {
            'runtime/train.py': fs.digest(parent / 'runtime/train.py'),
            'plan/UPSTREAM_SOURCE_MANIFEST.json': fs.digest(parent / 'plan/UPSTREAM_SOURCE_MANIFEST.json')}})
        parents[str(parent)] = fs.digest(parent / 'plan/PACKAGE_MANIFEST.json')
    fs.json(root / 'plan/PACKAGE_MANIFEST.json', {'files': {}})
    membership, source, csv_rows, details, candidates = [], [], [], [], []
    markers, locations, native_markers, warnings = {}, {}, {}, []
    for i in range(1030):
        case = f'BraTS-MET-{i:05d}-000'
        split = 'train' if i < 823 else 'val' if i < 926 else 'test'
        member = dict(source_case_id=case, patient_group=case[:-4], split=split,
                      nnunet_case_id=f'BraTSMET_{i:06d}', t2w_status='authentic', eligible_for_realonly='True')
        membership.append(member)
        source.append(dict(id=case, split=split))
        if split == 'test':
            continue  # No held-out image or native marker is available to the identity reader.
        directory = root / 'dataset/cases' / case
        native_path = Path('/recon/dataset/cases') / case / 'case_manifest.json'
        inputs = {m: hashlib.sha256((case + m).encode()).hexdigest() for m in (*training.MODALITIES, 'seg')}
        fs.json(native_path, dict(case_id=case, split=split, geometry='native', native_shape=[8, 8, 8],
                checkpoint_sha256=training.CHECKPOINT_SHA256,
                files={m: {'sha256': h} for m, h in inputs.items()}, source_sha256=inputs))
        native_markers[case] = fs.digest(native_path)
        counts = {str(k): {'voxels': 19 if k == 4 else 1} for k in range(5)}
        target_counts = copy.deepcopy(counts)
        lost = ['4'] if i in (2, 825) else []
        if lost:
            target_counts['4']['voxels'] = 0
            warnings.append(dict(case=case, lost_classes=lost))
        contract = dict(loss_policy=training.LOSS_POLICY, label_resampling_algorithm=training.LABEL_ALGORITHM,
                        support_contract=training.SUPPORT_CONTRACT, grid_version='physical_spacing_v2',
                        target_spacing=[1., 1., 1.], original_shape=[8, 8, 8], bbox=[[0, 8]] * 3,
                        normalization='whole_case_common_mask_zscore_before_resampling_zero_outside_support',
                        mask_algorithm='authentic_T1N_T1C_T2F_robust_0_100_nonzero_union_holefill',
                        local_label_exception=None, label_native=counts, label_target_1mm=target_counts,
                        lost_classes=lost, resolution_warning=bool(lost))
        outputs, references, parent_outputs = {}, {}, {}
        parent = Path('/r7' if i == 0 else '/r8')
        for m in (*training.MODALITIES, 'seg', 'brainmask'):
            name = case + '-' + m + '.nii.gz'
            contents = (case + ':' + m).encode()
            if i < 3:
                target = parent / 'dataset/cases' / case / name
                fs.put(target, contents)
                parent_outputs[name] = fs.digest(target)
                if i != 2 or m != 'seg':
                    fs.links[directory / name] = target
                    fs.directories.update((directory / name).parents)
                    references[name] = dict(path=str(target), sha256=fs.digest(target))
                    outputs[name] = fs.digest(target)
                    continue
                contents = b'original resampling, not the old patched label'
            fs.put(directory / name, contents)
            outputs[name] = fs.digest(directory / name)
        info = dict(status='complete', case_id=case, split=split, patient_group=member['patient_group'],
                    contract=contract, outputs=outputs, input_sha256=inputs,
                    authentic_mask_source_sha256={m: inputs[m] for m in ('t1c', 't1n', 't2f')},
                    native_case_manifest_sha256=native_markers[case])
        if references:
            parent_directory = parent / 'dataset/cases' / case
            fs.json(parent_directory / 'preprocess_complete.json', dict(case_id=case, split=split,
                    native_case_manifest_sha256=native_markers[case], outputs=parent_outputs))
            parent_marker_sha = fs.digest(parent_directory / 'preprocess_complete.json')
            candidates.append(dict(case_id=case, directory=str(parent_directory), marker_sha256=parent_marker_sha,
                              mode='label_only_rebuild' if i == 2 else 'all_outputs_reference'))
            info.update(file_references=references, parent_directory=str(parent_directory), parent_marker_sha256=parent_marker_sha)
        fs.json(directory / 'spatial_mapping.json', contract)
        fs.json(directory / 'preprocess_complete.json', info)
        marker_sha = fs.digest(directory / 'preprocess_complete.json')
        markers[case], locations[case] = marker_sha, str(directory)
        row = dict(patient_id=case[-9:], source_case_id=case, patient_group=member['patient_group'], split=split,
                   lesion_id=case + '_cc0_t0', n_voxels=1, t2w_valid=1, label=str(directory / (case + '-seg.nii.gz')))
        row.update({'scan_' + m: str(directory / (case + '-' + m + '.nii.gz')) for m in training.MODALITIES})
        row.update({f'{ax}_{key}': value for ax in 'xyz' for key, value in
                    [('extreme_min', 0), ('extreme_max', 1), ('size', 1)]})
        row.update({'center_' + ax: 0 for ax in 'xyz'})
        index_rows = [] if i == 825 else [row]
        fs.json(root / 'dataset/indices' / (case + '.json'), dict(case_id=case, rows=index_rows, marker_sha256=marker_sha))
        csv_rows.extend(dict(r, patient_n_crops=1) for r in index_rows)
        details.append(dict(case=case, split=split, resolved_directory=str(directory), marker_sha256=marker_sha,
                            outputs=outputs, file_references=references, lost_classes=lost))
    fs.csv(root / 'plan/splits_final_train_val_test_membership.csv', membership)
    fs.csv('/source/data/data_csv.csv', source)
    fs.csv(root / 'dataset/lesions.csv', csv_rows)
    fs.json(root / 'config/RUN.json', dict(reuse_parent='/r8', reuse_parent_package_sha256=parents['/r8'],
            source_code='/source', recon='/recon', source_csv_sha256=fs.digest('/source/data/data_csv.csv'),
            loss_policy=training.LOSS_POLICY, label_resampling_algorithm=training.LABEL_ALGORITHM))
    fs.json(root / 'plan/REUSE_CANDIDATES.json', dict(parents=parents, candidates=candidates))
    fs.json('/recon/dataset/COMPLETE.ok', dict(status='pass', checkpoint_sha256=training.CHECKPOINT_SHA256,
                                           manifest_sha256=native_markers))
    fs.json(root / 'dataset/PREPROCESS_REPORT.json', dict(status='complete', errors=[], cases_ok=926,
            cases_processed=926, warnings=warnings, loss_policy=training.LOSS_POLICY,
            label_resampling_algorithm=training.LABEL_ALGORITHM))
    fs.json(root / 'dataset/COMPLETE.ok', dict(status='complete', cases=926, split_counts={'train': 823, 'val': 103},
            support_contract=training.SUPPORT_CONTRACT, STRICT_ORIGINAL_GATE=False,
            loss_policy=training.LOSS_POLICY, label_resampling_algorithm=training.LABEL_ALGORITHM,
            reference_parents=parents, case_manifests=markers, case_locations=locations, cases_detail=details,
            csv_sha256=fs.digest(root / 'dataset/lesions.csv'), rows=len(csv_rows), train_rows=823,
            resolution_warnings=warnings, report_sha256=fs.digest(root / 'dataset/PREPROCESS_REPORT.json')))
    return fs


class TrainingIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = cohort()

    def setUp(self):
        self.fs = copy.deepcopy(self.base)
        self.case = 'BraTS-MET-00000-000'
        self.directory = Path('/r9/dataset/cases') / self.case

    def check_identity(self, modality='t1c'):
        with self.fs.patched():
            return training.data_identity(modality)

    def change_json(self, path, change):
        value = self.fs.read_json(path)
        change(value)
        self.fs.json(path, value)

    def reseal_marker(self):
        marker_sha = self.fs.digest(self.directory / 'preprocess_complete.json')
        self.change_json('/r9/dataset/COMPLETE.ok', lambda c: c['case_manifests'].update({self.case: marker_sha}))

    def test_full926_memory_success_all_modalities(self):
        for modality in training.MODALITIES:
            with self.subTest(modality=modality):
                identity = self.check_identity(modality)
                self.assertEqual(len(identity['case_manifests']), 926)
                self.assertEqual(len(identity['index_sha256']), 926)
                self.assertEqual(len(identity['consumed']), 823 * 2)
                self.assertEqual(set(identity['reference_parents']), {'/r7', '/r8'})
                self.assertEqual(len(identity['resolution_warnings']), 2)
                self.assertNotIn('local_exception_sha256', identity)
                self.assertNotIn('reused_cases', identity)
                patched_case = 'BraTS-MET-00002-000'
                self.assertNotIn(patched_case + '-seg.nii.gz', identity['file_references'][patched_case])
                self.assertTrue(all('/r9/' in p for p in identity['consumed']))

    def test_marker_drift(self):
        self.fs.put(self.directory / 'preprocess_complete.json', '{}')
        with self.assertRaisesRegex(AssertionError, 'Preprocessing marker drift'):
            self.check_identity()

    def test_same_content_link_redirect(self):
        link = self.directory / (self.case + '-t1c.nii.gz')
        other = Path('/r8/dataset/cases') / self.case / link.name
        self.fs.put(other, self.fs.files[self.fs.resolve(link)])
        self.fs.links[link] = other
        with self.assertRaisesRegex(AssertionError, 'Output reference target drift'):
            self.check_identity()

    def test_all_six_content_hashes_including_val(self):
        for case, suffix in [(self.case, m) for m in (*training.MODALITIES, 'seg', 'brainmask')] + [('BraTS-MET-00823-000', 't2f')]:
            with self.subTest(case=case, suffix=suffix):
                path = Path('/r9/dataset/cases') / case / (case + '-' + suffix + '.nii.gz')
                target = self.fs.resolve(path)
                previous = self.fs.files[target]
                self.fs.put(target, previous + b'drift')
                with self.assertRaisesRegex(AssertionError, 'Consumed data SHA drift'):
                    self.check_identity()
                self.fs.put(target, previous)

    def test_loss_policy_mismatch(self):
        for path in ('/r9/config/RUN.json', '/r9/dataset/COMPLETE.ok'):
            with self.subTest(path=path):
                self.fs = copy.deepcopy(self.base)
                self.change_json(path, lambda value: value.update(loss_policy='local_fix'))
                with self.assertRaisesRegex(AssertionError, 'R9 policy mismatch'):
                    self.check_identity()

    def test_resealed_marker_policy_mismatch(self):
        self.change_json(self.directory / 'preprocess_complete.json',
                         lambda p: p['contract'].update(loss_policy='local_fix'))
        self.reseal_marker()
        with self.assertRaisesRegex(AssertionError, 'Marker policy mismatch'):
            self.check_identity()

    def test_label_algorithm_mismatch(self):
        self.change_json('/r9/dataset/COMPLETE.ok', lambda value: value.update(label_resampling_algorithm='case_exception'))
        with self.assertRaisesRegex(AssertionError, 'R9 policy mismatch'):
            self.check_identity()

    def test_local_patch_forbidden(self):
        self.change_json(self.directory / 'preprocess_complete.json',
                         lambda p: p['contract'].update(local_label_exception={'case': self.case}))
        self.reseal_marker()
        with self.assertRaisesRegex(AssertionError, 'local label fix forbidden'):
            self.check_identity()

    def test_frozen_parent_runtime_and_source_files(self):
        for path in ('/r7/runtime/train.py', '/r8/runtime/upstream_source/model.py'):
            with self.subTest(path=path):
                self.fs = copy.deepcopy(self.base)
                self.fs.put(path, b'drift')
                with self.assertRaisesRegex(AssertionError, 'Frozen parent (file|source) drift'):
                    self.check_identity()

    def test_parent_marker_drift(self):
        self.fs.put(Path('/r7/dataset/cases') / self.case / 'preprocess_complete.json', '{}')
        with self.assertRaisesRegex(AssertionError, 'Parent marker drift'):
            self.check_identity()

    def test_case_directory_cannot_be_link(self):
        self.fs.links[self.directory] = Path('/r7/dataset/cases') / self.case
        with self.assertRaisesRegex(AssertionError, 'case directory must be real'):
            self.check_identity()

    def test_persisted_index_marker_and_rows(self):
        path = Path('/r9/dataset/indices') / (self.case + '.json')
        for mutate, message in [(lambda i: i.update(marker_sha256='wrong'), 'index marker drift'),
                                (lambda i: i['rows'][0].update(center_x=7), 'index rows drift')]:
            with self.subTest(message=message):
                self.fs = copy.deepcopy(self.base)
                self.change_json(path, mutate)
                with self.assertRaisesRegex(AssertionError, message):
                    self.check_identity()

    def test_marker_membership_and_native_manifest(self):
        self.change_json(self.directory / 'preprocess_complete.json', lambda p: p.update(split='val'))
        self.reseal_marker()
        with self.assertRaisesRegex(AssertionError, 'Marker membership drift'):
            self.check_identity()
        self.fs = copy.deepcopy(self.base)
        self.fs.put(Path('/recon/dataset/cases') / self.case / 'case_manifest.json', '{}')
        with self.assertRaisesRegex(AssertionError, 'Native manifest drift'):
            self.check_identity()

    def test_csv_mapping_even_with_resealed_csv(self):
        rows = list(csv.DictReader(io.StringIO(self.fs.files[Path('/r9/dataset/lesions.csv')].decode())))
        rows[0]['scan_t1c'] = rows[1]['scan_t1c']
        self.fs.csv('/r9/dataset/lesions.csv', rows)
        self.change_json('/r9/dataset/COMPLETE.ok', lambda c: c.update(csv_sha256=self.fs.digest('/r9/dataset/lesions.csv')))
        with self.assertRaisesRegex(AssertionError, 'Lesion file mapping drift'):
            self.check_identity()

    def test_report_drift(self):
        self.fs.put('/r9/dataset/PREPROCESS_REPORT.json', '{}')
        with self.assertRaisesRegex(AssertionError, 'Preprocessing report drift'):
            self.check_identity()


class TrainingHelperTests(unittest.TestCase):
    def test_empty_cursor_and_bad_position_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'Empty training dataset'):
            training.new_cursor(0, 42)
        cursor = training.new_cursor(3, 42)
        cursor['position'] = 4
        with self.assertRaisesRegex(AssertionError, 'sampler position'):
            training.next_indices(cursor, 1)

    def test_cpu_upstream_update_and_checkpoint_replay(self):
        from monai.transforms import Compose

        class Denoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.5))

            def forward(self, x, t, cond):
                return x * self.weight

        config = json.loads((ROOT / 'config/RUN.json').read_text())['training']
        args = SimpleNamespace(**config, modality='t1c')
        model = Denoiser()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, betas=(args.adam_beta1, .999))
        scaler = torch.amp.GradScaler('cpu')
        transform = Compose([])
        schedule = training.upstream._diffusion_utils.make_noise_schedule(schedule='edm', sigma_data=args.sigma_data)
        batch = {'scan_t1c_crop_pad': torch.ones(1, 1, 4, 4, 4), 'label_crop_pad': torch.ones(1, 4, 4, 4, 4)}
        cursor = training.new_cursor(3, 42)
        cursor['position'] = 1
        before = model.weight.detach().clone()
        step, _, _, losses, _, _ = training.upstream.train(args, 0, [batch], model, optimizer, scaler, schedule, '')
        self.assertEqual(step, 1)
        self.assertEqual(len(losses), 1)
        self.assertTrue(np.isfinite(losses[0]))
        self.assertFalse(torch.equal(before, model.weight))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            training.save_checkpoint(path, model, optimizer, scaler, step, cursor, transform, {'r9': True}, args, schedule)
            expected = training.upstream.train(args, step, [batch], model, optimizer, scaler, schedule, '')
            expected_weight = model.weight.detach().clone()
            restored_step, restored_cursor = training.restore_checkpoint(path, model, optimizer, scaler, transform, {'r9': True}, args)
            actual = training.upstream.train(args, restored_step, [batch], model, optimizer, scaler, schedule, '')
            self.assertEqual(actual[0], expected[0])
            self.assertEqual(actual[3], expected[3])
            self.assertTrue(torch.equal(model.weight, expected_weight))
            self.assertEqual(restored_cursor['position'], 1)
            with self.assertRaisesRegex(AssertionError, 'Resume identity drift'):
                training.restore_checkpoint(path, model, optimizer, scaler, transform, {'r9': False}, args)


if __name__ == '__main__':
    unittest.main()
