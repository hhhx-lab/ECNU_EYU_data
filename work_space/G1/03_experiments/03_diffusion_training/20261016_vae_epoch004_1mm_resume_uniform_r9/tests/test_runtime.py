"""Focused CPU regression tests; not a substitute for the waived GPU/quality smoke."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
import prepare
spec = importlib.util.spec_from_file_location('run_training', ROOT / 'runtime/train.py')
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


class RuntimeTests(unittest.TestCase):
    def test_uniform_loss_warning_keeps_quantization(self):
        for cls in range(1, 5):
            label = np.zeros((8, 8, 4), np.uint8)
            label[2, 3, 1] = cls
            shape, factor, offset = prepare.physical_grid(label.shape, (.5, .5, 1), (1, 1, 1))
            out = prepare.resample_physical(label[None], shape, factor, offset, (.5, .5, 1), (1, 1, 1), is_seg=True)[0]
            record = prepare.loss_record(prepare.label_summary(label, (.5, .5, 1)), prepare.label_summary(out, (1, 1, 1)))
            self.assertEqual(record['lost_classes'], [str(cls)])
            self.assertTrue(record['resolution_warning'])
            self.assertIsNone(record['local_label_exception'])
            self.assertEqual(int((out == cls).sum()), 0)

    def test_real_errors_collected_not_downgraded(self):
        from unittest.mock import patch
        job = ({}, {'source_case_id': 'broken'}, {}, 'sha')
        with patch.object(prepare, 'process', side_effect=ValueError('Native geometry mismatch')):
            record = prepare.process_record(job)
        self.assertFalse(record['ok'])
        self.assertIn('geometry mismatch', record['error'])

    def test_membership(self):
        rows = prepare.members()
        self.assertEqual(len(rows), 926)
        self.assertEqual(sum(r['split'] == 'train' for r in rows), 823)
        self.assertNotIn('test', {r['split'] for r in rows})

    def test_mask_and_normalization(self):
        a = np.zeros((16, 18, 20), np.float32)
        a[2:14, 3:15, 4:16] = np.arange(12**3).reshape(12, 12, 12) + 1
        mask = prepare.common_mask([a, a * 2, a * 3])
        out = prepare.normalize(a + .1, mask)
        self.assertTrue((out[~mask] == 0).all())
        self.assertAlmostEqual(float(out[mask].mean()), 0, places=5)
        self.assertAlmostEqual(float(out[mask].std()), 1, places=5)
        with self.assertRaises(ValueError):
            prepare.common_mask([np.zeros_like(a)] * 3)

    def test_physical_affine(self):
        affine = np.diag([2., 1.5, .8, 1.])
        bbox = [[2, 12], [3, 15], [4, 19]]
        shape, factor, offset = prepare.physical_grid((10, 12, 15), (2, 1.5, .8), (1, 1, 1))
        target, spacing = prepare.new_affine(affine, bbox, (10, 12, 15), shape, (2, 1.5, .8), (1, 1, 1))
        self.assertTrue(np.allclose(np.linalg.norm(target[:3, :3], axis=0), 1))
        j = np.array([3, 4, 5])
        self.assertTrue(np.allclose(target @ [*j, 1], affine @ [*(np.array([2, 3, 4]) + offset + factor * j), 1]))

    def test_cursor_resume(self):
        c = training.new_cursor(19, 20261013)
        indices, end = training.next_indices(c, 16)
        self.assertEqual(len(indices), 16)
        c['position'] = end
        saved = __import__('copy').deepcopy(c)
        self.assertEqual(training.next_indices(c, 16), training.next_indices(saved, 16))
        self.assertEqual(len(training.next_indices(c, 16)[0]), 3)
        c['position'] = 19
        nxt, _ = training.next_indices(c, 16)
        self.assertEqual(c['epoch'], 2)
        self.assertEqual(len(nxt), 16)

    def test_complete_checkpoint_and_rng(self):
        from monai.transforms import Compose
        from src.utils.gaussian_noise_tumour import GaussianNoiseTumour
        transform = Compose([GaussianNoiseTumour('scan_t1c')])
        transform.set_random_state(seed=20261013)
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(.9, .999))
        scaler = torch.amp.GradScaler('cpu')
        loss = model(torch.ones((1, 3))).sum()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        cfg = json.loads((ROOT / 'config/RUN.json').read_text())['training']
        args = SimpleNamespace(**cfg, modality='t1c')
        schedule = training.upstream._diffusion_utils.make_noise_schedule(schedule='edm')
        identity = {'test': 'synthetic'}
        cursor = training.new_cursor(19, 20261013)
        cursor['position'] = 16
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'checkpoint.pt'
            digest = training.save_checkpoint(path, model, optimizer, scaler, 1, cursor, transform, identity, args, schedule)
            expected = [__import__('random').random(), np.random.rand(), torch.rand(1).item(), transform.transforms[0].R.rand()]
            step, restored = training.restore_checkpoint(path, model, optimizer, scaler, transform, identity, args)
            actual = [__import__('random').random(), np.random.rand(), torch.rand(1).item(), transform.transforms[0].R.rand()]
            self.assertEqual(actual, expected)
            self.assertEqual(step, 1)
            self.assertEqual(restored['position'], 16)
            self.assertEqual(len(digest), 64)
            with self.assertRaises(AssertionError):
                training.restore_checkpoint(path, model, optimizer, scaler, transform, {'different': True}, args)

    def test_stop_signal(self):
        training.STOP = False
        training.request_stop(10, None)
        self.assertTrue(training.STOP)
        training.STOP = False

    def test_actual_preprocess_csv_loader(self):
        import csv
        import nibabel as nib
        from src.utils.data_utils import get_loader
        old_root = prepare.ROOT
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            case = 'BraTS-MET-99999-000'
            native = root / 'recon/dataset/cases' / case
            real = root / 'source/data/input' / case
            native.mkdir(parents=True)
            real.mkdir(parents=True)
            (root / 'config').mkdir()
            (root / 'config/P0_SOURCE_MAPPING.json').write_text(json.dumps({'corrected_label_paths': {}}))
            a = np.zeros((32, 34, 36), dtype=np.float32)
            a[3:29, 4:30, 5:31] = np.random.RandomState(3).uniform(1, 10, (26, 26, 26))
            label = np.zeros_like(a, dtype=np.uint8)
            label[14:18, 15:19, 16:20] = 4
            label[0, 0, 0] = 4
            source_row = {'id': case, 'split': 'train'}
            files, source_sha = {}, {}
            for modality in (*prepare.MODS, 'seg'):
                name = case + '-' + modality + '.nii.gz'
                image = label if modality == 'seg' else a
                nib.save(nib.Nifti1Image(image, np.eye(4)), real / name)
                nib.save(nib.Nifti1Image(image if modality == 'seg' else image + .01, np.eye(4)), native / name)
                source_row[modality] = name
                source_sha[modality] = prepare.sha(real / name)
                files[modality] = {'sha256': prepare.sha(native / name)}
            marker = native / 'case_manifest.json'
            marker.write_text(json.dumps(dict(case_id=case, split='train', checkpoint_sha256=prepare.CHECKPOINT,
                                    geometry='native', native_shape=list(a.shape), files=files, source_sha256=source_sha)))
            prepare.ROOT = root
            try:
                result = prepare.process(({'recon': str(root / 'recon'), 'source_code': str(root / 'source')},
                              {'source_case_id': case, 'split': 'train', 'patient_group': '99999'}, source_row, prepare.sha(marker)))
                self.assertEqual(result['labels'], [0, 4])
                actual = nib.load(root / 'dataset/cases' / case / (case + '-seg.nii.gz')).get_fdata()
                self.assertEqual(int(actual[0, 0, 0]), 4)
                signal = nib.load(root / 'dataset/cases' / case / (case + '-brainmask.nii.gz')).get_fdata()
                image = nib.load(root / 'dataset/cases' / case / (case + '-t1c.nii.gz')).get_fdata()
                self.assertEqual(signal[0, 0, 0], 0)
                self.assertEqual(image[0, 0, 0], 0)
                contract = json.loads((root / 'dataset/cases' / case / 'spatial_mapping.json').read_text())
                self.assertEqual(contract['native_label_outside_signal_voxels'], 1)
                self.assertEqual(contract['bbox'], [[0, 32], [0, 34], [0, 36]])
                rows = result['rows']
                self.assertTrue(rows)
                for row in rows:
                    row['patient_n_crops'] = len(rows)
                path = root / 'lesions.csv'
                with path.open('w', newline='') as h:
                    writer = csv.DictWriter(h, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                config = json.loads((ROOT / 'config/RUN.json').read_text())['training']
                args = SimpleNamespace(**{**config, 'num_workers': 1, 'batch_size': 1}, csv_path=str(path), modality='t1c')
                loader = get_loader(args)
                batch = loader.collate_fn([loader.dataset[0]])
                self.assertEqual(tuple(batch['scan_t1c_crop_pad'].shape), (1, 1, 64, 64, 64))
                self.assertEqual(tuple(batch['label_crop_pad'].shape), (1, 4, 64, 64, 64))
                self.assertGreater(int(batch['label_crop_pad'][:, 3].sum()), 0)
                self.assertTrue(torch.isfinite(batch['scan_t1c_crop_pad']).all())
                parent_root = root
                new_root = root / 'resume'
                new_root.mkdir()
                prepare.ROOT = new_root
                reused = prepare.reuse_case({'reuse_parent': str(parent_root), 'recon': str(root / 'recon')},
                    {'source_case_id': case, 'split': 'train', 'patient_group': '99999'},
                    {'marker_sha256': result['marker_sha256'], 'directory': str(root / 'dataset/cases' / case), 'mode': 'all_outputs_reference'}, prepare.sha(marker))
                self.assertEqual(reused['origin'], 'readonly_file_reference')
                self.assertTrue((new_root / 'dataset/cases' / case / (case+'-t1c.nii.gz')).is_symlink())
                self.assertEqual(reused['rows'][0]['n_voxels'], result['rows'][0]['n_voxels'])
                parent_case = root / 'dataset/cases' / case
                parent_marker = parent_case / 'preprocess_complete.json'
                parent_info = json.loads(parent_marker.read_text())
                patch_label = nib.load(parent_case / (case+'-seg.nii.gz')).get_fdata().astype(np.uint8)
                patch_label[1, 1, 1] = 4
                nib.save(nib.Nifti1Image(patch_label, np.eye(4)), parent_case / (case+'-seg.nii.gz'))
                parent_info['outputs'][case+'-seg.nii.gz'] = prepare.sha(parent_case / (case+'-seg.nii.gz'))
                parent_info['contract']['local_label_exception'] = {'historical': 'test_patch'}
                parent_info['contract']['label_target_1mm'] = prepare.label_summary(patch_label, (1, 1, 1))
                parent_marker.write_text(json.dumps(parent_info))
                (parent_case / 'spatial_mapping.json').write_text(json.dumps(parent_info['contract']))
                rebuild_root = root / 'label_only_resume'
                rebuild_root.mkdir()
                prepare.ROOT = rebuild_root
                rebuilt = prepare.reuse_case({'recon': str(root / 'recon')},
                    {'source_case_id': case, 'split': 'train', 'patient_group': '99999'},
                    {'marker_sha256': prepare.sha(parent_marker), 'directory': str(parent_case), 'mode': 'label_only_rebuild'},
                    prepare.sha(marker))
                rebuilt_case = rebuild_root / 'dataset/cases' / case
                self.assertEqual(rebuilt['origin'], 'label_only_rebuild')
                self.assertFalse((rebuilt_case / (case+'-seg.nii.gz')).is_symlink())
                self.assertTrue((rebuilt_case / (case+'-t1c.nii.gz')).is_symlink())
                self.assertEqual(int(nib.load(rebuilt_case / (case+'-seg.nii.gz')).get_fdata()[1, 1, 1]), 0)
                self.assertEqual(len(rebuilt['file_references']), 5)
                bad_root = root / 'bad_resume'
                bad_root.mkdir()
                prepare.ROOT = bad_root
                with self.assertRaises(AssertionError):
                    prepare.reuse_case({'reuse_parent': str(parent_root), 'recon': str(root / 'recon')},
                        {'source_case_id': case, 'split': 'train', 'patient_group': '99999'},
                        {'marker_sha256': 'bad', 'directory': str(root / 'dataset/cases' / case), 'mode': 'all_outputs_reference'}, prepare.sha(marker))
                corruption = parent_root / 'dataset/cases' / case / (case + '-t1c.nii.gz')
                with corruption.open('ab') as f:
                    f.write(b'drift')
                with self.assertRaises(AssertionError):
                    prepare.reuse_case({'reuse_parent': str(parent_root), 'recon': str(root / 'recon')},
                        {'source_case_id': case, 'split': 'train', 'patient_group': '99999'},
                        {'marker_sha256': result['marker_sha256'], 'directory': str(root / 'dataset/cases' / case), 'mode': 'all_outputs_reference'}, prepare.sha(marker))
            finally:
                prepare.ROOT = old_root


if __name__ == '__main__':
    unittest.main()
