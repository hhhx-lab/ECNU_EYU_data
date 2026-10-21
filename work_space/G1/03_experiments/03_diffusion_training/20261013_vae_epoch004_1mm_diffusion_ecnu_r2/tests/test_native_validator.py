"""Focused independent rejection tests for P0 native file validation."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np

spec = importlib.util.spec_from_file_location('native_validator', Path(__file__).resolve().parents[1] / 'runtime/validate_native.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


class NativeValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.reference = nib.Nifti1Image(np.ones((7, 9, 11), dtype=np.float32), np.diag([1, 0.8, 1.4, 1]))

    def tearDown(self):
        self.temp.cleanup()

    def image(self, data, affine=None):
        path = self.root / 'image.nii.gz'
        nib.save(nib.Nifti1Image(data.astype(np.float32), self.reference.affine if affine is None else affine), path)
        return path

    def test_valid_image(self):
        result = v.validate_volume(self.image(np.ones(self.reference.shape)), self.reference)
        self.assertEqual(result['shape_xyz'], [7, 9, 11])

    def test_affine_shift_rejected(self):
        affine = self.reference.affine.copy()
        affine[0, 3] = 10
        with self.assertRaisesRegex(ValueError, 'affine mismatch'):
            v.validate_volume(self.image(np.ones(self.reference.shape), affine), self.reference)

    def test_nonfinite_rejected(self):
        data = np.ones(self.reference.shape)
        data[1, 2, 3] = np.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            v.validate_volume(self.image(data), self.reference)

    def test_empty_image_rejected(self):
        with self.assertRaisesRegex(ValueError, 'empty image'):
            v.validate_volume(self.image(np.zeros(self.reference.shape)), self.reference)

    def test_rc_and_empty_label_allowed(self):
        for value in (0, 4):
            result = v.validate_volume(self.image(np.full(self.reference.shape, value)), self.reference, label=True)
            self.assertEqual(result['label_values'], [value])

    def test_illegal_and_fractional_labels_rejected(self):
        for value in (8, 1.5):
            with self.assertRaises(ValueError):
                v.validate_volume(self.image(np.full(self.reference.shape, value)), self.reference, label=True)

    def test_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            v.validate_volume(self.image(np.ones((11, 9, 7))), self.reference)

    def test_sha_detects_byte_mutation(self):
        path = self.image(np.ones(self.reference.shape))
        original = v.sha(path)
        self.image(np.full(self.reference.shape, 2))
        self.assertNotEqual(original, v.sha(path))

    def test_evidence_cannot_be_overwritten(self):
        path = self.root / 'audit.json'
        v.exclusive(path, {'status': 'pass'})
        with self.assertRaises(FileExistsError):
            v.exclusive(path, {'status': 'other'})

    def test_capacity_is_lightweight_p0_only(self):
        root = Path(__file__).resolve().parents[1]
        contract = json.loads((root / 'config/P0_CAPACITY_CONTRACT.json').read_text())
        self.assertEqual(contract['minimum_free_bytes'], 1024**3)
        self.assertEqual(contract['stage'], 'P0_native_integrity_only')
        self.assertFalse(contract['writes_image_arrays'])
        self.assertFalse(contract['writes_checkpoints'])
        self.assertIn('P3_checkpoint_capacity', contract['not_authorization_for'])

    def test_launch_and_environment_use_new_attempt(self):
        root = Path(__file__).resolve().parents[1]
        for relative in ('slurm/validate_native.slurm', 'config/p0.env.example'):
            value = (root / relative).read_text()
            self.assertIn('g1_diffusion_1mm_vae_epoch004_20261013_r2', value)
            self.assertNotIn('g1_diffusion_1mm_vae_epoch004_20261013_r1', value)

    def test_retry_authorization_retains_training_budget(self):
        root = Path(__file__).resolve().parents[1]
        auth = json.loads((root / 'plan/EXECUTION_AUTHORIZATION.json').read_text())
        self.assertEqual(auth['attempt'], 'r2')
        self.assertEqual(auth['per_modality_successful_updates'], 200000)
        self.assertEqual(len(auth['modalities']), 4)
        self.assertIn('P3_only_after_all_gates', auth['authorized_scope'])
        self.assertIn('locked104_quality_evaluation', auth['excluded_scope'])


if __name__ == '__main__':
    unittest.main()
