"""Focused independent rejection tests for P0 native file validation."""

import importlib.util
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


if __name__ == '__main__':
    unittest.main()
