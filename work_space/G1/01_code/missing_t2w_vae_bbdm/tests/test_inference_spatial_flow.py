import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from main import prepare_s_data
from synthesis.pipeline import prepare_inference_subject


class InferenceSpatialFlowTests(unittest.TestCase):
    def test_missing_t2w_subject_uses_seg_aware_shared_transform(self):
        with tempfile.TemporaryDirectory() as temporary:
            case_id = "BraTS-MET-00001-000"
            case_dir = Path(temporary) / case_id
            case_dir.mkdir()
            shape = (64, 64, 40)
            affine = np.diag([0.5, 0.5, 0.5, 1.0])
            image = np.zeros(shape, dtype=np.float32)
            image[4:60, 5:59, 3:37] = 0.5
            image[7:11, 28:34, 16:22] = 1.0
            seg = np.zeros(shape, dtype=np.int16)
            seg[7:11, 28:34, 16:22] = 4
            for index, modality in enumerate(("t1n", "t1c", "t2f")):
                nib.save(
                    nib.Nifti1Image(image + index * 0.02, affine),
                    case_dir / f"{case_id}-{modality}.nii.gz",
                )
            nib.save(
                nib.Nifti1Image(seg, affine),
                case_dir / f"{case_id}-seg.nii.gz",
            )

            subject = prepare_s_data(str(case_dir), load_seg=True)
            prepared = prepare_inference_subject(
                subject,
                target_shape=(32, 32, 20),
                base_spacing_mm=1.0,
                margin_mm=1.0,
            )

            self.assertEqual(len(prepared["images"]), 3)
            self.assertTrue(all(image.shape == (32, 32, 20) for image in prepared["images"]))
            self.assertIn(4, np.unique(prepared["segmentation"]))
            self.assertEqual(subject["seg_path"], str(case_dir / f"{case_id}-seg.nii.gz"))


if __name__ == "__main__":
    unittest.main()
