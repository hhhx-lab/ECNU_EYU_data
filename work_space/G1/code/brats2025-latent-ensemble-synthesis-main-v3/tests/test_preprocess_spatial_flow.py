import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from preprocess import prepare_training_subject_space, write_spatial_metadata


class PreprocessSpatialFlowTests(unittest.TestCase):
    def _subject(self, root: Path, target_outlier: bool):
        case_id = "BraTS-MET-00001-000"
        case_dir = root / case_id
        case_dir.mkdir(parents=True)
        shape = (64, 64, 40)
        affine = np.diag([0.5, 0.5, 0.5, 1.0])
        image = np.zeros(shape, dtype=np.float32)
        image[4:60, 5:59, 3:37] = 0.5
        seg = np.zeros(shape, dtype=np.int16)
        seg[7:11, 28:34, 16:22] = 3
        modality_map = {}
        for modality in ("t1n", "t1c", "t2w", "t2f"):
            value = image.copy()
            if modality == "t2w" and target_outlier:
                value[63, 63, 39] = 100.0
            name = f"{case_id}-{modality}.nii.gz"
            nib.save(nib.Nifti1Image(value, affine), case_dir / name)
            modality_map[modality] = name
        seg_name = f"{case_id}-seg.nii.gz"
        nib.save(nib.Nifti1Image(seg, affine), case_dir / seg_name)
        return {
            "id": case_id,
            "path": case_dir,
            "modality_map": modality_map,
            "seg": seg_name,
        }

    def test_training_transform_does_not_depend_on_target_t2w_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean = self._subject(root / "clean", target_outlier=False)
            outlier = self._subject(root / "outlier", target_outlier=True)

            clean_prepared = prepare_training_subject_space(
                clean,
                target_shape=(32, 32, 20),
                base_spacing_mm=1.0,
                margin_mm=1.0,
            )
            outlier_prepared = prepare_training_subject_space(
                outlier,
                target_shape=(32, 32, 20),
                base_spacing_mm=1.0,
                margin_mm=1.0,
            )

            np.testing.assert_allclose(
                clean_prepared["transform"].target_affine,
                outlier_prepared["transform"].target_affine,
            )
            self.assertIn(3, np.unique(clean_prepared["segmentation"]))

    def test_spatial_metadata_records_audits_and_transform(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subject = self._subject(root / "case", target_outlier=False)
            prepared = prepare_training_subject_space(
                subject,
                target_shape=(32, 32, 20),
                base_spacing_mm=1.0,
                margin_mm=1.0,
            )
            output = root / "latents" / subject["id"]

            metadata_path = write_spatial_metadata(prepared, subject, output)
            metadata = json.loads(metadata_path.read_text())

            self.assertEqual(metadata["case_id"], subject["id"])
            self.assertEqual(metadata["transform"]["target_shape"], [32, 32, 20])
            self.assertEqual(metadata["lesion_support_audit"]["outside_voxel_count"], 0)


if __name__ == "__main__":
    unittest.main()
