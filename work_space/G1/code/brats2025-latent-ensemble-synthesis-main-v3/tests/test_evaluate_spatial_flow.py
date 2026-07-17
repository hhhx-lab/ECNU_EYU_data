import csv
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from evaluate import (
    find_eval_subjects,
    prepare_eval_subject,
    save_synthesized_output,
)


class EvaluateSpatialFlowTests(unittest.TestCase):
    def _write_case(self, root: Path, case_id: str):
        case_dir = root / case_id
        case_dir.mkdir(parents=True)
        shape = (64, 64, 40)
        affine = np.diag([0.5, 0.5, 0.5, 1.0])
        image = np.zeros(shape, dtype=np.float32)
        image[4:60, 5:59, 3:37] = 0.4
        image[7:11, 28:34, 16:22] = 1.0
        seg = np.zeros(shape, dtype=np.int16)
        seg[7:11, 28:34, 16:22] = 3
        files = {}
        for index, modality in enumerate(("t1n", "t1c", "t2w", "t2f")):
            name = f"{case_id}-{modality}.nii.gz"
            nib.save(nib.Nifti1Image(image + index * 0.02, affine), case_dir / name)
            files[modality] = name
        seg_name = f"{case_id}-seg.nii.gz"
        nib.save(nib.Nifti1Image(seg, affine), case_dir / seg_name)
        files["seg"] = seg_name
        return case_dir, files, image, seg, affine

    def test_explicit_case_list_filters_validation_without_changing_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "data.csv"
            rows = []
            for case_id in ("BraTS-MET-00001-000", "BraTS-MET-00002-000"):
                _, files, _, _, _ = self._write_case(root, case_id)
                rows.append({"id": case_id, **files, "split": "val"})
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            subjects = find_eval_subjects(
                csv_path,
                root,
                split="val",
                case_ids={"BraTS-MET-00002-000"},
            )

            self.assertEqual([subject["id"] for subject in subjects], ["BraTS-MET-00002-000"])
            self.assertIn("seg", subjects[0]["files"])

    def test_prepare_and_save_use_one_transform_and_native_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_id = "BraTS-MET-00001-000"
            case_dir, files, _, seg, affine = self._write_case(root, case_id)
            subject = {"id": case_id, "path": str(case_dir), "files": files}

            prepared = prepare_eval_subject(
                subject,
                target_shape=(32, 32, 20),
                base_spacing_mm=1.0,
                margin_mm=1.0,
            )
            destination = root / "generated-t2w.nii.gz"
            save_synthesized_output(
                prepared["images_by_modality"]["t2w"],
                prepared,
                destination,
            )
            restored = nib.load(destination)
            restored_data = restored.get_fdata(dtype=np.float32)

            self.assertGreater(int(np.count_nonzero(prepared["segmentation"])), 0)
            self.assertEqual(restored.shape, seg.shape)
            np.testing.assert_allclose(restored.affine, affine, atol=1e-6)
            self.assertGreater(float(restored_data[seg > 0].mean()), 0.5)


if __name__ == "__main__":
    unittest.main()
