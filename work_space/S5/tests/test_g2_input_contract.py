import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


CODE_DIR = Path(__file__).resolve().parents[1] / "code"


def load_preprocessing_script():
    sys.path.insert(0, str(CODE_DIR))
    path = CODE_DIR / "2_preprocessing_mri.py"
    spec = importlib.util.spec_from_file_location("s5_preprocessing", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class S5G2InputContractTest(unittest.TestCase):
    def test_official_channel_order_and_prefixed_case_files(self):
        module = load_preprocessing_script()
        self.assertEqual(
            module.MODALITY_FILENAMES,
            ["t1n.nii.gz", "t1c.nii.gz", "t2w.nii.gz", "t2f.nii.gz"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_id = "BraTS-MET-00001-000"
            case_dir = root / "train" / case_id
            case_dir.mkdir(parents=True)
            for suffix in ("t1n.nii.gz", "t1c.nii.gz", "t2w.nii.gz", "t2f.nii.gz", "seg.nii.gz"):
                (case_dir / f"{case_id}-{suffix}").touch()
            preprocessor = module.MultiModalityPreprocessor(
                base_dir=str(root),
                image_dir="train",
                data_filenames=module.MODALITY_FILENAMES,
                seg_filename="seg.nii.gz",
            )
            resolved = Path(preprocessor.resolve_case_file(case_id, "t1n.nii.gz"))
            self.assertEqual(resolved, case_dir / f"{case_id}-t1n.nii.gz")


if __name__ == "__main__":
    unittest.main()
