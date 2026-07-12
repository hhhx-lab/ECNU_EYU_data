import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "code" / "g2_v3_completion_intake.py"


def load_module():
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    spec = importlib.util.spec_from_file_location("g2_v3_completion_intake", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def valid_metadata() -> dict[str, object]:
    return {
        "generation_run_id": "v3-unit",
        "generator_name": "g1_missing_t2w_v3",
        "generation_mode": "completion",
        "seed": 42,
        "source_csv": "data/g1_v3_data_placement_manifest.csv",
        "vae_weights": "vae.pt",
        "encdec_checkpoint": "encdec.pt",
        "bbdm_checkpoint": "bbdm.pt",
        "bbdm_s": 0.005,
        "validation_run": "validation/run-unit",
    }


class G2V3CompletionIntakeTest(unittest.TestCase):
    def test_requires_validation_provenance(self):
        mod = load_module()
        metadata = valid_metadata()
        metadata.pop("validation_run")
        with self.assertRaises(ValueError):
            mod.validate_v3_metadata(metadata)

    def test_rejects_noncompletion_mode(self):
        mod = load_module()
        metadata = valid_metadata()
        metadata["generation_mode"] = "full_generation"
        with self.assertRaises(ValueError):
            mod.validate_v3_metadata(metadata)

    def test_accepts_complete_v3_metadata(self):
        mod = load_module()
        mod.validate_v3_metadata(valid_metadata())


if __name__ == "__main__":
    unittest.main()
