import importlib.util
import inspect
from pathlib import Path
import unittest

SHIM_PATH = (
    Path(__file__).resolve().parents[1]
    / "custom_nnunet"
    / "nnUNetTrainerBraTS2026RC_inference.py"
)


class InferenceTrainerShimTest(unittest.TestCase):
    def test_shim_preserves_base_network_builder_contract(self):
        self.assertTrue(SHIM_PATH.is_file(), SHIM_PATH)
        self.assertNotIn(
            "nnunetv2.training.nnUNetTrainer",
            SHIM_PATH.read_text(encoding="utf-8"),
        )
        spec = importlib.util.spec_from_file_location("s2_inference_trainer_shim", SHIM_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        trainer = module.nnUNetTrainerBraTS2026RC

        self.assertEqual(
            list(inspect.signature(trainer.build_network_architecture).parameters),
            [
                "plans_manager",
                "configuration_manager",
                "num_input_channels",
                "num_output_channels",
                "enable_deep_supervision",
            ],
        )


if __name__ == "__main__":
    unittest.main()
