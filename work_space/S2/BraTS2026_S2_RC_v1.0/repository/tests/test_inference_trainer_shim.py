import importlib.util
import inspect
from pathlib import Path
import unittest

SHIM_PATH = (
    Path(__file__).resolve().parents[1]
    / "custom_nnunet"
    / "nnUNetTrainerBraTS2026RC_inference.py"
)
FROZEN_INFERENCE_PATH = Path(__file__).resolve().parents[1] / "inference_frozen.py"
INFER_SH_PATH = Path(__file__).resolve().parents[1] / "infer.sh"


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

    def test_met_aug_checkpoint_uses_the_non_augmenting_inference_shim(self):
        frozen_source = FROZEN_INFERENCE_PATH.read_text(encoding="utf-8")
        infer_source = INFER_SH_PATH.read_text(encoding="utf-8")

        self.assertIn("nnUNetTrainerBraTS2026RCMetAugFocalCompletionFineTune", frozen_source)
        self.assertIn("nnUNetTrainerBraTS2026RCMetAugControlFocalCompletionFineTune", frozen_source)
        self.assertIn("met_aug_route_a)", infer_source)
        self.assertIn("DEFAULT_USE_INFERENCE_TRAINER_SHIM=1", infer_source)
        self.assertIn("training-only Diffusion code is forbidden", infer_source)


if __name__ == "__main__":
    unittest.main()
