import tempfile
import unittest
from pathlib import Path

import numpy as np

from precompute_lesion_weights import downsample_weight_map
from synthesis.utils import get_chkpoint_path


class V3RuntimeHelperTests(unittest.TestCase):
    def test_divisible_lesion_weight_pooling_preserves_block_max(self):
        weights = np.zeros((8, 8, 8), dtype=np.float32)
        weights[1, 1, 1] = 2.0
        weights[6, 6, 6] = 5.0

        pooled = downsample_weight_map(weights, (2, 2, 2))

        self.assertEqual(pooled.shape, (2, 2, 2))
        self.assertEqual(float(pooled[0, 0, 0]), 2.0)
        self.assertEqual(float(pooled[1, 1, 1]), 5.0)
        self.assertTrue(np.all(pooled > 0))

    def test_latest_checkpoint_uses_numeric_training_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary)
            for name in ("model_9.pt", "model_100.pt", "latest.pt"):
                (checkpoint_dir / name).touch()

            selected = Path(get_chkpoint_path(str(checkpoint_dir)))

            self.assertEqual(selected.name, "model_100.pt")


if __name__ == "__main__":
    unittest.main()
