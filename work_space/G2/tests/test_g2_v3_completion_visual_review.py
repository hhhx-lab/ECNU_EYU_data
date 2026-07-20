from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "g2_v3_completion_visual_review.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("completion_visual_review", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CompletionVisualReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()

    def test_component_selection_keeps_smallest_and_largest(self):
        rows = [
            {"component_id": index, "volume_mm3": float(index), "centroid": (1, 1, 1)}
            for index in range(1, 11)
        ]
        selected = self.mod.select_components(rows, 4)
        ids = {int(row["component_id"]) for row in selected}
        self.assertEqual(ids, {1, 2, 9, 10})

    def test_signed_shape_and_focus_contract(self):
        segmentation = np.zeros((20, 20, 20), dtype=np.int16)
        segmentation[2:3, 2:3, 2:3] = 3
        segmentation[10:15, 10:15, 10:15] = 1
        _, rows = self.mod.component_rows(segmentation, (1.0, 1.0, 1.0))
        tiny_focus = self.mod.choose_focus(rows, True, False, segmentation.shape)
        z_focus = self.mod.choose_focus(rows, False, True, segmentation.shape)
        self.assertEqual(tiny_focus, (2, 2, 2))
        self.assertEqual(z_focus, (12, 12, 12))

    def test_overview_writes_nonempty_png(self):
        shape = (20, 20, 20)
        image = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
        images = {modality: image.copy() for modality in self.mod.MODALITIES}
        segmentation = np.zeros(shape, dtype=np.int16)
        segmentation[8:12, 8:12, 8:12] = 3
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "overview.png"
            self.mod.render_overview(
                "BraTS-MET-00001-000",
                images,
                segmentation,
                (10, 10, 10),
                "tiny",
                output,
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
