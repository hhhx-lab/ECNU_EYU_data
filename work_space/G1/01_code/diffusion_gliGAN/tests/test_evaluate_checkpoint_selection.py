#!/usr/bin/env python3
"""Regression tests for explicit Diffusion evaluation checkpoint selection."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "infer"))

from src.infer.evaluate_generation import _find_checkpoint


class EvaluateCheckpointSelectionTests(unittest.TestCase):
    def test_explicit_step_selects_exact_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            weights = Path(tmpdir) / "t1c" / "weights"
            weights.mkdir(parents=True)
            for step in (140000, 145000, 150000):
                (weights / f"diffusion_{step}.pt").write_bytes(b"checkpoint")

            selected = _find_checkpoint(tmpdir, "t1c", checkpoint_step=145000)

            self.assertEqual(Path(selected).name, "diffusion_145000.pt")

    def test_explicit_step_rejects_missing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            weights = Path(tmpdir) / "t2w" / "weights"
            weights.mkdir(parents=True)
            (weights / "diffusion_150000.pt").write_bytes(b"checkpoint")

            with self.assertRaisesRegex(FileNotFoundError, "145000"):
                _find_checkpoint(tmpdir, "t2w", checkpoint_step=145000)


if __name__ == "__main__":
    unittest.main()
