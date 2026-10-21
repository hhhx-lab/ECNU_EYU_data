#!/usr/bin/env python3
"""Regression checks for native NIfTI (x, y, z) lesion coordinates."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.csv_creator import (
    connected_component_analysis,
    tile_oversized_lesion,
)


def main() -> int:
    mask = np.zeros((36, 24, 14), dtype=bool)
    mask[10:34, 4:9, 2:7] = True
    component = connected_component_analysis(mask)[0]
    assert component["bbox"] == (10, 34, 4, 9, 2, 7), component
    assert component["centroid"] == (22, 6, 4), component

    tiles = list(tile_oversized_lesion(component, mask, crop_size=16, stride=12))
    assert len(tiles) >= 2, tiles
    assert all(tile["n_voxels"] > 0 for tile in tiles), tiles
    assert sum(tile["n_voxels"] for tile in tiles) >= int(mask.sum())
    print("axis_contract=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
