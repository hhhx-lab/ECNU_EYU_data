from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from custom_nnunet.met_aug_core import ComponentRecord, canonical_json_sha256, sha256_file
from custom_nnunet.met_aug_diffusion import G1FourModalityInpaintingBackend
from custom_nnunet.met_aug_k1245 import (
    ANCHOR_LABELS,
    K1245Config,
    K1245Manifest,
    LEGACY_ANCHOR_LABELS,
    RCAwareFixV3CandidateProcessor,
    _phenotype_scaled,
    anchor_statistics,
    extract_k1245_components,
)


class K1245ContractTests(unittest.TestCase):
    def test_rc_label4_passes_shared_input_contract(self):
        image = np.zeros((4, 64, 64, 64), dtype=np.float32)
        segmentation = np.zeros((64, 64, 64), dtype=np.int16)
        label_cube = np.zeros((64, 64, 64), dtype=np.int16)
        label_cube[28:32, 28:32, 28:32] = 4
        valid_mask = np.ones((64, 64, 64), dtype=bool)
        RCAwareFixV3CandidateProcessor._validate_inputs(
            image, segmentation, label_cube, valid_mask, (1.0, 1.0, 1.0)
        )

    def test_rc_only_component_is_extracted_with_label4_anchor(self):
        label = np.zeros((16, 16, 16), dtype=np.int16)
        label[4:7, 5:8, 6:9] = 4
        components, exclusions = extract_k1245_components(
            label, (1, 1, 1), min_anchor_volume_mm3=27, include_rc=True
        )
        self.assertEqual(exclusions, {})
        self.assertEqual(len(components), 1)
        self.assertEqual(components[0]["stats"]["classes_present"], (4,))
        self.assertEqual(components[0]["stats"]["core_volume_mm3"], 27.0)

    def test_mixed_rc_anchor_gets_uniquely_adjacent_snfh(self):
        label = np.zeros((20, 20, 20), dtype=np.int16)
        label[5:8, 5:8, 5:8] = 3
        label[8:11, 5:8, 5:8] = 4
        label[4, 5:8, 5:8] = 2
        components, _ = extract_k1245_components(
            label, (1, 1, 1), min_anchor_volume_mm3=27, include_rc=True
        )
        self.assertEqual(len(components), 1)
        self.assertEqual(components[0]["stats"]["classes_present"], (2, 3, 4))

    def test_non_rc_x3_pool_preserves_source_rc_exclusion(self):
        label = np.zeros((10, 10, 10), dtype=np.int16)
        label[1:3, 1:3, 1:3] = 3
        label[7, 7, 7] = 4
        components, exclusions = extract_k1245_components(
            label, (1, 1, 1), min_anchor_volume_mm3=8, include_rc=False
        )
        self.assertEqual(components, [])
        self.assertEqual(exclusions, {"source_contains_rc": 1})

    def test_direct_crop_contract_rejects_bbox_over_56_mm(self):
        label = np.zeros((70, 4, 4), dtype=np.int16)
        label[:, 1:3, 1:3] = 4
        components, exclusions = extract_k1245_components(
            label, (1, 1, 1), min_anchor_volume_mm3=27, include_rc=True
        )
        self.assertEqual(components, [])
        self.assertEqual(exclusions["bbox_exceeds_limit"], 1)

    def test_g1_condition_exposes_rc_channel(self):
        label = np.zeros((4, 4, 4), dtype=np.int16)
        label[1:3, 1:3, 1:3] = 4
        condition = G1FourModalityInpaintingBackend._label_to_multichannel(label)
        self.assertEqual(condition.shape, (4, 4, 4, 4))
        self.assertEqual(int(condition[3].sum()), 8)
        self.assertEqual(int(condition[:3].sum()), 0)

    def test_b1_rc_exemption_is_exact(self):
        label = np.full((3, 3, 3), 4, dtype=np.int16)
        record = ComponentRecord(
            component_id="rc", manifest_version="test", source_case_id="case-000",
            patient_group="case", split="train", component_path="rc.npz",
            label_sha256="0" * 64, source_label_sha256="0" * 64,
            source_modalities_sha256={key: "0" * 64 for key in ("t1n", "t1c", "t2w", "t2f")},
            source_affine_sha256="0" * 64, spacing_mm=(1, 1, 1),
            core_volume_mm3=27, total_volume_mm3=27, bbox_mm=(3, 3, 3),
            bbox_voxels=(3, 3, 3), class_counts={"4": 27}, classes_present=(4,),
            core_centroid_norm=(0.5, 0.5, 0.5),
        )
        scaled, updated, audit = _phenotype_scaled(
            label, record, np.random.default_rng(1), (record,), anchors=ANCHOR_LABELS
        )
        self.assertTrue(np.array_equal(scaled, label))
        self.assertEqual(updated, record)
        self.assertEqual(audit["b_result"], "B_RC_EXEMPT")

    def test_anchor_statistics_legacy_and_rc_are_explicit(self):
        label = np.full((3, 3, 3), 4, dtype=np.int16)
        self.assertEqual(anchor_statistics(label, (1, 1, 1), anchor_labels=ANCHOR_LABELS)["core_volume_mm3"], 27)
        with self.assertRaisesRegex(Exception, "no configured anchor"):
            anchor_statistics(label, (1, 1, 1), anchor_labels=LEGACY_ANCHOR_LABELS)


if __name__ == "__main__":
    unittest.main()
