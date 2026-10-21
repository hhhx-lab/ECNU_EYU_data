#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
PROJECT = ROOT.parents[4]
MODULE_PATH = ROOT / "runtime/build_case_views.py"
SPEC = importlib.util.spec_from_file_location("build_case_views", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EvaluationOnlyMappingTests(unittest.TestCase):
    def test_locked_104_mapping_is_exact_bijection(self) -> None:
        test_list = PROJECT / "work_space/S2/BraTS2026_S2_RC_v1.0/repository/data/splits/completion_warmstart/test_locked.txt"
        mapping_csv = PROJECT / "work_space/G2/results/manifests/nnunet_case_mapping_master.csv"
        pairs = MODULE.load_case_contract(test_list, mapping_csv, 104)
        self.assertEqual(len(pairs), 104)
        self.assertEqual(len({item[0] for item in pairs}), 104)
        self.assertEqual(len({item[1] for item in pairs}), 104)
        encoded = "".join(f"{internal},{source}\n" for internal, source in sorted(pairs)).encode()
        self.assertEqual(
            hashlib.sha256(encoded).hexdigest(),
            "78dd2710043171dd1de75043ff5d9f7be7ff9a226b71191a11c17a6ec493ffaf",
        )

    def test_known_six_digit_ids_map_to_canonical_source_ids(self) -> None:
        test_list = PROJECT / "work_space/S2/BraTS2026_S2_RC_v1.0/repository/data/splits/completion_warmstart/test_locked.txt"
        mapping_csv = PROJECT / "work_space/G2/results/manifests/nnunet_case_mapping_master.csv"
        pairs = dict(MODULE.load_case_contract(test_list, mapping_csv, 104))
        self.assertEqual(pairs["BraTSMET_001248"], "BraTS-MET-01333-002")
        self.assertEqual(pairs["BraTSMET_001266"], "BraTS-MET-01341-002")

    def test_slurm_is_cpu_evaluation_only(self) -> None:
        text = (ROOT / "slurm/01_evaluate_only_table7.slurm").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --partition=cpu_192G", text)
        self.assertNotIn("#SBATCH --gres", text)
        self.assertNotIn("frozen_s2_inference", text)
        self.assertNotIn("nnUNetv2_predict", text)

    def test_r2_job_id_comes_from_started_evidence(self) -> None:
        builder = (ROOT / "runtime/build_case_views.py").read_text(encoding="utf-8")
        validator = (ROOT / "runtime/validate_run.py").read_text(encoding="utf-8")
        self.assertNotIn('inference["job_id"]', builder)
        self.assertNotIn('r2_inference["job_id"]', validator)
        self.assertIn('started["job_id"]', builder)
        self.assertIn('r2_started["job_id"]', validator)


if __name__ == "__main__":
    unittest.main()
