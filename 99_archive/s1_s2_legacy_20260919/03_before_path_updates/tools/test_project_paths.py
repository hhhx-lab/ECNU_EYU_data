"""Read-only checks for layout adapters; no external datasets or ML libraries."""
import ast
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.project_paths import ROOT, get_path, resolve_legacy_path


class LayoutTests(unittest.TestCase):
    def test_configured_local_assets_exist(self):
        for key in ["G1_CODE_ROOT", "G1_DATA_ROOT", "G1_WEIGHTS_ROOT", "G1_VAE_WEIGHTS",
                    "G2_RELEASE_ROOT", "S1_CODE_ROOT", "S2_CODE_ROOT", "OFFICIAL_MET_CODE_ROOT"]:
            self.assertTrue(get_path(key).exists(), key)

    def test_environment_precedes_env_file(self):
        with patch.dict(os.environ, {"G2_RELEASE_ROOT": "override_not_created"}):
            self.assertEqual(get_path("G2_RELEASE_ROOT"), ROOT / "override_not_created")

    def test_old_and_intermediate_paths_resolve(self):
        expected = get_path("G2_RELEASE_ROOT") / "splits/splits_final_train_val_test.json"
        for old in ["work_space/G2/results/splits/splits_final_train_val_test.json",
                    "work_space/G2/30_results/splits/splits_final_train_val_test.json"]:
            self.assertEqual(resolve_legacy_path(old), expected)

    def test_all_moved_original_files_resolve(self):
        plan = json.loads((ROOT / "99_archive/reorganization_20260919_r2/FINAL_PLAN_LABEL_CORRECTED.json").read_text())
        for f in plan["files"]:
            self.assertEqual(resolve_legacy_path(f["old"]), ROOT / f["new"], f["old"])

    def test_remote_paths_are_not_rewritten(self):
        path = Path("/public/home/zqchen/projects/ECNU_EYU_data/work_space/G2/results")
        self.assertEqual(resolve_legacy_path(path), path)

    def test_new_output_respects_anchor(self):
        anchor = ROOT / "work_space/G1"
        self.assertEqual(resolve_legacy_path("uncreated_output.bin", anchor), anchor / "uncreated_output.bin")

    def test_repository_ancestry(self):
        for name in ["S1", "S2"]:
            self.assertEqual(get_path(name + "_CODE_ROOT").parents[3], ROOT)

    def test_adapted_source_syntax(self):
        source = ROOT / "99_archive/reorganization_20260919_r2/SOURCE_ADAPTATIONS.json"
        for record in json.loads(source.read_text())["files"]:
            ast.parse((ROOT / record["active"]).read_text(encoding="utf-8"))

    def test_vae_configuration_without_ml_imports(self):
        code = get_path("G1_CODE_ROOT")
        with patch.object(sys, "path", [str(code), *sys.path]):
            spec = importlib.util.spec_from_file_location("g1_config_layout_test", code / "configs.py")
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            self.assertEqual(Path(config.PATH_NAME_WEIGHTS_VAE), get_path("G1_VAE_WEIGHTS"))
            self.assertEqual(Path(config.PATH_DATA), get_path("G1_DATA_ROOT"))


if __name__ == "__main__":
    unittest.main()
