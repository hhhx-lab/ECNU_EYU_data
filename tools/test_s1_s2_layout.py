"""Structural regression checks without executing training or server scripts."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "99_archive/s1_s2_reorganization_20260919_r3"
REPOS = [ROOT / f"work_space/{name}/01_code/repository" for name in ("S1", "S2")]


class RefinedLayoutTests(unittest.TestCase):
    def test_scripts_have_one_unambiguous_responsibility_group(self):
        for repo, count in zip(REPOS, (6, 51)):
            scripts = [p for p in (repo / "scripts").rglob("*") if p.suffix in {".py", ".sh"}]
            self.assertEqual(len(scripts), count)
            self.assertEqual(len({p.name for p in scripts}), count)
            self.assertTrue(all(p.parent.parent == repo / "scripts" for p in scripts))

    def test_script_bootstraps_still_point_to_repository(self):
        expected = ast.dump(ast.parse("Path(__file__).resolve().parents[2]", mode="eval").body)
        count = 0
        for repo in REPOS:
            for path in (repo / "scripts").rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in tree.body:
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"REPO_ROOT", "REPOSITORY_ROOT"} for t in node.targets):
                        self.assertEqual(ast.dump(node.value), expected, str(path))
                        self.assertEqual(path.parents[2], repo)
                        count += 1
        self.assertGreater(count, 25)

    def test_literal_repository_script_references_resolve(self):
        def parts(node):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                left = parts(node.left)
                if left is not None and isinstance(node.right, ast.Constant) and isinstance(node.right.value, str):
                    return left + [node.right.value]
            if isinstance(node, ast.Name) and node.id in {"REPOSITORY_ROOT", "REPO_ROOT", "repository"}:
                return []
            return None
        for repo in REPOS:
            for path in repo.rglob("*.py"):
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                    values = parts(node)
                    if values and (values[0] == "scripts" or values[0].startswith("scripts/")):
                        self.assertTrue(repo.joinpath(*values).exists(), f"{path}: {values}")

    def test_nnunet_checkpoint_import_package_is_unchanged(self):
        records = json.loads((AUDIT / "BASELINE.json").read_text())["files"]
        selected = [r for r in records if "/repository/custom_nnunet/" in r["path"]]
        self.assertEqual(len(selected), 30)
        for item in selected:
            self.assertEqual(hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest(), item["sha256"])

    def test_fixed_split_preserves_original_bytes_and_shell_override(self):
        moves = json.loads((AUDIT / "PLAN.json").read_text())["moves"]
        baseline = {r["path"]: r for r in json.loads((AUDIT / "BASELINE.json").read_text())["files"]}
        selected = [m for m in moves if "/repository/data/splits/" in m["old"]]
        self.assertEqual(len(selected), 5)
        for move in selected:
            self.assertEqual(hashlib.sha256((ROOT / move["new"]).read_bytes()).hexdigest(), baseline[move["old"]]["sha256"])
        for name in ("train.sh", "infer.sh"):
            text = (REPOS[1] / name).read_text()
            self.assertIn("${BRATS_SPLIT_DIR:-${REPO_DIR}/../../02_datasets/undated_fixed_split_contracts/", text)

    def test_no_legacy_stubs_in_active_repository(self):
        for name in ("train.py", "inference.py", "models", "inference", "configs", "data"):
            self.assertFalse((REPOS[1] / name).exists(), name)
        for repo in REPOS:
            self.assertFalse(list(repo.rglob("*.bak*")))
            self.assertFalse(list(repo.rglob("*_old.py")))
        self.assertFalse((REPOS[0] / "trainers/trainer_v1_final_before_dice.py").exists())

    def test_shell_syntax_without_execution(self):
        for module in (ROOT / "work_space/S1", ROOT / "work_space/S2"):
            for category in ("01_code", "06_operations"):
                for path in (module / category).rglob("*"):
                    if path.suffix in {".sh", ".slurm"}:
                        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, f"{path}: {result.stderr}")

    def test_authored_manual_versions_are_both_retained(self):
        directory = ROOT / "work_space/S1/05_documents/03_operations"
        first = (directory / "S1_服务器运行手册.md").read_bytes()
        second = (directory / "S1_服务器运行手册_原代码文档版.md").read_bytes()
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
