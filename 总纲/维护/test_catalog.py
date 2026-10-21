"""Exercise catalog coverage and privacy without touching project artifacts."""
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("catalog", Path(__file__).with_name("catalog.py"))
catalog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog)


class CatalogTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / "总纲"
        (self.home / "维护").mkdir(parents=True)
        (self.root / "data").mkdir()
        (self.root / "data/evidence.md").write_text("# Evidence\n", encoding="utf-8")
        (self.root / "data/code.py").write_text("def train():\n    pass\n", encoding="utf-8")
        (self.root / "data/table.csv").write_text("case,metric\na,1\nb,2\n", encoding="utf-8")
        (self.root / "data/broken.json").write_text("{", encoding="utf-8")
        (self.root / ".env").write_text("PASSWORD=never_export_this_value\n", encoding="utf-8")
        (self.root / "external_link").symlink_to("/unavailable/external/data")
        self.config = {"reviewed_at": "2026-09-19T00:00:00Z", "units": [
            {"path": "data", "description": "Test scientific unit", "kind": "experiment", "state": "unknown",
             "evidence": ["data/evidence.md"]}]}
        catalog.dump(self.home / "维护/知识登记.json", self.config)
        (self.home / "README.md").write_text("# Test\n[Current](当前索引.md)\n", encoding="utf-8")

    def build(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return catalog.build(self.root, self.home, "test")

    def snapshot(self):
        pointer = json.loads((self.home / "当前索引.json").read_text())
        return self.root / pointer["snapshot"]

    def test_complete_coverage_and_current_index(self):
        self.assertEqual(self.build()["status"], "pass")
        self.assertEqual(catalog.verify(self.root, self.home)["status"], "pass")
        with (self.snapshot() / "全部文件.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = {r["path"]: r for r in csv.DictReader(stream)}
        self.assertIn("external_link", rows)
        self.assertEqual(rows["external_link"]["kind"], "symlink")
        self.assertIn("train", rows["data/code.py"]["structure"])
        self.assertIn("数据行=2", rows["data/table.csv"]["structure"])
        self.assertEqual(rows["data/broken.json"]["inspection"], "parse_error")
        self.assertTrue((self.home / "当前索引.md").exists())

    def test_secrets_are_metadata_only(self):
        self.build()
        text = (self.snapshot() / "全部文件.csv").read_text(encoding="utf-8-sig")
        self.assertNotIn("never_export_this_value", text)
        with (self.snapshot() / "全部文件.csv").open(encoding="utf-8-sig", newline="") as stream:
            row = next(r for r in csv.DictReader(stream) if r["path"] == ".env")
        self.assertEqual(row["sha256"], "")
        self.assertEqual(row["structure"], "")
        self.assertTrue(catalog.restricted("archive/.git/config"))
        self.assertTrue(catalog.restricted("archive/server.pem"))

    def test_changes_and_new_files_are_detected(self):
        self.build()
        (self.root / "data/code.py").write_text("def other(): pass\n", encoding="utf-8")
        (self.root / "data/new.txt").write_text("new\n", encoding="utf-8")
        check = catalog.verify(self.root, self.home)
        self.assertEqual(check["status"], "fail")
        self.assertIn("data/code.py", check["changed_since_snapshot"])
        self.assertIn("data/new.txt", check["missing_from_index"])

    def test_old_snapshot_preserved_and_self_reference_stable(self):
        self.build()
        old = self.snapshot()
        old_hash = catalog.digest(old / "全部文件.csv")
        self.assertEqual(self.build()["status"], "pass")
        self.assertNotEqual(old, self.snapshot())
        self.assertEqual(catalog.digest(old / "全部文件.csv"), old_hash)
        self.assertEqual(catalog.verify(self.root, self.home)["status"], "pass")

    def test_evidence_change_requires_human_review(self):
        self.build()
        (self.root / "data/evidence.md").write_text("# Changed evidence\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Curated evidence changed"):
            self.build()
        self.config["reviewed_at"] = "2026-09-20T00:00:00Z"
        catalog.dump(self.home / "维护/知识登记.json", self.config)
        self.assertEqual(self.build()["status"], "pass")

    def test_broken_links_stop_build(self):
        (self.home / "README.md").write_text("[missing](../does-not-exist.md)\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Broken curated links"):
            self.build()

    def test_no_evidence_context_is_supported(self):
        self.config["units"][0]["evidence"] = []
        catalog.dump(self.home / "维护/知识登记.json", self.config)
        self.assertEqual(self.build()["status"], "pass")

    def test_unregistered_experiment_blocks_pass(self):
        run = self.root / "work_space/S2/03_experiments/12_new/run_new"
        run.mkdir(parents=True)
        (run / "evidence.txt").write_text("unreviewed\n", encoding="utf-8")
        result = self.build()
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["unregistered_G1_S2_experiment_units"], [run.relative_to(self.root).as_posix()])

    def test_source_is_not_modified(self):
        before = {p.relative_to(self.root): (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in (self.root / "data").iterdir()}
        self.build()
        for p, values in before.items():
            self.assertEqual(((self.root / p).read_bytes(), (self.root / p).stat().st_mtime_ns), values)


if __name__ == "__main__":
    unittest.main()
