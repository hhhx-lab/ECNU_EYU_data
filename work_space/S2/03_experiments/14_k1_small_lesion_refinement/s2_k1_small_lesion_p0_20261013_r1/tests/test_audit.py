import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("audit", Path(__file__).parents[1] / "runtime/audit_existing.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def test_boundary(self):
        refs = [dict(volume=26, is_matched=1, sq_dsc=.2), dict(volume=27, is_matched=1, sq_dsc=.1), dict(volume=1, is_matched=0)]
        self.assertEqual(audit.count_region(dict(reference_instances=refs, n_ref_instances=3, fp=2)), (1, 1, 2))

    def test_undefined_match(self):
        for value in (None, float("nan"), float("inf"), -.1, 1.1):
            with self.assertRaises(ValueError):
                audit.count_region(dict(reference_instances=[dict(volume=1, is_matched=1, sq_dsc=value)], n_ref_instances=1, fp=0))

    def test_duplicate_case(self):
        with self.assertRaises(ValueError):
            audit.unique_cases([dict(id="a"), dict(id="a")], "id")

    def test_missing_id(self):
        with self.assertRaises(KeyError):
            audit.unique_cases([{}], "id")

    def test_count_mismatch(self):
        with self.assertRaises(ValueError):
            audit.count_region(dict(reference_instances=[], n_ref_instances=1, fp=0))


if __name__ == "__main__":
    unittest.main()
