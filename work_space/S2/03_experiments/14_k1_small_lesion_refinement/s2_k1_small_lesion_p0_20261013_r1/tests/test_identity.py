import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "runtime"))
from official_identity import stable_id


class IdentityTests(unittest.TestCase):
    def identity(self, coords, region="wt", sha="a"*64):
        return stable_id("case", region, sha, dict(shape=[8,8,8], spacing=[.5]*3), coords)

    def test_coordinate_order_invariant(self):
        self.assertEqual(self.identity([[1,2,3],[4,5,6]]),self.identity([[4,5,6],[1,2,3]]))

    def test_equal_volume_not_identity(self):
        self.assertNotEqual(self.identity([[1,2,3]]),self.identity([[1,2,4]]))

    def test_region_identity(self):
        self.assertNotEqual(self.identity([[1,2,3]]),self.identity([[1,2,3]],"rc"))

    def test_source_sha_identity(self):
        self.assertNotEqual(self.identity([[1,2,3]]),self.identity([[1,2,3]],sha="b"*64))

    def test_empty_or_duplicate_rejected(self):
        for coords in ([],[[1,2,3],[1,2,3]]):
            with self.assertRaises(RuntimeError):
                self.identity(coords)


if __name__ == "__main__":
    unittest.main()
