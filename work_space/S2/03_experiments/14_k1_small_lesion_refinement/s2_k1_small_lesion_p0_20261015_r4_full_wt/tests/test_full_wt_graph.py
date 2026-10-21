import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from full_wt_transport import classify_sources


def row(identifier, small, destinations):
    return dict(stable_reference_id=identifier, small=small, destination_components=destinations,
        outside_crop=False,standalone_training_voxels=1,training_region_voxels=1,
        restored_trace_voxels=1,detected=False)


class AllSources(unittest.TestCase):
    def test_small_large_merge_before_filter(self):
        rows,degree=classify_sources([row('small',True,[1]),row('large',False,[1])])
        self.assertEqual(degree[1],2)
        self.assertEqual(rows[0]['transport_category'],'merged')
        self.assertEqual(sum(r['small'] for r in rows),1)

    def test_small_small_merge(self):
        rows,_=classify_sources([row('a',True,[1]),row('b',True,[1])])
        self.assertTrue(all(r['transport_category']=='merged' for r in rows))

    def test_large_fragment_not_requalified(self):
        rows,_=classify_sources([row('large',False,[1,2])])
        self.assertEqual(rows[0]['transport_category'],'split')
        self.assertFalse(rows[0]['small'])

    def test_complex_and_unique(self):
        rows,_=classify_sources([row('a',True,[1,2]),row('b',False,[2])])
        self.assertEqual(rows[0]['transport_category'],'ambiguous_merge_and_split')

    def test_duplicate_identity_rejected(self):
        with self.assertRaises(ValueError):classify_sources([row('a',True,[1]),row('a',False,[2])])

    def test_duplicate_destination_rejected(self):
        with self.assertRaises(ValueError):classify_sources([row('a',True,[1,1])])

    def test_empty_and_preserved_failed_matching(self):
        self.assertEqual(classify_sources([]),([],{}))
        rows,_=classify_sources([row('a',True,[1])])
        self.assertEqual(rows[0]['transport_category'],'survived_roundtrip_match_failed')


if __name__=='__main__':unittest.main()
