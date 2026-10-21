"""Synthetic tests for the diagnostic, not patient-data acceptance."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from support_audit import coverage


class SupportTests(unittest.TestCase):
    def test_negative_source_is_reported_not_repaired(self):
        a=np.zeros((7,7,7));a[1:6,1:6,1:6]=-2
        label=np.zeros_like(a);label[3,3,3]=4
        r,_,_=coverage(label,[a,a,a])
        self.assertEqual(r['positive_missing_voxels'],1)
        self.assertEqual(r['nonzero_missing_voxels'],0)
        self.assertEqual(r['old_missing_with_negative_source'],1)

    def test_zero_background_label_remains_failure(self):
        a=np.zeros((7,7,7));a[2:5,2:5,2:5]=2
        label=np.zeros_like(a);label[1,2,2]=3
        r,_,_=coverage(label,[a,a,a])
        self.assertEqual(r['nonzero_missing_voxels'],1)
        self.assertEqual(r['old_missing_all_three_zero'],1)
        self.assertEqual(label[1,2,2],3)

    def test_closed_hole_fill_is_original_algorithm(self):
        a=np.zeros((7,7,7));a[1:6,1:6,1:6]=2;a[3,3,3]=0
        label=np.zeros_like(a);label[3,3,3]=1
        r,_,_=coverage(label,[a,a,a])
        self.assertEqual(r['label_all_three_zero'],1)
        self.assertEqual(r['positive_missing_voxels'],0)

    def test_illegal_label_and_nonfinite_stop(self):
        a=np.ones((3,3,3));label=np.zeros_like(a);label[0,0,0]=5
        with self.assertRaises(ValueError):coverage(label,[a,a,a])
        label*=0;a[0,0,0]=np.nan
        with self.assertRaises(ValueError):coverage(label,[a,a,a])

    def test_no_label_no_anomaly(self):
        a=np.ones((3,3,3));a[0,0,0]=0
        r,_,_=coverage(np.zeros_like(a),[a,a,a])
        self.assertEqual(r['positive_missing_voxels'],0)


if __name__=='__main__':unittest.main(verbosity=2)
