import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('rules', Path(__file__).resolve().parents[1]/'runtime/transport_rules.py')
rules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rules)


class Categories(unittest.TestCase):
    def row(self, **changes):
        return dict(outside_crop=False,standalone_training_voxels=8,training_region_voxels=8,
                    restored_trace_voxels=8,destination_components=[1],detected=False,**changes)

    def test_survival_is_not_detection(self):
        r=self.row()
        self.assertEqual(rules.classify(r,{1:1}),'survived_roundtrip_match_failed')
        r['detected']=True
        self.assertEqual(rules.classify(r,{1:1}),'survived_roundtrip_match_passed')

    def test_disappearance_requires_direct_trace(self):
        r=self.row();r['standalone_training_voxels']=0
        self.assertEqual(rules.classify(r,{1:1}),'disappeared_in_label_pipeline')
        r=self.row();r['restored_trace_voxels']=0
        self.assertEqual(rules.classify(r,{1:1}),'disappeared_in_label_pipeline')

    def test_crop_not_downsample_disappearance(self):
        r=self.row();r['outside_crop']=True
        self.assertEqual(rules.classify(r,{1:1}),'cropped_out')

    def test_competition_not_generic_disappearance(self):
        r=self.row();r['training_region_voxels']=0
        self.assertEqual(rules.classify(r,{1:1}),'ambiguous_label_competition')

    def test_merge(self):
        self.assertEqual(rules.classify(self.row(),{1:2}),'merged')

    def test_split(self):
        r=self.row();r['destination_components']=[1,2]
        self.assertEqual(rules.classify(r,{1:1,2:1}),'split')

    def test_ambiguous_complex(self):
        r=self.row();r['destination_components']=[1,2]
        self.assertEqual(rules.classify(r,{1:2,2:1}),'ambiguous_merge_and_split')

    def test_no_destination_not_disappearance(self):
        r=self.row();r['destination_components']=[]
        self.assertEqual(rules.classify(r,{}),'ambiguous_no_destination_overlap')


if __name__=='__main__':
    unittest.main()
