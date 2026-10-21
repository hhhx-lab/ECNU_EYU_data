"""Test the independent P1/P3 space budget, including locked-test exclusion."""

import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('storage_estimate',Path(__file__).resolve().parents[1]/'operator/estimate_p1_storage.py')
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def records():
    for case in range(926):
        for modality in ('t1n','t1c','t2w','t2f','seg'):
            yield {'case_id':str(case),'split':'train' if case<823 else 'val','modality':modality,
                   'status':'pass','geometry':{'shape_xyz':[7,9,11],'spacing_xyz':[1,0.8,1.4]}}
    yield {'case_id':'locked','split':'test','status':'fail'}


class StorageEstimateTests(unittest.TestCase):
    def test_reserves_dataset_temporary_and_all_modalities(self):
        value=v.estimate(records(),53712244)
        self.assertEqual(value['cases'],926)
        self.assertEqual(value['full_fov_1mm_voxels'],926*8*9*17)
        self.assertEqual(value['preprocessing_temporary_reserve_bytes'],value['derived_dataset_upper_bound_bytes'])
        self.assertGreater(value['total_required_bytes'],2*value['derived_dataset_upper_bound_bytes'])
        self.assertEqual(value['four_modality_checkpoint_reserve_bytes'],4*49*(53712244+8*1024**2))

    def test_missing_modality_rejected(self):
        with self.assertRaisesRegex(ValueError,'incomplete modality'):
            v.estimate((r for r in records() if r.get('modality')!='seg'),53712244)

    def test_incomplete_cohort_rejected(self):
        with self.assertRaisesRegex(ValueError,'926'):
            v.estimate([],53712244)

    def test_duplicate_t1n_rejected(self):
        values=list(records())
        values.append(values[0])
        with self.assertRaisesRegex(ValueError,'duplicate'):
            v.estimate(values,53712244)


if __name__=='__main__':
    unittest.main()
