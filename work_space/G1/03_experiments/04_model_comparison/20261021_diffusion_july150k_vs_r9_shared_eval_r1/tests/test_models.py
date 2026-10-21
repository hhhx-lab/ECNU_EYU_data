"""Trusted local eight-checkpoint compatibility; remote audit repeats real loading."""
import json
import os
import sys
import unittest
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
import common


class Models(unittest.TestCase):
    @unittest.skipUnless(os.getenv('SHARED_LOCAL_MODEL_TEST')=='1','Local checkpoint paths not present on remote; CPU audit loads all eight there')
    def test_all_eight_strict(self):
        project=ROOT.parents[4]
        original=common.cfg
        c=original()
        local=json.loads((ROOT/'LOCAL_MODELS.json').read_text())
        c['training_root']=str(project/'work_space/G1/03_experiments/03_diffusion_training/20261016_vae_epoch004_1mm_resume_uniform_r9')
        c['reference_root']=str(project/'work_space/G1/03_experiments/03_diffusion_training/20261018_r9_fixedval_selection_r1')
        common.cfg=lambda:c
        torch.set_num_threads(1)
        try:
            for version in ('july150k','r9'):
                for mod in common.MODS:
                    with self.subTest(version=version,modality=mod):
                        info=local[version][mod]
                        model=common.load_model(info,version)
                        with torch.inference_mode():
                            pred=model(torch.zeros(1,1,64,64,64),torch.zeros(1),torch.zeros(1,4,64,64,64))
                        self.assertEqual(tuple(pred.shape),(1,1,64,64,64))
                        self.assertTrue(torch.isfinite(pred).all())
                        del model
        finally:
            common.cfg=original


if __name__=='__main__':unittest.main()
