"""Independent numerical oracles and denominator/axis regression checks."""
import copy
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
import metrics as m
from evaluate import components, generate, measure, multichannel
from common import checkpoint, model_from, sha
R9=Path(os.environ.get('R9_TEST_ROOT',str(ROOT.parent/'20261016_vae_epoch004_1mm_resume_uniform_r9')))


def real_paths():
    if (R9/'training/t1c/RUN_CONTRACT.json').exists():
        return R9/'training/t1c/RUN_CONTRACT.json',R9/'training/t1c/weights/diffusion_200000.pt'
    return R9/'operator/pull_final_20261018T065725Z/lightweight/training/t1c/RUN_CONTRACT.json',R9/'final_models_200000/t1c/diffusion_200000.pt'


class MetricsTests(unittest.TestCase):
    def test_xyz_asymmetric_bbox(self):
        a=np.zeros((17,23,31));a[2:5,8:12,21:27]=4
        sl=m.bbox_slices((2,5,8,12,21,27),a.shape)
        self.assertEqual(a[sl].shape,(3,4,6));self.assertEqual(a[sl].sum(),288)

    def test_invalid_bbox(self):
        for b in ((-1,2,0,1,0,1),(0,18,0,1,0,1),(1,1,0,1,0,1)):
            with self.assertRaises(ValueError):m.bbox_slices(b,(17,23,31))

    def test_crop64_edge_preserves_xyz(self):
        a=np.arange(17*23*31).reshape(17,23,31)
        cube,sl,content=m.crop64(a,(2,5,8,12,21,27))
        self.assertEqual(cube.shape,(64,64,64));np.testing.assert_array_equal(cube[content],a[sl])

    def test_crop64_interior_noresize(self):
        a=np.zeros((101,112,123));a[40,50,70]=4
        cube,sl,content=m.crop64(a,(40,41,50,51,70,71))
        self.assertEqual(cube.sum(),4);self.assertEqual(cube[31,31,31],4)

    def test_oversized_reject(self):
        with self.assertRaises(ValueError):m.crop64(np.zeros((100,100,100)),(0,65,0,64,0,64))

    def test_multichannel_rc_xyz(self):
        a=np.zeros((9,11,13),np.uint8);a[2,7,10]=4;a[4,1,2]=3
        b=multichannel(a);self.assertEqual(b[3,2,7,10],1);self.assertEqual(b[0,4,1,2],1)
        self.assertEqual(b[:,2,7,10].sum(),1)

    def test_components_unique_weight(self):
        a=np.zeros((20,30,40),np.uint8);a[1,8,30]=4;a[10:13,2:4,3:7]=2
        ids,cc=components(a);self.assertEqual(len(cc),2);self.assertEqual(sum(c['n_voxels'] for c in cc),25)
        self.assertEqual(cc[0]['bbox'],(1,2,8,9,30,31))

    def test_ssim_identity(self):
        a=np.random.RandomState(3).normal(size=(9,11,13))
        self.assertGreater(m.ssim_xyz(a,a),.999999)

    def test_ssim_single_voxel_population_oracle(self):
        a=np.array([[[2.]]]);b=np.array([[[3.]]]);c1=.06**2;c2=.18**2
        expected=(12+c1)*c2/((13+c1)*c2+1e-8)
        self.assertAlmostEqual(m.ssim_xyz(a,b),expected,places=12)

    def test_ssim_gaussian_independent_oracle(self):
        a=np.random.RandomState(9).normal(size=(9,11,13));b=a*.8+.1
        vals=[];c1=.06**2;c2=.18**2
        for z in range(13):
            x,y=a[:,:,z],b[:,:,z]
            conv=lambda v:gaussian_filter(v,1.5,truncate=2.,mode='constant')[3:-3,3:-3]
            u,v=conv(x),conv(y);va,vb,cov=conv(x*x)-u*u,conv(y*y)-v*v,conv(x*y)-u*v
            vals.append(np.mean(((2*u*v+c1)*(2*cov+c2))/((u*u+v*v+c1)*(va+vb+c2)+1e-8)))
        self.assertAlmostEqual(m.ssim_xyz(a,b),float(np.mean(vals)),places=12)

    def test_nonfinite_metrics_reject(self):
        with self.assertRaises(ValueError):m.ssim_xyz(np.ones((2,3,4))*np.nan,np.zeros((2,3,4)))

    def test_masked_empty_na_perfect_inf(self):
        a=np.zeros((2,3,4));self.assertEqual(m.masked_errors(a,a,a>0)['reason'],'empty_region')
        self.assertEqual(m.masked_errors(a,a,np.ones(a.shape,bool))['reason'],'perfect_match_infinite_psnr')

    def test_volume_weight_case_equal(self):
        self.assertAlmostEqual(m.weighted_lesions([dict(ssim=.2,n_voxels=1),dict(ssim=.8,n_voxels=3)]),.65)
        self.assertAlmostEqual(np.mean([.65,.1]),.375)

    def test_empty_weight_reject(self):
        with self.assertRaises(ValueError):m.weighted_lesions([])

    def test_tie_earliest_relative_global_max(self):
        scores={s:.1 for s in m.STEPS};scores[20000]=.9999988;scores[40000]=.9999994;scores[60000]=1.
        self.assertEqual(m.choose_step(scores),40000)

    def test_missing_candidate_reject(self):
        with self.assertRaises(ValueError):m.choose_step({200000:1.})

    def test_seed_stable_order_and_step_absent(self):
        self.assertEqual(m.stream_seed('a','b','t1c',20261013),m.stream_seed('a','b','t1c',20261013))
        self.assertNotEqual(m.stream_seed('a','b','t1c',20261013),m.stream_seed('a','b','t1n',20261013))

    def test_cluster_bootstrap_reproducible_case_mean(self):
        a=m.cluster_bootstrap([0,1,1],['a','a','b'],[0,0,0]);b=m.cluster_bootstrap([0,1,1],['a','a','b'],[0,0,0])
        self.assertEqual(a,b);self.assertAlmostEqual(a['mean'],2/3);self.assertEqual(a['patient_groups'],2)

    def test_pair_denominator_reject(self):
        with self.assertRaises(ValueError):m.cluster_bootstrap([1,2],['a','b'],[1])

    def test_support_transaction(self):
        a=np.ones((8,9,10));label=np.zeros(a.shape,np.uint8);label[2,4,7]=4
        ids,cc=components(label);r=measure(a,np.zeros(a.shape),label,ids,cc,'test')
        self.assertTrue(r['support_outside_unchanged']);self.assertEqual(r['mse'],1)

    def test_generation_tile_blend_coverage(self):
        label=np.zeros((89,19,21),np.uint8);label[2:80,8:11,12:15]=4
        rows=[dict(lesion_id=str(i),x_extreme_min=x,x_extreme_max=min(x+64,80),
          y_extreme_min=8,y_extreme_max=11,z_extreme_min=12,z_extreme_max=15) for i,x in enumerate((2,58))]
        class EDM:
            @staticmethod
            def sample_edm(*args,**kwargs):return torch.ones((1,1,64,64,64))
        out=generate(None,EDM,None,label,rows,'case','t1c',20261013,'cpu')
        np.testing.assert_allclose(out[label>0],1)

    def test_generation_uncovered_reject(self):
        label=np.zeros((89,19,21),np.uint8);label[70,8,12]=4
        with self.assertRaises(AssertionError):generate(None,None,None,label,[],'a','t1c',20261013,'cpu')

    def test_real_final_checkpoint_contract(self):
        cp,p=real_paths();c=json.loads(cp.read_text())
        self.assertEqual(sha(p),'cc1eb4e2ae6b0607dc9d1235ca01ac4745e858937d1e4af49a68e355d32686e0')
        v=checkpoint(p,c,200000,'t1c');sys.path.insert(0,str(R9/'runtime/upstream_source'))
        model_from(v,'cpu')

    def test_wrong_step_checkpoint_reject(self):
        cp,p=real_paths();c=json.loads(cp.read_text())
        with self.assertRaises(AssertionError):checkpoint(p,c,180000,'t1c')

    def test_checkpoint_wrong_modality_reject(self):
        cp,p=real_paths();c=json.loads(cp.read_text())
        with self.assertRaises(AssertionError):checkpoint(p,c,200000,'t2f')

    def test_checkpoint_contract_drift_reject(self):
        cp,p=real_paths();c=json.loads(cp.read_text());c['config']['crop_size']=32
        with self.assertRaises(AssertionError):checkpoint(p,c,200000,'t1c')

    def test_secondary_identical_generated_transaction(self):
        recon=np.ones((8,9,10));real=np.zeros_like(recon);label=np.zeros(recon.shape,np.uint8)
        label[2,4,7]=4;ids,cc=components(label)
        r=measure(real,np.zeros_like(recon),label,ids,cc,'secondary',background=recon)
        self.assertTrue(r['support_outside_unchanged'])


if __name__=='__main__':unittest.main()
