import sys
import unittest
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from metrics import calibration,components,cube,foreground_ssim,halo_slices,holm,measure,paired,stream_seed,tile_starts


class Shared(unittest.TestCase):
    def test_identity(self):
        a=np.random.RandomState(2).normal(size=(13,14,15));m=np.zeros(a.shape,bool);m[4:9,4:9,4:9]=True
        self.assertAlmostEqual(foreground_ssim(a,a,m),1.)

    def test_background_inflation(self):
        rng=np.random.RandomState(3);a=rng.normal(size=(40,40,40));b=rng.normal(size=a.shape);m=np.zeros(a.shape,bool)
        m[5:8,5:8,5:8]=True;m[30:33,30:33,30:33]=True
        zero=structural_similarity(a*m,b*m,win_size=7,data_range=6,use_sample_covariance=False)
        score=foreground_ssim(a,b,m)
        self.assertGreater(zero,.9);self.assertLess(score,.2)

    def test_context_not_replaced(self):
        a=np.ones((15,15,15));b=a.copy();mask=np.zeros(a.shape,bool);mask[7,7,7]=True;b[~mask]=-2
        self.assertLess(foreground_ssim(a,b,mask),.5)

    def test_cover_thin_edge_large(self):
        for shape in ((8,11,9),(75,86,94),(128,121,119)):
            label=np.zeros(shape,np.uint8);label[0,0,0]=4;label[3:-2,4:-2,5:-2]=3
            ids,cc=components(label);starts=tile_starts(cc,shape);covered=np.zeros(shape,bool)
            self.assertEqual(len(starts),len(set(starts)))
            for start in starts:
                a,sl,content=cube(label,start);self.assertEqual(a.shape,(64,64,64));self.assertTrue(np.array_equal(a[content],label[sl]));covered[sl]=True
            for comp in cc:self.assertTrue(covered[halo_slices(comp,shape)].all())
            self.assertTrue(np.isfinite(measure(np.zeros(shape),np.ones(shape),label,ids,cc,covered)['ssim']))

    def test_fixed_reference_calibration(self):
        ref=np.arange(24,dtype=np.float32).reshape(2,3,4);mean,std=calibration(ref)
        np.testing.assert_allclose((ref-mean)/std*std+mean,ref,atol=1e-5)
        self.assertEqual(calibration(np.ones((2,2,2))),(1.,1.))

    def test_rng_pair_and_tile(self):
        self.assertEqual(stream_seed('case',(1,2,3),'t1c',20261013),stream_seed('case',(1,2,3),'t1c',20261013))
        self.assertNotEqual(stream_seed('case',(1,2,3),'t1c',20261013),stream_seed('case',(2,2,3),'t1c',20261013))

    def test_bootstrap_and_holm(self):
        p=paired([2,2,2],[1,1,1],['a','a','b']);self.assertEqual(p['paired_ci95'],[1.,1.]);self.assertEqual(p['patient_groups'],2)
        self.assertEqual(paired([1,1],[1,1],['a','b'])['r9_wtl'],[0,2,0])
        self.assertEqual(holm({'a':.01,'b':.02,'c':.8,'d':.9}),{'a':.04,'b':.06,'c':1.,'d':1.})

    def test_nan_and_bad_label(self):
        with self.assertRaises(ValueError):foreground_ssim(np.full((8,8,8),np.nan),np.zeros((8,8,8)),np.ones((8,8,8),bool))
        with self.assertRaises(ValueError):components(np.full((8,8,8),5))

    def test_regions_use_foreground_map(self):
        label=np.zeros((15,15,15),np.uint8);label[6:9,6:9,6:9]=4
        ids,cc=components(label);a=np.ones(label.shape);b=a.copy();b[label==0]=-2
        result=measure(a,b,label,ids,cc,np.ones(label.shape,bool))
        self.assertEqual(result['ssim'],result['regions']['RC']['ssim'])
        self.assertIsNone(result['regions']['ET']['ssim'])
        self.assertLess(result['ssim'],.5)


if __name__=='__main__':unittest.main()
