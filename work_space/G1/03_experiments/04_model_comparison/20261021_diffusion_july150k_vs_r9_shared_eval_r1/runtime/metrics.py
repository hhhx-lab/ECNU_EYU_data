"""Foreground-center SSIM on genuine generated context, and paired statistics."""
import hashlib
import itertools
import json
import math
import numpy as np
from scipy import ndimage
from scipy.stats import binomtest
from skimage.metrics import structural_similarity


def stream_seed(case, start, modality, seed):
    payload = json.dumps([case, list(start), modality, seed], separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], 'big') % (2**63-1)


def components(label):
    if label.ndim != 3 or not np.isin(label, [0,1,2,3,4]).all():
        raise ValueError('Invalid label')
    ids, n = ndimage.label(label > 0, structure=np.ones((3,3,3)))
    cc = []
    for i, sl in enumerate(ndimage.find_objects(ids), 1):
        cc.append(dict(id=i, bbox=[v for s in sl for v in (s.start, s.stop)],
                       n_voxels=int(np.count_nonzero(ids[sl] == i))))
    return ids, cc


def halo_slices(comp, shape, halo=5):
    return tuple(slice(max(0, comp['bbox'][2*i]-halo), min(shape[i], comp['bbox'][2*i+1]+halo)) for i in range(3))


def tile_starts(cc, shape):
    starts = set()
    for comp in cc:
        axes = []
        sl = halo_slices(comp, shape)
        for s, size in zip(sl, shape):
            if s.stop-s.start <= 64:
                mid = (s.start+s.stop)//2
                axes.append([min(max(0, mid-32), size-64) if size >= 64 else (size-64)//2])
            else:
                first = min(s.start, size-64)
                last = min(s.stop-64, size-64)
                positions = list(range(first, last+1, 32))
                if positions[-1] != last:
                    positions.append(last)
                axes.append(positions)
        starts.update(itertools.product(*axes))
    return sorted(starts)


def cube(array, start):
    sl = tuple(slice(max(0, s), min(n, s+64)) for s,n in zip(start, array.shape[-3:]))
    pads = tuple((max(0,-s), max(0,s+64-n)) for s,n in zip(start, array.shape[-3:]))
    a = np.pad(array[(...,)+sl], [(0,0)]*(array.ndim-3)+list(pads), constant_values=0)
    content = tuple(slice(p[0],64-p[1]) for p in pads)
    assert a.shape[-3:] == (64,64,64)
    return a, sl, content


def calibration(real_content):
    assert real_content.size > 1 and np.isfinite(real_content).all()
    mean, std = float(real_content.mean()), float(real_content.std(ddof=1))
    return mean, std if std >= 1e-8 else 1.


def ssim_map(reference, generated):
    if reference.shape != generated.shape or reference.ndim != 3:
        raise ValueError('Invalid SSIM geometry')
    if not np.isfinite(reference).all() or not np.isfinite(generated).all():
        raise ValueError('Nonfinite SSIM input')
    # External reflect padding supplies the same real image-edge convention to both.
    a, b = np.pad(reference.astype(np.float64),3,mode='reflect'), np.pad(generated.astype(np.float64),3,mode='reflect')
    _, score = structural_similarity(a,b,win_size=7,data_range=6.,full=True,
        gaussian_weights=False,use_sample_covariance=False,K1=.01,K2=.03)
    return score[3:-3,3:-3,3:-3]


def foreground_ssim(reference, generated, mask):
    if mask.shape != reference.shape or not mask.any():
        raise ValueError('Invalid SSIM denominator')
    return float(ssim_map(reference, generated)[mask].mean())


def errors(a, b, mask):
    if not mask.any():
        return dict(mae=None,rmse=None,psnr=None,reason='empty_region')
    x, y = a[mask].astype(np.float64), b[mask].astype(np.float64)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    d = y-x
    mse = float(np.mean(d*d))
    return dict(mae=float(np.mean(np.abs(d))),rmse=math.sqrt(mse),
                psnr=10*math.log10(36/mse) if mse else None,
                reason=None if mse else 'perfect_match_infinite_psnr')


def measure(reference, generated, label, ids, cc, covered):
    result = []
    regions = [('NETC',1),('SNFH',2),('ET',3),('RC',4)]
    region_scores = {name:[] for name,_ in regions}
    for comp in cc:
        sl = halo_slices(comp,label.shape)
        assert covered[sl].all(), 'Generated-context coverage gap'
        mask = ids[sl] == comp['id']
        a,b = reference[sl],generated[sl]
        scores = ssim_map(a,b)
        score = float(scores[mask].mean())
        for name,value in regions:
            rm = mask & (label[sl] == value)
            if rm.any():region_scores[name].append((float(scores[rm].sum()),int(rm.sum())))
        result.append(dict(**comp,ssim=score))
    assert result
    score = float(np.average([r['ssim'] for r in result],weights=[r['n_voxels'] for r in result]))
    foreground = label>0
    out = dict(ssim=score,lesions=result,**errors(reference,generated,foreground))
    out['regions'] = {name:dict(errors(reference,generated,label==value),
        ssim=sum(s for s,n in region_scores[name])/sum(n for s,n in region_scores[name]) if region_scores[name] else None)
        for name,value in regions}
    out['size_ssim'] = {}
    for name, records in [('small_le27',[r for r in result if r['n_voxels']<=27]),('large_gt27',[r for r in result if r['n_voxels']>27])]:
        out['size_ssim'][name] = float(np.average([r['ssim'] for r in records],weights=[r['n_voxels'] for r in records])) if records else None
    y=generated[foreground]
    out['extreme_fraction'] = float(np.mean(np.abs(y)>8))
    out['zero_fraction'] = float(np.mean(np.isclose(y,0,atol=1e-7)))
    ring=ndimage.binary_dilation(foreground,iterations=3)&~foreground&covered
    out['contrast_abs_error']=float(abs((generated[foreground].mean()-generated[ring].mean())-(reference[foreground].mean()-reference[ring].mean()))) if ring.any() else None
    return out


def paired(a,b,groups,higher=True):
    a,b=np.asarray(a,float),np.asarray(b,float)
    assert a.shape==b.shape and len(a)==len(groups) and len(a)>0 and np.isfinite(a).all() and np.isfinite(b).all()
    names=sorted(set(groups));ix=[np.flatnonzero(np.asarray(groups)==g) for g in names]
    counts=np.array([len(i) for i in ix]);sa=np.array([a[i].sum() for i in ix]);sb=np.array([b[i].sum() for i in ix])
    draws=np.random.RandomState(20261013).randint(0,len(ix),(20000,len(ix)))
    n=counts[draws].sum(1);ma=sa[draws].sum(1)/n;mb=sb[draws].sum(1)/n
    d=(sa-sb)/counts*(1 if higher else -1);w=int((d>1e-6).sum());l=int((d< -1e-6).sum())
    return dict(r9_mean=float(a.mean()),july150k_mean=float(b.mean()),difference_r9_minus_old=float((a-b).mean()),
        r9_ci95=np.quantile(ma,[.025,.975]).tolist(),old_ci95=np.quantile(mb,[.025,.975]).tolist(),
        paired_ci95=np.quantile(ma-mb,[.025,.975]).tolist(),cases=len(a),patient_groups=len(ix),replicates=20000,
        bootstrap_seed=20261013,r9_wtl=[w,len(ix)-w-l,l],higher_is_better=higher,
        group_sign_p_uncorrected=float(binomtest(w,w+l).pvalue) if w+l else 1.)


def holm(p):
    keys=sorted(p,key=p.get);out={};running=0.
    for i,k in enumerate(keys):
        running=max(running,min(1.,(len(keys)-i)*p[k]));out[k]=running
    return out
