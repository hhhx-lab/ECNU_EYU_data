"""Frozen XYZ lesion metrics and case-balanced, patient-cluster selection."""
import hashlib
import json
import math
import numpy as np
from scipy.signal import convolve2d
from scipy.stats import binomtest

STEPS = tuple(range(20000, 200001, 20000))
SEEDS = tuple(range(20261013, 20261017))


def stream_seed(case, crop, modality, seed):
    text = json.dumps([case, crop, modality, seed], separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big') % (2 ** 63 - 1)


def bbox_slices(bbox, shape):
    values = tuple(int(v) for v in bbox)
    if len(values) != 6 or any(not 0 <= values[2 * i] < values[2 * i + 1] <= shape[i] for i in range(3)):
        raise ValueError('Invalid XYZ half-open bbox')
    return tuple(slice(values[2 * i], values[2 * i + 1]) for i in range(3))


def crop64(array, bbox):
    """Deterministic center window matching the no-shift training crop in XYZ."""
    bbox_slices(bbox, array.shape[-3:])
    starts, ends, pads = [], [], []
    for i, size in enumerate(array.shape[-3:]):
        lo, hi = int(bbox[2 * i]), int(bbox[2 * i + 1])
        extent = hi - lo
        if extent > 64:
            raise ValueError('Oversized un-tiled bbox')
        pad = (64 - extent) / 2
        base, top = lo - int(pad), hi + int(pad + .5)
        starts.append(max(0, base)); ends.append(min(size, top))
        pads.append((max(0, -base), max(0, top - size)))
    slices = tuple(slice(lo, hi) for lo, hi in zip(starts, ends))
    cropped = array[(...,) + slices]
    prefix = [(0, 0)] * (array.ndim - 3)
    result = np.pad(cropped, prefix + pads, constant_values=0)
    if result.shape[-3:] != (64, 64, 64):
        raise ValueError('Unexpected crop geometry')
    content = tuple(slice(p[0], 64 - p[1]) for p in pads)
    return result, slices, content


def gaussian64():
    x = np.arange(64) - 31.5
    g = np.exp(-.5 * (x / 16.) ** 2)
    return (g[:, None, None] * g[None, :, None] * g[None, None, :]).astype(np.float32)


def ssim_xyz(a, b, data_range=6.):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3 or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid SSIM inputs')
    c1, c2 = (.01 * data_range) ** 2, (.03 * data_range) ** 2
    if min(a.shape) < 7:
        u, v = a.mean(), b.mean()
        va, vb = a.var(), b.var()
        cov = np.mean((a - u) * (b - v))
        return float(((2*u*v+c1)*(2*cov+c2))/((u*u+v*v+c1)*(va+vb+c2)+1e-8))
    g = np.exp(-np.arange(-3, 4, dtype=float) ** 2 / (2 * 1.5 ** 2)); g /= g.sum()
    kernel = np.outer(g, g)
    vals = []
    for z in range(a.shape[2]):
        x, y = a[:, :, z], b[:, :, z]
        conv = lambda t: convolve2d(t, kernel, mode='valid')
        u, v = conv(x), conv(y)
        va, vb, cov = conv(x*x)-u*u, conv(y*y)-v*v, conv(x*y)-u*v
        vals.append(np.mean(((2*u*v+c1)*(2*cov+c2))/((u*u+v*v+c1)*(va+vb+c2)+1e-8)))
    return float(np.mean(vals))


def masked_errors(a, b, mask):
    diff = np.asarray(a, dtype=np.float64)[mask] - np.asarray(b, dtype=np.float64)[mask]
    if not diff.size:
        return dict(mse=None, mae=None, psnr=None, reason='empty_region')
    if not np.isfinite(diff).all():
        raise ValueError('Nonfinite metric region')
    mse, mae = float(np.mean(diff * diff)), float(np.mean(np.abs(diff)))
    return dict(mse=mse, mae=mae, psnr=None if mse == 0 else 10 * math.log10(36 / mse),
                reason='perfect_match_infinite_psnr' if mse == 0 else None)


def weighted_lesions(records):
    if not records:
        raise ValueError('Empty positive-case denominator')
    values = np.array([r['ssim'] for r in records], float)
    weights = np.array([r['n_voxels'] for r in records], float)
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Invalid lesion denominator')
    return float(np.average(values, weights=weights))


def choose_step(scores, tolerance=1e-6):
    if set(scores) != set(STEPS) or not all(np.isfinite(v) for v in scores.values()):
        raise ValueError('Incomplete candidate denominator')
    best = max(scores.values())
    return min(step for step, score in scores.items() if best - score <= tolerance)


def cluster_bootstrap(values, groups, other=None):
    values, groups = np.asarray(values, float), np.asarray(groups)
    if len(values) != len(groups) or not len(values) or not np.isfinite(values).all():
        raise ValueError('Invalid patient bootstrap denominator')
    names = sorted(set(groups)); indices = [np.flatnonzero(groups == name) for name in names]
    n = np.array([len(i) for i in indices])
    sums = np.array([values[i].sum() for i in indices])
    rng = np.random.RandomState(20261013)
    samples = rng.randint(0, len(names), size=(20000, len(names)))
    denominator = n[samples].sum(axis=1)
    means = sums[samples].sum(axis=1) / denominator
    result = dict(mean=float(values.mean()), ci95=np.quantile(means, [.025, .975]).tolist(),
                  cases=len(values), patient_groups=len(names), replicates=20000, seed=20261013)
    if other is not None:
        other = np.asarray(other, float)
        if other.shape != values.shape or not np.isfinite(other).all():
            raise ValueError('Unpaired bootstrap comparison')
        delta = values - other
        ds = np.array([delta[i].sum() for i in indices])
        dm = ds[samples].sum(axis=1) / denominator
        group_delta = ds / n
        wins, losses = int((group_delta > 1e-6).sum()), int((group_delta < -1e-6).sum())
        result.update(paired_mean_difference=float(delta.mean()), paired_ci95=np.quantile(dm, [.025, .975]).tolist(),
                      group_wtl=[wins, len(names)-wins-losses, losses],
                      uncorrected_group_sign_p=float(binomtest(wins, wins+losses).pvalue) if wins+losses else 1.)
    return result
