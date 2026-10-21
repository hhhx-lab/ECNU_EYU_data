"""Compare label-independent source supports; never repair labels or expand masks."""
import numpy as np
from scipy.ndimage import binary_fill_holes


VERSION = 'source_support_comparison_v1_no_label_repair'


def supports(arrays):
    if len(arrays) != 3 or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError('Three common-geometry authentic source arrays required')
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('Nonfinite authentic source')
    positive = np.zeros(arrays[0].shape, bool)
    nonzero = positive.copy()
    negative = positive.copy()
    for image in arrays:
        low, high = np.percentile(image, [0, 100])
        low = max(0., float(low))
        if high > low:
            positive |= np.clip(image, low, high) > low
        nonzero |= image != 0
        negative |= image < 0
    return positive, binary_fill_holes(positive), nonzero, binary_fill_holes(nonzero), negative


def coverage(label, arrays):
    if label.shape != arrays[0].shape or not np.isfinite(label).all():
        raise ValueError('Invalid label geometry or finite contract')
    if not np.equal(label, np.rint(label)).all() or not np.isin(label, [0, 1, 2, 3, 4]).all():
        raise ValueError('Illegal label')
    raw_positive, positive, raw_nonzero, nonzero, negative = supports(arrays)
    lesion = label > 0
    old_missing = lesion & ~positive
    missing = lesion & ~nonzero
    count = lambda mask: int(np.count_nonzero(mask))
    result = {'algorithm_version': VERSION, 'label_voxels': count(lesion),
              'positive_support_voxels': count(positive), 'nonzero_support_voxels': count(nonzero),
              'positive_missing_voxels': count(old_missing), 'nonzero_missing_voxels': count(missing),
              'old_missing_with_negative_source': count(old_missing & raw_nonzero & negative),
              'old_missing_all_three_zero': count(old_missing & ~raw_nonzero),
              'nonzero_missing_all_three_zero': count(missing & ~raw_nonzero),
              'label_all_three_zero': count(lesion & ~raw_nonzero),
              'label_counts': {str(k): count(label == k) for k in range(5)},
              'positive_missing_by_label': {str(k): count(old_missing & (label == k)) for k in range(1, 5)},
              'nonzero_missing_by_label': {str(k): count(missing & (label == k)) for k in range(1, 5)},
              'positive_support_nonempty': bool(positive.any()), 'nonzero_support_nonempty': bool(nonzero.any())}
    return result, old_missing, missing
