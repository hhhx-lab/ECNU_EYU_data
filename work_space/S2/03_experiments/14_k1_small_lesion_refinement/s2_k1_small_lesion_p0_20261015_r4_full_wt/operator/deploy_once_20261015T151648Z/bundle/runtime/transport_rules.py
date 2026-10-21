"""Conservative instance tracing categories, separate from detection success."""


def classify(row, source_degree):
    if row['outside_crop']:
        return 'cropped_out'
    if row['standalone_training_voxels'] == 0:
        return 'disappeared_in_label_pipeline'
    if row['training_region_voxels'] == 0:
        return 'ambiguous_label_competition'
    if row['restored_trace_voxels'] == 0:
        return 'disappeared_in_label_pipeline'
    destinations = row['destination_components']
    merged = any(source_degree[x] > 1 for x in destinations)
    split = len(destinations) > 1
    if merged and split:
        return 'ambiguous_merge_and_split'
    if merged:
        return 'merged'
    if split:
        return 'split'
    if not destinations:
        return 'ambiguous_no_destination_overlap'
    return 'survived_roundtrip_match_passed' if row['detected'] else 'survived_roundtrip_match_failed'


def working_point(probabilities, threshold, baseline):
    """Keep baseline RC fixed; outside RC WT must exceed RC probability."""
    import numpy as np
    if probabilities.ndim != 4 or probabilities.shape[0] != 5:
        raise ValueError('five class probabilities required')
    if probabilities.shape[1:] != baseline.shape:
        raise ValueError('geometry mismatch')
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError('invalid probabilities')
    if not 0 <= threshold <= 1 or not set(np.unique(baseline)) <= set(range(5)):
        raise ValueError('invalid threshold/baseline')
    wt = probabilities[1:4].sum(axis=0)
    rc = baseline == 4
    choose = (~rc) & (wt >= threshold) & (wt > probabilities[4])
    output = np.zeros_like(baseline)
    output[rc] = 4
    output[choose] = (probabilities[1:4].argmax(axis=0) + 1)[choose]
    if not np.array_equal(output == 4, rc):
        raise RuntimeError('RC protection drift')
    return output
