"""Trace every original WT source before classifying or selecting small lesions."""
import copy
from collections import Counter
from official_identity import stable_id
from transport_rules import classify


def classify_sources(rows):
    identifiers = [r['stable_reference_id'] for r in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('duplicate source identity')
    degree = Counter()
    for row in rows:
        destinations = row['destination_components']
        if len(destinations) != len(set(destinations)) or any(x <= 0 for x in destinations):
            raise ValueError('invalid destination adjacency')
        degree.update(destinations)
    output = copy.deepcopy(rows)
    for row in output:
        row['transport_category'] = classify(row, degree)
    return output, degree


def trace_volume(ref, geometry, case, reference_sha, transpose, box, target,
                 spacing, target_spacing, resample, export, expected=None):
    import numpy as np
    from scipy import ndimage
    structure = np.ones((3, 3, 3), dtype=bool)
    source_labels, count = ndimage.label(np.isin(ref, [1, 2, 3]), structure=structure)
    full = resample(ref.transpose(transpose)[box][None], target, spacing, target_spacing)[0]
    full_wt = np.isin(full, [1, 2, 3])
    full_scores = np.stack([full == value for value in range(5)]).astype(np.float32)
    restored = export(full_scores)
    if restored.shape != ref.shape or not set(np.unique(restored)) <= set(range(5)):
        raise ValueError('invalid restored volume')
    destinations, dest_count = ndimage.label(np.isin(restored, [1, 2, 3]), structure=structure)
    rows = []
    for index, bbox in enumerate(ndimage.find_objects(source_labels), 1):
        if bbox is None:
            continue
        coords = np.argwhere(source_labels[bbox] == index) + np.array([s.start for s in bbox])
        key = stable_id(case, 'wt', reference_sha, geometry, coords)
        volume = float(len(coords) * np.prod(geometry['spacing']))
        prior = expected[key] if expected is not None else None
        if prior is not None and (int(prior['original_voxel_count']) != len(coords)
                or not np.isclose(float(prior['original_volume_mm3']), volume, rtol=1e-9, atol=1e-9)):
            raise ValueError('original identity/volume drift')
        mask = np.zeros(ref.shape, dtype=np.uint8)
        mask[tuple(coords.T)] = 1
        cut = mask.transpose(transpose)[box][None]
        standalone = resample(cut, target, spacing, target_spacing)[0] > 0
        supported = standalone & full_wt
        scores = np.zeros((5, *target), dtype=np.float32)
        scores[0] = ~supported
        scores[1] = supported
        returned = export(scores) > 0
        adjacent = sorted(int(v) for v in np.unique(destinations[returned]) if v)
        rows.append(dict(subject_id=case, stable_reference_id=key, reference_sha256=reference_sha,
            original_volume_mm3=volume, original_voxel_count=len(coords), small=volume < 27,
            outside_crop=not bool(cut.any()), standalone_training_voxels=int(standalone.sum()),
            training_region_voxels=int(supported.sum()), restored_trace_voxels=int(returned.sum()),
            destination_components=adjacent, detected=(prior['detected'] == 'True') if prior else None,
            sq_dsc=prior['sq_dsc'] if prior else None,
            trace_semantics='all_original_WT_binary_identity_with_full_WT_support_proxy_not_unique_multiclass_provenance'))
    if len(rows) != count or (expected is not None and set(expected) != {r['stable_reference_id'] for r in rows}):
        raise ValueError('all WT source coverage mismatch')
    rows, degree = classify_sources(rows)
    targets = []
    for index in range(1, dest_count + 1):
        sources = sorted(r['stable_reference_id'] for r in rows if index in r['destination_components'])
        if len(sources) != degree[index]:
            raise ValueError('source degree not independently reproducible')
        targets.append(dict(subject_id=case, destination_component=index,
            original_grid_voxels=int((destinations == index).sum()), source_degree=len(sources),
            source_ids=sources, small_source_count=sum(r['small'] and index in r['destination_components'] for r in rows)))
    return rows, targets, restored
