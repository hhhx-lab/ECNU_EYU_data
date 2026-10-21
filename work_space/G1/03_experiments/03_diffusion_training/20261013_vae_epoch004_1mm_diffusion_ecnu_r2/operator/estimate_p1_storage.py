"""Bound P1 storage from a completed P0 geometry manifest, not compressed source size."""

import argparse
import json
import math
from pathlib import Path


def estimate(records, network_optimizer_bytes):
    geometries = {}
    modalities = {}
    for row in records:
        if row['split'] not in ('train', 'val'):
            continue
        if row['status'] != 'pass':
            raise ValueError('P0 record is not pass')
        modalities.setdefault(row['case_id'], set()).add(row['modality'])
        if row['modality'] == 't1n':
            if row['case_id'] in geometries:
                raise ValueError('duplicate t1n record')
            geometries[row['case_id']] = row['geometry']
    if len(geometries) != 926 or len(modalities) != 926:
        raise ValueError('expected all 926 train/val cases')
    if any(values != {'t1n','t1c','t2w','t2f','seg'} for values in modalities.values()):
        raise ValueError('incomplete modality records')
    # Full native FOV at 1 mm bounds a source-mask crop. No gzip savings assumed.
    voxels = sum(math.prod(max(1, math.ceil(n*s)+1) for n,s in zip(g['shape_xyz'],g['spacing_xyz']))
                 for g in geometries.values())
    dataset_bytes = voxels * (4*4 + 1 + 1) + 926*6*4096
    checkpoint_count = 40 + 1 + 8  # every 5k, plus safe TERM and smoke/resume reserves
    checkpoint_bytes = 4 * checkpoint_count * (network_optimizer_bytes + 8*1024**2)
    # Includes a full-size preprocessing temporary reserve, plus logs, CSV and montages.
    reserve = dataset_bytes + checkpoint_bytes + 8*1024**3
    return {'artifact_status':'experimental_unvalidated','operator_approved':False,
            'formal_gate_status':'not_run_not_passed','stage':'P1_P3_storage_estimate',
            'cases':926,'full_fov_1mm_voxels':voxels,'derived_dataset_upper_bound_bytes':dataset_bytes,
            'four_modality_checkpoint_reserve_bytes':checkpoint_bytes,
            'preprocessing_temporary_reserve_bytes':dataset_bytes,
            'total_required_bytes':dataset_bytes+reserve,
            'assumptions':['4 float32 modalities; uint8 label and mask; gzip no savings',
                           'full FOV upper bound, actual common brain bbox may be smaller',
                           'no persistent full-volume disk cache',
                           'new controlled resume jobs require remaining-capacity recheck'],
            'operator_action':'fresh statvfs >= total_required_bytes before P1; not a P0 gate'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--p0-manifest',type=Path,required=True)
    p.add_argument('--network-optimizer-bytes',type=int,required=True)
    args=p.parse_args()
    with args.p0_manifest.open() as h:
        result=estimate((json.loads(line) for line in h),args.network_optimizer_bytes)
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__ == '__main__':
    main()
