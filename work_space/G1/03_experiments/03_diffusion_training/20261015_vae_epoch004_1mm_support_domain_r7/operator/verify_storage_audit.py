"""Validate local read-only audit arithmetic without another remote operation."""
import hashlib
import json

from common import ROOT, m


def main():
    audit = ROOT / 'operator/storage_targeted_20261015T160756Z'
    path = audit / 'STORAGE_TARGETED.json'
    value = json.loads(path.read_text())
    receipt = json.loads((audit / 'LOCAL_OPERATION_RECEIPT.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['file_sha256']
    assert receipt['operation_vpn_closed'] and not receipt['remote_writes']
    assert not receipt['remote_submissions'] and not receipt['large_data_downloaded']
    g = value['g1_remaining_budget']
    completed = g['completed']
    assert len(completed) == g['completed_metadata_checked'] == 297
    assert len({r['case_id'] for r in completed}) == 297 and not g['errors']
    bound = 0
    for row in completed:
        x, y, z = row['shape']
        assert x > 0 and y > 0 and z > 0
        n = x * y * z
        one = 0
        for payload in (4*n, 4*n, 4*n, 4*n, n, n):
            payload += 4096
            one += payload + (payload >> 12) + (payload >> 14) + (payload >> 25) + 31
        assert one == row['compressed_bound_bytes']
        bound += one
    assert bound == g['completed_payload_bound_bytes']
    assert sum(r['allocated_bytes'] for r in completed) == g['completed_payload_allocated_bytes']
    original = g['original_budget']
    reserves = (original['temporary_upper_bytes'] + original['checkpoint_reserve_bytes']
                + original['metadata_reserve_bytes'] + 4*1024**3)
    remaining = original['compressed_worst_bound_bytes'] - bound
    assert remaining == g['remaining_payload_upper_bytes']
    additional = remaining + reserves
    assert additional == g['additional_conservative_reserve_bytes']
    probability = 139852677120 + 16151810440
    argmax = 139852677120 // 20
    k1 = probability + argmax + 5*1024**3
    free = value['capacity_end']['available_bytes']
    assert free + value['capacity_end']['used_bytes'] == value['capacity_end']['total_bytes']
    projected_g1 = g['completed_payload_allocated_bytes'] / len(completed) * (926-len(completed)) + reserves
    result = dict(**m.META, status='pass_local_SHA_JSON_counts_arithmetic_only',
        source_sha256=receipt['file_sha256'], metadata_cases=297, total_cases=926,
        current_free_bytes=free, g1_additional_conservative_reserve_bytes=additional,
        probability_float32_bytes=probability, argmax_bytes=argmax,
        k1_additional_budget_bytes=k1, combined_additional_budget_bytes=k1+additional,
        conservative_capacity_gap_bytes=max(0,k1+additional-free),
        g1_additional_empirical_projection_bytes=projected_g1,
        empirical_projection_is_submission_gate=False,
        remote_content_SHA_validated=False, remote_file_geometry_validated=False,
        remote_snapshot_utc=value['remote_end_utc'], local_received_utc=value['local_received_utc'])
    m.exclusive(audit / 'LOCAL_STORAGE_VERIFICATION.json', result)
    print(json.dumps({k:round(v/1024**3,6) for k,v in result.items() if k.endswith('_bytes')}))


if __name__ == '__main__':
    main()
