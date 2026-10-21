"""Accept existing read-only evidence after separate VPN logout confirmation."""
import csv
import hashlib
import json
import subprocess

import numpy as np

from common import ROOT, m


def main():
    audit = ROOT / 'operator/raw_source_sha_header_audit_20261020T072240Z'
    result = json.loads((audit / 'RESULT.json').read_text())
    assert result['status'] == 'pass' and result['cases_ok'] == 1165 and not result['errors']
    state = subprocess.run([m.VPN, 'status'], capture_output=True, text=True, timeout=40)
    assert state.returncode == 0
    observed = [json.loads(line[line.index('{'):]) for line in state.stdout.splitlines() if '{' in line]
    assert len(observed) == 1 and observed[0]['status'] == 'logout' and observed[0]['tunnel_status'] == 'tunnelUnavailable'
    m.exclusive(audit / 'INDEPENDENT_VPN_LOGOUT_LEGACY_AFFINE_ACCEPTANCE.json', dict(m.META, status='pass',
                utc=m.datetime.now(m.timezone.utc).isoformat(), logout=True, tunnel_unavailable=True,
                original_stop_returned_nonzero=True, scope='read-only status; no logout/retry/new SSH'))
    raw = {}
    for name in ('raw_manifest.csv', 'ucsd_manifest.csv'):
        with (ROOT / 'plan' / name).open(newline='') as f:
            for row in csv.DictReader(f):
                raw.setdefault(row['case_id'], row)
    errors = []
    for case, record in result['cases'].items():
        for mod, file in record['files'].items():
            a = np.round(np.asarray(file['affine'], dtype=np.float64), 6)
            legacy_digest = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
            a[np.isclose(a, 0, atol=5e-7, rtol=0)] = 0
            digest = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
            expected = raw[case]['affine_hash_' + mod]
            file['archived_affine_hash'] = expected
            file['observed_affine_hash'] = digest
            file['observed_legacy_signed_zero_affine_hash'] = legacy_digest
            if legacy_digest != expected:
                errors.append(dict(case=case, modality=mod, expected=expected, observed_legacy=legacy_digest, observed_canonical=digest))
    m.exclusive(audit / 'LOCAL_SOURCE_LEGACY_AFFINE_ACCEPTANCE.json', dict(m.META, status='failed' if errors else 'pass',
                utc=m.datetime.now(m.timezone.utc).isoformat(), source_audit_sha256=m.sha(audit / 'RESULT.json'),
                cases=1165, archived_affine_errors=errors,
                missing_t2w_original_byte_baseline='new_frozen_source_SHA_not_prior_historical_byte_SHA',
                locked_test_images_read=False, no_remote_retry=True))
    assert not errors, 'Archived affine mismatch: ' + str(errors[:3])
    result['archived_affine_review'] = 'all_selected_modality_headers_match_archived_round6_float64_signed_zero_byte_hash'
    result['prior_canonical_hash_mismatch'] = 'retained_failed_local_acceptance; historical CSV preserves signed zero, current G2 canonicalizes it'
    result['missing_t2w_source_identity'] = 'new_frozen_original_source_SHA_with_archived_header_not_prior_byte_baseline'
    m.exclusive(ROOT / 'config/SOURCE_RECORDS.json', result)
    cfgpath = ROOT / 'config/RUN.json'
    cfg = json.loads(cfgpath.read_text())
    assert cfg['source_records_sha256'] == 'pending_source_audit'
    cfg['source_records_sha256'] = m.sha(ROOT / 'config/SOURCE_RECORDS.json')
    cfgpath.chmod(0o640)
    cfgpath.write_text(json.dumps(cfg, indent=2, sort_keys=True) + '\n')
    cfgpath.chmod(0o440)
    print(json.dumps(dict(status='pass', cases=1165, source_bytes=result['source_bytes'], source_records_sha256=cfg['source_records_sha256'])))


if __name__ == '__main__':
    main()
