"""Offline QA and source/audit identity for S0; never marks runtime pass."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[4]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, payload):
    with (ROOT/name).open('x', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    checks = []
    for path in sorted((ROOT/'operator').glob('*.py')):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
                if any(isinstance(t,ast.Name) and t.id=='CODE' or isinstance(t,ast.Attribute) and t.attr=='CODE' for t in node.targets):
                    ast.parse(node.value.value)
        checks.append(dict(check='local_and_embedded_AST', file=str(path.relative_to(ROOT)), status='pass'))
    receipts = []
    for directory in sorted((ROOT/'operator').glob('s0_preflight_*')):
        receipt = json.loads((directory/'LOCAL_OPERATION_RECEIPT.json').read_text())
        assert receipt['stage'] == 'remote_read_only'
        assert not receipt['vpn_owned'] or receipt['own_vpn_stop_completed']
        result = json.loads((directory/'RESULT.json').read_text())
        assert result['artifact_status']=='experimental_unvalidated' and result['operator_approved'] is False
        assert result['formal_gate_status']=='not_run_not_passed'
        assert result['scope']['remote_writes'] is False and result['scope']['jobs_created'] is False
        receipts.append(dict(path=str(directory.relative_to(ROOT)),result_sha256=sha(directory/'RESULT.json'),
            local_receipt_sha256=sha(directory/'LOCAL_OPERATION_RECEIPT.json'),remote_utc=result['remote_utc'],
            received_utc=result['local_received_utc'],vpn_stop_confirmed=receipt['own_vpn_stop_completed']))
    public = ROOT/'operator/public_sources_20261103T164839Z/PUBLIC_SOURCE_AUDIT.json'
    original = json.loads(public.read_text())
    index = {**original, 'source_responses': [
        {key:value for key,value in r.items() if key!='author_environment_text'} for r in original['source_responses']
    ]}
    index['local_full_source_audit_sha256'] = sha(public)
    index_path = public.with_name('PUBLIC_SOURCE_INDEX.json')
    if not index_path.exists():
        with index_path.open('x',encoding='utf-8') as f:
            json.dump(index,f,ensure_ascii=False,indent=2);f.write('\n')
    author = json.loads((ROOT/'INSTALL_AUTHORIZATION.json').read_text())
    assert author['environment_installation_authorized_by_user'] if 'environment_installation_authorized_by_user' in author else author['necessary_transitive_dependencies_authorized']
    assert author['direct_dependencies']['nnunetv2']=='2.7.0' and author['installation_status']=='not_performed'
    brain = json.loads((ROOT/'BRAINIAC_DOWNLOAD_IDENTITY.json').read_text())
    assert brain['public_preview_response_bytes']==362119240 and brain['sha256'] is None and not brain['downloaded']
    paths = sorted((ROOT/'operator').glob('*.py'))
    identities = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    transport = PROJECT/'work_space/S2/03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260923_r3_fallback/operator/access.py'
    meta = dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed')
    write('OPERATOR_SOURCE_IDENTITY.json',dict(meta,source_sha256=identities,
        inherited_transport_path=str(transport.relative_to(PROJECT)),inherited_transport_sha256=sha(transport),
        scope='read_only_S0_operator_not_frozen_inference_package',recorded_utc=datetime.now(timezone.utc).isoformat()))
    write('PREPARATION_VERIFICATION.json',dict(meta,status='offline_preparation_QA_pass_not_runtime_or_scientific_pass',
        checked_utc=datetime.now(timezone.utc).isoformat(),checks=checks,read_only_snapshot_receipts=receipts,
        source_metadata_records=len(index['source_responses']),raw_public_text_git_excluded=True,
        jobs_created=False,model_assets_downloaded=False,dependencies_installed=False,
        known_local_probe_failures=['Unquoted GitHub query URL rejected by zsh before any network request',
            'Local Dropbox HEAD timed out after 35 seconds; no model body requested',
            'Playwright network-request argument validation rejected once; corrected to index/response-body; no remote action']))
    print(json.dumps(dict(status='offline_preparation_QA_pass',snapshot_count=len(receipts))))


if __name__ == '__main__':
    main()
