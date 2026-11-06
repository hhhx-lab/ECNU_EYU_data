"""Local independent submission acceptance; no SSH, image reads or job mutation."""
import json
from pathlib import Path
from submit_fixed103_comparison import ROOT, R1_PAYLOAD, encode, digest, validate_cpu_held
from session import access_module


def main():
    audits=sorted((ROOT/'operator').glob('submit_fixed103_comparison_r2_*/RELEASE_RESULT.json'))
    assert len(audits)==1, 'comparison release evidence absent or ambiguous'
    audit=audits[0].parent
    payload=json.loads((audit/'SUBMISSION_PAYLOAD.json').read_text())
    original=json.loads(R1_PAYLOAD.read_text())
    contract=payload['contract'];files=payload['files']
    package=json.loads(files['contract/PACKAGE_SHA256.json'])['files']
    assert 'contract/PACKAGE_SHA256.json' not in package
    assert all(digest(files[name])==expected for name,expected in package.items())
    changed={name for name in files if files[name]!=original['files'][name]}
    assert changed=={'contract/COMPARISON_CONTRACT.json','contract/PACKAGE_SHA256.json','slurm/compare.slurm'}
    held=json.loads((audit/'HOLD_RESULT.json').read_text())
    verified=json.loads((audit/'LOCAL_HELD_VERIFICATION.json').read_text())
    release=json.loads(audits[0].read_text())
    validate_cpu_held(held['raw_scontrol'],held['job_id'],contract['attempt'])
    assert held['job_id']==release['job_id']==verified['job_id']
    assert verified['local_held_verified'] and verified['held_audit_sha256']==digest(encode(held))
    assert release['release_returncode']==0
    assert held['contract_sha256']==release['contract_sha256']==digest(files['contract/COMPARISON_CONTRACT.json'])
    assert contract['models']==original['contract']['models'] and contract['references']==original['contract']['references']
    assert contract['evaluator']==original['contract']['evaluator']
    assert contract['submission_parent']['package_sha256']==digest(original['files']['contract/PACKAGE_SHA256.json'])
    result=dict(access_module.META,status='independent_local_CPU_submission_identity_pass_not_scoring_complete',
        job_id=release['job_id'],attempt=contract['attempt'],changed_identity_members=sorted(changed),
        contract_sha256=release['contract_sha256'],package_sha256=digest(files['contract/PACKAGE_SHA256.json']),
        cases=103,patient_groups=69,metrics_per_model=72,model_names=sorted(contract['models']),
        inference_or_locked_test_read=False,SSH_or_job_mutation=False,
        remote_release_utc=release['remote_utc'],local_accepted_utc=access_module.datetime.now(access_module.timezone.utc).isoformat())
    access_module.exclusive(audit/'INDEPENDENT_LOCAL_ACCEPTANCE.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
