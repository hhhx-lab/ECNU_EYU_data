"""Freeze the verified CPU scope without changing any parent runtime."""
import ast
import hashlib
import json
import math
import shutil
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
S2=ROOT.parents[2]
OLD=ROOT.parent/'s2_k1_small_lesion_p0_20261014_r2'
INV=ROOT/'operator/inventory_once_20261015T073016Z/ASSET_SNAPSHOT.json'
META=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(payload,f,indent=2,allow_nan=False);f.write('\n')


def main():
    old=json.loads((OLD/'CPU_EXECUTION_PLAN.json').read_text())
    assert sha(OLD/'CPU_EXECUTION_PLAN.json')=='e3169bf735a39d05c066ca34c5568285b48ba060c79007331a850c903c936beb'
    inv=json.loads(INV.read_text())
    assert not inv['dependency_drift']
    remote=old['remote_root'].rsplit('/',1)[0]+'/'+ROOT.name
    for name in ('runtime/cpu_roundtrip.py','runtime/official_identity.py','runtime/boundary_tests.py'):
        assert sha(OLD/name)==old['static_sha256'][name]
        shutil.copyfile(OLD/name,ROOT/name)
    for name in ('manifests/FIXED103_REAL_T2W_CASES.csv','manifests/INPUT_REFERENCE_SHA256.json'):
        dst=ROOT/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(OLD/name,dst)
    for method in ('K1','K4'):
        path=ROOT/'sources'/method/'panoptica_evaluation_summary.json';path.parent.mkdir(parents=True)
        shutil.copyfile(OLD/'sources'/method/'panoptica_evaluation_summary.json',path)
    base=json.loads((OLD/'sources/INVENTORY_VALIDATION.json').read_text()) if (OLD/'sources/INVENTORY_VALIDATION.json').exists() else {}
    for method in ('B175','D50'):
        record=inv['routes'][method]['evaluation']['panoptica_evaluation_summary.json']
        path=ROOT/'sources'/method/'panoptica_evaluation_summary.json';path.parent.mkdir(parents=True)
        # JSON is source text; reject normalization rather than silently replace SHA.
        with path.open('x') as f:f.write(record['content'])
        assert sha(path)==record['sha256'],'source text byte normalization drift'
    contract=S2/'05_documents/02_plans/20261009_k1_small_lesion_nnunet_refinement/revisions/20261015/P0_FOLLOWUP_DIAGNOSTICS_PREREG_20261015.md'
    shutil.copyfile(contract,ROOT/'sources/P0_FOLLOWUP_DIAGNOSTICS_PREREG_20261015.md')
    parent_geometry=OLD/'operator/pull_observed_lightweight_20261015T070910Z/remote_exact/roundtrip/GEOMETRY.json'
    geometry=json.loads(parent_geometry.read_text())['cases']
    original=sum(math.prod(r['original_shape']) for r in geometry)*5*4*2
    training=sum(math.prod(r['training_shape']) for r in geometry)*5*4*2
    g1path=S2.parent/'G1/03_experiments/03_diffusion_training/20261014_vae_epoch004_1mm_diffusion_manual_r5/operator/pull_launch_evidence_20261014T075642Z/pulled/r5/STATIC_PREFLIGHT.json'
    # S2 is work_space/S2; the shared G1 stage is a sibling, not a new scan.
    g1=json.loads(g1path.read_text())['storage_header_upper_bound']['required_bytes']
    available=int(inv['disk']['stdout'].splitlines()[1].split()[3])
    capacity=dict(**META,status='hold_probability_capacity',observed_available_bytes=available,
        other_authorized_G1_full_reserve_bytes=g1,reserve_evidence_path=str(g1path),reserve_evidence_sha256=sha(g1path),
        cpu_lightweight_reserve_bytes=2**30,original_probabilities_float32_two_epochs_bytes=original,
        training_probabilities_float32_two_epochs_bytes=training,probability_min_payload_bytes=original+training,
        probability_precision_or_case_reduction=False,remote_progress_snapshots=1,
        note='conservatively reserve the full authorized G1 requirement, no claim all reserve is still unmaterialized')
    assert available-g1>2**30,'CPU capacity conflict'
    assert available-g1-2**30<original+training,'re-evaluate probability hold if capacity sufficient'
    write(ROOT/'operator/PROBABILITY_CAPACITY_HOLD.json',capacity)
    plan=old.copy()
    plan.update(**META,remote_root=remote,created_local_utc=datetime.now(timezone.utc).isoformat(),
        job_name='s2k1p0supp3',phase='P0-B/C/D CPU supplement; P0-E/F held not submitted',
        authorization_source='direct_user_20261015_execute_four_diagnostic_steps_no_long_training',
        parent_plan_sha256=sha(OLD/'CPU_EXECUTION_PLAN.json'),asset_snapshot_sha256=sha(INV),
        parent_geometry=old['remote_root']+'/roundtrip/GEOMETRY.json',parent_geometry_sha256=inv['r2_geometry_sha256'],
        parent_matrix=old['remote_root']+'/results/cpu_identity/REFERENCE_DETECTION_MATRIX.csv',parent_matrix_sha256=inv['r2_matrix_sha256'],
        parent_roundtrip=old['remote_root']+'/roundtrip/prediction',parent_roundtrip_sha256=inv['r2_roundtrip'],
        train_ids=inv['train']['ids'],val_ids=inv['val']['ids'],mapping_rows=inv['mapping']['rows'],train_labels=inv['train_labels'],
        resource_contract=dict(partition='cpu_192G',cpus=4,mem_gb=64,time='1-00:00:00',gpus=0),
        original_E_status='blocked_original_prediction_masks_not_located',probability_capacity=capacity,
        training_authorized=False,full_p0_complete=False)
    for method in ('B175','D50'):
        plan['prediction_roots'][method]=inv['routes'][method]['root']+'/prediction'
        plan['prediction_sha256'][method]=inv['routes'][method]['predictions']
        assert len(plan['prediction_sha256'][method])==103
    plan['dependency_sha256']=dict(old['dependency_sha256'])
    plan['dependency_sha256'].update({r['path']:r['sha256'] for r in inv['train_labels']})
    plan['dependency_sha256'][inv['mapping']['path']]=inv['mapping']['sha256']
    plan['dependency_sha256'][plan['parent_geometry']]=plan['parent_geometry_sha256']
    plan['dependency_sha256'][plan['parent_matrix']]=plan['parent_matrix_sha256']
    static=[p for parent in ('runtime','sources','manifests','slurm','tests') for p in (ROOT/parent).rglob('*') if p.is_file()]
    plan['static_sha256']={str(p.relative_to(ROOT)):sha(p) for p in static}
    for p in (ROOT/'runtime').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    write(ROOT/'CPU_EXECUTION_PLAN.json',plan)
    write(ROOT/'operator/FREEZE_RECEIPT.json',dict(**META,status='frozen_cpu_only',plan_sha256=sha(ROOT/'CPU_EXECUTION_PLAN.json'),
        static_count=len(static),train_count=1035,available_identity_routes=4,probability_status='hold_capacity',original_E='blocked'))
    print(json.dumps(dict(status='frozen_cpu_only',plan_sha256=sha(ROOT/'CPU_EXECUTION_PLAN.json'),static_count=len(static),available_GiB=available/2**30,G1_reserve_GiB=g1/2**30,probability_payload_GiB=(original+training)/2**30)))


if __name__=='__main__':main()
