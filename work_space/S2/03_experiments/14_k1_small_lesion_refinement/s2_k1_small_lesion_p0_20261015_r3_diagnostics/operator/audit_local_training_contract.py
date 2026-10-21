"""Inventory local historical code evidence without claiming measured gradients."""
import ast
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
S2=ROOT.parents[2]
REPO=S2/'01_code/repository/custom_nnunet'
META=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed')


def main():
    paths=[REPO/n for n in ('nnUNetTrainerBraTS2026RC.py','nnUNetTrainerBraTS2026RCFocalCompletionFineTune.py',
                           'small_lesion_trainer_mixins.py','small_lesion_variants.py','met_aug_data_loader.py','met_aug_paired_training.py')]
    paths += [S2/'03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260929_k1_175_r3/runtime/nnUNetTrainerBraTS2026RCMetAugK1245FocalCompletionFineTune175.py',
              S2/'03_experiments/12_k1245_rc_aware/s2_met_aug_k1_175_resume_20261002_r1/runtime/research_environment.sh']
    records=[]
    for p in paths:
        raw=p.read_bytes()
        if p.suffix=='.py':ast.parse(raw,filename=str(p))
        records.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),evidence_scope='local_source_only_not_new_remote_runtime_verification'))
    inv=json.loads((ROOT/'operator/inventory_once_20261015T073016Z/ASSET_SNAPSHOT.json').read_text())
    installed={n:dict(path=r['path'],sha256=r.get('sha256'),missing=r.get('missing',False)) for n,r in inv['installed_sources'].items()}
    payload=dict(**META,status='contract_evidence_prepared_measurement_not_run',local_utc=datetime.now(timezone.utc).isoformat(),
        local_source_files=records,installed_source_snapshot=installed,
        observed_local_loss=dict(family='Dice + focal weighted CE',gamma=2.0,class_weights=[1,1,1,1,3],deep_supervision='geometric weights with last output zero, normalized'),
        historical_patch_exposure='unavailable_no_complete_historical_instance_transform_trace_verified',
        measured_parameter_gradients=False,measured_deep_supervision_identity_survival=False,
        G1_lineage_scope='frozen selection JSON SHA and declared path confirmed; full generation code/CSV/consumption chain not yet verified',
        next_gate='CPU full training instance inventory/transport complete, then freeze actual sampler and loss remote SHA plus independent diagnostic sample protocol',
        optimizer_step_authorized=False,long_training_authorized=False,
        caution='local source settings are not proof every historical patch used them; no proxy reported as gradient or true175 exposure')
    target=ROOT/'operator/TRAINING_DIAGNOSTIC_PREPARATION.json'
    with target.open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=payload['status'],local_sources=len(records),installed_sources=len(installed),measured_gradients=False)))


if __name__=='__main__':main()
