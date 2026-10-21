"""Freeze this new attempt after local checks; never rewrite a frozen plan."""
import ast
import json
import math
import shutil
from datetime import datetime,timezone
from common import ROOT,S2,REMOTE,META,transport


def main():
    t=transport();old=ROOT.parent/'s2_k1_small_lesion_p0_20261015_r3_diagnostics'
    require_sha='471ce8b9ff87b4aaf1a83ca8eee86cc9f0c4525d83e7fd8c632ad42a6f6236bd'
    assert t.sha(old/'CPU_EXECUTION_PLAN.json')==require_sha
    p=json.loads((old/'CPU_EXECUTION_PLAN.json').read_text())
    for folder in ('sources','manifests'):shutil.copytree(old/folder,ROOT/folder)
    correction=S2/'05_documents/02_plans/20261009_k1_small_lesion_nnunet_refinement/revisions/20261015/R3_STOP_AND_CORRECTION_REQUIREMENTS_20261015.md'
    shutil.copyfile(correction,ROOT/'sources'/correction.name)
    inv=json.loads((old/'operator/inventory_once_20261015T073016Z/ASSET_SNAPSHOT.json').read_text())
    snap=ROOT/'operator/inspect_once_20261015T150744Z/SNAPSHOT.json'
    observed=json.loads(snap.read_text())
    geometry_path=ROOT.parent/'s2_k1_small_lesion_p0_20261014_r2/operator/pull_observed_lightweight_20261015T070910Z/remote_exact/roundtrip/GEOMETRY.json'
    geometry=json.loads(geometry_path.read_text())['cases']
    reserve=p['probability_capacity']['other_authorized_G1_full_reserve_bytes']
    # Preregister disjoint deterministic 3-case batches; only batch000 is currently submitted.
    batches=[]
    for offset in range(0,len(geometry),3):
        group=geometry[offset:offset+3]
        payload=sum((math.prod(c['original_shape'])+math.prod(c['training_shape']))*5*4*2
                    +math.prod(c['original_shape'])*2 for c in group)
        batches.append(dict(index=len(batches),cases=[c['subject_id'] for c in group],
            probability_and_argmax_payload_bytes=payload,stage_budget_bytes=payload+128*2**20))
    assert sum(len(b['cases']) for b in batches)==103
    probability=dict(**META,status='hold_server_only_capacity',
        authorization_source='latest_direct_user_20261015_server_only_no_local_probabilities',
        remote_destination=REMOTE+'/probabilities',local_probability_storage_authorized=False,
        checkpoints=inv['checkpoints'],geometry=geometry,
        dtype='float32',epochs=[175,150],cases=103,preprocess_workers=0,export_workers=0,
        multiprocessing_queue_used=False,training=False,wt_thresholds=[.10,.15,.20,.25,.30,.35,.40,.45,.50],
        RC_policy='baseline label4 voxelwise fixed; outside WTsum>=threshold and WTsum>pRC',
        batches=batches,other_G1_full_reserve_bytes=reserve,cpu_reserve_bytes=2**30,
        server_full_probability_payload_bytes=156004487560,
        server_full_argmax_bytes=sum(math.prod(c['original_shape'])*2 for c in geometry),
        server_safety_reserve_bytes=4*2**30,cleanup_or_local_transfer_authorized=False,
        future_batches='not_submitted_full_server_capacity_required',full_probability_complete=False)
    p.update(**META,remote_root=REMOTE,created_local_utc=datetime.now(timezone.utc).isoformat(),
        parent_failed_plan_sha256=require_sha,authorization_source='direct_user_20261015_execute_independent_full_WT_repair',
        job_name='s2k1p0supp4',phase='full_WT_CPU_only_probability_server_capacity_hold',probability=probability,
        all_original_wt_sources_required=True,progress_snapshot_sha256=t.sha(snap))
    static=[f for folder in ('runtime','sources','manifests','tests','slurm') for f in (ROOT/folder).rglob('*') if f.is_file()]
    p['static_sha256']={str(f.relative_to(ROOT)):t.sha(f) for f in static}
    for f in static:
        if f.suffix=='.py':ast.parse(f.read_text(),filename=str(f))
    t.exclusive(ROOT/'CPU_EXECUTION_PLAN.json',p)
    t.exclusive(ROOT/'operator/FREEZE_RECEIPT.json',dict(**META,status='frozen',plan_sha256=t.sha(ROOT/'CPU_EXECUTION_PLAN.json'),
        static_count=len(static),old_plan_unchanged=t.sha(old/'CPU_EXECUTION_PLAN.json')==require_sha,
        observed_available_bytes=observed['capacity'][REMOTE.rsplit('/',1)[0]]['available_bytes']))
    print(json.dumps(dict(status='frozen',plan_sha256=t.sha(ROOT/'CPU_EXECUTION_PLAN.json'),static_files=len(static),
        probability_batches=len(batches),first_batch_budget_bytes=batches[0]['stage_budget_bytes'])))


if __name__=='__main__':main()
