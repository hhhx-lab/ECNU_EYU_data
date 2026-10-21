"""Independent full-label training inventory, identity expansion and transport trace."""
import copy
import csv
import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path
from cpu_roundtrip import ROOT, emit, require, sha
from official_identity import stable_id, match_arrays, validate_reference_fingerprint
from transport_rules import classify
from full_wt_transport import trace_volume

REGIONS = dict(wt=[1,2,3],tc=[1,3],et=[3],rc=[4])


def plan():
    p=json.loads((ROOT/'CPU_EXECUTION_PLAN.json').read_text())
    for n,d in p['static_sha256'].items():
        require(sha(ROOT/n)==d,'static drift: '+n)
    for n,d in p['dependency_sha256'].items():
        require(sha(Path(n))==d,'dependency drift: '+n)
    return p


def image(path, digest=None):
    import numpy as np
    import SimpleITK as sitk
    if digest:
        require(sha(Path(path))==digest,'image SHA drift')
    obj=sitk.ReadImage(str(path))
    arr=sitk.GetArrayFromImage(obj)
    require(np.isfinite(arr).all() and set(np.unique(arr))<=set(range(5)),'illegal labels')
    geom=dict(shape=list(arr.shape),spacing=list(obj.GetSpacing()),origin=list(obj.GetOrigin()),direction=list(obj.GetDirection()))
    return arr,geom


def components(binary):
    import numpy as np
    from scipy import ndimage
    labels,_=ndimage.label(binary,structure=np.ones((3,3,3),bool))
    items=[]
    for index,box in enumerate(ndimage.find_objects(labels),1):
        if box is None:
            continue
        coords=np.argwhere(labels[box]==index)+np.array([s.start for s in box])
        items.append((index,coords))
    return labels,items


def write_csv(path, rows):
    require(bool(rows),'empty results')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def trace(subset=False):
    import numpy as np
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
    p=plan()
    pm=PlansManager(p['plans_path']);cm=pm.get_configuration('3d_fullres')
    dataset=json.loads(Path(p['dataset_path']).read_text())
    g=json.loads(Path(p['parent_geometry']).read_text())['cases']
    props={r['subject_id']:r for r in g}
    base_rows=list(csv.DictReader(Path(p['parent_matrix']).open()))
    wanted={(r['subject_id'],r['stable_reference_id']):r for r in base_rows if r['method']=='label_roundtrip' and r['region']=='wt'}
    require(sum(r['small']=='True' for r in wanted.values())==286,'small reference denominator')
    all_rows=[];all_targets=[]
    selected=[next(c for c in p['case_ids'] if any(key[0]==c for key in wanted))] if subset else p['case_ids']
    for case in selected:
        refpath=Path(p['cohort_root'])/'reference'/(case+'.nii.gz')
        ref,geom=image(refpath,p['input_reference_sha256'][str(refpath)])
        candidates={key:val for (c,key),val in wanted.items() if c==case}
        if not candidates:
            continue
        old=props[case];properties=copy.deepcopy(old['original_properties'])
        properties['shape_before_cropping']=[ref.shape[i] for i in pm.transpose_forward]
        properties['bbox_used_for_cropping']=old['bbox']
        properties['shape_after_cropping_and_before_resampling']=[b-a for a,b in old['bbox']]
        box=tuple(slice(a,b) for a,b in old['bbox'])
        spacing=[properties['spacing'][i] for i in pm.transpose_forward]
        target=old['training_shape']
        exporter=lambda scores:convert_predicted_logits_to_segmentation_with_correct_shape(scores,pm,cm,pm.get_label_manager(dataset),properties,False,num_threads_torch=1)
        per_case,targets,restored=trace_volume(ref,geom,case,sha(refpath),pm.transpose_forward,box,target,
            spacing,cm.spacing,cm.resampling_fn_seg,exporter,candidates)
        previous,_=image(Path(p['parent_roundtrip'])/(case+'.nii.gz'),p['parent_roundtrip_sha256'][case+'.nii.gz'])
        require(np.array_equal(restored,previous),'full label roundtrip reproduction drift')
        for row in per_case:
            row['destination_components']=json.dumps(row['destination_components'])
        for row in targets:row['source_ids']=json.dumps(row['source_ids'])
        require(len(per_case)==len(candidates),'trace identity mismatch')
        all_rows.extend(per_case)
        all_targets.extend(targets)
        print('P0_TRANSPORT_CASE',case,len(per_case),flush=True)
    small=[r for r in all_rows if r['small']]
    missed=[r for r in small if not r['detected']]
    if subset:
        require(bool(all_rows),'empty real trace subset')
        return dict(case_count=len(selected),reference_count=len(all_rows),missed_count=len(missed),categories=dict(Counter(r['transport_category'] for r in all_rows)))
    require(len(all_rows)==len(wanted) and len(small)==286 and len(missed)==28,'trace denominator mismatch')
    write_csv(ROOT/'results/transport/ALL_ORIGINAL_WT_SOURCES.csv',all_rows)
    write_csv(ROOT/'results/transport/ALL_TARGET_ADJACENCY.csv',all_targets)
    write_csv(ROOT/'results/transport/ALL_286.csv',small)
    write_csv(ROOT/'results/transport/MISSED_28.csv',missed)
    emit(ROOT/'evidence/TRANSPORT_VALIDATION.json',dict(status='pass',case_count=103,small_reference_count=286,missed_count=28,
        all_original_wt_sources=len(all_rows),all_target_components=len(all_targets),
        categories=dict(Counter(r['transport_category'] for r in small)),missed_categories=dict(Counter(r['transport_category'] for r in missed)),
        scope='fixed pipeline support/adjacency trace; ambiguous classes retained, not theoretical upper bound'))


def training_inventory(p):
    import numpy as np
    mapping={r['nnunet_case_id']:r for r in p['mapping_rows']}
    train=set(p['train_ids']);val=set(p['val_ids'])
    require(len(train)==1035 and len(val)==103 and not train&val,'split drift')
    require(len(mapping)==len(p['mapping_rows']) and train|val<=set(mapping),'mapping duplicate/missing')
    require(not {mapping[c]['patient_group'] for c in train}&{mapping[c]['patient_group'] for c in val},'patient leakage')
    records=[];case_counts=[]
    for index,item in enumerate(p['train_labels'],1):
        case=item['case'];row=mapping[case]
        require(case in train and row['label_source'] in ('raw','corrected'),'unknown full-label provenance')
        arr,geom=image(item['path'],item['sha256'])
        totals={}
        for region,labels in REGIONS.items():
            _,items=components(np.isin(arr,labels));small_count=0
            for _,coords in items:
                volume=float(len(coords)*np.prod(geom['spacing']))
                small=volume<27;small_count+=int(small)
                records.append(dict(nnunet_case_id=case,source_case_id=row['source_case_id'],patient_group=row['patient_group'],
                    region=region,stable_reference_id=stable_id(row['source_case_id'],region,item['sha256'],geom,coords),
                    original_voxel_count=len(coords),original_volume_mm3=volume,small=small,label_sha256=item['sha256'],
                    label_path=item['path'],geometry_json=json.dumps(geom,sort_keys=True),label_source=row['label_source'],
                    declared_seg_source_path=row['seg_source_path'],
                    completion_required=row['completion_required'],t2w_status=row['t2w_status']))
            totals[region+'_instances']=len(items);totals[region+'_small']=small_count
        case_counts.append(dict(nnunet_case_id=case,source_case_id=row['source_case_id'],**totals))
        if index%25==0 or index==1035:
            print('P0_TRAIN_LABEL',index,'/1035',flush=True)
    require(len({r['stable_reference_id'] for r in records})==len(records),'training identity collision')
    write_csv(ROOT/'results/train_instances/FULL_ORIGINAL_REGIONAL_INSTANCES.csv',records)
    write_csv(ROOT/'results/train_instances/CASE_COUNTS_1035.csv',case_counts)
    small=[r for r in records if r['small']]
    write_csv(ROOT/'results/train_instances/SMALL_ORIGINAL_REGIONAL_INSTANCES.csv',small)
    emit(ROOT/'evidence/TRAIN_INSTANCE_VALIDATION.json',dict(status='pass',train_cases=1035,validation_cases=103,
        full_instances=len(records),small_instances=len(small),small_by_region=dict(Counter(r['region'] for r in small)),
        eligibility='original complete region physical volume <27mm3, no preprocessed fragment qualification',
        historical_exposure='unavailable_not_inferred_from_inventory'))


def expand_identity(p):
    import numpy as np
    methods=('K1','K4','B175','D50')
    prior={m:{r['subject_name']:r for r in json.loads((ROOT/'sources'/m/'panoptica_evaluation_summary.json').read_text())['metrics']} for m in methods}
    all_rows=[];counts=[]
    original=list(csv.DictReader(Path(p['parent_matrix']).open()))
    baseline={(r['subject_id'],r['region'],r['stable_reference_id'],r['method']):r for r in original if r['method'] in ('K1','K4')}
    for case in p['case_ids']:
        refpath=Path(p['cohort_root'])/'reference'/(case+'.nii.gz')
        ref,geom=image(refpath,p['input_reference_sha256'][str(refpath)])
        for method in methods:
            path=Path(p['prediction_roots'][method])/(case+'.nii.gz')
            pred,pgeom=image(path,p['prediction_sha256'][method][path.name])
            require(geom==pgeom,'prediction geometry drift')
            for region,labels in REGIONS.items():
                refs,fp=match_arrays(np.isin(pred,labels).astype(np.uint8),np.isin(ref,labels).astype(np.uint8))
                validate_reference_fingerprint(refs,fp,prior[method][case+'.nii.gz'][region],float(np.prod(geom['spacing'])))
                tp=fn=0
                for record in refs:
                    key=stable_id(case,region,sha(refpath),geom,record['coords'])
                    volume=float(record['voxel_count']*np.prod(geom['spacing']))
                    hit=bool(record['is_matched'] and record['sq_dsc']>=.2)
                    if volume<27:
                        tp+=int(hit);fn+=int(not hit)
                    if method in ('K1','K4'):
                        previous=baseline[(case,region,key,method)]
                        require((previous['detected']=='True')==hit,'r2 identity detection drift')
                    all_rows.append(dict(subject_id=case,region=region,stable_reference_id=key,method=method,
                        detected=hit,is_matched=record['is_matched'],sq_dsc=record['sq_dsc'],
                        original_voxel_count=record['voxel_count'],original_volume_mm3=volume,small=volume<27,
                        reference_sha256=sha(refpath),geometry_json=json.dumps(geom,sort_keys=True)))
                counts.append(dict(subject_id=case,region=region,method=method,small_tp=tp,small_fn=fn,all_fp=fp))
        print('P0_IDENTITY_EXPANDED',case,flush=True)
    require(len(all_rows)==2254*4 and len(counts)==103*4*4,'expanded identity coverage')
    write_csv(ROOT/'results/identity/FOUR_ROUTE_MATRIX.csv',all_rows)
    write_csv(ROOT/'results/identity/FOUR_ROUTE_CASE_COUNTS.csv',counts)
    small={}
    for row in all_rows:
        if row['region']=='wt' and row['small']:
            small.setdefault(row['stable_reference_id'],{})[row['method']]=row['detected']
    require(len(small)==286 and all(set(v)==set(methods) for v in small.values()),'small4 matrix')
    pairs=[]
    import itertools
    for a,b in itertools.combinations(methods,2):
        pairs.append(dict(method_a=a,method_b=b,reference_count=286,
            common_missed=sum(not v[a] and not v[b] for v in small.values()),
            new_b=sum(v[b] and not v[a] for v in small.values()),lost_b=sum(v[a] and not v[b] for v in small.values())))
    write_csv(ROOT/'results/identity/SMALL_WT_PAIRS.csv',pairs)
    emit(ROOT/'evidence/IDENTITY_EXPANSION_VALIDATION.json',dict(status='pass_available_four_routes',case_count=103,
        methods=list(methods),reference_count=2254,records=len(all_rows),small_wt_count=286,
        original_E='blocked_original_prediction_mask_not_located_not_invented',five_route_complete=False))


def preflight():
    p=plan()
    from cpu_roundtrip import preflight as old_preflight
    old_preflight()
    import subprocess
    subprocess.run([p['eval_python'],'-B',str(ROOT/'runtime/supplement.py'),'eval_preflight'],check=True,timeout=180)
    require(sha(Path(p['parent_geometry']))==p['parent_geometry_sha256'],'r2 geometry drift')
    require(sha(Path(p['parent_matrix']))==p['parent_matrix_sha256'],'r2 matrix drift')
    # Real frozen export plus empty/crop tests from r2, and new policy boundaries.
    subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],check=True,timeout=60)
    subset=trace(subset=True)
    from native_transport_tests import run as native_tests
    native=native_tests()
    emit(ROOT/'evidence/SUPPLEMENT_STATIC_PREFLIGHT.json',dict(status='pass',plan_sha256=sha(ROOT/'CPU_EXECUTION_PLAN.json'),
        real_r2_boundary_tests=True,all_source_native_tests=native,actual_transport_subset=subset,scope='technical_preflight_not_completed_full_train_or_transport'))


def eval_preflight():
    p=plan()
    for method in ('B175','D50'):
        prior=json.loads((ROOT/'sources'/method/'panoptica_evaluation_summary.json').read_text())
        require(not prior['missings'] and len(prior['metrics'])==103,'official source count')
        first=p['case_ids'][0]
        ref,geom=image(Path(p['cohort_root'])/'reference'/(first+'.nii.gz'))
        pred,pgeom=image(Path(p['prediction_roots'][method])/(first+'.nii.gz'),p['prediction_sha256'][method][first+'.nii.gz'])
        require(geom==pgeom,'preflight geometry')
        case=next(r for r in prior['metrics'] if r['subject_name']==first+'.nii.gz')
        import numpy as np
        for region,labels in REGIONS.items():
            rows,fp=match_arrays(np.isin(pred,labels).astype(np.uint8),np.isin(ref,labels).astype(np.uint8))
            validate_reference_fingerprint(rows,fp,case[region],float(np.prod(geom['spacing'])))
    print('SUPPLEMENT_EVAL_SUBSET_PASS',flush=True)


def run():
    p=plan()
    emit(ROOT/'evidence/SUPPLEMENT_STARTED.json',dict(status='started',job_id=os.environ.get('SLURM_JOB_ID'),hostname=os.uname().nodename,
        plan_sha256=sha(ROOT/'CPU_EXECUTION_PLAN.json')))
    trace()
    import subprocess
    subprocess.run([p['eval_python'],'-B',str(ROOT/'runtime/supplement.py'),'eval_run'],check=True)
    emit(ROOT/'evidence/SUPPLEMENT_VALIDATION.json',dict(status='pass_cpu_scope_only',train_cases=1035,
        transport_references=286,transport_missed=28,available_identity_routes=4,five_route_complete=False,
        original_E='blocked_missing_original_prediction_masks',probability='not_run_capacity_hold',exposure_gradient='not_run'))
    with (ROOT/'evidence/SUPPLEMENT_COMPLETE.ok').open('x') as f:
        f.write('status=pass_cpu_scope_only\nartifact_status=experimental_unvalidated\noperator_approved=false\nformal_gate_status=not_run_not_passed\n')
    light=[f for name in ('results','evidence') for f in (ROOT/name).rglob('*') if f.is_file() and f.suffix in ('.json','.csv','.ok','.md')]
    emit(ROOT/'evidence/LIGHTWEIGHT_SHA256.json',dict(status='pass',files={str(f.relative_to(ROOT)):sha(f) for f in light}))


if __name__=='__main__':
    try:
        action=sys.argv[1]
        if action=='preflight':preflight()
        elif action=='eval_preflight':eval_preflight()
        elif action=='run':run()
        elif action=='eval_run':
            p=plan();training_inventory(p);expand_identity(p)
        else:raise ValueError('unknown action')
    except Exception:
        emit(ROOT/'evidence'/('SUPPLEMENT_PREFLIGHT_FAILED.json' if 'preflight' in sys.argv[1] else 'SUPPLEMENT_FAILED.json'),dict(status='failed',error=traceback.format_exc()))
        raise
