"""Inactive audit candidate after transport timeout; not wired to snapshot.py.

No complete result was received. Retained for review, not automatic retry.
Never open image or checkpoint payloads.
"""

REMOTE_CODE = r'''
def audit_metadata(root):
 import collections,csv,math,time
 started=time.monotonic()
 records=[]
 errors=[]
 parents_checked={}
 def require(condition,message):
  if not condition:raise ValueError(message)
 def read_json(path):
  require(path.suffix in ('.json','.ok'),'Non-metadata JSON path')
  return json.loads(path.read_text())
 def csv_rows(path):
  require(path.suffix=='.csv','Non-CSV path')
  with path.open(newline='') as stream:return list(csv.DictReader(stream))
 def row_key(row):
  return tuple(sorted((k,str(v)) for k,v in row.items() if k!='patient_n_crops'))
 def finite(value):
  if isinstance(value,float):return math.isfinite(value)
  if isinstance(value,dict):return all(finite(v) for v in value.values())
  if isinstance(value,list):return all(finite(v) for v in value)
  return True
 complete_path=root/'dataset/COMPLETE.ok'
 if not complete_path.is_file():return {'status':'not_available','reason':'No dataset COMPLETE'}
 try:
  cfg=read_json(root/'config/RUN.json')
  complete=read_json(complete_path)
  report_path=root/'dataset/PREPROCESS_REPORT.json'
  report=read_json(report_path)
  require(complete['status']==report['status']=='complete','Incomplete reports')
  require(complete['cases']==report['cases_ok']==report['cases_processed']==926,'Report count drift')
  require(complete['split_counts']=={'train':823,'val':103},'Report split drift')
  require(not (root/'dataset/FAILED.json').exists() and not report['errors'],'Dataset errors')
  require(complete['report_sha256']==sha(report_path),'Report SHA drift')
  require(report['warnings']==complete['resolution_warnings'],'Report warning drift')
  for key in ('loss_policy','label_resampling_algorithm','support_contract'):
   require(complete[key]==cfg[key],'Policy drift: '+key)
  require(cfg['loss_policy']=='uniform_original_quantization_warning_v1','Unexpected loss policy')
  require(cfg['label_resampling_algorithm']=='original_physical_spacing_v2_labelwise_threshold_no_case_exception','Unexpected algorithm')
  require(not complete.get('local_exception_sha256') and not (root/'plan/LOCAL_LABEL_EXCEPTION.json').exists(),'Local exception forbidden')
  require(complete['STRICT_ORIGINAL_GATE'] is False and complete['NONZERO_COMPARISON_GATE'] is False,'Historical gate drift')
  require(complete['independent_full_revalidation']=='waived_by_user_not_pass','Waiver drift')
  membership_rows=csv_rows(root/'plan/splits_final_train_val_test_membership.csv')
  membership={r['source_case_id']:r for r in membership_rows}
  require(len(membership)==len(membership_rows)==1030,'Membership identity/count drift')
  require(collections.Counter(r['split'] for r in membership_rows)=={'train':823,'val':103,'test':104},'Membership split drift')
  groups=collections.defaultdict(set)
  for row in membership_rows:
   groups[row['patient_group']].add(row['split'])
   require(row['t2w_status']=='authentic' and row['eligible_for_realonly'].lower()=='true','Membership eligibility drift')
  require(all(len(v)==1 for v in groups.values()),'Patient leakage')
  members={k:v for k,v in membership.items() if v['split']!='test'}
  details={r['case']:r for r in complete['cases_detail']}
  require(len(details)==len(complete['cases_detail'])==926,'Case detail duplicate/count drift')
  require(set(details)==set(members)==set(complete['case_manifests'])==set(complete['case_locations']),'Case coverage drift')
  require({p.name for p in (root/'dataset/indices').glob('*.json')}=={k+'.json' for k in members},'Index coverage drift')
  plan=read_json(root/'plan/REUSE_CANDIDATES.json')
  candidates={r['case_id']:r for r in plan['candidates']}
  require(len(candidates)==len(plan['candidates'])==514,'Candidate count drift')
  require(complete['reference_parents']==plan['parents'],'Reference parent drift')
  for name,digest in plan['parents'].items():
   parent=pathlib.Path(name)
   require(parent.is_absolute() and parent.resolve()==parent,'Parent path redirect')
   manifest_path=parent/'plan/PACKAGE_MANIFEST.json'
   require(sha(manifest_path)==digest,'Parent package SHA drift')
   package=read_json(manifest_path)
   for relative,expected in package['files'].items():
    path=parent/relative
    require(not pathlib.Path(relative).is_absolute() and path.resolve().is_relative_to(parent),'Parent package path escape')
    require(not path.name.endswith(('.nii','.nii.gz','.pt','.pth','.ckpt')),'Payload in parent package')
    require(sha(path)==expected,'Parent package file drift: '+relative)
   parents_checked[name]={'package_sha256':digest,'files_checked':len(package['files'])}
  source_path=pathlib.Path(cfg['source_code'])/'data/data_csv.csv'
  require(sha(source_path)==cfg['source_csv_sha256'],'Source CSV SHA drift')
  source_rows=csv_rows(source_path)
  source={r['id']:r for r in source_rows}
  require(len(source)==len(source_rows),'Source CSV duplicate')
  recon=pathlib.Path(cfg['recon'])/'dataset'
  native_complete=read_json(recon/'COMPLETE.ok')
  require(native_complete['status']=='pass' and native_complete['checkpoint_sha256']=='5d86a0b3cbc5df34d64d5a0ff376eae453634a609bd24dec1a1191fde800b13b','Native manifest identity drift')
  csv_path=root/'dataset/lesions.csv'
  require(sha(csv_path)==complete['csv_sha256'],'Current CSV SHA drift')
  rows=csv_rows(csv_path)
  require(len(rows)==complete['rows'] and sum(r['split']=='train' for r in rows)==complete['train_rows'],'Lesion row count drift')
  by_case=collections.defaultdict(list)
  crop_counts=collections.Counter(r['patient_id'] for r in rows)
  modalities=('t1c','t1n','t2w','t2f')
  for row in rows:
   case=row['source_case_id']
   require(case in members,'CSV includes non-train/val case')
   member=members[case]
   require(row['split']==member['split'] and row['patient_group']==member['patient_group'],'CSV membership drift')
   require(row['patient_id']==case[-9:] and int(row['t2w_valid'])==1,'CSV patient/target drift')
   require(int(row['patient_n_crops'])==crop_counts[row['patient_id']],'Patient crop count drift')
   for key,suffix in [('label','seg')]+[('scan_'+m,m) for m in modalities]:
    require(row[key]==str(root/'dataset/cases'/case/(case+'-'+suffix+'.nii.gz')),'CSV path mapping drift')
   by_case[case].append(row)
  warnings=[]
  origins=collections.Counter()
  for case in sorted(members):
   try:
    member=members[case]
    detail=details[case]
    directory=root/'dataset/cases'/case
    require(directory.is_dir() and not directory.is_symlink() and directory.resolve()==directory,'Case directory redirect')
    require(complete['case_locations'][case]==detail['resolved_directory']==str(directory),'Case location drift')
    marker_path=directory/'preprocess_complete.json'
    marker_sha=sha(marker_path)
    require(not marker_path.is_symlink() and marker_sha==complete['case_manifests'][case]==detail['marker_sha256'],'Marker SHA drift')
    marker=read_json(marker_path)
    require(marker['status']=='complete' and marker['case_id']==case,'Marker identity drift')
    require(marker['split']==detail['split']==source[case]['split']==member['split'] and marker['patient_group']==member['patient_group'],'Case split/group drift')
    contract=marker['contract']
    for record in (marker,contract,detail):
     require(not record.get('local_label_exception') and not record.get('local_exception'),'Case label exception forbidden')
    require(finite(contract),'Nonfinite metadata')
    require(contract['loss_policy']==cfg['loss_policy'] and contract['label_resampling_algorithm']==cfg['label_resampling_algorithm'],'Case quantization policy drift')
    require(contract['support_contract']==cfg['support_contract'] and contract['grid_version']=='physical_spacing_v2','Case geometry contract drift')
    require(contract['target_spacing']==[1.,1.,1.] and contract['bbox']==[[0,int(n)] for n in contract['original_shape']],'Grid/FOV metadata drift')
    require(contract['normalization']=='whole_case_common_mask_zscore_before_resampling_zero_outside_support','Normalization drift')
    require(contract['mask_algorithm']=='authentic_T1N_T1C_T2F_robust_0_100_nonzero_union_holefill','Mask contract drift')
    spatial_path=directory/'spatial_mapping.json'
    spatial=read_json(spatial_path)
    require(all(spatial.get(k)==v for k,v in contract.items()),'Spatial mapping drift')
    native_path=recon/'cases'/case/'case_manifest.json'
    native_sha=sha(native_path)
    require(native_sha==marker['native_case_manifest_sha256']==native_complete['manifest_sha256'][case],'Native marker SHA drift')
    native=read_json(native_path)
    require(native['case_id']==case and native['split']==member['split'] and native['geometry']=='native','Native metadata identity drift')
    require(native['checkpoint_sha256']==native_complete['checkpoint_sha256'] and native['native_shape']==contract['original_shape'],'Native shape/weight identity drift')
    require(set(marker['input_sha256'])==set(modalities+('seg',)),'Native input set drift')
    require(all(h==native['files'][m]['sha256'] for m,h in marker['input_sha256'].items()),'Recorded input SHA drift')
    require(marker['input_sha256']['seg']==native['source_sha256']['seg'],'Recorded label source drift')
    require(set(marker['authentic_mask_source_sha256'])=={'t1n','t1c','t2f'},'Mask source set drift')
    require(all(h==native['source_sha256'][m] for m,h in marker['authentic_mask_source_sha256'].items()),'Recorded mask source SHA drift')
    lost=[str(k) for k in range(1,5) if contract['label_native'][str(k)]['voxels'] and not contract['label_target_1mm'][str(k)]['voxels']]
    require(lost==contract['lost_classes']==detail['lost_classes'] and contract['resolution_warning']==bool(lost),'Warning counts drift')
    if lost:warnings.append({'case':case,'lost_classes':lost,'label_native':contract['label_native'],'label_target_1mm':contract['label_target_1mm']})
    expected={case+'-'+m+'.nii.gz' for m in modalities+('seg','brainmask')}
    require(set(marker['outputs'])==expected and detail['outputs']==marker['outputs'],'Six-output recorded SHA set drift')
    references=detail['file_references']
    require(marker.get('file_references',{})==references,'Recorded references drift')
    if case in candidates:
     candidate=candidates[case]
     parent_directory=pathlib.Path(candidate['directory'])
     require(parent_directory in {pathlib.Path(p)/'dataset/cases'/case for p in plan['parents']},'Candidate parent path drift')
     parent_marker_path=parent_directory/'preprocess_complete.json'
     require(sha(parent_marker_path)==marker['parent_marker_sha256']==candidate['marker_sha256'],'Parent marker SHA drift')
     require(marker['parent_directory']==str(parent_directory),'Parent directory record drift')
     parent_marker=read_json(parent_marker_path)
     require(parent_marker['case_id']==case and parent_marker['split']==member['split'] and parent_marker['native_case_manifest_sha256']==native_sha,'Parent/native identity drift')
     rebuild=candidate['mode']=='label_only_rebuild'
     require(candidate['mode'] in ('label_only_rebuild','all_outputs_reference'),'Unknown candidate mode')
     require(not rebuild or case=='BraTS-MET-01110-002','Unexpected label-only rebuild')
     require(set(references)==expected-({case+'-seg.nii.gz'} if rebuild else set()),'Reference coverage drift')
     origin='label_only_rebuild' if rebuild else 'readonly_file_reference'
     require(detail['origin']==origin,'Reference origin drift')
     require(all(parent_marker['outputs'][name]==reference['sha256'] for name,reference in references.items()),'Parent/reference recorded output SHA drift')
    else:
     require(not references and detail['origin']=='new_preprocessing','New case reference drift')
     origin='new_preprocessing'
    for name,digest in marker['outputs'].items():
     path=directory/name
     require(path.is_file() and path.stat().st_size>0,'Missing/empty output')
     require(isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest) is not None,'Invalid recorded content SHA')
     if name in references:
      reference=references[name]
      target=pathlib.Path(reference['path'])
      require(path.is_symlink() and target.is_absolute() and target.resolve()==target and path.resolve()==target,'Actual link target drift')
      require(target in {(pathlib.Path(p)/'dataset/cases'/case/name).resolve() for p in plan['parents']},'Link outside frozen parent/case')
      require(reference['sha256']==digest,'Recorded link content SHA drift')
     else:require(not path.is_symlink() and path.resolve()==path,'Undeclared output link')
    index_path=root/'dataset/indices'/(case+'.json')
    index=read_json(index_path)
    require(index['case_id']==case and index['marker_sha256']==marker_sha,'Index marker drift')
    require(collections.Counter(map(row_key,index['rows']))==collections.Counter(map(row_key,by_case[case])),'Index/current CSV row drift')
    origins[origin]+=1
    records.append({'case':case,'split':member['split'],'origin':origin,'marker_sha256':marker_sha,'index_sha256':sha(index_path),'spatial_sha256':sha(spatial_path),'native_manifest_sha256':native_sha,'csv_rows':len(by_case[case]),'outputs_recorded_sha256':marker['outputs'],'file_references':references,'lost_classes':lost})
   except Exception as error:errors.append({'case':case,'message':str(error)})
  require(not errors,'Per-case metadata errors collected')
  require(origins=={'readonly_file_reference':513,'label_only_rebuild':1,'new_preprocessing':412},'Reference/rebuild origin counts drift')
  require([{'case':w['case'],'lost_classes':w['lost_classes']} for w in warnings]==sorted(report['warnings'],key=lambda r:r['case']),'Full warning identity drift')
  require(sorted(k for k in members if not by_case[k])==complete['cases_without_lesion_rows'],'Zero-row case drift')
  return {'status':'pass','scope':'metadata_SHA_and_actual_link_targets_only','payload_content_rehashed':False,'source_images_read':0,'test_images_read':0,'checkpoint_payloads_read':0,'cases_checked':len(records),'split_counts':dict(collections.Counter(r['split'] for r in records)),'origin_counts':dict(origins),'csv_rows':len(rows),'csv_sha256':sha(csv_path),'complete_sha256':sha(complete_path),'report_sha256':sha(report_path),'finished_utc':complete['finished_utc'],'parents_checked':parents_checked,'warnings':warnings,'errors':errors,'case_records':records,'duration_seconds':time.monotonic()-started}
 except Exception as error:
  return {'status':'failed','scope':'metadata_SHA_and_actual_link_targets_only','payload_content_rehashed':False,'source_images_read':0,'test_images_read':0,'checkpoint_payloads_read':0,'message':str(error),'errors':errors,'cases_checked':len(records),'case_records':records,'parents_checked':parents_checked,'duration_seconds':time.monotonic()-started}
'''
