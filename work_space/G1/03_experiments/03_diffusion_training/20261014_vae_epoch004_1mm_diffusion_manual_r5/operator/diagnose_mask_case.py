"""One bounded read-only native-mask diagnostic; no remote output or job submission."""
import ast
import base64
import datetime
import json
import shlex

from common import ROOT, REMOTE, m, transport


REMOTE_CODE = r'''
import csv,datetime,hashlib,json,os,pathlib,re,subprocess,sys,traceback
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes
root=pathlib.Path(__ROOT__);case='BraTS-MET-00006-000'
meta={'artifact_status':'experimental_unvalidated','operator_approved':False,'formal_gate_status':'not_run_not_passed'}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def cmd(args):
 r=subprocess.run(args,text=True,capture_output=True,timeout=25)
 return {'argv':args,'returncode':r.returncode,'stdout':r.stdout[-14000:],'stderr':r.stderr[-2000:]}
def count(mask):return int(np.count_nonzero(mask))
def summary(a):
 return {'voxels':int(a.size),'finite':bool(np.isfinite(a).all()),'minimum':float(a.min()) if a.size else None,
  'maximum':float(a.max()) if a.size else None,'negative':count(a<0),'zero':count(a==0),'positive':count(a>0),
  'percentiles':np.percentile(a,[0,1,50,99,100]).tolist() if a.size else []}
def points(mask,volumes,reference):
 indices=np.argwhere(mask)
 if not len(indices):return []
 # Evenly spaced examples are descriptive only; decisions use full voxel counts.
 chosen=indices[np.unique(np.linspace(0,len(indices)-1,min(24,len(indices)),dtype=int))]
 return [{'ijk':p.tolist(),'xyz_mm':(reference.affine @ np.r_[p,1])[:3].tolist(),
          'label':int(label[tuple(p)]),'source_values':{k:float(v[tuple(p)]) for k,v in volumes.items()}}
         for p in chosen]
result={**meta,'status':'diagnostic_started','remote_utc':now(),'hostname':os.uname().nodename,'case':case}
try:
 result['squeue']=cmd(['squeue','-h','-j','4086739,4069916,4069917,4069918,4069919','-o','%i|%T|%j|%P|%r|%M'])
 result['sacct']=cmd(['sacct','-n','-P','-j','4086739,4069916,4069917,4069918,4069919','--format=JobID,State,ExitCode,Elapsed'])
 result['account_disk']=cmd(['df','-B1',str(root)])
 result['public_disk']=cmd(['df','-h','/public'])
 result['available_bytes']=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize
 result['failed_marker']=json.loads((root/'dataset/FAILED.json').read_text())
 result['failed_marker_sha256']=sha(root/'dataset/FAILED.json')
 result['complete_exists']=(root/'dataset/COMPLETE.ok').exists()
 cfg=json.loads((root/'config/RUN.json').read_text())
 source=pathlib.Path(cfg['source_code'])
 assert sha(source/'data/data_csv.csv')==cfg['source_csv_sha256']
 result['source_csv_sha256']=cfg['source_csv_sha256']
 package=json.loads((root/'plan/PACKAGE_MANIFEST.json').read_text())
 result['package_sha256']=sha(root/'plan/PACKAGE_MANIFEST.json')
 assert result['package_sha256']=='13f188b2609ba57544c1258aa368daeabe0201ecbdb81c3ec86700f6441b6ede'
 result['drift']=[p for p,v in package['files'].items() if not (root/p).is_file() or sha(root/p)!=v]
 assert not result['drift']
 binding=json.loads((root/'evidence/PREPARE_REPLACEMENT_ACTIVE.json').read_text())
 assert sha(pathlib.Path(binding['remote_audit'])/'prepare_cpu192.slurm')==binding['script_sha256']
 result['replacement_binding']=binding
 with (source/'data/data_csv.csv').open(newline='') as h:
  row=next(r for r in csv.DictReader(h) if r['id']==case)
 with (root/'plan/splits_final_train_val_test_membership.csv').open(newline='') as h:
  member=next(r for r in csv.DictReader(h) if r['source_case_id']==case)
 assert row['split']==member['split'] and member['split'] in ('train','val')
 result['membership']=member
 native=pathlib.Path(cfg['recon'])/'dataset/cases'/case
 cm=json.loads((native/'case_manifest.json').read_text())
 complete=json.loads((pathlib.Path(cfg['recon'])/'dataset/COMPLETE.ok').read_text())
 assert sha(native/'case_manifest.json')==complete['manifest_sha256'][case]
 assert cm['case_id']==case and cm['split']==member['split'] and cm['geometry']=='native'
 result['case_manifest_sha256']=sha(native/'case_manifest.json')
 ref=nib.load(str(native/(case+'-seg.nii.gz')))
 label=ref.get_fdata(dtype=np.float32)
 assert np.isfinite(label).all() and np.equal(label,np.rint(label)).all() and np.isin(label,[0,1,2,3,4]).all()
 assert list(ref.shape)==cm['native_shape']
 source_paths={k:source/'data/input'/case/pathlib.Path(row[k]).name for k in ('t1n','t1c','t2f','seg')}
 volumes={};details={}
 for k,path in source_paths.items():
  image=nib.load(str(path));a=image.get_fdata(dtype=np.float32)
  digest=sha(path)
  assert digest==cm['source_sha256'][k]
  assert image.shape==ref.shape and np.isfinite(a).all()
  corners=np.array([[i,j,l,1] for i in (0,ref.shape[0]-1) for j in (0,ref.shape[1]-1) for l in (0,ref.shape[2]-1)])
  corner_mm=float(np.linalg.norm((image.affine @ corners.T-ref.affine @ corners.T)[:3],axis=0).max())
  assert corner_mm<=.1
  details[k]={'path':str(path),'sha256':digest,'shape':list(image.shape),'affine':image.affine.tolist(),
              'spacing':list(map(float,image.header.get_zooms()[:3])),'max_corner_shift_mm':corner_mm,'values':summary(a)}
  if k=='seg':assert np.array_equal(a,label)
  else:volumes[k]=a
 native_seg=native/(case+'-seg.nii.gz')
 assert sha(native_seg)==details['seg']['sha256']==cm['files']['seg']['sha256']
 overlay=json.loads((root/'config/P0_SOURCE_MAPPING.json').read_text())['corrected_label_paths']
 result['overlay']={'listed':case in overlay,'path':overlay.get(case)}
 if case in overlay:
  result['overlay']['sha256']=sha(overlay[case])
  assert sha(overlay[case])==sha(native_seg)
 result['reference']={'path':str(native_seg),'sha256':sha(native_seg),'shape':list(ref.shape),
  'affine':ref.affine.tolist(),'spacing':list(map(float,ref.header.get_zooms()[:3]))}
 positive=np.zeros(ref.shape,bool);nonzero=np.zeros(ref.shape,bool);negative=np.zeros(ref.shape,bool)
 for a in volumes.values():
  low,high=np.percentile(a,[0,100]);low=max(0.,float(low))
  if high>low:positive|=np.clip(a,low,high)>low
  nonzero|=a!=0;negative|=a<0
 raw_positive=positive.copy();raw_nonzero=nonzero.copy()
 positive=binary_fill_holes(positive);nonzero=binary_fill_holes(nonzero)
 lesion=label>0;missing=lesion & ~positive
 all_zero=lesion & ~raw_nonzero;negative_recovered=missing & raw_nonzero & negative
 result['source_details']=details
 result['coverage']={'label_voxels':count(lesion),'old_positive_union_voxels':count(raw_positive),
  'old_positive_holefilled_voxels':count(positive),'source_nonzero_union_voxels':count(raw_nonzero),
  'source_nonzero_holefilled_voxels':count(nonzero),'old_missing_label_voxels':count(missing),
  'nonzero_missing_label_voxels':count(lesion & ~nonzero),'all_three_source_zero_label_voxels':count(all_zero),
  'old_missing_with_negative_signal':count(negative_recovered),'old_missing_source_zero':count(missing & ~raw_nonzero),
  'old_missing_recovered_only_by_nonzero_holefill':count(missing & nonzero & ~raw_nonzero),
  'label_counts':{str(k):count(label==k) for k in (0,1,2,3,4)},
  'old_missing_by_label':{str(k):count(missing & (label==k)) for k in (1,2,3,4)},
  'nonzero_missing_by_label':{str(k):count((label==k) & ~nonzero) for k in (1,2,3,4)},
  'source_values_at_old_missing':{k:summary(a[missing]) for k,a in volumes.items()},
  'old_missing_bbox_ijk':np.array([np.argwhere(missing).min(0),np.argwhere(missing).max(0)]).tolist() if missing.any() else None,
  'old_missing_samples':points(missing,volumes,ref),'nonzero_missing_samples':points(lesion & ~nonzero,volumes,ref)}
 result['interpretation']='unresolved'
 if not missing.any():result['interpretation']='failure_not_reproduced_stop'
 elif count(negative_recovered)==count(missing) and not (lesion & ~nonzero).any():
  result['interpretation']='all_old_missing_have_negative_source_signal_nonzero_covers_label'
 elif (lesion & ~nonzero).any():result['interpretation']='nonzero_source_support_also_fails_stop_no_signal_or_boundary_issue'
 else:result['interpretation']='nonzero_recovers_via_holefill_requires_review'
 result['status']='read_only_diagnostic_complete';result['finished_remote_utc']=now()
except BaseException as error:
 result.update(status='diagnostic_failed',message=str(error),traceback=traceback.format_exc(),finished_remote_utc=now())
print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(result,sort_keys=True),flags=re.I))
'''


def main():
    access = transport('diagnose_mask_case')
    code = REMOTE_CODE.replace('__ROOT__', repr(str(REMOTE)))
    ast.parse(code)
    m.exclusive(access.audit / 'AUTHORIZATION.json', {**m.META, 'scope': 'one failing case read-only voxel diagnostic',
        'user_authorization': '只读诊断失败病例；按诊断结果修复并独立新批次',
        'remote_writes': False, 'submission': False, 'test104_read': False})
    try:
        access.start_vpn()
        env = 'env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 '
        output = access.ssh(env + m.PYTHON + ' -B -u -c ' + shlex.quote(code), timeout=240)
        result = json.loads(output.strip().splitlines()[-1])
        result['local_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit / 'CASE_DIAGNOSTIC.json', result)
        print(json.dumps({'audit': str(access.audit), **result}, ensure_ascii=False))
        if result['status'] != 'read_only_diagnostic_complete':
            raise RuntimeError('Read-only case diagnostic failed; preserve evidence')
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
