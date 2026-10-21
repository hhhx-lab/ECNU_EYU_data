"""Read-only class-loss diagnosis; no source images, resubmission, or label edits."""
import datetime
import hashlib
import json
import shlex
from pathlib import Path

from common import REMOTE, m, transport


def remote_code():
    code = r'''
import collections,csv,datetime,hashlib,importlib.util,json,os,pathlib,sys,time
import nibabel as nib
import numpy as np
from scipy import ndimage
root=pathlib.Path(__ROOT__)
start=time.monotonic()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
remote_start_utc=now()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def bounded():
 if time.monotonic()-start>170:raise TimeoutError('One diagnostic exceeded 170s bound')
cfg=read(root/'config/RUN.json')
package=read(root/'plan/PACKAGE_MANIFEST.json')
drift=[n for n,h in package['files'].items() if not (root/n).is_file() or sha(root/n)!=h]
assert not drift
members=list(csv.DictReader((root/'plan/splits_final_train_val_test_membership.csv').open()))
active={r['source_case_id']:r for r in members if r['split'] in ('train','val')}
assert len(active)==926
source_csv=pathlib.Path(cfg['source_code'])/'data/data_csv.csv'
assert sha(source_csv)==cfg['source_csv_sha256']
sources={r['id']:r for r in csv.DictReader(source_csv.open())}
recon=pathlib.Path(cfg['recon'])/'dataset'
complete=read(recon/'COMPLETE.ok')
assert complete['status']=='pass'
spec=importlib.util.spec_from_file_location('readonly_label_grid',root/'runtime/upstream_source/src/preprocess/nnunet_like.py')
grid=importlib.util.module_from_spec(spec);spec.loader.exec_module(grid)
case='BraTS-MET-01132-002';cls=2
assert case in active
native=recon/'cases'/case
cm=read(native/'case_manifest.json')
assert sha(native/'case_manifest.json')==complete['manifest_sha256'][case]
assert cm['case_id']==case and cm['split']==active[case]['split']
src=pathlib.Path(cfg['source_code'])/'data/input'/case/pathlib.Path(sources[case]['seg']).name
rec=native/(case+'-seg.nii.gz')
source_sha=sha(src);recon_sha=sha(rec)
assert source_sha==recon_sha==cm['source_sha256']['seg']==cm['files']['seg']['sha256']
ref=nib.load(str(rec));src_ref=nib.load(str(src))
raw=ref.get_fdata(dtype=np.float32)
assert np.isfinite(raw).all() and np.equal(raw,np.rint(raw)).all() and np.isin(raw,[0,1,2,3,4]).all()
label=raw.astype(np.uint8);del raw
spacing=np.asarray(ref.header.get_zooms()[:3],float)
shape,factor,offset=grid.physical_grid(label.shape,spacing,(1.,1.,1.))
assert int(np.prod(shape))<50000000
bounded()
default=grid.resample_physical(label[None],shape,factor,offset,spacing,(1.,1.,1.),is_seg=True)[0]
nearest=ndimage.affine_transform(label,np.diag(factor),offset,output_shape=tuple(shape),order=0,mode='nearest')
counts=lambda a:{str(k):int(np.count_nonzero(a==k)) for k in range(5)}
native_counts=counts(label);default_counts=counts(default);nearest_counts=counts(nearest)
points=np.argwhere(label==cls);assert len(points)
target=np.argwhere(nearest==cls)
aniso=None
for candidate in (spacing,np.ones(3)):
 if grid.get_do_separate_z(candidate):
  axes=grid.get_lowres_axis(candidate)
  if len(axes)==1:aniso=int(axes[0])
  break
probability={'anisotropic_separate_axis':aniso}
if aniso is None:
 projection=(points+.5)*spacing-.5
 lo=np.maximum(0,np.floor(projection.min(0)).astype(int)-2)
 hi=np.minimum(shape,np.ceil(projection.max(0)).astype(int)+3)
 roi=hi-lo
 if int(np.prod(roi))<1000000:
  def prob(k):return ndimage.affine_transform((label==k).astype(np.float32),np.diag(factor),offset+lo*factor,output_shape=tuple(roi),order=1,mode='nearest',prefilter=False)
  pcls=prob(cls);pbg=prob(0)
  samples=[]
  for q in target[:64]:
   r=q-lo
   if np.all(r>=0) and np.all(r<roi):
    continuous=q*factor+offset
    samples.append(dict(target_ijk=q.tolist(),source_nearest_ijk=np.floor(continuous+.5).astype(int).tolist(),probability_class=float(pcls[tuple(r)]),probability_background=float(pbg[tuple(r)]),default_label=int(default[tuple(q)])))
  probability.update(roi_origin=lo.tolist(),roi_shape=roi.tolist(),class_probability_max=float(pcls.max()),threshold_hits=int(np.count_nonzero(pcls>=.5)),samples=samples)
 else:probability['not_evaluated']='ROI exceeds one million voxels'
else:probability['not_evaluated']='No isotropic probability claim for anisotropic path'
components,n=ndimage.label(label==cls)
sizes=np.bincount(components.ravel())[1:]
report=dict(case_id=case,split=active[case]['split'],source_seg_sha256=source_sha,recon_seg_sha256=recon_sha,
 native_manifest_sha256=sha(native/'case_manifest.json'),native_shape=list(ref.shape),source_shape=list(src_ref.shape),native_spacing_mm=spacing.tolist(),
 source_recon_affine_max_abs=float(np.abs(src_ref.affine-ref.affine).max()),native_affine=ref.affine.tolist(),target_shape=shape.tolist(),factor=factor.tolist(),offset=offset.tolist(),
 native_counts=native_counts,default_counts=default_counts,pure_nearest_counts=nearest_counts,
 native_class2_volume_mm3=float(native_counts['2']*np.prod(spacing)),class2_components_6_connected=n,component_sizes_voxels=sorted(sizes.tolist(),reverse=True),
 native_class2_bbox=[points.min(0).tolist(),points.max(0).tolist()],nearest_class2_target_first64=target[:64].tolist(),
 nearest_class2_overwritten_to_background=int(np.count_nonzero((nearest==cls)&(default==0))),nearest_class2_overwritten_to_other_foreground=int(np.count_nonzero((nearest==cls)&(default!=0)&(default!=cls))),
 default_lost_classes=[str(k) for k in range(1,5) if native_counts[str(k)] and not default_counts[str(k)]],
 nearest_lost_classes=[str(k) for k in range(1,5) if native_counts[str(k)] and not nearest_counts[str(k)]],probability=probability,
 pure_nearest_diagnostic_only=True,no_label_modified=True)
del label,default,nearest,components
bounded()
inventory=[];errors=[];incomplete=[]
for p in sorted((root/'dataset/cases').iterdir()):
 bounded()
 marker=p/'preprocess_complete.json';index=root/'dataset/indices'/(p.name+'.json')
 if not marker.is_file():incomplete.append(p.name);continue
 try:
  v=read(marker);member=active[p.name]
  assert v['case_id']==p.name and v['status']=='complete' and v['split']==member['split']
  assert v['patient_group']==member['patient_group']
  expected={p.name+'-'+k+'.nii.gz' for k in ('t1c','t1n','t2w','t2f','seg','brainmask')}
  assert set(v['outputs'])==expected and all((p/n).is_file() and (p/n).stat().st_size>0 for n in expected)
  index_ok=False
  if index.is_file():
   idx=read(index);assert idx['case_id']==p.name and idx['marker_sha256']==sha(marker);index_ok=True
  inventory.append(dict(case_id=p.name,split=member['split'],symlink=p.is_symlink(),resolved_directory=str(p.resolve()),
   marker_sha256=sha(marker),index_present_and_consistent=index_ok,index_sha256=sha(index) if index_ok else None,
   logical_output_bytes=sum((p/n).stat().st_size for n in expected),contract=v['contract'],content_sha_not_reverified=True))
 except (AssertionError,KeyError,ValueError,OSError) as e:errors.append(dict(case_id=p.name,error=type(e).__name__+': '+str(e)))
result=dict(remote_start_utc=remote_start_utc,hostname=os.uname().nodename,scope='one failed source label resampling + existing outputs metadata only; no four-modality/full-cohort rescan',
 package_sha256=sha(root/'plan/PACKAGE_MANIFEST.json'),drift=drift,single_case=report,
 existing_outputs=dict(metadata_candidates=len(inventory),symlink_candidates=sum(r['symlink'] for r in inventory),index_candidates=sum(r['index_present_and_consistent'] for r in inventory),
  candidates=inventory,incomplete=incomplete,errors=errors,full_content_sha_not_reverified=True),
 failed=read(root/'dataset/FAILED.json'),failed_sha256=sha(root/'dataset/FAILED.json'),available_bytes=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize,
 remote_end_utc=now(),seconds=round(time.monotonic()-start,3),test_images_read=0,source_images_read=0)
print(json.dumps(result))
'''
    return code.replace('__ROOT__', repr(str(REMOTE)))


def main():
    code = remote_code()
    compile(code, 'class2_readonly_diagnostic', 'exec')
    access = transport('diagnose_class2')
    local_start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        access.start_vpn()
        out = access.ssh('env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 '
                         + m.PYTHON + ' -B -u -c ' + shlex.quote(code), timeout=200)
        value = json.loads(out.strip().splitlines()[-1])
        value.update(m.META)
        value.update(local_start_utc=local_start, local_received_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        m.exclusive(access.audit / 'DIAGNOSTIC.json', value)
    finally:
        access.close_vpn()
    m.exclusive(access.audit / 'LOCAL_OPERATION_RECEIPT.json', dict(
        **m.META, diagnostic_sha256=hashlib.sha256((access.audit / 'DIAGNOSTIC.json').read_bytes()).hexdigest(),
        operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        local_end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), operation_vpn_closed=True,
        remote_writes=False, remote_submissions=False, large_data_downloaded=False))
    print(json.dumps(dict(audit=str(access.audit), single_case=value['single_case'],
        existing_outputs={k: v for k, v in value['existing_outputs'].items() if k != 'candidates'}, seconds=value['seconds']), ensure_ascii=False))


if __name__ == '__main__':
    main()
