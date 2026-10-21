"""Bounded read-only single-case resampling diagnosis and reuse candidate inventory."""
import datetime
import hashlib
import json
import shlex

from common import REMOTE, m, transport


def remote_code():
    code = r'''
import collections,csv,datetime,hashlib,importlib.util,json,os,pathlib,subprocess,sys,time
import nibabel as nib
import numpy as np
from scipy import ndimage
root=pathlib.Path(__ROOT__)
case='BraTS-MET-01110-002'
start=time.monotonic()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def bounded():
    if time.monotonic()-start>165:raise TimeoutError('Diagnostic 165s read-only bound reached')
def cmd(a):
    r=subprocess.run(a,capture_output=True,text=True,timeout=20)
    return dict(returncode=r.returncode,stdout=r.stdout[-8000:],stderr=r.stderr[-1000:])
cfg=read(root/'config/RUN.json')
members=list(csv.DictReader((root/'plan/splits_final_train_val_test_membership.csv').open()))
active={r['source_case_id']:r for r in members if r['split'] in ('train','val')}
assert len(active)==926 and case in active
parent=pathlib.Path(cfg['recon'])/'dataset'
parent_complete=read(parent/'COMPLETE.ok')
package=read(root/'plan/PACKAGE_MANIFEST.json')
v=dict(remote_start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=os.uname().nodename,
       case_id=case,split=active[case]['split'],patient_group=active[case]['patient_group'],
       scope='one native label + exact resampling; prior outputs metadata inventory and up to three content-SHA samples; no cohort image scan',
       package_sha256=sha(root/'plan/PACKAGE_MANIFEST.json'),
       drift=[n for n,h in package['files'].items() if not (root/n).is_file() or sha(root/n)!=h],
       squeue=cmd(['squeue','-h','-j','4096309,4096310,4096311,4096312,4096313','-o','%i|%T|%P|%r|%M']),
       sacct=cmd(['sacct','-n','-P','-j','4096309,4096310,4096311,4096312,4096313','--format=JobID,State,ExitCode,Elapsed']),
       failed_marker=read(root/'dataset/FAILED.json'),failed_marker_sha256=sha(root/'dataset/FAILED.json'))
assert not v['drift'] and parent_complete['status']=='pass'
native=parent/'cases'/case
cm_path=native/'case_manifest.json';cm=read(cm_path)
assert sha(cm_path)==parent_complete['manifest_sha256'][case]
assert cm['case_id']==case and cm['split']==active[case]['split']
source_csv=pathlib.Path(cfg['source_code'])/'data/data_csv.csv'
assert sha(source_csv)==cfg['source_csv_sha256']
source=next(r for r in csv.DictReader(source_csv.open()) if r['id']==case)
src_seg=pathlib.Path(cfg['source_code'])/'data/input'/case/pathlib.Path(source['seg']).name
recon_seg=native/(case+'-seg.nii.gz')
source_hash=sha(src_seg);recon_hash=sha(recon_seg)
assert source_hash==recon_hash==cm['source_sha256']['seg']==cm['files']['seg']['sha256']
ref=nib.load(str(recon_seg));source_ref=nib.load(str(src_seg))
raw=ref.get_fdata(dtype=np.float32)
assert np.isfinite(raw).all() and np.equal(raw,np.rint(raw)).all() and np.isin(raw,[0,1,2,3,4]).all()
label=raw.astype(np.uint8);del raw
spacing=np.asarray(ref.header.get_zooms()[:3],float)
points=np.argwhere(label==4)
assert len(points)>0
lo,hi=points.min(0),points.max(0)
spec=importlib.util.spec_from_file_location('exact_grid_probe',root/'runtime/upstream_source/src/preprocess/nnunet_like.py')
grid=importlib.util.module_from_spec(spec);spec.loader.exec_module(grid)
shape,factor,offset=grid.physical_grid(label.shape,spacing,(1.,1.,1.))
assert int(np.prod(shape))<50000000
bounded()
exact=grid.resample_physical(label[None],shape,factor,offset,spacing,(1.,1.,1.),is_seg=True)[0]
nearest=ndimage.affine_transform(label,np.diag(factor),offset,output_shape=tuple(shape),order=0,mode='nearest')
aniso=None
for candidate in (spacing,np.ones(3)):
    if grid.get_do_separate_z(candidate):
        axes=grid.get_lowres_axis(candidate)
        if len(axes)==1:aniso=int(axes[0])
        break
bounded()
target_points=np.argwhere(nearest==4)
projection=(points+.5)*spacing-.5
target_lo=np.maximum(0,np.floor(projection.min(0)).astype(int)-2)
target_hi=np.minimum(shape,np.ceil(projection.max(0)).astype(int)+3)
roi_shape=target_hi-target_lo
assert int(np.prod(roi_shape))<1000000
if aniso is None:
    probability4=ndimage.affine_transform((label==4).astype(np.float32),np.diag(factor),offset+target_lo*factor,
        output_shape=tuple(roi_shape),order=1,mode='nearest',prefilter=False)
    probability0=ndimage.affine_transform((label==0).astype(np.float32),np.diag(factor),offset+target_lo*factor,
        output_shape=tuple(roi_shape),order=1,mode='nearest',prefilter=False)
    probability_report=dict(roi_origin=target_lo.tolist(),roi_shape=roi_shape.tolist(),max_class4_probability=float(probability4.max()),
        class4_threshold_hits=int(np.count_nonzero(probability4>=.5)))
    samples=[]
    for q in target_points[:64]:
        r=q-target_lo
        if np.all(r>=0) and np.all(r<roi_shape):
            prob4=float(probability4[tuple(r)]);prob0=float(probability0[tuple(r)])
            native_cont=q*factor+offset
            samples.append(dict(target_ijk=q.tolist(),sample_native_continuous_ijk=native_cont.tolist(),
                sample_native_nearest_ijk=np.floor(native_cont+.5).astype(int).tolist(),
                probability4=prob4,probability0=prob0,background_would_overwrite_nearest4=prob0>=.5,
                final_label=int(exact[tuple(q)])))
    probability_report['nearest4_samples']=samples
else:
    probability_report=dict(not_evaluated='anisotropic separate-axis probability path; no isotropic approximation claimed')
affine_target=ref.affine.copy();affine_target[:3,:3]=ref.affine[:3,:3]@np.diag(factor)
affine_target[:3,3]=ref.affine[:3,3]+ref.affine[:3,:3]@offset
v['single_case']=dict(native_shape=list(ref.shape),native_spacing_mm=spacing.tolist(),native_affine=ref.affine.tolist(),
    source_shape=list(source_ref.shape),source_affine=source_ref.affine.tolist(),source_recon_affine_max_abs=float(np.abs(source_ref.affine-ref.affine).max()),
    source_seg_sha256=source_hash,reconstruction_seg_sha256=recon_hash,native_manifest_sha256=sha(cm_path),
    native_label_counts={str(k):int(np.count_nonzero(label==k)) for k in range(5)},
    native_class4_volume_mm3=float(len(points)*np.prod(spacing)),native_class4_bbox_inclusive=[lo.tolist(),hi.tolist()],
    native_class4_ijk_first64=points[:64].tolist(),native_class4_world_mm_first64=nib.affines.apply_affine(ref.affine,points[:64]).tolist(),
    native_class4_projected_target_ijk_first64=projection[:64].tolist(),target_shape=shape.tolist(),target_affine=affine_target.tolist(),
    sample_factor=factor.tolist(),sample_offset=offset.tolist(),anisotropic_separate_axis=aniso,
    exact_label_counts={str(k):int(np.count_nonzero(exact==k)) for k in range(5)},
    pure_nearest_label_counts={str(k):int(np.count_nonzero(nearest==k)) for k in range(5)},
    exact_lost_classes=[str(k) for k in range(1,5) if np.any(label==k) and not np.any(exact==k)],
    nearest_class4_target_ijk_first64=target_points[:64].tolist(),probability=probability_report,
    pure_nearest_is_diagnostic_only_not_authorized_runtime=True)
del nearest,exact,label
bounded()
directories=sorted(p for p in (root/'dataset/cases').iterdir() if p.is_dir())
inventory=[];incomplete=[];errors=[];cache={}
fields={'grid_version':'physical_spacing_v2','support_contract':'signal_statistics_vs_full_native_FOV_geometry_v1',
        'geometric_domain':'full_native_FOV_not_claimed_as_tissue_support',
        'normalization':'whole_case_common_mask_zscore_before_resampling_zero_outside_support'}
for p in directories:
    bounded()
    marker=p/'preprocess_complete.json'
    if not marker.is_file():incomplete.append(p.name);continue
    try:
        info=read(marker);contract=info['contract'];member=active[p.name]
        assert info['case_id']==p.name and info['status']=='complete' and info['split']==member['split']
        assert info['patient_group']==member['patient_group']
        assert all(contract[k]==expected for k,expected in fields.items())
        assert contract['target_spacing']==[1.,1.,1.]
        assert contract['bbox']==[[0,int(n)] for n in contract['original_shape']]
        assert all(not contract['label_native'][str(k)]['voxels'] or contract['label_target_1mm'][str(k)]['voxels'] for k in range(1,5))
        spatial=read(p/'spatial_mapping.json');assert all(spatial[k]==value for k,value in contract.items())
        native_marker=parent/'cases'/p.name/'case_manifest.json';ident=read(native_marker)
        assert sha(native_marker)==info['native_case_manifest_sha256']==parent_complete['manifest_sha256'][p.name]
        assert all(h==ident['files'][mod]['sha256'] for mod,h in info['input_sha256'].items())
        assert all(h==ident['source_sha256'][mod] for mod,h in info['authentic_mask_source_sha256'].items())
        expected_names={p.name+'-'+mod+'.nii.gz' for mod in ('t1c','t1n','t2w','t2f','seg','brainmask')}
        assert set(info['outputs'])==expected_names
        assert all((p/n).is_file() and not (p/n).is_symlink() and (p/n).stat().st_size>0 for n in expected_names)
        inventory.append(dict(case_id=p.name,split=member['split'],marker_sha256=sha(marker),
            logical_output_bytes=sum((p/n).stat().st_size for n in expected_names),metadata_consistent=True,
            content_sha_all_files_not_checked=True))
        cache[p.name]=(p,info)
    except (AssertionError,KeyError,ValueError,OSError) as e:errors.append(dict(case_id=p.name,error=type(e).__name__+': '+str(e)))
samples=[]
ids=sorted(cache)
selection=list(dict.fromkeys([ids[0],ids[len(ids)//2],ids[-1]])) if ids else []
for name in selection:
    bounded()
    p,info=cache[name];observed={n:sha(p/n) for n in info['outputs']}
    samples.append(dict(case_id=name,observed_output_sha256=observed,mismatches=[n for n,h in observed.items() if h!=info['outputs'][n]]))
v['reuse_inventory']=dict(case_directories=len(directories),metadata_consistent_candidates=len(inventory),
    candidates=inventory,incomplete_case_directories=incomplete,metadata_errors=errors,
    output_sha_samples=samples,all_candidate_content_sha_verified=False,
    per_case_crop_rows_persisted=False,marker_written_before_crop_index_generation=True,
    reuse_authorized_or_performed=False,no_cohort_voxel_scan=True)
v['remote_end_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
v['seconds']=round(time.monotonic()-start,3)
v['available_bytes']=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize
print(json.dumps(v,sort_keys=True))
'''
    return code.replace('__ROOT__', repr(str(REMOTE)))


def main():
    code = remote_code()
    compile(code, 'readonly_single_case_diagnostic', 'exec')
    access = transport('diagnose_class_loss')
    local_start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        access.start_vpn()
        output = access.ssh('env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 '
                            + m.PYTHON + ' -B -u -c ' + shlex.quote(code), timeout=190)
        value = json.loads(output.strip().splitlines()[-1])
        value.update(m.META)
        value['local_start_utc'] = local_start
        value['local_received_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit / 'DIAGNOSTIC.json', value)
    finally:
        access.close_vpn()
    evidence = access.audit / 'DIAGNOSTIC.json'
    m.exclusive(access.audit / 'LOCAL_OPERATION_RECEIPT.json', dict(
        **m.META, file_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        operator_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),
        local_end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        vpn_started_by_operation=access.owns_vpn, operation_vpn_closed=True,
        remote_writes=False, remote_submissions=False, large_data_downloaded=False))
    print(json.dumps(dict(audit=str(access.audit),single_case=value['single_case'],
        reuse_summary={k:v for k,v in value['reuse_inventory'].items() if k!='candidates'},
        drift=value['drift'],seconds=value['seconds'],local_received_utc=value['local_received_utc']),ensure_ascii=False))


if __name__ == '__main__':
    main()
