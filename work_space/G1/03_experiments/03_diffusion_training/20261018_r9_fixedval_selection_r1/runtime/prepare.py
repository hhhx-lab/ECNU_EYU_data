"""CPU-only validation reference preparation and forty checkpoint audits."""
import collections
import csv
import json
import os
import traceback
from pathlib import Path
import nibabel as nib
import numpy as np
from common import META, MODS, ROOT, R9_SHA, checkpoint, model_from, now, settings, sha, write
from metrics import STEPS, crop64


def volume(path, reference=None, labels=False):
    image = nib.load(str(path)); a = image.get_fdata(dtype=np.float32)
    assert np.isfinite(a).all() and np.isfinite(image.affine).all(), str(path)
    if reference is not None:
        assert image.shape == reference.shape and np.allclose(image.affine, reference.affine, atol=1e-5), str(path)
    if labels:
        assert np.isin(a, [0, 1, 2, 3, 4]).all()
    return image, a


def main():
    write(ROOT / 'reference/STARTED.json', dict(META, utc=now(), job_id=os.getenv('SLURM_JOB_ID')))
    try:
        cfg, r9 = settings()
        from scipy.ndimage import binary_fill_holes
        from src.preprocess.nnunet_like import physical_grid, resample_physical
        from scripts.preprocess_dataset import new_affine
        done = json.loads((r9 / 'dataset/COMPLETE.ok').read_text())
        assert done['cases'] == 926 and done['split_counts'] == {'train': 823, 'val': 103}
        assert not (r9 / 'dataset/FAILED.json').exists()
        assert sha(r9 / 'dataset/lesions.csv') == done['csv_sha256']
        assert sha(r9 / 'dataset/PREPROCESS_REPORT.json') == done['report_sha256']
        report=json.loads((r9/'dataset/PREPROCESS_REPORT.json').read_text())
        assert report['status']=='complete' and report['cases_ok']==report['cases_processed']==926 and report['errors']==[]
        assert report['warnings']==done['resolution_warnings']
        assert done['package_sha256']==R9_SHA and done['status']=='complete'
        run = json.loads((r9 / 'config/RUN.json').read_text())
        with (r9 / 'plan/splits_final_train_val_test_membership.csv').open() as f:
            members = list(csv.DictReader(f))
        assert collections.Counter(r['split'] for r in members) == {'train':823,'val':103,'test':104}
        groups = collections.defaultdict(set)
        for m in members: groups[m['patient_group']].add(m['split'])
        assert all(len(v) == 1 for v in groups.values())
        val = sorted([m for m in members if m['split'] == 'val'], key=lambda m:m['source_case_id'])
        with (r9 / 'dataset/lesions.csv').open() as f: all_rows = list(csv.DictReader(f))
        rows = collections.defaultdict(list)
        for row in all_rows:
            if row['split'] == 'val': rows[row['source_case_id']].append(row)
        assert sum(map(len, rows.values())) == 602
        assert set(rows).issubset({m['source_case_id'] for m in val})
        assert sha(Path(run['source_code']) / 'data/data_csv.csv') == run['source_csv_sha256']
        with (Path(run['source_code']) / 'data/data_csv.csv').open() as f:
            source_rows=list(csv.DictReader(f));source = {r['id']:r for r in source_rows}
        assert len(source)==len(source_rows) and all(source[m['source_case_id']]['split']=='val' for m in val)
        available = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
        shapes=[json.loads((Path(done['case_locations'][m['source_case_id']])/'preprocess_complete.json').read_text())['contract']['shape'] for m in val]
        data_upper=sum(int(np.prod(s))*33 for s in shapes)
        required=data_upper*1.01+max(int(np.prod(s))*200 for s in shapes)+16*1024**3
        assert available>required, 'Insufficient measured reference/temporary/results reserve'
        contracts = {}; inventory = {}
        preflight=json.loads((ROOT/'evidence/smoke/PREFLIGHT.json').read_text())
        for mod in MODS:
            d = r9 / 'training' / mod
            c = json.loads((d / 'RUN_CONTRACT.json').read_text())
            assert c['package_sha256'] == R9_SHA and c['identity']['csv_sha256'] == done['csv_sha256']
            complete = json.loads((d / 'COMPLETE.ok').read_text())
            assert complete['global_step'] == 200000 and complete['status'] == 'training_complete'
            assert not (d / 'FAILED.json').exists() and not (d / 'SAFE_TERM.json').exists()
            contracts[mod] = c; inventory[mod] = {}
            for step in STEPS:
                p = d / 'weights' / f'diffusion_{step}.pt'
                assert p.is_file() and not p.is_symlink() and p.resolve() == p
                digest=sha(p)
                assert digest==preflight['candidate_inventory'][mod+'_'+str(step)]['sha256']
                v = checkpoint(p, c, step, mod); model_from(v, 'cpu')
                if step == 200000: assert digest == complete['checkpoint_sha256']
                inventory[mod][str(step)] = dict(path=str(p), sha256=digest, bytes=p.stat().st_size)
                del v
        common_identity = {k:v for k,v in contracts['t1c']['identity'].items() if k not in ('modality','consumed')}
        assert all({k:v for k,v in c['identity'].items() if k not in ('modality','consumed')} == common_identity for c in contracts.values())
        for key,expected in dict(dataset_complete_sha256=sha(r9/'dataset/COMPLETE.ok'),report_sha256=done['report_sha256'],
                membership_sha256=sha(r9/'plan/splits_final_train_val_test_membership.csv'),source_csv_sha256=run['source_csv_sha256'],
                package_sha256=R9_SHA,reference_parents=done['reference_parents'],case_manifests=done['case_manifests'],
                case_locations=done['case_locations'],loss_policy='uniform_original_quantization_warning_v1',
                label_resampling_algorithm='original_physical_spacing_v2_labelwise_threshold_no_case_exception').items():
            assert common_identity[key]==expected, 'Training identity drift: '+key
        details={d['case']:d for d in done['cases_detail']};assert len(details)==926
        for parent,digest in common_identity['reference_parents'].items():
            assert sha(Path(parent)/'plan/PACKAGE_MANIFEST.json')==digest
        records = []; errors = []
        for member in val:
            case = member['source_case_id']
            try:
                directory = Path(done['case_locations'][case])
                marker_path = directory / 'preprocess_complete.json'
                assert sha(marker_path) == done['case_manifests'][case]
                marker = json.loads(marker_path.read_text()); contract = marker['contract']
                detail=details[case]
                assert directory==r9/'dataset/cases'/case and directory.resolve()==directory
                assert marker['case_id'] == case and marker['split'] == 'val' and marker['status']=='complete'
                assert marker['patient_group']==member['patient_group'] and contract['local_label_exception'] is None
                assert contract['grid_version']=='physical_spacing_v2' and contract['target_spacing']==[1.,1.,1.]
                assert contract['normalization']=='whole_case_common_mask_zscore_before_resampling_zero_outside_support'
                assert contract['mask_algorithm']=='authentic_T1N_T1C_T2F_robust_0_100_nonzero_union_holefill'
                assert contract['loss_policy']==common_identity['loss_policy'] and contract['label_resampling_algorithm']==common_identity['label_resampling_algorithm']
                assert marker['outputs']==detail['outputs']
                spatial=json.loads((directory/'spatial_mapping.json').read_text())
                assert all(spatial.get(k)==v for k,v in marker['contract'].items())
                index_path=r9/'dataset/indices'/(case+'.json')
                assert sha(index_path)==common_identity['index_sha256'][case]
                index=json.loads(index_path.read_text());assert index['marker_sha256']==sha(marker_path)
                assert index['case_id']==case
                def rowkey(row):return tuple(sorted((k,str(v)) for k,v in row.items() if k!='patient_n_crops'))
                assert collections.Counter(map(rowkey,index['rows']))==collections.Counter(map(rowkey,rows[case]))
                for name,reference in marker.get('file_references',{}).items():
                    assert (directory/name).is_symlink() and str((directory/name).resolve())==reference['path']
                    assert reference['sha256']==marker['outputs'][name]
                assert marker.get('file_references',{})==common_identity['file_references'][case]==detail['file_references']
                if marker.get('file_references'):
                    parent=Path(marker['parent_directory']);assert str(parent.parents[2]) in common_identity['reference_parents']
                    assert sha(parent/'preprocess_complete.json')==marker['parent_marker_sha256']
                    parent_marker=json.loads((parent/'preprocess_complete.json').read_text())
                    assert parent_marker['case_id']==case and parent_marker['status']=='complete'
                    assert parent_marker['native_case_manifest_sha256']==marker['native_case_manifest_sha256']
                    assert all(parent_marker['outputs'][n]==v['sha256'] for n,v in marker['file_references'].items())
                assert contract['bbox'] == [[0, v] for v in contract['original_shape']]
                for name,digest in marker['outputs'].items(): assert sha(directory / name) == digest
                ref, label = volume(directory / (case+'-seg.nii.gz'), labels=True)
                _, mask = volume(directory / (case+'-brainmask.nii.gz'), ref)
                assert np.isin(mask, [0,1]).all() and np.allclose(np.linalg.norm(ref.affine[:3,:3],axis=0),1,atol=1e-5)
                label=label.astype(np.uint8); mask=mask>0
                recon = np.stack([volume(directory/(case+'-'+mod+'.nii.gz'),ref)[1] for mod in MODS])
                native_dir = Path(run['recon'])/'dataset/cases'/case
                nm_path=native_dir/'case_manifest.json';nm=json.loads(nm_path.read_text())
                assert sha(nm_path)==common_identity['native_case_manifests'][case] and nm['split']=='val'
                assert marker['native_case_manifest_sha256']==sha(nm_path) and nm['case_id']==case and nm['geometry']=='native'
                assert all(nm['files'][m]['sha256']==h for m,h in marker['input_sha256'].items())
                assert all(nm['source_sha256'][m]==h for m,h in marker['authentic_mask_source_sha256'].items())
                assert sha(native_dir/(case+'-seg.nii.gz'))==nm['files']['seg']['sha256']
                native_ref, native_label=volume(native_dir/(case+'-seg.nii.gz'),labels=True)
                real=[]; source_sha={}
                for mod in MODS:
                    p=Path(run['source_code'])/'data/input'/case/Path(source[case][mod]).name
                    source_sha[mod]=sha(p);assert source_sha[mod]==nm['source_sha256'][mod]
                    real.append(volume(p,native_ref)[1])
                native_mask=np.zeros(native_label.shape,dtype=bool)
                for mod in ('t1n','t1c','t2f'):
                    a=real[MODS.index(mod)];lo=max(0.,float(a.min()));hi=float(a.max())
                    if hi>lo:native_mask|=np.clip(a,lo,hi)>lo
                native_mask=binary_fill_holes(native_mask);assert native_mask.any()
                spacing=np.asarray(native_ref.header.get_zooms()[:3],float)
                shape,factor,offset=physical_grid(native_label.shape,spacing,(1.,1.,1.))
                affine,_=new_affine(native_ref.affine,contract['bbox'],native_label.shape,shape,spacing,(1.,1.,1.))
                assert tuple(shape)==label.shape and np.allclose(affine,ref.affine,atol=1e-5)
                checkmask=resample_physical(native_mask[None].astype(np.uint8),shape,factor,offset,spacing,(1.,1.,1.),is_seg=True)[0]>0
                checklabel=resample_physical(native_label[None],shape,factor,offset,spacing,(1.,1.,1.),is_seg=True)[0]
                assert np.array_equal(mask,checkmask) and np.array_equal(label,checklabel)
                authentic=[];statistics={}
                for mod,a in zip(MODS,real):
                    values=a[native_mask];mean=float(values.mean());std=max(float(values.std()),1e-8)
                    normalized=np.zeros(a.shape,np.float32);normalized[native_mask]=(a[native_mask]-mean)/std
                    out=resample_physical(normalized[None],shape,factor,offset,spacing,(1.,1.,1.))[0];out[~mask]=0
                    assert np.isfinite(out).all();authentic.append(out)
                    statistics[mod]=dict(mean=mean,effective_std=std,mask_voxels=int(values.size))
                for row in rows[case]:
                    b=tuple(int(row[k+'_extreme_'+side]) for k in 'xyz' for side in ('min','max'))
                    cube,_,_=crop64(label,b);assert np.count_nonzero(cube)>0
                assert bool(label.any())==bool(rows[case]), 'Positive denominator mismatch'
                dest=ROOT/'reference/cases'/(case+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
                with dest.open('xb') as f:np.savez_compressed(f,label=label,recon=recon,authentic=np.stack(authentic),affine=ref.affine)
                records.append(dict(case=case,patient_group=member['patient_group'],positive=bool(label.any()),
                  reference=str(dest),reference_sha256=sha(dest),rows=rows[case],source_sha256=source_sha,
                  native_marker_sha256=sha(nm_path),marker_sha256=sha(marker_path),normalization_statistics=statistics,
                  label_volume=int(np.count_nonzero(label)),resolution_warnings=contract['lost_classes']))
                print(case, 'reference pass', flush=True)
            except Exception as error:
                errors.append(dict(case=case,message=str(error),traceback=traceback.format_exc()))
        result=dict(META,utc=now(),cases=len(records),positive_cases=sum(r['positive'] for r in records),
                    negative_cases=sum(not r['positive'] for r in records),errors=errors,records=records,
                    checkpoints=inventory,contracts=contracts,training_package_sha256=R9_SHA,
                    dataset_complete_sha256=sha(r9/'dataset/COMPLETE.ok'),source_csv_sha256=run['source_csv_sha256'],
                    evaluation_package_sha256=sha(ROOT/'PACKAGE_MANIFEST.json'),
                    storage=dict(available_before_bytes=available,required_upper_bytes=int(required),data_upper_bytes=data_upper))
        write(ROOT/'reference/REPORT.json',result)
        assert not errors and len(records)==103 and result['positive_cases']==94 and result['negative_cases']==9
        write(ROOT/'reference/COMPLETE.ok',dict(META,status='pass',utc=now(),report_sha256=sha(ROOT/'reference/REPORT.json')))
    except Exception as error:
        write(ROOT/'reference/FAILED.json',dict(META,utc=now(),message=str(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__': main()
