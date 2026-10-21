"""Independent fixed-val candidate inference; no saved generated volumes."""
import argparse
import json
import os
import signal
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from common import META, ROOT, checkpoint, frozen_model_module, model_from, now, settings, sha, write
from metrics import SEEDS, STEPS, bbox_slices, crop64, gaussian64, masked_errors, ssim_xyz, stream_seed, weighted_lesions

STOP = False


def stop(*_):
    global STOP
    STOP = True


def components(label):
    ids, n = ndimage.label(label > 0, structure=np.ones((3,3,3)))
    result=[]
    for i, sl in enumerate(ndimage.find_objects(ids),1):
        if sl is None: raise ValueError('Missing connected component')
        bbox=tuple(v for s in sl for v in (s.start,s.stop))
        result.append(dict(id=i,bbox=bbox,n_voxels=int(np.count_nonzero(ids[sl]==i))))
    return ids,result


def multichannel(label):
    return np.stack([np.isin(label,[1,3]),np.isin(label,[1,2,3]),label==3,label==4]).astype(np.float32)


def generate(model, edm, schedule, label, rows, case, modality, seed, device):
    cond=multichannel(label);numerator=np.zeros(label.shape,np.float32);denominator=np.zeros(label.shape,np.float32)
    weight=gaussian64()
    for row in rows:
        bbox=tuple(int(row[k+'_extreme_'+side]) for k in 'xyz' for side in ('min','max'))
        cube,sl,content=crop64(cond,bbox)
        torch.manual_seed(stream_seed(case,row['lesion_id'],modality,seed))
        if device=='cuda':torch.cuda.manual_seed_all(stream_seed(case,row['lesion_id'],modality,seed))
        with torch.inference_mode():
            pred=edm.sample_edm(model,(1,64,64,64),torch.from_numpy(cube[None]).to(device),schedule,
                                device,num_steps=18,S_churn=0.,cfg_weight=1.)
        assert pred.shape==(1,1,64,64,64) and torch.isfinite(pred).all(), 'Nonfinite generation'
        data=pred[0,0].cpu().numpy()[content];w=weight[content]
        numerator[sl]+=data*w;denominator[sl]+=w
    mask=label>0
    assert np.all(denominator[mask]>0), 'Uncovered lesion voxels'
    generated=np.zeros(label.shape,np.float32);generated[mask]=numerator[mask]/denominator[mask]
    assert np.isfinite(generated).all()
    return generated


def measure(reference, generated, label, ids, cc, target, background=None):
    background=reference if background is None else background
    transaction=background.copy();transaction[label>0]=generated[label>0]
    assert np.array_equal(transaction[label==0],background[label==0]), 'Support transaction drift'
    lesions=[]
    for comp in cc:
        sl=bbox_slices(comp['bbox'],label.shape)
        localmask=ids[sl]==comp['id']
        row=dict(component=comp['id'],bbox=comp['bbox'],n_voxels=comp['n_voxels'],
                 ssim=ssim_xyz(reference[sl],transaction[sl]),**masked_errors(reference[sl],transaction[sl],localmask))
        lesions.append(row)
    return dict(target=target,lesion_ssim=weighted_lesions(lesions),lesions=lesions,
                **masked_errors(reference,transaction,label>0),
                regions={name:masked_errors(reference,transaction,np.isin(label,classes))
                         for name,classes in {'ET':[3],'NETC':[1],'SNFH':[2],'RC':[4]}.items()},
                size_ssim={name:weighted_lesions(selected) if selected else None for name,selected in
                           [('small_le27',[l for l in lesions if l['n_voxels']<=27]),
                            ('large_gt27',[l for l in lesions if l['n_voxels']>27])]},support_outside_unchanged=True)


def visual(path, recon, authentic, pred, label, cc):
    from PIL import Image
    comp=max(cc,key=lambda c:c['n_voxels']);sl=bbox_slices(comp['bbox'],label.shape)
    z=(sl[2].start+sl[2].stop-1)//2
    transaction=recon.copy();transaction[label>0]=pred[label>0]
    panels=[]
    for a in (recon,transaction,authentic):
        gray=np.round(np.clip((a[sl[0],sl[1],z]+3)/6,0,1)*255).astype(np.uint8).T
        rgb=np.repeat(gray[:,:,None],3,axis=2);mask=label[sl[0],sl[1],z].T>0
        border=mask & ~ndimage.binary_erosion(mask)
        rgb[border]=[255,70,70]
        panels.append(Image.fromarray(rgb).resize((256,256),Image.Resampling.NEAREST))
    out=Image.new('RGB',(768,256))
    for i,p in enumerate(panels):out.paste(p,(i*256,0))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:out.save(f,format='PNG')
    return dict(path=str(path),sha256=sha(path),case_slice_z=z,component=comp['id'],
                panel_order=['reconstruction','generated_same_transaction','authentic'],fixed_window=[-3,3],
                manual_review='not_performed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--modality',choices=('t1c','t1n','t2w','t2f'),required=True)
    parser.add_argument('--smoke',action='store_true');a=parser.parse_args()
    output=ROOT/('smoke' if a.smoke else 'evaluation')/a.modality
    write(output/'STARTED.json',dict(META,utc=now(),job_id=os.getenv('SLURM_JOB_ID')))
    try:
        cfg,r9=settings();assert torch.cuda.is_available() and torch.cuda.device_count()==1
        if not a.smoke:
            for mod in ('t1c','t1n','t2w','t2f'):
                smoke=json.loads((ROOT/'smoke'/mod/'COMPLETE.ok').read_text())
                assert smoke['status']=='smoke_pass' and smoke['modality']==mod
                assert not (ROOT/'smoke'/mod/'FAILED.json').exists()
                assert smoke['evaluation_package_sha256']==sha(ROOT/'PACKAGE_MANIFEST.json')
                assert smoke['reference_report_sha256']==sha(ROOT/'reference/REPORT.json')
                assert smoke['estimated_full_seconds'] < 48 * 3600 * .85, 'Measured runtime exceeds allocation margin'
        assert 'A100' in torch.cuda.get_device_name(0) and int(os.getenv('WORLD_SIZE','1'))==1
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
        done=json.loads((ROOT/'reference/COMPLETE.ok').read_text());assert done['status']=='pass'
        assert sha(ROOT/'reference/REPORT.json')==done['report_sha256']
        ref=json.loads((ROOT/'reference/REPORT.json').read_text())
        assert not (ROOT/'reference/FAILED.json').exists() and ref['errors']==[] and ref['cases']==103
        assert ref['evaluation_package_sha256']==sha(ROOT/'PACKAGE_MANIFEST.json')
        records=ref['records'];contract=ref['contracts'][a.modality];edm=frozen_model_module(r9)
        indices={m:i for i,m in enumerate(('t1c','t1n','t2w','t2f'))};start=time.monotonic()
        if a.smoke:
            positives=[r for r in records if r['positive']]
            special=[]
            for rec in positives:
                with np.load(rec['reference']) as f:label=f['label']
                _,cc=components(label)
                flags=set()
                if np.any(label==4):flags.add('RC')
                if any(c['n_voxels']<=27 for c in cc):flags.add('small')
                if any(any(c['bbox'][i+1]-c['bbox'][i]>64 for i in (0,2,4)) for c in cc):flags.add('tile')
                covered=set().union(*(v[1] for v in special)) if special else set()
                if flags-covered:special.append((rec,flags))
            records=list({r['case']:r for r in [positives[0],*(v[0] for v in special),
                          next(r for r in records if not r['positive'])]}.values())
            assert {'RC','small','tile'}.issubset(set().union(*(v[1] for v in special))), 'Missing representative smoke strata'
        samples=0;candidate_shas={}
        steps=(200000,) if a.smoke else STEPS
        for step in steps:
            info=ref['checkpoints'][a.modality][str(step)];assert sha(info['path'])==info['sha256']
            saved=checkpoint(info['path'],contract,step,a.modality);model=model_from(saved,'cuda')
            s=saved['schedule_config'];schedule=edm.make_noise_schedule(schedule='edm',n_steps=s['n_steps'],
                sigma_data=s['sigma_data'],sigma_min=s['sigma_min'],sigma_max=s['sigma_max'],rho=s['rho'],snr_shift=s['snr_shift'])
            del saved
            summaries=[];seed_evidence={};visual_evidence=[]
            for rec in records:
                assert sha(rec['reference'])==rec['reference_sha256']
                with np.load(rec['reference']) as f:
                    label=f['label'];recon=f['recon'][indices[a.modality]];authentic=f['authentic'][indices[a.modality]]
                if not rec['positive']:
                    assert not label.any()
                    summaries.append(dict(case=rec['case'],patient_group=rec['patient_group'],positive=False,status='NA_no_lesion_noop'))
                    continue
                ids,cc=components(label);seedrows=[]
                for seed in SEEDS:
                    if STOP:raise RuntimeError('Controlled evaluation stop; no automatic retry')
                    pred=generate(model,edm,schedule,label,rec['rows'],rec['case'],a.modality,seed,'cuda')
                    samples+=len(rec['rows'])
                    if a.smoke and seed==SEEDS[0]:
                        repeat=generate(model,edm,schedule,label,rec['rows'],rec['case'],a.modality,seed,'cuda')
                        assert np.array_equal(pred,repeat), 'Sampling reproducibility mismatch'
                    value=dict(META,utc=now(),case=rec['case'],patient_group=rec['patient_group'],step=step,
                        modality=a.modality,seed=seed,checkpoint_sha256=info['sha256'],reference_sha256=rec['reference_sha256'],
                        primary=measure(recon,pred,label,ids,cc,'vae_reconstructed'),
                        secondary=measure(authentic,pred,label,ids,cc,'authentic_real',background=recon))
                    evidence=output/str(step)/(rec['case']+'_'+str(seed)+'.json')
                    write(evidence,value);seed_evidence[evidence.name]=sha(evidence)
                    if seed==SEEDS[0] and rec['case']==records[0]['case']:
                        visual_evidence.append(visual(output/str(step)/'visual'/ (rec['case']+'.png'),recon,authentic,pred,label,cc))
                    seedrows.append(value)
                summaries.append(dict(case=rec['case'],patient_group=rec['patient_group'],positive=True,
                    primary_ssim=float(np.mean([v['primary']['lesion_ssim'] for v in seedrows])),
                    secondary_ssim=float(np.mean([v['secondary']['lesion_ssim'] for v in seedrows])),
                    primary_mse=float(np.mean([v['primary']['mse'] for v in seedrows])),
                    secondary_mse=float(np.mean([v['secondary']['mse'] for v in seedrows]))))
                print(a.modality,step,rec['case'],'4seed pass',flush=True)
                with (output/'progress.jsonl').open('a') as f:
                    f.write(json.dumps(dict(utc=now(),step=step,case=rec['case'],seconds=time.monotonic()-start))+'\n')
            result=dict(META,utc=now(),step=step,modality=a.modality,checkpoint_sha256=info['sha256'],cases=summaries,
                reference_report_sha256=done['report_sha256'],evaluation_package_sha256=sha(ROOT/'PACKAGE_MANIFEST.json'),
                seed_evidence_sha256=seed_evidence,visual_evidence=visual_evidence,seeds=list(SEEDS),
                all_qc_pass=True,selection_status='not_run')
            if not a.smoke:
                assert len(summaries)==103 and sum(r['positive'] for r in summaries)==94
            summary_path=output/str(step)/'CANDIDATE_COMPLETE.json';write(summary_path,result)
            candidate_shas[str(step)]=sha(summary_path)
            del model;torch.cuda.empty_cache()
        write(output/'COMPLETE.ok',dict(META,status='smoke_pass' if a.smoke else 'candidate_evaluation_complete',
              utc=now(),modality=a.modality,steps=list(steps),elapsed_seconds=time.monotonic()-start,
              sampled_cubes=samples,estimated_full_seconds=(time.monotonic()-start)/max(1,samples)*602*4*10*1.5,
              peak_cuda_bytes=torch.cuda.max_memory_allocated(),cases=[r['case'] for r in records],
              candidate_summary_sha256=candidate_shas,reference_report_sha256=done['report_sha256'],
              evaluation_package_sha256=sha(ROOT/'PACKAGE_MANIFEST.json')))
    except Exception as error:
        write(output/'FAILED.json',dict(META,utc=now(),message=str(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
