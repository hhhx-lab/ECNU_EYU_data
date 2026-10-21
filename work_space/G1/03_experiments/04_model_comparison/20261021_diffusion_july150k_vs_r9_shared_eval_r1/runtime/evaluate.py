"""Paired old/new inference, no persistent generated volumes."""
import argparse
import json
import os
import signal
import time
import traceback
import numpy as np
import torch
from common import ROOT,META,MODS,SEEDS,cfg,completed,load_model,now,own_package,parent,sampler,sha,write
from metrics import calibration,components,cube,halo_slices,measure,stream_seed,tile_starts

STOP=False


def stop(*_):
    global STOP
    STOP=True


def generate(model,edm,schedule,label,real,starts,case,mod,seed,version):
    cond=np.stack([np.isin(label,[1,3]),np.isin(label,[1,2,3]),label==3,label==4]).astype(np.float32)
    numerator=np.zeros(label.shape,np.float32);rawsum=np.zeros_like(numerator);denominator=np.zeros_like(numerator)
    x=np.arange(64)-31.5;g=np.exp(-.5*(x/16.)**2);weight=(g[:,None,None]*g[None,:,None]*g[None,None,:]).astype(np.float32)
    calibration_rows=[]
    for start in starts:
        if STOP:raise RuntimeError('Controlled stop; no automatic retry')
        a,sl,content=cube(cond,start)
        derived=stream_seed(case,start,mod,seed)
        torch.manual_seed(derived);torch.cuda.manual_seed_all(derived)
        with torch.inference_mode():
            pred=edm.sample_edm(model,(1,64,64,64),torch.from_numpy(a[None]).to('cuda'),schedule,'cuda',num_steps=18,S_churn=0.,cfg_weight=1.)
        assert pred.shape==(1,1,64,64,64) and torch.isfinite(pred).all()
        raw=pred[0,0].cpu().numpy()[content];w=weight[content]
        if version=='july150k':
            mean,std=calibration(real[sl]);data=raw*std+mean
            calibration_rows.append(dict(start=list(start),mean=mean,sample_std=std))
        else:data=raw
        numerator[sl]+=data*w;rawsum[sl]+=raw*w;denominator[sl]+=w
    covered=denominator>0
    out=np.zeros_like(numerator);rawout=np.zeros_like(numerator)
    out[covered]=numerator[covered]/denominator[covered];rawout[covered]=rawsum[covered]/denominator[covered]
    assert np.isfinite(out).all()
    return out,rawout,covered,calibration_rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--modality',choices=MODS,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    output=ROOT/('smoke' if a.smoke else 'evaluation')/a.modality
    write(output/'STARTED.json',dict(META,utc=now(),job_id=os.getenv('SLURM_JOB_ID')))
    started=time.monotonic()
    try:
        c=cfg();pkg=own_package();ref=parent()
        done=json.loads((ROOT/'audit/COMPLETE.ok').read_text())
        assert done['package_sha256']==pkg and done['report_sha256']==sha(ROOT/'audit/REPORT.json')
        assert not (ROOT/'audit/FAILED.json').exists()
        assert torch.cuda.is_available() and torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0)
        assert int(os.getenv('WORLD_SIZE','1'))==1 and os.getenv('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
        audit=json.loads((ROOT/'audit/REPORT.json').read_text());records=ref['records']
        if a.smoke:records=[r for r in records if r['case'] in audit['smoke_cases']]
        else:
            for mod in MODS:
                d=ROOT/'smoke'/mod;v=json.loads((d/'COMPLETE.ok').read_text())
                assert v['package_sha256']==pkg and v['status']=='smoke_pass' and v['estimated_full_seconds']<48*3600*.85
                assert not (d/'FAILED.json').exists() and v['report_sha256']==sha(d/'REPORT.json')
        assert records
        completed(['audit']+([] if a.smoke else ['smoke_'+m for m in MODS]))
        edm,schedule=sampler();sampled=0;versions={};visuals=[]
        for version in ('july150k','r9'):
            info=c['models'][version][a.modality];model=load_model(info,version,'cuda');summaries=[]
            for rec in records:
                assert sha(rec['reference'])==rec['reference_sha256']
                with np.load(rec['reference']) as f:
                    label=f['label'];real=f['authentic'][MODS.index(a.modality)];recon=f['recon'][MODS.index(a.modality)]
                ids,cc=components(label)
                if not rec['positive']:
                    assert not label.any();summaries.append(dict(case=rec['case'],patient_group=rec['patient_group'],positive=False,status='NA_no_lesion_noop'));continue
                starts=tile_starts(cc,label.shape);seedrows=[];foreground_samples=[]
                for seed in SEEDS:
                    pred,raw,covered,cal=generate(model,edm,schedule,label,real,starts,rec['case'],a.modality,seed,version)
                    sampled+=len(starts)
                    if a.smoke and seed==SEEDS[0]:
                        repeat,_,other,_=generate(model,edm,schedule,label,real,starts,rec['case'],a.modality,seed,version)
                        assert np.array_equal(pred,repeat) and np.array_equal(covered,other),'Reproducibility failed'
                    primary=measure(real,pred,label,ids,cc,covered);secondary=measure(recon,pred,label,ids,cc,covered)
                    foreground_samples.append(pred[label>0].copy())
                    raw_metric=measure(real,raw,label,ids,cc,covered) if version=='july150k' else None
                    value=dict(META,utc=now(),case=rec['case'],patient_group=rec['patient_group'],version=version,
                        modality=a.modality,seed=seed,checkpoint_sha256=info['sha256'],reference_sha256=rec['reference_sha256'],
                        primary=primary,secondary=secondary,old_raw_sensitivity=raw_metric,calibration=cal,tiles=len(starts))
                    dest=output/version/(rec['case']+'_'+str(seed)+'.json');write(dest,value)
                    seedrows.append(value)
                    if seed==SEEDS[0] and rec['case']==audit['first_positive']:
                        from PIL import Image
                        comp=max(cc,key=lambda v:v['n_voxels']);sl=halo_slices(comp,label.shape);z=(sl[2].start+sl[2].stop)//2
                        panels=[]
                        for v in (real,recon,pred):
                            gray=np.round(np.clip((v[sl[0],sl[1],z]+3)/6,0,1)*255).astype(np.uint8).T
                            panels.append(Image.fromarray(gray).resize((256,256)))
                        canvas=Image.new('L',(768,256))
                        for i,v in enumerate(panels):canvas.paste(v,(256*i,0))
                        vp=output/version/'visual.png'
                        with vp.open('xb') as f:canvas.save(f,format='PNG')
                        visuals.append(dict(version=version,path=str(vp),sha256=sha(vp),panel_order=['real','reconstruction','generated'],manual_review='not_performed'))
                row=dict(case=rec['case'],patient_group=rec['patient_group'],positive=True,seed_count=4,
                         primary={},secondary={},old_raw_sensitivity={},size_ssim={},regions={},seed_evidence={})
                voxel_std=np.std(np.stack(foreground_samples).astype(np.float64),axis=0,ddof=0)
                row['seed_variability']=dict(foreground_voxel_std_mean=float(voxel_std.mean()),
                    foreground_voxel_std_rms=float(np.sqrt(np.mean(voxel_std**2))),
                    interpretation='Four fixed seeds, descriptive only; larger is not better')
                for target in ('primary','secondary','old_raw_sensitivity'):
                    if target=='old_raw_sensitivity' and version!='july150k':continue
                    for metric in ('ssim','mae','rmse','psnr','extreme_fraction','zero_fraction','contrast_abs_error'):
                        vals=[r[target][metric] for r in seedrows]
                        row[target][metric]=float(np.mean(vals)) if all(v is not None for v in vals) else None
                for name in ('small_le27','large_gt27'):
                    vals=[r['primary']['size_ssim'][name] for r in seedrows]
                    row['size_ssim'][name]=float(np.mean(vals)) if vals[0] is not None else None
                for name in ('ET','NETC','SNFH','RC'):
                    row['regions'][name]={}
                    for metric in ('ssim','mae','rmse','psnr'):
                        vals=[r['primary']['regions'][name][metric] for r in seedrows]
                        row['regions'][name][metric]=float(np.mean(vals)) if all(v is not None for v in vals) else None
                for seed in SEEDS:
                    pp=output/version/(rec['case']+'_'+str(seed)+'.json');row['seed_evidence'][str(seed)]=sha(pp)
                summaries.append(row)
                with (output/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(utc=now(),version=version,case=rec['case'],elapsed_seconds=time.monotonic()-started))+'\n')
                print(version,a.modality,rec['case'],'4seed pass',flush=True)
            versions[version]=summaries;del model;torch.cuda.empty_cache()
        assert [r['case'] for r in versions['july150k']]==[r['case'] for r in versions['r9']]
        if not a.smoke:assert len(records)==103 and sum(r['positive'] for r in records)==94
        elapsed=time.monotonic()-started
        result=dict(META,status='smoke_pass' if a.smoke else 'evaluation_complete',utc=now(),modality=a.modality,
            package_sha256=pkg,reference_report_sha256=c['reference_report_sha256'],models=c['models'],versions=versions,
            seeds=list(SEEDS),visuals=visuals,elapsed_seconds=elapsed,sampled_cubes=sampled,
            estimated_full_seconds=elapsed/max(1,sampled)*audit['total_tiles']*4*2*1.5,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),mean_primary={v:float(np.mean([r['primary']['ssim'] for r in rows if r['positive']])) for v,rows in versions.items()})
        if a.smoke:assert result['estimated_full_seconds']<48*3600*.85,'Measured runtime exceeds allocation'
        write(output/'REPORT.json',result)
        write(output/'COMPLETE.ok',dict(META,utc=now(),status=result['status'],package_sha256=pkg,
            report_sha256=sha(output/'REPORT.json'),estimated_full_seconds=result['estimated_full_seconds']))
    except BaseException as e:
        write(output/'FAILED.json',dict(META,utc=now(),message=str(e),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
