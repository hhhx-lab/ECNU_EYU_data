"""CPU audit: every shared reference and all eight trusted weights."""
import json
import os
import traceback
from pathlib import Path
import numpy as np
import torch
from common import ROOT,META,MODS,cfg,load_model,now,own_package,parent,sha,write
from metrics import components,cube,halo_slices,tile_starts


def main():
    write(ROOT/'audit/STARTED.json',dict(META,utc=now(),job_id=os.getenv('SLURM_JOB_ID')))
    try:
        c=cfg();pkg=own_package();ref=parent();errors=[];records=[];smoke=[];covered=set();total=0
        torch.set_num_threads(1)
        models={}
        for version in ('july150k','r9'):
            models[version]={}
            for mod in MODS:
                info=c['models'][version][mod];model=load_model(info,version)
                with torch.inference_mode():
                    out=model(torch.zeros((1,1,64,64,64)),torch.zeros(1),torch.zeros((1,4,64,64,64)))
                assert out.shape==(1,1,64,64,64) and torch.isfinite(out).all()
                models[version][mod]=dict(sha256=sha(info['path']),step=info['step'],strict_cpu_loaded=True)
                del model
        first=None;negative=None
        for rec in ref['records']:
            try:
                p=Path(rec['reference']);assert sha(p)==rec['reference_sha256']
                with np.load(p) as f:
                    label=f['label'];real=f['authentic'];recon=f['recon'];affine=f['affine']
                assert label.ndim==3 and real.shape==recon.shape==(4,)+label.shape
                assert np.isfinite(real).all() and np.isfinite(recon).all() and np.isfinite(affine).all()
                assert affine.shape==(4,4) and np.allclose(np.linalg.norm(affine[:3,:3],axis=0),1,atol=1e-5)
                ids,cc=components(label);assert bool(cc)==rec['positive']
                assert int(np.count_nonzero(label))==rec['label_volume']
                starts=tile_starts(cc,label.shape);mask=np.zeros(label.shape,bool)
                for start in starts:
                    _,sl,_=cube(label,start);mask[sl]=True
                for comp in cc:assert mask[halo_slices(comp,label.shape)].all()
                total+=len(starts);flags=set()
                if cc:
                    first=first or rec['case']
                    if np.any(label==4):flags.add('RC')
                    if any(v['n_voxels']<=27 for v in cc):flags.add('small')
                    if any(any(v['bbox'][2*i+1]-v['bbox'][2*i]>64 for i in range(3)) for v in cc):flags.add('tile')
                    if flags-covered:smoke.append(rec['case']);covered|=flags
                else:negative=negative or rec['case']
                records.append(dict(case=rec['case'],patient_group=rec['patient_group'],positive=rec['positive'],
                    reference_sha256=rec['reference_sha256'],components=len(cc),tiles=len(starts),flags=sorted(flags)))
            except Exception as e:errors.append(dict(case=rec['case'],error=str(e),traceback=traceback.format_exc()))
        assert not errors and len(records)==103 and sum(r['positive'] for r in records)==94
        assert {'RC','small','tile'}<=covered and first and negative
        available=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize;assert available>16*1024**3
        result=dict(META,status='pass',utc=now(),package_sha256=pkg,reference_report_sha256=c['reference_report_sha256'],
            models=models,records=records,errors=errors,total_tiles=total,smoke_cases=sorted(set(smoke+[first,negative])),
            first_positive=first,storage_available_bytes=available,versions=dict(torch=torch.__version__))
        write(ROOT/'audit/REPORT.json',result)
        write(ROOT/'audit/COMPLETE.ok',dict(META,status='pass',utc=now(),package_sha256=pkg,report_sha256=sha(ROOT/'audit/REPORT.json')))
    except BaseException as e:
        write(ROOT/'audit/FAILED.json',dict(META,utc=now(),message=str(e),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
