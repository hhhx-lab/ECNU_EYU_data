"""Paired case-balanced comparison; never reselect checkpoints."""
import csv
import json
import traceback
import numpy as np
from common import ROOT,META,MODS,cfg,completed,now,own_package,parent,sha,write
from metrics import holm,paired


def main():
    write(ROOT/'comparison/STARTED.json',dict(META,utc=now()))
    try:
        c=cfg();pkg=own_package();reference=parent();results={};digests={};table=[]
        completed(['audit',*['smoke_'+m for m in MODS],*['eval_'+m for m in MODS]])
        for mod in MODS:
            d=ROOT/'evaluation'/mod;done=json.loads((d/'COMPLETE.ok').read_text())
            assert not (d/'FAILED.json').exists() and done['status']=='evaluation_complete' and done['package_sha256']==pkg
            assert done['report_sha256']==sha(d/'REPORT.json')
            report=json.loads((d/'REPORT.json').read_text());digests[mod]=sha(d/'REPORT.json')
            assert report['reference_report_sha256']==c['reference_report_sha256'] and report['seeds']==[20261013,20261014,20261015,20261016]
            old=report['versions']['july150k'];new=report['versions']['r9']
            expected=[r['case'] for r in reference['records']]
            assert [r['case'] for r in old]==[r['case'] for r in new]==expected
            assert len(new)==103 and sum(r['positive'] for r in new)==94
            for version,rows in [('july150k',old),('r9',new)]:
                for r in rows:
                    if not r['positive']:continue
                    assert r['seed_count']==4
                    for seed,h in r['seed_evidence'].items():
                        f=d/version/(r['case']+'_'+seed+'.json');assert sha(f)==h
                        v=json.loads(f.read_text());assert v['checkpoint_sha256']==c['models'][version][mod]['sha256']
            o=[r for r in old if r['positive']];n=[r for r in new if r['positive']];groups=[r['patient_group'] for r in n]
            values={}
            for target in ('primary','secondary'):
                for metric in ('ssim','mae','rmse','psnr','contrast_abs_error'):
                    keep=[i for i in range(94) if n[i][target][metric] is not None and o[i][target][metric] is not None]
                    assert keep if metric!='contrast_abs_error' else True
                    if not keep:continue
                    name=target+'_'+metric
                    values[name]=paired([n[i][target][metric] for i in keep],[o[i][target][metric] for i in keep],
                        [groups[i] for i in keep],higher=metric in ('ssim','psnr'))
            values['primary_ssim_raw_old_sensitivity']=paired([r['primary']['ssim'] for r in n],
                [r['old_raw_sensitivity']['ssim'] for r in o],groups)
            for size in ('small_le27','large_gt27'):
                keep=[i for i in range(94) if n[i]['size_ssim'][size] is not None and o[i]['size_ssim'][size] is not None]
                if keep:values[size]=paired([n[i]['size_ssim'][size] for i in keep],[o[i]['size_ssim'][size] for i in keep],[groups[i] for i in keep])
            for region in ('ET','NETC','SNFH','RC'):
                for metric in ('ssim','mae','rmse','psnr'):
                    keep=[i for i in range(94) if n[i]['regions'][region][metric] is not None and o[i]['regions'][region][metric] is not None]
                    if keep:values[region+'_'+metric]=paired([n[i]['regions'][region][metric] for i in keep],[o[i]['regions'][region][metric] for i in keep],[groups[i] for i in keep],higher=metric in ('ssim','psnr'))
            values['seed_variability_descriptive']={version:float(np.mean([r['seed_variability']['foreground_voxel_std_mean'] for r in rows if r['positive']])) for version,rows in [('july150k',old),('r9',new)]}
            results[mod]=values
            for metric,v in values.items():
                if metric=='seed_variability_descriptive':continue
                table.append(dict(modality=mod,metric=metric,**{k:val for k,val in v.items() if not isinstance(val,list)},paired_ci95=json.dumps(v['paired_ci95']),group_sign_p_holm4=None))
        adjusted=holm({m:v['primary_ssim']['group_sign_p_uncorrected'] for m,v in results.items()})
        for m,p in adjusted.items():results[m]['primary_ssim']['group_sign_p_holm4']=p
        for row in table:
            if row['metric']=='primary_ssim':row['group_sign_p_holm4']=adjusted[row['modality']]
        with (ROOT/'comparison/VALUES_AND_PAIRED.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
        result=dict(META,status='exploratory_comparison_complete',utc=now(),package_sha256=pkg,models=c['models'],
            reference_report_sha256=c['reference_report_sha256'],evaluation_report_sha256=digests,results=results,
            cases=103,scored_cases=94,negative_noop=9,seeds=[20261013,20261014,20261015,20261016],
            csv_sha256=sha(ROOT/'comparison/VALUES_AND_PAIRED.csv'),manual_review='not_performed',selection='frozen_not_reselected',
            limitation='Conditional paired similarity on repeatedly used validation; not clinical realism or downstream segmentation benefit.')
        write(ROOT/'comparison/REPORT.json',result)
        write(ROOT/'comparison/COMPLETE.ok',dict(META,status='comparison_complete',utc=now(),package_sha256=pkg,report_sha256=sha(ROOT/'comparison/REPORT.json')))
    except BaseException as e:
        write(ROOT/'comparison/FAILED.json',dict(META,utc=now(),message=str(e),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
