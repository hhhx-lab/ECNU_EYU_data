"""Select candidates without shadowing Python's native select module."""
import json
import math
import traceback
from common import META, MODS, ROOT, now, settings, sha, write
from metrics import SEEDS, STEPS, choose_step, cluster_bootstrap, weighted_lesions


def audit_candidate(directory, value, records, modality, step, reference_sha, package_sha, checkpoint_sha):
    assert value['step']==step and value['modality']==modality and value['all_qc_pass'] is True
    assert value['reference_report_sha256']==reference_sha and value['evaluation_package_sha256']==package_sha
    assert value['checkpoint_sha256']==checkpoint_sha and value['seeds']==list(SEEDS)
    rows={r['case']:r for r in value['cases']};expected={r['case']:r for r in records}
    assert len(rows)==len(value['cases'])==len(expected)==103 and set(rows)==set(expected)
    assert sum(r['positive'] for r in rows.values())==94
    files={r['case']+'_'+str(seed)+'.json' for r in records if r['positive'] for seed in SEEDS}
    assert set(value['seed_evidence_sha256'])==files
    assert not list(directory.rglob('FAILED.json'))
    for case,row in rows.items():
        rec=expected[case]
        assert row['patient_group']==rec['patient_group'] and row['positive']==rec['positive']
        if not row['positive']:
            assert row['status']=='NA_no_lesion_noop' and 'primary_ssim' not in row
            continue
        values=[];components=None
        for seed in SEEDS:
            name=case+'_'+str(seed)+'.json';path=directory/name
            assert sha(path)==value['seed_evidence_sha256'][name]
            v=json.loads(path.read_text())
            assert (v['case'],v['patient_group'],v['step'],v['modality'],v['seed'])==(case,rec['patient_group'],step,modality,seed)
            assert v['checkpoint_sha256']==checkpoint_sha and v['reference_sha256']==rec['reference_sha256']
            for target in ('primary','secondary'):
                t=v[target];assert t['support_outside_unchanged'] is True
                assert t['target']==('vae_reconstructed' if target=='primary' else 'authentic_real')
                assert t['lesions'] and all(math.isfinite(l['ssim']) and l['n_voxels']>0 for l in t['lesions'])
                identity=[(l['component'],l['bbox'],l['n_voxels']) for l in t['lesions']]
                assert len({l['component'] for l in t['lesions']})==len(t['lesions'])
                if components is None:components=identity
                assert identity==components and sum(l['n_voxels'] for l in t['lesions'])==rec['label_volume']
                assert abs(t['lesion_ssim']-weighted_lesions(t['lesions']))<1e-12
                assert math.isfinite(t['mse']) and t['mse']>=0
            values.append(v)
        for key,target,metric in [('primary_ssim','primary','lesion_ssim'),('secondary_ssim','secondary','lesion_ssim'),
                                  ('primary_mse','primary','mse'),('secondary_mse','secondary','mse')]:
            assert abs(row[key]-sum(v[target][metric] for v in values)/4)<1e-12
    assert value['visual_evidence']
    for visual in value['visual_evidence']:
        assert sha(visual['path'])==visual['sha256'] and visual['manual_review']=='not_performed'
    return rows


def main():
    write(ROOT/'selection/STARTED.json',dict(META,utc=now()))
    try:
        cfg,r9=settings();package_sha=sha(ROOT/'PACKAGE_MANIFEST.json')
        assert not (ROOT/'reference/FAILED.json').exists()
        reference_sha=json.loads((ROOT/'reference/COMPLETE.ok').read_text())['report_sha256']
        assert reference_sha==sha(ROOT/'reference/REPORT.json')
        ref=json.loads((ROOT/'reference/REPORT.json').read_text());assert ref['errors']==[]
        results={}
        for mod in MODS:
            d=ROOT/'evaluation'/mod
            complete=json.loads((d/'COMPLETE.ok').read_text())
            assert complete['status']=='candidate_evaluation_complete' and complete['steps']==list(STEPS)
            assert complete['modality']==mod and complete['reference_report_sha256']==reference_sha
            assert complete['evaluation_package_sha256']==package_sha
            assert not (d/'FAILED.json').exists()
            candidates={};identities={};names=None
            for step in STEPS:
                p=d/str(step)/'CANDIDATE_COMPLETE.json';v=json.loads(p.read_text())
                assert sha(p)==complete['candidate_summary_sha256'][str(step)]
                info=ref['checkpoints'][mod][str(step)]
                assert sha(info['path'])==info['sha256']
                all_cases=audit_candidate(d/str(step),v,ref['records'],mod,step,reference_sha,package_sha,info['sha256'])
                cases={n:r for n,r in all_cases.items() if r['positive']}
                if names is None:names=sorted(cases)
                assert sorted(cases)==names
                candidates[step]=[cases[n]['primary_ssim'] for n in names]
                identities[step]=dict(candidate_summary_sha256=sha(p),checkpoint_sha256=v['checkpoint_sha256'])
                groups=[cases[n]['patient_group'] for n in names]
            scores={s:sum(v)/len(v) for s,v in candidates.items()};best=choose_step(scores)
            results[mod]=dict(selected_step=best,selected_checkpoint_sha256=identities[best]['checkpoint_sha256'],
                rule='max_case_equal_volume_weighted_lesion_ssim_mean4seed_tol1e-6_earliest',
                cases=names,patient_groups=groups,candidates={str(s):dict(score=scores[s],**identities[s],
                    **cluster_bootstrap(candidates[s],groups,candidates[200000])) for s in STEPS},
                inference_candidate_only=True,manual_review='not_performed',quality_approval='not_run_not_passed')
            results[mod]['selected_checkpoint_path']=ref['checkpoints'][mod][str(best)]['path']
            results[mod]['secondary_scores']={str(s):sum(r['secondary_ssim'] for r in json.loads((d/str(s)/'CANDIDATE_COMPLETE.json').read_text())['cases'] if r['positive'])/94 for s in STEPS}
        write(ROOT/'selection/SELECTION.json',dict(META,utc=now(),modalities=results,status='exploratory_selected'))
        write(ROOT/'selection/COMPLETE.ok',dict(META,utc=now(),status='complete',selection_sha256=sha(ROOT/'selection/SELECTION.json')))
    except Exception as error:
        write(ROOT/'selection/FAILED.json',dict(META,utc=now(),message=str(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
