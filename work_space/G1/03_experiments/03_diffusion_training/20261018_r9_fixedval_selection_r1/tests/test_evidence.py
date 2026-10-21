"""Exercise complete seed evidence and exclusive marker publication."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from common import write, sha
from choose import audit_candidate
from metrics import SEEDS


class EvidenceTests(unittest.TestCase):
    def test_exclusive_atomic_marker(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'COMPLETE.ok';write(p,{'status':'pass'})
            with self.assertRaises(FileExistsError):write(p,{'status':'bad'})
            self.assertEqual(json.loads(p.read_text()),{'status':'pass'})
            self.assertEqual([v.name for v in Path(d).iterdir()],['COMPLETE.ok'])

    def test_nonfinite_never_publishes_marker(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'COMPLETE.ok'
            with self.assertRaises(ValueError):write(p,{'score':float('nan')})
            self.assertFalse(p.exists())

    def test_all_seed_evidence_and_mutations(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d);records=[];rows=[];files={}
            for i in range(103):
                case='case'+str(i);positive=i<94
                rec=dict(case=case,patient_group='group'+str(i//2),positive=positive,label_volume=1,reference_sha256='ref'+str(i))
                records.append(rec);row={k:rec[k] for k in ('case','patient_group','positive')}
                if positive:
                    row.update(primary_ssim=.8,secondary_ssim=.8,primary_mse=1.,secondary_mse=1.)
                    for seed in SEEDS:
                        metric=dict(target='vae_reconstructed',support_outside_unchanged=True,mse=1.,lesion_ssim=.8,
                            lesions=[dict(component=1,bbox=[1,2,3,4,5,6],n_voxels=1,ssim=.8)])
                        v=dict(case=case,patient_group=rec['patient_group'],step=200000,modality='t1c',seed=seed,
                            checkpoint_sha256='checkpoint',reference_sha256=rec['reference_sha256'],primary=metric,
                            secondary={**metric,'target':'authentic_real'})
                        p=directory/(case+'_'+str(seed)+'.json');write(p,v);files[p.name]=sha(p)
                else:row['status']='NA_no_lesion_noop'
                rows.append(row)
            visual=directory/'visual.png';visual.write_bytes(b'fixture')
            value=dict(step=200000,modality='t1c',all_qc_pass=True,reference_report_sha256='report',evaluation_package_sha256='package',
                checkpoint_sha256='checkpoint',seeds=list(SEEDS),cases=rows,seed_evidence_sha256=files,
                visual_evidence=[dict(path=str(visual),sha256=sha(visual),manual_review='not_performed')])
            audit=lambda v:audit_candidate(directory,v,records,'t1c',200000,'report','package','checkpoint')
            self.assertEqual(len(audit(value)),103)
            for mutation in ('duplicate','missing_seed','wrong_group','wrong_package','wrong_mean','negative_score'):
                bad=copy.deepcopy(value)
                if mutation=='duplicate':bad['cases'][-1]=bad['cases'][0]
                if mutation=='missing_seed':bad['seed_evidence_sha256'].pop(next(iter(files)))
                if mutation=='wrong_group':bad['cases'][0]['patient_group']='other'
                if mutation=='wrong_package':bad['evaluation_package_sha256']='drift'
                if mutation=='wrong_mean':bad['cases'][0]['primary_ssim']=.7
                if mutation=='negative_score':bad['cases'][-1]['primary_ssim']=0
                with self.subTest(mutation=mutation),self.assertRaises(AssertionError):audit(bad)


if __name__=='__main__':unittest.main()
