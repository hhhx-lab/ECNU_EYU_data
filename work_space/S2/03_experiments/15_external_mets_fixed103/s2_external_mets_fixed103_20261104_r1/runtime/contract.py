"""Strict fixed103 inputs, native labels and immutable package checks."""
from datetime import datetime,timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import re

META=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed')
MODALITIES=('t1n','t1c','t2w','t2f')
BIO_MODALITIES=('t1c','t1n','t2f','t2w')
CASE_PATTERN=re.compile(r'^BraTS-MET-(\d{5})-(\d{3})$')


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(payload,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    path.chmod(0o440)


def evidence(path,status,**values):
    result=dict(META,status=status,remote_utc=datetime.now(timezone.utc).isoformat(),job_id=os.environ.get('SLURM_JOB_ID'),**values)
    write(path,result);return result


def fixed_rows(path):
    with Path(path).open(newline='') as f:rows=list(csv.DictReader(f))
    require(len(rows)==103,'fixed103 count drift')
    ids=[r['source_case_id'] for r in rows]
    require(len(set(ids))==103 and all(CASE_PATTERN.fullmatch(x) for x in ids),'fixed103 ID/duplicate drift')
    require(all(r['source_t2w_allowed']=='True' and r['t2w_status']=='authentic' and r['completion_required']=='False' for r in rows),'missing or generated T2W prohibited')
    require(len({CASE_PATTERN.fullmatch(x).group(1) for x in ids})==69,'fixed103 patientgroups drift')
    return rows


def input_map(rows,manifest):
    files=manifest['files'];require(len(files)==515,'515 input/reference file identities required')
    mapped={}
    for row in rows:
        case=row['source_case_id'];out={}
        for index,modality in enumerate((*MODALITIES,'seg')):
            name=case+'.nii.gz' if modality=='seg' else f'{case}_{index:04d}.nii.gz'
            matches=[p for p in files if Path(p).name==name]
            require(len(matches)==1,'input/reference identity missing/ambiguous: '+name)
            p=Path(matches[0]);require(p.is_absolute(),'absolute input path required')
            require(p.parent.name==('reference' if modality=='seg' else 'input'),'input/reference role drift')
            out[modality]=dict(path=str(p),sha256=files[str(p)])
        mapped[case]=out
    require(len({v['path'] for case in mapped.values() for v in case.values()})==515,'input/reference aliases')
    return mapped


def internal_map(rows):
    return {r['source_case_id']:f'BraTS-MET-{i:05d}-000' for i,r in enumerate(rows)}


def label_array(image):
    import numpy as np
    a=np.asarray(image.dataobj)
    require(a.ndim==3,'NIfTI must be 3D')
    require(np.isfinite(a).all(),'NIfTI has nonfinite voxels')
    require(np.equal(a,np.rint(a)).all(),'segmentation has fractional labels')
    require(set(np.unique(a).tolist())<=set(range(5)),'segmentation illegal labels')
    return a


def native_geometry(image,reference):
    import nibabel as nib
    import numpy as np
    require(image.shape==reference.shape and len(image.shape)==3,'native shape mismatch')
    require(np.allclose(image.affine,reference.affine,atol=1e-4,rtol=0),'native affine mismatch')
    require(np.allclose(image.header.get_zooms()[:3],reference.header.get_zooms()[:3],atol=1e-4,rtol=0),'native spacing mismatch')
    require(nib.aff2axcodes(image.affine)==nib.aff2axcodes(reference.affine),'native axes mismatch')


def static_contract(root):
    root=Path(root);contract=read(root/'contract/EXECUTION_CONTRACT.json')
    require(all(contract[k]==v for k,v in META.items()),'scientific status drift')
    require(str(root)==contract['remote_root'],'remote root drift')
    for name,digest in read(root/'contract/PACKAGE_SHA256.json')['files'].items():
        require(sha(root/name)==digest,'frozen runtime drift: '+name)
    require(contract['stage']=='S1_only' and contract['full_jobs_created'] is False,'stage/job graph drift')
    return contract
