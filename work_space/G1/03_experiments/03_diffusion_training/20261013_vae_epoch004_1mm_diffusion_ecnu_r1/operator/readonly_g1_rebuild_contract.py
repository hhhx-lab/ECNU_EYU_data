"""Check actual mask regeneration inputs and public runtime references, read-only."""

import datetime
import json
import shlex

from common import m, transport


def main():
    access = transport('readonly_g1_rebuild_contract')
    code = r'''
import csv,datetime,json,os,pathlib,re
root=pathlib.Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v3')
data=root/'data'
with (data/'data_csv.csv').open() as f:rows=list(csv.DictReader(f))
records=[]
for row in rows:
    case=row['id'];p=data/'latents'/case
    records.append({'case_id':case,'split':row.get('split'),
                    'spatial_transform_exists':(p/'spatial_transform.json').is_file(),
                    'source_seg_exists':any((data/'input'/case).glob('*-seg.nii*'))})
refs=[]
for p in list(root.glob('*.py'))+list((root/'synthesis').glob('*.py'))+list((root/'slurm').glob('*.slurm')):
    if p.stat().st_size>250000:continue
    for i,line in enumerate(p.read_text(errors='replace').splitlines()):
        if re.search(r'256.?256.?160|attmasks_shapes_list|attmask_|lesion_weights_|spatial_transform|val_attmasks_path',line) and not re.search(r'password|secret|token|credential',line,re.I):
            refs.append({'path':str(p),'line':i+1,'text':line[:300]})
output={}
for base in (data/'output/run_3104668',data/'evaluation/val'):
    groups={}
    for p in base.rglob('*'):
        if not p.is_file() or p.is_symlink():continue
        rel=p.relative_to(base);g='intermediate' if any('intermediate' in x for x in rel.parts) else 'other'
        s=p.stat();d=groups.setdefault(g,{'files':0,'allocated_bytes':0});d['files']+=1;d['allocated_bytes']+=s.st_blocks*512
    output[str(base)]=groups
v={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,
   'cases':records,'transform_present_count':sum(x['spatial_transform_exists'] for x in records),
   'source_seg_present_count':sum(x['source_seg_exists'] for x in records),'public_consumption_references':refs,
   'output_file_groups':output,'boundary':'existence and source references only, no voxel reads or regeneration; intermediate names are not deletion proof'}
print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(v,sort_keys=True),flags=re.I))
'''
    compile(code, '<readonly_g1_rebuild_contract>', 'exec')
    try:
        access.start_vpn()
        data = json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -c ' + shlex.quote(code), timeout=55).strip().splitlines()[-1])
        data.update(m.META)
        data['local_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit / 'G1_REBUILD_CONTRACT.json', data)
        print(json.dumps({'audit':str(access.audit),'cases':len(data['cases']),
                          'transform_present_count':data['transform_present_count'],
                          'source_seg_present_count':data['source_seg_present_count'],
                          'output_file_groups':data['output_file_groups']},ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
