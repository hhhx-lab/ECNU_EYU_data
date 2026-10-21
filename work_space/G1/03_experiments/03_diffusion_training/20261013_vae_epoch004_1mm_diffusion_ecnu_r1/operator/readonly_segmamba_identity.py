"""Read selected legacy SegMamba public training definitions and model metadata."""

import datetime
import json
import shlex

from common import m, transport


def main():
    access=transport('readonly_segmamba_identity')
    code=r'''
import datetime,json,os,pathlib,re
root=pathlib.Path('/public/home/zqchen/SegMamba-main')
script_summary={}
for name in ('3_train_brats2026.py','config_generator.py','loss_functions.py','2_preprocessing_mri.py'):
    path=root/name
    if path.is_file():
        lines=path.read_text(errors='replace').splitlines()
        selected=[{'line':i+1,'text':line} for i,line in enumerate(lines)
                  if re.search(r'SegMamba|nnUNet|epoch|loss|num_class|in_chan|data_dir|data_path|fullres|patch|batch|exp_|BraTS',line,re.I)]
        script_summary[name]={'bytes':path.stat().st_size,'selected_lines':selected[:90]}
configs=[]
if (root/'configs').is_dir():
    for p in sorted((root/'configs').iterdir()):
        if p.is_file():
            record={'name':p.name,'bytes':p.stat().st_size}
            if p.suffix=='.json' and p.stat().st_size<30000:
                try:
                    v=json.loads(p.read_text())
                    allowed=('experiment_name','exp_name','model','model_name','loss','loss_type','loss_name','epochs','max_epochs','num_epochs','patch_size','batch_size','data_dir','data_path','train_data_dir','val_data_dir','in_channels','out_channels','num_classes')
                    record['public_fields']={k:v[k] for k in allowed if k in v}
                except Exception as e:record['parse_error']=type(e).__name__
            configs.append(record)
models=[]
for path in sorted((root/'results').glob('*/model/*')):
    if path.is_file() and not path.is_symlink():
        s=path.stat()
        models.append({'path':str(path),'bytes':s.st_size,'mtime_utc':datetime.datetime.fromtimestamp(s.st_mtime,datetime.timezone.utc).isoformat()})
dataset_counts={}
for name in ('data/fullres/train','data/validation','data/validation_preprocessed'):
    path=root/name
    if path.is_dir():
        entries=list(path.iterdir())
        dataset_counts[name]={'direct_entries':len(entries),'directories':sum(p.is_dir() for p in entries),'files':sum(p.is_file() for p in entries),
                              'first_names':[p.name for p in sorted(entries)[:6]]}
v={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,
   'scripts':script_summary,'config_files':configs,'model_files':models,'dataset_directory_counts':dataset_counts,
   'boundary':'selected public definitions and file metadata only; not full checkpoint/data/metrics validation, no job status or modifications'}
print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(v,sort_keys=True),flags=re.I))
'''
    try:
        access.start_vpn()
        data=json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+m.PYTHON+' -B -c '+shlex.quote(code),timeout=60).strip().splitlines()[-1])
        data.update(m.META)
        data['local_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit/'SEGMAMBA_IDENTITY.json',data)
        print(json.dumps({'audit':str(access.audit),'scripts':data['scripts'],'models':data['model_files'],'configs':data['config_files'],'dataset_counts':data['dataset_directory_counts']},ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__=='__main__':
    main()
