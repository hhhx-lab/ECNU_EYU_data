"""Read selected public source and artifact metadata without executing loaders."""

import datetime
import json
import shlex

from common import m, transport


def main():
    access = transport('readonly_two_project_identity')
    code = r'''
import collections,datetime,json,os,pathlib,re
home=pathlib.Path('/public/home/zqchen');other=home/'zdn_deepcpg'
old=home/'projects/ECNU_EYU_data/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v3'
sources=[]
names=[other/'setup.py',other/'deepcpg/__init__.py',other/'gpu_nhomo_cpg_20e.slurm',
       other/'gpu_nhomo_dna_20e.slurm',other/'convert_neuron_h5.slurm',
       other/'melissa/README.md',old/'generate_attmask.py',old/'precompute_lesion_weights.py',
       old/'compute_weights.py',old/'slurm/03_encode_and_prepare_aux.slurm',
       old/'training_bbdm.py',old/'training_endec.py']
for p in names:
    if not p.is_file() or p.is_symlink() or p.stat().st_size>250000:continue
    lines=p.read_text(errors='replace').splitlines();selected=[]
    for i,line in enumerate(lines):
        wanted=len(lines)<180 or re.search(r'numpy|np\.|load|save|mask|weight|attention|channel|resume|checkpoint|check_points|model_|val_img|DeepCpG|methyl|HDF|h5|dna|cpg|hemato|neuron|create',line,re.I)
        if wanted and not re.search(r'password|token|secret|api.?key|private.?key|authorization|credential',line,re.I):
            selected.append({'line':i+1,'text':line[:350]})
    sources.append({'path':str(p),'total_lines':len(lines),'selected_lines':selected[:140]})
arrays=[]
for relative in ('attention_masks','lesion_weights','latents'):
    root=old/'data'/relative;groups=collections.defaultdict(lambda:{'files':0,'allocated_bytes':0});examples=[]
    for p in root.rglob('*.npy'):
        s=p.stat();g=groups[p.name];g['files']+=1;g['allocated_bytes']+=s.st_blocks*512
        if len(examples)<8:examples.append({'path':str(p),'logical_bytes':s.st_size})
    arrays.append({'path':str(root),'names':dict(groups),'examples':examples})
jsons=[]
for name in ('data/selected_vae.json','data/g1_v3_preparation_validation.json','data/g1_v3_data_adoption.json'):
    p=old/name
    if p.is_file() and p.stat().st_size<100000:
        d=json.loads(p.read_text());jsons.append({'path':str(p),'data':d})
checkpoints=[]
for p in sorted((old/'training').rglob('*.pt')):
    s=p.stat();checkpoints.append({'path':str(p),'allocated_bytes':s.st_blocks*512,'size_bytes':s.st_size,'nlink':s.st_nlink})
headers=[]
probe=r"""
import json,numpy as np,sys
out=[]
for path in json.loads(sys.argv[1]):
    a=np.load(path,mmap_mode='r',allow_pickle=False)
    out.append({'path':path,'shape':list(a.shape),'dtype':str(a.dtype)})
print(json.dumps(out))
"""
import subprocess
paths=[x['examples'][0]['path'] for x in arrays if x['examples']]
r=subprocess.run([PYTHON,'-B','-c',probe,json.dumps(paths)],capture_output=True,text=True,timeout=20)
if r.returncode==0:headers=json.loads(r.stdout)
else:headers={'exit_code':r.returncode,'stderr':r.stderr[-1000:]}
links=[]
for name in ('input','input_inference'):
    p=old/'data'/name;links.append({'path':str(p),'symlink':p.is_symlink(),'target':os.readlink(p) if p.is_symlink() else None})
v={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,
   'selected_sources':sources,'array_files':arrays,'array_header_samples':headers,'selected_metadata':jsons,
   'training_checkpoint_files':checkpoints,'data_links':links,
   'boundary':'public selected source/JSON/header only; no checkpoint load, generation, deletion, full array comparison or dependency proof'}
print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(v,sort_keys=True),flags=re.I))
'''.replace('PYTHON', repr(m.PYTHON))
    compile(code, '<readonly_two_project_identity>', 'exec')
    try:
        access.start_vpn()
        data = json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -c ' + shlex.quote(code), timeout=65).strip().splitlines()[-1])
        data.update(m.META)
        data['local_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit / 'TWO_PROJECT_IDENTITY.json', data)
        print(json.dumps({'audit':str(access.audit),'array_header_samples':data['array_header_samples'],
                          'checkpoint_count':len(data['training_checkpoint_files']),
                          'data_links':data['data_links']},ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
