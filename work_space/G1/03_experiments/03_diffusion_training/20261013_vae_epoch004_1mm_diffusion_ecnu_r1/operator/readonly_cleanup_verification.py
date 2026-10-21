"""Read-only loader and representative compressed/unpacked cache verification."""

import datetime
import json
import shlex

from common import m, transport


def main():
    access=transport('readonly_cleanup_verification')
    code=r'''
import datetime,json,os,pathlib,re,subprocess
home=pathlib.Path('/public/home/zqchen')
seg=home/'SegMamba-main'
sources={}
for name in ('light_training/dataloading/dataset.py','light_training/dataloading/utils.py'):
    p=seg/name
    if p.is_file():
        lines=p.read_text(errors='replace').splitlines()
        selected=lines[:90] if name.endswith('dataset.py') else lines[:150]
        sources[name]={'excerpt':'\n'.join(selected),'total_lines':len(lines)}
samples=[('data/fullres/train','BraTS-MET-00001-000'),('data/fullres/train','BraTS-MET-00002-000'),
         ('data/validation_preprocessed','BraTS-MET-00833-000')]
probe=r"""
import json,pathlib,sys,numpy as np
root=pathlib.Path(sys.argv[1]);pairs=json.loads(sys.argv[2]);out=[]
for relative,case in pairs:
    p=root/relative
    with np.load(p/(case+'.npz'),allow_pickle=False) as z:
        record={'directory':str(p),'case_id':case,'npz_keys':z.files,'checks':{}}
        for key,suffix in (('data','.npy'),('seg','_seg.npy')):
            if key not in z.files:
                record['checks'][key]={'present':False};continue
            a=z[key];b=np.load(p/(case+suffix),mmap_mode='r',allow_pickle=False)
            record['checks'][key]={'present':True,'compressed_shape':list(a.shape),'unpacked_shape':list(b.shape),
                                  'compressed_dtype':str(a.dtype),'unpacked_dtype':str(b.dtype),'finite':bool(np.isfinite(a).all()),
                                  'array_equal':bool(a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b))}
        out.append(record)
print(json.dumps(out,sort_keys=True))
"""
try:
    r=subprocess.run([PYTHON,'-B','-c',probe,str(seg),json.dumps(samples)],capture_output=True,text=True,timeout=60)
    sample_result={'exit_code':r.returncode,'stderr':r.stderr[-2500:]}
    if r.returncode==0:sample_result['samples']=json.loads(r.stdout)
    else:sample_result['stdout_tail']=r.stdout[-2500:]
except subprocess.TimeoutExpired:sample_result={'exit_code':124,'status':'bounded_timeout'}
installed=[]
envs=home/'.conda/envs'
for name in ('segmamba','diffusion','brats_g1_v2','brats_swin'):
    env=envs/name
    for site in env.glob('lib/python*/site-packages'):
        for module in ('mamba_ssm','causal_conv1d'):
            p=site/module
            installed.append({'environment':name,'module':module,'exists':p.exists(),'symlink':p.is_symlink(),
                              'resolved':str(p.resolve()),'references_source_build':str(seg) in str(p.resolve())})
        for p in site.glob('*.pth'):
            if p.stat().st_size>50000:continue
            matches=[line for line in p.read_text(errors='replace').splitlines() if str(seg) in line or 'mamba/build' in line or 'causal-conv1d/build' in line]
            if matches:installed.append({'environment':name,'pth':p.name,'matching_lines':matches})
maps=[]
for p in pathlib.Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        if p.stat().st_uid!=os.getuid():continue
        for line in (p/'maps').read_text(errors='replace').splitlines():
            if str(seg/'mamba/build') in line or str(seg/'causal-conv1d/build') in line:
                maps.append({'pid':p.name,'mapped_path':line.split(maxsplit=5)[-1]})
    except (OSError,PermissionError):continue
v={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,
   'loader_source_excerpt':sources,'sample_array_comparison':sample_result,'selected_environment_module_paths':installed,
   'local_process_build_mappings':maps,'boundary':'only three representative data/seg array pairs; not full1475case equality; no writes/import project/automatic unpack/cleanup; login process maps not cluster-wide dependency proof'}
print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(v,sort_keys=True),flags=re.I))
'''.replace('PYTHON',repr(m.PYTHON))
    compile(code,'<readonly_cleanup_verification>','exec')
    try:
        access.start_vpn()
        data=json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+m.PYTHON+' -B -c '+shlex.quote(code),timeout=95).strip().splitlines()[-1])
        data.update(m.META)
        data['local_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit/'CLEANUP_VERIFICATION.json',data)
        print(json.dumps({'audit':str(access.audit),'samples':data['sample_array_comparison'],
                          'installed':data['selected_environment_module_paths'],'mappings':data['local_process_build_mappings'],
                          'loader':data['loader_source_excerpt']},ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__=='__main__':
    main()
