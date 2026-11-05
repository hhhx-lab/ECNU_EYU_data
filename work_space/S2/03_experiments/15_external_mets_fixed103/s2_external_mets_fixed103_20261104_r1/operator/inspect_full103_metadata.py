"""Bounded static interfaces and fresh submission metadata, without inference."""
import hashlib
import json
import os
import shlex

from session import session, access_module

CODE = r'''
import hashlib,json,os,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
def read(path):return json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def command(argv):
 p=subprocess.run(argv,capture_output=True,text=True,timeout=40)
 assert len(p.stdout)+len(p.stderr)<100000,'metadata budget'
 return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
identity=read(root/'operator/BIO_SOURCE_IDENTITY.json')
source=root/'assets/bio_source'
interfaces={}
for name,item in identity['files'].items():
 assert sha(source/name)==item['sha256'],'author source drift'
 if name.endswith('config.py') or ('trainers/' in name and name.endswith('.py')) or name.endswith('entrypoint.sh'):
  assert (source/name).stat().st_size<65536,'interface size'
  interfaces[name]=(source/name).read_text()
k1=Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_met_aug_k1_selected103_official_met_20261008_r1')
assert sha(k1/'manifests/INPUT_REFERENCE_SHA256.json')=='5f52bfdc3a957d848063850460226a7c3d9c25f0926e827f728f21e525fbd9df','fixed input manifest identity drift'
manifest=read(k1/'manifests/INPUT_REFERENCE_SHA256.json')
layout={}
for name,digest in manifest['files'].items():
 if Path(name).parent.name!='input':continue
 p=Path(name)
 assert p.is_file(),'source image missing'
 layout[name]=dict(realpath=str(p.resolve()),symlink=p.is_symlink(),bytes=p.stat().st_size,sha256=digest)
runtime=root/'operator/environment_r4'
metadata=dict(author_identity=identity,bio_identity=read(root/'operator/asset_materialization/BIO_FULL_MODEL_IDENTITY.json'),
 sif=read(root/'attempts/s0_sif_recovery_r2/reports/COMPLETE.json')['sif'],image_layout=layout,
 runtime_markers={p.name:read(p) for p in runtime.glob('*.json') if p.name in ('FAILED.json','COMPLETE.json','PLAN_ACCEPTANCE.json')},
 runtime_log_tails={p.name:p.read_text(errors='replace')[-6000:] for p in (root/'logs/environment_r4').glob('*.log')},
 free_bytes=shutil.disk_usage(root).free,quota=command(['quota','-s']),
 backend='/public/software/singularity-ce/4.1.1/bin/singularity',
 backend_help=command(['/public/software/singularity-ce/4.1.1/bin/singularity','exec','--help']),
 partitions=command(['scontrol','show','partition','a100','-o']),
 associations=command(['sacctmgr','-n','-P','show','assoc','where','user=zqchen','format=User,Account,Partition,QOS,MaxJobs,MaxSubmitJobs']),
 qos=command(['sacctmgr','-n','-P','show','qos']),
 own_queue=command(['squeue','-h','-u','zqchen','-o','%i|%T|%P|%j|%r|%b']),
 remote_utc=datetime.now(timezone.utc).isoformat(),images_read=False,models_loaded=False,jobs_created=False,remote_writes=False)
print(json.dumps(dict(metadata=metadata,interfaces=interfaces)))
'''


def main():
    compile(CODE, 'full103_metadata', 'exec')
    with session('inspect_full103_metadata') as access:
        access_module.exclusive(access.audit / 'EXECUTED_SOURCE.json', dict(source=CODE, sha256=hashlib.sha256(CODE.encode()).hexdigest()))
        command = 'env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT=' + shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command += ' ' + shlex.quote(os.environ['ECNU_REMOTE_PYTHON']) + ' -B -c ' + shlex.quote(CODE)
        result = json.loads(access.ssh(command, timeout=300).strip().splitlines()[-1])
        access_module.exclusive(access.audit / 'RESULT.json', result['metadata'])
        metadata = result['metadata']
        print(json.dumps(dict(audit=str(access.audit), interfaces=result['interfaces'],
              runtime_markers=metadata['runtime_markers'], runtime_log_tails=metadata['runtime_log_tails'],
              image_symlinks=sum(x['symlink'] for x in metadata['image_layout'].values()),
              image_realparents=sorted({str(__import__('pathlib').Path(x['realpath']).parent) for x in metadata['image_layout'].values()}),
              partitions=metadata['partitions'], associations=metadata['associations'], qos=metadata['qos'],
              own_queue=metadata['own_queue'], free_bytes=metadata['free_bytes'], remote_utc=metadata['remote_utc'])), flush=True)


if __name__ == '__main__':
    main()
