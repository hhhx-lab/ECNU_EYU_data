"""One bounded read-only capacity and parent identity probe."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[4]
spec = importlib.util.spec_from_file_location('shared_access', PROJECT / 'work_space/G1/03_experiments/03_diffusion_training/20261020_raw_real_modality_specific_1mm_r2/operator/common.py')
access = importlib.util.module_from_spec(spec)
spec.loader.exec_module(access)
access.m.ROOT = ROOT

CODE = r'''
import json,hashlib,os,subprocess
from pathlib import Path
from datetime import datetime,timezone
base=Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated')
root=base/'g1_r9_fixedval_selection_20261018_r1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def call(a):
 p=subprocess.run(a,capture_output=True,text=True,timeout=30)
 return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
report=root/'reference/REPORT.json';sel=root/'selection/SELECTION.json'
assert sha(report)=='13e57c28d597e7801e503219da15350eda68a044550ed0d436edee84c1c70d3c'
assert sha(sel)=='541d0fd245ba66841f35574aed3ad227efa9363c580211a1c8733f528d808902'
r=json.loads(report.read_text());assert r['cases']==103 and r['positive_cases']==94 and r['errors']==[]
out=dict(status='pass',remote_utc=datetime.now(timezone.utc).isoformat(),reference_sha256=sha(report),selection_sha256=sha(sel),
 reference_cache_files=sum(Path(v['reference']).is_file() for v in r['records']),
 account_queue=call(['squeue','-h','-u','zqchen','-o','%i|%T|%j|%P|%r|%M|%b']),
 disk=call(['df','-B1',str(base)]),available_bytes=os.statvfs(base).f_bavail*os.statvfs(base).f_frsize,
 a100=call(['scontrol','show','partition','a100','-o']),cpu=call(['scontrol','show','partition','cpu_192G','-o']),
 qos=call(['sacctmgr','-n','-P','show','qos','where','name=user_zqchen,partition_a100','format=Name,MaxWall,MaxJobsPU,MaxSubmitPU']))
print(json.dumps(out))
'''

if __name__ == '__main__':
    access.run_remote('probe', CODE, timeout=120)
