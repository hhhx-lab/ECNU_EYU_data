"""Single bounded read-only access, raw-root, storage and scheduler probe."""
from common import ROOT, run_remote

CODE = r'''
import datetime,json,os,pathlib,subprocess,traceback
result={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'pass'}
def call(argv):
 p=subprocess.run(argv,capture_output=True,text=True,timeout=25)
 assert p.returncode==0, str(argv)+': '+p.stderr
 return p.stdout
try:
 roots=['/public/home/zqchen/data/MICCAI-LH-BraTS2025-MET-Challenge-Training',
 '/public/home/zqchen/data/MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels']
 result['roots']={r:{'exists':pathlib.Path(r).is_dir(),'resolved':str(pathlib.Path(r).resolve())} for r in roots}
 code=pathlib.Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v3')
 result['samples']={}
 for case in ['BraTS-MET-00001-000','BraTS-MET-01094-003','BraTS-MET-01184-002']:
  p=code/'data/input'/case
  result['samples'][case]={'directory':str(p),'exists':p.exists(),'resolved':str(p.resolve()),
   'files':{q.name:str(q.resolve()) for q in p.glob('*.nii.gz')} if p.exists() else {}}
 result['df']=call(['df','-B1','/public/home/zqchen'])
 result['inodes']=call(['df','-i','/public/home/zqchen'])
 result['queue']=call(['squeue','-h','-u','zqchen','-o','%i|%T|%j|%P|%r'])
 result['partitions']={p:call(['scontrol','show','partition',p,'-o']) for p in ['cpu_192G','a100']}
 result['qos']=call(['sacctmgr','-n','-P','show','qos','where','name=user_zqchen,partition_a100','format=Name,MaxWall,MaxJobsPU,MaxSubmitPU'])
except BaseException as e:result.update(status='failed',message=str(e),traceback=traceback.format_exc())
print(json.dumps(result))
'''

if __name__ == '__main__':
    run_remote('source_resource_probe', CODE, timeout=150)
