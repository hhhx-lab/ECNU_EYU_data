"""Targeted, read-only installed-module and K1 metadata follow-up."""
import preflight

preflight.CODE = r'''
import hashlib,importlib.metadata,json,os,stat,subprocess
from pathlib import Path
from datetime import datetime,timezone

def command(argv,timeout=25):
    try:
        p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
        return dict(returncode=p.returncode,stdout=p.stdout[:42000],stderr=p.stderr[:8000])
    except (OSError,subprocess.TimeoutExpired) as e:
        return dict(returncode=124 if isinstance(e,subprocess.TimeoutExpired) else 127,error=type(e).__name__)
def metadata(p):
    if not p.exists():return dict(exists=False)
    s=p.stat()
    return dict(exists=True,bytes=s.st_size,mode=oct(stat.S_IMODE(s.st_mode)),uid=s.st_uid,gid=s.st_gid)
def document(p):
    out=metadata(p)
    if p.is_file() and p.stat().st_size<=100000:
        b=p.read_bytes();out.update(sha256=hashlib.sha256(b).hexdigest(),text=b.decode())
    return out

out=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed',
    status='capability_details_not_execution_pass',remote_started_utc=datetime.now(timezone.utc).isoformat(),hostname=os.uname().nodename)
load='module load singularity-ce/4.1.1; '
out['module_load_and_buildcfg']=command(['bash','-lc',load+'command -v singularity; singularity version; singularity buildcfg'])
out['module_help']=command(['bash','-lc','module help singularity-ce/4.1.1 2>&1'])
out['container_exec_help']=command(['bash','-lc',load+'singularity exec --help'])
out['container_network_help']=command(['bash','-lc',load+'singularity instance start --help'])
out['container_config_files']={}
out['container_helpers']={}
lines=out['module_load_and_buildcfg'].get('stdout','').splitlines()
config={}
for line in lines:
    if '=' in line:
        key,value=line.split('=',1);config[key.strip()]=value.strip().strip('"')
if 'SYSCONFDIR' in config:
    directory=Path(config['SYSCONFDIR'])/'singularity'
    for name in ['singularity.conf','capability.json','ecl.toml']:
        p=directory/name
        if p.is_file() and os.access(p,os.R_OK):out['container_config_files'][str(p)]=document(p)
if 'LIBEXECDIR' in config:
    directory=Path(config['LIBEXECDIR'])/'singularity/bin'
    out['container_helpers']={str(directory/n):metadata(directory/n) for n in ['starter','starter-suid']}
out['partitions']=command(['sinfo','-h','-p','a100,cpu_192G','-o','%P|%a|%l|%D|%G|%c|%m'])
out['qos']=command(['sacctmgr','-n','-P','show','qos','where','name=user_zqchen','format=Name,MaxJobsPU,MaxSubmitJobsPU,MaxTRESPU,MaxWall,Flags'])
out['partition_details']=command(['scontrol','show','partition','a100'])
out['cpu_partition_details']=command(['scontrol','show','partition','cpu_192G'])
k1=Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_met_aug_k1_selected103_official_met_20261008_r1')
model=k1/'cohort/fixed103/models/K1'
out['k1_prediction_manifest']=document(model/'PREDICTION_SHA256.json')
out['k1_inference_audit']=document(model/'inference_audit.json')
prediction=model/'prediction'
out['k1_prediction_inventory']=[dict(name=p.name,**metadata(p)) for p in sorted(prediction.iterdir())] if prediction.is_dir() else []
out['k1_evaluation_inventory']=[dict(name=p.name,**metadata(p)) for p in sorted((model/'evaluation').iterdir())] if (model/'evaluation').is_dir() else []
names=['BraTS-evaluation','panoptica','numpy','scipy','nibabel','SimpleITK']
probe='import importlib.metadata as m,json; result={};\nfor n in '+repr(names)+':\n try: result[n]=m.version(n)\n except m.PackageNotFoundError: result[n]=None\nprint(json.dumps(result))'
out['evaluation_environment_metadata']=command(['/public/home/zqchen/.conda/envs/brats_eval/bin/python','-B','-c',probe])
out['evaluation_source_inventory']=[dict(name=p.name,**metadata(p)) for p in sorted((k1/'runtime').iterdir())] if (k1/'runtime').is_dir() else []
out['scope']=dict(remote_writes=False,models_downloaded=False,image_voxels_read=False,dependencies_installed=False,jobs_created=False,
    module_loaded_only_in_child_shell=True,container_executed=False,namespace_policy_changed=False)
out['remote_utc']=datetime.now(timezone.utc).isoformat()
print(json.dumps(out))
'''

if __name__ == '__main__':
    preflight.main()
