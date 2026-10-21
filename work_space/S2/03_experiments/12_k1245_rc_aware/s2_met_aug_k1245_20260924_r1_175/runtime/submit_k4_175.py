import argparse, datetime as dt, hashlib, json, os, subprocess
from pathlib import Path

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p, x):
    fd=os.open(p, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o400)
    with os.fdopen(fd, 'w') as f: json.dump(x,f,indent=2,sort_keys=True); f.write('\n')

ap=argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True); a=ap.parse_args(); root=a.root.resolve(); ev=root/'evidence'; now=dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00','Z')
plan=root/'K1245_RC_AWARE_175_EXECUTION_PLAN_20260924_R1.json'; slurm=root/'slurm/01_train_k4_175.slurm'
assert plan.is_file() and slurm.is_file() and '175' in plan.read_text() and '175' in slurm.read_text()
assert not any((ev/n).exists() for n in ('K4_175_TRAINING_STATIC_PREFLIGHT.json','K4_175_TRAINING_LAUNCH.json','K4_175_TRAINING_RELEASED.json'))
q=subprocess.run(['squeue','-h','-n','s2k4train175r1','-o','%i'],capture_output=True,text=True,check=True).stdout.strip(); assert not q, q
for rel in ('config/K1245_ROUTE_K4.json','pools/small8/component_manifest.json','evidence/REMOTE_STATIC_PREFLIGHT.json','evidence/POOL_BUILD_COMPLETE.ok'):
 assert (root/rel).exists(), rel
pre={'artifact_status':'experimental_unvalidated','operator_approved':False,'formal_gate_status':'not_run_not_passed','status':'pass','job_name':'s2k4train175r1','epochs':175,'checkpoint_epochs':[25,50,75,100,125,150,175],'plan_sha256':sha(plan),'slurm_sha256':sha(slurm),'timestamp_utc':now,'F_STRICT_PREREG_GATE':False}
put(ev/'K4_175_TRAINING_STATIC_PREFLIGHT.json',pre)
res=subprocess.run(['sbatch','--hold','--parsable',str(slurm)],capture_output=True,text=True,check=True); jid=res.stdout.strip().split(';',1)[0]; assert jid.isdigit()
put(ev/'K4_175_TRAINING_LAUNCH.json',{**pre,'status':'submitted_held','job_id':jid})
subprocess.run(['scontrol','release',jid],check=True)
put(ev/'K4_175_TRAINING_RELEASED.json',{**pre,'status':'released','job_id':jid,'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00','Z')})
print(json.dumps({'job_id':jid,'job_name':'s2k4train175r1','status':'released'},sort_keys=True))
