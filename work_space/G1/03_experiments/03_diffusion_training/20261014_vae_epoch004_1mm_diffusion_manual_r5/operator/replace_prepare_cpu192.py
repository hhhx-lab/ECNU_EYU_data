"""One held CPU replacement, audit dependent migration, retire old producer, release."""
import datetime
import json
import shlex

from common import REMOTE, ROOT, SOURCE_CODE, m, transport


def main():
    access = transport('replace_prepare_cpu192')
    original = (ROOT / 'slurm/prepare.slurm').read_text()
    script = original.replace('--partition=cpu_96G', '--partition=cpu_192G')
    script = script.replace('g1d1m5p1', 'g1d1m5p2')
    assert script != original and '--mem=90G' in script and '--time=24:00:00' in script
    auth = {**m.META, 'status': 'replacement_authorized',
            'user_authorization': '总之我希望，立刻提交训练，而不是等待管理员处理，那种方式能让我最快进入训练提交',
            'context': 'replace CPU on cpu_192G; migrate existing four GPU dependencies and retire old held CPU',
            'old_prepare_job': '4069915', 'gpu_jobs': ['4069916','4069917','4069918','4069919'],
            'resource': {'partition': 'cpu_192G','cpus': 8,'mem': '90G','walltime': '24:00:00'},
            'original_package_sha256': '13f188b2609ba57544c1258aa368daeabe0201ecbdb81c3ec86700f6441b6ede',
            'no_admin_contact': True, 'no_duplicate_training_submission': True,
            'script': script, 'local_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    m.exclusive(access.audit / 'AUTHORIZATION_AND_SCRIPT.json', auth)
    code = r'''
import datetime,hashlib,json,os,pathlib,re,subprocess,traceback
root=pathlib.Path(__ROOT__);source=pathlib.Path(__SOURCE__);auth=__AUTH__
meta={k:auth[k] for k in ('artifact_status','operator_approved','formal_gate_status')}
audit=root/'evidence'/__AUDIT_NAME__
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name,p):
 with os.fdopen(os.open(audit/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o440),'w') as h:
  json.dump(p,h,sort_keys=True,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
def fields(text):return dict(re.findall(r'([A-Za-z][A-Za-z0-9_/:]*)=([^\s]+)',text))
def command(args):
 r=subprocess.run(args,text=True,capture_output=True,timeout=30)
 return {'argv':args,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'utc':now()}
def checked(args):
 r=command(args)
 assert r['returncode']==0,json.dumps(r)
 return r
def job(job_id):return checked(['scontrol','show','job',job_id,'-o'])
def dependent(job_id,producer):
 r=job(job_id);f=fields(r['stdout'])
 assert f['JobId']==job_id and f['JobState']=='PENDING' and f['RunTime']=='00:00:00'
 assert f['Account']=='zqchen' and f['Partition']=='a100'
 assert f['Dependency'].split('(')[0]=='afterok:'+producer
 assert f['NumCPUs']=='8' and f['MinMemoryNode']=='64G' and f['TimeLimit']=='2-00:00:00'
 assert 'gres/gpu=1' in f['TRES'].split(',') and f.get('KillOInInvalidDependent')=='Yes'
 assert f['Command'].startswith(str(root/'slurm')+'/')
 return r
def verify_new(r,job_id,held=True):
 f=fields(r['stdout'])
 assert f['JobId']==job_id and f['JobName']=='g1d1m5p2' and f['Account']=='zqchen'
 assert f['Partition']=='cpu_192G' and f['NumCPUs']=='8' and f['NumTasks']=='1' and f['CPUs/Task']=='8'
 assert f['MinMemoryNode']=='90G' and f['TimeLimit']=='1-00:00:00'
 assert f['NumNodes'] in ('1','1-1') and f['Dependency']=='(null)' and f['Features']=='(null)'
 assert f['Command']==str(audit/'prepare_cpu192.slurm')
 assert f['StdOut']==str(root/'logs'/('g1d1m5p2_'+job_id+'.out'))
 assert f['StdErr']==str(root/'logs'/('g1d1m5p2_'+job_id+'.err'))
 if held:
  assert f['JobState']=='PENDING' and f['Reason']=='JobHeldUser' and f['RunTime']=='00:00:00'
 else:
  assert f['JobState'] in ('PENDING','RUNNING') and f['Reason'] not in ('JobHeldUser','JobHeldAdmin','BadConstraints')
 return f
audit.mkdir(mode=0o700,exist_ok=False)
stage='precheck';new_id=None;mutations=[];migrated=[];old_cancelled=False
try:
 write('AUTHORIZATION.json',{**auth,'remote_utc':now()})
 assert not (root/'evidence/PREPARE_REPLACEMENT_ACTIVE.json').exists(),'Replacement already bound; refuse duplication'
 assert not (root/'evidence/PREPARE_REPLACEMENT_RESERVATION.json').exists(),'A replacement was already reserved; do not resubmit'
 old=job(auth['old_prepare_job']);of=fields(old['stdout'])
 assert of['JobState']=='PENDING' and of['Reason']=='JobHeldAdmin' and of['RunTime']=='00:00:00'
 assert of['Account']=='zqchen' and of['JobName']=='g1d1m5p1' and of['Partition']=='cpu_192G'
 assert of['NumCPUs']=='8' and of['MinMemoryNode']=='90G' and of['TimeLimit']=='1-00:00:00'
 assert of['Command']==str(root/'slurm/prepare.slurm') and of['Dependency']=='(null)'
 assert not (root/'dataset').exists() and not (root/'training').exists(),'Prior producer output exists; stop'
 package=json.loads((root/'plan/PACKAGE_MANIFEST.json').read_text())
 assert sha(root/'plan/PACKAGE_MANIFEST.json')==auth['original_package_sha256'] and len(package['files'])==57
 assert all(sha(root/p)==h for p,h in package['files'].items())
 assert sha(source/'data/data_csv.csv')=='0d1765e12cdad7d8a5df57a5d2c46c79414ed99a2015a4d8fed57cd4c4fc836b'
 before={j:dependent(j,auth['old_prepare_job']) for j in auth['gpu_jobs']}
 partition=checked(['scontrol','show','partition','cpu_192G','-o']);pf=fields(partition['stdout'])
 assert pf['State']=='UP' and pf['MaxTime']=='UNLIMITED' and pf['ReqResv']=='NO'
 assert pf['AllowAccounts']=='ALL' or 'zqchen' in pf['AllowAccounts'].split(',')
 assert pf['AllowGroups']=='ALL' or 'users' in pf['AllowGroups'].split(',')
 assert pf['AllowQos']=='ALL' or 'user_zqchen' in pf['AllowQos'].split(',')
 names=checked(['scontrol','show','hostnames',pf['Nodes']])['stdout'].splitlines()
 assert names
 nodes=checked(['scontrol','show','node',','.join(names),'-o'])
 eligible=[]
 for line in nodes['stdout'].splitlines():
  f=fields(line)
  capacity=int(f['RealMemory'])-int(f.get('MemSpecLimit','0'))
  if int(f['CPUTot'])>=8 and capacity>=92160 and not any(s in f['State'] for s in ('DOWN','DRAIN','FAIL','NO_RESPOND','FUTURE')):
   eligible.append({'node':f['NodeName'],'allocatable_mib':capacity,'state':f['State']})
 assert eligible,'No usable node meets unchanged request'
 qos=checked(['sacctmgr','-n','-P','show','qos','where','name=user_zqchen,'+pf['QoS'],
             'format=Name,Flags,MaxWall,MaxTRESPerJob,MaxTRESPerNode,MinTRESPerJob,MaxJobsPU,MaxSubmitPU'])
 qr=[line.split('|') for line in qos['stdout'].splitlines() if line.strip()]
 assert any(r[0].lower()=='user_zqchen' for r in qr)
 assert all(not r[i] for r in qr for i in (2,3,4,5)), 'Populated QOS limit requires review'
 disk=checked(['df','-B1',str(root)]);public=checked(['df','-h','/public'])
 available=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize
 assert available>=213513415862,'Unchanged storage budget no longer met'
 script=audit/'prepare_cpu192.slurm'
 old_text=(root/'slurm/prepare.slurm').read_text()
 assert auth['script']==old_text.replace('--partition=cpu_96G','--partition=cpu_192G').replace('g1d1m5p1','g1d1m5p2')
 with os.fdopen(os.open(script,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o440),'w') as h:h.write(auth['script'])
 checked(['bash','-n',str(script)])
 env=os.environ.copy();env['G1_STATIC_DRY_RUN']='1'
 dry=subprocess.run(['bash','-u',str(script)],env=env,text=True,capture_output=True,timeout=10)
 assert dry.returncode==0 and 'G1_R5_BASH_U_DRY_RUN_PASS' in dry.stdout
 static={**meta,'status':'pass','remote_utc':now(),'hostname':os.uname().nodename,'old_job':old,
         'gpu_jobs_before':before,'partition':partition,'node_details':nodes,'eligible_nodes':eligible,
         'qos':qos,'account_disk':disk,'public_disk':public,'available_bytes':available,
         'package_sha256':auth['original_package_sha256'],'script_sha256':sha(script),'dry_run_stdout':dry.stdout}
 write('STATIC_PREFLIGHT.json',static)
 reservation={**meta,'status':'exclusive_reserved','remote_utc':now(),'audit':str(audit),
              'old_prepare_job':auth['old_prepare_job'],'script_sha256':sha(script),'static_sha256':sha(audit/'STATIC_PREFLIGHT.json')}
 with os.fdopen(os.open(root/'evidence/PREPARE_REPLACEMENT_RESERVATION.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o440),'w') as h:
  json.dump(reservation,h,sort_keys=True,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 stage='submit'
 response=command(['sbatch','--parsable','--hold','--export=NONE',str(script)])
 mutations.append(response);write('SUBMISSION_RESPONSE.json',response)
 assert response['returncode']==0,'Unique replacement submission rejected; do not retry'
 new_id=response['stdout'].strip().split(';')[0];assert re.fullmatch(r'[0-9]+',new_id)
 write('CANONICAL_JOB.json',{**meta,'status':'held','job_id':new_id,'script_sha256':sha(script),'replaces':auth['old_prepare_job']})
 held=job(new_id);verify_new(held,new_id)
 write('HELD_LAUNCH_AUDIT.json',{**meta,'status':'held_audited','remote_utc':now(),'job':held,
       'script_sha256':sha(script),'static_sha256':sha(audit/'STATIC_PREFLIGHT.json'),
       'package_sha256':auth['original_package_sha256']})
 stage='migrate_dependencies'
 for j in auth['gpu_jobs']:
  dependent(j,auth['old_prepare_job'])
  r=command(['scontrol','update','JobId='+j,'Dependency=afterok:'+new_id])
  mutations.append(r);write('DEPENDENCY_UPDATE_'+j+'.json',r)
  assert r['returncode']==0,'Dependency update rejected; leave new producer held'
  observed=dependent(j,new_id);migrated.append(j);write('DEPENDENCY_AUDIT_'+j+'.json',observed)
 after={j:dependent(j,new_id) for j in auth['gpu_jobs']}
 verify_new(job(new_id),new_id)
 current_old=fields(job(auth['old_prepare_job'])['stdout'])
 assert current_old['JobState']=='PENDING' and current_old['Reason']=='JobHeldAdmin' and current_old['RunTime']=='00:00:00'
 assert not (root/'dataset').exists()
 write('ALL_DEPENDENCIES_MIGRATED.json',{**meta,'status':'pass','remote_utc':now(),
       'new_prepare_job':new_id,'dependency':'afterok:'+new_id,'jobs':after})
 stage='retire_old'
 r=command(['scancel',auth['old_prepare_job']]);mutations.append(r);write('OLD_CANCEL_RESPONSE.json',r)
 assert r['returncode']==0,'Cannot retire old producer; do not release new producer'
 observation=command(['scontrol','show','job',auth['old_prepare_job'],'-o'])
 accounting=checked(['sacct','-n','-P','-j',auth['old_prepare_job'],'--format=JobID,State,ExitCode,Elapsed'])
 cancelled=(observation['returncode']==0 and fields(observation['stdout']).get('JobState','').startswith('CANCELLED'))
 cancelled=cancelled or any(line.startswith(auth['old_prepare_job']+'|CANCELLED') for line in accounting['stdout'].splitlines())
 assert cancelled,'Cancellation not yet proven; retain new producer held'
 old_cancelled=True;write('OLD_CANCEL_AUDIT.json',{**meta,'status':'cancelled_proven','job':observation,'sacct':accounting,'remote_utc':now()})
 assert all(sha(root/p)==h for p,h in package['files'].items())
 dependent_final={j:dependent(j,new_id) for j in auth['gpu_jobs']}
 binding={**meta,'status':'active_producer_bound','job_id':new_id,'old_job_id':auth['old_prepare_job'],
          'old_status':'cancelled_zero_runtime','dependent_jobs':auth['gpu_jobs'],'dependency':'afterok:'+new_id,
          'remote_audit':str(audit),'script_sha256':sha(script),'package_sha256':auth['original_package_sha256'],
          'partition':'cpu_192G','cpus':8,'memory':'90G','walltime':'24:00:00','remote_utc':now()}
 with os.fdopen(os.open(root/'evidence/PREPARE_REPLACEMENT_ACTIVE.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o440),'w') as h:
  json.dump(binding,h,sort_keys=True,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 stage='release'
 r=command(['scontrol','release',new_id]);mutations.append(r);write('RELEASE_RESPONSE.json',r)
 assert r['returncode']==0,'Release denied; do not retry or convert hold'
 released=job(new_id);verify_new(released,new_id,held=False)
 queue=checked(['squeue','-h','-j',','.join([new_id,*auth['gpu_jobs']]),'-o','%i|%T|%j|%P|%r|%M'])
 receipt={**meta,'status':'replacement_released_dependencies_migrated_old_retired','remote_utc':now(),
          'hostname':os.uname().nodename,'new_prepare_job':new_id,'old_prepare_job':auth['old_prepare_job'],
          'gpu_jobs':auth['gpu_jobs'],'dependency':'afterok:'+new_id,'old_cancelled':True,
          'jobs_submitted':[new_id],'jobs_cancelled':[auth['old_prepare_job']], 'partition':'cpu_192G',
          'resource':auth['resource'],'package_sha256':auth['original_package_sha256'],
          'script_sha256':sha(script),'remote_audit':str(audit),'job_after_release':released,
          'queue_after_release':queue,'dependent_jobs_verified':dependent_final,'available_bytes':available,
          'active_binding':binding,'evidence':{p.name:json.loads(p.read_text()) for p in audit.glob('*.json')}}
 write('REPLACEMENT_RECEIPT.json',receipt)
 print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(receipt,sort_keys=True),flags=re.I))
except BaseException as error:
 failure={**meta,'status':'stopped','stage':stage,'remote_utc':now(),'new_prepare_job':new_id,
          'dependencies_already_migrated':migrated,'old_cancelled':old_cancelled,
          'message':str(error),'traceback':traceback.format_exc(),'mutation_responses':mutations,'remote_audit':str(audit)}
 write('REPLACEMENT_FAILED.json',failure)
 print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(failure,sort_keys=True),flags=re.I))
'''
    for key, value in {'__ROOT__': str(REMOTE),'__SOURCE__': str(SOURCE_CODE),'__AUDIT_NAME__': access.audit.name}.items():
        code = code.replace(key, repr(value))
    code = code.replace('__AUTH__', repr(auth))
    compile(code, '<one_held_cpu_replacement>', 'exec')
    try:
        access.start_vpn()
        result = json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+m.PYTHON+' -B -u -c '+shlex.quote(code), timeout=450).strip().splitlines()[-1])
        result['local_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit / 'REPLACEMENT_RECEIPT.json', result)
        print(json.dumps({'audit': str(access.audit),**{k:v for k,v in result.items() if k not in ('evidence','dependent_jobs_verified','traceback')}}, ensure_ascii=False))
        if result['status']=='stopped':
            raise RuntimeError('Replacement stopped; preserve state and do not automatically repeat')
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
