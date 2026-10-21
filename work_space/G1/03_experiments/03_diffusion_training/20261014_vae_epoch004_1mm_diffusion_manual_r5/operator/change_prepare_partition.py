"""Audit one pending-job partition correction, preserving runtime and dependent IDs."""
import datetime
import json
import shlex

from common import REMOTE, SOURCE_CODE, m, transport


def main():
    access = transport('prepare_cpu192_correction')
    authorization = {**m.META, 'status': 'resource_correction_authorized',
        'user_authorization': '核验并改用 cpu_192G，保持原 8 CPU、90G、24h，同时正确处理四路训练依赖',
        'prepare_job': '4069915', 'dependent_jobs': ['4069916', '4069917', '4069918', '4069919'],
        'scope': 'one existing pending job; hold, update partition only, audit, release; no sbatch or cancellation',
        'package_sha256': '13f188b2609ba57544c1258aa368daeabe0201ecbdb81c3ec86700f6441b6ede'}
    m.exclusive(access.audit / 'AUTHORIZATION.json', authorization)
    code = r'''
import datetime,hashlib,json,os,pathlib,re,subprocess,traceback
root=pathlib.Path(__ROOT__);source=pathlib.Path(__SOURCE__);auth=__AUTH__
audit=root/'evidence'/__AUDIT_NAME__
meta={k:auth[k] for k in ('artifact_status','operator_approved','formal_gate_status')}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name,value):
 with os.fdopen(os.open(audit/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o440),'w') as h:
  json.dump(value,h,sort_keys=True,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
def command(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=30)
 return {'argv':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'utc':now()}
def checked(args):
 p=command(args)
 if p['returncode']:raise RuntimeError('Read rejected: '+json.dumps(p))
 return p
def fields(text):return dict(re.findall(r'([A-Za-z][A-Za-z0-9_/:]*)=([^\s]+)',text))
def job(job_id):return checked(['scontrol','show','job',job_id,'-o'])
def verify_cpu(raw,partition,held=False):
 f=fields(raw['stdout'])
 assert f['JobId']=='4069915' and f['JobName']=='g1d1m5p1' and f['JobState']=='PENDING'
 assert f['RunTime']=='00:00:00' and f['Restarts']=='0' and f['Partition']==partition
 assert f['NumCPUs']=='8' and f['NumTasks']=='1' and f['CPUs/Task']=='8'
 assert f['NumNodes'] in ('1','1-1') and f['MinMemoryNode']=='90G' and f['TimeLimit']=='1-00:00:00'
 assert f['Account']=='zqchen' and f['QOS']=='user_zqchen' and f['Features']=='(null)'
 assert f['Command']==str(root/'slurm/prepare.slurm')
 assert f['StdOut']==str(root/'logs/g1d1m5p1_4069915.out')
 assert f['StdErr']==str(root/'logs/g1d1m5p1_4069915.err') and f['Dependency']=='(null)'
 if held:assert f['Reason'] in ('JobHeldUser','JobHeldAdmin')
 return f
def verify_dependent(job_id):
 raw=job(job_id);f=fields(raw['stdout'])
 assert f['JobId']==job_id and f['JobState']=='PENDING' and f['RunTime']=='00:00:00'
 assert f['Partition']=='a100' and f['Dependency'].split('(')[0]=='afterok:4069915'
 assert f['NumCPUs']=='8' and f['MinMemoryNode']=='64G' and f['TimeLimit']=='2-00:00:00'
 assert f['Command'].startswith(str(root/'slurm')+'/')
 return raw
audit.mkdir(mode=0o700,exist_ok=False)
stage='precheck';commands=[]
try:
 write('AUTHORIZATION.json',{**auth,'remote_utc':now()})
 original=job('4069915');cpu=verify_cpu(original,'cpu_96G')
 assert cpu['Reason']=='BadConstraints','No longer the diagnosed blocked job; stop without change'
 assert not (root/'dataset').exists() and not (root/'training').exists()
 package=json.loads((root/'plan/PACKAGE_MANIFEST.json').read_text())
 assert sha(root/'plan/PACKAGE_MANIFEST.json')==auth['package_sha256']
 assert len(package['files'])==57 and all(sha(root/p)==h for p,h in package['files'].items())
 assert sha(source/'data/data_csv.csv')=='0d1765e12cdad7d8a5df57a5d2c46c79414ed99a2015a4d8fed57cd4c4fc836b'
 dependent_before={j:verify_dependent(j) for j in auth['dependent_jobs']}
 partition=checked(['scontrol','show','partition','cpu_192G','-o']);pf=fields(partition['stdout'])
 assert pf['State']=='UP' and pf['PartitionName']=='cpu_192G'
 assert pf['AllowAccounts']=='ALL' or 'zqchen' in pf['AllowAccounts'].split(',')
 assert pf['AllowGroups']=='ALL' or 'users' in pf['AllowGroups'].split(',')
 assert pf['AllowQos']=='ALL' or 'user_zqchen' in pf['AllowQos'].split(',')
 assert pf['MaxTime']=='UNLIMITED' and pf['ReqResv']=='NO'
 names=checked(['scontrol','show','hostnames',pf['Nodes']])['stdout'].splitlines()
 assert names
 node_response=checked(['scontrol','show','node',','.join(names),'-o'])
 nodes=[fields(line) for line in node_response['stdout'].splitlines() if line.strip()]
 eligible=[{'node':n['NodeName'],'cpu_total':int(n['CPUTot']),
            'real_memory_mib':int(n['RealMemory']),'reserved_mib':int(n.get('MemSpecLimit','0')),
            'allocatable_mib':int(n['RealMemory'])-int(n.get('MemSpecLimit','0')),'state':n['State']}
           for n in nodes if int(n['CPUTot'])>=8 and int(n['RealMemory'])-int(n.get('MemSpecLimit','0'))>=92160
           and not any(s in n['State'] for s in ('DOWN','DRAIN','FAIL','NO_RESPOND','FUTURE'))]
 assert eligible,'No usable node meets the unchanged resource request'
 qos=checked(['sacctmgr','-n','-P','show','qos','where','name=user_zqchen,'+pf['QoS'],
              'format=Name,Flags,MaxWall,MaxTRESPerJob,MaxTRESPerNode,MinTRESPerJob,MaxJobsPU,MaxSubmitPU'])
 qos_rows=[line.split('|') for line in qos['stdout'].splitlines() if line.strip()]
 assert any(r[0].lower()=='user_zqchen' for r in qos_rows)
 # A populated limit requires interpretation; do not silently ignore it.
 assert all(not r[i] for r in qos_rows for i in (2,3,4,5)), 'QOS memory/wall limits need separate verification'
 disk=checked(['df','-B1',str(root)])
 available=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize
 assert available>=213513415862,'The unchanged storage bound is no longer met'
 queue=checked(['squeue','-h','-j',','.join([auth['prepare_job'],*auth['dependent_jobs']]),'-o','%i|%T|%r|%P|%M'])
 accounting=checked(['sacct','-n','-P','-j',','.join([auth['prepare_job'],*auth['dependent_jobs']]),'--format=JobID,State,ExitCode,Elapsed'])
 static={**meta,'status':'pass','remote_utc':now(),'hostname':os.uname().nodename,
         'package_sha256':auth['package_sha256'],'drift':[], 'old_job':original,
         'dependents_before':dependent_before,'partition':partition,'nodes':node_response,
         'eligible_nodes':eligible,'qos':qos,'account_disk':disk,'available_bytes':available,
         'squeue':queue,'sacct':accounting,'correction':'Partition cpu_96G -> cpu_192G only'}
 write('STATIC_RESOURCE_PREFLIGHT.json',static)
 write('RESERVATION.json',{**meta,'remote_utc':now(),'status':'exclusive_pending_job_correction_reserved',
       'job_id':'4069915','old_partition':'cpu_96G','new_partition':'cpu_192G',
       'old_held_audit_sha256':sha(root/'evidence/PREPARE_LAUNCH_AUDIT.json'),
       'unchanged_script_sha256':sha(root/'slurm/prepare.slurm'),
       'static_sha256':sha(audit/'STATIC_RESOURCE_PREFLIGHT.json')})
 # Retain the existing producer ID, so consumers never acquire an invalid dependency.
 stage='hold';r=command(['scontrol','hold','4069915']);commands.append(r);write('HOLD_RESPONSE.json',r)
 assert r['returncode']==0,'Hold rejected; do not update'
 held_old=job('4069915');verify_cpu(held_old,'cpu_96G',held=True);write('HELD_OLD_JOB.json',held_old)
 stage='partition_update';r=command(['scontrol','update','JobId=4069915','Partition=cpu_192G'])
 commands.append(r);write('UPDATE_RESPONSE.json',r);assert r['returncode']==0,'Partition update rejected; leave held for operator review'
 held_new=job('4069915');verify_cpu(held_new,'cpu_192G',held=True)
 dependent_after={j:verify_dependent(j) for j in auth['dependent_jobs']}
 assert all(sha(root/p)==h for p,h in package['files'].items())
 write('CORRECTED_HELD_LAUNCH_AUDIT.json',{**meta,'status':'held_audited','remote_utc':now(),
       'job_id':'4069915','effective_partition':'cpu_192G','unchanged_script_partition_header':'cpu_96G',
       'scheduler_override_authorized':True,'held_job':held_new,'dependents_after':dependent_after,
       'package_sha256':auth['package_sha256'],'unchanged_script_sha256':sha(root/'slurm/prepare.slurm'),
       'static_sha256':sha(audit/'STATIC_RESOURCE_PREFLIGHT.json')})
 stage='release';r=command(['scontrol','release','4069915']);commands.append(r);write('RELEASE_RESPONSE.json',r)
 assert r['returncode']==0,'Release rejected; retain corrected job held'
 released=job('4069915');rf=fields(released['stdout'])
 assert rf['Partition']=='cpu_192G' and rf['Reason'] not in ('JobHeldUser','JobHeldAdmin') and rf['JobState'] in ('PENDING','RUNNING')
 assert rf['NumCPUs']=='8' and rf['MinMemoryNode']=='90G' and rf['TimeLimit']=='1-00:00:00'
 receipt={**meta,'status':'existing_prepare_partition_corrected_and_released','remote_utc':now(),
          'hostname':os.uname().nodename,'remote_audit':str(audit),'prepare_job':'4069915',
          'dependent_jobs':auth['dependent_jobs'],'dependency':'afterok:4069915','dependent_checks':dependent_after,
          'jobs_submitted':[],'jobs_cancelled':[],'old_partition':'cpu_96G','new_partition':'cpu_192G',
          'num_cpus':8,'memory':'90G','walltime':'24:00:00','package_sha256':auth['package_sha256'],
          'job_after_release':released,'eligible_nodes':eligible,'available_bytes':available,
          'evidence':{p.name:json.loads(p.read_text()) for p in audit.glob('*.json')}}
 write('CORRECTION_RECEIPT.json',receipt)
 print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(receipt,sort_keys=True),flags=re.I))
except BaseException as error:
 failure={**meta,'status':'stopped','stage':stage,'remote_utc':now(),'message':str(error),
          'traceback':traceback.format_exc(),'mutation_responses':commands,'remote_audit':str(audit)}
 write('CORRECTION_FAILED.json',failure)
 print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(failure,sort_keys=True),flags=re.I))
'''
    replacements = {'__ROOT__': str(REMOTE), '__SOURCE__': str(SOURCE_CODE),
                    '__AUDIT_NAME__': access.audit.name}
    for key, value in replacements.items():
        code = code.replace(key, repr(value))
    code = code.replace('__AUTH__', repr(authorization))
    compile(code, '<cpu192_pending_correction>', 'exec')
    try:
        access.start_vpn()
        result = json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+m.PYTHON+' -B -u -c '+shlex.quote(code),timeout=420).strip().splitlines()[-1])
        result['local_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit / 'CORRECTION_RECEIPT.json', result)
        print(json.dumps({'audit': str(access.audit), **{k:v for k,v in result.items() if k not in ('evidence','traceback','dependent_checks')}},ensure_ascii=False))
        if result['status']=='stopped':
            raise RuntimeError('Partition correction stopped; see immutable receipt')
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
