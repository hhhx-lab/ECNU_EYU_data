"""One bounded read-only Slurm resource diagnosis; never modify or submit jobs."""
import datetime
import json
import shlex

from common import REMOTE, SOURCE_CODE, m, transport


def main():
    access = transport('constraints_readonly')
    code = r'''
import datetime,hashlib,json,os,pathlib,re,subprocess
root=pathlib.Path(__ROOT__);source=pathlib.Path(__SOURCE__)
def cmd(args):
 try:
  r=subprocess.run(args,capture_output=True,text=True,timeout=25)
  return {'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr[-3000:]}
 except subprocess.TimeoutExpired:
  return {'exit_code':124,'stdout':'','stderr':'bounded read timed out'}
result={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,
 'job':cmd(['scontrol','show','job','4069915','-o']),
 'partition':cmd(['scontrol','show','partition','cpu_96G','-o']),
 'nodes_sinfo':cmd(['sinfo','-N','-h','-p','cpu_96G','-o','%N|%t|%m|%c|%f|%e']),
 'partitions_summary':cmd(['sinfo','-h','-o','%P|%a|%l|%m|%c|%f']),
 'squeue':cmd(['squeue','-h','-j','4069915,4069916,4069917,4069918,4069919','-o','%i|%T|%r|%P|%M']),
 'sacct':cmd(['sacct','-n','-P','-j','4069915,4069916,4069917,4069918,4069919','--format=JobID,State,ExitCode,Elapsed,ReqMem,ReqCPUS,Timelimit']),
 'qos':cmd(['sacctmgr','-n','-P','show','qos','where','name=user_zqchen,partition_cpu_96G','format=Name,Flags,MaxWall,MaxTRESPerJob,MaxTRESPerNode,MinTRESPerJob,MaxJobsPU,MaxSubmitPU']),
 'public_disk':cmd(['df','-h','/public']),'account_disk':cmd(['df','-B1',str(root)]),
 'available_bytes':os.statvfs(root).f_bavail*os.statvfs(root).f_frsize}
names=sorted(set(line.split('|')[0] for line in result['nodes_sinfo']['stdout'].splitlines() if line))
result['nodes_scontrol']=cmd(['scontrol','show','node',','.join(names),'-o']) if names else {'exit_code':1,'stdout':'','stderr':'no node inventory'}
config=cmd(['scontrol','show','config'])
config['stdout']='\n'.join(line for line in config['stdout'].splitlines() if re.match(r'\s*(SelectType|SelectTypeParameters|DefMemPer|MaxMemPer|SchedulerType|SchedulerParameters|AccountingStorageEnforce)\s*=',line))
result['config_resource_fields']=config
package=json.loads((root/'plan/PACKAGE_MANIFEST.json').read_text())
result['package_sha256']=hashlib.sha256((root/'plan/PACKAGE_MANIFEST.json').read_bytes()).hexdigest()
result['drift']=[p for p,h in package['files'].items() if not (root/p).is_file() or hashlib.sha256((root/p).read_bytes()).hexdigest()!=h]
result['source_csv_sha256']=hashlib.sha256((source/'data/data_csv.csv').read_bytes()).hexdigest()
result['logs']={p.name:{'bytes':p.stat().st_size,'tail':p.read_text(errors='replace')[-3000:]} for p in (root/'logs').glob('*') if p.is_file() and p.stat().st_size<1048576}
result['prepared_case_directories']=len(list((root/'dataset/cases').glob('BraTS-*')))
result['markers']={name:(json.loads((root/'dataset'/name).read_text()) if (root/'dataset'/name).is_file() else None) for name in ('STARTED.json','FAILED.json','COMPLETE.ok')}
print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(result,sort_keys=True),flags=re.I))
'''
    code = code.replace('__ROOT__', repr(str(REMOTE))).replace('__SOURCE__', repr(str(SOURCE_CODE)))
    try:
        access.start_vpn()
        result = json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -c ' + shlex.quote(code), timeout=300).strip().splitlines()[-1])
        result.update(m.META)
        result['local_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit / 'CONSTRAINT_DIAGNOSIS.json', result)
        print(json.dumps({'audit': str(access.audit), 'job': result['job'], 'squeue': result['squeue'],
                          'nodes_sinfo': result['nodes_sinfo'], 'partition': result['partition'],
                          'qos': result['qos'], 'drift': result['drift'], 'local_utc': result['local_utc'],
                          'remote_utc': result['remote_utc']}, ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
