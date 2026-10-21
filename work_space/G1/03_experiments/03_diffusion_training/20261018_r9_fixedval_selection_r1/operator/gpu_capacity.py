"""One read-only A100 allocation and own-job scheduling diagnosis."""
import datetime
import json
import shlex
from common import m, transport

CODE = r'''
import datetime,json,re,subprocess
def command(argv):
 v=subprocess.run(argv,capture_output=True,text=True,timeout=30)
 return dict(argv=argv,returncode=v.returncode,stdout=v.stdout,stderr=v.stderr)
def fields(s):return dict(re.findall(r'([A-Za-z][A-Za-z0-9_/:]*)=([^\s]+)',s))
part=command(['scontrol','show','partition','a100','-o'])
assert part['returncode']==0
hosts=command(['scontrol','show','hostnames',fields(part['stdout'])['Nodes']])
assert hosts['returncode']==0
nodes=command(['scontrol','show','node',','.join(hosts['stdout'].splitlines()),'-o'])
jobs=['4145278','4145279','4145280','4145281']
result=dict(status='read_only_gpu_capacity',remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 partition=part,nodes=nodes,
 own_jobs={j:command(['scontrol','show','job',j,'-o']) for j in jobs},
 queue=command(['squeue','-h','-j',','.join(jobs),'-o','%i|%T|%j|%P|%r|%M|%b|%C|%m']),
 start_estimate=command(['squeue','--start','-j',','.join(jobs),'-o','%.18i %.9P %.8j %.8u %.2t %.19S %.6D %20Y %R']),
 priorities=command(['sprio','-j',','.join(jobs)]))
print(json.dumps(result))
'''


def main():
    access=transport('gpu_capacity');started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    result=None;started_ok=False
    try:
        access.start_vpn();started_ok=True
        out=access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+m.PYTHON+' -B -c '+shlex.quote(CODE),timeout=240)
        result=json.loads(out.strip().splitlines()[-1])
        result['local_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit/'GPU_CAPACITY.json',result)
        print(json.dumps(dict(audit=str(access.audit),**result)))
    finally:
        if started_ok:access.close_vpn()
        m.exclusive(access.audit/'LOCAL_OPERATION_RECEIPT.json',dict(m.META,started_utc=started,
            local_completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            startup_confirmed=started_ok,vpn_owned=access.owns_vpn if started_ok else None,
            preexisting_vpn_left_unchanged=started_ok and not access.owns_vpn,
            remote_snapshot_created=result is not None,read_only=True))


if __name__=='__main__':main()
