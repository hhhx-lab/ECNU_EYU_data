"""One read-only immediate scan snapshot, no submission or repeated checking."""
import datetime
import json
import shlex
from common import ROOT, REMOTE, m, transport


def main():
    access=transport('snapshot_scan')
    code=r'''
import datetime,hashlib,json,os,pathlib,re,subprocess
root=pathlib.Path(__ROOT__)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cmd(a):
 r=subprocess.run(a,capture_output=True,text=True,timeout=25)
 return {'returncode':r.returncode,'stdout':r.stdout[-16000:],'stderr':r.stderr[-2000:]}
def report(p):
 if not p.is_file():return {'exists':False}
 value=json.loads(p.read_text())
 return {'exists':True,'sha256':sha(p),**{k:v for k,v in value.items() if k not in ('case_report_sha256',)}}
def log(p):
 if not p.is_file():return {'exists':False}
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-10000));s=f.read().decode(errors='replace')
 return {'exists':True,'bytes':p.stat().st_size,'tail':s[-3500:],
  'critical':re.findall(r'^.*(?:Traceback|ImportError|OutOfMemory|No space left|nonfinite|DUE TO TIME LIMIT|RuntimeError|AssertionError|unbound variable).*$',s,re.I|re.M)[-8:]}
canonical=json.loads((root/'evidence/CANONICAL_JOB.json').read_text());job=canonical['job_id']
package=json.loads((root/'plan/PACKAGE_MANIFEST.json').read_text())
cfg=json.loads((root/'config/SCAN.json').read_text())
data={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,'job_id':job,
 'squeue':cmd(['squeue','-h','-j',job,'-o','%i|%T|%j|%P|%r|%M']),
 'sacct':cmd(['sacct','-n','-P','-j',job,'--format=JobID,State,ExitCode,Elapsed,Timelimit,AllocTRES']),
 'job_detail':cmd(['scontrol','show','job',job,'-o']),
 'account_disk':cmd(['df','-B1',str(root)]),'public_disk':cmd(['df','-h','/public']),
 'available_bytes':os.statvfs(root).f_bavail*os.statvfs(root).f_frsize,
 'package_sha256':sha(root/'plan/PACKAGE_MANIFEST.json'),
 'drift':[n for n,v in package['files'].items() if not (root/n).is_file() or sha(root/n)!=v],
 'source_csv_sha256':sha(pathlib.Path(cfg['source_code'])/'data/data_csv.csv'),
 'reports':{n:report(root/'scan'/n) for n in ('STARTED.json','FAILED.json','VALIDATION.json','COMPLETE.ok')},
 'logs':{p.name:log(p) for p in (root/'logs').glob('*') if p.is_file()},
 'case_reports':len(list((root/'scan/cases').glob('*.json'))),
 'training_submitted':False,'test_images_read_by_scan_contract':0}
print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(data,sort_keys=True),flags=re.I))
'''
    code=code.replace('__ROOT__',repr(str(REMOTE)))
    compile(code,'scan_snapshot','exec')
    try:
        access.start_vpn()
        out=access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+m.PYTHON+' -B -c '+shlex.quote(code),timeout=200)
        result=json.loads(out.strip().splitlines()[-1]);result.update(m.META)
        result['local_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit/'SNAPSHOT.json',result)
        print(json.dumps({'audit':str(access.audit),**result},ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__=='__main__':main()
