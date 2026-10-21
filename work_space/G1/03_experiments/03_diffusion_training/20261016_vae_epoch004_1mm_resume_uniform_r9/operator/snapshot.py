"""One read-only R9 job snapshot; no polling, retries, or submission."""
import datetime
import json
import shlex
from common import REMOTE, m, transport


def main():
    access = transport('snapshot')
    code = r'''
import datetime,hashlib,json,os,pathlib,re,subprocess
root=pathlib.Path(__ROOT__)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cmd(a):
 r=subprocess.run(a,capture_output=True,text=True,timeout=25)
 return {'returncode':r.returncode,'stdout':r.stdout[-16000:],'stderr':r.stderr[-2000:]}
def report(p):
 if not p.is_file():return {'exists':False}
 v=json.loads(p.read_text())
 return {'exists':True,'sha256':sha(p),**{k:v[k] for k in ('status','utc','cases','cases_ok','cases_processed','split_counts','global_step','support_contract','loss_policy','label_resampling_algorithm','STRICT_ORIGINAL_GATE','NONZERO_COMPARISON_GATE','independent_full_revalidation','safe_boundary','message') if k in v},'warning_cases':len(v.get('warnings',v.get('resolution_warnings',[]))),'error_cases':len(v.get('errors',[]))}
def log(p):
 if not p.is_file():return {'exists':False}
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-10000));s=f.read().decode(errors='replace')
 return {'exists':True,'bytes':p.stat().st_size,'tail':s[-3000:],
 'critical':re.findall(r'^.*(?:Traceback|CUDA error|OutOfMemory|ImportError|No space left|nonfinite|RuntimeError|DUE TO TIME LIMIT|FloatingPointError|AssertionError|unbound variable).*$',s,re.I|re.M)[-8:]}
receipt=json.loads((root/'evidence/SUBMISSION_RECEIPT.json').read_text());jobs=receipt['jobs']
package=json.loads((root/'plan/PACKAGE_MANIFEST.json').read_text())
v={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,'jobs':jobs,
 'squeue':cmd(['squeue','-h','-j',','.join(jobs.values()),'-o','%i|%T|%j|%P|%r|%b|%M']),
 'sacct':cmd(['sacct','-n','-P','-j',','.join(jobs.values()),'--format=JobID,State,ExitCode,Elapsed,Timelimit,AllocTRES']),
 'account_disk':cmd(['df','-B1',str(root)]),'public_disk':cmd(['df','-h','/public']),
 'available_bytes':os.statvfs(root).f_bavail*os.statvfs(root).f_frsize,'package_sha256':sha(root/'plan/PACKAGE_MANIFEST.json'),
 'drift':[n for n,h in package['files'].items() if not (root/n).is_file() or sha(root/n)!=h],
 'evidence':{str(p.relative_to(root)):report(p) for p in [root/'dataset/STARTED.json',root/'dataset/FAILED.json',root/'dataset/PREPROCESS_REPORT.json',root/'dataset/COMPLETE.ok',
 *(root/'training'/m/n for m in ('t1c','t1n','t2w','t2f') for n in ('STARTED.json','RUN_CONTRACT.json','FAILED.json','SAFE_TERM.json','LATEST_CHECKPOINT.json','COMPLETE.ok'))]},
 'logs':{p.name:log(p) for p in (root/'logs').glob('*') if p.is_file()},
 'training_progress':{m:log(root/'training'/m/'progress.jsonl') for m in ('t1c','t1n','t2w','t2f')},
 'prepared_case_directories':len(list((root/'dataset/cases').glob('BraTS-*'))) if (root/'dataset/cases').exists() else 0,
 'STRICT_ORIGINAL_GATE':False,'NONZERO_COMPARISON_GATE':False,'independent_full_revalidation':'waived_by_user_not_pass'}
print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(v),flags=re.I))
'''.replace('__ROOT__', repr(str(REMOTE)))
    compile(code, 'r9_snapshot', 'exec')
    try:
        access.start_vpn()
        out = access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -c ' + shlex.quote(code), timeout=200)
        value = json.loads(out.strip().splitlines()[-1])
        value.update(m.META)
        value['local_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit / 'SNAPSHOT.json', value)
        print(json.dumps({'audit': str(access.audit), **value}, ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
