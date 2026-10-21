"""One bounded read-only P0 snapshot; never deploy, resubmit or read raw launch logs."""

import datetime
import json
import shlex

from common import REMOTE, ROOT, m, transport


def main():
    receipt_path = ROOT / 'operator/submit_p0_20261013T155644Z/SUBMISSION_RECEIPT.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['job_id'] == '4058680' and receipt['status'] == 'released'
    access = transport('snapshot_p0')
    code = r'''
import datetime,hashlib,json,os,pathlib,re,subprocess
root=pathlib.Path(REMOTE_ROOT)
job='4058680'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def cmd(args):
    r=subprocess.run(args,capture_output=True,text=True,timeout=25)
    return {'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr[-2000:]}
def marker(name):
    p=root/'evidence/p0_native'/name
    if not p.is_file():return {'exists':False}
    value=json.loads(p.read_text())
    return {'exists':True,'sha256':sha(p),'value':value}
def log(suffix):
    p=root/'logs'/('g1d1mmp0r2_'+job+'.'+suffix)
    if not p.is_file():return {'exists':False}
    with p.open('rb') as h:
        h.seek(max(0,p.stat().st_size-24000));tail=h.read().decode(errors='replace')
    return {'exists':True,'bytes':p.stat().st_size,'tail':tail[-12000:],
            'critical':re.findall(r'^.*(?:Traceback|OutOfMemory|CUDA error|ImportError|nonfinite|No space left|FAILED|RuntimeError|unbound variable|AssertionError|ValueError|Killed|oom-kill).*$',tail,re.I|re.M)}
package=json.loads((root/'plan/P0_PACKAGE_MANIFEST.json').read_text())
drift=[p for p,v in package['files'].items() if not (root/p).is_file() or sha(root/p)!=v]
progress=root/'evidence/p0_native/NATIVE_FILE_VALIDATION.jsonl'
rows=sum(1 for _ in progress.open()) if progress.is_file() else 0
disk=os.statvfs(root)
result={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,
        'job_id':job,'squeue':cmd(['squeue','-h','-j',job,'-o','%i|%T|%j|%P|%r|%N']),
        'sacct':cmd(['sacct','-n','-P','-j',job,'--format=JobID,State,ExitCode,Elapsed,Start,End,AllocTRES']),
        'public_disk':cmd(['df','-B1','/public',str(root)]),'available_bytes':disk.f_bavail*disk.f_frsize,
        'markers':{n:marker(n) for n in ('STARTED.json','FAILED.json','VALIDATION.json','COMPLETE.ok')},
        'logs':{s:log(s) for s in ('out','err')},'file_rows_written':rows,
        'package_manifest_sha256':sha(root/'plan/P0_PACKAGE_MANIFEST.json'),'drift':drift,
        'operation_failed_exists':(root/'evidence/P0_OPERATION_FAILED.json').exists()}
print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(result,sort_keys=True),flags=re.I))
'''.replace('REMOTE_ROOT', repr(str(REMOTE)))
    try:
        access.start_vpn()
        output = access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -c ' + shlex.quote(code), timeout=100)
        result = json.loads(output.strip().splitlines()[-1])
        result.update(m.META)
        result['local_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        result['submission_receipt_sha256'] = m.sha(receipt_path)
        m.exclusive(access.audit / 'P0_SNAPSHOT.json', result)
        print(json.dumps({'audit':str(access.audit),'remote_utc':result['remote_utc'],
                          'squeue':result['squeue'],'sacct':result['sacct'],
                          'file_rows_written':result['file_rows_written'],'drift':result['drift'],
                          'markers':result['markers'],'critical':{k:v.get('critical',[]) for k,v in result['logs'].items()}}, ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
