"""Read only the deployment rejection and capacity fields; never retry submission."""

import json
import shlex

from common import REMOTE, m, transport


def main():
    access = transport('capacity_diagnostic')
    code = '''
import datetime,json,os,pathlib,re,subprocess
root=pathlib.Path(ROOT)
s=os.statvfs(root)
def cmd(args):
    try:
        r=subprocess.run(args,text=True,capture_output=True,timeout=15)
        return {'exit_code':r.returncode,'stdout':r.stdout[-5000:],'stderr':r.stderr[-1500:]}
    except subprocess.TimeoutExpired:return {'error':'bounded_timeout'}
p={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,
   'statvfs':{k:getattr(s,k) for k in ('f_bsize','f_frsize','f_blocks','f_bfree','f_bavail','f_files','f_ffree','f_favail')},
   'available_bytes':s.f_frsize*s.f_bavail,'total_free_bytes':s.f_frsize*s.f_bfree,
   'df_bytes':cmd(['df','-B1',str(root)]),'df_inodes':cmd(['df','-i',str(root)]),
   'operation_failure':json.loads((root/'evidence/P0_OPERATION_FAILED.json').read_text()),
   'canonical_job_exists':(root/'evidence/P0_CANONICAL_JOB.json').exists(),
   'submission_response_exists':(root/'evidence/P0_SUBMISSION_RESPONSE.json').exists(),
   'static_preflight_exists':(root/'evidence/P0_STATIC_PREFLIGHT.json').exists()}
print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(p,sort_keys=True),flags=re.I))
'''.replace('ROOT', repr(str(REMOTE)))
    try:
        access.start_vpn()
        data=json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+m.PYTHON+' -B -c '+shlex.quote(code),timeout=80).strip().splitlines()[-1])
        data.update(m.META)
        m.exclusive(access.audit/'CAPACITY_DIAGNOSTIC.json',data)
        print(json.dumps(data,ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
