"""One queue snapshot plus one bounded instrumented test, with no deployment/submission."""
import datetime
import json
import shlex
from common import ROOT, REMOTE, PARENT_REMOTE, RECON, m, transport


def main():
    access = transport('diagnose_parent')
    code = r'''
import datetime,hashlib,json,os,pathlib,re,subprocess,time
parent=pathlib.Path(__PARENT__);root=pathlib.Path(__ROOT__);recon=pathlib.Path(__RECON__)
def cmd(argv):
 r=subprocess.run(argv,capture_output=True,text=True,timeout=30)
 return {'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
def evidence(p):
 if not p.exists():return {'exists':False}
 value=json.loads(p.read_text())
 return {'exists':True,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'value':value}
result={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,
 'root_exists':root.exists(),'public_disk':cmd(['df','-h','/public']),'account_disk':cmd(['df','-B1',str(parent)]),
 'available_bytes':os.statvfs(parent).f_bavail*os.statvfs(parent).f_frsize,
 'squeue':cmd(['squeue','-h','-u','zqchen','-o','%i|%T|%j|%P|%r|%b']),
 'sacct':cmd(['sacct','-n','-P','-j','3941042,4058680','--format=JobID,State,ExitCode,Elapsed']),
 'partitions':{p:cmd(['scontrol','show','partition',p]) for p in ('cpu_96G','a100')},
 'parent_failure':evidence(parent/'evidence/OPERATION_FAILED.json'),
 'parent_package':evidence(parent/'plan/PACKAGE_MANIFEST.json'),
 'parent_canonical_jobs':[p.name for p in (parent/'evidence').glob('*CANONICAL*')],
 'reconstruction_complete':evidence(recon/'dataset/COMPLETE.ok'),
 'thread_env_before':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')},
 'cpu_affinity_count':len(os.sched_getaffinity(0))}
env=os.environ.copy()
env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
 PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='')
probe="import faulthandler,time,runpy,sys;faulthandler.enable();faulthandler.dump_traceback_later(90,repeat=False);print('PROBE_START',time.time(),flush=True);sys.argv=["+repr(str(parent/'tests/test_runtime.py'))+",'-v'];runpy.run_path(sys.argv[0],run_name='__main__')"
started=time.monotonic()
try:
 r=subprocess.run([__PYTHON__,'-B','-u','-X','importtime','-c',probe],env=env,capture_output=True,text=True,timeout=300)
 result['probe']={'returncode':r.returncode,'elapsed_seconds':time.monotonic()-started,'stdout':r.stdout[-16000:],'stderr':r.stderr[-30000:],
  'bounded_timeout':False,'thread_policy':'OMP/MKL/OPENBLAS/NUMEXPR=1; CUDA disabled for CPU tests'}
except subprocess.TimeoutExpired as error:
 def decode(v):return v.decode(errors='replace') if isinstance(v,bytes) else v or ''
 result['probe']={'returncode':124,'elapsed_seconds':time.monotonic()-started,'stdout':decode(error.stdout)[-16000:],
  'stderr':decode(error.stderr)[-30000:],'bounded_timeout':True}
print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(result,sort_keys=True),flags=re.I))
'''
    for key, value in {'__PARENT__':str(PARENT_REMOTE), '__ROOT__':str(REMOTE), '__RECON__':str(RECON), '__PYTHON__':m.PYTHON}.items():
        code = code.replace(key, repr(value))
    try:
        access.start_vpn()
        result = json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -u -c ' + shlex.quote(code), timeout=420).strip().splitlines()[-1])
        result.update(m.META)
        result['local_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit / 'DIAGNOSIS.json', result)
        print(json.dumps({'audit':str(access.audit),'remote_utc':result['remote_utc'],'local_utc':result['local_utc'],
                         'available_gib':result['available_bytes']/1024**3,'squeue':result['squeue'],
                         'parent_canonical_jobs':result['parent_canonical_jobs'],'parent_failure':result['parent_failure'],
                         'thread_env_before':result['thread_env_before'],'probe':result['probe']},ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
