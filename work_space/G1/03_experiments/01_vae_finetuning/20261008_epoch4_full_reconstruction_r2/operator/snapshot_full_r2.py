"""Take one bounded, read-only status snapshot of the submitted r2 job."""

import json
import pathlib
import runpy
import shlex
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
submission = runpy.run_path(str(ROOT / "operator/submit_full_r2.py"))
m = submission["m"]
REMOTE = submission["REMOTE"]
JOB_ID = "3941042"


def main():
    access = m.Access("readonly_full_r2")
    code = r'''
import datetime,hashlib,json,os,pathlib,re,subprocess
root=pathlib.Path(ROOT)
job=JOB
def command(args):
    result=subprocess.run(args,text=True,capture_output=True)
    return {'code':result.returncode,'stdout':result.stdout[-10000:],'stderr':result.stderr[-3000:]}
def json_file(path):
    if not path.exists(): return {'exists':False}
    try: return {'exists':True,'content':json.loads(path.read_text())}
    except Exception as error: return {'exists':True,'parse_error':type(error).__name__}
def log(path):
    if not path.exists(): return {'exists':False}
    content=path.read_text(errors='replace')
    errors=re.findall(r'^.*(?:Traceback|OutOfMemory|OOM|CUDA error|ImportError|nonfinite|No space left|FAILED|RuntimeError|unbound variable).*$',content,re.I|re.M)
    return {'exists':True,'bytes':path.stat().st_size,'tail':content[-7000:],'critical':errors[-12:]}
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()
def evidence(path):
    if not path.exists(): return {'exists':False}
    payload=json.loads(path.read_text())
    return {'exists':True,'sha256':sha(path),'status':payload.get('status'),'job_id':payload.get('job_id')}
cases=root/'dataset/cases'
paths={'plan':root/'RECONSTRUCTION_RECOVERY_PLAN.json','slurm':root/'slurm/run_full_reconstruction_r2.slurm','runtime':root/'runtime/reconstruct_epoch004.py'}
payload={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'hostname':os.uname().nodename,'disk':command(['df','-h','/public']),
         'squeue':command(['squeue','-h','-j',job,'-o','%i|%T|%j|%b']),
         'sacct':command(['sacct','-n','-P','-j',job,'--format=JobID,State,ExitCode,Start,End,Elapsed,AllocTRES']),
         'files':{name:json_file(root/'dataset'/name) for name in ('RUN_CONTRACT.json','DATASET_VALIDATION.json','COMPLETE.ok','FAILED.json')},
         'preflight':evidence(root/'evidence/REMOTE_STATIC_PREFLIGHT.json'),
         'launch':evidence(root/'evidence/FULL_LAUNCH_AUDIT.json'),
         'case_dirs':sum(path.is_dir() and not path.name.startswith('.') for path in cases.iterdir()) if cases.exists() else 0,
         'logs':{name:log(root/'logs'/name) for name in (f'g1vae4fullr2_{job}.out',f'g1vae4fullr2_{job}.err')},
         'sha256':{name:sha(path) if path.is_file() else None for name,path in paths.items()}}
# The SSH expect adapter treats any later "password:" substring as a second
# authentication challenge. Keep archived audit text out of the response.
rendered=json.dumps(payload,sort_keys=True)
print(re.sub('password:', '[AUTH_PROMPT]:', rendered, flags=re.I))
'''.replace("ROOT", repr(str(REMOTE))).replace("JOB", repr(JOB_ID))
    owns_vpn = False
    try:
        result = subprocess.run([m.VPN, "start"], env=access.env, text=True,
                                capture_output=True, timeout=100)
        if result.returncode:
            errors = [line for line in (result.stdout + "\n" + result.stderr).splitlines()
                      if line.startswith("ERROR:")]
            raise RuntimeError(f"VPN start failed ({result.returncode}): {errors[-2:]}")
        owns_vpn = "OpenConnect is already running" not in result.stdout
        output = access.ssh("env PYTHONDONTWRITEBYTECODE=1 " + m.PYTHON
                            + " -B -c " + shlex.quote(code), timeout=150)
        data = json.loads(output.strip().splitlines()[-1])
        data.update(m.META)
        m.exclusive(access.audit / "FULL_STATUS.json", data)
        print(json.dumps({
            "audit": str(access.audit), "remote_utc": data["remote_utc"],
            "squeue": data["squeue"], "sacct": data["sacct"],
            "case_dirs": data["case_dirs"],
            "files": {name: record["exists"] for name, record in data["files"].items()},
            "critical": {name: record.get("critical", []) for name, record in data["logs"].items()},
            "sha256": data["sha256"],
        }, ensure_ascii=False))
    finally:
        if owns_vpn:
            stop = subprocess.run([m.VPN, "stop"], env=access.env,
                                  text=True, capture_output=True, timeout=40)
            print("VPN_STOP_RC=" + str(stop.returncode))
            if stop.returncode:
                raise RuntimeError("could not confirm shutdown of this operation's VPN")
        else:
            print("NO_OWNED_VPN_TO_STOP")


if __name__ == "__main__":
    main()
