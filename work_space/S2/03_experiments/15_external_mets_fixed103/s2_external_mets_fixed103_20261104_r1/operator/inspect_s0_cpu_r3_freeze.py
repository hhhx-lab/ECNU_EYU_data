"""Read-only diagnosis of the stopped R3 freeze; no retry, install or submission."""
import json
import os
import shlex
from session import session, access_module
from s0_cpu_r3_sources import common_source

CODE = '''
from datetime import datetime,timezone
import os,shutil,subprocess
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
parent=root/'attempts/s0_cpu_prepare_r2'
attempt=root/'attempts/s0_cpu_prepare_r3'
contract=json.loads((parent/'contract/PREPARATION_CONTRACT.json').read_text())
identity=parent_r2_guard(root)
observations={}
for relative in ('reports/SOURCE_BUILD_REQUIREMENTS.json','operator/SOURCE_BUILD_REQUIREMENTS.json'):
    path=parent/relative
    observations[relative]=dict(exists=path.is_file(),sha256=sha(path) if path.is_file() else None,
                               expected_sha256=contract['source_build_requirements_sha256'])
missing={name:not (attempt/'operator'/name).exists() for name in
         ('RESERVATION.json','SBATCH_RESULT.json','HELD_CREATED.json','HELD_AUDIT.json','RELEASE_RESULT.json','SUBMISSION_RECEIPT.json')}
queries={}
for name,argv in {
 'queue':['squeue','-h','-u','zqchen','-n','s2exts0p3','-o','%i|%T|%j|%r'],
 'accounting':['sacct','-P','-u','zqchen','--name=s2exts0p3','--starttime=2026-11-05T00:00:00','--format=JobID,JobName,State,ExitCode'],
 'quota':['quota','-s']}.items():
 p=subprocess.run(argv,capture_output=True,text=True,timeout=30)
 require(len(p.stdout.encode())+len(p.stderr.encode())<65536,'diagnosis output budget')
 queries[name]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
failed=attempt/'operator/SUBMISSION_FAILED.json'
files={str(p.relative_to(attempt)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(attempt.rglob('*')) if p.is_file()}
require(parent_r2_guard(root)==identity,'parent R2 changed during diagnosis')
print(json.dumps(dict(META,status='read_only_R3_freeze_diagnosis_no_retry',observations=observations,
 submission_markers_absent=missing,queries=queries,failed_sha256=sha(failed),failed=json.loads(failed.read_text()),
 R3_partial_files=files,free_bytes=shutil.disk_usage(root).free,remote_utc=datetime.now(timezone.utc).isoformat(),
 jobs_created=False,remote_writes=False,images_read=False,environment_reinstalled=False)))
'''


def main():
    source=common_source()+'\n'+CODE
    compile(source,'readonly_R3_freeze_diagnosis','exec')
    with session('inspect_s0_cpu_r3_freeze') as access:
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(source)
        result=json.loads(access.ssh(command,timeout=180).strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'],remote_utc=result['remote_utc'])))


if __name__=='__main__':
    main()
