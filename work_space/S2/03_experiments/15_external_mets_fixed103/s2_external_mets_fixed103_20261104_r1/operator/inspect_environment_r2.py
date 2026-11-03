"""One bounded read-only view of the running installer and pinned entry points."""
import json
import os
from pathlib import Path
import shlex
from session import session, access_module

CODE = r'''
from pathlib import Path
import json,hashlib,os,subprocess,shutil
from datetime import datetime,timezone
root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
out=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed',
    remote_utc=datetime.now(timezone.utc).isoformat(),free_bytes=shutil.disk_usage(root).free,markers={},logs={},source_text={})
for name in ['STARTED.json','COMPLETE.json','FAILED.json']:
    p=root/'operator/environment_r2'/name
    out['markers'][name]=json.loads(p.read_text()) if p.exists() else None
for p in sorted((root/'logs/environment_r2').glob('*.log')):
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-4000));tail=f.read().decode(errors='replace')
    out['logs'][p.name]=dict(bytes=p.stat().st_size,tail=tail)
identity=json.loads((root/'operator/BIO_SOURCE_IDENTITY.json').read_text())
out['source_identity']=identity
for name,item in identity['files'].items():
    if name.endswith('.py') and not name.startswith('trainers/'):
        p=root/'assets/bio_source'/name
        assert p.stat().st_size<131072 and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
        out['source_text'][name]=p.read_text()
print(json.dumps(out))
'''


def main():
    with session('inspect_environment_r2') as access:
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(CODE)
        result=json.loads(access.ssh(command,timeout=90).strip().splitlines()[-1])
        source_text=result.pop('source_text')
        access_module.exclusive(access.audit/'RAW_BIO_SOURCE.json',source_text)
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),markers={k:v is not None for k,v in result['markers'].items()},
                              log_sizes={k:v['bytes'] for k,v in result['logs'].items()},remote_utc=result['remote_utc'])))


if __name__=='__main__':main()
