"""Audited offline S0 conversion using the server's existing native backend."""
import json
import os
from pathlib import Path
import shlex
from session import session,access_module
from download_assets import ROOT
from download_segmented import sha


def main():
    assert len(list((ROOT/'operator').glob('transfer_assets_*/TRANSFER_ACCEPTANCE.json')))==1
    remote=os.environ['EXTERNAL_MET_REMOTE_ROOT']
    backend='/public/software/singularity-ce/4.1.1/bin/singularity'
    with session('materialize_server_assets') as access:
        py=shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])
        def run(code,timeout=240):
            return json.loads(access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+py+' -B -c '+shlex.quote(code),timeout=timeout).strip().splitlines()[-1])
        check='''from pathlib import Path
import json,shutil
from datetime import datetime,timezone
root=Path('''+repr(remote)+''')
assert json.loads((root/'operator/ASSET_TRANSFER_COMPLETE.json').read_text())['status']=='all_server_asset_bytes_SHA_verified'
free=shutil.disk_usage(root).free
assert free>150*1024**3
assert not (root/'operator/asset_materialization').exists()
(root/'operator/s0_runtime').mkdir(mode=0o700,exist_ok=False)
print(json.dumps(dict(remote_utc=datetime.now(timezone.utc).isoformat(),free_bytes=free,initial_conversion_reservation_bytes=150*1024**3,final_smoke_budget_pass=False)))
'''
        access_module.exclusive(access.audit/'CAPACITY_GATE.json',run(check))
        identities={}
        for name in ['contract.py','materialize_assets.py']:
            source=ROOT/'runtime'/name
            access.scp(source,remote+'/operator/s0_runtime/'+name)
            identities[name]=sha(source)
        validate='''from pathlib import Path
import json,hashlib
root=Path('''+repr(remote)+''')
expected='''+repr(identities)+'''
for name,digest in expected.items():
    p=root/'operator/s0_runtime'/name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
    p.chmod(0o440)
print(json.dumps(dict(status='S0_helper_identity_pass',files=expected)))
'''
        access_module.exclusive(access.audit/'HELPER_IDENTITY.json',run(validate))
        command='env PYTHONDONTWRITEBYTECODE=1 '+py+' -B '+shlex.quote(remote+'/operator/s0_runtime/materialize_assets.py')
        command+=' --root '+shlex.quote(remote)+' --singularity '+shlex.quote(backend)
        command='bash -lc '+shlex.quote('set -e; module load singularity-ce/4.1.1; exec '+command)
        # Subprocess output is retained by the server helper; pull only bounded metadata/log tails.
        try:
            access.ssh(command,timeout=2700)
        finally:
            inspect='''from pathlib import Path
import json
from datetime import datetime,timezone
root=Path('''+repr(remote)+''')
audit=root/'operator/asset_materialization'
out=dict(remote_utc=datetime.now(timezone.utc).isoformat(),markers={})
for name in ['COMPLETE.json','FAILED.json','BIO_FULL_MODEL_IDENTITY.json','OCI_EXPANDED_IDENTITIES.json','DOCKER_ARCHIVE_IDENTITY.json']:
    p=audit/name
    out['markers'][name]=json.loads(p.read_text()) if p.exists() else None
p=audit/'SIF_BUILD.log'
if p.exists():
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-4000));out['sif_log_tail']=f.read().decode(errors='replace')
print(json.dumps(out))
'''
            result=run(inspect)
            access_module.exclusive(access.audit/'RESULT.json',result)
            print(json.dumps(dict(audit=str(access.audit),complete=result['markers']['COMPLETE.json'] is not None,
                                 failed=result['markers']['FAILED.json'] is not None,remote_utc=result['remote_utc'])),flush=True)
        if result['markers']['COMPLETE.json'] is None:
            raise RuntimeError('Offline asset materialization not complete; preserve evidence; no automatic retry')


if __name__=='__main__':main()
