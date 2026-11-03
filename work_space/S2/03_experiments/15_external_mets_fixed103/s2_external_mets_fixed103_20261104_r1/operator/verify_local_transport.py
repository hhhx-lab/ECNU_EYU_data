"""Local preparation tests only; does not validate any external model runtime."""
import ast
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
from download_assets import ROOT,META,write
from download_segmented import sha


def main():
    audit=ROOT/'operator'/('verify_transport_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    audit.mkdir(exist_ok=False)
    source={}
    for path in sorted([*(ROOT/'operator').glob('*.py'),*(ROOT/'runtime').glob('*.py')]):
        ast.parse(path.read_text())
        source[str(path.relative_to(ROOT))]=sha(path)
    with (audit/'UNITTEST.log').open('x') as log:
        result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],
                              stdout=log,stderr=subprocess.STDOUT,timeout=120)
    write(audit/'VERIFICATION.json',dict(META,status='pass' if result.returncode==0 else 'failed',
        local_utc=datetime.now(timezone.utc).isoformat(),python=sys.executable,ast_count=len(source),source_sha256=source,
        test_log_sha256=sha(audit/'UNITTEST.log'),test_source_sha256={p.name:sha(p) for p in (ROOT/'tests').glob('*.py')},
        external_model_runtime_pass=False,A100_pass=False,jobs_created=False))
    print(json.dumps(dict(audit=str(audit),status='pass' if result.returncode==0 else 'failed')))
    if result.returncode:raise RuntimeError('local transport preparation tests failed')


if __name__=='__main__':main()
