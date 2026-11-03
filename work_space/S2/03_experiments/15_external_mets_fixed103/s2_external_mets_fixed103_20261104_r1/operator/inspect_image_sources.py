"""Read only small, digest-verified code layers; no model execution."""
from datetime import datetime,timezone
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import urllib.request
from download_assets import IMAGE,META

ROOT=Path(__file__).resolve().parents[1]


def main():
    temp=Path(os.environ['EXTERNAL_MET_LOCAL_TRANSFER_ROOT'])
    manifest=json.loads((temp/'docker/manifest.json').read_text())
    proxy=os.environ.get('EXTERNAL_MET_LOCAL_HTTP_PROXY','')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':proxy,'https':proxy} if proxy else {}))
    def fetch(url,headers=None):
        with opener.open(urllib.request.Request(url,headers=headers or {}),timeout=40) as r:return r.read(300000)
    token=json.loads(fetch('https://auth.docker.io/token?service=registry.docker.io&scope=repository:'+IMAGE+':pull'))['token']
    audit=ROOT/'operator'/('inspect_image_sources_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    audit.mkdir(exist_ok=False)
    files,raw,layers=[],[],[]
    for layer in manifest['layers']:
        if layer['size']>100000:continue
        b=fetch('https://registry-1.docker.io/v2/'+IMAGE+'/blobs/'+layer['digest'],{'Authorization':'Bearer '+token})
        assert len(b)==layer['size'] and hashlib.sha256(b).hexdigest()==layer['digest'].split(':')[1]
        with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as t:
            members=t.getmembers()
            layers.append(dict(digest=layer['digest'],members=[dict(path=m.name,bytes=m.size,type=m.type.decode()) for m in members]))
            for m in members:
                if not m.isfile() or m.size>500000 or not m.name.endswith(('.py','.json','.txt','.md','.sh','LICENSE')):continue
                body=t.extractfile(m).read();text=body.decode()
                item=dict(path=m.name,layer_digest=layer['digest'],bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
                if m.name.endswith('.py'):
                    tree=ast.parse(text)
                    item['imports']=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
                files.append(item);raw.append(dict(path=m.name,text=text))
                if m.name.endswith('main.py'):print(text)
    for name,value in [('IMAGE_SOURCE_AUDIT.json',dict(META,files=files,layers=layers,model_loaded=False)),('RAW_IMAGE_SOURCE.json',raw)]:
        with (audit/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(dict(audit=str(audit),files=len(files))))


if __name__=='__main__':main()
