"""Extract selected image dataset metadata without downloading source MRI."""
from datetime import datetime,timezone
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
    layer=next(x for x in manifest['layers'] if x['digest']=='sha256:e1fb9995d7ec3c2a1c6d4507b49d1350508f3412daa202fcf183966f6b3a2ef2')
    proxy=os.environ.get('EXTERNAL_MET_LOCAL_HTTP_PROXY','')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':proxy,'https':proxy} if proxy else {}))
    def fetch(url,headers=None):
        with opener.open(urllib.request.Request(url,headers=headers or {}),timeout=40) as r:return r.read(1000000)
    token=json.loads(fetch('https://auth.docker.io/token?service=registry.docker.io&scope=repository:'+IMAGE+':pull'))['token']
    b=fetch('https://registry-1.docker.io/v2/'+IMAGE+'/blobs/'+layer['digest'],{'Authorization':'Bearer '+token})
    assert len(b)==layer['size'] and hashlib.sha256(b).hexdigest()==layer['digest'].split(':')[1]
    with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as t:body=t.extractfile('raw/Dataset101_submission/dataset.json').read()
    dataset=json.loads(body)
    audit=ROOT/'operator'/('inspect_image_dataset_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    audit.mkdir(exist_ok=False)
    result=dict(META,status='image_metadata_not_model_runtime_pass',source_layer_digest=layer['digest'],dataset_bytes=len(body),
        dataset_sha256=hashlib.sha256(body).hexdigest(),dataset_keys=sorted(dataset),
        metadata={k:dataset[k] for k in ['channel_names','labels','numTraining','file_ending','name','description','license'] if k in dataset},
        source_images_read=False,raw_metadata_git_excluded=True)
    for name,value in [('IMAGE_DATASET_AUDIT.json',result),('RAW_IMAGE_DATASET.json',dataset)]:
        with (audit/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(result));print(json.dumps(dict(audit=str(audit))))


if __name__=='__main__':main()
