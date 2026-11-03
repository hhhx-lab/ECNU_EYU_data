"""Check selected public asset endpoints from ECNU; metadata only."""
import preflight

preflight.CODE = r'''
from datetime import datetime,timezone
import hashlib,json,os,shutil,ssl,urllib.request,urllib.error

urls={
 'bio_weights':'https://huggingface.co/keramik05/MICCAI-METS_challenge_bi-intel_model/resolve/4a8291024dafb79b4c0cc381438e976a36072201/weights.zip',
 'brainiac':'https://www.dropbox.com/scl/fo/i51xt63roognvt7vuslbl/AMblt6reQVvlSrORTB3_2lE/BrainIAC.ckpt?rlkey=9w55le6tslwxlfz6c0viylmjb&dl=1',
 'bio_source':'https://api.github.com/repos/thekeramik05/MICCAI-METS_challenge_bi-intel/contents/environment.yaml?ref=83c73e0cac0b5a3c77a298547040ea3ea7f99854',
 'torch_index':'https://download.pytorch.org/whl/cu124/torch/',
 'conda_repodata':'https://conda.anaconda.org/conda-forge/linux-64/repodata.json',
}
out=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed',
 status='asset_endpoint_metadata_only',remote_started_utc=datetime.now(timezone.utc).isoformat(),sources={})
for name,url in urls.items():
    record=dict(url=url,requested_utc=datetime.now(timezone.utc).isoformat())
    try:
        req=urllib.request.Request(url,method='HEAD',headers={'User-Agent':'ECNU-selected-asset-preflight'})
        with urllib.request.urlopen(req,timeout=35) as response:
            record.update(status=response.status,headers={k:v for k,v in response.headers.items()
              if k.lower() in ['content-length','content-type','etag','x-linked-size','x-linked-etag','x-dropbox-content-length']})
    except Exception as error:
        record.update(status='error',error_type=type(error).__name__,error=str(error)[:500],no_automatic_retry=True)
    out['sources'][name]=record
try:
    url='https://auth.docker.io/token?service=registry.docker.io&scope=repository:brainles/brats25_met_gli-team:pull'
    with urllib.request.urlopen(url,timeout=35) as r:token=json.load(r)['token']
    url='https://registry-1.docker.io/v2/brainles/brats25_met_gli-team/manifests/sha256:f83d914d4d7a9d316f7605abaf03fa6adbcf0e86d840d93276e86ad65b2a2a82'
    req=urllib.request.Request(url,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json'})
    with urllib.request.urlopen(req,timeout=35) as r:b=r.read(1000000)
    out['docker_manifest']=dict(status=200,url=url,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),manifest=json.loads(b))
except Exception as error:
    out['docker_manifest']=dict(status='error',error_type=type(error).__name__,error=str(error)[:500],no_automatic_retry=True)
out['conda_executable']=shutil.which('conda')
out['scope']=dict(remote_writes=False,binary_download=False,dependencies_installed=False,jobs_created=False,tls_verification_disabled=False)
out['remote_utc']=datetime.now(timezone.utc).isoformat()
print(json.dumps(out))
'''

if __name__ == '__main__':
    preflight.main()
