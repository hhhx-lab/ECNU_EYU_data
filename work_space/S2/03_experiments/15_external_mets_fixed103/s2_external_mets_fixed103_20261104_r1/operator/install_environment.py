"""Install only the explicitly approved independent ECNU inference environment."""
import json
import os
from pathlib import Path
import shlex
from session import session,access_module

ROOT=Path(__file__).resolve().parents[1]

CODE=r'''
import base64,hashlib,json,os,shutil,subprocess,urllib.request
from datetime import datetime,timezone
from pathlib import Path

root=Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
assert root.name=='s2_external_mets_fixed103_20261104_r1' and not root.exists(),'exclusive remote preparation root required'
free=shutil.disk_usage(root.parent).free
assert free>=100*1024**3,'S0 preparation capacity gate: 100GiB free required; not full-run budget'
root.mkdir(mode=0o700)
for name in ['assets','envs','cache','operator','logs','reports','contract','scratch']:(root/name).mkdir(mode=0o700)
os.environ.update(CONDA_PKGS_DIRS=str(root/'cache/conda'),PIP_CONFIG_FILE='/dev/null',PIP_DISABLE_PIP_VERSION_CHECK='1',
    PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
environment=root/'envs/bio_intel_py310'
meta=dict(artifact_status='experimental_unvalidated',operator_approved=False,formal_gate_status='not_run_not_passed')
def write(name,payload):
    with (root/'operator'/name).open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
started=datetime.now(timezone.utc).isoformat()
write('ENV_INSTALL_STARTED.json',dict(meta,started_utc=started,free_bytes_before=free,environment=str(environment),
    authorized_dependencies=True,preparation_free_gate_bytes=100*1024**3,full_inference_capacity_gate_pass=False))
stage='source_metadata'
try:
    commit='83c73e0cac0b5a3c77a298547040ea3ea7f99854'
    base='https://api.github.com/repos/thekeramik05/MICCAI-METS_challenge_bi-intel'
    def fetch(url):
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'ECNU-pinned-source-preparation'}),timeout=40) as r:return json.load(r)
    tree=fetch(base+'/git/trees/'+commit+'?recursive=1')
    assert tree['truncated'] is False
    selected=[x for x in tree['tree'] if x['type']=='blob' and (
        x['path'].startswith('src/') or x['path'].startswith('trainers/') or
        x['path'] in ['LICENSE','requirements.txt','environment.yaml','Dockerfile','docker/install_trainers.py','docker/entrypoint.sh'])]
    assert len(selected)<=40
    source=root/'assets/bio_source';source.mkdir()
    identities={}
    for item in selected:
        obj=fetch(base+'/contents/'+item['path']+'?ref='+commit)
        b=base64.b64decode(obj['content'])
        gitsha=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        assert gitsha==item['sha']==obj['sha'] and len(b)==obj['size']
        p=source/item['path'];p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(b)
        p.chmod(0o444);identities[item['path']]=dict(bytes=len(b),git_blob_sha=gitsha,sha256=hashlib.sha256(b).hexdigest())
    write('BIO_SOURCE_IDENTITY.json',dict(meta,commit=commit,files=identities))
    stage='conda_create'
    def run(argv,name,timeout=1800,extra=None):
        env=dict(os.environ);env.update(extra or {})
        with (root/'logs'/name).open('x') as log:
            p=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=timeout)
        assert p.returncode==0,'dependency preparation command failed: '+name
    run(['/public/software/apps/miniconda3/bin/conda','create','--prefix',str(environment),'--override-channels','-c','conda-forge',
         '--repodata-fn','current_repodata.json','-y','python=3.10','pip'],'conda_create.log',timeout=900)
    py=str(environment/'bin/python')
    stage='torch_cu124_install'
    run([py,'-B','-m','pip','install','--no-cache-dir','--index-url','https://download.pytorch.org/whl/cu124','torch==2.6.0','torchvision==0.21.0'],
        'torch_cu124_install.log')
    stage='author_dependencies_install'
    run([py,'-B','-m','pip','install','--no-cache-dir','--index-url','https://pypi.org/simple','-r',str(source/'requirements.txt'),'matplotlib==3.10.9'],
        'author_dependencies_install.log',timeout=1200)
    stage='trainer_registration'
    run([py,'-B',str(source/'docker/install_trainers.py')],'trainer_registration.log',timeout=120,
        extra={'TRAINER_SOURCE_DIR':str(source/'trainers')})
    stage='version_verification'
    expected={'torch':'2.6.0','nnunetv2':'2.7.0','monai':'1.5.2','SimpleITK':'2.5.5','nibabel':'5.4.0','scipy':'1.15.2',
        'numpy':'2.2.6','pandas':'2.3.3','timm':'1.0.22','einops':'0.8.2','threadpoolctl':'3.6.0','matplotlib':'3.10.9'}
    code='import importlib.metadata as m,json,torch; print(json.dumps(dict(versions={n:m.version(n) for n in '+repr(list(expected))+'},cuda_build=torch.version.cuda)))'
    observed=json.loads(subprocess.check_output([py,'-B','-c',code],text=True))
    assert all(observed['versions'][n].split('+')[0]==v for n,v in expected.items()) and observed['cuda_build']=='12.4'
    run([py,'-B','-m','pip','check'],'pip_check.log',timeout=60)
    inventory=subprocess.check_output([py,'-B','-m','pip','freeze','--all'],text=True)
    with (root/'reports/BIO_ENVIRONMENT_FREEZE.txt').open('x') as f:f.write(inventory)
    out=dict(meta,status='environment_preparation_pass_not_model_runtime_pass',remote_utc=datetime.now(timezone.utc).isoformat(),
        environment=str(environment),versions=observed,free_bytes_after=shutil.disk_usage(root).free,
        jobs_created=False,model_checkpoint_loaded=False,existing_environment_modified=False)
    write('ENV_INSTALL_COMPLETE.json',out);print(json.dumps(out))
except BaseException as error:
    out=dict(meta,status='environment_preparation_failed',stage=stage,error_type=type(error).__name__,
        message=str(error) if isinstance(error,AssertionError) else 'bounded preparation failed; inspect batch log',
        remote_utc=datetime.now(timezone.utc).isoformat(),no_automatic_retry=True,jobs_created=False)
    write('ENV_INSTALL_FAILED.json',out);print(json.dumps(out))
'''


def main():
    authorization=json.loads((ROOT/'INSTALL_AUTHORIZATION.json').read_text())
    assert authorization['necessary_transitive_dependencies_authorized'] and authorization['author_trainer_registration_authorized']
    with session('install_environment') as access:
        command='env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT='+shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command+=' '+shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])+' -B -c '+shlex.quote(CODE)
        out=access.ssh(command,timeout=3000)
        result=json.loads(out.strip().splitlines()[-1])
        access_module.exclusive(access.audit/'RESULT.json',result)
        print(json.dumps(dict(audit=str(access.audit),status=result['status'])),flush=True)
        if result['status']=='environment_preparation_failed':raise RuntimeError('Independent environment preparation failed; no automatic retry')


if __name__=='__main__':main()
