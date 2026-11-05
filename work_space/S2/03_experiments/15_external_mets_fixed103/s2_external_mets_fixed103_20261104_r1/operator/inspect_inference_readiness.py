"""Bounded readiness inventory; does not install, load models, or submit jobs."""
import hashlib
import json
import os
import shlex

import submit_sif_recovery_r2
from session import session, access_module


DETAILS = r'''
source_root=root/'assets/bio_source'
identity_path=root/'operator/BIO_SOURCE_IDENTITY.json'
require(identity_path.stat().st_size<262144,'author identity metadata budget')
identity=json.loads(identity_path.read_text())
require(identity['commit']=='83c73e0cac0b5a3c77a298547040ea3ea7f99854','author revision drift')
source_files={}
interfaces={}
for name,item in identity['files'].items():
    path=source_root/name
    require(path.resolve().is_relative_to(source_root.resolve()),'author source path escape')
    require(sha(path)==item['sha256'],'author source identity drift: '+name)
    source_files[name]=dict(bytes=path.stat().st_size,sha256=item['sha256'])
    if path.name=='requirements.txt' or any(word in path.name.lower() for word in ('predict','inference','install_trainers')):
        require(path.stat().st_size<65536,'author interface per-file budget')
        interfaces[name]=path.read_text()
require(sum(len(v.encode()) for v in interfaces.values())<196608,'author interface total budget')
versions_code="import importlib.metadata as m,json; out={};\nfor n in ['torch','torchvision','pandas','numpy','numexpr','connected-components-3d','ndindex','nnunetv2','monai','SimpleITK','nibabel','scipy','timm','einops','threadpoolctl','matplotlib','pillow']:\n try: out[n]=m.version(n)\n except m.PackageNotFoundError: out[n]=None\nprint(json.dumps(out))"
old_environment=root/'envs/bio_intel_py310_r3'
p=subprocess.run([str(old_environment/'bin/python'),'-B','-c',versions_code],capture_output=True,text=True,timeout=40)
require(p.returncode==0,'R3 installed-package inventory failed')
require(len(p.stdout)<16384,'environment metadata budget')
installed=json.loads(p.stdout)
assets_path=root/'operator/ASSETS_MATERIALIZED.json'
assets_metadata=None
if assets_path.is_file():
    require(assets_path.stat().st_size<262144,'asset manifest metadata budget')
    assets_metadata=json.loads(assets_path.read_text())
readiness=dict(META,status='read_only_inference_readiness_not_submission',preparation=result,
    author_identity_sha256=sha(identity_path),author_revision=identity['commit'],author_files=source_files,
    environment_R3_versions=installed,environment_R4_exists=(root/'envs/bio_intel_py310_r4').exists(),
    environment_R4_audit_exists=(root/'operator/environment_r4').exists(),
    asset_metadata=assets_metadata,root_inventory=[p.name for p in sorted(root.iterdir())],
    remote_utc=datetime.now(timezone.utc).isoformat(),remote_writes=False,jobs_created=False,
    models_loaded=False,images_read=False,author_interface_text_not_publication_evidence=True)
print(json.dumps(dict(metadata=readiness,author_interfaces=interfaces)))
'''


def remote_code():
    code = submit_sif_recovery_r2.snapshot_code('4309579') + '\n' + DETAILS
    compile(code, 'read_only_inference_readiness', 'exec')
    return code


def main():
    code = remote_code()
    with session('inspect_inference_readiness') as access:
        access_module.exclusive(access.audit / 'EXECUTED_SOURCE.json',
                                dict(source=code, sha256=hashlib.sha256(code.encode()).hexdigest()))
        command = 'env PYTHONDONTWRITEBYTECODE=1 EXTERNAL_MET_REMOTE_ROOT=' + shlex.quote(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
        command += ' ' + shlex.quote(os.environ['ECNU_REMOTE_PYTHON']) + ' -B -c ' + shlex.quote(code)
        response = json.loads(access.ssh(command, timeout=600).strip().splitlines()[-1])
        access_module.exclusive(access.audit / 'RESULT.json', response['metadata'])
        metadata = response['metadata']
        print(json.dumps(dict(audit=str(access.audit), status=metadata['status'],
                              environment_R3_versions=metadata['environment_R3_versions'],
                              environment_R4_exists=metadata['environment_R4_exists'],
                              environment_R4_audit_exists=metadata['environment_R4_audit_exists'],
                              author_files=metadata['author_files'], root_inventory=metadata['root_inventory'],
                              remote_utc=metadata['remote_utc'],
                              author_interfaces=response['author_interfaces'])), flush=True)


if __name__ == '__main__':
    main()
