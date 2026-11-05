"""Independently join the completed R5 runtime with immutable full103 metadata."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex

from install_runtime_r4 import DIRECT
from session import session, access_module


def validate_result(result, remote):
    assert result['status'] == 'runtime_R5_dependencies_imports_trainers_pass_not_model_or_GPU_pass'
    assert result['environment'] == remote + '/envs/bio_intel_py310_r5'
    assert result['versions'] == dict(versions=DIRECT, cuda_build='12.4')
    assert result['author_revision'] == '83c73e0cac0b5a3c77a298547040ea3ea7f99854'
    assert result['plan_sha256'] == 'f310957e4a49ad06fca73740590da52b4892b0bb9a2765cc719052614783c1da'
    assert result['models_loaded'] is False and result['jobs_created'] is False
    assert result['images_read'] is False and result['system_compiler_installed'] is False
    assert result['artifact_status'] == 'experimental_unvalidated'
    assert result['operator_approved'] is False and result['formal_gate_status'] == 'not_run_not_passed'


CODE = r'''
import hashlib,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
root=Path(REMOTE);audit=root/'operator/environment_r5';prefix=Path(EXPECTED['environment'])
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
assert not (audit/'FAILED.json').exists(),'R5 conflicting failure'
assert sha(audit/'COMPLETE.json')==COMPLETE_SHA,'R5 completion transport drift'
assert json.loads((audit/'COMPLETE.json').read_text())==EXPECTED,'R5 completion drift'
for name,key in [('ENVIRONMENT_FREEZE.txt','freeze_sha256'),('PIP_INSTALL_ACTUAL.json','actual_install_report_sha256')]:
 assert sha(audit/name)==EXPECTED[key],'accepted runtime report drift: '+name
assert sha(root/'operator/environment_r4/PIP_INSTALL_PLAN.json')==EXPECTED['plan_sha256'],'accepted dependency plan drift'
for name,digest in EXPECTED['prior_failure_guard'].items():assert sha(root/name)==digest,'prior failure drift'
for name,digest in EXPECTED['old_R3_guard'].items():assert sha(root/'envs/bio_intel_py310_r3'/name)==digest,'old environment drift'
for item in EXPECTED['compiled_wheels']:
 p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],'compiled wheel drift'
source=root/'assets/bio_source'
assert json.loads((root/'operator/BIO_SOURCE_IDENTITY.json').read_text())==AUTHOR,'author identity drift'
assert json.loads((root/'operator/asset_materialization/BIO_FULL_MODEL_IDENTITY.json').read_text())==BIO,'model manifest drift'
for name,item in AUTHOR['files'].items():assert sha(source/name)==item['sha256'],'author source drift'
lock=json.loads((audit/'BASE_PACKAGE_LOCK.json').read_text())
base={json.loads(p.read_text())['name']:json.loads(p.read_text()) for p in (prefix/'conda-meta').glob('*.json')}
assert len(base)==len(lock['records']),'exact base inventory drift'
for item in lock['records']:
 actual=base[item['name']]
 assert all(actual[k]==item[k] for k in ('name','version','build','sha256')),'exact base identity drift'
env={k:v for k,v in os.environ.items() if k in ('PATH','USER','LOGNAME','LANG','LC_ALL')}
env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
p=subprocess.run([str(prefix/'bin/python'),'-B','-m','pip','freeze','--all'],capture_output=True,text=True,env=env,timeout=90)
assert p.returncode==0 and p.stdout==(audit/'ENVIRONMENT_FREEZE.txt').read_text(),'current dependency freeze drift'
# Use the real SIF and the same absolute read-only prefix, without MRI/model mounts or --nv.
probe="import ast,importlib.metadata as m,json; import torch,torchvision,numpy,pandas,scipy,nibabel,SimpleITK,monai,timm,einops,threadpoolctl,matplotlib,numexpr,cc3d,ndindex,nnunetv2; from nnunetv2.utilities.find_class_by_name import recursive_find_python_class; import nnunetv2.training.nnUNetTrainer as package; from pathlib import Path; source=Path("+repr(str(source))+" ); files=['nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE.py','nnUNetTrainerBraTS_BrainIACWrapper_RCOversample.py']; names=sorted({n.name for file in files for n in ast.walk(ast.parse((source/'trainers'/file).read_text())) if isinstance(n,ast.ClassDef) and n.name.startswith('nnUNetTrainer')}); assert len(names)>=2; found={n:recursive_find_python_class(str(Path(package.__file__).parent),n,package.__name__) for n in names}; assert all(found.values()); print(json.dumps(dict(versions={n:m.version(n) for n in "+repr(list(DIRECT))+"},cuda_build=torch.version.cuda,trainers=sorted(found),models_loaded=False,images_read=False)))"
argv=[BACKEND,'exec','--cleanenv','--containall','--no-home','--no-mount','hostfs,bind-paths','--net','--network','none','--pwd','/tmp',
 '--bind',str(prefix)+':'+str(prefix)+':ro','--bind',str(source)+':'+str(source)+':ro',SIF['path'],str(prefix/'bin/python'),'-B','-c',probe]
env.update(SINGULARITYENV_PYTHONDONTWRITEBYTECODE='1',SINGULARITYENV_PYTHONNOUSERSITE='1',SINGULARITYENV_CUDA_VISIBLE_DEVICES='',
 SINGULARITYENV_OMP_NUM_THREADS='2',SINGULARITYENV_MKL_NUM_THREADS='2')
p=subprocess.run(argv,capture_output=True,text=True,env=env,timeout=180)
assert len(p.stdout)+len(p.stderr)<65536,'runtime import output budget'
assert p.returncode==0,'R5 isolated CPU imports failed: '+p.stderr[-3000:]
observed=json.loads(p.stdout.strip().splitlines()[-1])
assert observed['versions']==DIRECT and observed['cuda_build']=='12.4','R5 isolated versions drift'
interpreter=(prefix/'bin/python').resolve()
assert interpreter.is_relative_to(prefix.resolve()),'R5 interpreter escapes environment'
files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in
 [audit/'COMPLETE.json',audit/'ENVIRONMENT_FREEZE.txt',audit/'PIP_INSTALL_ACTUAL.json',audit/'BASE_PACKAGE_LOCK.json',interpreter]]
print(json.dumps(dict(status='independent_R5_environment_acceptance_not_GPU_or_quality_pass',
 accepted_bio_runtime=EXPECTED,complete_sha256=COMPLETE_SHA,base_records=len(base),isolated_CPU_imports=observed,files=files,
 remote_utc=datetime.now(timezone.utc).isoformat(),remote_writes=False,models_loaded=False,images_read=False,jobs_created=False)))
'''


def code_for(result, complete_sha, metadata, remote):
    values = dict(EXPECTED=result, COMPLETE_SHA=complete_sha, DIRECT=DIRECT, REMOTE=remote,
                  AUTHOR=metadata['author_identity'], BIO=metadata['bio_identity'],
                  BACKEND=metadata['backend'], SIF=metadata['sif'])
    code = '\n'.join(k + '=' + repr(v) for k, v in values.items()) + '\n' + CODE
    compile(code, 'independent_R5_acceptance', 'exec')
    return code


def main():
    parser = argparse.ArgumentParser()
    evidence = parser.add_mutually_exclusive_group(required=True)
    evidence.add_argument('--installation', type=Path)
    evidence.add_argument('--snapshot', type=Path)
    parser.add_argument('--metadata', type=Path, required=True)
    args = parser.parse_args()
    if args.snapshot:
        marker = json.loads(args.snapshot.read_text())['runtime_r5']['COMPLETE.json']
        result, complete_sha = marker['data'], marker['sha256']
    else:
        result, complete_sha = json.loads(args.installation.read_text()), access_module.sha(args.installation)
    metadata = json.loads(args.metadata.read_text())
    remote = os.environ['EXTERNAL_MET_REMOTE_ROOT']
    validate_result(result, remote)
    code = code_for(result, complete_sha, metadata, remote)
    with session('accept_runtime_r5') as access:
        access_module.exclusive(access.audit/'EXECUTED_SOURCE.json', dict(source=code,sha256=hashlib.sha256(code.encode()).hexdigest()))
        origin = args.snapshot or args.installation
        access_module.exclusive(access.audit/'EVIDENCE_ORIGIN.json', dict(path=str(origin.resolve()),sha256=access_module.sha(origin),
            remote_complete_sha256=complete_sha,installation_repeated=False))
        command = 'env LC_ALL=C PYTHONDONTWRITEBYTECODE=1 ' + shlex.quote(os.environ['ECNU_REMOTE_PYTHON']) + ' -B -c ' + shlex.quote(code)
        accepted = json.loads(access.ssh(command, timeout=420).strip().splitlines()[-1])
        assert accepted['accepted_bio_runtime'] == result, 'runtime response drift'
        access_module.exclusive(access.audit/'RESULT.json', accepted)
        metadata.update(accepted_bio_runtime=result, independent_runtime_acceptance=accepted)
        access_module.exclusive(access.audit/'ACCEPTED_METADATA.json', metadata)
        print(json.dumps(dict(audit=str(access.audit),status=accepted['status'],remote_utc=accepted['remote_utc'])),flush=True)


if __name__ == '__main__':
    main()
