"""Manifest-first 2025 output download and read-only K1/evaluator reconciliation."""
import ast
import json
import os
from pathlib import Path
import shlex
import shutil
from session import session, access_module

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[4]
K1 = PROJECT / 'work_space/S2/03_experiments/12_k1245_rc_aware/s2_met_aug_k1_selected103_official_met_20261008_r1'
ACCEPT = ROOT / 'operator/accept_2025_full103_completion_20261106T073525Z'

CODE = r'''
import hashlib,importlib.metadata as m,json,os,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def command(argv):
 p=subprocess.run(argv,capture_output=True,text=True,timeout=60)
 assert p.returncode==0,(argv,p.returncode,p.stderr)
 assert len(p.stdout)+len(p.stderr)<262144,'command output budget'
 return p.stdout
base=Path(REMOTE_ROOT)/'attempts/full103_2025_r1'
k1=Path(REMOTE_ROOT).parent/'s2_met_aug_k1_selected103_official_met_20261008_r1'
reports=base/'reports/EXT2025_MET_GLI'
assert sha(reports/'COMPLETE.json')==COMPLETE_SHA,'2025 COMPLETE drift'
assert sha(reports/'PREDICTION_SHA256.json')==MANIFEST_SHA,'2025 manifest drift'
complete=json.loads((reports/'COMPLETE.json').read_text())
accounting=command(['sacct','-n','-P','-j','4310575','--format=JobID,State,ExitCode,Elapsed,NodeList'])
records=[r.split('|') for r in accounting.splitlines() if r]
assert all(any(r[0]==identity and r[1:3]==['COMPLETED','0:0'] for r in records) for identity in ('4310575','4310575.batch')),'2025 successful accounting missing'
pred=base/'predictions/EXT2025_MET_GLI'
assert {p.name for p in pred.glob('*.nii.gz')}==set(EXPECTED2025),'2025 case identity drift'
for name,e in EXPECTED2025.items():
 p=pred/name;assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256'],'2025 output drift: '+name
assert sha(k1/'cohort/fixed103/models/K1/PREDICTION_SHA256.json')==K1_MANIFEST_SHA,'K1 manifest drift'
assert sha(k1/'cohort/fixed103/models/K1/evaluation/leaderboard_metrics.csv')==K1_CSV_SHA,'K1 historical CSV drift'
assert sha(k1/'manifests/FIXED103_REAL_T2W_CASES.csv')==FIXED_CSV_SHA,'K1 cohort drift'
assert sha(k1/'manifests/INPUT_REFERENCE_SHA256.json')==REFERENCE_MANIFEST_SHA,'K1 reference contract drift'
kpred=k1/'cohort/fixed103/models/K1/prediction'
km=json.loads((k1/'cohort/fixed103/models/K1/PREDICTION_SHA256.json').read_text())['files']
assert set(km)==set(EXPECTED2025) and {p.name for p in kpred.glob('*.nii.gz')}==set(km),'K1 prediction case drift'
files={}
for name,e in km.items():
 p=kpred/name;assert sha(p)==e,'K1 prediction drift: '+name
 files[name]=dict(path=str(p),bytes=p.stat().st_size,sha256=e)
refs={}
for p,e in REFERENCE_FILES.items():
 p=Path(p);assert p.name in EXPECTED2025 and sha(p)==e,'reference SHA drift: '+str(p)
 refs[p.name]=dict(path=str(p),bytes=p.stat().st_size,sha256=e)
assert len(refs)==103
env=Path('/public/home/zqchen/.conda/envs/brats_eval')
probe=r"""
import hashlib,importlib.metadata as m,json,sys
from pathlib import Path
import brats_evaluation,panoptica,numpy
sources={}
for name,module in [('brats_evaluation',brats_evaluation),('panoptica',panoptica)]:
 root=Path(module.__file__).parent
 for p in sorted(root.rglob('*')):
  if p.is_file() and p.suffix in ('.py','.yaml','.yml'):sources[name+'/'+str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps(dict(python=sys.version.split()[0],versions={n:m.version(n) for n in ('BraTS-evaluation','panoptica','numpy')},source_sha256=sources)))
"""
environment=json.loads(command([str(env/'bin/python'),'-B','-c',probe]))
assert environment['python']=='3.10.18' and environment['versions']=={'BraTS-evaluation':'0.0.8','panoptica':'2.1.0','numpy':'1.26.4'},'evaluator version drift'
bins={n:sha(env/'bin'/n) for n in ('python','brats-evaluate','brats-parse-metrics')}
assert bins==EVALUATOR_BIN_SHA,'evaluator binary drift'
print(json.dumps(dict(status='comparison_inputs_and_environment_identity_pass',complete2025=complete,
 accounting2025=accounting,prediction2025=dict(path=str(pred),files=EXPECTED2025),K1_predictions=files,
 references=refs,evaluator=environment,evaluator_bin_sha256=bins,remote_utc=datetime.now(timezone.utc).isoformat(),
 free_bytes=shutil.disk_usage(Path(REMOTE_ROOT)).free,quota_stdout=command(['quota','-s']),
 Bio_queue=command(['squeue','-h','-u','zqchen','-n','s2ext26full','-o','%i|%T|%M|%R']),
 remote_writes=False,raw_MRI_read=False,locked_test_read=False)))
'''


def main():
    manifest = json.loads((ACCEPT / 'PREDICTION_SHA256.json').read_text())
    assert access_module.sha(ACCEPT / 'PREDICTION_SHA256.json') == 'c0ffe07f452aa911d7e1b17128add3db25c443ddfb64fe45e7dac2bb84a34717'
    refs = json.loads((K1 / 'manifests/INPUT_REFERENCE_SHA256.json').read_text())['files']
    refs = {p:e for p,e in refs.items() if '/reference/' in p}
    values = dict(REMOTE_ROOT=os.environ['EXTERNAL_MET_REMOTE_ROOT'], EXPECTED2025=manifest['files'], REFERENCE_FILES=refs,
        COMPLETE_SHA='fb26a4ab47a8922fae89a78c62b73583ff959d982c2733f6e7277e14743228ff',
        MANIFEST_SHA='c0ffe07f452aa911d7e1b17128add3db25c443ddfb64fe45e7dac2bb84a34717',
        K1_MANIFEST_SHA='933d513156f5f1193d9c5eadbc2ffdf33be2ca97c138ce4f47f7726869082701',
        K1_CSV_SHA='62e25bd3c6402bc8ad2d25ab9a5c7c0d71ba4b06a80a20319dafa76c7eaf65a8',
        FIXED_CSV_SHA='86ba853dadc64ffc45357895ba0bd543bf5b6b14c0c83eb789f1e875d4aa2e7e',
        REFERENCE_MANIFEST_SHA='5f52bfdc3a957d848063850460226a7c3d9c25f0926e827f728f21e525fbd9df',
        EVALUATOR_BIN_SHA=dict(python='74c3c4930e54299fc01a41f7797ec316f11ccd02f5c8fda622e68753ff901d59',
            **{'brats-evaluate':'5a8245b52f63bb18b4e82d8cafd51f6cf2ac484b039094fb918c0aef86dda71f',
               'brats-parse-metrics':'5958f9c2338ff945b8aa49343e39fc531e87f05173ea012a272cbd94ff7c8a49'}))
    code = '\n'.join(k+'='+repr(v) for k,v in values.items()) + '\n' + CODE
    ast.parse(code, feature_version=(3,10))
    with session('collect_comparison_inputs_r2') as access:
        access_module.exclusive(access.audit / 'EXECUTED_SOURCE.json', dict(source=code))
        result = json.loads(access.ssh('env LC_ALL=C PYTHONDONTWRITEBYTECODE=1 ' + shlex.quote(os.environ['ECNU_REMOTE_PYTHON']) + ' -B -c ' + shlex.quote(code),timeout=240).strip().splitlines()[-1])
        access_module.exclusive(access.audit / 'RESULT.json', dict(access_module.META,**result))
        destination = ROOT / 'results' / ('2025_full103_download_' + access.audit.name.rsplit('_',1)[1])
        assert shutil.disk_usage(ROOT).free > sum(e['bytes'] for e in manifest['files'].values()) + 1024**3
        destination.mkdir(parents=True, exist_ok=False)
        remote = result['prediction2025']['path']
        access.call(['scp','-r','-P',access.env['HPC_LOGIN_PORT'],'-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',
            access.env['HPC_SSH_USER']+'@'+access.env['HPC_LOGIN_HOST']+':'+remote,str(destination / 'prediction')],timeout=240)
        local = destination / 'prediction'
        assert {p.name for p in local.glob('*.nii.gz')} == set(manifest['files'])
        for name, e in manifest['files'].items():
            p = local / name
            assert p.stat().st_size == e['bytes'] and access_module.sha(p) == e['sha256'], 'download SHA mismatch: ' + name
        acceptance = dict(access_module.META,status='2025_103_prediction_download_size_SHA_pass',cases=103,
            files=manifest['files'],remote_source=remote,local_prediction_root=str(local),manifest_sha256=values['MANIFEST_SHA'],
            remote_utc=result['remote_utc'],local_received_utc=access_module.datetime.now(access_module.timezone.utc).isoformat(),
            total_bytes=sum(e['bytes'] for e in manifest['files'].values()),models_or_raw_MRI_downloaded=False,scoring_run=False)
        access_module.exclusive(destination / 'DOWNLOAD_ACCEPTANCE.json', acceptance)
        access_module.exclusive(access.audit / 'DOWNLOAD_ACCEPTANCE.json', acceptance)
        print(json.dumps(dict(audit=str(access.audit),download=str(destination),status=acceptance['status'],Bio_queue=result['Bio_queue'])),flush=True)


if __name__ == '__main__': main()
