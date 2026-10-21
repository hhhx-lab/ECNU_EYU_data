"""One exclusive P0 deployment, static gate, held launch audit and release."""

import base64
import datetime
import json
import shlex
import subprocess

from common import EXPECTED_PLAN, RECON, REMOTE, ROOT, SOURCE_CODE, m, transport


def main():
    auth = json.loads((ROOT / 'plan/EXECUTION_AUTHORIZATION.json').read_text())
    assert auth['approved_plan_sha256'] == EXPECTED_PLAN
    assert m.sha(ROOT / 'plan/G1_EPOCH004_RECON_1MM_DIFFUSION_ECNU_PLAN_20261013_R1.md') == EXPECTED_PLAN
    test = subprocess.run(['conda', 'run', '--no-capture-output', '-n', 'g1_t2w_bbdm', 'python', '-B', str(ROOT / 'tests/test_native_validator.py')], capture_output=True, text=True, timeout=60)
    if test.returncode:
        raise RuntimeError('local P0 tests failed; no remote submission')
    local_sources = {
        'README.md': ROOT / 'README.md',
        'runtime/validate_native.py': ROOT / 'runtime/validate_native.py',
        'tests/test_native_validator.py': ROOT / 'tests/test_native_validator.py',
        'slurm/validate_native.slurm': ROOT / 'slurm/validate_native.slurm',
        'config/p0.env': ROOT / 'config/p0.env.example',
        'config/p0.env.example': ROOT / 'config/p0.env.example',
        'config/P0_SOURCE_MAPPING.json': ROOT / 'config/P0_SOURCE_MAPPING.json',
    }
    local_sources.update({str(p.relative_to(ROOT)):p for p in (ROOT / 'plan').iterdir() if p.is_file()})
    assert 'plan/P0_PACKAGE_MANIFEST.json' not in local_sources, 'P0 package already frozen; refuse repeat'
    files = {relative:m.sha(source) for relative,source in local_sources.items()}
    m.exclusive(ROOT / 'plan/P0_PACKAGE_MANIFEST.json', {**m.META, 'files':files, 'stage':'P0', 'local_tests':'9 pass'})
    local_sources['plan/P0_PACKAGE_MANIFEST.json'] = ROOT / 'plan/P0_PACKAGE_MANIFEST.json'
    encoded = {relative:base64.b64encode(source.read_bytes()).decode() for relative,source in local_sources.items()}
    access = transport('submit_p0')
    payload_path = access.audit / 'PACKAGE_PAYLOAD.json'
    m.exclusive(payload_path, {'files_b64':encoded})
    m.exclusive(access.audit / 'LOCAL_TESTS.json', {**m.META,'status':'pass','exit_code':test.returncode,'output':test.stdout+test.stderr})
    code = r'''
import ast,base64,collections,csv,datetime,hashlib,importlib.util,json,os,pathlib,re,subprocess,tempfile,py_compile
root=pathlib.Path(REMOTE_ROOT)
recon=pathlib.Path(RECON_ROOT)
source_code=pathlib.Path(SOURCE_CODE_ROOT)
payload_path=root/'evidence/P0_PACKAGE_PAYLOAD.json'
assert hashlib.sha256(payload_path.read_bytes()).hexdigest()==PAYLOAD_SHA
encoded=json.loads(payload_path.read_text())['files_b64']
meta={'artifact_status':'experimental_unvalidated','operator_approved':False,'formal_gate_status':'not_run_not_passed'}
def write(path,value):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o440)
    with os.fdopen(fd,'w') as h: json.dump(value,h,sort_keys=True,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def cmd(args,timeout=30,env=None):
    r=subprocess.run(args,capture_output=True,text=True,timeout=timeout,env=env)
    if r.returncode: raise RuntimeError('command failed: '+repr(args)+': '+r.stderr[-2500:])
    return r.stdout
assert (root/'evidence/P0_RESERVATION.json').is_file()
assert json.loads((root/'evidence/P0_RESERVATION.json').read_text())['payload_sha256']==PAYLOAD_SHA
try:
    for relative,data in encoded.items():
        p=root/relative;p.parent.mkdir(parents=True,exist_ok=True)
        fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o440)
        with os.fdopen(fd,'wb') as h:h.write(base64.b64decode(data))
    package=json.loads((root/'plan/P0_PACKAGE_MANIFEST.json').read_text())
    assert all(sha(root/p)==v for p,v in package['files'].items()), 'package SHA drift'
    auth=json.loads((root/'plan/EXECUTION_AUTHORIZATION.json').read_text())
    assert auth['approved_plan_sha256']==EXPECTED_PLAN_SHA
    assert sha(root/'plan/G1_EPOCH004_RECON_1MM_DIFFUSION_ECNU_PLAN_20261013_R1.md')==EXPECTED_PLAN_SHA
    assert auth['operator_approved'] is False and auth['status']=='execution_authorized'
    assert sha(source_code/'data/data_csv.csv')==EXPECTED_SOURCE_CSV_SHA
    assert sha(recon/'RECONSTRUCTION_RECOVERY_PLAN.json')=='545a8d6b2fa9fe155bd1da17914734d5da9762e72e321253e82098bf8a25d83e'
    assert sha(recon/'runtime/reconstruct_epoch004.py')=='525b2dd071c956a2d24d6ad9885118a2f45bfe5f7401998e6ea553d96e33a5da'
    assert not (recon/'dataset/FAILED.json').exists()
    complete=json.loads((recon/'dataset/COMPLETE.ok').read_text())
    assert complete==json.loads((recon/'dataset/DATASET_VALIDATION.json').read_text())
    assert complete['status']=='pass' and complete['cases']==1030 and complete['images']==4120 and complete['labels']==1030
    assert '3941042|COMPLETED|0:0' in cmd(['sacct','-n','-P','-j','3941042','--format=JobID,State,ExitCode'])
    mapping=json.loads((root/'config/P0_SOURCE_MAPPING.json').read_text())['corrected_label_paths']
    with (root/'plan/corrected_label_overlay.csv').open(newline='') as h: overlays={r['case_id']:r for r in csv.DictReader(h) if r['applied'].lower()=='true'}
    with (root/'plan/splits_final_train_val_test_membership.csv').open(newline='') as h: members=list(csv.DictReader(h))
    assert len(members)==1030 and len({r['source_case_id'] for r in members})==1030
    assert collections.Counter(r['split'] for r in members)=={'train':823,'val':103,'test':104}
    assert set(mapping)==set(overlays)&{r['source_case_id'] for r in members}
    for case,path in mapping.items():
        p=pathlib.Path(path)
        source=source_code/'data/input'/case/(case+'-seg.nii.gz')
        assert p.is_file() and source.is_file() and p.resolve()==source.resolve(), 'corrected source mapping drift'
    assert os.statvfs(root).f_bavail*os.statvfs(root).f_frsize>100*1024**3, 'public disk capacity'
    script=root/'slurm/validate_native.slurm'
    cmd(['bash','-n',str(script)])
    env=os.environ.copy();env['G1_STATIC_DRY_RUN']='1'
    dry=cmd(['bash','-u',str(script)],env=env)
    assert 'P0_BASH_U_DRY_RUN_PASS' in dry
    for p in (root/'runtime/validate_native.py',root/'tests/test_native_validator.py'):
        ast.parse(p.read_text())
        with tempfile.TemporaryDirectory() as d: py_compile.compile(str(p),cfile=str(pathlib.Path(d)/'test.pyc'),doraise=True)
    spec=importlib.util.spec_from_file_location('p0_native_runtime',root/'runtime/validate_native.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tests=cmd([PYTHON_BIN,'-B',str(root/'tests/test_native_validator.py')],timeout=60)
    queue=cmd(['squeue','-h','-u','zqchen','-o','%i|%T|%j|%P|%r'])
    partition=cmd(['scontrol','show','partition','cpu_96G'])
    assert 'State=UP' in partition and 'zqchen' in partition
    preflight={**meta,'status':'pass','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,'package_manifest_sha256':sha(root/'plan/P0_PACKAGE_MANIFEST.json'),'files':package['files'],'bash_n':'pass','bash_u_dry_run':'pass','py_compile':'pass','real_import':'pass','native_tests':'9 pass','user_queue_before':queue,'partition':'cpu_96G','cpus':8,'memory_gb':32,'walltime':'04:00:00','gpu':0}
    write(root/'evidence/P0_STATIC_PREFLIGHT.json',preflight)
    result=subprocess.run(['sbatch','--parsable','--hold','--export=NONE',str(script)],capture_output=True,text=True,timeout=30)
    write(root/'evidence/P0_SUBMISSION_RESPONSE.json',{**meta,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    if result.returncode: raise RuntimeError('unique sbatch refused: '+result.stderr[-2000:])
    job=result.stdout.strip().split(';',1)[0]
    assert re.fullmatch('[0-9]+',job), 'noncanonical job id'
    write(root/'evidence/P0_CANONICAL_JOB.json',{**meta,'job_id':job,'status':'held','script_sha256':sha(script)})
    held=cmd(['scontrol','show','job',job])
    assert 'JobState=PENDING' in held and 'JobName=g1d1mmp0r1' in held and 'Partition=cpu_96G' in held and 'Reason=JobHeldUser' in held
    assert str(root/'logs') in held and 'g1d1mmp0r1_'+job+'.out' in held and 'g1d1mmp0r1_'+job+'.err' in held
    write(root/'evidence/P0_LAUNCH_AUDIT.json',{**meta,'status':'held_audited','job_id':job,'script_sha256':sha(script),'package_manifest_sha256':sha(root/'plan/P0_PACKAGE_MANIFEST.json'),'held_job':held,'preflight_sha256':sha(root/'evidence/P0_STATIC_PREFLIGHT.json'),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    assert sha(script)==package['files']['slurm/validate_native.slurm']
    cmd(['scontrol','release',job])
    receipt={**meta,'status':'released','job_id':job,'stage':'P0_native_integrity','remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,'remote_root':str(root),'held_job':held,'script_sha256':sha(script),'preflight':preflight}
    write(root/'evidence/P0_RELEASE_RECEIPT.json',receipt)
    print(re.sub('password:','[AUTH_PROMPT]:',json.dumps(receipt,sort_keys=True),flags=re.I))
except Exception as e:
    write(root/'evidence/P0_OPERATION_FAILED.json',{**meta,'status':'failed','exception':type(e).__name__,'message':str(e),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    raise
'''
    replacements = {'REMOTE_ROOT':repr(str(REMOTE)), 'RECON_ROOT':repr(str(RECON)),
                    'SOURCE_CODE_ROOT':repr(str(SOURCE_CODE)), 'PAYLOAD_SHA':repr(m.sha(payload_path)),
                    'EXPECTED_PLAN_SHA':repr(EXPECTED_PLAN), 'EXPECTED_SOURCE_CSV_SHA':repr('0d1765e12cdad7d8a5df57a5d2c46c79414ed99a2015a4d8fed57cd4c4fc836b'), 'PYTHON_BIN':repr(m.PYTHON)}
    for key,value in replacements.items(): code=code.replace(key,value)
    compile(code, '<remote_p0_submission>', 'exec')
    try:
        access.start_vpn()
        reservation = (
            "import pathlib,json,os,datetime;root=pathlib.Path("+repr(str(REMOTE))+");root.mkdir(mode=0o700,exist_ok=False);"
            "(root/'evidence').mkdir(mode=0o700);(root/'logs').mkdir(mode=0o700);"
            "p=root/'evidence/P0_RESERVATION.json';fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o440);"
            "h=os.fdopen(fd,'w');json.dump("+repr({**m.META,'status':'reserved','payload_sha256':m.sha(payload_path),'local_audit':str(access.audit)})+",h);h.close();print('P0_ROOT_RESERVED')"
        )
        access.ssh('env PYTHONDONTWRITEBYTECODE=1 '+m.PYTHON+' -B -c '+shlex.quote(reservation),timeout=40)
        access.scp(payload_path, str(REMOTE/'evidence/P0_PACKAGE_PAYLOAD.json'))
        output = access.ssh('env PYTHONDONTWRITEBYTECODE=1 ' + m.PYTHON + ' -B -c ' + shlex.quote(code), timeout=220)
        receipt=json.loads(output.strip().splitlines()[-1])
        receipt['local_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        m.exclusive(access.audit / 'SUBMISSION_RECEIPT.json', receipt)
        print(json.dumps({'job_id':receipt['job_id'],'stage':receipt['stage'],'status':receipt['status'],'audit':str(access.audit)},ensure_ascii=False))
    finally:
        access.close_vpn()


if __name__ == '__main__':
    main()
