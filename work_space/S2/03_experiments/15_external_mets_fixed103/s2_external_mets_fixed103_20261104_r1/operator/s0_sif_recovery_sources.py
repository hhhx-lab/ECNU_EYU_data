"""Scoped recovery package and protocol; retain the tested scheduler and held gates."""
import hashlib
from pathlib import Path

import submit_s0_cpu_r3a as parent
from s0_cpu_r2_sources import replace_function, replace_one

ROOT=parent.ROOT
ATTEMPT='s0_sif_recovery_r1'
NAME='s2exts0sifr1'

GUARDS='''
def recovery_parent(root):
    path=Path(root)/'attempts/s0_cpu_prepare_r3a'
    require(sha(path/'reports/FAILED.json')=='8a5d33b7f6900d27c8f9c282c82b08cd373330ca3dcb2f321a1520a53c4c4432','original CPU failure drift')
    require(sha(path/'contract/PREPARATION_CONTRACT.json')=='6615846dca72b6b0cd2dd861cfa8e656ae9e2c9f51ffbb952f566fc62481063c','parent R3a contract drift')
    require(sha(path/'slurm/prepare.slurm')=='cc76d642f087be1b5b83d3b7208ffb152ca256e303071017f7f7ee66438795d5','parent R3a Slurm drift')
    require(sha(path/'reports/WHEELS_COMPLETE.json')=='79b288745b7912cdd3b4ddebe4782ee4b8a66d96c8fd159bec6a0440394eb6d5','original wheels manifest drift')
    contract=json.loads((path/'contract/PREPARATION_CONTRACT.json').read_text())
    for relative,digest in contract['package_files'].items():require(sha(path/relative)==digest,'parent R3a frozen runtime drift')
    wheels=json.loads((path/'reports/WHEELS_COMPLETE.json').read_text())['wheels']
    require(len(wheels)==3,'three reused wheels required')
    for item in wheels:
        wheel=Path(item['path'])
        require(wheel.is_relative_to(path/'wheels') and not wheel.is_symlink() and wheel.is_file(),'reused wheel path boundary')
        require(wheel.stat().st_size==item['bytes'] and sha(wheel)==item['sha256'],'reused wheel bytes/SHA drift')
    return contract,wheels

def recovery_guard(attempt,contract):
    root=Path(contract['remote_root'])
    require(Path(attempt)==root/'attempts/s0_sif_recovery_r1' and contract['attempt']==str(attempt),'recovery attempt drift')
    require(contract['resource']==dict(partition='cpu_192G',cpus_per_task=8,memory_gib=64,time_limit='04:00:00',gpu=0),'recovery resource drift')
    require(contract['maximum_normalized_layer_bytes']==16*1024**3 and contract['incremental_budget_bytes']==150*1024**3,'recovery storage budget drift')
    require(contract['single_conversion_deadline_seconds']==7200 and not contract['automatic_retry'],'recovery timing policy drift')
    require(prior_guard(root)==contract['prior_guard'],'prior environment/failure drift')
    require(parent_staging_guard(root)==contract['parent_attempt_files'],'R1 staging drift')
    require(parent_r2_guard(root)==contract['parent_R2_attempt_files'],'R2 attempt drift')
    require(failed_r3_guard(root)==contract['failed_R3_attempt_files'],'failed R3 inventory drift')
    _,wheels=recovery_parent(root)
    require(wheels==contract['reused_wheels'],'reused wheel manifest changed')
    for relative,key in [('contract/USER_AUTHORIZATION.json','user_authorization_sha256'),
                         ('contract/COMPATIBILITY_VERIFICATION.json','compatibility_verification_sha256')]:
        require(sha(Path(attempt)/relative)==contract[key],'recovery evidence drift')
    proof=json.loads((Path(attempt)/'contract/COMPATIBILITY_VERIFICATION.json').read_text())
    remote_proof=root/'operator/sif_hardlink_compatibility_r1/COMPLETE.json'
    require(sha(remote_proof)==proof['remote_complete_sha256'],'real minimal compatibility proof drift')
    require(proof['normalizer_source_sha256']==contract['package_files']['runtime/normalize_oci_hardlinks.py'],'normalizer not the tested version')
    for relative,digest in contract['package_files'].items():require(sha(Path(attempt)/relative)==digest,'recovery package drift')
    for relative,digest in contract['build_env_files'].items():require(sha(Path(contract['build_env'])/relative)==digest,'accepted build env drift')
    require(Path(contract['build_env']).name=='bio_build_glibc217_r2','recovery build env identity')
'''

VALIDATE='''
def validate_package(attempt,contract):
    recovery_guard(attempt,contract)
    for relative in contract['package_files']:compile((attempt/relative).read_text(),relative,'exec')
'''

FREEZE='''
def freeze_r3(root,parent_contract,files,authorization,bootstrap_proof):
    attempt=root/'attempts/s0_sif_recovery_r1'
    require(not attempt.exists(),'recovery attempt exists; no repeat')
    accepted,wheels=recovery_parent(root)
    require(accepted==parent_contract,'parent contract identity mismatch')
    require(prior_guard(root)==accepted['prior_guard'],'historical provenance drift')
    proof=bootstrap_proof['compatibility_verification']
    remote_proof=root/'operator/sif_hardlink_compatibility_r1/COMPLETE.json'
    require(sha(remote_proof)==proof['remote_complete_sha256'],'minimal compatibility proof SHA drift')
    result=json.loads(remote_proof.read_text())
    require(result['normalized_native_probe_pass'] and result['original_error_reproduced'],'real compatibility gate missing')
    require(hashlib.sha256(files['normalize_oci_hardlinks.py'].encode()).hexdigest()==proof['normalizer_source_sha256']==result['normalizer_source_sha256'],'tested normalizer identity drift')
    require(sha(accepted['archive'])==ARCHIVE_SHA,'original archive SHA drift')
    require(shutil.disk_usage(root).free>150*1024**3,'fresh recovery storage gate')
    require(set(files)=={'s0_build_contract.py','prepare_s0_cpu_r1.py','prepare_sif_recovery_r1.py','normalize_oci_hardlinks.py'},'recovery runtime set drift')
    __HEADER_BUDGET__
    attempt.mkdir(mode=0o700)
    CONTEXT.update(owns_audit=True,audit=str(attempt/'operator'),stage='freeze_recovery_package',job_id=None)
    for name in ('runtime','contract','logs','reports','scratch','operator','slurm'):(attempt/name).mkdir(mode=0o700)
    write(attempt/'contract/USER_AUTHORIZATION.json',authorization)
    write(attempt/'contract/COMPATIBILITY_VERIFICATION.json',proof)
    for name,source in files.items():
        compile(source,name,'exec')
        path=attempt/'runtime'/name
        with path.open('x') as stream:stream.write(source)
        path.chmod(0o440)
    contract=json.loads(json.dumps(accepted))
    contract.update(attempt=str(attempt),package_files={'runtime/'+name:hashlib.sha256(source.encode()).hexdigest() for name,source in files.items()},
        reused_wheels=wheels,parent_wheels_manifest_sha256='79b288745b7912cdd3b4ddebe4782ee4b8a66d96c8fd159bec6a0440394eb6d5',
        maximum_normalized_layer_bytes=16*1024**3,incremental_budget_bytes=150*1024**3,
        user_authorization_sha256=sha(attempt/'contract/USER_AUTHORIZATION.json'),
        compatibility_verification_sha256=sha(attempt/'contract/COMPATIBILITY_VERIFICATION.json'),
        wheels_recompiled=False,assets_redownloaded=False,normalization_method='materialize_original_tar_hardlinks_before_native_OCI_merger',
        original_Docker_digest_preserved_as_source_not_derivative=True)
    path=attempt/'contract/PREPARATION_CONTRACT.json'
    write(path,contract)
    script=render_slurm(attempt,contract['build_env'],os.environ['EXTERNAL_MET_CONDA_BIN'],sha(path))
    with (attempt/'slurm/prepare.slurm').open('x') as stream:stream.write(script)
    (attempt/'slurm/prepare.slurm').chmod(0o440)
    return attempt,json.loads(path.read_text())
'''

HEADER_BUDGET='''
    import tarfile,time
    namespace={}
    normalizer=files['normalize_oci_hardlinks.py'].replace('from s0_build_contract import require, sha\\n','')
    namespace.update(require=require,sha=sha)
    exec(compile(normalizer,'normalizer_static_header_budget','exec'),namespace)
    total=0
    deadline=time.monotonic()+120
    with tarfile.open(accepted['archive'],mode='r:') as outer:
        manifest=json.load(outer.extractfile('manifest.json'))
        for name in manifest[0]['Layers']:
            earlier={}
            with outer.extractfile(name) as stream,tarfile.open(fileobj=stream,mode='r:') as layer:
                for item in layer:
                    require(len(earlier)<300000 and time.monotonic()<deadline,'header/time budget')
                    key=namespace['canonical'](item.name)
                    require(key not in earlier,'duplicate layer member')
                    payload=namespace['resolve_link'](item,earlier).size if item.islnk() else item.size
                    total+=((payload+511)//512)*512+4096
                    earlier[key]=item
                total+=10240
    require(total<16*1024**3,'normalized archive upper bound exceeds 16GiB; no job')
    proof=dict(proof,full_header_normalized_bytes_upper_bound=total,
               simultaneous_copy_upper_bound_bytes=6*(16*1024**3)+20*1024**3)
'''


def runtime_files():
    common=parent.runtime_files()['s0_build_contract.py']+GUARDS
    return {
        's0_build_contract.py':common,
        'prepare_s0_cpu_r1.py':(ROOT/'runtime/prepare_s0_cpu_r1.py').read_text(),
        'prepare_sif_recovery_r1.py':(ROOT/'runtime/prepare_sif_recovery_r1.py').read_text(),
        'normalize_oci_hardlinks.py':(ROOT/'runtime/normalize_oci_hardlinks.py').read_text()}


def common_source():
    source=parent.common_source()
    source=source.replace("attempt.name == 's0_cpu_prepare_r3a'", "attempt.name == 's0_sif_recovery_r1'")
    source=source.replace('s2exts0p3a',NAME).replace('prepare_s0_cpu_r3a.py','prepare_sif_recovery_r1.py')
    return source+'\n'+GUARDS


def protocol_source():
    source=parent.protocol_source().replace('s0_cpu_prepare_r3a',ATTEMPT).replace('s2exts0p3a',NAME)
    source=replace_function(source,'validate_package',VALIDATE)
    freeze=FREEZE.replace('    __HEADER_BUDGET__',HEADER_BUDGET.strip('\n'))
    source=replace_function(source,'freeze_r3',freeze)
    source=source.replace('4304740','4304783')
    source=replace_one(source,"root/'attempts/s0_cpu_prepare_r2/contract/PREPARATION_CONTRACT.json'",
        "root/'attempts/s0_cpu_prepare_r3a/contract/PREPARATION_CONTRACT.json'")
    compile(source,'sif_recovery_protocol','exec')
    return source
