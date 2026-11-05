"""R2 identity and proven empty-layer fix; reuse the existing submission gates."""
import hashlib

import s0_sif_recovery_sources as prior
from s0_cpu_r2_sources import replace_one
from sif_empty_layer_sources import candidate_source

ROOT = prior.ROOT
ATTEMPT = 's0_sif_recovery_r2'
NAME = 's2exts0sifr2'

EXTRA_GUARDS = r'''
def failed_sif_r1_guard(root):
    parent=Path(root)/'attempts/s0_sif_recovery_r1'
    expected={
        'contract/PREPARATION_CONTRACT.json':'d957f9e1240186436b7e8ecb607a0f47ef52e7be2b1303cd384959bffe428784',
        'slurm/prepare.slurm':'4368622f008a106baf159dde2842723f4b387d24bcd208b1b4ad5285779df60b',
        'reports/FAILED.json':'526085de8e5a3209ab43cd38cce22826aad54e1ea5a89f4a8fe338da79ad2e9d'}
    for name,digest in expected.items():require(sha(parent/name)==digest,'failed SIF R1 identity drift: '+name)
    contract=json.loads((parent/'contract/PREPARATION_CONTRACT.json').read_text())
    for name,digest in contract['package_files'].items():require(sha(parent/name)==digest,'failed SIF R1 package drift')
    failure=json.loads((parent/'reports/FAILED.json').read_text())
    require(failure['job_id']=='4304987' and failure['status']=='S0_SIF_recovery_failed','failed SIF R1 marker identity')
    require(not (parent/'reports/COMPLETE.json').exists(),'failed SIF R1 COMPLETE conflict')
    return dict(expected,package_files=contract['package_files'])

def remote_python310_guard(root,proof):
    reference=proof['python310_verification']
    path=Path(root)/'operator/sif_empty_layer_verification_r2/COMPLETE.json'
    require(path.is_file() and path.stat().st_size<131072 and sha(path)==reference['remote_complete_sha256'],'same-Python verification missing or drift')
    result=json.loads(path.read_text())
    require(result['status']=='same_remote_Python310_empty_layer_and_small_native_probe_pass_not_full_SIF_or_GPU_pass','same-Python verification status')
    require(result['python_major_minor']==[3,10] and result['all_eight_checks_pass'] and len(result['checks'])==8,'same-Python checks incomplete')
    require(result['original_empty_errno22_reproduced'] and result['small_same_backend_native_probe_pass'],'empty/native compatibility incomplete')
    require(result['candidate_source_sha256']==reference['candidate_source_sha256']==proof['normalizer_source_sha256'],'tested candidate source drift')
    require(result['original_source_sha256']==proof['original_normalizer_source_sha256']=='8c8ba3ac4daad96a4f6a8fe043757e4d102f7e913a0a2a88ec28960005258205','original normalizer identity drift')
    require(result['failed_recovery_sha256']=='526085de8e5a3209ab43cd38cce22826aad54e1ea5a89f4a8fe338da79ad2e9d','tested prior failure drift')
    require(all(result[name] is False for name in ('models_loaded','images_read','jobs_created','full_image_conversion_attempted','packages_installed')),'verification scope drift')
    return result
'''


def mapped(source):
    return source.replace('s0_sif_recovery_r1', ATTEMPT).replace('s2exts0sifr1', NAME).replace(
        'prepare_sif_recovery_r1.py', 'prepare_sif_recovery_r2.py')


def guarded(source):
    source = mapped(source)
    source = replace_one(source, "    proof=json.loads((Path(attempt)/'contract/COMPATIBILITY_VERIFICATION.json').read_text())",
        "    require(failed_sif_r1_guard(root)==contract['failed_SIF_R1_guard'],'failed SIF R1 changed')\n"
        "    proof=json.loads((Path(attempt)/'contract/COMPATIBILITY_VERIFICATION.json').read_text())\n"
        "    remote_python310_guard(root,proof)")
    return source + '\n' + EXTRA_GUARDS


def runtime_files():
    old = prior.runtime_files()
    runtime = mapped(old['prepare_sif_recovery_r1.py'])
    runtime = replace_one(runtime, "original_failed_job_preserved='4304783',",
                          "original_failed_job_preserved='4304783',previous_failed_recovery_preserved='4304987',")
    return {'s0_build_contract.py': guarded(old['s0_build_contract.py']),
            'prepare_s0_cpu_r1.py': old['prepare_s0_cpu_r1.py'],
            'prepare_sif_recovery_r2.py': runtime,
            'normalize_oci_hardlinks.py': candidate_source()}


def common_source():
    return guarded(prior.common_source())


def protocol_source():
    source = mapped(prior.protocol_source())
    source = source.replace('4304783', '4304987')
    source = replace_one(source, "    accepted,wheels=recovery_parent(root)",
        "    failed_before=failed_sif_r1_guard(root)\n    accepted,wheels=recovery_parent(root)")
    source = replace_one(source,
        "    require(hashlib.sha256(files['normalize_oci_hardlinks.py'].encode()).hexdigest()==proof['normalizer_source_sha256']==result['normalizer_source_sha256'],'tested normalizer identity drift')",
        "    require(result['normalizer_source_sha256']==proof['original_normalizer_source_sha256'],'original hardlink proof identity drift')\n"
        "    py310=remote_python310_guard(root,proof)\n"
        "    require(hashlib.sha256(files['normalize_oci_hardlinks.py'].encode()).hexdigest()==proof['normalizer_source_sha256']==py310['candidate_source_sha256'],'tested candidate identity drift')")
    source = replace_one(source, '        reused_wheels=wheels,parent_wheels_manifest_sha256=',
                          '        failed_SIF_R1_guard=failed_before,reused_wheels=wheels,parent_wheels_manifest_sha256=')
    source = replace_one(source,
        "    require(authorization['single_R3_CPU_held_submission_and_release_authorized'], 'R3 not authorized')",
        "    require(authorization['single_R3_CPU_held_submission_and_release_authorized'], 'R3 not authorized')\n"
        "    require(authorization['independent_R2_authorized'] and authorization['same_remote_Python310_verification_before_submission_required'],'R2 authorization missing')")
    compile(source, 'SIF_recovery_R2_protocol', 'exec')
    return source


def source_identities():
    return {name: hashlib.sha256(source.encode()).hexdigest() for name, source in runtime_files().items()}
