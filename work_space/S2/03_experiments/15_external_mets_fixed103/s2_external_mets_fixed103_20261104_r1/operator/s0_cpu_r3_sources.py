"""Independent R3 identities; reuse the accepted R2 environment without installation."""
import ast
import hashlib
import inspect
from pathlib import Path

from review_s0_cpu_preparation import bootstrap_only, PROBE
from s0_cpu_r2_sources import ROOT, runtime_files as r2_files, held_helpers, replace_one, replace_function
from s0_slurm_fields_r2 import job_fields

PARENT_GUARD = '''
def parent_r2_guard(root):
    parent=Path(root)/'attempts/s0_cpu_prepare_r2'
    require(sha(parent/'contract/PREPARATION_CONTRACT.json')=='8a2246326f58d7351712785e2395c54fb81949ad7a4390f19e18e80980329980', 'parent R2 contract drift')
    require(sha(parent/'slurm/prepare.slurm')=='d879a1addf9756b3bfa8561b96e33f600deb87c1b14f9483e79217d8506620ff', 'parent R2 Slurm drift')
    require(sha(parent/'operator/submit_usage_r2/SUBMISSION_RECEIPT.json')=='592c93f24b50993446ff38210c5e0ef352a554beb7c0c3dea147b41ee4f71eda', 'parent R2 receipt drift')
    paths=[p for p in sorted(parent.rglob('*')) if p.is_file()]
    require(len(paths)<100 and all(p.stat().st_size<1024**2 for p in paths), 'parent R2 light identity budget')
    return {str(p.relative_to(parent)):sha(p) for p in paths}

def r3_evidence_guard(attempt, contract):
    for relative,key in [('contract/USER_AUTHORIZATION.json','user_authorization_sha256'),
                         ('contract/BOOTSTRAP_VERIFICATION.json','bootstrap_verification_sha256'),
                         ('contract/BUILD_ENV_EXPLICIT.txt','build_env_explicit_sha256'),
                         ('reports/SOURCE_BUILD_REQUIREMENTS.json','source_build_requirements_sha256')]:
        require(sha(Path(attempt)/relative)==contract[key], 'R3 evidence drift: '+relative)
'''


def runtime_files():
    files = r2_files()
    files['s0_build_contract.py'] = replace_one(files['s0_build_contract.py'],
        "ATTEMPT = 's0_cpu_prepare_r2'", "ATTEMPT = 's0_cpu_prepare_r3'")+PARENT_GUARD
    runtime = replace_one(files.pop('prepare_s0_cpu_r2.py'),
        "root/'attempts/s0_cpu_prepare_r2'", "root/'attempts/s0_cpu_prepare_r3'")
    runtime = replace_one(runtime, 'prior_guard, parent_staging_guard\n',
                          'prior_guard, parent_staging_guard, parent_r2_guard, r3_evidence_guard\n')
    for before, after in (
        ("        require(parent_staging_guard(root)==contract['parent_attempt_files'], 'parent CPU staging files drift')",
         "        require(parent_r2_guard(root)==contract['parent_R2_attempt_files'], 'parent R2 attempt drift')"),
        ("        require(parent_staging_guard(root)==contract['parent_attempt_files'], 'parent CPU staging files changed after preparation')",
         "        require(parent_r2_guard(root)==contract['parent_R2_attempt_files'], 'parent R2 attempt changed after preparation')"),
    ):
        runtime = replace_one(runtime, before, before+'\n'+after+'\n        r3_evidence_guard(attempt, contract)')
    files['prepare_s0_cpu_r3.py'] = runtime
    for name, source in files.items():
        compile(source, name, 'exec')
    return files


def common_source():
    source = runtime_files()['s0_build_contract.py']+'\n'
    helpers = replace_one(held_helpers(), "attempt.name == 's0_cpu_prepare_r2'",
                          "attempt.name == 's0_cpu_prepare_r3'")
    helpers = helpers.replace('s2exts0p2', 's2exts0p3').replace('prepare_s0_cpu_r2.py', 'prepare_s0_cpu_r3.py')
    helpers = replace_function(helpers, 'job_fields', inspect.getsource(job_fields))
    original_render = next(n for n in ast.parse(helpers).body
                           if isinstance(n, ast.FunctionDef) and n.name == 'render_slurm')
    lines = helpers.splitlines(keepends=True)
    render = ''.join(lines[original_render.lineno-1:original_render.end_lineno])
    render = replace_one(render, "'set -euo pipefail'", "'set -eo pipefail'")
    render = replace_one(render, "'module load singularity-ce/4.1.1',",
                          "'module load singularity-ce/4.1.1', 'set -u',")
    helpers = replace_function(helpers, 'render_slurm', render)
    qos = (ROOT/'operator/s0_qos_table.py').read_text()
    qos = qos.replace('from s0_cpu_submission_contract import memory_mib\n', '')
    qos = qos.replace('    from s0_cpu_submission_contract import job_fields\n', '')
    qos = replace_one(qos, "                assert count==0, 'aggregate QoS TRES requires current usage audit'",
                       '                validate_cpu_usage_limit(snapshot,row,key)')
    usage = (ROOT/'operator/s0_cpu_usage.py').read_text().replace(
        'from s0_cpu_submission_contract import job_fields\n', '')
    source += helpers+'\n'+qos+'\n'+usage+'\n'+inspect.getsource(bootstrap_only)+'\nPROBE='+repr(PROBE)+'\n'
    compile(source, 'R3_shared_identity_and_scheduler_helpers', 'exec')
    return source


def source_identities(files):
    return {name: hashlib.sha256(source.encode()).hexdigest() for name, source in files.items()}
