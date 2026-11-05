"""R3 remote protocol: freeze once, submit held, independently release the same ID."""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

CONTEXT = {}
LOCK_SHA = '866d02c7106d809d4f14b87c0da101e2c3c722ab50f7a8f78e3ea89fb5fc2d77'


def command(argv, timeout=30):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    require(len(p.stdout.encode())+len(p.stderr.encode()) < 131072, 'command output budget')
    return dict(argv=argv, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)


def light_result(attempt, outcome):
    files = {}
    for path in sorted((attempt/'operator').glob('*.json')):
        require(path.stat().st_size < 131072, 'light audit file budget')
        files[path.name] = dict(bytes=path.stat().st_size, sha256=sha(path), content=json.loads(path.read_text()))
    return dict(META, **outcome, light_audit_files=files,
                contract=json.loads((attempt/'contract/PREPARATION_CONTRACT.json').read_text()),
                frozen_slurm=(attempt/'slurm/prepare.slurm').read_text())


def validate_package(attempt, contract):
    root = Path(contract['remote_root'])
    require(attempt == root/'attempts/s0_cpu_prepare_r3' and contract['attempt'] == str(attempt), 'R3 attempt drift')
    require(Path(contract['build_env']).name == 'bio_build_glibc217_r2', 'R3 must reuse R2 environment')
    require(contract['resource'] == dict(partition='cpu_192G', cpus_per_task=8, memory_gib=64,
                                       time_limit='04:00:00', gpu=0), 'R3 resource drift')
    require(contract['single_conversion_deadline_seconds'] == 7200 and not contract['automatic_retry'], 'R3 conversion policy drift')
    require(prior_guard(root) == contract['prior_guard'], 'prior environment or failure drift')
    require(parent_staging_guard(root) == contract['parent_attempt_files'], 'parent R1 staging drift')
    require(parent_r2_guard(root) == contract['parent_R2_attempt_files'], 'parent R2 attempt drift')
    r3_evidence_guard(attempt, contract)
    for name, digest in contract['package_files'].items():
        require(sha(attempt/name) == digest, 'R3 runtime drift: '+name)
        compile((attempt/name).read_text(), name, 'exec')
    for name, digest in contract['build_env_files'].items():
        require(sha(Path(contract['build_env'])/name) == digest, 'R2 build environment drift: '+name)
    for name, item in SOURCES.items():
        path = attempt/'sources'/item['filename']
        require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'R3 sdist drift: '+name)
    require(sha(Path(contract['compiler_preflight'])/'COMPLETE.json') == contract['compiler_preflight_sha256'], 'compiler acceptance drift')
    require(sha(Path(contract['compiler_solver_preflight'])/'FAILED.json') == contract['original_R3_failed_sha256'], 'original solver failure drift')


def freeze_r3(root, parent_contract, files, authorization, bootstrap_proof):
    attempt = root/'attempts/s0_cpu_prepare_r3'
    require(not attempt.exists(), 'R3 attempt already exists; inspect without retry')
    prior, first_parent, second_parent = prior_guard(root), parent_staging_guard(root), parent_r2_guard(root)
    require(prior == parent_contract['prior_guard'] and first_parent == parent_contract['parent_attempt_files'], 'parent provenance drift')
    require(shutil.disk_usage(root).free > parent_contract['incremental_budget_bytes'], 'fresh R3 staging capacity')
    require(sha(parent_contract['archive']) == parent_contract['archive_sha256'] == ARCHIVE_SHA, 'accepted archive drift')
    require(set(files) == {'s0_build_contract.py', 'prepare_s0_cpu_r3.py', 's0_source_requirements_r2.py'}, 'R3 runtime set drift')
    for name, source in files.items():
        compile(source, name, 'exec')
    references = {relative: root/'attempts/s0_cpu_prepare_r2'/relative for relative in
                  ('contract/BUILD_ENV_EXPLICIT.txt', 'reports/SOURCE_BUILD_REQUIREMENTS.json')}
    for relative, original in references.items():
        expected = LOCK_SHA if relative.endswith('BUILD_ENV_EXPLICIT.txt') else parent_contract['source_build_requirements_sha256']
        require(original.is_file() and original.stat().st_size < 131072, 'parent reference missing or above light budget: '+relative)
        require(sha(original) == expected, 'parent reference SHA drift: '+relative)
    attempt.mkdir(mode=0o700)
    CONTEXT.update(owns_audit=True, audit=str(attempt/'operator'), stage='freeze_R3_package', job_id=None)
    for name in ('runtime', 'contract', 'sources', 'logs', 'reports', 'wheels', 'wheel_probe', 'scratch', 'operator', 'slurm'):
        (attempt/name).mkdir(mode=0o700)
    write(attempt/'contract/USER_AUTHORIZATION.json', authorization)
    write(attempt/'contract/BOOTSTRAP_VERIFICATION.json', bootstrap_proof)
    for relative, original in references.items():
        with original.open('rb') as src, (attempt/relative).open('xb') as dst:
            shutil.copyfileobj(src, dst)
        (attempt/relative).chmod(0o440)
    for name, source in files.items():
        path = attempt/'runtime'/name
        with path.open('x') as stream:
            stream.write(source)
        path.chmod(0o440)
    for item in SOURCES.values():
        original = root/'attempts/s0_cpu_prepare_r2/sources'/item['filename']
        require(original.stat().st_size == item['bytes'] and sha(original) == item['sha256'], 'parent sdist drift')
        with original.open('rb') as src, (attempt/'sources'/item['filename']).open('xb') as dst:
            shutil.copyfileobj(src, dst)
        (attempt/'sources'/item['filename']).chmod(0o440)
    contract = json.loads(json.dumps(parent_contract))
    contract.update(attempt=str(attempt), parent_R2_attempt_files=second_parent,
                    package_files={'runtime/'+name: hashlib.sha256(source.encode()).hexdigest() for name, source in files.items()},
                    parent_R2_contract_sha256=sha(root/'attempts/s0_cpu_prepare_r2/contract/PREPARATION_CONTRACT.json'),
                    initialization_policy='nounset_after_system_conda_module_initialization',
                    user_authorization_sha256=sha(attempt/'contract/USER_AUTHORIZATION.json'),
                    build_env_explicit_sha256=sha(attempt/'contract/BUILD_ENV_EXPLICIT.txt'),
                    bootstrap_verification_sha256=sha(attempt/'contract/BOOTSTRAP_VERIFICATION.json'))
    path = attempt/'contract/PREPARATION_CONTRACT.json'
    write(path, contract)
    script = render_slurm(attempt, contract['build_env'], os.environ['EXTERNAL_MET_CONDA_BIN'], sha(path))
    slurm = attempt/'slurm/prepare.slurm'
    with slurm.open('x') as stream:
        stream.write(script)
    slurm.chmod(0o440)
    return attempt, json.loads(path.read_text())


def prepare_held(payload):
    root = Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
    authorization, proof = payload['authorization'], payload['bootstrap_proof']
    require(authorization['single_R3_CPU_held_submission_and_release_authorized'], 'R3 not authorized')
    require(not authorization['environment_reinstallation_authorized'] and not authorization['solver_rerun_authorized'], 'R3 scope drift')
    require(proof['pass_bootstrap_only'] and not proof['full_preparation_pass'] and not proof['GPU_pass'], 'bootstrap proof invalid')
    parent_r2_guard(root)
    terminal = command(['sacct', '-P', '-j', '4304740', '--format=JobID,State,ExitCode,Elapsed'])
    rows = list(csv.DictReader(io.StringIO(terminal['stdout']), delimiter='|'))
    allocation = [r for r in rows if r['JobID'] == '4304740']
    require(terminal['returncode'] == 0 and len(allocation) == 1 and allocation[0]['State'] == 'FAILED'
            and allocation[0]['ExitCode'] == '1:0', 'parent R2 failed terminal not established')
    parent_contract = json.loads((root/'attempts/s0_cpu_prepare_r2/contract/PREPARATION_CONTRACT.json').read_text())
    require(parent_contract['remote_root'] == str(root), 'remote root identity drift')
    attempt, contract = freeze_r3(root, parent_contract, payload['files'], authorization, proof)
    audit = attempt/'operator'
    CONTEXT['stage'] = 'exact_R3_bootstrap_and_static_preflight'
    validate_package(attempt, contract)
    require(command(['/bin/bash', '-n', str(attempt/'slurm/prepare.slurm')])['returncode'] == 0, 'R3 Bash syntax failed')
    script = (attempt/'slurm/prepare.slurm').read_text()
    clean = dict(HOME=os.environ['HOME'], USER='zqchen', LOGNAME='zqchen', SHELL='/bin/bash',
                 PATH='/usr/bin:/bin', LANG='C', PYTHONDONTWRITEBYTECODE='1')
    p = subprocess.run(['/bin/bash', '-s'], input=bootstrap_only(script, PROBE), env=clean,
                       capture_output=True, text=True, timeout=180)
    require(len(p.stdout.encode())+len(p.stderr.encode()) < 131072, 'exact R3 bootstrap output budget')
    markers = [line[len('S0_BOOTSTRAP_PROBE='):] for line in p.stdout.splitlines() if line.startswith('S0_BOOTSTRAP_PROBE=')]
    write(audit/'BOOTSTRAP_EXECUTION.json', dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr,
                                               slurm_sha256=sha(attempt/'slurm/prepare.slurm')))
    require(p.returncode == 0 and len(markers) == 1, 'exact R3 bootstrap failed; no job')
    probe = json.loads(markers[0])
    require(probe['status'] == 'bootstrap_import_compiler_CLI_probe_pass_not_artifact_pass', 'R3 bootstrap probe status')
    require(all(probe[k] is False for k in ('images_read', 'models_loaded', 'packages_installed', 'assets_written')), 'R3 probe scope drift')
    CONTEXT['stage'] = 'fresh_R3_scheduler_and_capacity'
    queries = {
        'qos': ['sacctmgr', '-P', 'show', 'qos'],
        'partition': ['scontrol', 'show', 'partition', 'cpu_192G', '-o'],
        'nodes': ['sinfo', '-N', '-h', '-p', 'cpu_192G', '-o', '%N|%t|%c|%m|%G'],
        'association': ['sacctmgr', '-P', 'show', 'assoc', 'where', 'user=zqchen',
                        'format=User,Account,Partition,QOS,MaxJobs,MaxSubmitJobs'],
        'own_queue': ['squeue', '-h', '-u', 'zqchen', '-o', '%i|%T|%P|%j|%r|%b'],
        'quota': ['quota', '-s'], 'libc': ['getconf', 'GNU_LIBC_VERSION'],
    }
    scheduler = {name: command(argv) for name, argv in queries.items()}
    snapshot = dict(scheduler=scheduler, free_bytes=shutil.disk_usage(root).free,
                    cpu_usage=collect_cpu_usage(scheduler['own_queue']['stdout']))
    write(audit/'RESOURCE_SNAPSHOT.json', snapshot)
    validation = validate_cpu_snapshot(snapshot)
    require(not any(r['name'] == 's2exts0p3' for r in queue_records(scheduler['own_queue']['stdout']).values()), 'R3 job already queued')
    require(snapshot['free_bytes'] > contract['incremental_budget_bytes'], 'R3 fresh capacity failed')
    validate_package(attempt, contract)
    static = dict(status='CPU_static_and_fresh_scheduler_pass', validation=validation,
                  cpu_usage=snapshot['cpu_usage'], contract_sha256=sha(attempt/'contract/PREPARATION_CONTRACT.json'),
                  slurm_sha256=sha(attempt/'slurm/prepare.slurm'), remote_utc=datetime.now(timezone.utc).isoformat())
    write(audit/'STATIC_PREFLIGHT.json', static)
    write(attempt/'reports/STAGING_COMPLETE.json', dict(status='S0_offline_CPU_input_staging_pass_not_model_runtime_pass',
          contract_sha256=static['contract_sha256'], environment_reinstalled=False, solver_rerun=False,
          remote_utc=datetime.now(timezone.utc).isoformat()))
    write(audit/'RESERVATION.json', dict(static, status='R3_single_submission_reserved'))
    CONTEXT['stage'] = 'sbatch_held_once'
    result = command(['sbatch', '--hold', '--parsable', '--chdir='+str(attempt), str(attempt/'slurm/prepare.slurm')], timeout=60)
    write(audit/'SBATCH_RESULT.json', result)
    require(result['returncode'] == 0, 'R3 sbatch nonzero; no retry')
    match = re.fullmatch(r'([0-9]+)(?:;[A-Za-z0-9_.-]+)?', result['stdout'].strip())
    require(match is not None, 'R3 JobID unclear; inspect without retry')
    job_id = match.group(1)
    CONTEXT.update(stage='independent_held_identity_audit', job_id=job_id)
    write(audit/'HELD_CREATED.json', dict(static, status='CPU_held_created', job_id=job_id))
    observed = command(['scontrol', 'show', 'job', job_id, '-o'])
    require(observed['returncode'] == 0, 'R3 held query failed')
    fields = validate_held(observed['stdout'], job_id, attempt, [validation['account']])
    require(fields['QOS'].lower() in [q.lower() for q in validation['applicable_qos']], 'R3 held QoS drift')
    write(audit/'HELD_AUDIT.json', dict(status='CPU_held_identity_pass', job_id=job_id, fields=fields,
          raw_scontrol=observed['stdout'], remote_utc=datetime.now(timezone.utc).isoformat()))
    return light_result(attempt, dict(status='R3_CPU_submitted_held_not_released', job_id=job_id,
                        slurm_sha256=static['slurm_sha256'], contract_sha256=static['contract_sha256'],
                        remote_utc=datetime.now(timezone.utc).isoformat(), original_inference_jobs_created=False))


def release_held(payload):
    root = Path(os.environ['EXTERNAL_MET_REMOTE_ROOT'])
    attempt, audit = root/'attempts/s0_cpu_prepare_r3', root/'attempts/s0_cpu_prepare_r3/operator'
    require(not (audit/'RELEASE_RESULT.json').exists() and not (audit/'SUBMISSION_RECEIPT.json').exists(), 'R3 release already attempted')
    created = json.loads((audit/'HELD_CREATED.json').read_text())
    job_id = created['job_id']
    require(payload['job_id'] == job_id and payload['local_held_verification_pass'] is True, 'local held verification missing')
    require(sha(audit/'HELD_AUDIT.json') == payload['held_audit_sha256'], 'held audit identity drift')
    require(sha(attempt/'contract/PREPARATION_CONTRACT.json') == created['contract_sha256'] == payload['contract_sha256'], 'R3 contract changed while held')
    require(sha(attempt/'slurm/prepare.slurm') == created['slurm_sha256'] == payload['slurm_sha256'], 'R3 Slurm changed while held')
    contract = json.loads((attempt/'contract/PREPARATION_CONTRACT.json').read_text())
    validate_package(attempt, contract)
    require(shutil.disk_usage(root).free > contract['incremental_budget_bytes'], 'R3 release fresh capacity failed')
    observed = command(['scontrol', 'show', 'job', job_id, '-o'])
    require(observed['returncode'] == 0, 'held identity requery failed')
    validation = json.loads((audit/'STATIC_PREFLIGHT.json').read_text())['validation']
    fields = validate_held(observed['stdout'], job_id, attempt, [validation['account']])
    require(fields['QOS'].lower() in [q.lower() for q in validation['applicable_qos']], 'release QoS identity drift')
    CONTEXT.update(owns_audit=True, audit=str(audit), stage='release_same_held_job_once', job_id=job_id)
    write(audit/'LOCAL_HELD_VERIFICATION.json', payload)
    release = command(['scontrol', 'release', job_id])
    write(audit/'RELEASE_RESULT.json', dict(job_id=job_id, **release, remote_utc=datetime.now(timezone.utc).isoformat()))
    require(release['returncode'] == 0, 'R3 release failed; no retry')
    observed = command(['scontrol', 'show', 'job', job_id, '-o'])
    require(observed['returncode'] == 0, 'post-release state unavailable; inspect without resubmission')
    outcome = dict(status='CPU_preparation_job_released_not_artifact_or_model_runtime_pass', job_id=job_id,
                   slurm_sha256=created['slurm_sha256'], contract_sha256=created['contract_sha256'],
                   observed_job=job_fields(observed['stdout']), remote_utc=datetime.now(timezone.utc).isoformat(),
                   models_loaded=False, images_read=False, original_inference_jobs_created=False)
    write(audit/'SUBMISSION_RECEIPT.json', outcome)
    return light_result(attempt, outcome)


def execute(action, payload):
    CONTEXT.clear()
    try:
        require(action in ('prepare_held', 'release_held'), 'unrecognized R3 protocol action')
        return prepare_held(payload) if action == 'prepare_held' else release_held(payload)
    except BaseException as error:
        result = dict(META, status='R3_CPU_operation_stopped_no_automatic_retry', stage=CONTEXT.get('stage', 'preflight'),
                      job_id=CONTEXT.get('job_id'), message=str(error), error_type=type(error).__name__,
                      remote_utc=datetime.now(timezone.utc).isoformat(), no_automatic_retry=True)
        if CONTEXT.get('owns_audit'):
            audit = Path(CONTEXT['audit'])
            audit.mkdir(mode=0o700, exist_ok=True)
            name = 'SUBMISSION_FAILED.json' if action == 'prepare_held' else 'RELEASE_FAILED.json'
            if not (audit/name).exists():
                write(audit/name, result)
        return result
