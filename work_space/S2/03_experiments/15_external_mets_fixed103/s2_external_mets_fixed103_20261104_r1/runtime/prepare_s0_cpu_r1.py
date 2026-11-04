"""Allocated CPU-only offline wheel construction and one fixed-image conversion."""
import argparse
from datetime import datetime, timezone
from email.parser import Parser
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import time
import zipfile

from s0_build_contract import META, ARCHIVE_SHA, SOURCES, require, sha, write, prior_guard


def run_owned(command, log, env, timeout, compression=False):
    import resource

    def limits():
        resource.setrlimit(resource.RLIMIT_CPU, (7200, 7200))
        if compression:
            os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
            resource.setrlimit(resource.RLIMIT_AS, (2*1024**3, 2*1024**3))

    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                               start_new_session=True, preexec_fn=limits)
    try:
        require(process.wait(timeout=timeout) == 0, 'CPU preparation subprocess failed')
    except BaseException:
        # Only the process group created for this exact subprocess is terminated.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
        raise


def validate_sdist(path):
    import tomli
    from packaging.requirements import Requirement
    from importlib.metadata import version
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        require(len(members) < 10000 and sum(x.size for x in members) < 50*1024**2, 'sdist expansion budget')
        for member in members:
            p = PurePosixPath(member.name)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in member.name,
                    'sdist path traversal')
            require(member.isfile() or member.isdir(), 'sdist special file or link')
        configs = [x for x in members if x.name.count('/') == 1 and x.name.endswith('/pyproject.toml')]
        require(len(configs) == 1, 'sdist build configuration ambiguous')
        config = tomli.loads(archive.extractfile(configs[0]).read().decode())
    requires = config['build-system']['requires']
    for text in requires:
        item = Requirement(text)
        if item.marker is None or item.marker.evaluate():
            require(not item.url and version(item.name) in item.specifier, 'unstaged build requirement: '+text)
    return dict(build_backend=config['build-system']['build-backend'], requires=requires)


def wheel_identity(path, name, expected, destination, readelf, env):
    from packaging.utils import parse_wheel_filename, canonicalize_name
    distribution, version, _, tags = parse_wheel_filename(path.name)
    require(distribution == canonicalize_name(name) and str(version) == expected, 'wheel identity drift')
    require(any(t.interpreter == 'cp310' and t.abi == 'cp310' and t.platform == 'linux_x86_64' for t in tags),
            'native CPython310 wheel tag drift')
    destination.mkdir(mode=0o700, exist_ok=False)
    with zipfile.ZipFile(path) as archive:
        names = set()
        require(sum(x.file_size for x in archive.infolist()) < 100*1024**2, 'wheel expansion budget')
        for item in archive.infolist():
            p = PurePosixPath(item.filename)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in item.filename
                    and item.filename not in names, 'wheel path traversal or duplicate')
            names.add(item.filename)
            require((item.external_attr >> 16) & 0o170000 in (0, 0o100000, 0o040000), 'wheel special file')
            if item.is_dir():
                continue
            target = destination / item.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(archive.read(item))
        metadata = [x for x in names if x.endswith('.dist-info/METADATA')]
        require(len(metadata) == 1, 'wheel metadata missing')
        parsed = Parser().parsestr(archive.read(metadata[0]).decode())
        require(canonicalize_name(parsed['Name']) == canonicalize_name(name) and parsed['Version'] == expected,
                'wheel distribution metadata drift')
    binaries = []
    for library in sorted(destination.rglob('*.so')):
        text = subprocess.check_output([readelf, '--version-info', str(library)], env=env, text=True, timeout=30)
        versions = [tuple(map(int, v.split('.'))) for v in re.findall(r'GLIBC_(\d+\.\d+(?:\.\d+)?)', text)]
        require(all(v <= (2, 17) for v in versions), 'wheel requires newer system glibc')
        binaries.append(dict(path=str(library.relative_to(destination)), sha256=sha(library), glibc_versions=versions))
    require(binaries, 'compiled wheel has no extension')
    return dict(name=name, version=expected, path=str(path), bytes=path.stat().st_size, sha256=sha(path),
                binaries=binaries, requires_dist=parsed.get_all('Requires-Dist', []))


def convert_sif(attempt, contract, env):
    backend = contract['backend']
    require(subprocess.check_output([backend, 'version'], text=True, env=env, timeout=20).strip() == '4.1.1',
            'native backend version drift')
    conversion = attempt/'conversion'
    conversion.mkdir(mode=0o700, exist_ok=False)
    temporary = conversion/'tmp'
    temporary.mkdir()
    cache = conversion/'cache'
    cache.mkdir()
    env = {k:v for k,v in env.items() if not k.startswith(('SINGULARITY', 'APPTAINER'))}
    env.update(SINGULARITY_TMPDIR=str(temporary), SINGULARITY_CACHEDIR=str(cache))
    sandbox, squash, partial = [conversion/name for name in ('native_rootfs', 'native_rootfs.squashfs', 'fixed2025.part.sif')]
    commands = [
        [backend, 'build', '--disable-cache', '--sandbox', '--no-cleanup', str(sandbox), 'docker-archive://'+contract['archive']],
        ['/usr/sbin/mksquashfs', str(sandbox), str(squash), '-noappend', '-all-root', '-comp', 'gzip',
         '-processors', '2', '-read-queue', '64', '-write-queue', '128', '-fragment-queue', '64', '-no-progress'],
        [backend, 'sif', 'new', str(partial)],
        [backend, 'sif', 'add', str(partial), str(squash), '--datatype', '4', '--parttype', '2', '--partfs', '1', '--partarch', '2'],
    ]
    write(attempt/'reports/NATIVE_BUILD_COMMANDS.json', dict(commands=commands, deadline_seconds=7200,
          compression_threads=2, compression_queue_mib=256, system_configuration_changed=False))
    deadline = time.monotonic()+7200
    with (attempt/'logs/SIF_BUILD.log').open('x') as log:
        for index, command in enumerate(commands):
            remaining = deadline-time.monotonic()
            require(remaining > 0, 'single conversion deadline exhausted')
            write(attempt/f'reports/SIF_STAGE_{index}_STARTED.json',
                  dict(remote_utc=datetime.now(timezone.utc).isoformat(), command=command))
            run_owned(command, log, env, remaining, compression=index == 1)
            if index == 0:
                require((sandbox/'.singularity.d/runscript').is_file(), 'native runscript absent')
        listing = subprocess.check_output([backend, 'sif', 'list', str(partial)], env=env, text=True, timeout=30)
    require(listing.count('FS (Squashfs/*System/amd64)') == 1, 'native primary descriptor drift')
    final = conversion/'fixed2025.sif'
    require(not final.exists(), 'SIF destination already exists')
    partial.chmod(0o440)
    os.link(partial, final)
    return dict(path=str(final), bytes=final.stat().st_size, sha256=sha(final), descriptor=listing,
                source_archive_sha256=ARCHIVE_SHA, failed_old_conversion_preserved=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=Path, required=True)
    args = parser.parse_args()
    attempt = args.attempt.resolve()
    contract = json.loads((attempt/'contract/PREPARATION_CONTRACT.json').read_text())
    root = Path(contract['remote_root'])
    require(attempt == root/'attempts/s0_cpu_prepare_r1', 'CPU preparation path drift')
    reports = attempt/'reports'
    require(not (reports/'STARTED.json').exists(), 'CPU preparation already started; no retry')
    stage = 'allocated_CPU_identity'
    try:
        require(sha(attempt/'contract/PREPARATION_CONTRACT.json') == os.environ.get('S0_EXPECTED_CONTRACT_SHA256'),
                'frozen CPU contract SHA drift')
        require(os.environ.get('SLURM_JOB_ID') and int(os.environ['SLURM_CPUS_PER_TASK']) == 8, 'CPU allocation missing')
        require(not os.environ.get('SLURM_JOB_GPUS') and not os.environ.get('CUDA_VISIBLE_DEVICES'), 'GPU in CPU preparation')
        require(Path(sys.executable).resolve() == (Path(contract['build_env'])/'bin/python').resolve(), 'build interpreter drift')
        for name, digest in contract['package_files'].items():
            require(sha(attempt/name) == digest, 'CPU package drift: '+name)
        for name, digest in contract['build_env_files'].items():
            require(sha(Path(contract['build_env'])/name) == digest, 'build environment drift: '+name)
        require(prior_guard(root) == contract['prior_guard'], 'old environment or failure drift')
        require(sha(Path(contract['compiler_preflight'])/'COMPLETE.json')==contract['compiler_preflight_sha256'],
                'independent compiler/scheduler acceptance drift')
        require(sha(Path(contract['compiler_solver_preflight'])/'FAILED.json')==contract['original_R3_failed_sha256'],
                'original R3 failure drift')
        require(sha(contract['archive']) == ARCHIVE_SHA, 'accepted archive drift')
        import shutil
        require(shutil.disk_usage(attempt).free > contract['incremental_budget_bytes'], 'CPU preparation fresh capacity failed')
        env = dict(os.environ)
        for name in list(env):
            if name.upper().endswith('_PROXY') or name.startswith('CONDA_OVERRIDE_'):
                del env[name]
        env.update(PIP_CONFIG_FILE='/dev/null', PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1',
                   CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', MAX_JOBS='8', CMAKE_BUILD_PARALLEL_LEVEL='8')
        cc, cxx = [shlex.split(env[name]) for name in ('CC', 'CXX')]
        for command in (cc, cxx):
            binary = shutil.which(command[0], path=env.get('PATH')) if len(command) == 1 else None
            require(binary and Path(binary).resolve().is_relative_to(Path(contract['build_env'])), 'compiler outside build env')
        write(reports/'STARTED.json', dict(status='CPU_offline_preparation_started', job_id=os.environ['SLURM_JOB_ID'],
              remote_utc=datetime.now(timezone.utc).isoformat(), contract_sha256=sha(attempt/'contract/PREPARATION_CONTRACT.json')))
        stage = 'offline_same_version_wheel_build'
        wheels = []
        for name, item in SOURCES.items():
            source = attempt/'sources'/item['filename']
            require(source.stat().st_size == item['bytes'] and sha(source) == item['sha256'], 'source identity drift')
            configuration = validate_sdist(source)
            output = attempt/'wheels'/name
            output.mkdir(mode=0o700, exist_ok=False)
            with (attempt/'logs'/f'{name}_BUILD.log').open('x') as log:
                run_owned([sys.executable, '-B', '-m', 'pip', 'wheel', '--no-index', '--no-deps', '--no-build-isolation',
                           '--no-cache-dir', '--wheel-dir', str(output), str(source)], log, env, 1800)
            candidates = list(output.glob('*.whl'))
            require(len(candidates) == 1, 'exact single built wheel required')
            destination = attempt/'wheel_probe'/name
            identity = wheel_identity(candidates[0], name, item['version'], destination, contract['readelf'], env)
            identity.update(source_sha256=item['sha256'], configuration=configuration)
            wheels.append(identity)
            probe = {'numexpr': 'import numexpr,numpy as n; a=n.arange(8); assert n.array_equal(numexpr.evaluate("a+1"),a+1)',
                     'connected-components-3d': 'import cc3d,numpy as n; a=n.zeros((4,4,4),n.uint8); a[0,0,0]=1; a[3,3,3]=1; assert int(cc3d.connected_components(a,connectivity=26).max())==2',
                     'ndindex': 'from ndindex import ndindex; assert ndindex(slice(1,3)).raw==slice(1,3)'}[name]
            code = 'import sys;sys.path.insert(0,'+repr(str(destination))+');'+probe
            subprocess.check_call([sys.executable, '-B', '-c', code], env=env, timeout=60)
            write(reports/f'WHEEL_{name}.json', identity)
        write(reports/'WHEELS_COMPLETE.json', dict(status='three_same_version_wheels_CPU_probe_pass', wheels=wheels))
        stage = 'single_offline_SIF_conversion'
        sif = convert_sif(attempt, contract, env)
        require(prior_guard(root) == contract['prior_guard'], 'old environment/failure changed after preparation')
        require(sha(contract['archive']) == ARCHIVE_SHA, 'source archive changed after conversion')
        write(reports/'COMPLETE.json', dict(status='S0_CPU_artifacts_complete_runtime_environment_still_requires_consumption',
              job_id=os.environ['SLURM_JOB_ID'], remote_utc=datetime.now(timezone.utc).isoformat(), sif=sif,
              wheels_manifest_sha256=sha(reports/'WHEELS_COMPLETE.json'), prior_environments_preserved=True,
              models_loaded=False, images_read=False, runtime_R4_installed=False, original_inference_jobs_created=False))
    except BaseException as error:
        write(reports/'FAILED.json', dict(status='S0_CPU_preparation_failed', stage=stage, error_type=type(error).__name__,
              message=str(error), job_id=os.environ.get('SLURM_JOB_ID'), remote_utc=datetime.now(timezone.utc).isoformat(),
              no_automatic_retry=True, models_loaded=False, images_read=False))
        raise


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise RuntimeError('CPU preparation interrupted: '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main()
