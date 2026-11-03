"""Offline fixed OCI conversion and exclusive safe ZIP expansion, no model loading."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import zipfile
from contract import META, evidence, read, require, sha, write

IMAGE_DIGEST = 'f83d914d4d7a9d316f7605abaf03fa6adbcf0e86d840d93276e86ad65b2a2a82'


def zip_inventory(archive):
    items = archive.infolist()
    names = set()
    for item in items:
        path = PurePosixPath(item.filename)
        normalized = str(path)
        require(not path.is_absolute() and '..' not in path.parts and '\\' not in item.filename,
                'ZIP path traversal')
        require(normalized not in ('', '.') and normalized.rstrip('/') not in names, 'ZIP duplicate/empty path')
        require(normalized == item.filename.rstrip('/'), 'ZIP noncanonical path')
        names.add(normalized.rstrip('/'))
        kind = (item.external_attr >> 16) & 0o170000
        require(kind in (0, 0o100000, 0o040000), 'ZIP symlink or special file forbidden')
        require(not item.flag_bits & 1, 'encrypted ZIP not supported')
    require(sum(item.file_size for item in items) < 40 * 1024**3, 'ZIP expansion budget exceeds 40GiB')
    for name in names:
        ancestors = PurePosixPath(name).parents
        for ancestor in ancestors:
            if str(ancestor) in names:
                matching = next(item for item in items if item.filename.rstrip('/') == str(ancestor))
                require(matching.is_dir(), 'ZIP file used as directory')
    return items


def expand_zip(path, destination):
    with zipfile.ZipFile(path) as archive:
        items = zip_inventory(archive)
        destination.mkdir(mode=0o700, exist_ok=False)
        output = []
        for item in items:
            target = destination / item.filename
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, target.open('xb') as sink:
                shutil.copyfileobj(source, sink, 1024**2)
            require(target.stat().st_size == item.file_size, 'ZIP extracted length drift')
            target.chmod(0o440)
            output.append(dict(path=str(target.relative_to(destination)), bytes=item.file_size, sha256=sha(target)))
    return output


def uncompressed_identity(path, maximum):
    digest = hashlib.sha256()
    size = 0
    with gzip.open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            size += len(block)
            require(size <= maximum, 'OCI expanded layer budget exceeded')
            digest.update(block)
    return size, digest.hexdigest()


def add_bytes(archive, name, data):
    item = tarfile.TarInfo(name)
    item.size = len(data)
    item.mode = 0o444
    item.mtime = 0
    archive.addfile(item, io.BytesIO(data))


def squashfs_command(binary, sandbox, destination):
    return [str(binary), str(sandbox), str(destination), '-noappend', '-all-root',
            '-comp', 'gzip', '-processors', '2', '-read-queue', '64',
            '-write-queue', '128', '-fragment-queue', '64', '-no-progress']


def validate_sif_listing(listing):
    require(listing.count('FS (Squashfs/*System/amd64)') == 1, 'native primary SIF descriptor drift')


def build_sif(backend, archive_path, conversion, sif, audit, env):
    # CE 4.1.1 has no build --mksquashfs-args; use its native SIF tools instead.
    import resource
    import time

    version = subprocess.check_output([str(backend), 'version'], text=True, env=env).strip()
    require(version == '4.1.1', 'container backend version drift')
    compressor = Path('/usr/sbin/mksquashfs')
    require(compressor.is_file(), 'existing system mksquashfs unavailable')
    help_result = subprocess.run([str(compressor), '-help'], capture_output=True, text=True, env=env, timeout=30)
    help_text = help_result.stdout + help_result.stderr
    require(all(flag in help_text for flag in ('-processors', '-read-queue', '-write-queue', '-fragment-queue')),
            'bounded existing compressor options unavailable')
    sandbox = conversion / 'native_rootfs'
    squash = conversion / 'native_rootfs.squashfs'
    partial = conversion / 'fixed2025.part.sif'
    require(not any(p.exists() for p in (sandbox, squash, partial, sif)), 'native build destination already exists')
    commands = [
        [str(backend), 'build', '--disable-cache', '--sandbox', '--no-cleanup',
         str(sandbox), 'docker-archive://' + str(archive_path)],
        squashfs_command(compressor, sandbox, squash),
        [str(backend), 'sif', 'new', str(partial)],
        [str(backend), 'sif', 'add', str(partial), str(squash), '--datatype', '4',
         '--parttype', '2', '--partfs', '1', '--partarch', '2'],
    ]
    write(audit / 'NATIVE_BUILD_COMMANDS.json', dict(META, commands=commands,
          method='existing_CE_sandbox_system_squashfs_native_primary_SIF',
          compression_threads=2, compression_queue_mib=256, compression_address_space_limit_bytes=2 * 1024**3,
          administrator_configuration_modified=False, model_code_executed=False))

    def limits(compression):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
        resource.setrlimit(resource.RLIMIT_CPU, (1800, 1800))
        if compression:
            resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))

    deadline = time.monotonic() + 1800
    with (audit / 'SIF_BUILD.log').open('x') as log:
        for index, command in enumerate(commands):
            log.write(json.dumps(command) + '\n'); log.flush()
            remaining = deadline - time.monotonic()
            require(remaining > 0, 'native build total time budget exhausted')
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=remaining, preexec_fn=lambda: limits(index == 1))
            require(result.returncode == 0, 'native build stage failed: ' + str(index))
            if index == 0:
                require((sandbox / '.singularity.d/runscript').is_file(), 'native OCI runscript absent')
        listing = subprocess.check_output([str(backend), 'sif', 'list', str(partial)], text=True, env=env, timeout=30)
        log.write(listing)
    validate_sif_listing(listing)
    partial.chmod(0o440)
    os.link(partial, sif)
    partial.unlink()
    return dict(native_descriptor_listing=listing, sandbox_retained=True, compression_queue_mib=256)


def docker_archive(incoming, destination, audit):
    manifest = read(incoming / 'docker/manifest.json')
    require(sha(incoming / 'docker/manifest.json') == IMAGE_DIGEST, 'OCI manifest identity drift')
    config_path = incoming / 'docker/config.json'
    config = read(config_path)
    require(sha(config_path) == manifest['config']['digest'].split(':')[1], 'OCI config drift')
    require(config['os'] == 'linux' and config['architecture'] == 'amd64', 'OCI architecture drift')
    diff_ids = config['rootfs']['diff_ids']
    require(len(diff_ids) == len(manifest['layers']) == 11, 'OCI layer count drift')
    records = []
    expanded = 0
    for index, layer in enumerate(manifest['layers']):
        compressed_sha = layer['digest'].split(':')[1]
        path = incoming / 'docker/layers' / (compressed_sha + '.tar.gz')
        require(sha(path) == compressed_sha and path.stat().st_size == layer['size'], 'OCI compressed layer drift')
        size, digest = uncompressed_identity(path, 80 * 1024**3 - expanded)
        require('sha256:' + digest == diff_ids[index], 'OCI uncompressed diff_id drift')
        expanded += size
        records.append(dict(index=index, compressed_sha256=compressed_sha, bytes=size, diff_id='sha256:' + digest))
    write(audit / 'OCI_EXPANDED_IDENTITIES.json', dict(META, records=records, expanded_tar_bytes=expanded,
          identity_checked_before_build=True, original_layers_preserved=True))
    require(shutil.disk_usage(destination.parent).free > expanded * 4 + 20 * 1024**3,
            'fresh archive/build/SIF simultaneous capacity budget fails')
    config_name = manifest['config']['digest'].split(':')[1] + '.json'
    layer_names = [str(index) + '/layer.tar' for index in range(len(records))]
    saved = [dict(Config=config_name, RepoTags=['ecnu-external-mets:fixed2025'], Layers=layer_names)]
    with destination.open('xb') as sink, tarfile.open(fileobj=sink, mode='w|', format=tarfile.PAX_FORMAT) as archive:
        add_bytes(archive, config_name, config_path.read_bytes())
        add_bytes(archive, 'manifest.json', json.dumps(saved, separators=(',', ':')).encode())
        for layer, record, name in zip(manifest['layers'], records, layer_names):
            item = tarfile.TarInfo(name)
            item.size = record['bytes']
            item.mode = 0o444
            with gzip.open(incoming / 'docker/layers' / (record['compressed_sha256'] + '.tar.gz'), 'rb') as stream:
                archive.addfile(item, stream)
    destination.chmod(0o440)
    return dict(path=str(destination), bytes=destination.stat().st_size, sha256=sha(destination),
                source_digest='sha256:' + IMAGE_DIGEST, expanded_tar_bytes=expanded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--singularity', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    audit = root / 'operator/asset_materialization'
    audit.mkdir(mode=0o700, exist_ok=False)
    evidence(audit / 'STARTED.json', 'asset_materialization_started_no_model_loaded')
    stage = 'transfer_acceptance'
    try:
        acceptance = read(root / 'operator/ASSET_TRANSFER_COMPLETE.json')
        require(acceptance['status'] == 'all_server_asset_bytes_SHA_verified', 'asset transfer incomplete')
        for item in acceptance['files']:
            path = Path(item['path'])
            require(path.is_relative_to(root / 'assets/incoming') and not path.is_symlink(), 'asset path boundary')
            require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'transferred asset drift')
        require(len(acceptance['files']) == 15, 'complete asset set required')
        stage = 'bio_zip_expansion'
        incoming = root / 'assets/incoming'
        bio = root / 'assets/bio_weights'
        files = expand_zip(incoming / 'weights.zip', bio)
        write(audit / 'BIO_EXPANDED_SHA256.json', dict(META, root=str(bio), files=files,
              expanded_bytes=sum(x['bytes'] for x in files)))
        datasets = list(bio.rglob('Dataset001_BraTS'))
        require(len(datasets) == 1 and datasets[0].parent.name == 'nnUNet_results', 'Bio canonical model root ambiguous')
        dataset = datasets[0]
        models = {
            'A': ('nnUNetTrainerBraTS_TriadInit_SmallLesionWeightedCE_CEw3__3_FT900__nnUNetPlans__3d_fullres', (0,1,2,3,4)),
            'B': ('nnUNetTrainerBraTS_BrainIACWrapper_RCOversample__3__nnUNetPlans__3d_fullres', (1,3)),
        }
        selected = {}
        for name, (directory, folds) in models.items():
            model = dataset / directory
            required = [model / 'plans.json', model / 'dataset.json',
                        *(model / f'fold_{fold}/checkpoint_final.pth' for fold in folds)]
            require(all(p.is_file() for p in required), 'Bio full ensemble/plans incomplete')
            require(set(read(model / 'dataset.json')['labels'].values()) == set(range(5)), 'Bio label map drift')
            channels = read(model / 'dataset.json')['channel_names']
            require([channels[str(i)].lower() for i in range(4)] == ['t1c','t1n','t2f','t2w'], 'Bio modality order drift')
            selected[name] = dict(directory=str(model), folds=list(folds), files=[
                dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in required])
        brain = incoming / 'BrainIAC.ckpt'
        write(audit / 'BIO_FULL_MODEL_IDENTITY.json', dict(META, models=selected,
            nnunet_results=str(dataset.parent), brainiac=dict(path=str(brain),bytes=brain.stat().st_size,sha256=sha(brain)),
            model_loaded=False, original_zip_preserved=True))
        stage = 'offline_oci_archive'
        conversion = root / 'assets/container_conversion'
        conversion.mkdir(mode=0o700, exist_ok=False)
        archive_path = conversion / 'fixed2025.docker-archive.tar'
        archive_identity = docker_archive(incoming, archive_path, audit)
        write(audit / 'DOCKER_ARCHIVE_IDENTITY.json', dict(META, **archive_identity))
        stage = 'native_sif_build'
        temporary = conversion / 'tmp'
        temporary.mkdir()
        cache = conversion / 'cache'
        cache.mkdir()
        sif = root / 'assets/fixed2025.sif'
        require(not sif.exists(), 'SIF destination already exists')
        env = {k:v for k,v in os.environ.items() if not k.startswith(('SINGULARITY','APPTAINER'))}
        env.update(SINGULARITY_TMPDIR=str(temporary), SINGULARITY_CACHEDIR=str(cache), LC_ALL='C', OMP_NUM_THREADS='2')
        build = build_sif(args.singularity, archive_path, conversion, sif, audit, env)
        sif.chmod(0o440)
        evidence(audit/'COMPLETE.json','assets_materialized_not_model_runtime_pass',
            sif=dict(path=str(sif),bytes=sif.stat().st_size,sha256=sha(sif)),
            bio_identity_sha256=sha(audit/'BIO_FULL_MODEL_IDENTITY.json'),
            offline_conversion=True, docker_daemon_installed=False, namespace_policy_changed=False,
            model_loaded=False, jobs_created=False, native_build=build, free_bytes_after=shutil.disk_usage(root).free)
    except BaseException as error:
        evidence(audit/'FAILED.json','asset_materialization_failed',stage=stage,error_type=type(error).__name__,
                 message=str(error),no_automatic_retry=True,jobs_created=False)
        raise


if __name__ == '__main__':
    main()
