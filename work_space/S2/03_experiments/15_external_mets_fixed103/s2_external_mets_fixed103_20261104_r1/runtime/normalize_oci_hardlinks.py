"""Materialize tar hardlinks before native OCI flattening; never extract a rootfs."""
import copy
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

from s0_build_contract import require, sha


def canonical(name):
    path = PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts and '\\' not in name
            and str(path) not in ('', '.'), 'unsafe archive member')
    require(str(path) == name.rstrip('/'), 'noncanonical archive member')
    return str(path)


def resolve_link(member, earlier):
    seen = {member.name}
    target = canonical(member.linkname)
    while True:
        require(target not in seen, 'hardlink cycle')
        seen.add(target)
        require(target in earlier, 'hardlink target absent or forward/cross-layer reference')
        item = earlier[target]
        if item.isfile():
            return item
        require(item.islnk(), 'hardlink target is not a regular file')
        target = canonical(item.linkname)


class DigestReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes = 0

    def read(self, size=-1):
        block = self.stream.read(size)
        self.digest.update(block)
        self.bytes += len(block)
        return block


def normalize_layer(stream, destination, deadline, maximum_bytes):
    links = []
    earlier = {}
    source_headers = 0
    with tarfile.open(fileobj=stream, mode='r:') as source:
        with destination.open('xb') as sink, tarfile.open(fileobj=sink, mode='w|', format=tarfile.PAX_FORMAT) as output:
            for member in source:
                require(time.monotonic() < deadline and source_headers < 300000, 'normalization header/time budget')
                name = canonical(member.name)
                require(name not in earlier, 'duplicate layer member')
                require(member.type in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE,
                        tarfile.SYMTYPE, tarfile.LNKTYPE), 'unsupported layer special file')
                require(not member.sparse, 'sparse tar not supported')
                emitted = copy.deepcopy(member)
                content = source.extractfile(member) if member.isfile() else None
                if member.islnk():
                    target = resolve_link(member, earlier)
                    require(member.size == 0, 'hardlink data size must be zero')
                    content = source.extractfile(target)
                    emitted.type = tarfile.REGTYPE
                    emitted.linkname = ''
                    emitted.size = target.size
                    emitted.pax_headers.pop('linkpath', None)
                    emitted.pax_headers.pop('size', None)
                if content is None:
                    output.addfile(emitted)
                else:
                    with content:
                        reader = DigestReader(content)
                        output.addfile(emitted, reader)
                    require(reader.bytes == emitted.size, 'layer content short read')
                    if member.islnk():
                        links.append(dict(name=name, original_target=member.linkname,
                            resolved_regular_target=target.name, bytes=reader.bytes,
                            content_sha256=reader.digest.hexdigest()))
                earlier[name] = member
                source_headers += 1
                require(sink.tell() <= maximum_bytes, 'normalized layer size budget')
    require(destination.stat().st_size <= maximum_bytes, 'normalized layer final size budget')
    destination.chmod(0o440)
    return dict(headers=source_headers, hardlinks=links, bytes=destination.stat().st_size, sha256=sha(destination))


def verify_layer(stream, normalized, links, deadline):
    identities = {item['name']:item for item in links}
    earlier = {}
    count = 0
    with tarfile.open(fileobj=stream, mode='r:') as source, tarfile.open(normalized, mode='r:') as derived:
        for original in source:
            require(time.monotonic() < deadline, 'normalized verification time budget')
            emitted = derived.next()
            require(emitted is not None, 'normalized archive lost header')
            wanted = copy.deepcopy(original)
            if original.islnk():
                target = resolve_link(original, earlier)
                wanted.type, wanted.linkname, wanted.size = tarfile.REGTYPE, '', target.size
                wanted.pax_headers.pop('linkpath', None)
                wanted.pax_headers.pop('size', None)
            keys = ('name','mode','uid','gid','uname','gname','mtime','type','linkname','size','devmajor','devminor')
            require(all(getattr(wanted,k)==getattr(emitted,k) for k in keys), 'normalized metadata/order drift')
            # PAX path/size may be reconstructed by tarfile; all unrelated extended metadata must remain equal.
            for key,value in wanted.pax_headers.items():
                require(emitted.pax_headers.get(key)==value, 'normalized extended metadata drift')
            if emitted.isfile():
                before = source.extractfile(resolve_link(original,earlier) if original.islnk() else original)
                after = derived.extractfile(emitted)
                digest = hashlib.sha256()
                with before, after:
                    while True:
                        block = before.read(1024**2)
                        other = after.read(1024**2)
                        require(block==other, 'normalized file bytes differ from original')
                        if not block:break
                        digest.update(other)
                if original.islnk():
                    require(digest.hexdigest()==identities[original.name]['content_sha256'], 'hardlink content identity drift')
            earlier[original.name]=original
            count += 1
        require(derived.next() is None, 'normalized archive added header')
    return count


def add_bytes(archive, name, data):
    item = tarfile.TarInfo(name)
    item.size, item.mode, item.mtime = len(data), 0o444, 0
    archive.addfile(item, io.BytesIO(data))


def normalize_archive(original, directory, deadline, maximum_bytes=80*1024**3):
    directory.mkdir(mode=0o700, exist_ok=False)
    records = []
    with tarfile.open(original, mode='r:') as source:
        manifest = json.load(source.extractfile('manifest.json'))
        require(len(manifest)==1, 'ambiguous Docker archive')
        config_name = canonical(manifest[0]['Config'])
        raw_config = source.extractfile(config_name).read()
        require(hashlib.sha256(raw_config).hexdigest()+'.json'==config_name, 'original config SHA drift')
        config = json.loads(raw_config)
        layers = manifest[0]['Layers']
        require(config['os']=='linux' and config['architecture']=='amd64', 'OCI platform drift')
        require(len(layers)==len(config['rootfs']['diff_ids']) and 0<len(layers)<=11, 'OCI layer count drift')
        total = 0
        for index,name in enumerate(layers):
            name = canonical(name)
            require(name==str(index)+'/layer.tar', 'original layer sequence drift')
            path = directory/(str(index)+'.tar')
            with source.extractfile(name) as stream:
                reader = DigestReader(stream)
                while reader.read(1024**2):
                    require(time.monotonic()<deadline, 'source diff_id deadline')
                original_sha = reader.digest.hexdigest()
            require('sha256:'+original_sha==config['rootfs']['diff_ids'][index], 'original layer diff_id drift')
            with source.extractfile(name) as stream:
                record = normalize_layer(stream,path,deadline,maximum_bytes-total)
            with source.extractfile(name) as stream:
                require(verify_layer(stream,path,record['hardlinks'],deadline)==record['headers'], 'layer count drift')
            record.update(index=index,path=str(path),original_diff_id='sha256:'+original_sha)
            records.append(record)
            total += record['bytes']
        derived_config = copy.deepcopy(config)
        derived_config['rootfs']['diff_ids'] = ['sha256:'+r['sha256'] for r in records]
        require({k:v for k,v in config.items() if k!='rootfs'} ==
                {k:v for k,v in derived_config.items() if k!='rootfs'}, 'OCI inference configuration changed')
        config_bytes = json.dumps(derived_config,sort_keys=True,separators=(',',':')).encode()
        new_config_name = hashlib.sha256(config_bytes).hexdigest()+'.json'
        derived_manifest = copy.deepcopy(manifest)
        derived_manifest[0]['Config'] = new_config_name
        archive_path = directory/'hardlinks_materialized.docker-archive.tar'
        with archive_path.open('xb') as sink, tarfile.open(fileobj=sink,mode='w|',format=tarfile.PAX_FORMAT) as output:
            add_bytes(output,new_config_name,config_bytes)
            add_bytes(output,'manifest.json',json.dumps(derived_manifest,separators=(',',':')).encode())
            for name,item in zip(layers,records):
                header = tarfile.TarInfo(name)
                header.size, header.mode = item['bytes'], 0o444
                with Path(item['path']).open('rb') as stream:output.addfile(header,stream)
                require(time.monotonic()<deadline, 'derived archive writing deadline')
        archive_path.chmod(0o440)
    return dict(path=str(archive_path),bytes=archive_path.stat().st_size,sha256=sha(archive_path),layers=records,
        original_archive_sha256=sha(original),original_config_sha256=hashlib.sha256(raw_config).hexdigest(),
        derived_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        method='regular_file_materialization_of_original_tar_hardlinks_then_unchanged_native_OCI_merger',
        every_member_bytes_and_metadata_compared=True,only_hardlink_storage_and_rootfs_diff_ids_changed=True,
        inode_sharing_not_preserved=True,not_original_byte_identical_Docker_digest=True,
        original_archive_preserved=True,rootfs_extracted_by_this_program=False,models_loaded=False)
